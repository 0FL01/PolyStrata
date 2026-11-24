# Tideglass serial decode: dense launch grouping on gfx1031

**Do not prototype another Q8_0 grouping pass.** Each visit already groups every Q8_0 input projection in
one dispatch and reuses its prequantized activation inputs. The only split inside `project()` is the Q4_0
relative projection on 12 non-core visits. A mixed-format kernel could remove those 12 launches per token,
but not their dot products or weight reads. The whole singleton kernel budget is **0.057 ms/token**; even
deleting it and both surrounding dispatch gaps for free removes only **0.203 ms/token (+0.56% throughput on
the traced wall)**. This is an intentionally generous trace-based scenario, not a measured speedup.

## What the existing trace shows

RX 6700 XT (`gfx1031`), Ryzen 7 5700X, HIP Release `c0dd0e52`, Tideglass Inkling-2-Mini Unbound v2.1,
Q8_0 KV, 2,048-token `mixed` prefix, no lookup/drafter. `roc2k` contains 32 serial greedy decode steps,
1,171.658 ms of logits-copy-to-logits-copy time (36.614 ms/token) and 30,612 complete dispatches. See the
[capture conditions and boundary check](../profiler/results-2k.md). Counts below were checked per token
and logical visit against the actual GGUF header and dispatch order/geometry, not just by kernel name.

| Stage | Total launches | Per token | Mean ms/token |
|---|---:|---:|---:|
| `segments<8>`: 8,576 output rows (core Q/gate/rel/K/V) | 1,152 | 36 | 2.362 |
| `segments<8>`: 8,192 output rows (non-core Q/gate/K/V) | 384 | 12 | 0.761 |
| Q4_0 `rel` fallback: 384 output rows | 384 | 12 | 0.057 |
| Q8_0 attention output: 2,304 output rows, after attention | 1,536 | 48 | 1.158 |
| Fused stream RMSNorm and Q8_1: one stream | 640 | 20 | 0.097 |
| Fused stream RMSNorm and Q8_1: three streams | 896 | 28 | 0.122 |
| Standalone activation quantization for these attention projections | 0 | 0 | 0 |

The trace does have 32 standalone `native_quantize_q8_1_kernel` dispatches, all immediately between the
final RMSNorm and the Q6_K vocabulary head. Those are not part of this grouping. The Q4_0 dense flag
was on, but the specialized grouped Q4_0 kernel did not dispatch. The Q4_0 expert slot kernels are a
different path.

## Weight types, dependencies and arithmetic

- In the Tideglass GGUF, Q and gate are Q8_0 `[2304,3072]` and K/V are Q8_0 `[2304,1024]` on all 24 physical
  layers (dimensions shown as `[in,out]`). `rel` is `[2304,384]`: Q8_0 on core layers 6–17 (visits 6–41),
  but **Q4_0 on physical 0–5 and 18–23** (visits 0–5 and 42–47). The attention output is Q8_0 `[3072,2304]`
  on every layer. [`load_layer()`](../../../../src/poly/inkling2.cu#L712) preserves these matrix types;
  [`Engine::global_scale()`](../../../../src/poly/engine.cpp#L1775) gives all of them scale 1.
- [`layer_head()`](../../../../src/poly/inkling2.cu#L1325) supplies Q/gate/rel with the same Q8_1 row.
  The 20 shared-input visits also give K and V that row. The 28 other visits already use one fused
  RMSNorm+Q8_1 launch for three **different** normalized streams, producing separate K and V rows. The
  `!shared_kv && !k8` fallback in [`project()`](../../../../src/poly/inkling2.cu#L1457) is not taken with
  the recorded flags. The attention-output projection already gets `attq_` from
  [`decode_combine_kernel`](../../../../src/poly/inkling_attend.cu#L751). It cannot join the front batch:
  the combine stage needs the projected Q/K/V/rel/gate and sink, and each output dot consumes the combined
  result across heads. The next visit depends on the current attention output and FFN/residual.
- `project()` groups equal type, input width and unit scale in `MmvqSegments`; segment pointers can differ
  even if their activation rows alias. [`native_mmvq_segments()`](../../../../src/kernels/cuda/native_mmvq.cu#L3524)
  supports Q8_0 at width 2304; generic segments do not support Q4_0, and the specialized Q4_0 entry needs at
  least two segments. Thus the non-core singleton takes ordinary `lin()` with the **existing** Q8_1 row.
  The precise hypothetical merge is to add 384 Q4_0 workgroups to the 8,192-row Q8_0 grid on those 12 visits,
  select format per workgroup, and call the existing one-row (`SmallK=false`) body for each format. It must
  not replace Q4_0 with Q8_0 or alter either dot product. The total is 60 → 48 `project()` launches per token;
  zero quantization launches or activations are removed.
- Disassembly of the **actual profiled Release binary** confirms the Q8_0 segment and standalone kernels use
  two `v_dot4c_i32_i8`, a half-scale product (`v_fma_mix_f32` with zero), then `v_fmac_f32` into the lane
  accumulator. The source lane takes `kbx=tid/4 + 32*j`, `kqs=2*(tid%4)`. The four wave32 partials are added
  as `(((p0+p1)+p2)+p3)` per lane after the LDS barrier, then an XOR tree `16,8,4,2,1` (`ds_bpermute_b32`
  plus `v_add_f32`). Q4_0 instead takes `tid/2 + 64*j`, `kqs=2*(tid%2)`, with four integer dots and three
  `v_fma_mix_f32` instructions for its affine correction using `ds.y` and its accumulation; it has the same
  final reduction tree. Preserve the order **and** the fusion, not merely the dot-product formula. A one-wave
  row with a different lane map or an independent mul/add sequence is not an exact replacement.

The adjacent sink and full-visit K/V tap GEMVs are uploaded as F32 from BF16 tensors, not Q8_0
[`Dev`](../../../../src/poly/engine.cpp#L1836) projections. In this trace they use 256 threads per row, vs. 128
for the quantized grid. The two tap calls on each of 8 full visits could share the existing equal-shape F32
segments entry; all 16 tap dispatches together are only 0.050 ms/token and grouping would still compute both.
The 48 sinks take 0.151 ms/token and differ in row count from the taps. These are separate from the Q8_0
grouping hypothesis and do not duplicate activation quantization.

## Gate: no patch or GPU slot

- Q8_0 is already at the minimum of one front projection launch per visit. The 3.123 ms/token is not a
  launch-only budget: these workgroups consume **996.434 MB/token** of logical dense weights (319 GB/s
  effective payload rate, not measured DRAM traffic). The separately dependent output accounts for another
  360.972 MB/token at 312 GB/s effective. The same-card historical 376 GB/s streaming-read reference is
  a scale check, not a bandwidth bound in cache. Even attaining that payload rate in the front group
  would leave ~2.65 of 3.12 ms/token; grouping does not change the bytes. The ISA shows dependent scalar
  segment-index loads before the hot loop, but no timing for them; replacing their lookup with fixed shape
  offsets would not remove a dispatch or quantizer. There is no evidence to book time for that rewrite.
- Removing **all** measured singleton device time instantaneously would yield +0.16% on 36.614 ms/token.
  The 12 intervals from Q8_0 dispatch completion to sink start, including the singleton and **both**
  surrounding gaps, total 0.203 ms/token; deleting that entire interval yields +0.56%. The no-Q4 visits
  still had a median 8.24 µs gap from Q8_0 completion to sink start. The gap is neither all HIP overhead nor
  necessarily removable, and the Q4_0 grid work would still run. For scale, the separate historical
  same-card untraced 3.4–4.4 µs small-launch floor would map 12 launches to only 0.041–0.053 ms/token;
  do not add this to the traced scenario. A +25% gain on this trace would require 7.323 ms/token.
- [`ec320c9d` already grouped five all-Q8_0 projections](../../2026-11-21-inkling2-dense-decode.md) on the
  earlier IQ4_XS model. The [Bartowski mixed Q4_0/Q8_0 trials](../../2026-11-24-inkling2-q4-0-decode.md#L91)
  removed 48 launches/token with nearly unchanged grouped GPU time; the untraced 4K pairs were mixed.
  Q8_0 four **output rows** per workgroup regressed: 1.036 vs 0.938–1.004 ms/token for K/V alone, or
  ~2.13 vs ~2.05 ms/token in the same-launch mixed configuration. That is distinct from the DSpark
  multi-**token** kernel and is the relevant prior for workgroup tiling. Neither history is a Tideglass A/B.
- No engine edit, build, service change, GPU run, or new parity claim. Reopen only if a supported
  fixed-weight model/trace has substantially more split work, or an independent exact memory/dispatch
  experiment shows savings beyond noise. If reopened, first compare raw Q8_1 and five FP32 projection
  buffers against the original on the actual 4/5-segment shapes and both input-sharing cases, then require
  full logits and greedy IDs at matched placement before an untraced paired test.

## CPU-only reproduction

```sh
D="$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK"
python -B docs/goals/arch-decode-recon/dense/attribute.py "$D/roc2k"
```

Optional ISA spot-check from the pinned `877fbdfed412…` binary: the `gfx1031` native-MMVQ code object is
at file offset `0xe8b000`, size `0x42b7f0` **for that binary only**. Extract to the disassembler's stdin,
without running the binary:

```sh
sha256sum build-poly-hip/strata-poly
OBJDUMP=$(find .venv/lib -path '*/_rocm_sdk_devel/lib/llvm/bin/llvm-objdump' -print -quit)
SYMS=_ZN6strata7kernels12_GLOBAL__N_127native_mmvq_segments_kernelILi8EEEvNS0_12MmvqSegmentsEi
SYMS=$SYMS,_ZN6strata7kernels12_GLOBAL__N_124native_small_mmvq_kernelINS1_8Q80BlockELi8ELb0EEEvPKT_PKNS1_8Q81BlockEPfii
SYMS=$SYMS,_ZN6strata7kernels12_GLOBAL__N_124native_small_mmvq_kernelINS1_8Q40BlockELi4ELb0EEEvPKT_PKNS1_8Q81BlockEPfii
dd if=build-poly-hip/strata-poly iflag=skip_bytes,count_bytes skip=$((0xe8b000)) count=$((0x42b7f0)) status=none \
  | "$OBJDUMP" -d --disassemble-symbols="$SYMS" -
```
