# First target-verifier difference

This is an instrumented target-only diagnostic, not a throughput run. It does not change the production engine.
The private run directory is `$R/divergence`, where `$R` is the `dflash2-recon` run directory. Do not publish the
FP32 or quantized tapes, request files, token IDs, or snapshots.

## Method

- RX 6700 XT (gfx1031), Tideglass Inkling 2 Mini Unbound v2.1, dense Q8 KV and compact SWA. Fixed combined expert
  arena and global KV budget of 4,968 MiB, 2,048 MiB VRAM reserve, eight CPU threads, zero expert swaps, no
  drafter or lookup. W1 and windows have the same allocated verifier capacity (eight rows).
- Restore one archived mixed session and append one identical fresh token. The standalone W1 calls
  `Engine::eval_decode` and records the next eight greedy input IDs. One `Engine::window` call in each wider leg
  reads those same IDs. Compare only aligned rows; all window-stage captures and logits precede `keep(width)`.
  There is no rejected row or rollback on this witness.
- Compare the prompt logits and layout hash before comparing per-row FP32 buffers and Q8 bytes. The hash covers
  the exact resident expert IDs and their relative offsets within each layer. Trace points include projections,
  head norms, the new cached KV row, attention, pre-router norms, router scores, picked experts, their weights
  and backend, CPU expert sum, visit residuals, final norm and post-softcap logits. The W1 fused GPU MoE sum
  has no separate buffer, so it is compared at the shared visit output rather than at `moe_gpu_sum`.
- `build.py` copies and instruments the relevant translation units into the private directory and relinks from
  an existing Release HIP build. While tracing, W1 emits the final norm outside its graph (same operations).
  `leg.py` takes the shared GPU lock for each leg and invokes `run_guarded.py --warn-amdgpu-faults`.
  `compare.py` only emits aggregates. Each new leg name is write-once.

## Short witness: first query at position 2305 (zero-based)

Each process restored 2,304 IDs, appended one identical ID, and had 1,143 resident experts with layout hash
`e2855b3d9687beda`. The first query's embedding, incoming convolutions, and every traced array through the
visit-0 FFN norm (including its Q8 bytes) matched.

| Window | First unequal operation in row 0 | Differing FP32 words | Max absolute / RMS error | Greedy top-1 IDs |
|---|---|---:|---:|---:|
| W2 | None in 1,709 common stage records | 0 | 0 / 0 | 2/2 |
| W4 | None in 1,709 common stage records | 0 | 0 / 0 | 4/4 |
| W8 | Visit 0, F32 router GEMV | 78 / 128 | 4.77e-7 / 2.32e-7 | 8/8 |

W2/W4 also matched every compared boundary and every post-softcap logit on the later rows. All eight stock W8
rows first differed at the same router call (59–78 scores changed, max absolute 4.77e-7). In row 0, the first
eight selected experts and their CPU/GPU assignments still matched, but two pick weights changed (max 1.79e-7)
and the first visit's residual changed by up to 4.77e-7. The final row-0 post-softcap logits eventually differed
in 200,057/200,058 FP32 words, max 1.531. Equal greedy IDs are not evidence of equal sampling distributions.

The dispatch in `src/poly/kernels.cu::gemv_f32_waves` uses a 256-thread block per output when
`ceil(n_out / 8) * rows < 80`, otherwise a 32-lane wave per output for this shape. There are 128 router scores
and 2,304 inputs. W1 (including the dual-write path), W2, and W4 use the same block reduction. W8 changes
the grouping of the FP32 additions; this happens before selecting experts, not in attention or `keep()`.

## Isolation

`DIV_FIX_ROUTER=1` only changes the traced multi-row router: one batched launch of the W1 block-per-output
kernel, with the same subsequent copy and routing. A forced-ID W1 rebuild/repeat matched all 13,336 stage
records, including its eight logit vectors. The router control removed the visit-0 mismatch in all eight
W8 rows and exposed a second source at visit 5, the first global Q8 attention. Its row-0 Q/K/V, relative
terms, gate, sink, and appended K/V cache bytes were bit-identical. The FP32 attention output differed in
2,388/3,072 words (max absolute 7.15e-7, RMS 6.19e-8).

At this position the dense Q8 reader's W1 uses a chunk of 32 keys (73 chunks in row 0); W8 uses 40 keys
per chunk (58 chunks) because the scratch permits only `4096 / (8 KV groups * 8 rows) = 64` chunks per row.
Forcing all eight queries to use 73 chunks in one launch would exceed that bound. Changing chunk size changes
where the partial online sums restart and how their results are combined. With only the router change, the
first four rows' logits matched W1 exactly, but offsets 4–7 did not (largest absolute logit difference 3.525).
All eight greedy top-1 IDs still matched.

`DIV_FIX_GLOBAL_ATTN=1` gives each traced global Q8 query its exact W1 launch and chunk boundaries, reusing
the existing scratch; it does not change the stored KV or the SWA reader. With **both** controls, W8 matched
W1 on all **13,336 common stage records**, including all **1,600,464 post-softcap logit words** over the eight
rows and the 1,852 GPU / 1,220 CPU expert calls. The unmodified W8 repeat on the same diagnostic ELF reproduced
the original W8 tape byte for byte. This establishes exactness for this fixture and these captured stages,
not for other contexts, models, flag combinations, or rejected-state behavior. The serial attention control
is not a throughput candidate.

## Reproduce

The host scripts expect the archived mixed checkpoints, model and profile already present at the paths in
`leg.py`. From the repository root, using a Python with NumPy for the comparer:

```sh
d=rdna2-rnd/dflash2/divergence
python -B "$d/build.py"
python -B "$d/leg.py" short-w1 --width 1
python -B "$d/leg.py" short-w8 --width 8 --baseline short-w1
python -B "$d/compare.py" short-w1 short-w8
python -B "$d/leg.py" short-w8-router --width 8 --baseline short-w1 --setting DIV_FIX_ROUTER=1
python -B "$d/compare.py" short-w1 short-w8-router
python -B "$d/leg.py" short-w8-router-attention --width 8 --baseline short-w1 \
  --setting DIV_FIX_ROUTER=1 --setting DIV_FIX_GLOBAL_ATTN=1
python -B "$d/compare.py" short-w1 short-w8-router-attention
```

The stock `short-w{1,2,4,8}` legs, `short-w1-aa`, `short-w8-stock-repeat`, `short-w8-router`, and
`short-w8-router-attention` each contain a `receipt.json` and guard receipts. The last four used the same ELF
(`6b7044a98a62e65d…`). Across the eight legs the minimum sampled free VRAM was 2,190.1 MiB with no guard stop
or fresh fault. Guard warnings were earlier boot records. No timings from trace runs are used as performance data.

## Live DFlash2 exact verifier

`live_exact.py` calls the live research builder with extra source overlays. With no overlay that builder still
uses the original live arithmetic. The corrected ELF is in `$R/divergence/live-exact-bin-v2/strata-poly-dflash2-live`;
the builder rejects an existing output directory. Neither `src/poly/`, the Ninja objects, nor `$R/live-bin/`
are changed.

- The FP32 router runs the W1 reduction in one batched launch. A tall Q8_0 W8 projection dispatches W1 for
  each row instead of taking the alternative `warps_cols` reduction. The Q6_K target head remains batched.
- Inside `Engine::window` only, short decode attention (2–8 rows) computes each row's W1 chunk size and count. Contiguous rows share a
  launch only when both numbers match and the scratch fits. It does this for SWA and dense global attention;
  W1 and non-windowed prefill retain the original kernel and geometry. The fp16-KV fallback uses serial W1 launches
  for short multi-row attention; the live parity tests below use only Q8 KV.
- The existing live sampler picks from complete post-softcap target rows. `--parity` records every picked
  row and the final convolution state, chronological SWA rings and real dense global KV. `live_compare.py`
  compares all bytes, checks the initial exact expert placement, and reports any first changed region without
  printing logits, KV, or IDs. It also reports any separately persisted expert profile count differences.

```sh
base=$HOME/.cache/opencode-tmp/opencode
R=$base/polystrata-runs/dflash2-recon
d=rdna2-rnd/dflash2/divergence
weights=$HOME/.lmstudio/models/unsloth/Inkling-2-Mini/DFlash2/Inkling-2-Mini-DFlash2-Q4_0.gguf
python -B "$d/live_exact.py"
for temperature in 0 1; do
  for k in 0 3 7; do
    python -B rdna2-rnd/dflash2/live/measure.py --run "$R" \
      --out "$R/divergence/exact-live-parity-v2/t$temperature-k$k" \
      --config "$base/polystrata-runs/native-donor-bc2b/baseline/fixed.json" \
      --profile "$base/arch-campaign.XWM1yK/profile.seed" --domain json_struct \
      --binary "$R/divergence/live-exact-bin-v2/strata-poly-dflash2-live" \
      --mode live --weights "$weights" --library "$R/native-opt/baseline/libstrata-dflash2.so" \
      --steps 65 --cap 3072 --seed 42 --temperature "$temperature" --k "$k" --parity
  done
  p=$R/divergence/exact-live-parity-v2/t$temperature
  python -B "$d/live_compare.py" "$p-k0" "$p-k3" "$p-k7"
done
```

Both K0 and the window arms attach the same drafter. K0 uses `eval_decode` and does not call the draft proposer.
Never use timings from a `--parity` run. For throughput, use fresh output names and the same arguments but
omit `--parity`.

The earlier `live-exact-bin-v1` ELF put the attention change in the shared reader, so a short multi-row
prefill tail could change sums before any draft was checked. Its paired short `json_struct` receipts test
only that common-prefix fixture; do not use them as evidence for unchanged prefill. In the v2 ELF, the
prefill reader is unmodified and the new attention reader is called only from the genuine verifier window.

The resumed cold-prompt test keeps the first 4,105 IDs from the private `long-taps/16k/request.json` base
and uses the installed 2,048-row batches: 2,048 + 2,048 + 8 rows, then the last token alone for its logits.
Stock attention uses 72-key chunks for that eight-row tail; the ungated v1 overlay would impose 32-key W1 chunks. For these
legs the private config sets the expert+KV cap directly to 3,072 MiB with swaps at zero. It does not use
`measure.py --cap` (which also enables a freeze and prevents making room for the initial longer KV).
The load starts with 701 slots; the single KV growth before prompt work leaves 672 slots. The stock K0
control uses `live_exact.py --stock-prefill --out <new directory>`; only its driver writes a tape and its
engine reports the placement fingerprint, with no draft attached or change to the target kernels.

On the RX 6700 XT, v2 K0 matched the original live ELF and the stock no-drafter K0 control on all
200,058 prefill logits, all 4,105 global KV rows, and 991 shared prefill rows from the terminal SWA rings.
The stock control and v2 had the same 672-slot placement (`9c4a971975eda3f9`); the original live ELF reports
the same count but predates the layout fingerprint.

At temperature 1, seed 42, the v2 W1/W4/W8 legs on this same input generated the same 33 IDs. Each W4/W8
tape equaled W1 across all 191,300,328 bytes: 6,601,914 chosen-row FP32 logits, the terminal convolution
state and SWA rings, and the real dense global KV through the committed position. W4 accepted 19 of 36
proposals (seven rollbacks, three after partial acceptance); W8 accepted 21 of 70 (nine rollbacks, five
after partial acceptance). The tapes do not include the separate persisted expert usage profile: the
W4/W8 profiles counted 6,528/18,816 extra expert uses, exactly 384 per discarded row. This did not affect
the matched placement or target state; it is not a claim that the profile has rollback semantics.

Private receipts: `$R/divergence/exact-resume-4105-v1/{prefill-kv-comparison,target-comparison}.json`.
Tape-free speed samples are separate, in `$R/divergence/exact-resume-speed-v1/speed-comparison.json`. They use the short
115-token `json_struct` prompt, 65 greedy outputs, the same Q4_0 draft and exact verifier, the fixed 701-slot
layout `0d634fa4251d0df7`, and W1/W4/W1 run order. W1 produced 26.60 and 25.83 tok/s, W4 42.44 tok/s
(1.62× the W1 median, one W4 sample). All 65 IDs matched; W4 accepted 47/51 proposals. Inter-token p90
increased from 40.4–42.2 ms to 90.7 ms. Do not infer throughput for the 4,105-token parity fixture from
these short-prompt samples. Across five resumed parity and three timing legs the minimum sampled free
VRAM was 4,480.2 MiB, with no guard stops or fresh GPU faults.

### Commit-only profile, grouped Q8, and CPU reuse

`live_exact.py --commit-profile` separates tentative route counts by row and applies them only
for the prefix passed to `keep()`. This change is only for zero swaps; natural-placement
`recent_` decay and GPU/CPU admission still do not have W1 transaction semantics. With the same
701 slots, T=1/seed42 and one partial accept discarding two rows, FP32 logits, canonical committed
state and the saved usage profile matched W1 byte-for-byte (`$R/target-commits-v1`).

`--group4` includes the commit-only profile and gives long, dense Q8 attention about 1.52 MiB
of extra split scratch. At spans ≥32,768, group up to four consecutive queries with identical
W1 chunk size and count (≤160), and use the four-wave stock leaf. The combine is unchanged.
Other geometries fall back to the original W1-ordered overlay. This is batching, not the more
expensive LDS-based sharing [experiment](../multiq/README.md).

- At 67,841 rows, Q4_0 W4 with Group4 matched W1's 49 greedy IDs, all **1,313,898,856** tape bytes,
  and the saved profile, even with two rejected windows (five discarded rows). A separate sampled
  25-ID pair also matched its tape, but had no misses. Same 940 resident experts, zero swaps.
- Separate tape-free Group4 ABBA at that 64K fixture: baseline **34.89/34.58** vs Group4
  **36.05/36.41 tok/s**; all 49 IDs and 34/39 accepted proposals matched. See `$R/group4-live-v1`.
- `--cpu-reuse` shares Q4 unpack across inputs at W3–W8 on Zen 3. On 128 greedy Python outputs
  at 701 equal slots, the 8-worker ABBA measured **33.93/33.38** baseline and **35.97/36.71 tok/s**
  with reuse. Output and accept decisions matched, 88/114 accepted in each. Do not project these
  gains onto W1 or other CPU types. Untraced receipts: `$R/cpu-reuse-live-v1`.
- `--group4 --cpu-reuse` combines both. The complete 64K matched tape, 49 IDs, resident layout,
  and profile were checked again after combining (`$R/combined-64k-parity-v1`).

### Live near the context limit

Target-only restore, followed by 2,049 new IDs for genuine draft SWA warmup; there is no draft
state SAVE/RESTORE. Layout was equal within each context after restore and held through decode.
The arena cap was 5,200 MiB, zero swaps, RX 6700 XT, all 49 greedy IDs match in each comparison.
The DSO contains the separately [qualified exact draft attention](../native/opt/README.md).
TTFT excludes model load and the restore itself. p90 is internal token-to-token latency, not wire.

| First anchor | Experts | W1 → W4 tok/s | TTFT W1 → W4 | p90 W1 → W4 | W4 accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 133,377 | 685 | 20.89 → 24.19 | 4.76 → 5.92 s | 51.4 → 146.6 ms | 34/39 |
| 199,070 | 430 | 18.03 → 22.89 | 6.67 → 6.13 s | 58.4 → 182.5 ms | 36/36 |
| 199,070, combined | 430 | 18.31 → 26.18–28.11 | 6.15 → 6.22–6.50 s | 57.1 → 145.9–156.6 ms | 36/36 |

The adjacent ordinary exact-W4 ABBA control at 199,070 ran at **19.69/22.86 tok/s**. This drift
limits estimates of the combined kernel gain, but both combined runs outperformed both controls
with the same outputs, placement and decisions. Peak process RAM was ~18.0 GiB; module and card
allocations are not process RSS. These short near-cap continuations are not graded agent runs or
an evaluation of active swapping. Receipts: `$R/warm-exact-v1` and `$R/combined-near-cap-v1`.
