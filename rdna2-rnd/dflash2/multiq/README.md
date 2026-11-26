# Sharing dense Q8 K/V across target queries on gfx1031

This is a standalone target attention experiment. It makes no changes to the engine, the verifier, or the installed server.

**Result:** exact on the tested inputs. Sharing gives a further **3.4–6.3%** lower full attention latency than
the four-wave nonsharing control after a read flush. Most of the gain over W1 comes from batching without software
sharing. Use exact four-wave batching as the reference in any in-engine work; do not infer a verifier or generation
speedup from this test.

## What is measured

The fixture uses deterministic synthetic inputs in the layout and value ranges of `src/poly/inkling_attend_test.cu`.
All arms use the same FP32 Q, relative table, gate and sink, and the same Q8_0 K/V code planes and FP16 scales.
Each of the `W` independent query positions sees only its own causal prefix and keeps all 24 query heads. This
is reuse **across query positions**, not merely the native sharing among three GQA heads. The long tests time
one full attention visit at position 65,535 and 199,069 (zero-based), using the program baseline's
`STRATA_INKLING_DECODE_CHUNKS=160` ([prior qualification](../../../docs/goals/native-donor-recon/qualification.md)).

- **Serial:** call stock `attend_decode()` separately for each query, or launch its unchanged eight-wave leaf in
  the same sequence for leaf-only timing. Scratch is reused row by row when timing.
- **Batch8 / Group4:** unchanged eight-wave or grouped four-wave stock shaders, batched with the **serial**
  chunk size and count. These measure batching and possible hardware cache reuse without software K/V sharing.
  The test allocates enough scratch for W4 × 160; current production W4's 4,096-block scratch cap cannot hold
  this exact batched layout. These controls are not the existing public W4 dispatch.
- **LDS 2G/32 and 1G/64:** two K/V groups with 32-key tiles, or one group with 64-key tiles. One wave owns each
  `(query, K/V group)` pair. The query waves fill disjoint slices of the same raw K/V tile in LDS; all waves then use
  it, keeping their own FP32 reductions and online softmax. A wave processes the same key subsequence, in order,
  as in W1. The partially visible tile is masked separately for each query. Each wave scatters directly to the
  stock combine layout. There is no separate scatter launch, copy or transpose.

All batched arms split into contiguous cohorts when **either** serial chunk size or count differs (with singleton
cohorts falling back to stock). Besides the long cases, checks exercise short causal tails and each type of
geometry crossing. Comparisons use the stock W1 split `(max, sum, unnormalized values)`, final FP32 and bytewise
`Q8_1` output. `leaf` includes the scatter; `full` adds stock combine, sink, gate and packed Q8_1. Times are HIP
stream events around one graph replay. The warm arm reheats on the same graph; the read-flushed arm reads a
disjoint 256 MiB allocation before each *whole graph*, not between its rows. The flush itself is not timed.
This is not an assertion of zero cache hits or a target-layer/engine wall-clock measurement.

## RX 6700 XT results

HIP runtime 7.13.26193, Release `-O3`, wave32. Each number below is the median of 21 graph replays with rotating
and reversing arm order. Time is **ms for all W rows**, not per row. Each cell gives **leaf / full**, after a
256 MiB read flush. Bold marks the lowest full time. The 64K serial geometry is 158 chunks of up to 416 keys;
the 199K geometry is 160 of up to 1,248.

| Context / queries | Serial W1 | Batch8 | Group4 | LDS 2G/32 | LDS 1G/64 |
|---|---:|---:|---:|---:|---:|
| 64K / W2 | 0.921 / 0.959 | 0.880 / 0.902 | 0.671 / 0.693 | **0.646 / 0.668** | 0.781 / 0.806 |
| 64K / W4 | 1.829 / 1.911 | 1.724 / 1.764 | 1.208 / 1.234 | 1.279 / 1.327 | **1.162 / 1.192** |
| 199K / W2 | 2.925 / 2.974 | 2.848 / 2.885 | 1.978 / 2.018 | **1.825 / 1.892** | 2.289 / 2.314 |
| 199K / W4 | 5.608 / 5.698 | 5.518 / 5.570 | 3.552 / 3.583 | 3.726 / 3.757 | **3.405 / 3.434** |

The winning LDS shapes saved 25/42 µs at 64K and 127/149 µs at 199K over Group4 (W2/W4 respectively).
Group4 alone accounts for 88–94% of the reduction from serial W1. Warm graphs agree on the winning shapes:
W2 full Group4 → LDS was **0.690 → 0.666 ms** and **1.992 → 1.844 ms**; W4 was **1.231 → 1.187 ms** and
**3.612 → 3.448 ms**. The other LDS geometry lost to Group4 at both spans for both widths: 1G/64 at W2 was
14.6–16.2% slower after a flush, and 2G/32 at W4 was 4.8–7.5% slower. Neither LDS shape is a general replacement.

All 48 checks against separate stock W1 calls matched: **47,748,480** split FP32 words, **442,368** final FP32
words, and **497,664** Q8_1 bytes across the four arms. Of these, half exercised the two sharing kernels and half
the two batched controls. At position 65,726, W4 spans a 158→159 chunk-count change; at 66,558 it spans a 416→424
key chunk-size change. Both are split into two W2 cohorts. No FP reassociation, approximate attention, real prompt,
token check, or cache-commit behavior is involved.

Generated gfx1031 ISA metadata (32-lane waves):

| Shader | Threads/block | VGPRs | SGPRs | LDS/block | Spill / private bytes |
|---|---:|---:|---:|---:|---:|
| Stock leaf (serial/Batch8) | 256 | 173 | 44 | 0 | 0 / 0 |
| Group4 leaf | 128 | 175 | 44 | 0 | 0 / 0 |
| LDS 2G/32 W2 | 128 | 188 | 26 | 20 KiB | 0 / 0 |
| LDS 2G/32 W4 | 256 | 188 | 28 | 20 KiB | 0 / 0 |
| LDS 1G/64 W2 | 64 | 186 | 26 | 20 KiB | 0 / 0 |
| LDS 1G/64 W4 | 128 | 186 | 26 | 20 KiB | 0 / 0 |
| Stock combine | 128 (launch) | 47 | 46 | 0 | 0 / 0 |

Sharing replaces repeated global K/V fetches with LDS stores, reads and two barriers per tile. It costs
11–13 more VGPRs per thread than Group4, without spills. This shows a real reuse gain at two long positions, but
does not measure DRAM traffic or isolate the cost of any one resource. A production design also has to budget
the exact W4 partials (7.52–7.62 MiB here versus 6.09 MiB allocated by stock) and prove parity through
real placement, rollback and next-token sampling before running a wall-clock comparison.

Receipt: `$R/multiq/v2/{summary.txt,bench-gfx1031.s,SHA256SUMS,guard/command.log}`. The guard did not stop and
sampled at least 10,248.7 MiB free; warnings referred to boot history. `$R/multiq/v1` is retained: the first
test failed because its sentinel fills ran on the default stream while the test used a nonblocking stream. V2
orders both on the test stream; there are no timings from V1.

## Reproduce

Use a built gfx1031 Release `poly_inkling_attend_test` target (the runner takes its compiler and linker flags
from Ninja and reuses the built lean/H2 support objects; neither is called). It writes the binary, exact build
commands, device assembly, hashes, and raw guard and event results into a fresh private directory.

```sh
R="$HOME/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon"
python -B rdna2-rnd/dflash2/multiq/run.py --out "$R/multiq/my-test"
python -B rdna2-rnd/dflash2/multiq/report.py "$R/multiq/my-test"
```

GPU work is serialized with `flock "$R/gpu.lock"` and `run_guarded.py --warn-amdgpu-faults`.
