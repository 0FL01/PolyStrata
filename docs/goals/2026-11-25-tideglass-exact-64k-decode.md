# Goal: faster exact Tideglass serial decode at 64K on RDNA2

Status: unmet
Source: 2026-11-25 user request to implement the 64K wait RECON, follow measurements rather than a fixed plan,
and commit, push, deploy and reprofile each proven win.
Last updated: 2026-11-25

## Objective and finish line

Improve repeatable 64K serial decode throughput for Tideglass Inkling-2-Mini Unbound v2.1 on the RX 6700 XT
with no numerical divergence. Continue while an in-scope hypothesis has a concrete discriminating experiment;
close with the measured wins and losses, limits, and the final gfx1031 Release binary. Do not ship a leaf win.

- R1 — Prove useful exact engine changes on the same hardware/model/inputs. Primary evidence: matched greedy
  IDs, touched operator/committed state bytes and untraced live-swaps A/B on saved mixed and independent code.
  Require a repeatable gain of at least 1% and above observed A/A on both at 64K, without 2K/p90 regression.
  Status: pending; no screened candidate met this.
- R2 — For each proven candidate, rebuild Release, commit to `main`, push `origin/main`, deploy and smoke the
  local production engine; keep the byte-identical baseline when the feature is off. Primary evidence: paired
  result, remote commit, running executable hash and completed local request. Status: not applicable; no proven
  candidate to integrate.
- R3 — Reprofile after substantial changes and use the new baseline for further tests; report failed approaches
  and remaining limits and give the final binary and its build verification. Status: verified for the retained
  baseline; no substantial release change triggered an after-deploy profile.

Constraints: preserve weights, quantization and FP32 reduction order; do not accept changed expert backends or
different route/slot assignments as an exact A/B. Keep Q8_0 KV, reserve, no-draft behavior, and loopback API.
GPU runs are serial and guarded with `run_guarded.py --warn-amdgpu-faults` on gfx1031.

Envelope: `src/poly/` serial Tideglass CPU/GPU expert and attention paths and their immediate kernels, flags and
docs; isolated temporary profilers/benchmarks outside the tree. No API, model file, verifier or disk-format change.

## Checkpoint

Baseline `db7046a7`, Release `877fbdfed412…`. Prior measurements and exact fixtures are indexed in
`~/.cache/opencode-tmp/opencode/wait64-recon.dg0bMA/PLAN.md`; prior campaign decisions are in
[architecture RECON results](2026-11-24-inkling2-architecture-aware-results.md).
At 64K the exposed CPU-not-ready median was 8.96/6.01 ms/token mixed/code (instrumented), mostly visits with
at least three CPU experts. Swap DMA completed before settlement on the sampled trajectories. No-go: moving
the fence, staging the mapped CPU sum, shared-activation AVX2 dots with no cold gain, and W1 4/2-wave grouping.

## Evidence and next checkpoint

Implementation scratch: `~/.cache/opencode-tmp/opencode/tideglass-impl/`.

- Capped standalone A/A (5,050 MiB arena+KV budget, fresh prefill, 256 measured steps each, live swaps)
  reproduced all 257 greedy IDs on both requests. Mixed 32.66 / 32.27 t/s, code 35.18 / 34.72 t/s;
  run-to-run differences 1.19% and 1.32%. Time comparisons must clear that spread, not just 1%.
- Two real decode captures gave 2,062 CPU visits at positions 65,536–65,559, with 33/33 greedy ID parity per
  workload. Down expert-major traversal within a tile kept all 4,750,848 captured sum words exact. At 3–8
  CPU misses, median paired cold savings were 0.84/0.06 µs/visit mixed/code. The fixed-GPU estimate was only
  0.065/0.061 ms/token; rejected. Ready-down is a separate prototype with a same-scratch barrier control.
- Rolling K/V load issue plus DPP8 left all partials, outputs, and packed outputs exact on eight 64K tail
  lengths. The best evicted arm saved 18.90 µs/visit in 24/24 pairs (463.04 → 444.86 µs median) versus the
  40 µs leaf gate; only eight full-attention visits use it. No engine change.
- Swap phase replay at fixed timestamps suggests follower RAM packing, not just H2D, overlaps the CPU join.
  An 8 ms shift reduced modeled pack intersection with it by 0.84/1.13 ms/token without a predicted late
  copy; that does not measure contention. The same-binary live-swaps phase A/B tests the cause. A separate
  pinned-buffer microbench with exactly the same stream wait favored AVX2 non-temporal stores at typical
  12 swaps: Q4_0 concurrent-GPU batch 3.98 → 3.22 ms p50; Q4_1 4.28 → 3.43 ms. Every checked DMA payload
  matched. Test that arm separately in the engine; faster staging alone does not imply faster tokens.
- Early-ready `down` passed real-input CPU boundary parity. Ordered replay projected only +0.085/+0.128
  ms/token p50 on mixed/code versus stock at fixed GPU producer timing (scratch barrier control regressed
  both). Using a SIMD reducer for independent rows (same add order within a row) raised the projection to
  +0.188/+0.378 ms/token. The leaf omits contention with the real GPU submit task and swap worker; a live
  mirrored-window screen tests whether the signal survives.
- In the first phase diagnostic, delaying all requested packing by 8 ms retained 257/257 IDs on both and
  went 31.41 → 32.57 t/s mixed, 33.17 → 35.39 code. The on arms nearly match prior A/A, the off arms were
  slower than A/A, and code off had a 192 ms outlier. Mirrored 8-step windows with two boundary steps excluded
  gave 32.59 → 32.92 / 35.29 → 35.33 t/s (192 steps scored each). Both 257-ID streams matched. The code
  signal fell to +0.10%; no general gain and no fixed production delay.
- Non-temporal pinned staging did not pass the live engine screen: 32.61 → 32.22 t/s mixed, 35.17 → 34.81
  code, with p90 up 0.72/0.39 ms. Exact 257/257 IDs in each pair. The standalone staging win did not survive
  the end-to-end test; don't promote it or assume measured microbench savings shorten the critical path.
- Ryzen exact F16C weight-scale conversion and worker tile reservations of four both matched captured CPU
  sums and failed the cold leaf on both fixtures (−1.66/−1.20 and −15.51/−9.29 µs paired savings at 3–8 misses).
- Exact one-wrap W1 SWA ring indexing plus DPP8 (same 32 chunks and additions) matched partials, FP32 and
  packed Q8_1 before/after wrap through 200K. Evicted timings at position 65,536 saved 0.92 µs out of
  27.84 µs per visit. Forty local visits are only about 0.037 ms/token at that measured gain; rejected.
- Live mirrored early-down + exact SIMD: mixed 23.50 → 22.19 t/s (−5.56%, p90 60.05 → 62.25 ms),
  code 30.70 → 31.40 (+2.28%, p90 42.51 → 41.21 ms). All 257/257 IDs matched on both. These legs were
  slower overall than the capped stock A/A; this is a noisy conflicting result, not qualification or a
  universal cost estimate. The replay omitted live RAM/PCIe contention and actual HIP calls.
- Biased AVX2 Q4_0 dot passed all 4,750,848 captured word checks. Cold paired savings −5.48/−4.86 µs;
  warm +11.98/+14.13 µs. The warm gain does not transfer to the cold fixture. CPU-only `perf` replay on
  real activations put 90.70/90.83% of sampled user cycles in the Q4 dots and 3.83/3.62% in worker loops.
  Samples include stalls and are not proof of arithmetic saturation. The replay also counted about
  206k/178k 4K L2-DTLB misses per replayed token (whole-process counter, including setup).
- Anonymous huge-page A/B could not establish matching 4K/2M backing at representative size on this host:
  1,590,534,144 B per arm, `MADV_COLLAPSE` returned `ENOMEM`, only 36 MiB backed by 2M pages on the huge arm.
  This is a backing failure, not a speed measurement; stock weights remain file-mapped.
- The apparent 0.41 ms/token final-sync-to-next-token host interval was dominated by trace flush. The
  stock ROC-only mixed prefix showed 0.053 ms p50 for the entire interval. No sampler rewrite.
- Pinning each CPU participant to one physical core gave 32.18 → 32.00 t/s mixed and 33.85 → 35.27 code
  in mirrored windows, with 257/257 IDs on both. The code off arm had a 189 ms step; omitting the single
  slowest step from each arm for diagnosis leaves +1.43%. Mixed did not improve, and there is no pinned default.
- The bounded same-row prefetch/unroll arm matched all 4,750,848 captured words; zero timed major faults.
  Cold paired savings at 3–8 misses were −3.51/−6.26 µs and warm −0.51/+0.06. No engine A/B for this arm.

## Closure

- `cmake --build build-poly-hip --target strata-poly poly_workers_test poly_inkling_attend_test
  poly_mmvq_segments_test -j4` verified the current `gfx1031` Release target; `strata-poly` was already up to date.
  The retained artifact is `~/.cache/opencode-tmp/opencode/tideglass-impl/final/strata-poly`, SHA-256
  `877fbdfed4120abe7ee9f5e093a3af17f9c8220ebb28a8388a650cf338a775ef`.
- `poly_workers_test`, `poly_inkling_attend_test` and `poly_mmvq_segments_test` passed on the RX 6700 XT/5700X
  after supplying the production ROCm library directory. The same binary completed a guarded 2K mixed smoke at
  40.58 t/s over 64 measured steps (65 output IDs, `DONE length`, no reuse/draft). Not a speed comparison.
- No engine source or production config changed. `main` and `origin/main` remain at `db7046a7`; the service stayed
  inactive/disabled. Raw data, commands and rejected alternatives are indexed in `tideglass-impl/REPORT.md`.
- Remaining distinct CPU-memory hypothesis needs reproducible, verified 2 MiB backing of the exact working set.
  The scratch mapping could not allocate it on this boot. Do not retune a boundary or use changed placement as
  an exact speedup in lieu of a working control.

Complete the frozen outcomes using this envelope and evidence. Stop at a proven blocker or when no remaining
in-scope experiment has a falsifiable expected result; record exact evidence and the smallest unlock.
