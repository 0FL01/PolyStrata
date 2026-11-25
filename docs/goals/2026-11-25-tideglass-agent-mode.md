# Tideglass: faster English reasoning and code on RDNA2

Status: complete. Source: user request to implement the Agent Mode measurements, ship each repeatable win, and test the next candidate against the resulting binary.

- R1: measure exact candidates on the RX 6700 XT and Ryzen 7 5700X, including English reasoning, emitted code and tool continuations. Check real output IDs and touched arithmetic against the baseline separately from timing. Keep the existing model, precision, expert placement, decoding settings and memory reserve. Status: verified.
- R2: only promote a repeatable end-to-end win above control variation with no material p90 or short-context regression. Commit, push, deploy the qualified Release binary on loopback, verify and profile it before considering another promotion. Status: verified.
- R3: leave a verified Release build; report the measured wins, failures and limitations. Status: verified.

Baseline `db7046a7`. Compare matched CPU/GPU placement: the final small prefill batch can change backend even between stock runs. The test-only replay is not a deployment setting. First test the full-vocabulary sampler at temperature 1; greedy and constrained sampling are controls. In parallel prepare exact full-attention load-scheduling leaves; test registered-source expert copies without changing who gets a slot. Run GPU tests serially with the RDNA2 guard and boot-fault warning. Do not modify weights, answer budgets or client semantics to improve a measurement.

Evidence: private scratch `~/.cache/opencode-tmp/opencode/agent-impl.l4wfKL`; preceding Agent Mode study `../agentmode64.LOFrhb/AGENT-PROFILE.md` relative to that directory. Continue on a new mechanism only when the measurements give a falsifiable next test; if a candidate fails, do not deploy it.

## Checkpoint: sampled decode

The candidate specializes only `Sampler::draw_all` in `src/poly/main.cpp`, behind `STRATA_POLY_AVX2_SAMPLE=1`. The scalar path remains the default. On this Release build the Q weights, double run masses, selected IDs, and RNG states matched across 2,808 cases per setting, including 66 real logit rows and signed zeros, NaNs, infinities, and underflow boundaries. The production target built with just this change; SHA-256 `a8cfc09f1c14d3f17c64920683bce7ec15544fe65f060b5bd379b0fb6d322da3`.

`mirrored/phase-results.json`: two independent 65,536-ID Agent Mode requests, real loopback HTTP, T=1, top-k=0, top-p=1, min-p=0, seed 42. Each request ran twice using the opposite sampler schedule in short blocks. No token was forced. All 409 IDs on the English review (including two tool calls) and all 2,288 native IDs on the code request matched. The code answer continued on the same engine with 67,265 prompt IDs reused. The client's SSE intervals improved 28.165 → 27.442 ms/token for English reasoning (+2.64% throughput; stock/stock +0.15%), and 30.792 → 30.136 ms/token for fenced code (+2.18%; stock/stock +0.67%). Both p90s fell. This is an in-request crossover, not the improvement in total time from a cold request.

Independently closed: direct H2D from registered GGUF experts rather than copying into the pinned bounce buffer (+0.41% reasoning, −0.10% code, 449/449 IDs and 33 full logit rows matched on each); removing a per-tile CPU worker completion counter (cold paired median −1.92/−0.81 µs saved). For cold H2 prompt attention, two later fetch points and a two-launch 2+1 head split each matched 39,601,152 FP32 words. All three lost on all five measured shapes: later fetch raised live registers or spilled; the split repeated K/V reads. None belongs in the binary.

## Whole-request qualification

Each arm started a fresh production HTTP frontend and a Release engine with benchmark-only placement controls, outside the profiler. `http/current.py` checks both original 65,536-ID inputs, the entire recorded prefill assignment, and the exact slot identities and offsets before/after warm growth. All generated IDs and returned reasoning, text and tool arguments matched. No sampled tokens were fed back from a reference. Full-logit capture is separate from timing. For the coding request the frontend's ordinary answer budget reached a second native `GEN`; it reused 67,265 IDs and read 17 more. The final diff was cut off by the token limit; identical output is not a claim that the patch is correct.

| Request | Order | Decode tok/s, off → on | Two paired gains | Same-arm variation | Cold request through final SSE, mean s |
|---|---|---:|---:|---:|---:|
| English statistics design review | ABBA | 32.58 → 33.39 | +2.67%, +2.31% | +0.41% | 89.087 → 88.633 |
| `argparse` patch with English reasoning | BAAB | 35.41 → 36.30 | +2.46%, +2.53% | +0.04% | 141.854 → 140.189 |

Rates pool equal-length runs by total decode time. On only the indicated output phase the rates were 35.45 → 36.35 tok/s (295 English reasoning IDs) and 32.38 → 33.26 tok/s (413 fenced-code IDs as classified by the HTTP parser). Both orientations improved p90 on both requests. Cold TTFT stayed around 76.5–76.8 s; sampling does not accelerate the prefill. The paired comparisons share runs; these are not independent confidence intervals. Raw pairs: `http/time-full/`, `decision.json`.

At 2,048 input IDs, sampled decode was 41.36/41.54 tok/s off and 42.60/42.88 on; 257/257 sampled IDs and 65/65 greedy IDs matched and the captured 33 logit rows were byte-identical for each mode. A short greedy timing moved 40.51 → 39.97 tok/s; the AVX2 draw is not called for greedy. No speed claim for that mode. At 64K, the full-vocabulary post-softcap bits matched for 33 scored rows on each sampled fixture, including a window inside the naturally emitted code. All 409 and 2,288 raw sampled IDs matched; both independent greedy controls matched 33/33 IDs and 33 full logit rows. `http/{reason-parity,code-parity,greedy}/` contains the separate comparisons.

A separate genuine `read_case` turn on a shorter prefix reached natural EOS. Returning the local fixture to the same engine reused all 773 previously evaluated IDs. Both prompts and both naturally generated ID sequences matched off/on.

Two further limits: a 36 MiB anonymous bank held entirely on 2 MiB pages for the short correctness check, but did not retain the required ≥95% backing in timed runs; no latency score. The bank also covered no whole held-out visits with ≥3 CPU misses. A guarded HIP VMM leaf compared the current per-piece access grants with one contiguous grant at unchanged addresses and sizes. Its first 256 MiB A/B check passed, but a sentinel failed in the following stock/stock control. No timings were scored and neither engine VMM nor deployment settings use this experiment.


## Shipped baseline and remaining limits

Code commit `50effa3a` was pushed to `origin/main`. The installed Release binary matched the checked SHA-256 above, and `STRATA_POLY_AVX2_SAMPLE=1` was enabled only in the local engine config. No placement replay or test caps were installed. The loopback-only service accepted an authenticated chat, returned `len(items)` with a natural stop and rejected the unauthenticated `/v1/models` request. The previous binary and an atomic backup of its config were retained. Service autostart policy was not changed.

Profiled that Release executable again. On 32 serial English steps at 64K, rocprof recorded 32.99 ms/token elapsed, 17.73 ms in the union of GPU kernels, 3.76 ms in copies and 1.99 ms overlapping both. These are profiled intervals, not the untraced A/B or removable stalls. The cold prefill region was 76.84 s; Q4_0 grouped products (27.66 s), flash attention (24.75 s), and Q8_0 grouped products (9.28 s) still dominate. On the deployed service a separate real coding request produced 172 completion tokens. From its first meaningful SSE onward, perf attributed 47.78% of sampled host user cycles to Q4_0/Q4_1 dot routines. This is an on-CPU sample, not a wall-time share or a DRAM counter. That uncontrolled installed run did not enter the speed comparison.

The deployed sample also put 47.11% of host user-cycle samples in `Workers::loop`, concentrated at the polling branch after `pause`. This motivated one more live test on the new AVX2 baseline. Reducing the idle spin budget from 20,000 to 1,500 (same job, completion and wakeup protocol) changed client throughput by -1.64% on reasoning and -0.14% on fenced code. Stock/stock controls were -7.19% and +0.39%. Process CPU time per token changed 167.4 → 106.2 and 190.1 → 123.2 ms (sums across threads, not wall time or wattage). The two complete natural histories and warm slot identities matched the controls. No repeatable latency win on both phases: keep the original wait policy.

Stop here on the measured candidates. Cold prompt cost did not move, the tested H2 changes were slower, and reducing host synchronization or bypassing weight staging did not release a comparable decode interval. The bounded huge-page bank lacks both repeatable backing and reuse; the VMM leaf did not establish a correct stock control. No second optimization cleared the same quality and latency checks. `closure.json`, `release-roc/`, `production-profile/`, `park-results/` and `final-health.json` in the private campaign directory hold the final measurements.
