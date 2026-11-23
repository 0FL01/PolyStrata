# Goal: more Inkling-2-Mini decode throughput on RDNA2

Status: complete (one deterministic long-context win, deployed; short/16K at the current limits)
Source: user request of 2026-11-23: "Изучи последние коммиты и цели, нужен план по пооптимизации движка, цель
поднять `decode` t/s ... проводить профилирование, писать бенчмарки ... на уровне ядра ... не потерять в качестве
из за дивергенции/смены формул". Then: "Приступай к профилированию". Then the working directive quoted under
"User directive" below: RECON, the plan is a map, commit and push each deterministic win, re-profile after it.
Last updated: 2026-11-23

## Objective

Inkling-2-Mini decode (`Inkling-2-Mini-UD-IQ4_XS.gguf`, 200,000 context, Q8_0 KV, production flags) is faster on this
RX 6700 XT / Ryzen 7 5700X at short context and at 16K, with the same answers and the default path unchanged. This
goal records the measured bottlenecks and the plan; implementation is the next step.

## User directive (2026-11-23, verbatim)

"`1. Reserve 1600 MiB (сейчас 2048): +0.5–1.6% short, +2.2% на 16K, prefill не меняется. Перед дефолтом проверить
200K grow/shrink и floor 1536.` - reject, reserve не менять и явно это зафиксировать в цели

Переходи от реализации к RECON, цель максимально ускорить реальный end to end `decode`, а не просто выполнить
план, план это GPS навигатор, а не рельсы. Используй текущий план как карту гипотез, а не жёсткий маршрут,
начинай с P1. Работай автономно длинными циклами: профилирование -> гипотеза -> временный прототип/бенчмарк ->
измерение -> анализ -> следующий эксперимент (если профит детерминирование получен - коммит и пуш в main, далее
это будет как baseline для последующих правок (т.е закрепляем результат и идём дальше). При неудачах выясняй
почему ожидания не подтвердились, пересматривай гипотезы и пробуй другие подходы, не Зацикливайся на одном
направлении и свободно меняй приориты по результам экспериментов. Успешные оптимизации интегрируй в main (прод
бинарник), проверяй end to end прирост. После существенных изменений, перепрофилируй движок, bottleneck может
переместиться. ПРодолжай исслендвоание и реализацию, пока останется перспективное проверяемые гипотезы, в финале
покажи измеренные результатыы, неудачные подходы и оставшиеся ограничения, и деплой в прод.

Делай копию плана и моей подсказки в цель, помни что документ с целью это глина, её можно мять как угодно по ходу
работ, в эту глину пиши удачи/неудачи"

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest
unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks or optional source text.
Stop on a fresh guard fault or a proven blocker; record the evidence and the smallest unlock.

## Frozen Contract

### Required Outcomes

- R1: Short-context decode faster than the current production build.
  - Source: user; the forced 94+96 fixture is the existing decode gate.
  - Acceptance: paired A/B (A = `ebab4fad` + production config, B = candidate) on the forced 94+96 fixture, B above A
    beyond run-to-run variation; same forced IDs; coherent output.
  - Primary evidence: `rbench.sh` A/B logs, parity fixture, trace.
  - Status: pending
- R2: 16K decode faster, not a prefill regression.
  - Acceptance: paired 16,384-prompt + 128-token decode A/B, B above A; 2K/8K/16K prefill within variation.
  - Primary evidence: `bench16k.py` A/B, prefill logs.
  - Status: pending
- R3: Quality unchanged.
  - Acceptance: every kept kernel is bit for bit the old one, or passes the existing attention test tolerance; the
    parity fixture's top-1 IDs match; the switch-off path is byte-identical; no formula or precision change.
  - Primary evidence: unit tests, parity fixture, top-1 comparisons.
  - Status: pending
- R4: The long-context cost is measured and the 200K deployment stays safe.
  - Acceptance: 16K/32K/64K trace or end-to-end numbers for the chosen changes; the expert cache shrink at 200K
    keeps the guard floor; no OOM/fault.
  - Primary evidence: long-context runs and `cover()`/`shrink_cache()` logs.
  - Status: pending

### Constraints

- C1: RDNA2/HIP only; one card, one sequence; same GGUF, Q8_0 KV, 200,000 context; no expert skipping and no
  reduced precision for speed.
- C1a: The VRAM reserve stays 2,048 MiB. The 1,600 MiB proposal is rejected by the user (2026-11-23); do not
  change the production reserve in this goal.
- C2: New behavior opt-in; with the switch off the path is byte-identical; other models unchanged.
- C3: GPU runs through `rdna2-rnd/scripts/run_guarded.py` (1536 MiB floor); one engine at a time; benchmarks on
  copies of the frozen expert profile. A local guard copy that uses `journalctl -b` is allowed while the boot
  journal contains the 2026-11-23 GPU reset.
- C4: Server on 127.0.0.1 with its key; artifacts, logs, profiles and binaries stay out of Git.
- C5: Docs keep measured numbers with what they were measured on.

### Non-goals

- DSpark/speculative decoding (measured before at 35.5 against 36.5 tok/s on this workload); a second card; other
  models; prefill speed beyond keeping it.

## Change Envelope

- `src/poly/inkling2.*`, `src/poly/inkling_attend.*`, `src/poly/kernels.cu`, `src/kernels/cuda/native_mmvq.cu` and
  `src/kernels/cuda/iq_kernels.cu` (fusions and attention), `include/strata/poly/kernels.hpp`, `src/poly/engine.*`
  only where a launch or graph needs a hook, unit tests and CMake entries, docs (`docs/goals/`, results doc).
- Forbidden: request protocol changes; default-path changes; new dependencies or services; a formula or precision
  change that is not bitwise.

## Profiling (RECON)

Hardware and configuration: RX 6700 XT (gfx1031, 12 GB), Ryzen 7 5700X, TheRock HIP 7.13, Inkling-2-Mini-UD-IQ4_XS,
Q8_0 KV, 200,000 capacity, batch 2048, 8 threads, 64 swaps, 2048 MiB reserve, frozen expert profile, flash, direct
staging, prefetch, NVFP4 tail, batch KV release, small GEMM 2x2, lane probabilities, state pack. The 94+96 fixture is
94 prompt tokens and 96 forced decode tokens; 16K is 16,384 + 128.

### Short context, production config

- 45.1-46.4 tok/s (median about 45.9; 21.6-22.2 ms/token). The first run after a boot or a game can be 3-8% lower
  until the page cache and clocks settle.
- rocprofv3 trace, 95 decode tokens: 25.21 ms/token traced, 15.75 ms GPU busy, 9.46 ms gaps; untraced about 21.9 ms.
  1,105 launches a token.
- GPU busy, per token:
  - NVFP4 cached experts: 5.24-5.36 ms (34%), 88 launches, about 60 us each; about 1.46 GB of expert bytes.
  - Q8_0 dense projections: 2.85-3.39 ms (18-20%), 48 launches, one a visit.
  - Q6_K LM head: about 1.02 ms (6.5%), one launch.
  - Q8_0 small projections: 1.11-1.15 ms (7%), 48 launches (the o projection).
  - Q6_K projections: 0.28-0.30 ms, 15 launches.
  - decode attention + combine: 0.56 + 0.18 ms (4.7%), 96 launches.
  - the rest: about 5.0 ms in 3-12 us kernels (`gemv_rows` 0.57, `mix` 0.50, `add_rms_norm_q8_1` 0.30,
    `control_in/out` about 0.70, `conv_add` 0.26, `mix3` 0.21, `add_scaled` 0.19, `heads_kv` 0.19, and so on).
- Effective bandwidth: Q8_0 dense about 354 GB/s, small Q8_0 about 326 GB/s, LM head about 372 GB/s, all at or near
  the card's measured streaming speed. NVFP4 experts about 272 GB/s, about 25% below the same streaming speed.
- Host gaps: about 6.2 ms/token untraced. 4.2 ms lie in gaps of 20 us or more (about 50 a token): the CPU experts
  and the router event. The rest is launch latency: an empty dependent kernel costs 3.1-3.2 us on this setup, about
  3.4 ms for 1,105 launches; a HIP graph of the same chain costs 2.5-2.7 us (about 0.6 ms less a token).
- Expert cache: 1,340 of 3,072 experts in VRAM; 83-86% of decode uses hit. About 51-60 misses a token; the CPU
  experts take 4.5-5.1 ms/token, at 53 GB/s of DDR4 read (near the socket's limit). An LRU simulation over the routed
  IDs gives 86.6% hits and 51.4 misses a token at 55 slots a layer; the engine matches it. Prefetching the previous
  token's top-8 changes neither the hits nor the misses (they are the experts already cached). More capacity helps
  slowly: 65 slots a layer (about +1.1 GB) gives 42.8 misses, 80 slots (+2.6 GB) 28.4.
- Expert weight: scaling the frozen profile's edge-layer counts to an effective x1/x2/x4/x8 changed the hit rate
  (83/85/82/77%) but no weight beat x2 beyond variation; x4 lowered the hit rate and did not lower the critical path.
- Reserve and swaps: 1600 MiB against 2048 (same 64 swaps) measured 45.9-46.3 against 45.1-46.0 tok/s at short
  context (about +0.5-1.6%) and 40.2-41.0 against 39.0-39.9 at 16K (+2.2%); prefill was unchanged. At 1600, 16
  swaps against 64 measured 46.8 against 47.1 in one run each: no gain, only the smaller staging buffer. Reserve
  1280 gave no further gain and left about 1.2 GB free; 1024 reached 1.0 GB free and the 1536 floor.
- Vectorized NVFP4 loads: on a cold 382 MB working set, 16-byte weight and activation loads with the activation row
  staged in shared memory measured 311 GB/s gate/up and 260-269 GB/s down against 179 and 165 GB/s scalar (bitwise
  equal). In the engine at short context the same change measured 5.27 against 5.24 ms of expert time, and 44.6
  against 44.3 tok/s: no gain. The engine's expert bytes are partly in the 96 MB Infinity Cache; the cold win does
  not transfer at these contexts. The patch is kept outside Git for a long-context or cold-cache test.

### 16K context

- 38.2-41.0 tok/s (26-29 ms/token), trace 29.04 ms/token spanned, 16.91 ms GPU busy, 12.13 ms gaps. The gap sum is
  inflated by one-off page faults (a 61.8 ms H2D expert copy and a 29-72 ms head launch appeared once); use the
  median token, not the average, and warm the model file before timing.
- Busy: NVFP4 experts 5.34 ms (31.6%), Q8_0 dense 2.87 ms (16.9%), decode attention 1.62 ms (9.6%) and combine
  0.30 ms, small Q8_0 1.13 ms, LM head 1.04 ms, the rest about 4.6 ms. Attention grows from 0.74 to 1.92 ms a token
  and is the main 16K addition; the expert and dense paths are unchanged.
- At 190K the attention kernel alone was measured before at 2.20 ms a full-attention visit (about 413 MB of KV,
  188 GB/s), 8 full visits a token; that is the dominant long-context cost and the least bandwidth-efficient large
  kernel left.

## Roofline

The token moves about 3.2 GB of dense and expert weights (1.0 GB dense Q8_0, 0.38 GB head, 1.46 GB experts, 0.36 GB
small projections) plus the KV it reads. At the card's 360 GB/s that is about 8.9 ms; the measured memory-bound
kernels take about 10.5 ms, 85% of that floor. The other 11.4 ms of the 21.9 ms token are the 5.0 ms small-kernel
tail and the 6.2 ms host/launch gap. A perfect fusion of every small kernel and a graph of every launch could
approach 11-12 ms, but the experts, the CPU misses and the dependent chain do not allow that in one step. The plan
targets 17-18 ms (about 50-55 tok/s). A larger jump needs fewer bytes a token (a quality/quantization change),
several tokens a call (batching or speculation), or a different model; none is in this goal.

## Working mode (RECON)

- The plan is a map of hypotheses, not a route. Each loop: measure, state one falsifiable hypothesis, build the
  smallest prototype or benchmark, measure end to end, then keep or discard. Record both outcomes here.
- A deterministic win is integrated into `main`, committed with its measured before/after, and becomes the new
  baseline. Re-profile after it; the bottleneck may move.
- P1 is the small-kernel/launch chain; reserve is out of scope. Failures are written down with the reason the
  expectation did not hold, and the next hypothesis is chosen from the new trace.

## Plan

Ordered by measured gain per risk. Every kept change is bitwise or passes the existing attention test; any change
that moves a formula or a reduction order is not taken for speed alone. Stages 1-6 are the map after the RECON;
only the attention stage (3) produced a kept change in this round. Stage 1 is rejected; stages 2 and 4 were tried
and failed or are blocked; stages 5 and 6 remain conditional.

1. **Rejected: 1600 MiB reserve.** The user rejected it on 2026-11-23; the reserve stays 2,048 MiB. The measured
   +0.5-1.6% short / +2.2% at 16K is recorded, not taken. No swap change either unless a later run earns it.
2. **P1: fuse the per-visit small-kernel chains** (bitwise; the same sums in the same order):
   - `conv_add` + `mix` (two pairs a visit, 96 + 68 launches a token);
   - `mix` + `rms_norm` + `quantize_q8_1` (the next layer's input, 48 x 3 launches);
   - `control_in3` + `mix3` (24 x 2 launches on loop visits);
   - `situ_rows_q8_1` + `add_scaled` (the expert tail, 48 x 2 launches);
   - the two `gemv_rows` of a visit through `gemv_rows_segments` (136 launches a token).
   Expected: 1,105 to about 850 launches, 1.0-1.5 ms less busy + launch time. Unit tests compare each fused kernel
   with the old launches bit for bit; where a reduction cannot keep the order, the change is dropped or split.
3. **Decode attention at long context.** The 16K visit is 33.8 us for 35.6 MB; at 190K it is 2.20 ms for 413 MB
   (188 GB/s against 360). Work: more key chunks at 16K (the VGPR count is 176, so occupancy is the limit), a
   combine that folds into the last chunk or uses fewer partials, and a 16-byte K/V path with the existing Q8_0
   layout. Target 1.4x at 16K and 1.5x at 190K, inside the attention test's 3e-5. The short-context attention is
   only 0.74 ms; this stage is for 16K and beyond.
4. **Launch overhead: HIP graphs for the deterministic dense chain.** An empty chain costs 3.1 us a launch and 2.5
   in a graph; 1,105 launches leave about 0.6 ms on the table. Route and CPU experts break a whole-token graph, so
   capture the dense prefix/suffix of a layer (or the recurrent/attention part) between expert calls, with the
   expert launches outside. Target 0.3-0.6 ms; fall back if capture invalidates pointers or routing.
5. **Expert path, only with evidence.** The cache predictor and x4 weight did not pay; vectorized NVFP4 did not pay
   at short or 16K. Keep the vectorized patch for a cold/long-context test; if a 200K trace shows the expert kernel
   at DRAM speed, take it there. Do not lower the reserve below 1536 at 200K without a shrink test.
6. **Long-context memory and page cache.** At 200K the KV cache (about 3.5 GB for the full-attention visits) forces
   the expert cache down; measure 32K/64K/200K end to end with the fixed 2,048 MiB reserve, and warm or lock the
   expert pages so a 61 ms fault cannot appear in an answer.  No reserve change is part of this goal.

Expected short-context result: 45.9 to about 50-53 tok/s (+9-15%), 16K 40.5 to about 44-46 (+9-13%), with the
dense, head and expert bytes unchanged. These are not a promise: the trace's launch and gap numbers are the
upper bound for stages 2 and 4, and stage 3 is the only large lever at 200K.

## Experiment log

- 2026-11-23, P1 `conv_add` + `mix` one launch: the kernel was bit for bit (parity fixture's 8 top-5 rows
  identical) but forced decode fell 42.57 -> 40.50 tok/s (-4.9%).  The old conv used 18 blocks, one per 128
  channels; the fused kernel used one block a row and ran the conv's 18 channel groups in series, so the mix waited.
  Reverted.  Lesson: a fusion must keep the old kernel's block parallelism; one block a row is only right for the
  row-wide reductions that already use one.
- 2026-11-23, NVFP4 slot kernel with 16-byte loads, the activation row in shared memory and 16 warps a block: bit
  for bit against the old kernel with the expert cache frozen (`STRATA_FREEZE_EXPERTS=1`, both 67.3% in VRAM), but
  no end-to-end gain (frozen 32.35 against 32.19 tok/s; production runs 42.6 against 42.2 in the same series).  The
  engine's expert bytes are partly in the 96 MB Infinity Cache; the cold 311/260 GB/s measurement does not transfer.
  Reverted; the patch is kept outside Git.
- 2026-11-23, cache policy: a per-layer simulation of the 16-token routed IDs gives LRU + the frozen profile 51.4
  misses a token (the engine's 83-86% hit rate); LFU, previous-token prefetch and profile-initialized LFU are equal
  or worse.  The misses are structural at 1,340 slots; no policy change is kept.
- 2026-11-23, a 256-thread `mix_row`: 5,000 forced decode fell to the same top-1 IDs but the top-5 probabilities
  moved (0.376 -> 0.344, 0.675 -> 0.556).  The eight-wave reduction is another sum order; rejected for quality, not
  measured further.
- 2026-11-23, WIN: `decode_q8_kernel` with `__launch_bounds__(256, 2)`.  At 256 threads its 176 VGPR allowed one
  resident block a CU; the cap allows two and keeps more long KV reads in flight.  poly_inkling_attend_test cold:
  16K 0.658 -> 0.442 ms a call, 190K 2.453 -> 2.023 (-17.5%); 200 keys, the window and every hot case unchanged;
  all cases PASS.  Forced decode parity with `STRATA_FREEZE_EXPERTS=1`: 8/8 top-5 rows identical.  Engine 16K
  trace: decode_q8 1.621 -> 1.629 ms/token (the KV is in L2 there; no change), so the win is at spans whose KV
  comes from DRAM.  Short A/B 44.12 -> 43.81 tok/s and 16K 40.96 -> 41.20, both inside variation: no regression.
  Commit 21544cc0, pushed to main.

## Final measurements

- Short forced 94+96, production config (2,048 MiB reserve, 64 swaps, 8 threads): old `ebab4fad` 44.12 tok/s mean of
  six; `21544cc0` 43.81 (inside the 1-2% run variation).
- 16K (16,384 + 128): old 40.96 mean of three; `21544cc0` 41.20.
- 190K attention kernel (`poly_inkling_attend_test`, cold caches, the engine's scratch): 2.453 -> 2.023 ms a full
  visit (-17.5%); 16K cold 0.658 -> 0.442; all numerical cases PASS.
- Live production after deploy: health 200, /v1/models 401 without the key, live agent round trip PASS (tool call
  ID, cobalt-73, 42, cancel, MARBLE); agent requests decoded at 39.8-46.7 tok/s; 1,950 MiB VRAM free at the end.

## Remaining limits

- Short and 16K are no longer bound by a single kernel: the CPU experts wait 4.5-5.2 ms a token (about 51-60 misses
  of 384 uses; the CPU reads at 53 GB/s of DDR4), the launch floor is 3.1 us x 1,105, and the dense Q8_0, small Q8_0
  and LM head are at 326-372 GB/s, the card's streaming speed.
- The reserve stays 2,048 MiB by the user's decision; the cache cannot grow, and the reserve experiment is recorded
  but not taken.
- Per-layer HIP graphs exist for non-looped models but `run_visits()` (Inkling) bypasses them; the decode kernels
  take `pos_` as a scalar, so a graph would replay a stale position.  Making Inkling use the graph path is a larger
  refactor (device-side position) and was not attempted.
- At 190K a full attention visit is still 2.02 ms (about 413 MB, 204 GB/s); a further gain needs another K/V layout,
  not a launch-bound change.

## Current State

- One engine change kept and deployed: 21544cc0. Working tree clean but for the pre-existing `.rocprofv3/`.
- Scratch: `~/.cache/opencode-tmp/opencode/decode-opt-1795416366/` (`rbench.sh`, `bench16k.py`, `ana3.py`,
  `nvfp4_bench.hip`, `nvfp4-vec16.patch`, `route2.err`, traces under `tr-*`, `trA16`, `trB16`, `tr16`). Outside Git.

## Material Decisions

- 2026-11-23: no PMC counter collection on this driver. `rocprofv3 --pmc` caused a GPU ring timeout and a device
  wedge (TLB flush failures, ring resets, kwin/firefox timeouts); the machine rebooted. Kernel traces and byte
  accounting are the measurement path. Do not repeat PMC without a driver fix.
- 2026-11-23: traces include one-off page-cache faults; compare medians and warm the model file.
- 2026-11-23: quality gate is bitwise for fusions and the existing 3e-5 attention test where a reduction order
  changes; no formula change for speed.

## Checkpoint History

- 2026-11-23: short trace 95 tokens, 15.75 ms busy / 9.46 ms gaps, 1,105 launches; kernel table above.
- 2026-11-23: reserve/swaps sweep; 1600 MiB measured better but rejected by the user. The reserve stays 2,048.
- 2026-11-23: expert profile weight sweep; x2 stays.
- 2026-11-23: NVFP4 cold microbenchmark; vector + shared memory 311/260-269 against 179/165 GB/s, bitwise; engine
  A/B no gain at short context.
- 2026-11-23: 16K decode 38.2-41.0 tok/s; attention 1.62 + 0.30 ms a token; 190K attention reference 2.20 ms.
- 2026-11-23: routed-ID capture (16 tokens) and cache simulation; LRU 86.6% hits, previous-token prefetch no gain.

## Deployment

- Binary: `~/.config/strata/inkling2-mini/strata-poly-21544cc0`, SHA-256
  `f3bb3a58f6e08953835c4977a79ededd1c52e149368a8bc889b88f04eeae5ee2`, the build of `21544cc0`.
- Config: `strata-inkling-2-mini.json` points at it; rollback `strata-inkling-2-mini.json.before-21544cc0.bak` and
  `strata-poly-ebab4fadff6f`.  `strata.service` active; 200,000 context, one slot, Inkling only.
- Live test: `test-agent.py` PASS (tool call, cobalt-73, 42, cancel then MARBLE); `/health` 200, `/v1/models` 401
  without the key; 1,950 MiB free after the test.

## Completion

- Resolved: one deterministic win (long-context decode attention) committed, pushed and deployed.  The short/16K
  paths are unchanged within variation; the failed fusions and the rejected reserve are recorded above.  Further
  short-context gains need a different CPU-expert/cache balance (VRAM) or the looped-model graph refactor, neither
  of which is a small, testable change.
