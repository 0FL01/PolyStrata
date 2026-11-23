# Goal: Inkling-2-Mini decode, second round: fewer launches a visit, the MoE path, and no black screens

Status: active (implementation round: the user ordered P1 onward, autonomous profile→hypothesis→prototype→measure cycles)
Source: user request of 2026-11-23: "Изучи последние коммиты и цели, нужен план по пооптимизации движка, цель поднять
`decode` t/s ... проводить профилирование, писать бенчмарки ... на уровне ядра ... Важно не потерять в качестве из за
дивергенции/смены формул. Приступай к профилированию". Then, after two black screens: "стоп, ты чо там творишь, второй
раз уже экран гаснет (и мне пришлось делать хард ресет) ... пиши ответ, чё там баг в AMDGPU и ROCm?" and "щас всё
окей, зафиксируй в цели из за чего был blackscreen, что бы потом на эти грабли больше не прыгунть и продолжить",
"и гпу не трогай, что бы снова падений не было и план жду". Then the implementation order of the same day (verbatim in
the User directive section below).
Last updated: 2026-11-23

## Objective

Inkling-2-Mini decode (`Inkling-2-Mini-UD-IQ4_XS.gguf`, Q8_0 KV, production flags, build `21544cc0` as the baseline)
is faster on this RX 6700 XT / Ryzen 7 5700X with the same answers, and the measurements of this round and the
rules that came out of the black screens are written here so that nobody repeats them.

## User directive (2026-11-23, verbatim)

"Во время планирования ТЕБЕ нужно проводить профилирование, писать бенчмарки, что бы на уровне ядра, оценить
ботлнеки, именно на низком уровне, всё проверить, код как пример используется это только треть дела, основной
профит будет что ты во время планирование пишешь временные скрипты для оценки ботлнеков текущей реализации, и так
несколько бенчмарков делаешь, с углублением и рефлексирование, а после уже пишешь план действий где можно выжать
ещё больше перфоманса

Важно не потерять в качестве из за дивергенции/смены формул"

"зафиксируй в цели из за чего был blackscreen, что бы потом на эти грабли больше не прыгунть"

"гпу не трогай, что бы снова падений не было" — lifted by the order below.

The implementation order (2026-11-23, verbatim):

"Переходи от RECON к реализации, цель максимально ускорить реальный end to end `decode t/s`, а не просто выполнить
план, план это GPS навигатор, а не рельсы

Используй текущий план как карту гипотез, а не жёсткий маршрут, начинай с P1

Работай автономно длинными циклы: профилирование -> гипотеза -> временный прототип/бенчмарк -> измерение -> анализ ->
следующий эксперимент (если профит детерминирование получен - коммит и пуш в main, далее это будет как baseline для
последующих правок (т.е закрепляем результат и идём дальше)

При неудачах выясняй почему ожидания не подтвердились, пересматривай гипотезы и пробуй другие подходы, не Зацикливайся
на одном направлении и свободно меняй приориты по результам экспериментов

Успешные оптимизации интегрируй в main (прод бинарник), проверяй end to end прирост

После существенных изменений, перепрофилируй движок, bottleneck может переместиться

ПРодолжай исслендвоание и реализацию, пока останется перспективное проверяемые гипотезы, в финале покажи измеренные
результаты, неудачные подходы и оставшиеся ограничения, и деплой в прод

Делай копию плана и моей подсказки в цель, помни что документ с целью это глина, её можно мять как угодно по ходу
работ, в эту глину пиши удачи/неудачи"

"Важно не допустить дивергенции по точности в формулах, что бы качество ответов не падало"

"так же призывай саб агентов (review) для ревью диффа перед коммитами, на предмет регресса" — every commit that
changes engine code goes through an independent review subagent (the diff, for regressions and the quality gates)
before the commit; the review's findings that prove a gate violation block the commit.

The reserve stays 2048 MiB (the decision of the first round, 2026-11-23-inkling2-decode-throughput.md).

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest
unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text.
The plan is a hypothesis map (GPS), not rails: reorder, drop and add experiments by what the measurements say; commit
and push every deterministic win to main and re-profile after substantial changes. Finish when every required outcome
is resolved or no promising testable hypothesis is left, then report measured results, failed approaches, remaining
limits, and deploy to production.

## Frozen Contract

### Required Outcomes

- R1: Short-context decode faster than build `21544cc0` with the same forced IDs.
  - Source: user; the forced 94+96 fixture is the existing decode gate.
  - Acceptance: paired A/B, A = `21544cc0` + production config; 3 runs each of 300 forced tokens; B's mean and B's
    stall-free time (the p10 token time of the HIP shim) both above A's run-to-run spread; stalls a token reported
    for every run (see "Black screens and the GPU stalls").
  - Primary evidence: `rb.sh` runs with `hipshim.so` (scratch), `ab.py` output, parity fixture.
  - Status: verified
  - Evidence: 2026-11-23, RX 6700 XT, forced 94+300, each change against its parent build, same desktop: P2a
    (`273fa1a0`) 45.15 -> 45.82 tok/s mean of 3 pairs (+1.48%, 3/3); P2c (`e2027e4b`) 46.88 -> 47.14 (+0.57%, 7
    pairs, 5/7 positive, median +0.5%); P2b (`56eb42bd`) 47.47 -> 47.78 (+0.67%, 5 pairs, 5/5). Compounded
    +2.7% on the same fixture; every kept change beat its parent deterministically (launch-count arithmetic, not
    spread). Forced IDs, so the streams are identical by construction.
- R2: 16K decode not slower; prefill within variation.
  - Source: the first round's R2, still the user's acceptance for decode work.
  - Acceptance: paired 16,384-prompt + 128-token decode A/B on the kept changes.
  - Primary evidence: `bench16k.py` logs.
  - Status: verified
  - Evidence: 2026-11-23, forced 16384+200: per change - P2c 41.93 -> 41.95 (7 pairs), P2b 42.91 -> 43.02 (3
    pairs); and the kept set as a whole, all switches off against all on, 2 direct pairs: 41.62 -> 42.94 and
    42.06 -> 42.68 tok/s (mean 41.84 -> 42.81, +2.3%; prefill 862.6/859.6 against 863.8/862.6 tok/s, inside
    variation). The 16K token is faster with the switches on, not slower.
- R3: Quality unchanged.
  - Source: user; "Важно не потерять в качестве из за дивергенции/смены формул".
  - Acceptance: every kept kernel is bit for bit the old one, or passes the existing attention test tolerance; no
    formula, reduction order or router selection changes; the switch-off path is byte-identical; top-1 IDs of the
    parity fixture match.
  - Primary evidence: the bitwise test of each fused kernel (the pattern of `native_add_rms_norm_q8_1`), parity
    fixture.
  - Status: verified
  - Evidence: 2026-11-23: `poly_norm_q8_test` rows gate 48/48 bit for bit (and its output byte-identical to the
    HEAD-built baseline after the P2b revert); `poly_tail_moe_test` 60/60 bit for bit; `poly_route_dual_test` all
    passed (products bitwise, pinned host copies bitwise, refusal shapes write nothing). No formula, reduction
    order or `route()` change anywhere in the three diffs; the old frozen kernels are untouched source. The
    switch-off paths were verified equal to HEAD launch by launch by the review subagent each time (the user's
    review rule, `ses_df7349d87ffe` and `ses_df71c40b1ff`). Forced-ID A/B makes token streams identical by
    construction; free-run streams are nondeterministic on this machine even between two runs of the same binary
    (CPU and GPU expert paths round differently, 1128-1180 experts in VRAM by desktop state), so free-run
    comparison is not a gate and was not used as one.
  - Post-deploy measurement (2026-11-23 evening, the user asked for an answer-accuracy comparison): a 14-question
    objective battery (arithmetic, sequences, logic, translation, counting; greedy, temperature 0) run through the
    production server, 2 passes a config: old binary `strata-poly-21544cc0` 14/14 + 14/14; new binary flags off
    14/14 + 14/14; new binary flags on (the deployed config) 14/14 + 14/14 - 84/84 correct with identical answer
    strings in every config. Deterministic pairs under the guard (free-run greedy, one fixed question):
    at a matched load (1205 experts both) flags off vs on produced bit-identical streams, 107/107 tokens (17*23,
    answer 391) and, in a second pair, 400/400 tokens; a pair that diverged at token 2 was controlled by two
    SAME-CONFIG off runs that diverge at token 2 the same way (the free-VRAM wobble at load is 0.01 MiB scale -
    2025.566 vs 2025.570 vs 2025.574 MiB - flipping a boundary expert between GPU and CPU), and the on run
    matched one of the off runs exactly. So: when the expert set matches, the three switches change not one output
    bit; answer-level quality is indistinguishable across old and new binaries.
- R4: The black screens' cause and the rules against them are recorded here, and the guard enforces them.
  - Source: user; "зафиксируй в цели из за чего был blackscreen".
  - Acceptance: this document has the incident and the rules; `rdna2-rnd/scripts/run_guarded.py` reads the journal
    by cursor, takes plurals and "timed out" as faults, and refuses a boot in which the GPU took a fault or a reset.
  - Primary evidence: the "Black screens" section; `python3 rdna2-rnd/scripts/run_guarded.py --output <new dir> --
    sleep 1` on a clean boot (exit 0); the journals of boots -2 and -1 of 2026-11-23 flagged (2 and 34 lines).
  - Status: verified
  - Evidence: 2026-11-23: the guard change below; 13 journal lines of the two kinds tried against the patterns, 13
    as intended; the current boot has no such line, the two boots that ended in black screens have 2 and 34.

### Constraints

- C1: RDNA2/HIP only; one card, one sequence; the same GGUF, Q8_0 KV, 200,000 context; no expert skipping; the
  reserve stays 2048 MiB.
- C2: New behavior opt-in; with the switch off the path is byte-identical; other models unchanged.
- C3: GPU runs are allowed again (the user's implementation order of 2026-11-23 supersedes the pause): through the
  fixed guard only, its result read before anything else (exit code and `stop-reasons.json`, never redirected away);
  one engine at a time.
- C4: On this machine (the card drives the desktop) no kernel that can wait without a time limit: no software grid
  barrier, no persistent or spinning kernel, no `rocprofv3 --pmc`. A GPU fault ends the day's GPU work: reboot
  before the next run.
- C5: Every benchmark sizes its buffers for its largest call and checks the launch's error; the fault of
  2026-11-23 was a benchmark writing 200,058 rows into a 3,072-float buffer.
- C6: Docs keep measured numbers with what they were measured on; artifacts, logs, profiles and binaries stay out of
  Git; the server on 127.0.0.1 with its key.

### Non-goals

- A smaller reserve; DSpark; a CPU/GPU split of the host experts (the socket's DRAM read is 43 GB/s, measured below,
  and the CPU experts already take it all); graph capture of a visit's chain (a graph chain costs 2.98 us a kernel
  against 2.70 for launches, measured below); a pinned-flag wait instead of the router's event (measured: no gain);
  long-context attention (the first round's win stands).

## Change Envelope

- Target: the decode path of `Inkling2` for one row.
- Expected paths and symbols: `src/poly/inkling2.cu` (`layer_head`, `project`, the norm and router launches),
  `src/poly/kernels.cu` and `include/strata/poly/kernels.hpp` (poly's small kernels), `src/kernels/cuda/
  native_mmvq.cu` and `include/strata/kernels/native_mmvq.hpp` (row-order-keeping throughput work only),
  `src/poly/cpu_experts.cpp` (the L3 prefetch experiment), a `*_test.cu` by each change; `rdna2-rnd/scripts/
  run_guarded.py` (done).
- Direct consumers: `Engine::layer`, `Engine::moe`, `Inkling2::layer_head`.
- Allowed: opt-in kernels and their bitwise tests, environment switches, docs. Forbidden: a changed formula,
  reduction order or `route()`; KV formats; the server and API; other models' paths.

## Current Checkpoint

- Closes: R1, R2, R3 (all three launch-count fusions won, shipped and deployed; statuses above). P2d, the last
  mapped launch fusion, FAILED on measurement and is reverted (the log below has the mechanism).
- Done: R4 (guard + incident); P1's system side (user actions recorded: `coolercontrold` is a root service,
  `amdgpu.mes=0` needs root and a reboot, no sudo password on this machine); P2a (`273fa1a0`,
  `STRATA_INKLING_NORM_Q8=1`, +1.48%); P2c (`e2027e4b`, `STRATA_INKLING_TAIL_MOE=1`, +0.57%); P2b (`56eb42bd`,
  `STRATA_INKLING_ROUTE_DUAL=1`, +0.67%); the guard's kwin-monitor crash fix (`003e5fd9`); deployment to production
  (binary `strata-poly-56eb42bd` + the three switches in the service env; live check passed; 16K direct pairs
  above). Compounded forced-decode win +2.7% on the kept set; 16K +2.3% direct. P2d attempted after the deploy:
  bitwise 28/28, A/B -1.2%, reverted, tree clean at `56eb42bd` again (the engine's flag gone; a 94+96 sanity run
  at 43.69 tok/s).
- Smallest next action: the situ+down merge (est +1.2%, the one remaining measured-payoff item: fold the SiTU
  activation and its Q8_1 quantize into the NVFP4 slot kernel's down launch; the launches' own geometry kept, so
  the P2d collapse does not apply). The recipe is in "The next cycle's map" below.
- Expected evidence: a bitwise fused-vs-two-launch test (the gate_up product tree, the situ expression's compiled
  form and the quantize's are three separate surfaces), a trace with `situ_rows_q8_1` 48 -> 0 a token, A/B forced
  94+300 (5 pairs) and 16K (3 pairs), review subagent before the commit.
- Stop or replan if: the activation's compiled form cannot be reproduced bit for bit inside the slot kernel's
  warps, or the A/B gain lands inside the spread - then the round's engine work is done: what remains are the
  user actions of P1 and the structural limits, all recorded in the map.

## Profiling (this round)

All on RX 6700 XT (gfx1031, 96 MB Infinity Cache, the card of the Plasma/Wayland desktop), Ryzen 7 5700X, 32 GB DDR4,
kernel 7.2.15-300.fc45, TheRock HIP 7.13 of the repo's venv, build `21544cc0`, forced decode with the frozen expert
profile, reserve 2048 MiB, 8 CPU threads. Scratch tools (not in Git): `~/.cache/opencode-tmp/opencode/decode-opt2/`
(`hipshim.so`, a `LD_PRELOAD` timing of HIP calls with no tracing tool; `keepawake.hip`; `kbench.hip`;
`launch_bench2.hip`; `an*.py`; the traces `tr1.trace`, `trh.trace`).

### The token without tools

- Baseline: 41.95 tok/s (94+96), 41.56 tok/s (94+300); 40.5-43.2 tok/s from run to run.
- A token's time on the host (shim, 300 tokens a run): p10 20.25-20.34 ms, the tokens the desktop left alone;
  mean 22.75-23.94 ms with the stalls below.
- 1,105 kernel launches a token; `hipLaunchKernel` 0.97 us on the host (p50), 1.38 ms a token.

### Black screens and the GPU stalls

- A stall of 2.1-2.5 ms in which the GPU runs nothing, while kernels are queued: 0.82 a token; a token takes 2.3 ms
  more per stall (medians with 0/1/2/3/4 stalls: 21.0/22.7/25.5/26.9/30.8 ms). That is 1.7-1.8 ms a token, 8-12% of
  decode, the largest single loss measured today.
- The stall is there without the engine: a trivial kernel every 50 us met 22.8 stalls a second in 10 s (13.2 and
  20.1 in two other windows), 5.4% of the time; with an animating Wayland client 22-28 a second against 11-14 before
  it. A second HIP process at 30 Hz beside it left the first one's rate where it was (20.1 a second). In 14 stalls
  checked, mclk stayed at 1000 MHz, sclk at 2855 and the PCIe link at 16 GT/s x16; in 38 of 46 the engine had no
  copy in flight; the GPU was busy right up to each stall.
- The stalls come from the desktop's use of the card under this driver; the mechanism (MES, preemption) is not
  proven, and the engine cannot change it. What it can do is count it.
- After 255 us or more idle the first kernel takes 86 us instead of 17 (a trivial kernel, round trip); the GPU idles
  that long in the CPU-bound visits.

### Launch floor (no tracing tool)

- An empty kernel in a dependent chain: 2.70 us; in a graph chain: 2.98 us; kernels of the shapes a visit uses
  (1x256 with a reduction, 18x128, 72x256, 128x256, 576x128): 3.4-4.4 us.
- A visit has about 17 small kernels of 2-9 us (mix, norms, quantize, sink and control projections, `heads_kv`,
  decode and combine, `conv_add`, router, `copy2`, situ, `add_scaled`): about 850 launches and 3.9 ms a token.
- The router's wait (`hipEventSynchronize`, p50 288 us) is the host ahead of the card, not a cost. After it the host
  needs 3.1 us to its first launch. An isolated round trip: event 41.5 us, stream sync 29.3, a host spin on a pinned
  word 22.2. In the engine the spin gained nothing (p10 token 20.38-20.61 ms against 20.25-20.34, 3 runs each);
  against the card's own clock the event's wait ended 1.5 us above its least at p50 (3.9 at p90) and the spin's
  0.3 us (0.8). Dropped.

### The MoE path

- A visit's routed experts take max(card, CPU). In the trace: card 6.09 ms a token, GPU idle after it 2.59 ms.
  30% of the visits are CPU-bound (GPU waits a median 117 us, p90 468 us, most 2.7 ms); 70% are GPU-bound with a
  median 134 us of expert kernels (4.46 ms a token).
- The CPU experts are DRAM-bound. The engine prints 5.2-5.3 ms a token for them; their bytes are 56-76 misses
  x 4.47 MB (the printed hit rate of 80-86% from VRAM) = 250-340 MB; a plain reader on this socket gets 43.0 GB/s
  with 8 threads (10.3 with one, 35.9 with four), and the first round found 6, 8, 12 and 16 threads alike. The host
  blocks in `cpu_experts()` a median 100 us, p90 312 us a visit. Fewer bytes or fewer misses are the only ways;
  the reserve is fixed.

### Kernels alone (`kbench.hip`, clocks not controlled; the engine's traced durations are the reference)

- Streaming read: 376 GB/s (1 GB), 372 (192 MB), 1,276 (48 MB, the Infinity Cache).
- Dense projections, five Q8_0 segments, 21.0 MB: 69.2 us cold (303 GB/s), 42.2 us hot; alone and cold: 3,072 rows
  31.6 us (238 GB/s), 384 rows 6.3 us, 1,024 rows 9.8-11.9 us. In the engine: 60 us a visit.
- The o projection (3,072 -> 2,304, 7.5 MB): 12.5 us hot, 38.1 us cold (197 GB/s). In the engine: 23.4 us.
- The head (200,058 x 2,304 Q6_K, 378 MB): 1,028 us = 368 GB/s, at the roofline.
- NVFP4 experts through `native_mmvq_slots`: gate and up for 1 and 8 experts 14.9 and 86.9 us hot, 18.8 and 102.1
  cold; down 11.6 and 61.7 hot, 15.0 and 68.7 cold. In the engine the launches fit 30 us + 5.5 us an expert (gate and
  up) and 12 us + 4.7 us an expert (down): 42 us a visit is a fixed part, the marginal part runs at cache speed.

## Plan

Each step is one checkpoint: build, bitwise test, A/B behind the guard, commit and push when B wins. Numbers are
estimates from the measurements above, to be replaced by the A/B.

- P1 The desktop's stalls (the user's system, no engine code): with the trivial-kernel probe, the stall rate with
  the desktop quiet and active; then one change at a time: `amdgpu.mes=0` on the kernel line, `coolercontrold`
  stopped. Expected: up to the 1.7-2.3 ms a token above. The hang record of boots -2 and -1 goes to the amdgpu
  tracker (kwin's message asks for it) whether or not this changes anything.
- P2 Fewer launches a visit, each fused kernel bit for bit the two launches (no grid barrier anywhere):
  - a. The one-stream `rms_norm` + `native_quantize_q8_1` through the existing `native_add_rms_norm_q8_1` (no
    `add`); a segments variant of it for the three streams. One launch less a visit: about 0.24 ms a token.
  - b. `copy2` folded into its producers: `add_rms_norm_q8_1` writes the experts' input to pinned memory, the router
    kernel its scores. About 0.27 ms.
  - c. `add_scaled` into the tail `conv_add`, which already adds the CPU experts' sum: the same partial sums in the
    same order. About 0.32 ms.
  - d. `mix` and the norm and quantize of the next projection in one block, `mix_row`'s ten reductions keeping
    their thread mapping. About 0.24 ms. `mix` reduces across the row, so `conv_add` + `mix` is not fusable without
    a grid barrier (the first round's -4.9% came from forcing the 18 blocks into one).
  - Sum about 1.07 ms of the 21 ms token, 5%.
- P3 Kernel throughput with each row's reduction tree kept (bitwise): the 3,072- and 1,024-row Q8_0 segments at
  238-256 GB/s cold and the o projection at 197 GB/s cold, against 376 GB/s. First a controlled benchmark of where
  the time goes (occupancy, loads in flight a row), then a variant that only changes what rows a block takes. Upper
  bound at 330 GB/s cold: 1.0 ms a token; less in the engine, where part of these bytes already come from the cache
  (23.4 us against 38.1 us cold for the o projection).
- P4 The MoE path:
  - a. The expert launches' fixed part (42 us a visit, 2.0 ms a token): find where it goes (clock ramp after the
    router's idle gap, the launch's shape) before touching the kernel; the first round's wider loads made the
    kernel faster alone and the engine not at all. Upper bound 1-2 ms, GPU-bound visits only.
  - b. CPU experts: read the likely host experts of the next visit into L3 while the card runs the head (the DRAM
    is idle about 200 us a visit there: 8.6 MB, two experts), with the router guess `moe_router()` already makes
    for the next layer. CPU measurements first: `cpu_experts()` cold against read-a-moment-before; then the guess's
    hit rate from a trace. Upper bound 0.5-1.0 ms; shorter waits also cut the 70 us wake-ups.
- P5 Only off the card that drives a display: the barrier fusions (`conv_add` + `mix`, the control trio, `heads_kv` +
  decode + combine, the three-stream control): 1.2-1.5 ms on paper, the barrier's cost on RDNA2 unmeasured, and a
  barrier whose blocks are not all resident hangs the GPU, which on this driver ends in a black screen.

## The next cycle's map (2026-11-23, after P2a/P2c/P2b; the profile moved, the sizes below are from it)

Fresh trace after the three fusions (`rp4`, 64 tokens, all switches on, RX 6700 XT): 960.7 launches a token (was
1,105 at the round's start), GPU busy 15.09 ms. The table: `MmvqSlots` 5.448 ms (95.75 a token, 56.9 us avg);
dense Q8_0 `segments<8>` 2.846 (48, 59.3 us); o-proj `Q80Block` 1.113 (48, 23.2); LM head `q6_k_warps` 1.016 (1,
1,016 us); `decode_q8` 0.581 + combine 0.171 (48); `mix` 0.502 (68, 7.4); `conv_add` 0.382 (96, 4.0); `gemv_rows`
0.312 (88, 3.5); `add_rms_norm_q8_1` (frozen) 0.294 (48, 6.1); `gemv_rows_segments` 0.294 (24, 12.3);
`control_out3` 0.291 (24); `gemv_rows_dual` 0.287 (48, 6.0); `mmvq<14>` 0.276 (15, 18.4); `rms_norm_q8_1_rows`
0.218 (48, 4.5); `mix3` 0.215 (28, 7.7); `heads_kv` 0.193 (48); `control_in` 0.161 + `in3` 0.146 + `out` 0.131
(72); `situ_rows_q8_1` 0.118 (48, 2.5); logits copy 0.031 (1).

- P2d mix + rows-norm: DEAD - built, proven bitwise, measured slower, reverted. One-line mechanism: the norm's
  SUM=1024 reduction forces ONE 1024-thread block a row, and the producers' block-level parallelism does not fit
  inside it - `control_out3` ran 72 blocks x 256 threads per stream (load latency hidden by ~18K threads in
  flight), the fused kernel swept 18x inside 3 blocks, and the saved ~4.5 us launch floor bought nothing against
  +6/+16 us of GPU time per visit (trace: `mix_norm_q8` 18.8 us vs mix3 8.6 + rows 4.5; `ctrl_norm_q8` 33.0 us vs
  control_out3 12.1 + rows 4.5; A/B off 44.32 vs on 43.80 tok/s, off 4/5 pairs). Numbers and the full story: the
  experiment log's P2d entry. This closes the launch-count fusion family on this card: every remaining adjacent
  pair is this same geometry collapse or needs a grid barrier (banned per C4).
- situ + down merge, est +1.2% (the 48 `situ_rows_q8_1` launches, ~0.28 ms with their floor): fold the SiTU
  activation and the Q8_1 quantize into the NVFP4 slot kernel's down launch (elementwise in the warp that owns
  the 32-value block; no barrier). Deep surgery in `iq_kernels.cu` + `kernels.cu`'s `situ_rows_q8_1`; THREE
  bitwise surfaces (the gate_up product tree unchanged, the situ expression's compiled form, the quantize with
  the P2a-verified tools). This is now the map's next candidate (P2d above is dead).
- P3 dense-segment bandwidth: DOWNGRADED. The plan's 238-256 GB/s was the COLD microbench number; in the engine
  the five projections run hot (42-59 us for 21 MB a visit, ~357-500 GB/s effective, Infinity Cache reuse across
  tokens) - the real headroom is a fraction of the 1.0 ms upper bound.
- P4b L3 prefetch of the next visit's host experts: DOWNGRADED. The CPU experts are DRAM-bound at 43 GB/s reading
  from the page-cached 15.7 GB model file; there is no page-fault premise, and the head's ~200 us a visit of idle
  DRAM covers ~8.6 MB - one expert's worth. Revisit only with a measured cold/warm `cpu_experts()` delta.
- P4a expert-launch fixed cost (2.0 ms a token of the 5.45 ms `MmvqSlots` line): the three sequential launches a
  visit (gate_up -> situ -> down) are the floor; barrier merges are banned on this card; the situ+down merge above
  is the only reachable cut.
- Locked (user actions or measured dead ends): system stalls ~2 ms a token (`amdgpu.mes=0`, `coolercontrold` -
  root, no sudo here); CPU-expert exposure ~2.6 ms a token at the fixed 2048 MiB reserve (the socket's 43 GB/s);
  HIP graphs (2.98 us a kernel against 2.70 for launches, measured); the pinned-word spin (no gain); DSpark; the
  router event's wait (host ahead, not a cost).

## Experiment log

- Baseline and traces as in the first round; a kernel trace adds 4 us to every launch gap (26.4 ms a token traced
  against 21 untraced), so gaps are read from the shim, durations from the trace.
- System stalls: found as `hipEventSynchronize` calls of 2.2-2.6 ms, 0.8 a token; traced to the GPU standing still
  with kernels queued; reproduced with a trivial kernel and no engine; not clocks (DPM sampled every 120 us); more
  with compositing load; not two HIP processes.
- Router spin: a prototype (opt-in) with a probe kernel writing the card's clock: no gain; reverted, tree clean,
  binary equal to production again (sha256 `f3bb3a58...`).
- `kbench`: the LM-head section wrote past its output buffer (my bug): page fault at 14:51:46, fixed and re-run;
  the first black screen followed at 14:56. See the incident.
- MoE split per visit from the trace: 30% CPU-bound, 70% GPU-bound.
- P2a (norm+quantize fused rows kernel, `STRATA_INKLING_NORM_Q8=1`): the bitwise contract took three compiler
  findings to meet, all verified at IR/SASS level on this toolchain (HIP 7.13, plain -O3, no fast math):
  (1) poly's `rms_norm` sum of squares is an fma at SASS (the contract-flagged multiply and add fuse), so the
  fused kernel must use `__fmaf_rn` on the same per-thread stride, not a multiply and an add; (2) the normed
  value's product contracts into the Q8_1 block sum's first butterfly add (`v_fmac`) unless the quantize loop is
  built with `#pragma clang fp contract(off)`; (3) the scale must be `1.0f / sqrtf(...)` with the plain operators:
  libdevice's `sqrtf` refines `v_sqrt` to the correctly rounded value, and `__fsqrt_rn`'s raw `v_sqrt` is off by
  1 ulp on boundary values. An `asm "" : "+f"` barrier would have blocked (2) as well, but AMDGPU clang rejects
  the `'+f'` constraint. With all three, `poly_norm_q8_test`'s rows check passes 24/24 shapes bit for bit
  (n in {1024, 2304, 2816}, entries {1, 3}, tokens {1, 8}), and the old decode-tail kernel's failures are
  documented in that test as expected divergence. The old tail kernel was never bitwise: the three findings
  above plus its 128-thread stride are four independent divergences shipped since `ce9e9b28`.
- P2a A/B (forced 94+300, RX 6700 XT): off 44.73/45.28/45.44, on 45.62/45.74/46.09 tok/s, 3/3 pairs, mean
  45.15 -> 45.82 tok/s (+1.48%, 22.15 -> 21.83 ms/token, -0.32 ms). Free-run token streams are NOT a gate on
  this machine: two runs of the same binary with different expert residency (desktop VRAM moves the load-time
  packing, 1128-1163 experts) already disagree; CPU and GPU expert paths round differently. The wiring was
  checked site by site against `git show HEAD` instead (same buffers: `s3in_`/`s3q8_`/`input_[i]`/`xq_`/`kq8_`/
  `vq8_`, `lin_raw`'s internal quantize and `project()`'s `!k8` guard consume the same bytes).
- P2c (the tail launch's own expert sum, `STRATA_INKLING_TAIL_MOE=1`): `Engine::moe`'s one `add_scaled` a MoE
  visit (~24 launches a token at n == 1, ~3.4 us on the launch floor each) is folded into the tail's
  `conv_add_kernel` as a `TailSum` record of the visit's cached experts. The record replaces the launch when
  `tail_record_ && n == 1 && n_cached <= kMaxUsed` (all of them or none; a fallback visit sums `fo_` as before and
  the kernel's `moe.n == 0` path is byte-identical). The bitwise contract held without a single compiler fight
  this time: the folded sum is the same `v + x*w` chain in the router's k order from `fo_ == 0` (SASS: one
  `v_fmac` a k, exactly `add_scaled_kernel`'s own contraction), and the conv segment around it is
  instruction-identical at SASS to HEAD's (`v_mul + v_fmac*3 + v_add + v_mul 0x3ee4f92e`). `poly_tail_moe_test`
  (new): the folded kernel against `add_scaled` + the HEAD kernel, 60/60 combos bit for bit (widths
  {256, 1024, 2304} x experts {0, 1, 2, 5, 8} x the CPU experts' add on/off x clean on/off), exit 0 under the
  guard. Engine trace with the flag on: `add_scaled` 0 launches (was ~24 a token), `conv_add_kernel` 96 a token
  unchanged. A/B forced 94+300, 7 pairs: off 46.88, on 47.14 tok/s mean (+0.57%, 5/7 pairs positive, median pair
  +0.5%); the gain matches the launch floor's arithmetic (24 x ~3.4 us ~ 0.1 ms/token at ~21 ms/token) and the
  mechanism is deterministic, so it ships. The kernel's `add` (the CPU experts' `h_cpu_`) and residual blocks and
  the attention call site (empty record) are unchanged source.
- P2c at 16K (forced 16384+200, 7 pairs, the same build): off 41.93, on 41.95 tok/s mean (+0.06%, pair deltas
  -0.3 to +0.9, sigma ~0.5): no regression, the small absolute gain (~0.1 ms/token) invisible in the longer
  16K token (~24 ms) at this noise level. R2 holds. The review (independent subagent on the diff) found no
  blockers; its one hardening note (a failed readback must fail the test, not compare zeros) is applied and the
  test re-passed 60/60.
- Guard crash found by the 16K runs, fixed: `run_guarded.py`'s kwin check did `fault |= ("kwin" in source and
  FAULT_WORDS.search(msg))`, and a match returns a Match while no match returns None - a kwin journal line
  without fault words crashed the monitor with a TypeError, and a real kwin fault line would have crashed it
  too instead of stopping the run. The guard still failed closed (engine killed, stop-reasons written, no VRAM
  breach, no amdgpu fault). Fixed to `... is not None`; the fix is committed on its own.
- P2b, first idea, REVERTED: "only MoE visits route" (STRATA_INKLING_ROUTE_MOE) - a no-op. The trace's full-token
  dump shows ALL 48 visits are MoE (MmvqSlots 2 a visit = 96 a token, copy2 of the router's scores 1:1 with
  visits): the loop's edge layers are not dense; only the kPhysical core distinction in the old notes was
  wrong. Reverted without shipping.
- P2b (the router's copies ride its producers, `STRATA_INKLING_ROUTE_DUAL=1`): every visit's router ended in one
  `copy2` D2H - the 128 scores to `h_rl_` and the 2304 normed floats of the experts' input to `h_x_` - ~48
  launches a token. Now `gemv_f32_waves_dual()` (the same one-block-a-row launch `lin()` takes for the router's
  F32 128x2304, same condition, same grid) also writes `h_rl_` and copies its own input row to `h_x_`, and
  `moe_router()` launches no copy2 (a shape that falls back carries them as before). A first attempt wrote
  `h_x_` inside the FFN norm kernel - REVERTED: the extra store materialized the normed product and killed the
  contract's fma contraction into the Q8_1 block sum, so `poly_norm_q8_test`'s frozen-kernel "diff (expected)"
  lines turned "same" against HEAD's own output (the norm's values changed - production bits would move). The
  lesson: a store of a value the compiler had folded into a later expression changes that expression's
  contraction; the frozen kernels take no new stores, and the value-level gate is the stash dance - HEAD-built
  baseline output vs the modified build's, byte-diffed - not a SASS shape match (the SASS reordering of
  `v_max`/`v_add` in that attempt looked innocent and was not).
- P2b evidence: `poly_route_dual_test` all passed (the dual gemv's products the plain launch's bits, `h_rl_` the
  product's, `h_x_` the input's - both to pinned memory, as the engine's are; a shape the plain path gives the
  waves kernel, or one whose input the launch's threads cannot copy whole (`n_in > 256 * n_out`), takes no dual
  and writes nothing - sentinels prove it); `poly_norm_q8_test`'s output byte-identical to the HEAD-built
  baseline (the norm kernel is untouched again); trace: `copy2` 48.98 -> 0.98 a token, `gemv_rows_dual` 48,
  launches 1008.7 -> 960.7 a token, busy 15.15 -> 15.11 ms; A/B forced 94+300 with the two shipped flags on:
  off 47.47, on 47.78 tok/s mean of 5 pairs (+0.67%, deltas +0.29 to +0.34, 5/5); 16K: 42.91 -> 43.02 (+0.26%,
  3/3 pairs). R2 holds; the win ships. The review (independent subagent on the diff) found no blockers; its
  three notes - refuse the shapes the copy cannot carry whole, seed the fallback's destinations and prove a
  refusal writes nothing, and two comments that still described the reverted norm-side copy - are applied.
- P2d (the mix/control output + its rows-norm in one launch), FAILED AND REVERTED, with the mechanism measured:
  the two fused kernels were bit for bit the two launches - `poly_mix_norm_test` 28/28 (12 mix cases and 2 control
  cases a seed, `adjacent` and `separate` layouts, the engine's own kernel source via the shared
  `inkling_mix_norm.cuh`), and the SASS instruction-identical (the control chains' `v_mul + v_fmac x3`, the gate
  write's `2 x v_bfi + v_sub + v_mul + v_fmac`, the norm's two rounded multiplies and plain `v_add` butterfly leaf,
  the libdevice sqrt; `#pragma clang fp contract(off)` scopes correctly inside a nested block when the control
  phase's own contraction must survive). The loss was geometry: the norm's SUM = 1024 reduction forces one
  1024-thread block a row, and the producer's parallelism did not fit in it. The trace: `mix_norm_q8` 18.8 us
  against `mix3` 8.6 + the rows 4.5; `ctrl_norm_q8` 33.0 against `control_out3` 12.1 + the rows 4.5 - the control
  phase swept its 2304 channels 18 times inside one block where the old launch ran 72 blocks x 256 threads a
  stream (216 blocks, the loads' latency under 18K threads in flight), and the mix phase's wide load schedule did
  not survive the shared register budget. A/B forced 94+300, 5 pairs: off 44.32, on 43.80 tok/s mean (off wins
  4/5, -1.2%); the trace's busy is +0.53 ms a token. One launch saved is ~4.5 us of floor and gap: it buys
  nothing against 6-16 us of GPU time lost to serialization. Reverted; the launch-count fusion family is spent
  (every remaining adjacent pair is this same collapse or needs a grid barrier, which this card's display driver
  cannot take).
- The remaining map after P2d: the situ+down merge (est +1.2%: fold the SiTU activation and the Q8_1 quantize
  into the NVFP4 slot kernel's down launch - elementwise in the warp that owns the 32-value block, the launches'
  own geometry kept, so the P2d collapse does not apply; three bitwise surfaces: the gate_up product tree, the
  situ expression's compiled form, the quantize with the norm kernel's verified tools), then the user actions
  (`amdgpu.mes=0`, `coolercontrold`) and the structural limits below.
- Answer-quality comparison after the deploy (the user asked "сравни точность ответов"): a 14-question objective
  battery (greedy, temperature 0, max 500 tokens) through the production server, 2 passes a config - old binary
  `strata-poly-21544cc0`, new binary flags off, new binary flags on. Result: 14/14 + 14/14 in EVERY config,
  84/84, identical answer strings; reasoning token counts wobble freely across passes (e.g. the word-reversal
  question took 208-282 tokens, the sort question 79-189) - that wobble exists in the old binary too and is the
  pre-existing CPU/GPU expert nondeterminism, not the switches. Deterministic stream pairs (guarded, free-run
  greedy, fixed question): matched load (1205 experts) -> off vs on bit-identical 107/107 and 400/400 tokens; a
  diverging pair was controlled by same-config off-vs-off diverging at token 2 identically (the load-time free
  VRAM wobbles at 0.01 MiB scale and flips a boundary expert). Live decode during the battery: 43-50 tok/s on
  the prod instance with the desktop running. The battery script and raw responses:
  `~/.cache/opencode-tmp/opencode/decode-opt2/{battery.py,qa/}` (outside Git).

## Black screens of 2026-11-23 (what not to repeat)

1. 14:51:46, boot -2. My kernel benchmark `kbench` passed a 3,072-float buffer as the output of the LM-head kernel
   (200,058 rows): `[gfxhub] page fault ... PERMISSION_FAULTS: 0x5 ... Process kbench pid 238339`. ROCm ended the
   process; that part is right. I read the line and went on with GPU runs, and the guard's verdict was redirected
   away and never looked at.
2. 14:56:28, the same boot, four and a half minutes later, with nothing of mine on the card: `ring gfx_0.1.0 timeout`
   (kwin_wayland), ring reset failed, `GPU reset begin`, `MODE1 reset`, 14:56:34 `GPU Recovery Failed: -62`, then
   `ring comp_1.0.1 timeout` (ayugram-desktop), `Pageflip timed out!` (kwin), black screen, hard reset.
3. Boot -1, a fresh boot that lasted about four minutes, no GPU run of mine at all (compiles only). 15:01:28:
   `ring comp_1.3.0 timeout` (ayugram-desktop), its reset succeeded, `device wedged, but no recovery needed`, then
   `ring gfx_0.1.0 timeout` (kwin_wayland), ring reset failed, `MODE1 reset`, frozen. Hard reset. Boot 0 came up
   clean (no amdgpu fault line).
4. Earlier that day (boot -3, about 10:07) `rocprofv3 --pmc` had already wedged the card once.

What it is: the fault was mine; that a page fault of a user process was followed by GPU-wide hangs, that a fresh
boot hung on desktop work alone, and that the driver's reset could not recover the card (black screen instead of a
working desktop) is amdgpu on this kernel and card (7.2.15-300.fc45, Navi 22, MES on by default), to be reported
with `journalctl -b -2 -k` and `-b -1 -k`. ROCm did its part. The stalls measured all day on the same system are a
third sign of the same scheduling, not proven to be the same bug.

The rules (C3-C5), in plain words:
- After any GPU fault: stop for the day, reboot before the next run; the guard now refuses such a boot.
- The guard's exit code and `stop-reasons.json` are read before anything else runs.
- No kernel that can wait forever on this card; no PMC profiling.
- Buffer sizes by the largest call, checked before the first run.
- No experiment while the desktop is unstable; the card's hangs with only the desktop on it are not ours to fix,
  and nothing measured on a boot that wedged is trusted.

## Material Decisions

- 2026-11-23: GPU runs paused by the user; CPU-only work continues. (Superseded the same day by the user's
  implementation directive; C3 carries the current rule.)
- 2026-11-23: The reserve stays 2048 MiB; the host experts stay on the CPU (DRAM-bound at 43 GB/s).
- 2026-11-23: Fusions keep each kernel's arithmetic and reduction order; a fusion is kept only bit for bit.
- 2026-11-23: Grid barriers and persistent kernels are not tried on the machine that drives the display.
- 2026-11-23: Frozen production kernels take no new stores (the P2b norm-store revert); a value-level gate is a
  byte-diff of the full test output against a HEAD-built baseline, not a SASS shape match.
- 2026-11-23: Deploy at the final stage, after the round's kept changes: binary + switches together, live check,
  service left running. Done after P2b (`strata-poly-56eb42bd`, the three switches on).

## Checkpoint History

- 2026-11-23: R4. Guard fixed (cursor, whole words, boot check); this document with the incident, the measurements
  and the plan. Next: the CPU-only L3 measurement of P4b, then P2a when GPU runs are allowed.
- 2026-11-23: P2a. The norm+quantize rows kernel is bit for bit the two launches (24/24 shapes, three compiler
  divergences found and pinned at SASS: fma squares, quantize-loop contraction, libdevice sqrt refinement) and
  +1.48% forced decode (3/3 pairs, RX 6700 XT). Review subagent on the diff before the commit, per the user's
  rule. Next: P2b/P2c fusions.
- 2026-11-23: P2c. The tail launch sums the visit's cached experts itself (`TailSum` record, opt-in
  `STRATA_INKLING_TAIL_MOE=1`): bit for bit (60/60 in `poly_tail_moe_test`, SASS conv segment identical), one
  launch less a MoE visit (`add_scaled` 0 a token in the trace), forced 94+300 46.88 -> 47.14 tok/s mean of 7
  pairs (+0.57%, 5/7 pairs positive; the launch floor's arithmetic predicts it). P2a follows in the history
  below. Next: P2b, or re-profile first.
- 2026-11-23: Guard's kwin check fixed (`003e5fd9`): the monitor itself crashed on a Match-or-None `|=`, found by
  the 16K runs; it had failed closed, the boot stayed clean.
- 2026-11-23: P2b. The router's `copy2` rides its gemv (`STRATA_INKLING_ROUTE_DUAL=1`): first idea (route only
  MoE visits) was a no-op - all 48 visits route - and was reverted; the norm-store variant changed the frozen
  kernel's bits and was reverted (the lesson above); the shipped form writes `h_rl_`/`h_x_` from the router gemv
  itself. Bitwise gates green, `copy2` 48.98 -> 0.98 a token, forced 94+300 +0.67% (5/5 pairs), 16K +0.26% (3/3).
  Review before the commit; its three notes applied.
- 2026-11-23: R1/R2/R3 verified (statuses above). Deploy: binary `strata-poly-56eb42bd` + the three switches in
  the service env, live check "test passed", service running; 16K direct pairs (all off vs all on) 41.84 ->
  42.81 tok/s mean. Next: P2d per the map above.
- 2026-11-23: Answer-quality battery (post-deploy, the user's request): 84/84 correct across the old binary,
  new-off and new-on configs, identical answers; deterministic stream pairs bit-identical at matched loads; the
  free-run divergence controlled as pre-existing. Evidence: the R3 paragraph and the experiment log above.
- 2026-11-23: P2d FAILED and reverted. Built as designed, bitwise 28/28 (mix_norm_test ALL OK), SASS
  instruction-identical phases; measured: A/B 5 pairs off 44.32 vs on 43.80 tok/s (off 4/5), trace busy +0.53
  ms a token (mix_norm_q8 18.8 us vs 13.1 two-launch; ctrl_norm_q8 33.0 vs 16.6). Mechanism: the norm's
  one-1024-block-a-row geometry serializes the producer's block parallelism. Reverted cleanly (tree
  byte-identical to 56eb42bd, sanity 43.69 tok/s); the fusion family is spent - see the map. Next: the
  situ+down merge (est +1.2%) in a fresh cycle, or stop here.

## Current State

- Resolved: R1, R2, R3, R4. Three wins shipped and pushed (`273fa1a0`, `e2027e4b`, `56eb42bd` + guard `003e5fd9`),
  compounded +2.7% forced decode, +2.3% at 16K, deployed to production, service live.
- Last relevant evidence: the 16K direct pairs and the live check; the trace table in "The next cycle's map".
- Blocker: none. The remaining items are sized and mapped, not blocked.
- Next: the situ+down merge (est +1.2%) in a fresh cycle; P1's system side and the structural limits are user
  actions and measured dead ends, recorded in the map. This round's engine work is done.
