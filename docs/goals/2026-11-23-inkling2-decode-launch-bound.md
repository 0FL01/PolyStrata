# Goal: Inkling-2-Mini decode, second round: fewer launches a visit, the MoE path, and no black screens

Status: active (GPU runs paused by the user until said otherwise; the plan below is the map)
Source: user request of 2026-11-23: "Изучи последние коммиты и цели, нужен план по пооптимизации движка, цель поднять
`decode` t/s ... проводить профилирование, писать бенчмарки ... на уровне ядра ... Важно не потерять в качестве из за
дивергенции/смены формул. Приступай к профилированию". Then, after two black screens: "стоп, ты чо там творишь, второй
раз уже экран гаснет (и мне пришлось делать хард ресет) ... пиши ответ, чё там баг в AMDGPU и ROCm?" and "щас всё
окей, зафиксируй в цели из за чего был blackscreen, что бы потом на эти грабли больше не прыгунть и продолжить",
"и гпу не трогай, что бы снова падений не было и план жду".
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

"гпу не трогай, что бы снова падений не было"

The reserve stays 2048 MiB (the decision of the first round, 2026-11-23-inkling2-decode-throughput.md).

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest
unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text.
Stop on a fresh guard fault or a proven blocker; record the evidence and the smallest unlock. No GPU run before the
user allows one again; CPU-only work (reading code, analysis of stored traces, CPU benchmarks) goes on.

## Frozen Contract

### Required Outcomes

- R1: Short-context decode faster than build `21544cc0` with the same forced IDs.
  - Source: user; the forced 94+96 fixture is the existing decode gate.
  - Acceptance: paired A/B, A = `21544cc0` + production config; 3 runs each of 300 forced tokens; B's mean and B's
    stall-free time (the p10 token time of the HIP shim) both above A's run-to-run spread; stalls a token reported
    for every run (see "Black screens and the GPU stalls").
  - Primary evidence: `rb.sh` runs with `hipshim.so` (scratch), `ab.py` output, parity fixture.
  - Status: pending
- R2: 16K decode not slower; prefill within variation.
  - Source: the first round's R2, still the user's acceptance for decode work.
  - Acceptance: paired 16,384-prompt + 128-token decode A/B on the kept changes.
  - Primary evidence: `bench16k.py` logs.
  - Status: pending
- R3: Quality unchanged.
  - Source: user; "Важно не потерять в качестве из за дивергенции/смены формул".
  - Acceptance: every kept kernel is bit for bit the old one, or passes the existing attention test tolerance; no
    formula, reduction order or router selection changes; the switch-off path is byte-identical; top-1 IDs of the
    parity fixture match.
  - Primary evidence: the bitwise test of each fused kernel (the pattern of `native_add_rms_norm_q8_1`), parity
    fixture.
  - Status: pending
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
- C3: No GPU run until the user allows it. Then: through the fixed guard only, its result read before anything else
  (exit code and `stop-reasons.json`, never redirected away); one engine at a time.
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

- Closes: R4.
- Done: the guard reads the journal from a cursor and by whole words (plurals, "timed out"), and refuses a boot in
  which amdgpu logged a page fault, a ring timeout, a wedge or a reset. Checked with 13 journal lines of both days'
  kinds and a harmless command on the clean boot.
- Smallest next action (CPU only, no GPU): time `cpu_experts()` for 1-4 experts with the weights cold and with them
  read a moment before (L3), the first measurement of P4b.
- Expected evidence: ms an expert both ways. If the warm case is not at least 1.5 times faster, P4b is dropped.
- Stop or replan if: the user allows GPU runs again: then P2a first, with its bitwise test, behind the guard.

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

- 2026-11-23: GPU runs paused by the user; CPU-only work continues.
- 2026-11-23: The reserve stays 2048 MiB; the host experts stay on the CPU (DRAM-bound at 43 GB/s).
- 2026-11-23: Fusions keep each kernel's arithmetic and reduction order; a fusion is kept only bit for bit.
- 2026-11-23: Grid barriers and persistent kernels are not tried on the machine that drives the display.

## Checkpoint History

- 2026-11-23: R4. Guard fixed (cursor, whole words, boot check); this document with the incident, the measurements
  and the plan. Next: the CPU-only L3 measurement of P4b, then P2a when GPU runs are allowed.

## Current State

- Resolved: R4.
- Last relevant evidence: the guard's 13-line check and the two flagged journals.
- Blocker: R1-R3 need GPU runs; the user said not to touch the GPU.
- Next: P4b's CPU measurement; P2a when allowed.
