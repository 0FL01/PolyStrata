# Goal: Inkling-2-Mini next levers on RDNA2 (decode, NVFP4 MMQ, prompt idle, R5, R6, 200K)

Status: active (RECON done and measured; plan below; no engine code changed in this round yet)
Source: user instruction of 2026-11-23 (verbatim below); the limits listed at the end of
`docs/goals/2026-11-23-inkling2-fork-vectors.md` (session 3 account)
Last updated: 2026-11-23

## Objective

Each soft goal the user listed is either moved by a measured, opt-in, committed change (deployed when it wins
end to end) or closed with a measured reason it does not pay on this card. The plan below is a map of
hypotheses with their measured upper bounds, not a fixed route: it is reshaped by each result.

## User directive (2026-11-23, verbatim)

> Изучи последние коммиты и цели, нужен план по оптимизации
> Во время планирования ТЕБЕ нужно проводить профилирование, писать бенчмарки, что бы на уровне ядра, оценить
> ботлнеки, именно на низком уровне, всё проверить, код как пример используется это только треть дела, основной
> профит будет что ты во время планирование пишешь временные скрипты для оценки ботлнеков текущей реализации, и так
> несколько бенчмарков делаешь, с углублением и рефлексирование, а после уже пишешь план действий где можно выжать
> ещё больше перфоманса
>
> Приступай к профилированию
>
> подсказка, далее сложить все гипотезы в цель (документ в репозитории сделать с планом работ), итеративно
> прощупать и после предложить план, уточняю, план это езда по GPS, а не по рельсам, план мнём как глину (глиномес)
>
> Цели (абстрактно, мягкие цели):
>
> - Decode при коротком контексте упирается в полосу DRAM, ядра уже на 327–370 GB/s из 380.
> - NVFP4 MMQ ограничен числом инструкций, а J=64 — максимальный тайл для RDNA2.
> - Простой GPU в префилле 32K всё ещё ~3.1 с.
> - Не сделаны R5 (слияние situ+down) и R6 (сохранение KV на диск).
> - На 200K сквозного замера нет; ускорение ядра full-attention в 2 раза — пока только оценка.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the
smallest unresolved outcome; the plan's order is a default, reorder by measured results. Do not add requirements
from reviews, tests, tools, speculative risks, or optional source text. A measured win is committed and pushed to
`main` as the next baseline; a refuted hypothesis is recorded with its mechanism. Finish when every required
outcome is resolved and affected constraints remain satisfied.

## Frozen Contract

### Required Outcomes

- R1: Short-context decode: the largest measured lever below is shipped opt-in with an A/B win, or every lever
  in the decode map is recorded as measured dead.
  - Source: soft goal 1.
  - Acceptance: forced-decode A/B (2048 + 256, ABBA, live config) faster beyond the run-to-run spread; greedy
    tokens identical to the switch off, or (for a precision-level change) the `q.py` KL protocol at the FP32
    reorder noise floor.
  - Primary evidence: `ab.sh` logs, token hashes.
  - Status: pending
  - Evidence:
- R2: NVFP4 MMQ: an instruction or load-phase cut of the J64 gate/up kernel (or the tail) ships opt-in with a
  kernel and 32K end-to-end win, or the two mapped cuts are recorded as measured dead.
  - Source: soft goal 2.
  - Acceptance: `mmqbench` faster and bitwise to the current lean kernel; 32K prefill A/B faster.
  - Primary evidence: `mmqbench` dumps compared byte for byte, `ab.sh` 32K pairs.
  - Status: pending
  - Evidence:
- R3: Prompt idle: the measured steady-state idle at 32K is cut by a shipped opt-in change, or the components are
  recorded as not reachable.
  - Source: soft goal 3.
  - Acceptance: trace idle (window after the first batch) lower and 32K A/B faster; bitwise.
  - Primary evidence: `rocprofv3 --kernel-trace` window idle, `ab.sh` 32K pairs.
  - Status: pending
  - Evidence:
- R4: R5 of the previous goal (situ + down/gate_up merge) resolved: shipped bitwise with an A/B win or measured
  dead.
  - Source: soft goal 4; `2026-11-23-inkling2-fork-vectors.md` R5.
  - Acceptance: a bitwise fused-vs-unfused test plus forced-decode A/B, or a recorded measured rejection.
  - Primary evidence: a scratch bitwise test, `ab.sh` logs.
  - Status: pending
  - Evidence:
- R5: R6 of the previous goal: the KV state can be saved to disk and loaded by a fresh `strata-poly` process.
  - Source: soft goal 4; `2026-11-23-inkling2-fork-vectors.md` R6 (its acceptance is kept unchanged).
  - Acceptance: versioned, validated snapshot (conv state, SWA ring tails, full-attention rows, token list,
    position); a fresh process after RESTORE reproduces the saved process's greedy tokens and prefills only the
    new suffix; an incompatible file is refused.
  - Primary evidence: a round-trip run with a token compare; save/restore times at 32K.
  - Status: pending
  - Evidence:
- R6: 200K end to end measured against the previous production build.
  - Source: soft goal 5.
  - Acceptance: one 199,000-token prompt on each binary, live config, same tokens.
  - Primary evidence: `eng.py` serve runs under the guard.
  - Status: verified
  - Evidence: RECON below: 580.5 s (`56eb42bd`) -> 383.1 s (`f442cbe2`), the same 32 greedy tokens.

### Constraints (inherited, unchanged)

- C1: Same GGUF (`Inkling-2-Mini-UD-IQ4_XS.gguf`), Q8_0 KV, 200K context, `--vram-reserve-mib 2048` fixed.
- C2: New paths opt-in; the default path byte-identical (normalized `-S` ISA compare for kernels in shared TUs).
- C3: GPU runs only through `rdna2-rnd/scripts/run_guarded.py` (`~/.cache/claude-fv/g.sh`); production
  stopped during standalone GPU work and restored after.
- C4: No unbounded-wait or persistent kernels, no `rocprofv3 --pmc`, buffers sized by the largest call.
- C5: Fusions bitwise or within the attend-test tolerance; precision-level changes through the KL protocol.
- C6: A review subagent (general) on every engine diff before its commit.
- C7: No artifacts or logs in Git; scratch lives in `~/.cache/claude-fv/`.

### Non-goals

- Other GPU architectures; changing the reserve, KV format, quantization or prompt batch of the deployment.
- Persistent/megakernel decode (C4), HIP graphs (measured slower per kernel in the decode round).

## Change Envelope

- Target: `src/poly/` (engine.cpp/hpp, inkling2.cu/hpp, kernels.cu, main.cpp, mmq*.cu), `src/kernels/cuda/`
  (iq_kernels.cu slot kernels), own TUs for opt-in kernels, `CMakeLists.txt` source lists, tests under
  `src/poly/*_test.cu`, docs (this file, DETAILS.md settings when a switch ships).
- Forbidden: edits to `third_party/` (overrides go in our own TUs, as P2 did), new dependencies, server
  (`serve/`) changes beyond letting the existing `/slots` save/restore reach `strata-poly` for R5.

## RECON (2026-11-23, this round; RX 6700 XT gfx1031, Ryzen 7 5700X, live config of `f442cbe2`)

Tools (scratch, not in Git): `eng.py`/`g.sh`/`ktab.py`/`gaps.py` from the previous round; new: `vram/mshim.so`
(LD_PRELOAD logger of hipMalloc/hipMemCreate >= 4 MiB with callers), `slots/vbench.hip` (decode expert
mat-vec at L lanes a row, back-to-back launches over a 600 MB weight pool), `mmq/vnodot`, `mmq/vnoload` (the lean
NVFP4 MMQ with its dot or its weight-tile load compiled out, linked into `mmqbench` against the engine's
`libstrata_poly_mmq.a`), `isamix.py` (VALU mix of the hottest basic blocks), `bin/hackroute` (timing-only hack,
reverted from the tree).

### Short-context decode (2,048 prompt + 96/256 forced tokens)

- Trace (`trd2k`): 960.9 launches a token, GPU busy 16.05 ms a token; untraced 45.7-46.3 tok/s (21.7 ms).
- Expert mat-vecs are not latency-bound: per launch `t(m) = intercept + slope * m` from the trace: gate_up
  8.3 us + 8.7 us/expert (**343 GB/s marginal, 90% of DRAM**), down 6.3 us + 5.5 us/expert (271 GB/s). Down
  has 36 dot calls a row on 16 lanes (lanes 0-3 do 3, the rest 2: 75% balance). `vbench`: down at 8 lanes a row
  -9% (50.7 -> 46.3 us for 7 experts), gate_up unchanged; both change the sum order (not bitwise). Unrolling with
  a constant call count (all loads first) is bitwise but **slower** (gate_up 75.0 -> 84.8 us, down 50.7 -> 57.5):
  REFUTED as a lever.
- Router round trip: a timing-only hack that launches the GPU experts before the host waits for the router
  (`FV_HACK_ROUTE`, garbage routing, forced tokens): 46.1-46.3 -> **48.9 tok/s (+5.8%)**; part of it is the hack's
  routing landing on more VRAM experts (91.9% vs 88.5%, CPU experts 0.4 vs 0.6 ms a token), so the reachable
  bound is about **+4.5%**.
- VRAM experts: 1,180 of 3,072 in VRAM (5.0 GiB). Sensitivity (single runs, spread about +-1.5%): 945 experts
  43.2 tok/s, 1,065 44.9, 1,180 45.7-46.0, 1,286 47.2, 1,317-1,395 46.3-47.5. CPU experts cost 0.6 ms a token at
  2K, so the short-context bound is about +3%.
- VRAM map (`mshim`): expert cache 5,109 MiB, dense weights 1,142 (`load()`), prompt scratch 934 (`alloc()`),
  prompt expert stage 668 (`ensure_stage()`), SWA rings 254, KV 64 at 2K, BLAS 32; the desktop holds 1,689 MiB.
  Scratch and stage (1.6 GB, ~367 experts) serve prompt batches only.
- Long context: decode at 199K is 21.4 tok/s (46.7 ms a token vs 21.7 at 2K). The KV of 199K rows (3.5 GB) comes
  out of the expert cache (`cover()` -> `shrink_cache()`), so long-context decode loses most of its VRAM experts;
  the 1.6 GB above would be a larger share there (not measured yet).
- `situ_rows_q8_1`: 48 launches, 0.128 ms a token kernel time, gaps 8.3 us before / 4.4 us after.

### NVFP4 MMQ (prefill gate/up J64, lean kernel, `mmqbench` 128 experts x 128 rows x [2304 x 2304])

- Whole kernel 11.0 ms (15.9 TOPS, 34% of the 46.8 dot4 peak). Dot compiled out: 3.57 ms; weight-tile load
  compiled out: 8.71 ms. So the dot phase is about 7.4 ms (67%, 23.5 TOPS = 50% of peak), the weight-tile
  load/convert about 2.3 ms (21%), activations, barriers and write-back the rest.
- Hot block ISA (`isamix.py nvL64.s`): 470 VALU per 256 `v_dot4`: dot 54%, `v_cvt_f32_i32` 64, `v_fmac` 64,
  `v_fma` 32, `v_add_nc_u32` 54 (LDS address arithmetic: 60 LDS reads need 54 adds). The NVFP4 format's 16-value
  scale costs 2 cvt + 3 FP ops per 8 dot4 (best case 8/13 = 62% dot share); the 54 adds are the only removable
  VALU (11% of the dot phase).

### Prompt idle at 32K (trace `tr32k4`, all session-3 switches)

- The "3.1 s" counts from the first kernel of the process (expert-cache fill and load-time memsets: 151 + 889 +
  114 ms gaps). Engine PP lines: the first batch costs ~0.25 s more than a steady one.
- Steady window (after 2 s): span 35.26 s, **idle 1.50 s (4.2%)**: memset -> `copy2` (router round trip and host
  routing of 2,048 rows) 406 ms in 798 gaps of 509 us; Q5_0 down MMQ (layers 20-22) -> memset 238 + 46 ms in 31
  gaps of ~9.2 ms (host waits after the non-NVFP4 coda experts); norm -> router GEMM 215 ms (754 x 285 us);
  `gemm_small_2x2` -> rocBLAS `Cijk` 148 ms (1,536 x 96 us: rocBLAS host overhead per call); the rest is ~6 us
  inter-kernel gaps by the thousand.

### 200K end to end (R6)

- 199,000-token prompt + 32 tokens, serve mode, live config: `56eb42bd` (previous production) **580.5 s
  (342.8 tok/s)**, `f442cbe2` (deployed) **383.1 s (519.4 tok/s), -34%**; TOKHASH equal (the same 32 greedy
  tokens); decode at 199K 21.3 -> 21.4 tok/s.

## Plan (GPS, not rails; order by measured payoff and risk)

1. **D3 decode routing on the card for resident experts** (R1, bound +4.5%). Inkling `route_card()`: the router
   scores' sigmoid, bias, top-8 and weights on the card, bit-exact to `Inkling2::route()` (a port of glibc's FMA
   `expf` in double, checked exhaustively against the host over the scores' range; ties and sum order as the
   host); the cached experts go through `MmvqIds` with a per-layer slot table (`d_slots_`, updated on swaps), a
   null slot skips; the host still waits for the router, routes the same picks for CPU experts and statistics and
   enqueues the tail. First step: the exhaustive `expf` check (falsifies "bitwise" cheaply). Stop if the A/B
   gain is inside the spread.
2. **D1 prompt scratch and stage lent to the expert cache between prompts** (R1, +3% at 2K; larger at long
   context). First step: measure decode at 64K with `--prompt-batch 512` vs 2048 (the cache size proxy) to size
   the long-context gain before any code.
3. **B2 LDS address VALU in the NVFP4 dot phase** (R2, up to -7% of the kernel). Own TU (as P2); bitwise by
   construction (same products and order).
4. **B1 weight-tile load phase** (R2, 21% of the kernel): cheaper E2M1/UE4M3 conversion or loads issued ahead of
   the barrier within the VGPR budget; measure with `vnoload`-style variants first.
5. **R5 situ in the gate_up launch** (R4, est. ~1.5%): rows i and i+1152 of an expert in one block (64 rows, 1024
   threads, the same wave count), situ + Q8_1 by one warp after a barrier, bitwise to `situ_rows_q8_1`. The
   literal "into down" form is dead by construction: each of 288 down blocks reads the whole activation.
6. **C1 prompt host routing on the card** (R3, ~0.4 s of 36 s); **C2** the coda's Q5_0 waits (0.28 s);
   **C3** rocBLAS `Cijk` calls replaced by our own kernel (0.15 s).
7. **R6 KV snapshot** (R5): `SAVE`/`RESTORE` in `strata-poly` using the conversation checkpoint copies
   (`save_state`, `save_card_rows`), STRSESS-like header (magic, version, model and config fingerprints, hashes),
   size at 32K ~660 MB; let the server's `/slots` reach it.
8. Down mat-vec at 8 lanes (D2, -9% kernel, ~0.8% E2E, not bitwise): only if the plan runs out of bitwise levers.

## Current Checkpoint

- Closes: R1 (D3), first step.
- Smallest next action: exhaustive GPU-vs-host check of a ported glibc `expf` (and the full `route()` on recorded
  router scores).
- Expected evidence: 0 mismatches over the scores' float range, or the count that kills bit-exactness.
- Stop or replan if: mismatches cannot be removed; then D3 ships only as a precision-level option, or D1 goes first.

## Current State

- Resolved: R6 (200K measured). RECON numbers above.
- Production: `strata.service` restarted on the deployed `f442cbe2` config after the GPU runs; tree clean.
- Next: plan step 1.

## Material Decisions

- 2026-11-23: The soft goal "prompt idle 3.1 s" is re-measured as 1.5 s steady at 32K; the rest was process
  start-up counted from the first kernel.
- 2026-11-23: "Decode is DRAM-bound" holds for gate_up (90% marginal); the remaining short-context levers are the
  router round trip (+4.5% bound) and VRAM experts (+3%), not mat-vec bandwidth.

## Checkpoint History

- 2026-11-23: RECON: decode trace and sensitivity runs, VRAM map, slot-kernel microbench, router hack, MMQ phase
  split, 32K idle re-analysis, 200K end to end on both binaries. R6 verified.

## Completion

- Pending.
