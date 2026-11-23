# Goal: lift the reference fork's prefill/TTFT and decode vectors into Inkling-2-Mini on RDNA2

Status: active (session 2 re-measured the RECON on the GPU and rewrote the plan from kernel experiments; next: P1a)
Source: user request of 2026-11-23 (verbatim quote below): study the latest commits and goals, port the optimization
vectors of the Canahmetozguven fork (RX 7900 XTX) "для поднятия prefill/TTTF/decode, по очереди", profile at the
kernel level with temporary benchmarks during planning, fold all hypotheses into this goal, probe iteratively, then
write the action plan.
Last updated: 2026-11-23

## Objective

Inkling-2-Mini prefill (TTFT) and long-context decode on this RX 6700 XT / Ryzen 7 5700X rise by porting the fork's
measured optimization vectors, one at a time, each kept bitwise or gated by the existing quality tests, each committed
and pushed when it wins, with the engine re-profiled after every kept change. The round ends with the qualified build
deployed and the final measured account here.

## User directive (2026-11-23, verbatim)

"Изучи последние коммиты и цели, нужен план по пооптимизации движка, цель поднять перетащить оптимизации из подсказки
ниже (для поднятия prefill/TTTF/decode, по очереди)
Во время планирования ТЕБЕ нужно проводить профилирование, писать бенчмарки, что бы на уровне ядра, оценить ботлнеки,
именно на низком уровне, всё проверить, код как пример используется это только треть дела, основной профит будет что ты
во время планирование пишешь временные скрипты для оценки ботлнеков текущей реализации, и так несколько бенчмарков
делаешь, с углублением и рефлексирование, а после уже пишешь план действий где можно выжать ещё больше перфоманса

Приступай к профилированию

подсказка, изучи всё с помощью массово вызова @explore на каждый вектор, далее сложить все гипотезы в цель, итеративно
прощупать и после предложить план"

The hint is the fork's public account (github.com/Canahmetozguven/inkling-rocm, docs/02_results.md, and the Strata
fork branch port-upstream-feat-inkling2): fused relative-bias flash attention for int8 KV with P in registers (WMMA),
a split-key decode attention (KV reads 250-270 -> 527 GB/s), fp16-output projection GEMMs (their hipBLAS f32-output
path: 11.1 -> 92.5 TFLOP/s), a split-K fix for a skinny 64x4608 gate GEMM at 1.7 TFLOP/s, a Windows BAR-eviction guard,
and SAVE/LOAD of the KV state to disk.

## User directive, session 3 (2026-11-23, verbatim)

"Переходи от RECON к реализаци цели, план работ это GPS навигатор, а не рельсы

Используй текущий план как карту гипотез, а не жёсткий маршрут, начинай с P1

Работай автономно длинными циклами: профилирование -> гипотеза -> временный прототип/бенчмарк -> измерение -> анализ
-> следующий эксперимент (если профит детерминирование получен - коммит и пуш в main, далее это будет как baseline для
последующих правок (т.е закрепляем результат и идём дальше)

При неудачах выясняй почему ожидания не подтвердились, пересматривай гипотезы и пробуй другие подходы, не Зацикливайся
на одном направлении и свободно меняй приориты по результам экспериментов

Успешные оптимизации интегрируй в main (прод бинарник), проверяй end to end прирост

После существенных изменений, перепрофилируй движок, bottleneck может переместиться

ПРодолжай исслендвоание и реализацию, пока останется перспективное проверяемые гипотезы, в финале покажи измеренные
результатыы, неудачные подходы и оставшиеся ограничения, и деплой в прод

---

Делай копию плана и моей подсказки в цель, помни что документ с целью это глина, её можно мять как угодно по ходу
работ, в эту глину пиши удачи/неудачи

Используй саб агента general для ревью"

How this session works: the Plan below is a hypothesis map; each kept, deterministic win is committed and pushed to
main and becomes the baseline for the next experiment; failures are recorded with why the expectation broke; the
engine is re-profiled after substantial changes; code review of each engine diff by a general-purpose subagent.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest
unresolved outcome; the plan below is a hypothesis map (GPS, not rails): reorder, drop and add experiments by what the
measurements say; commit and push every deterministic win to main and re-profile after substantial changes. Finish when
every required outcome is resolved or no promising testable hypothesis is left; report measured results, failed
approaches and remaining limits, and deploy to production.

## Frozen Contract

### Required Outcomes

- R1: The kernel-level RECON of this round is measured and recorded here: the card ceilings, the prefill attention and
  decode attention span curves (bench1), and the per-kernel prefill cost table from the retained traces.
  - Source: "проводить профилирование, писать бенчмарки ... именно на низком уровне, всё проверить".
  - Acceptance: the numbers below in "Profiling" carry this round's own measurements (trace archaeology done; the
    bench1 sweeps run once the reboot rule below is lifted), each with the hardware and config it was measured on.
  - Primary evidence: the 8K/32K trace tables, bench1 logs under the guard.
  - Status: verified (session 2)
  - Evidence: "Session 2 RECON" below; scratch `~/.cache/claude-fv/` (fvbench, eng.py, ktab.py, runs/, tr32k/). The
    first session's scratch `prefill-opt3/` does not exist.
- R2: The action plan is written from those measurements, ordered by measured gain per risk, with every fork vector
  either mapped to a concrete experiment or rejected with a measured reason.
  - Source: "сложить все гипотезы в цель, итеративно прощупать и после предложить план".
  - Acceptance: the "Plan" section below; no vector of the hint left unmapped.
  - Primary evidence: this document.
  - Status: verified
  - Evidence: the plan below, including the three fork vectors already rejected by measurement (the f32-output GEMM
    pathology, the skinny-gate SGEMM, and "add split-key" as such - our kernels already have the design).
- R3: Prefill is faster end to end, prefill-focused vectors first.
  - Source: the hint's prefill items and "по очереди".
  - Acceptance: paired A/B (A = `56eb42bd` production family, forced fixtures at 2K/8K/16K/32K, uncached), B faster
    beyond variation at 16K and 32K, with 2K/8K not slower; TTFT on the private HTTP fixture improved at 8K+.
  - Primary evidence: `rb.sh`/bench.py A/B logs (ABBA), `--pp` prefill runs, HTTP TTFT pairs.
  - Status: pending
- R4: Long-context decode is faster (the fork's split-key follow-ons), short-context decode not slower.
  - Source: the hint's decode items.
  - Acceptance: `poly_inkling_attend_test` span times improve at 64K-190K beyond the pre-launch_bounds baseline;
    paired 94+300 and 16K+200 decode A/B not slower.
  - Primary evidence: attend test logs, `rb.sh` A/B.
  - Status: pending
- R5: The decode round's remaining mapped item (the situ+down merge) is resolved: shipped or measured dead.
  - Source: the previous round's map ("est +1.2%").
  - Acceptance: a bitwise fused kernel plus an A/B win, or a recorded measured rejection.
  - Primary evidence: `poly_*_test` bitwise result, A/B logs.
  - Status: pending
- R6: The KV state can be saved to disk and loaded by a fresh engine process, so a restart does not re-prefill a long
  context (the fork's "a prefill should not happen twice").
  - Source: the hint's SAVE/LOAD item.
  - Acceptance: a versioned, validated snapshot file (conv state + SWA rings + full-attention rows + token list +
    position); a fresh process after LOAD reproduces the same greedy tokens as the process that saved, and a request
    prefills only the new suffix; incompatible formats are refused and re-prefilled.
  - Primary evidence: a snapshot round-trip test with byte compare and a same-tokens check (the fork's test pattern:
    416 MB in 172 ms out / 123 ms in at 8K on their machine).
  - Status: pending
- R7: Quality unchanged; precision-level changes (if any ship) are opt-in and measured.
  - Source: "не потерять в качестве" (the standing rule of both prior rounds) plus the hint's own accuracy tables.
  - Acceptance: every kept kernel bitwise or within the existing attend-test tolerance; the answer-quality battery
    (14-question greedy set) 14/14 twice on the deployed config; any F16/packed-math variant additionally shows a
    logits KL vs the current engine inside the engine's own CPU/GPU-expert noise floor and stays opt-in with a
    byte-identical off path.
  - Primary evidence: unit tests, the battery, a KL harness run, forced-ID parity fixtures.
  - Status: pending
- R8: The qualified build serves production with rollback, and the final account is here.
  - Source: the standing deploy step of both prior rounds.
  - Acceptance: deployed binary + switches verified from `/proc`, live check passes, rollback files in place; the
    final account lists kept wins with before/after, rejected ideas and limits.
  - Primary evidence: deployment record below, live smoke.
  - Status: pending

### Constraints

- C1: RDNA2/HIP only; one card, one sequence; the same GGUF (`Inkling-2-Mini-UD-IQ4_XS.gguf`), Q8_0 KV, 200,000
  context; no expert skipping; the VRAM reserve stays 2,048 MiB (the user's decision, first decode round).
- C2: New behavior opt-in; the switch-off path byte-identical; other models unchanged.
- C3: GPU runs only through `rdna2-rnd/scripts/run_guarded.py`, its exit code and `stop-reasons.json` read first; one
  engine at a time; the production unit is stopped during standalone GPU work and restored afterwards.
- C4: The black-screen rules of 2026-11-23 stand: no kernel that can wait without a limit (no software grid barrier,
  no persistent/spinning kernel), no `rocprofv3 --pmc`; buffers sized for the largest call, checked before the first
  run; after any GPU fault, GPU work stops for the day and the next run needs a reboot. For the 17:15:23 fault of
  2026-11-23 the user lifted the reboot part (session 2, Material Decisions); every other part stands, and a new
  fault stops GPU work again.
- C5: A fusion keeps its kernels' arithmetic and reduction order; a kept fusion is bitwise or inside the existing
  attention-test tolerance. Frozen production kernels take no new stores (the P2b lesson).
- C6: Every engine-code commit goes through an independent review subagent on the diff before the commit (the user's
  rule from the decode round).
- C7: Docs keep measured numbers with what they were measured on; artifacts, logs, profiles and binaries stay out of
  Git; the server stays on 127.0.0.1 with its key.

### Non-goals

- DSpark/speculative decoding (measured slower here at 35.5 vs 36.5 tok/s; the fork measured the same: 2.1-2.3
  drafts a verify against the 3+ it needs).
- Expert-cache policy changes (the fork's replay at 2-10% residency: no online policy beats LRU by much; ours at 43%
  residency already matches an LRU simulation).
- 1M context, int8-with-Hadamard KV, other GGUFs, other backends, Windows BAR work (Linux does not evict engine
  buffers the way Windows did; the fork's guard concept is recorded, not ported).
- A smaller reserve; chunk-size changes beyond what a measured prefill win requires.

## Change Envelope

- Prefill/decode attention: `src/poly/inkling_attend.cu` (+`_hpp`), a `*_test.cu` per change; opt-in env switches.
- Dense/expert product path: `src/poly/mmq.cu`, `src/kernels/cuda/native_mmvq.cu`, `src/kernels/cuda/iq_kernels.cu`,
  `src/poly/kernels.cu` + `include/strata/poly/kernels.hpp`, opt-in epilogue/precision variants with quality gates.
- Snapshot feature: `src/poly/engine.{hpp,cpp}`, `src/poly/inkling2.cu` (save_state/save_rows wiring), `src/poly/main.cpp`
  (a poly SAVE/RESTORE-style command or serve hook, Inkling-shaped, refusing incompatible formats), plus a round-trip
  test.
- CMake/test entries for retained changes; docs (this file, a results section at the end).
- Forbidden: request-protocol changes; default-path changes; new services or dependencies; formula changes that are
  not bitwise or not quality-gated per R7.

## Profiling (this round's RECON)

Hardware: RX 6700 XT (gfx1031, 12 GB, 96 MB Infinity Cache, the desktop's card), Ryzen 7 5700X, 32 GB DDR4, kernel
7.2.15-300.fc45, TheRock HIP 7.13 (the repo venv), build `56eb42bd` (deployed family), forced fixtures, reserve
2,048 MiB, batch 2048, 8 threads, 64 swaps, frozen-profile copies.

### The fork's reference numbers (RX 7900 XTX, gfx1100, from its public docs; every number is theirs, not ours)

- Their engine's prefill went 452 -> 1,677 -> ~3,090 tok/s at 32K and 52 min -> 37 min at 1M; decode 73 -> 81 tok/s
  at 16K, ~7 -> ~10.5 tok/s at 1M.
- Attention kernel lineage: table-based `kb_attn_tc` 15 TFLOP/s -> fused in-kernel bias 27 TFLOP/s -> transposed
  formulation (S^T = K Q^T, O^T = V^T P^T; softmax state one scalar a lane; P into the next product through a lane^16
  register exchange instead of an LDS round trip; LDS 56 -> 18.9 KB; VGPR 256/18 spilled -> 219-241/0) 53 TFLOP/s at
  64K; whole 64K prefill -10.7%; 256K prefill 272 -> 158 -> 119 s.
- Split-key decode attention: one block per KV head and its 3 query heads, 16-byte loads, up to 8 key rows in flight
  per lane; KV stream 250-270 -> 527 GB/s (57% of their 925 streaming); decode 1.39x at 256K, 1.62x at 512K.
- Projection GEMMs: hipBLAS GemmEx f16-in/f32-out runs 11.1 TFLOP/s while f16-out runs 92.5 on the same shape -> fp16
  output (+43% on their 16K prefill). Their KL vs a CPU reference stayed 0.00202 -> 0.00189 (their noise floor).
- Skinny gate: a 64-output, K=4608 SGEMM at 1.7 TFLOP/s; 18-way strided-batched split-K -> 344 ms from 1,115 ms of
  their 32K profile, +4.5% whole prefill.
- Accuracy method: fp32-rounding-size changes injected into their attention output move logits KL by ~1e-3 - the size
  of their whole engine-vs-reference KL; answer-level tests are the real gate.
- Snapshots: 416 MB written 172 ms, read 123 ms at 8K; a fresh process reproduces the same 24 greedy tokens and
  prefills only the 42 new tokens.
- Their negatives: attention register prefetch spilled 87 VGPRs and lost; speculative decoding lost; cache policy
  unchanged; a per-process 2x slow decode state at 256K (one cause: Windows moved the unembedding matrix out of a
  full card, +31 ms/token; a second state unexplained).

### Session 2 RECON (2026-11-23 evening): every number below re-measured today

The first session's commits are treated as evidence to re-check, not as facts. Hardware: RX 6700 XT (gfx1031, 20 WGP /
40 CU, 12 GB), Ryzen 7 5700X, 32 GB DDR4, kernel 7.2.15-300.fc45, TheRock HIP 7.13 (repo venv). GPU runs under
`run_guarded.py --accept-faults-before '2026-11-23 17:16:00'` (see Material Decisions); every run below exited 0 with
empty stop-reasons. Scratch outside Git: `~/.cache/claude-fv/` (fvbench.hip, variants v0..v3d and vd, eng.py,
ktab.py, idle.py, runs/, tr32k/).

**What the first session left that did not hold:**
- The scratch it names (`~/.cache/opencode-tmp/opencode/prefill-opt3/`, `bench1.hip`, `post-reboot.sh`) does not
  exist; only compiler temporaries of `bench1` remain. The benches here were written anew.
- Its decode driver (`decode-opt2/bench.py`) reads `claude-inkling-pp/prod-before.json`, an old config, not the
  live `~/.config/strata/inkling2-mini/strata-inkling-2-mini.json`. `eng.py` reads the live args and env and runs
  long prompts through `--serve` (a 32K id list is 160 KB, over Linux's 128 KB single-argument limit).
- "Cold" decode-attention numbers (its 2.023 ms at 190K, the attend test's cold lines, the previous round's
  2.453 -> 2.023 ms) flush with a 256 MiB **write** (`cudaMemsetAsync`). Dirty lines then cost write-backs during the
  measured kernel. With a 256 MiB **read** flush the same kernel takes 1.41 ms at 190K (write flush: 1.96 ms).
  Engine decode is read-dominated, so the read-flush number is the representative one.
- Confirmed: the 8K/32K prefill trace table (fresh trace below: flash 12,906 ms vs its 12,948), the 413.44 MB/visit
  KV payload at 190K, `kAttUnroll=2`, the flash geometry (16 tokens x 16 keys, 128 threads, char4 + 2-byte scale
  fetches), and `21544cc0` changing only `__launch_bounds__(256)` -> `(256, 2)`.

**Card ceilings (`fvbench ceil`):** DRAM read 380 GB/s (1 GiB), write 374 GB/s; Infinity Cache 1,000 GB/s for
64-96 MiB; FP32 FMA 11.69 TFLOP/s (2.28 GHz sustained); `v_pk_fma_f16` 23.8 and `v_dot2_f32_f16` 23.7 TFLOP/s;
`v_dot4_i32_i8` 46.8 TOPS (the first session's 39.2 was assumed, not measured); LDS `ds_read_b128` in flash's
pattern 6.04 TB/s.

**E2E, production binary `strata-poly-56eb42bd`, live config, 32K prompt, `rocprofv3 --kernel-trace`:**
untraced 43.28 s prefill (757 tok/s) and 33.1 tok/s decode at 32K; traced 42.75 s.

| Prefill line (32K) | ms | share of 40,772 ms kernel time |
|---|---:|---:|
| `flash_kernel<true,true>` (768) | 12,906 | 31.7% |
| NVFP4 gate/up `mul_mat_q<40,64>` (2,617) + other J | 10,244 + ~640 | 26.7% |
| `nvfp4_tail_kernel` (1,479) | 4,700 | 11.5% |
| Q8_0 dense `mul_mat_q<8,64>` (4,240) | 4,374 | 10.7% |
| mix 846, 2x2 GEMM 827, quantize 721, rms_norm 673, Q5_K 582, Q6_K 517, situ_rows 494, router 421, moe_sum 384, head_norm 371, control 278, others | ~7,300 | ~18% |
| GPU idle inside the 43.9 s span (union of busy intervals) | 3,178 | 7.2% of span |

Idle: ~1.23 s is the per-visit router round trip through the host (`add_rms_norm -> router` 397 us,
`router -> fill` 662 us, `fill -> quantize` 580 us, x768 visits); ~0.9 s is 147 gaps of ~6 ms between memsets (not yet
attributed; the memory-copy trace in `tr32k/` can tell whether expert uploads are behind them).

Decode at 32K (63 traced tokens): 954 launches/token, GPU busy **16.8 ms** of ~30 ms/token. Kernels:
`mmvq_slot_lanes<40,16>` 4.66, `native_mmvq_segments<8>` 2.88, `decode_q8_kernel` 2.69, o-proj 1.12, LM head 1.02 ms.
Idle per token: 5.96 ms waiting after the GPU experts for the CPU experts' sum (`mmvq_slot_lanes -> conv_add`,
137 us x 43.6), 2.3 ms on visits with no GPU expert, 1.43 ms router round trips. So decode at 32K is about half
GPU kernels, half CPU experts (DDR4) and syncs; 1,201 of 3,072 experts are in VRAM.

**Prefill flash attention (`fvbench flash`, n=2048, Q8_0 KV, ring 3,071 as the engine's `ring_`):** 5.0-5.5 TFLOP/s
= 43-47% of FP32 at every position (0 to 196K), so it is not memory-bound. ISA of the tile loop (one 16-key tile,
one wave): 2,547 VALU, 1,536 of them the useful FMAs. Each DPP8 reduction step compiled to three instructions
(copy, `v_mov_b32_dpp`, `v_add_f32`) where one `v_add_f32_dpp` does it; 201 such steps a tile. LDS is not the limit
(needs 3.5 of 6.04 TB/s at today's rate).

| Flash experiment (all bitwise vs production on 10 dumped configs: full/SWA flash, decode) | full 32K | full 196K | SWA |
|---|---:|---:|---:|
| v0 production | 160.1 ms | 983.8 ms | 5.08 ms |
| v1: fused DPP via `asm volatile` | 162.1 (+1.3%) | - | 5.21 |
| v1b: same, plain `asm` (schedulable), 203 VGPR | 145.7 (-9.0%) | - | 4.70 |
| v2: + in-tile branches (far-tile bias skip, inner-tile mask skip, rescale skip) | 161.8 (+1%), and NOT bitwise | - | 6.33 |
| v3 / v3b: tile body as template, chosen by a branch (3 or 2 copies) | spills 212 / 200 B | - | - |
| **v3c: v1b + far tiles in their own loop (no table slice, no bias), 221 VGPR, no spill** | **129.7 (-19.0%)** | **757.8 (-23.0%)** | **4.69 (-7.7%)** |
| v3d: v3c forced to 5 waves/SIMD | 191.7 (spills 120 B) | - | 6.39 |

Lessons: the tile loop's one big basic block is what hides LDS latency, so any branch inside it costs more than it
saves; uniform specializations must be separate loops. v2 also broke bitwise identity because the compiler had
contracted `l*shrink + y` into one FMA, which a branch splits.

**Decode attention (`fvbench decode-long`, read flush; variants in `vd/`):**

| 190K span (413 MB) | 128 chunks (production) | 80 | 160 | 40 |
|---|---:|---:|---:|---:|
| unroll 2 (production) | 1.407 ms (294 GB/s) | 1.243 (333) | 1.239 (334) | 1.252 (330) |
| unroll 4 + batched softmax | 1.335 | 1.269 | 1.310 | 1.262 |

At 65K: 0.506 ms (128 chunks) vs 0.454-0.465 (80/160). 128 chunks of 8-wave blocks are 3.2 rounds of the 40
resident blocks (2 a WGP at 173 VGPR); a multiple of 40 removes the partial round: -12% at 190K, 87% of DRAM read.
Unroll 4, batched online softmax (one rescale per 2-4 keys) and the write/read flush aside, nothing else moved the
long-span time: the per-lane-latency hypothesis is refuted. Decode attention at 16K is 0.14-0.15 ms a visit; its
lever is long context only (8 visits x ~0.17 ms saved at 190K = ~1.3 ms/token).

**NVFP4 MMQ (`mul_mat_q<40,64>`, 240 VGPR, 4 waves/SIMD):** 10.9 TOPS = 23% of the measured dot4 peak. Inner dot loop:
534 VALU, 256 `v_dot4c_i32_i8` (48%); 64 `v_mov` that zero the tied VOP2 accumulators, 64 `cvt` + 64 `fmac` (the
per-16 UE4M3 scales, inherent), 54 LDS address adds. With tile loading dot4 is ~30% of VALU: like flash, the line is
bound by instruction overhead, not bandwidth.

### Fork-vector map after session 2

| Fork vector | Here | Verdict |
|---|---|---|
| Fused relative-bias flash, P in registers | already ours (in-kernel bias from a 64 KB table; OWNED16 lane-owned P) | the fork's kernel work transfers as **instruction economy**: v3c -19..-23%, bitwise |
| Transposed formulation / WMMA | no WMMA on gfx1031 | the transferable part is "fewer instructions per FMA"; `v_dot2_f32_f16` (2x FP32) is the packed-math route, precision-gated |
| Split-key decode, 527 GB/s | ours, 294 GB/s at 190K | chunk count a multiple of the resident blocks: 334 GB/s (-12%) |
| fp16-output GEMMs | our MMQ is not the broken hipBLAS path | rejected as such; MMQ's own instruction overhead is the lever |
| Split-K skinny gate | 2x2 GEMM 827 ms + SGEMM 221 ms = 2.6% | rejected (no pathology) |
| KV SAVE/LOAD | missing (`SERR ... keeps no disk sessions`, HTTP 501) | P6, unchanged |
| Windows BAR guard | n/a on Linux | rejected |

## Plan (session 2; ordered by measured gain per risk; prefill/TTFT first, then decode)

Every step: a `STRATA_INKLING_*` opt-in switch, off path byte-identical; `fvbench dump` bitwise vs production (or
inside the attend test's 1e-4 for the non-bitwise ones); `poly_inkling_attend_test` both `LANE_PROB` settings; an
ABBA end-to-end A/B with `eng.py` (live config) at 8K and 32K, plus 128K for attention; independent review subagent
on the diff before each commit (C6).

1. **P1a: flash v3c into the engine** (`STRATA_INKLING_FLASH_FAST=1`, gfx1031 lane-prob path). Measured: -19% at
   32K, -23% at 196K, -7.7% SWA, bitwise. Expected E2E: ~-2.0 s at 32K (-4.6%), ~-85 s on a 200K prompt (~-15%).
   Then re-profile.
2. **P1b: flash, more of the same economy (bitwise):** the 48 owner `cndmask` per tile through a reduce-scatter
   ordering of the partial dots; the rescale skip as a third loop version only if it fits without spills; fold the
   SWA ring modulo out of the fetch (64-bit `%`, ~460 instructions on SWA tiles per fetch block). Expected +5-10% on
   the line; each kept only if measured.
3. **P1c: packed-math flash (precision; opt-in).** Minimal experiment first: QK with Q and K codes in F16 through
   `v_dot2_f32_f16` (FP32 sums; K codes are exact in F16, the scale applied per 32), PV unchanged; measure the
   kernel rate and the attend-test error. Only if the kernel gains >1.3x and the error is inside the fork's
   noise-floor protocol (logits KL vs the current engine within its own CPU/GPU-expert noise, answer battery 14/14)
   does it continue.
4. **P2: NVFP4 MMQ instruction economy (bitwise, integer):** the VOP3 `v_dot4_i32_i8` with a zero inline constant
   instead of `mov 0` + `v_dot4c`, and immediate LDS offsets; measure on a captured-shape microbench first
   (expected -10..-15% of the 37% line: -1.5..-2.3 s at 32K). The 4-byte tied-accumulator pattern is in
   `third_party/llama.cpp/ggml/src/ggml-cuda/mmq-load-tiles.cuh` / the NVFP4 vec_dot; same check for
   `nvfp4_tail_kernel`.
5. **P3: prefill router round trip (TTFT, 2.8%):** the ~1.2 s of host time between the router GEMM and the expert
   products. Measure first where the 662 + 580 us go (host top-k of 2048x128, grouping, uploads); a GPU top-k +
   grouping kernel only if the host part is the bulk. The 147 x 6 ms memset gaps get attributed from the copy trace.
6. **P4: decode attention chunk count** = a multiple of the resident blocks (160 at 190K): -12% on the long-span
   kernel, ~-1.3 ms/token at 190K, nothing at short context. Not bitwise (chunk sums), inside the attend-test
   tolerance.
7. **P5: decode CPU-expert wait (the largest decode term at 32K, ~6-8 ms/token).** RECON first: per-token count of
   CPU expert-visits and their bytes vs DDR4 read speed, to see whether the wait is bandwidth (then only VRAM
   residency helps: the fixed 2,048 MiB reserve stays, C1) or scheduling (CPU starts late after the router
   readback).
8. **P6: KV snapshot SAVE/LOAD** for `strata-poly` (unchanged from the first plan, mirrors the Qwen engine's
   `STRSESS` v1 format and the `SAVE/RESTORE` protocol, `src/core/conversation_file.cpp`).
9. **P7: deploy, battery, final account.**

Rejected on measurement this session: in-tile branches in flash; 5 waves/SIMD flash (spills); decode-attention
unroll 4 and batched softmax; fp16-output GEMMs and split-K gate as fork pathologies with no counterpart.

## Session 3 log (implementation; newest last; each kept step is the next baseline)

- **P1a kept: `STRATA_INKLING_FLASH_LEAN=1`** (`src/poly/inkling_attend_lean.cu`, used with `LANE_PROB=1` on gfx1031).
  v3c as its own kernel in its own translation unit: added next to `flash_kernel`, it changed `flash_kernel`'s ISA
  (register allocation) although that source was untouched; in a separate file all 13 functions of
  `inkling_attend.cu` keep the HEAD ISA (normalized `-S` compare). Review (general subagent) found that the far loop
  turns `fmaf(inf, 0, s)` = NaN into `s` for an infinite `rel` term; fixed by a block vote (`__syncthreads_and`) that
  sends such a block through the general loop. Checks: `fvbench dump` 10/10 configs bitwise vs production; attend test
  PASS for LANE_PROB 0/1 and LEAN 1; kernel full 32K 160.1 -> 131.7 ms, 196K 983.8 -> 752.0 ms, SWA 5.08 -> 4.78 ms.
  E2E (`ab.sh`, live config, serve mode, same binary, only the switch; two ABBA rounds): 32K prefill
  A 43.23/42.39/42.27 s (A1 of round 1, 47.71 s, an outlier with another expert residency), B 40.64/41.27/40.44/40.58
  s: **-4.9%**; 8K 9.13/9.73 -> 8.99/9.00 s. Generated tokens equal within a residency (both token hashes occur with
  either switch: 1,206-1,228 experts loaded per run). Accepted limit: like `LANE_PROB`, the switch is decided once per
  process for the first device (one card, C1).

- **P2 kept: `STRATA_HIP_NVFP4_LEAN=1`** (`src/poly/mmq_nvfp4_lean.cu`, `mmq_nvfp4_tail_lean.cu`). The NVFP4 vec_dot
  starts each 16-value group's sum at 0; hipcc then moves 0 into a register for the tied `v_dot4c_i32_i8` (222 such
  `v_mov` in the J64 kernel). The lean units route ggml's `ggml_cuda_dp4a` through a wrapper (renamed around
  `common.cuh`) that emits VOP3 `v_dot4_i32_i8 r, a, b, 0` when the sum is the constant 0: `v_mov` 222 -> 94, VGPR
  240 -> 218, same integers. third_party/llama.cpp is a gitignored FetchContent checkout, so the change lives in
  Strata's own units; the default units' ISA is unchanged (65/65 and the tail kernel). Microbench (`mmq/mmqbench`,
  128 experts, Mmq API, bitwise vs default): gate/up 2304x2304 at 128 rows 11.67 -> 10.90 ms, 64 rows 6.02 -> 5.58,
  48: 4.95 -> 4.79, 32: 4.00 -> 3.91, 24: 3.53 -> 3.40, 16: 3.05 -> 3.01; **8 rows 2.73 -> 2.89 (worse: the review
  found J8 grows to 175 VGPRs)**, so lean is used only above 8 rows; down tail 2304x1152 6.62 -> 6.23 ms. E2E 32K
  (both with FLASH_LEAN): A 41.63/40.85/40.95/41.17 s, B 40.53/40.06/40.07/41.47 s (-1.5%, near the noise; one B run
  had a slow decode too). Review (general subagent): no linkage or semantic defect; the J8 finding is applied.
  Not taken: the same wrapper for the Q8_0 dense MMQ (one zero start per 8 DP4A there, half NVFP4's share; est.
  <0.5% E2E).

## Current Checkpoint

- Closes: R1 (re-measured: ceilings, flash and decode span curves, fresh 32K trace, ISA mixes), R2 (plan above).
- Smallest next action: P1a - port v3c's two changes (`~/.cache/claude-fv/v3c/inkling_attend.cu`: fused DPP8 asm
  helpers and the far-tile loop) into `src/poly/inkling_attend.cu` behind an opt-in switch, run
  `poly_inkling_attend_test` and `fvbench dump`, then the E2E A/B.
- Stop or replan if: v3c is not bitwise inside the engine build, or the E2E A/B does not show the kernel gain.

## Current State

- Resolved: the first session's claims checked (list above); a fresh profile of the deployed binary; four kernel
  experiments on flash (one kept, bitwise, -19..-23%), three on decode attention (one kept, -12% at long span).
- Production: `strata.service` restarted at the end of the session on the deployed config; `/proc` shows
  `STRATA_INKLING_{FLASH,NORM_Q8,TAIL_MOE,ROUTE_DUAL}=1`.
- No engine code changed in this session.

## Material Decisions

- 2026-11-23: The three fork vectors below are rejected on our own measurements, not by assumption: the f32-output
  GEMM pathology, the skinny-gate split-K, and "add split-key decode" (already our design).
- 2026-11-23: Precision-level changes (P1c and any later packed-math path) only opt-in and only through the R7 gates.
- 2026-11-23 (session 2): the user lifted the reboot rule of C4 for this boot ("гпу использовать для бенчей и тестов,
  не смотря на глупую ошибку, исправь её и используй gpu"). `run_guarded.py` gained `--accept-faults-before`: it
  ignores only this boot's faults logged before the given time; any later fault, and any fresh fault during a run,
  still stops the run. Default behavior unchanged.
- 2026-11-23 (session 2): the first session's commits are evidence to be re-checked (the user's instruction); the
  checks are listed in the session 2 RECON.
- 2026-11-23: For the deploy step, the live switches are verified from `/proc` (see the first session's env-drift
  note; at the end of session 2 they are on).

## Checkpoint History

- 2026-11-23: RECON opened; plan P1-P7 frozen from trace archaeology; bench1 faulted at 17:15:23 (gate buffer sized
  `[n][24]` against the kernel's `[n][24][128]`), boot refused per C4.
- 2026-11-23 (session 2): guard flag; fvbench (ceilings, flash/decode sweeps, bitwise dumps) and eng.py (live config,
  serve mode) written; fresh 32K trace; flash v1..v3d and decode vd experiments; plan rewritten from them.

## Completion

- Pending; see R1-R8.
