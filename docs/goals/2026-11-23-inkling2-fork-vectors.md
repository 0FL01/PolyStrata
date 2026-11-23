# Goal: lift the reference fork's prefill/TTFT and decode vectors into Inkling-2-Mini on RDNA2

Status: active (RECON measured where the boot allows; the GPU-fault rule below stops the remaining kernel runs until
a reboot; the plan is frozen from the measurements already in hand)
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
  - Status: in_progress (trace tables and fork comparison measured; bench1 GPU runs blocked on the reboot, see the
    incident)
  - Evidence: the "Profiling" section; scratch `prefill-opt3/` (bench1.hip built, run-check/run-dbg logs, the fault
    record in `run-check2/`).
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
  run; after any GPU fault, GPU work stops for the day and the next run needs a reboot - this rule is ACTIVE now (see
  the incident below).
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

### Our engine's measured state (trace archaeology, retained build; the 32K/8K tables from
`inkling-prefill-e2e-20261122/attention-retained-trace`, decode from the second round's `rp4`)

32K prefill, 43.5 s wall, 40.85 s kernel-sum, 55,530 launches (16 chunks x 48 visits):

| Line | ms | share | measured rate |
|---|---:|---:|---|
| flash_kernel (48 visits, 768 launches) | 12,948 | 29.7% | ~5.95 TFLOP/s (77 TFLOP incl. 40 SWA visits) = ~51% of the 11.6 TFLOP/s F32-FMA peak |
| NVFP4 gate/up MMQ (4,641) | 10,956 | 25.2% | 12.2 nominal TOPS = 31% of the 39.2 TOPS dot4 roof |
| nvfp4 down tail (1,619) | 4,629 | 10.6% | 14.4 nominal TOPS = 37% of roof |
| Q8_0 dense MMQ (4,240) | 4,352 | 10.0% | 19.4 TOPS = 49% of roof; stores 89.2 MB/chunk-visit exceed the 7.5 MB weight payload per Q/gate launch |
| copy H2D (expert staging) | 8,899 | - | overlapped (the shipped prefetch); copy-only idle is small |
| mix 867, gemm_small_2x2 838, quantize 745, rms_norm 684, Q6_K dense 600+519, situ_rows (F32, unfused at prefill) 475, router gemm_tile 420, moe_sum 396, head_norm 373, Q5_K 364, control 281, coda experts (Q4_K/IQ4) 537, add_rms_norm 219, conv_span 172, axpy 153 | ~5,000 | 11.5% | the launch-bound tail |
| rocBLAS SGEMM (controller gain/forget, 3,072 launches, 2048x2304x64) | 226 | 0.55% | 8.2 TFLOP/s, at the store roofline - healthy |

8K (warm segment): MMQ-40 2,741, flash 1,387, tail 1,154, MMQ-8 1,097, then the same tail at 1/4 scale; span 8,780 ms.

Decode (from `rp4`, 64 tokens, all switches on): 960.7 launches/token, GPU busy 15.09 ms - MmvqSlots 5.448, dense Q8_0
segments 2.846, o-proj 1.113, LM head 1.016, decode_q8 0.581 + combine 0.171, mix 0.502, conv_add 0.382, gemv_rows
0.312, add_rms_norm 0.294, segments/dual/out3 ~0.29 each, mmvq<14> 0.276, rms_norm rows 0.218, mix3 0.215, heads_kv
0.193, situ_rows_q8_1 0.118 (48).

### Fork-vector comparison (what transfers, measured)

- **Fused relative-bias flash: already ours.** Our flash computes the bias in-kernel from the projected `rel_` rows
  and a 64 KB table (no dense bias table); OWNED16 already lane-owns the probabilities (P in registers). The fork's
  table->fused win (their 11.7) does not apply. What remains is raw kernel rate: we are at ~51% of the F32-FMA peak
  (their kernels: 22-45% of their WMMA peak) with known overheads inside our tile loop: the serialized per-tile
  stage (16 KB FP32 unpack into LDS behind two barriers), scalar 4-byte Q8 fetches (char4 + a 2-byte scale), the
  31-distance ts restage per tile, and FP32-only math. This is P1.
- **Split-key decode: already ours.** `decode_q8_kernel` splits the span into up to 128 chunk blocks (one block per
  chunk covering all 8 KV heads, 8 waves), 16-byte `global_load_dwordx4` code loads, online softmax with a combine
  kernel. Cold at 190K: 2.023 ms for 413.44 MB = 204 GB/s = 54% of our 376 GB/s streaming; the fork's post-fix is
  527 of 925 = 57% - the same fraction of the respective card. The remaining levers are per-lane memory-level
  parallelism (our `kAttUnroll=2`, the fork keeps up to 8 key rows in flight), the chunk cap, and scale-load pairing.
  This is P4. Note: `21544cc0`'s `__launch_bounds__(256,2)` has 173-176 VGPRs in every archived trace (<=128 is
  needed for two 256-thread blocks a CU) and the kernel symbol is byte-identical between `ebab4fad` and `21544cc0`
  binaries; the measured -17.5% cold win stands as recorded, but its residency mechanism is not established. The
  post-reboot bench re-measures the span curve.
- **fp16-output projection GEMMs: the pathology is not ours, the idea survives as an opt-in.** Our dense projections
  run custom Q8_0 MMQ at 49% of the dot4 roof (not a broken 9%-of-peak library path), and our gain/forget SGEMM runs
  at 8.2 TFLOP/s (0.55% of the profile). The fork's +43% was fixing their library; for us the analogous change is
  F16 (or Q8_1) output from our MMQ epilogue: every dense projection stores more F32 bytes than its weight payload
  (Q/gate: 25.2 MB out vs 7.5 MB weights per launch at M=2048). Expected 5-15% of the 10% line; a precision change,
  so opt-in with the R7 gates. This is P3.
- **Skinny-gate split-K: dead here (measured).** Our controller down (4608->64) already runs the pipelined 2x2 F32
  kernel (0.46 ms/launch), and the gain/forget SGEMM is at the store roofline (above). The fork's 1,115 ms pathology
  has no counterpart: our router+controller total is 646 ms = 1.5% of a 32K prefill.
- **KV SAVE/LOAD: a real gap.** `strata-poly` rejects the Qwen engine's SAVE/RESTORE (HTTP 501); a restart re-prefills
  everything. A 200K prefix is 3.57 GB (92.88 MB conv+SWA checkpoint + 3.48 GB full-attention rows; entry points
  `save_state`/`save_rows` exist). This machine's LUKS NVMe does 2.8 GB/s write / 1.5 GB/s read (O_DIRECT, measured
  today): a 200K snapshot is ~1.3 s to write, ~2.4 s to read, against a multi-minute re-prefill. This is P6.
- **BAR-eviction guard: N/A on Linux** (recorded, not ported).

### The 2026-11-23 GPU fault (this round's incident; the rules held)

19:15:23, this boot: `[gfxhub] page fault ... Process bench1 pid 228167 ... PERMISSION_FAULTS: 0x3` - my first
`bench1 check` wired the attention `gate` buffer as `[n][24]` while `gate_kernel` reads `[n][24][128]`; the OOB read
faulted, the guard caught it, ROCm killed the process, and the desktop survived (no ring timeout, no reset followed).
Per C4 the boot is now refused by the guard ("the GPU took a fault or a reset earlier in this boot; reboot first"),
so the remaining bench1 GPU runs wait for a reboot. The buffer is fixed (gate sized `[n][24][128]`, scales 64 B/row
- the same class of sizing bug as the round's C5 rule was written for) and `bench1` is rebuilt; `bench1 dbg` (the
allocation probe) passed clean before the fault and the wiring-check mode is ready to run first after the reboot.

## Plan (hypothesis map, ordered by measured gain per risk; prefill first, then decode, then the feature)

1. **P1 prefill flash attention rate (the largest scalable lever; 29.7% at 32K and growing with N^2 - at 200K it is
   the dominant prefill term).** Experiments, cheapest first, each behind the guard after the reboot:
   - a. bench1 spans: flash n=512/2048 at pos 8K-196K, hot and L2-flushed; the TFLOP/s curve and the flush delta tell
     how much of the gap is DRAM vs loop overhead before any kernel edit.
   - b. Structural, bitwise-or-tolerance candidates in `flash_kernel`: 16-byte Q8 fetches (uint4 instead of char4 +
     scalar scale); split the per-tile staging so K unpacks before QK and V unpacks only after the scores (halves the
     barrier-to-use distance and the live LDS); restage the 31-distance rel window as a sliding ring instead of a full
     62-float reload per tile. Measure each alone; keep only wins.
   - c. Occupancy/pipelining probes: 16-token x 16-key is the tested-best geometry on this card, but the 208-VGPR
     OWNED16 build allows 2 blocks a CU - check whether a 3rd query group (48 queries a block) fits and helps at long
     pos.
   - d. Only if b/c stall: packed-F16 QK/PV (`__hfma2`) - a precision change; opt-in, gated by the R7 KL/battery
     protocol, expected up to ~1.9x the line on paper, rejected if the KL moves outside the engine's own noise floor.
   Expected from b/c: 1.2-1.5x the line (-1.7 to -4.3 s at 32K; at 200K the same factor is worth several minutes of
   TTFT). Reference: the fork's kernel lineage went 3.5x on WMMA hardware; RDNA2 has no matrix path, so the VALU
   ceiling bounds us near 2x.
2. **P2 NVFP4 expert line (35.8%; incremental, known surfaces).** Port the decode round's fused
   `situ_rows_q8_1` to the prefill grouped path (today prefill runs F32 `situ_rows_kernel` 475 ms + a separate
   quantize of `hx_`); the three bitwise surfaces are already mapped from the decode round. Expected 0.5-1.5% E2E;
   low risk. The gate/up rate itself (31% of roof) has no cheap lever left (K128, tile sweeps, vectorized loads all
   measured dead in the earlier rounds); revisit only with a captured-groups bench1-style microbench if P1 finishes
   early.
3. **P3 dense MMQ output precision (10% line; opt-in).** F16 (or Q8_1) output from the Q8_0 MMQ epilogue where the
   consumer allows it, halving the 89.2 MB/chunk-visit store traffic. Precision change: same R7 protocol as P1d;
   the fork's own accuracy result (KL inside their noise floor) is the reference. Expected 0.5-1.5% E2E.
4. **P4 long-context decode attention (production context is 200K).** bench1 decode spans (16K-190K, hot/flushed)
   re-establish the curve; then: `kAttUnroll` 2->4 (more loads in flight a lane; ~10 VGPRs a key; watch the 128-VGPR
   2-blocks-a-CU line); chunk cap 128->256 at long spans (scratch 1.6->3.2 MB a row); scale loads paired per lane.
   Each is a small kernel edit with the attend test as the gate. Expected 10-30% on the span line at 64K+; at 200K
   decode that is ~4-12% tok/s; nothing at 16K (0.75 ms of 21 ms).
5. **P5 the situ+down merge at decode (the mapped remainder, est +1.2%).** The three bitwise surfaces and the merge
   site are recorded in the decode round's map; its own commit and A/B.
6. **P6 KV snapshot SAVE/LOAD (TTFT across restarts).** Versioned file around `save_state` + `save_rows` (conv
   history, SWA rings, full-attention rows, token list, position, layout/config metadata, integrity check); a LOAD
   path that provisions KV, uploads, restores state and registers the prefix with `Conversations` so only the suffix
   prefills; refuse incompatible formats. Acceptance per R6 with the fork's test pattern (fresh process, same greedy
   tokens, suffix-only prefill).
7. **P7 deploy and close.** Stop: re-run the answer battery on the deployed config; restore the service (and fix the
   live-env drift noted below); final account with kept/rejected/limits.

Rejected on measurement (do not reopen without new evidence): the fork's f32-output GEMM fix (our SGEMM is at the
store roofline, 0.55% of prefill); the router split-K (420 ms = 1%; a cheap follow-up only if the round idles);
"add split-key" as such (already our design); speculative decoding; cache policy; a transposed-WMMA port (no WMMA on
gfx1031; the register-level ideas that survive it are in P1b).

## Current Checkpoint

- Closes: R1 (partially - the trace tables are in), R2 (done - the plan above).
- Smallest next action: after the user reboots, run one command -
  `~/.cache/opencode-tmp/opencode/prefill-opt3/post-reboot.sh` - which chains `dbg` (staged
  allocation probes of all three geometries) -> `check` (wiring) -> `ceil` -> `flash` -> `decode`,
  every mode under the guard, stops/starts production itself, and stops the batch on the first
  nonzero exit or non-empty stop-reasons; then fold the numbers into the Profiling section and
  start P1b.
- Expected evidence: bench1 logs (hot/flushed span curves, card ceilings), guard-clean runs.
- Stop or replan if: the flash structural edits (P1b) cannot beat the tile-loop overhead (the flushed span curve
  will say), or the quality protocol rejects the precision paths (then P1 stops at the bitwise wins).

## Current State

- Resolved: the fork's account is fully mapped (webfetch of inkling-rocm docs/02_results.md incl. sections 11.2-11.12);
  six explore agents mapped our flash, decode attention, GEMM inventory, decode visit structure, KV persistence and
  bench tooling; the retained 8K/32K trace tables are extracted above; the fork-vector comparison is measured.
- Last relevant evidence: the tables above; `prefill-opt3/` scratch (bench1.hip built; `run-dbg3` alloc probe PASS;
  `run-check2` the fault record).
- Blocker: the guard refuses this boot after the 19:15:23 bench1 fault (my gate-buffer sizing bug; desktop survived);
  the remaining GPU RECON (bench1 runs) and all kernel experiments wait for a reboot, per the standing rule C4.
- Next: user reboots (or lifts the rule explicitly) -> `post-reboot.sh` (the whole bench1 batch) -> P1b.
- Blocked-time prep (19:17-19:27, no GPU touched): bench1 re-verified against the header
  (`Attend` field shapes, `attend_plain/flash/decode` signatures - all match); fixture gate fill
  fixed (the loop under-filled `[n][24][128]` to `[n][24]`, benign for parity but unrepresentative);
  `dbg` mode now stages allocation probes through the small, flash (198,656 rows) and decode
  (190,008 rows) geometries with a Buffers destructor between them, so an OOM/mapping problem
  surfaces before any kernel runs; `post-reboot.sh` written and bash -n checked. bench1 rebuilt
  (BUILD-OK 19:26).

## Material Decisions

- 2026-11-23: The three fork vectors below are rejected on our own measurements, not by assumption: the f32-output
  GEMM pathology (our rocBLAS SGEMM: 8.2 TFLOP/s, 226 ms = 0.55% of a 32K prefill), the skinny-gate split-K (the
  same product family; our down is the pipelined 2x2 kernel at 0.46 ms/launch), and "add split-key decode" (our
  decode_q8 is already that design at the same fraction-of-streaming as the fork's fixed kernel).
- 2026-11-23: Precision-level changes (P1d, P3) are allowed in this round but only opt-in and only through the R7
  gates (KL inside the engine's own noise floor + the answer battery + byte-identical off path). The user's hint
  itself ships such changes with measured accuracy; the fork's noise-floor method is adopted as the protocol.
- 2026-11-23: After the bench1 fault, the C4 reboot rule is enforced on this round too: no GPU run until a reboot;
  bench1 is fixed and rebuilt first so the post-reboot run starts clean.
- 2026-11-23: Noted for the deploy step: the live production process at inspection ran `strata-poly-21544cc0` with
  `STRATA_INKLING_{NORM_Q8,TAIL_MOE,ROUTE_DUAL}=0` in its environment while the on-disk config points at
  `strata-poly-56eb42bd` with all three `=1` - the deployed switches were not active in the running process. The
  19:21:25 service restart after the fault (done to restore production while GPU work is paused) loaded the on-disk
  config with the switches on; the deploy step (P7) still verifies the switches from `/proc` after its restart.

## Checkpoint History

- 2026-11-23: RECON opened. Read the two prior rounds' goals in full; fetched the fork's docs/02_results.md (all
  sections) and its repo layout; launched six explore agents (flash, decode attention, GEMM inventory, KV persistence,
  decode visit structure, tooling/dims); extracted the retained 8K/32K per-kernel trace tables and the decode token
  table; measured the snapshot-relevant disk rates (2.8/1.5 GB/s O_DIRECT); wrote and built `bench1` (ceilings,
  flash/decode span sweeps, wiring checks) in scratch.
- 2026-11-23: GPU fault at 19:15:23 (bench1 gate buffer `[n][24]` vs the kernel's `[n][24][128]`); guard caught it,
  desktop survived, boot refused per C4. Fixed the sizing (gate `[n][24][128]`, KV scales 64 B/row), rebuilt bench1;
  the allocation probe passes; GPU runs resume after a reboot. Plan frozen from the trace tables meanwhile.

## Completion

- Pending; see R1-R8.
