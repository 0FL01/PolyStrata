# Goal: faster end-to-end Inkling prefill on RDNA2

Status: active
Source: user direction of 2026-11-22, copied below; the same day's measured prefill RECON.
Last updated: 2026-11-22

## Objective

Raise real end-to-end prefill throughput of the deployed Inkling-2-Mini UD-IQ4_XS on this RX 6700 XT / Ryzen 7
5700X. Start with P1, then follow measurements, not the order of a fixed checklist. Integrate, commit and push
each repeatable improvement to main, use it as the next baseline, and deploy the qualified final build.
Continue while there is a concrete promising, falsifiable in-scope hypothesis; no claim of a global optimum.

## User directive (verbatim)

> Переходи от реализации к RECON, цель максимально ускорить реальный end to end prefill, а не просто выполнить план, план это GPS навигатор, а не рельсы
>
> Используй текущий план как карту гипотез, а не жёсткий маршрут, начинай с P1
>
> Работай автономно длинными циклами: профилирование -> гипотеза -> временный прототип/бенчмарк -> измерение -> анализ -> следующий эксперимент (если профит детерминирование получен - коммит и пуш в main, далее это будет как baseline для последующих правок (т.е закрепляем результат и идём дальше)
>
> При неудачах выясняй почему ожидания не подтвердились, пересматривай гипотезы и пробуй другие подходы, не Зацикливайся на одном направлении и свободно меняй приориты по результам экспериментов
>
> Успешные оптимизации интегрируй в main (прод бинарник), проверяй end to end прирост
>
> После существенных изменений, перепрофилируй движок, bottleneck может переместиться
>
> ПРодолжай исслендвоание и реализацию, пока останется перспективное проверяемые гипотезы, в финале покажи измеренные результатыы, неудачные подходы и оставшиеся ограничения, и деплой в прод
>
> ---
>
> Делай копию плана и моей подсказки в цель, помни что документ с целью это глина, её можно мять как угодно по ходу работ, в эту глину пиши удачи/неудачи

## Execution directive

Use the user's adaptive research loop. The outcomes and affected constraints below define the task; the plan is
a revisable hypothesis map, not a frozen implementation order or a substitute for an end-to-end result. Each
checkpoint records an improvement or a falsified expectation and changes the next experiment. Finish when
the qualified improvements serve production and no remaining concrete promising hypothesis is supported by
the current measurements; record untested limitations. Stop on a fresh guard fault or proven external blocker.

## Required outcomes

- R1: Repeatable real prefill improvement, beginning with P1.
  - Source: "цель максимально ускорить реальный end to end prefill", "начинай с P1".
  - Acceptance: paired uncached model requests, same fixtures/settings/profile seed, faster beyond control variation;
    kernels alone are not sufficient. Record correctness checks and any trade-off.
  - Primary evidence: guarded end-to-end runs on the RX 6700 XT, their `prompt_ms`, measured GPU tested.
  - Status: verified
  - Evidence: integrated opt-in P1, unfrozen ABBA at 2K/8K/16K and code/prose 8K, fixed-placement forced
    distributions, direct/bounce, SWA rewind checks and fresh trace; checkpoint history below.
- R2: Adaptive research and new baselines.
  - Source: profiling -> hypothesis -> prototype -> measurement -> analysis; reprofile after substantial changes;
    continue while promising falsifiable hypotheses remain.
  - Acceptance: measured successful and unsuccessful experiments recorded here, priorities updated from the latest
    timeline, each retained improvement committed/pushed to main before becoming the next baseline.
  - Primary evidence: checkpoint history, per-stage commits and paired runs, fresh profiles.
  - Status: in_progress
- R3: Production deployment and final measured account.
  - Source: "деплой в прод", "в финале покажи измеренные результатыы, неудачные подходы и оставшиеся ограничения".
  - Acceptance: tested final binary/config installed with rollback, authenticated local server and live answer/tool
    checks pass; final summary includes measured gains, rejected ideas, limitations.
  - Primary evidence: production binary hash, `strata.service`, authenticated live checks and final report.
  - Status: pending
- R4: Durable, editable goal with the plan and user directive.
  - Source: "Делай копию плана и моей подсказки в цель ... в эту глину пиши удачи/неудачи".
  - Acceptance: this file retains source intent, hypothesis map, current evidence and next action.
  - Primary evidence: this document and its checkpoint updates.
  - Status: verified

## Constraints and scope

- HIP/RDNA2 only, one card/sequence, same GGUF and Q8_0 KV; preserve model work and request semantics, no expert skipping.
- Repository rules: new features opt-in, default path byte-identical; production can enable measured flags. No other-backend
  compatibility work. Keep changes local to evidence-supported Inkling prefill paths and their direct primitives.
- GPU workloads sequential through `rdna2-rnd/scripts/run_guarded.py`; 1536 MiB free VRAM floor, stop on a fresh fault.
- Benchmarks use isolated copies of one frozen expert-profile seed, not the live additive profile. Timed measurements
  separate from traces/PMC and diagnostic logging. Freeze can isolate experiments but does not qualify production behavior.
- Keep server on 127.0.0.1 with its existing API key. Downtime is allowed. No binaries, traces, credentials or model files in Git.
- Final long HTTP timing uses an equivalent private server/config/profile copy; the actual deployed unit gets
  the short authenticated answer/tool smoke, not tens of thousands of synthetic tokens added to its live profile.
- No unrelated refactoring, services or dependencies. Scratch artifacts stay outside Git.

## Change envelope

- `src/poly/engine.*`, `inkling2.*`, `inkling_attend.*`, `mmq*`, local kernels and their declarations/direct callers;
  CMake/tests only when a retained change requires a target or a concrete regression check. Docs for measured settings.
- `rdna2-rnd/scripts/run_guarded.py` only for the reproduced R2 safety-gate blocker: journald byte-array text
  decoding without changing fault patterns, monitoring, termination or the VRAM floor (checkpoint history below).
- Work on `main`, push task commits to `origin/main`. After qualification install the production binary and only its
  measured opt-in switches, with a rollback copy. Preserve unrelated configuration and decode optimizations.

## Starting evidence

Baseline `a27c4a6c` / production code `ab8507f1`, binary SHA-256
`6b41731112562a7561c8ce9ff7dba30cf322d1fa82093ed1b262f5c3daca558a`.
RX 6700 XT gfx1031, Ryzen 7 5700X, 32 GiB RAM, TheRock HIP 7.13. Batch 2048, reserve 2048 MiB,
200000 context capacity, 8 CPU threads, swaps 64, conversation cache 0, no DSpark. Existing production env:
`STRATA_HIP_NVFP4_MMQ=1 STRATA_INKLING_FLASH=1 STRATA_POLY_DIRECT_STAGE=1 STRATA_INKLING_DECODE=1`.
Service was already inactive before this work. Full scratch RECON:
`~/.cache/opencode-tmp/opencode/inkling-prefill-recon-20261122/REPORT.md`.

- Clean 8K trace: span 11.174 s, kernel union 8.649 s, copy union 2.179 s, overlap 8.4 ms, copy-only 2.170 s.
  NVFP4 MMQ 4.034 s, Q8_0 MMQ 1.086 s, Flash 1.454 s, small F32 GEMMs 0.373 s.
- Frozen scratch reorder: ABBA 2K 771-772 -> 939 tok/s; 8K 739-746 -> 892-904; 16K 706 -> 844-847.
  A1/B1/B2 had identical 816 count/residency/group log rows. No-log repeat also faster.
  Trace B: overlap 2.084 s, copy-only 0.085 s; kernels/copy durations nearly unchanged. Not qualified without freeze;
  a mechanical move queries the previous router event. Specific HSA/SDMA cause not established by simplified probes.
- Real-weight MMQ sweep: gate ~14.7 nominal TOPS at 64 experts x128 rows, pure measured dot4 roof 39.2 TOPS.
  J32/J128 and I64 lose; I64 increases estimated active blocks but loses reuse. All compared outputs bitwise.
- Real routing: <=32-row expert-visits are 51.5% of experts but 2.53% of rows. Staged classes 8/16/32 screen ~1-2%.
- Full attention n2048/pos32768 ~165-167 ms, SWA ~5.3 ms. Head-serial variant bitwise but not faster, VGPR 232->248.
  Cold/hot and PMC at n512/pos8192 do not confirm external-VRAM saturation; do not extrapolate this to all contexts.
- Batch4096 screen loses at reduced residency; it does not separately identify the cause. H2D coalescing synthetic
  25.85->28.26 GB/s, streaming VRAM read ~379.5 GB/s. Existing attention test passes. No new GPU faults.

## Plan copied from RECON (hypotheses, freely revisable)

### P1: prefill staging scheduling

Make early Inkling prefetch opt-in and correct without freeze. Enqueue before router readback, keep `ev_free_ ->
upload -> ev_ahead_ -> consume` and staging/bounce lifetime. Separate enqueue from route timing feedback rather
than querying the previous `ev_`. Check end-to-end 2K/8K/16K, partial chunks, short requests, decode, SWA wrap/restore,
default-off parity and model distributions/forced replay. Reprofile the qualified new baseline.

### P2: NVFP4 MMQ inner-loop specialization

Largest compute cost after P1. Analyze dot4 versus unpack/scales/addressing/LDS. Try a smaller K-step/pipeline
that keeps I/J reuse (K128 needs a local kernel, not a configuration-only upstream edit). Check unpack reuse and
down K-tail 1152; do not globally expand weights and lose expert residency without an end-to-end comparison.
Use the captured uneven resident/staged distributions, not just balanced matrices. The earlier suggested
15% kernel / 5% model screens are experiment-selection heuristics, not the finish line or a reason to stop.

### P3: full attention at longer prefix

Investigate VALU/SFU/dependencies, lane ownership of replicated probability work and live registers. Keep FP32
operands. Head-serial was rejected. Compare full specialization at 8K/32K/64K, preserve SWA/short cases, numerical
spike/sink/tail checks and model behavior. Reorder P2/P3 if the new long-context profile warrants it. Capacity
200K does not mean measured full-window throughput.

### P4: follow the remaining profile

Qualify staged size classes; consider SiTU+MMQ packing and shared dense packing (native decode Q8_1 differs).
Only pursue control/mix/norm/conv fusions where their measured contribution warrants them. Do not increase batch
to 4096 by default after the negative screen. A looped layer-major rewrite is not the first move while cheaper
scheduling changes hide most DMA. These are hypotheses, not required implementation items.

## Current checkpoint

- Progresses: R2; R1 verified by P1.
- Next: retained baseline is **`c9bb85d9`**, actual build `09304bb1...`, with prefetch, down-tail, batch-KV,
  selective2x2 F32, OWNED16 and PACK enabled. Test the faulting cost exposed by plain snapshot timing as
  one remaining concrete private prototype, then final original-vs-final/production. F32's32K HTTP gain is verified;
  its8K API screen remains inconclusive.
  Empty-grid MMQ cost is small; no compact-grid mechanism is justified by that screen. Production is pending.
  Actual-root VMM frozen restore and unfrozen HTTP ABBA passed. SiTU remains unshipped; production not deployed.
  The query16 lane-owned attention, packed checkpoints and selective2x2 F32 projections remain model candidates;
  rebase them on each retained baseline. Do not disable checkpoints or merely move measured work afterDONE.
- Replan on: failure to retain the effect without freeze, correctness mismatch, or fresh profile showing another
  larger actionable cost. A rejected idea is recorded and replaced, not repeatedly retested unchanged.

## Checkpoint history

- 2026-11-22: prior RECON copied above and directive preserved; task active, no engine edit yet. Baseline tree clean.
- 2026-11-22: P1 integrated as `STRATA_INKLING_PREFETCH=1`, set only by Inkling; common prefetch work retains its old
  location with the switch off. Upload moves before router submission with the switch on; feedback observes the
  current router event after it is recorded, including without freeze. gfx1031 `strata-poly` builds. Independent
  static review found no blocking issue in event/lifetime/default-off behavior. No open threads in this fork's
  GitHub issues/PR list on 2026-11-22.
- 2026-11-22: P1 numerical checkpoint, **without freeze**, zero resident slots (reserve 10000 MiB only for placement
  isolation; batched experts still run on the GPU). Old baseline/current off/current on have identical printed
  top-10 distributions over 8 forced steps after 1064 tokens. Current off/on also identical after 8192 tokens,
  and with direct staging disabled (bounce fallback). `guard-p1-parity` reports no faults. Scratch for this goal:
  `~/.cache/opencode-tmp/opencode/inkling-prefill-e2e-20261122/` (`qualify.py`, `p1-par-*`, `p1-long-*`, `p1-bounce-*`).
  Full production-settings ABBA runs are the next evidence, not inferred from these zero-cache timings.
- 2026-11-22: P1 real end-to-end checkpoint without freeze, same binary with only prefetch 0/1 changed, fresh
  profile copies per start, batch2048/reserve2048/threads8/swaps64/Q8_0/capacity200000, no prefix reuse. ABBA:
  2K A 2694.7/2683.0 -> B 2186.7/2271.1 ms (760-763 -> 902-937 tok/s); 8K A11173.0/11100.1 ->
  B9107.8/9098.0 (733-738 ->899-900); 16K A24026.5/23825.9 -> B19845.1/19546.8 (682-688 ->826-838).
  B wins both pairs at all lengths. Separate 8K code: 11152.8/11157.7 ->9147.3/9139.2 ms (734->896 tok/s);
  prose:11133.2/11294.8 ->9119.5/9097.3 (731->899). All guards clean. Raw: `p1-e2e-{A1,B1,B2,A2}`,
  `p1-{code,prose}-{0,1}`, `results-summary.json`. Reprofile/state checks follow before the baseline commit.
- 2026-11-22: P1 qualification complete. SWA `state-wrap.py` at pivot4096 (>3071 physical ring) restores the same
  six IDs as a full read, both off and on. A production-settings 94+96 forced decode control gives 42.31/42.67
  tok/s, not a repeatability claim for decode. Its distributions differ: this unfrozen short PP uses timing-adaptive
  CPU/GPU expert splitting and swaps, so it is not the fixed-placement numerical check above. Investigate only
  if controlled checks expose a regression; this path does not upload ahead at n94. All guards clean.
- 2026-11-22: fresh **unfrozen integrated** P1 trace, measured 8K after8K: span9145.482 ms, kernel union8727.060,
  copy union2193.627, overlap2106.622, copy-only87.005, idle331.416. NVFP4 MMQ4029.895, Flash1457.140,
  Q8_0 MMQ1098.370, small GEMM375.029, quant181.574 ms. Thus P2 remains first compute hypothesis; P3 becomes
  more important with prefix length. Raw `p1-trace/trace`, `p1-trace/analysis.json`. P1 binary SHA-256
  `46b5d95ba2867a9f0d57237b0bf1fe11995c2c4abee8c568a4ef81f3f8c3a5f2`; no arithmetic/precision change.
- 2026-11-22: P1 committed/pushed as `4345219b`; this binary with `STRATA_INKLING_PREFETCH=1` is now baseline.
- 2026-11-22: P2 K128 local-copy prototype compiles and passes exhaustive CPU indexing checks plus 98,304 real-weight
  simulated outputs. GPU strict smoke (empty/partial groups) and balanced real-GGUF tests are bitwise identical.
  I128/J64/256threads kept; LDS51200->30720B, VGPR240->206, no spills, HIP-estimated active blocks1->2.
  **Expectation partly falsified:** gate at64experts x128/256rows gets 5976.9->6210.3 /11792.1->12174.8 us
  (3-4% slower), down3456.6->3207.1 /6665.9->6177.3 (about7.8% faster). Lower resources/greater occupancy alone
  do not accelerate gate. Down removes the old128-value paddedK tail, making a down-only model screen worthwhile.
  Captured uneven groups and gate PMC follow before attributing the cause. Raw `mmq-k128/`.
- 2026-11-22: P3 lane-owned probabilities, temporary kernel: each8-lane group retains2keys/head/lane instead of16;
  exp instructions per tile51->9, DPP8 broadcasts keep sequential16-key denominator sum and per-key PV order.
  VGPR232rounded->208, LDS unchanged18368B, zero spills. Eight GPU geometries (512/2048queries x8K/32Kprefix
  xSWA/full) **bitwise equal** to currentFlash;14 plain-reference casesPASS including190K dominantkey,60K sink,
  ringwrap/tails/FP16. Timings full2048@32K167.44->158.85ms, SWA5.245->4.986ms (~5% kernel gain).
  This is not yet an engine gain: model A/B is required, especially atlongerprefix. Raw `attention-prob/`.
- 2026-11-22: captured uneven P2 groups confirm gate K1280.961-0.971x and down1.069-1.087x, all bitwise. Gate's
  FETCH_SIZE counter is nearly unchanged (386217/385521 KiB per dispatch). **SQ PMC unusable in this experiment:** zero
  VALU/LDS after first sample, occupancy0 and SQ_WAVES65814 rather than18432 for589824threads. A corrected
  single-counter rerun reproduces invalid values; no hardware-occupancy claim is made. HIP's estimate of2blocks
  is not a measured occupancy. The first single-counter command only rejected `--warmup 0`; rerun used1.
- 2026-11-22: temporary **down-only** model uses the original auto-J selection; only supported NVFP4 down1152,
  max_rows>=128 and J64 qualify (129-row J48 is deliberately excluded). Control relink equals P1 bytes; ABI
  offsets checked. 8K zero-cache model forced top-10 is identical to P1. Unfrozen same-binary ABBA, P1 enabled:
  8K A9117.5/9108.8 ->B9030.3/9024.3 ms (+0.95% throughput), 16K A19750.8/19541.6 ->B19357.1/19263.7
  (+1.74%), 32K A44116.8/44121.0 ->B43713.1/43802.1 (+0.83%). All pairs positive, all guards clean.
  This clears an end-to-end signal, not the old hypothetical5% gate. Before production integration, test whether
  keeping K256 but skipping only down's final padded128-value slab does the same work with less overhead/code.
  Raw `k128-e2e-*`, `k128-model-parity`, `mmq-k128/downonly-model/`.
- 2026-11-22: attention no-lookahead CPU variant recovers17VGPRs (208->191raw/192rounded), not the expected~40;
  it misses the ~170 target for a third block. Guarded GPU timing will decide whether the lost memory prefetch
  outweighs this smaller resource benefit. No model build or permanent change for it yet.
- 2026-11-22: no-lookahead GPU check: eight geometries bitwise, but full512@8K10.288->10.623ms and
  full2048@32K159.018->166.973ms versus OWNED (3-7% slower across full cases). SWA roughly flat or~1% faster,
  too small to explain a real-model win. **Rejected:** fewer registers did not cross the residency threshold,
  and removing global-memory overlap costs more than it saves. Next attention hypothesis instead splits GQA
  heads across blocks: more K/V traffic, much smaller live Q/O state; test its reuse/occupancy trade-off.
- 2026-11-22: K256 tail-skip alternative keeps original loaders/config/LDS51200/VGPR240, but omits only down's
  padded second128 slab (10->9 activation slabs,40->36 sum updates,20->18 inner barriers). CPU196608 outputs and
  GPU empty/partial plus8captured groups are bitwise. Captured down1.082-1.101x vs original, gate~flat.
  It beats K128's typical1.07-1.08x without the doubled outer-loop count, so choose a focused tracked tail kernel
  over K128. Do not copy/fork upstream headers. Skipped inputs have zero D4 scales; packed padding bytes need
  not be zero (legacy0*infinity conversion). Model qualification of the tracked implementation follows.
- 2026-11-22: one-head attention GPU check:8geometries bitwise,14reference casesPASS, but~2.1x slower than OWNED3
  (full2048@32K158.958->342.346ms). RawVGPR190 still misses the third-block threshold; hoisted LDS operands consume
  much of the recovered Q/O space and K/V/table staging repeats three times. **Rejected**, do not integrate.
- 2026-11-22: SiTU->MMQ D4 fusion compiles (33rawVGPR, noLDS/spills) and42GPU row/scale/zero/tiny/cancellation
  cases have identical **entire packed bytes**, including pad512 and guards. Separate->fused at1152columns:
  rows2048 .10640->.08080ms; rows16384 .98797->.58713ms. This is a promising~0.5-1% model hypothesis,
  not yet a model result. Raw `situ-mmq/{parity,timing}.jsonl` (CSV content), no permanent source change yet.
- 2026-11-22: focused tracked down-tail source138lines reuses unchanged ggml load/dot/writeback helpers; no
  upstream-header copies. CMake adds it only to HIP/gfx1031; strict `STRATA_HIP_NVFP4_TAIL=1` dispatch preserves
  original auto-J and default path. Three actual-production GPU layouts (global prefix/suffix, arbitrary
  gather/scatter, padded cache/weight/output strides) are full-buffer bitwise with intact canaries;8captured
  down groups also pass. Old/currentoff/currenton 8K zero-resident forced top-10 identical. Independent review
  found no grouping/math issue; guarded device body is restricted to gfx1031. No other-backend builds required.
- 2026-11-22: **tracked** down-tail same-binary unfrozen ABBA:8K A9100.1/9068.4 ->B8953.0/9016.2ms (+1.11%);
  16K A19532.3/19465.9 ->B19115.6/19132.6 (+1.96%);32K A44036.5/43963.1 ->B43463.0/43354.4 (+1.36%).
  Both pairs positive at each length. Baseline P1 on, all other production settings/profile seed fixed; no reuse.
  Raw `tail-e2e-*`, `mmq-tail-production/`, guards clean. Rebuild after the two-line device scope guard before
  promoting the result; the previously qualified binary SHA is `c8a238762cf47e10975f514c5641cfff6a468771816c4d58914c8596cdf94ebe`.
- 2026-11-22: gfx1031 rebuild after the device guard produces the **same full executable bytes**, SHA
  `c8a238762cf47e10975f514c5641cfff6a468771816c4d58914c8596cdf94ebe`. Thus the GPU/model/E2E evidence above is
  current for the final source, not inferred across a changed binary. Default-off original path retained;
  no gate/up or decode algorithm change. Ready to promote this second measured baseline.
- 2026-11-22: down-tail committed/pushed as `914336f6` (P2 binary c8a238... with prefetch/tail enabled). Parent
  includes the intervening user documentation commit `753b8cdd`; it changes no measured engine code. All future
  experiments use this retained code/settings as baseline rather than the old P1-only engine.
- 2026-11-22: actual HTTP/SSE baseline (old `ab8507f1` bytes, private server/config/profile copies, loopback+auth):
  fresh8229-token requests,16outputtokens, zero reused prompt tokens. TTFT(first text, including reasoning)
  13034.7/12803.1ms afterwarmup; role-only SSE header is~9ms and is **not** TTFT. Reasoning-only short generations
  are not a coherence test; check a budgeted full reply separately at deployment. Artifacts `api-baseline-v4`.
  Harness setup needed a free port and a valid `strata-poly-<8/12hex>` basename (server feature detection); no
  production sampling/config/server-code change. Ordinary urllib bypassed loopback already; helper now explicitly
  disables proxies. Failed harness starts are not performance data.
- 2026-11-22: next R&D priority is real partial chunks, not another occupancy sweep. `Session::generate` leaves
  one token for logits; `Engine::read` splits at rows2048 and checkpoints. Tails32..1023 use CPU/GPU splitting,
  not the ahead path; aligned-length benchmarks can hide that. A scratch flag will test balancing the last two
  chunks within each mark segment, preserving total rows/call count/checkpoint positions and scratch limits.
- 2026-11-22: **read-rebalance expectation falsified for the actual defaults before GPU timing.** Cache every2048
  remains enabled even with conversation-cache-mib0; all due marks bound calls, including discarded/null-state
  marks. Thus8229 API input is2048+2048+2048+2048+36, then1, both flagstates. The prepared model/1,479,981CPUcases
  establish a no-op for all proposed default-policy sweeps. No production integration and no pointless GPU
  rerun. Larger prompt-batch values likewise cannot make text calls exceed2048 while this cadence remains.
  Changing/disabling checkpoints just for a positive cold benchmark would risk real prefix-reuse behavior.
- 2026-11-22: new same-cache-policy hypothesis: current8K trace has134.3ms idle between successive D2H copies,
  plus42.7ms D2H->runtimecopy, inside331ms idle. `save_state` performs many synchronous tiny transfers.
  Test packing the unchanged convolution/SWA snapshot into the existing expert stage buffer with asynchronous
  D2D copies, then one bulk D2H; no new persistent allocation. Preserve all bytes/checkpoints, clear overwritten
  staged-weight metadata, synchronize before reuse, and fall back when unavailable/DSpark. Temporary prototype
  and full-byte/restore validation precede any integration. This is a measured direction change, not a new outcome.
- 2026-11-22: fresh unfrozen P2 trace after8K warmup: measured8K span8999.561ms, kernel union8571.651,
  copy union2215.403, overlap2125.825, copy-only89.578, idle338.331. NVFP4 combined3888.697ms
  (original MMQ2736.351 + native down-tail1152.346), Flash1457.354, Q8_0 MMQ1097.163, smallGEMM375.353,
  packing181.850, SiTU116.864. New symbol names must not be mistaken for removed work. Checkpoint-related
  D2H->D2H idle134.714ms persists; fill->H2D107.894ms and D2H->runtimecopy42.080ms. The second baseline
  keeps P1's overlap; attention/fusion/checkpoint hypotheses remain measurable. Raw `p2-trace/analysis.json`.
- 2026-11-22: same real HTTP/SSE fixture on retained P2: fresh8229prompt+16output, cached0, first-text TTFT
  10269.1/10538.4ms versus original13034.7/12803.1ms. Mean12.919->10.404s,19.47% lower latency
  (24.18% higher reciprocal throughput). Separate private servers/profile copies, same payload/settings;
  clocks unlocked, not a fine-grained estimate of P2 alone. Role-only headers are excluded. Raw `api-p2`.
- 2026-11-22: current-P2 knob screen, A/T/C/D/A, warm8K, fresh8K/8229/8704: staged8/16/32 alone
  9544.8/10044.9/10502.4ms versus A means8960.2/9325.5/10097.6; CPU-share0 alone
  9018.2/9459.9/10183.2; both9153.4/9501.4/10178.4. **No retained win.** The earlier1-2% staged-tiles
  result does not survive this baseline/screen. Extra small-class launches only address5.4% of staged rows;
  reducing CPU work on partial tails is not automatically faster. One sample/arm and uncontrolled initial
  residency (A1424slots, T1429) do not prove exact regression sizes. Keep existing adaptive/default settings.
  Raw `p2-knobs-*`, all guards clean.
- 2026-11-22: packed-state prototype passes actual GPU **full92875776-byte** comparisons at positions93,
  2048/4096/6144/8192; existing rewind at4096 restores the same six generated IDs as a full read. It also
  byte-compares short/rewound states at6/4104/4199. No policy/marker change, no new GPU allocation, old serializer
  used as reference with verification on. Guard clean. Raw `state-pack-verify`, `guard-state-pack-verify`.
  Verification must be disabled for timing; numerical success alone is not an E2E improvement.
- 2026-11-22: SiTU->MMQ temporary model at P2, unfrozen ABBA:8K A8993.7/8957.9 ->B8894.0/8885.3ms
  (+0.969% throughput, both pairs faster, control drift0.40%);32K A43551.5/43503.1 ->B43382.8/43330.0
  (+0.394%, both faster, control drift0.11%). **16K inconclusive:**A19167.2/19292.1 ->B19310.1/19103.6
  (+0.119% mean, one pair0.74% slower). Fixed-placement8K+8forced top-10 equals the golden baseline.
  No GPU faults. Independent kernel review found no blocking arithmetic/index issue. Prepare the minimal
  opt-in tracked implementation, then qualify that build rather than promoting scratch timing alone.
  Raw `situ-e2e-*`, `situ-model-parity`, `measured-summary.json`.
- 2026-11-22: one additional attention hypothesis follows the failed occupancy variants:32query/256thread
  blocks instead of16/128, still16keys/GQA3/8lanes per query. This doubles KV reuse without the one-head
  variant's tripled loads; projected1x256 versus2x128 resident threads is not a measured occupancy claim.
  Compile/strict parity/kernel timing decide whether it survives. Keep only one such geometry, not a blind sweep.
- 2026-11-22: **query32 rejected.** Actual rawVGPR208->196, LDS18368->19392, no spills; fewer global loads
  did not translate to throughput. Four512-query geometries are bitwise but6-8% slower:full8K10.299->10.928ms,
  full32K40.389->43.580, SWA1.328->1.437. The sweep stops at2048/SWA/pos8192 with446 tiny bit differences
  (max3.73e-9, normalized4.57e-8), although14plain-reference casesPASS. A wider block can execute extra fully
  masked tiles; investigate compensated-zero updates rather than weakening the strict comparison. No model
  build/integration/retry, no fresh GPU faults. Raw `attention-query32/`, `guard-att-query32`.
- 2026-11-22: FP16 packed-state actual-GPU byte verification alsoPASS at93/2048,171442176bytes each. This
  validates the existing alternate KV layout only; production remains Q8_0. Raw `state-pack-fp16-verify`.
- 2026-11-22: opt-in lane-probability source uses separate compile-time Flash specializations. Actual native
  false Q8/FP16 machine bytes/resources equal P2, true equal qualifiedOWNED3, nine non-flash kernels unchanged.
  Actual host flag0/1 each passes8finite/bitwise GPU geometries;14reference casesPASS on1. No additional timing
  claim from these unchanged kernel bytes. Model rebase waits for the next retained baseline. Raw
  `attention-prob/optin-source`, `guard-att-optin-dispatch`.
- 2026-11-22: query32 arithmetic explained from actual ISA/source: worst differing element isquery43/head0/dim7.
  Valid keys7212..8235; query32 executes an additional fully masked8240..8255 tile. Withsum0/shrink1 the
  compensated denominator still computesFP32(l-lc), not necessarilyl; exact-FP32 emulation demonstrates
  a one-ULP compensation fold. Matched update ISA has the same contraction DAG, so a changed compiler
  reduction is not needed to explain differences. Internal GPUl/lc were not captured; not a replay of all446
  elements. Keep query16 and its tile schedule, do not relax parity or pursue the slower query32 geometry.
- 2026-11-22: provisional tracked SiTU fusion builds on914336f6: `Mmq::situ_nvfp4()` rejects unsupported
  shapes/devices before writes; Inkling alone sets the default-false flag. Cached/staged fallback keeps the
  original calls. New85-line kernel uses actual ggml D4 types; its machine bytes equal the qualified scratch
  kernel, all existing model HIP.text unchanged, Engine bool uses padding.71150CPUcondition casesPASS.
  Binary `cb6b5dd3bfb2b6862d7d2c79fd1303329cdfcd940fb409e838fb0fc353e5c32b`; actual production GPU/model
  checks and timing remain required before commit. Artifacts `situ-mmq/production`.
- 2026-11-22: actual tracked SiTU kernel passes all42GPU packed-byte/canary cases. Actual binary flag0/1 each
  matches the8K+8forced golden top-10 trace at fixed placement. Active-on banner confirmed, guard clean.
  This closes tracked numerical/default-off qualification; timing waits until CPU builds finish.
- 2026-11-22: SiTU review caught a shared-source link regression: a runtime-false call still required the new
  MMQ symbol in Metal. Restricting the call to the existing HIP compile guard, with old-path false fallback,
  removes the dependency; no backend implementation/test expansion. Independent review cleared it. Rebuilt
  gfx1031 **full executable remains byte-identical** cb6b5d..., so all prior tracked GPU checks remain current.
- 2026-11-22: private actual HTTP smoke on tracked SiTU passes auth401, coherent answer "17 plus25 is42",
  automatic `add_numbers(a=17,b=25)` call and tool-result answer "42". No external tool/code executed.
  This is functional qualification, not timed performance. Raw `api-situ-smoke/full-smoke`.
- 2026-11-22: existing F32 tile screen **rejected**:16shape/row cases full-buffer bitwise, finite and guardsPASS,
  but2048rows small->tile2304x24 .2215->.3280ms,2304x32 .2233->.3520,2304x64 .4278->.4323,
  4608x64 .8381->.8550. Short32/77rows are much slower. Actual old selector isout>64, not>=128.
  Keep it unchanged. The4x4 tile has fewer blocks, more masked output work and lacks the small kernel's
  next-step global-prefetch pipeline; this screen does not isolate one hardware cause. Raw `gemm-small-tiles`.
  One different candidate will preserve that pipeline with16x32 blocks,128threads and2x2 FP32 accumulators,
  reducing LDS data per FMA while keeping input order. No blind geometry sweep and no model claim yet.
- 2026-11-22: **extended actual-build SiTU ABBA does not qualify a win.** Two rounds of8K/16K/32K, warm8K:
  mean throughput differences-2.41%/-3.18%/-2.48%; both8Kpairs slower. Raw `situ-prod-e2e-*` is retained,
  not discarded as a successful test. No commit/push. The helper loops rounds outside lengths: the second16K
  follows32K->8K. In all four arms that16K spends3558/3708/3561/3573ms before its firstRESUME versus24-50ms
  normally, at the KV-trim/cache-grow boundary;16K totals22.8-25.2s versus19.3-20.5s initially. A separate
  slow period spans lateB1/earlyB2; its hardware cause is not established. Do not silently subtract these costs
  from the end-to-end claim. Raw `situ-prod-history-breakdown.json` separates the existing pre-layer boundary.
- 2026-11-22: diagnostic trace removes the32K history, keeping8K/16K/8K/16K. Both16K are stable again.
  Actual GPU union8K8607.86->8540.67 /8659.13->8592.78ms;16K18494.10->18346.66 /
  18622.67->18490.16. Flash/NVFP4/copy work and overlap remain near-identical; fused packing replaces about
  150ms separate work with89.5ms per8K. This supports a real local saving, **not** unprofiled E2E qualification.
  Next isolate mixed-history trim via HIP API and record clocks/power/temperature on the next timed comparison.
  Raw `situ-prod-repeat-trace-{0,1}`. No fault, and no claim that the negative series was disproved.
- 2026-11-22: **SiTU closed without promotion.** Local saving is real in the trace, but repeatable unprofiled
  production E2E gain was not established by the extended test. All seven provisional code changes were
  removed; tracked engine sources exactly match914336f6 again, private source/binaries/proofs retained. No
  unqualified flag ships merely for its microbenchmark result. New evidence would be needed to reopen it.
- 2026-11-22: mixed-history3.5s is independently visible in closed protocol timestamps beforeRESUME, before
  any model layer. Source has32KVregions and up to100x1ms memory-accounting polls **per region** after address
  release. Current free-memory baseline is sampled after physical release, so an already visible release can
  make a target unreachable; this is still a hypothesis until the exact HIP API pattern is checked. Candidate
  preserves all physical unmap/remap operations, checkpoints, residency policy and same `mind_vram` boundary,
  but measures before the entire trim and polls once for7/8 of the aggregate bytes, bounded100ms. No new
  worker/pool and no hiding work outside prompt time. Read-only telemetry will accompany the trace.
- 2026-11-22: K64-prefetch2x2 F32 screen succeeds numerically for all16cases. At2048rows old->new:
  2304x24 .22344->.13344ms,2304x32 .22388->.13384,2304x64 .42928->.23704,
  4608x64 .84385->.45968.512rows also win;32/77rows are1.6-1.8x slower. Preserve those short calls on
  the old kernel. Actual newVGPR121/allocated128 versus64, LDS12480versus10400, no spills; prefetch survives,
  but increased resources did not prevent the measured gain. Prepare one opt-in host selection only for the
  tested shapes androws>=512, then measure the model. Raw `gemm-small-2x2`, no root integration yet.
- 2026-11-22: exact mixed-history P2 trace confirms the KV hypothesis:32polling series occupy3490.211ms,
  including3200sleep gaps totaling3473.076ms. Actual polling HIP APIs take10.536ms. Whole trim3517.120ms
  includes DeviceSync0.132ms,288Unmap14.713ms,128Release3.028ms,32AddressFree/Reserve and160Map/SetAccess.
  Subsequent refill26.498ms host wall, not3.5s. This proves repeated waits, not the precise free-byte accounting
  cause; current CSV lacks MemGetInfo return values. Prototype diagnostics will capture them and actual refill.
  Raw `kv-mixed-trace-baseline`, read-only CPU analysis; no hardware fault.
- 2026-11-22: read-only telemetry on that trace records clocks/power/fan/temperature and process page faults.
  Busy GPU clocks are around2.26GHz, hotspot rises to92C during the last32K. The pre-layer wait lets GPU idle;
  all-sample clock means therefore cannot be used to normalize kernel throughput. No retrospective claim
  that thermal throttling caused the earlier SiTU outliers. Raw `kv-mixed-trace-baseline.telemetry.jsonl`.
- 2026-11-22: private batch-release prototype `5bcd495e...` built from frozen P2; control relinks exactly to
  `c8a23876...`. No new fields/vtable entries/GPU code. Source operation oracle and69native CPU driver-stub
  cases pass. Independent source review finds no blocker; timeout/external VRAM activity can still change
  refill through the existing fresh `mind_vram()` sample, so do not promise identical refill from the7/8 target.
- 2026-11-22: first **unfrozen** rewind/full-read equality fails in the batch arm. Native VRAM expert hits differ
  4055vs4961out of5376, so this is not a controlled numerical comparison. Nothing committed or waved through.
  Repeated the exact32K->8K->16K,14,336checkpoint fork and full reread with frozen placement and3072MiB reserve
  (validation only, enough KV headroom): **both modes pass six-token equality within each process**. Initial
  slots1187vs1185 differ, so no cross-process bitwise claim. Old/new trim2033.994->18.254ms; both free285212672B,
  new free2602684416->2887897088 exactly matches that delta, exceeds2852245504target without extra sleeps.
  An additional68MiB trim also passes and falls1857.183->12.212ms. Diagnostic timings, not final performance.
  Raw `kv-prototype-{frozen-,}restore-*`; all guards clean.
- 2026-11-22: actual opt-in2x2 F32 API dispatcher passes all16finite/full-buffer-bitwise/canary cases with
  flag0 and1, including short fallback. All141old GPU kernels remain identical; sole added kernel matches
  the qualified standalone. Canonical object `gemm-small-2x2/optin-source/kernels_optin.o`; model still pending.
- 2026-11-22: applied only production VMM `engine.cpp/hpp` draft, without temporary diagnostics; build passes
  on gfx1031. Stable binary `kv-release-batch/production/strata-poly-kv-release`, SHA
  `e4555437d9532c3610f562960ed55cebe4cc95465b2587e89acbfc7203b1e577`. **All20,355,104embedded HIP fatbin bytes
  equal P2**, not merely sampled kernels. No rejected SiTU object/member remains. Source review clears actual
  diff. Actual-root frozen restore followed by HTTP ABBA is in progress; no VMM performance promotion yet.
- 2026-11-22: **actual-root KV qualification succeeds.** Frozen actual-root14,336rewind and full reread
  produce the same six IDs. Unfrozen private HTTP A/B/B/A, exact same binary off/on, fresh-profile copies,
  body sequence32768/8192/16384 and16completion tokens: actual prompts32805/8229/16420, cached0 everywhere.
  Last-request TTFT24,164.64/24,482.80ms ->20,688.92/20,806.44ms: **-14.70% mean latency**, paired-14.38/-15.02%.
  This includes the real pre-layer wait, not a renamed timing bucket. First32K control44,548.36->44,588.48ms
  (+0.09%, A/A0.11%); no causal8K gain claimed because that request does not trim. Cache after refill:
  A1362/1354, B1356/1362; adaptive placement varied, not a fixed-capacity test. No systematic refill loss
  observed, and no claim it must be identical under external VRAM activity or timeout. All guards clean,
  both B logs confirm actual batch activation. Raw `kv-api-production-*`, summary JSON and telemetry.
  Retain this opt-in; production deployment is still pending. Next commit/push, reprofile, then F32 model.
- 2026-11-22: VMM improvement committed/pushed as `53348354`; this is the next baseline, not an uncommitted
  experiment. Post-change trace `kv-retained-trace` (warm32K, then fresh8/16/32K) has no hardware fault:
  8Kspan9082.97ms/kernel8571.36/NVFP43885.66/Flash1455.71/smallF32374.55;
  16Kspan19278.40/kernel18391.05/NVFP47737.94/Flash4227.44/smallF32749.42;
  32Kspan43696.28/kernel42003.16/NVFP415559.21/Flash13588.40/smallF321497.47.
  NVFP4 includes both legacy MMQ-40 and native tail. Copy overlap remains2100.53/4124.13/8445.89ms;
  kernel/copy intervals are unions, not additive wall shares. First-request warmup includes load/JIT and is
  not used as steady throughput. Full attention approaches NVFP4 cost at32K; F32's measured2x2 saving still
  makes it the cheaper next model experiment. Separate HIP API analysis checks that the polling is gone.
- 2026-11-22: retained-profile HIP API confirms32KVreservations unchanged, **zero polling sleeps**:
  trim19.727ms wall (18.334insideHIP), subsequent refill16.188ms, experts1293->1353. Start->RESUME about62.9ms
  versus3.57s before. Only two100-sleep series remain, one arena shrink per32K,109.409/109.188ms each
  (~0.25% of that request); no stage lending/widen groups occur for this looped Linux HIP path. Do not
  refactor the general allocator for these small residual waits while larger tested candidates remain.
- 2026-11-22: selective F32 model rebased on53348354, control relink exactlye455. Only canonicalkernels.o
  replaced; modelSHA48450c7eb5f1da13358543aaf074af0377202782b68d7190afac614d361189c3. All141old GPUkernels
  unchanged; sole added6444Bkernel matches standalone. Actual API off/on16casespass; zero-resident8K+8forced
  top10traces both equal oldgolden. Unfrozen8/16/32K ABBA (warm8K, fresh profiles, no reuse):
  A1[8984.5,19362.1,43621.9], B1[8793.8,18925.4,42845.6], B2[8788.3,19048.3,42715.8],
  A2[8946.8,19227.8,43474.8]ms. Throughput gains**+1.99/+1.62/+1.79%**, bothpairspositiveeachlength;
  Adrift0.42/0.69/0.34%. All guards clean, actual B activation banners. Rootkernels.cu now exactly the
  qualified opt-in source (89additions), actual build pending. Raw `gemm-e2e-533-*`; no F32 commit yet.
- 2026-11-22: actual-root F32 build `1cc5d022f7bca930f7e997c3c05a67ea4a14197dd6072a540bc4c4509174ee5d`
  passes old8K+8forced top10golden. Native proof confirms141oldkernels and new6444Bkernel unchanged from
  qualified code; all28otherGPUbundles exact. Wholefatbin initially differed367bytes: one generatedCUID,
  symbol ordering and SysV/GNUhash tables. Constructive renaming/table rebuilding reproduces the complete
  canonical fatbin exactly; this is metadata, not assumed shader equality. ggml-base still embeds914336f6.
  Actual-root8K HTTP ABBA (8229prompt+16output, cached0):
  A1[10267.787,10263.200], B1[10402.055,10213.146], B2[10212.840,10063.215],
  A2[10251.006,10275.033]ms TTFT. Mean**-0.40%latency**, pairs**+0.41%slower/-1.22%faster**;
  this does NOT establish a repeatable8K API gain. The engine's own prompt times also vary, so it is not
  merely SSE framing. Next: same actual ELF, warm8K then two fresh32K requests perprocess, to test whether
  larger measured GPU savings survive real API/short-tail costs. Raw `gemm-api-production-*`; unshipped.
- 2026-11-22: actual-root F32 large HTTP qualification passes. Same1cc5ELF, original production settings,
  P1/TAIL/BATCHfixed1, only F32flag0/1, noFREEZE; warm8229 then fresh32805/32805token prompts, output16,
  cached0. TTFTms A1[45549.116,45231.835], B1[44673.831,44739.403],
  B2[44661.264,44484.476], A2[46004.142,45962.035]. Mean45686.782->44639.743ms,
  **-2.292%latency/+2.346%throughput**, pairedlatencyreductions1.507/3.067%, Adrift1.306%.
  All four corresponding32K cases are faster; all guards clean. This qualifies a real long API-prefill win,
  not an8K API guarantee. Keep the512gate because short kernel32/77caseslost1.6-1.8x. No new buffers,
  defaultold141kernels/resources and FP32term order preserved. Raw `gemm-api32-production-*`, summaryJSON.
  Next: commit/pushF32, profile, then continue surviving OWNED16 and packed-state candidates.
- 2026-11-22: F32 committed/pushed as**af66c57e**, newbaseline. Retained trace `f32-retained-trace` warm8K,
  then fresh8/32K: span8891.36/42918.42ms, kernelunion8452.52/41242.69, copyoverlap2095.60/8372.75,
  NVFP43934.56/15449.66, Flash1456.31/13602.86. Actual F32newkernel206.33/824.99ms versus prior
  smallF32374.55/1497.47ms (different profile starts/residency, not an isolated E2E pair). Arithmetic/copy work
  otherwise stays on the same paths; do not add dispatch sums as wall shares. Flash is now the clear next
  tested candidate at longprefix. Checkpoint D2H->D2H idle remains138.76/230.69ms; packed-state source has
  been freshly rebased without SiTU and preserves VMMlogic. Its new virtual hook/performance remain unqualified.
- 2026-11-22: opt-in OWNED16 model freshly linked onaf66, controlbyte-identical1cc5, onlyattentionobject
  replaced; all28otherGPUbundles exact. Model5346adea8764593d1f3dcf4e3810469c85dc349643393d73cf03ae993bd339bc.
  Off/on zero-resident8K+8forcedtop10 both match oldgolden. Unfrozen8/32/64K ABBA warm8K with allretained
  flags1: A1[8829.5,43023.6,106867.9], B1[8695.2,41777.3,104154.2],
  B2[8705.2,42049.0,104109.2], A2[8772.9,42777.0,106672.6]ms. Throughput**+1.16/+2.36/+2.53%**;
  bothpairspositiveeverylength, Adrift0.64/0.57/0.18%. Allguards clean, actual64K model prefill now measured,
  not200K. NextactualRoot keeps false template oldcode and true qualifiedcode; removeonlynewmisleading
  comment clause about fullymasked Kahanstate. No performance credit from a comment. No attentioncommit yet.
- 2026-11-22: CPU-only MMQ-grid accounting `mmq-grid-cost` bounds the next possible bottleneck, not a win.
  Eight repacked realgroups:60123validrows/66496nonemptytileslots,9.58%padding; >128-rowexperts6.65%padding.
  Their repackedgrids have88.44%emptyCTAs, but physical resident-run boundaries are absent, so this is NOT
  productiongrid waste. Lasttilelive1-8/9-16/17-32/33-64hist41/25/38/88 (10full64). Preparedsingle guarded
  ABBA adds64zero experts whileholding64x128 or64x512usefulrows and cross-checks outputhashes. Cost must be
  measured; oldgeneric shared-ID/barrier setup differs from retaineddown-tail's earlyreturn. No compactgrid,
  newtailbody, groupingpolicy or additionalkernel was implemented from these ratios.
- 2026-11-22: actual-root OWNED16 build79efbccc834549dca97020bf2b4f97f33229ca533ef19a8685f4ff25aef02c8f
  has falseoldkernels, qualifiedtruekernels, unchanged104BABI and28otherexactGPUbundles. Metadata-only CUID,
  69localABSresourcesymbols/ordering/emptysection differences are constructively normalized; actual shader
  text/descriptors/args/resources already match before normalization. Actual-root8K+8forcedgolden PASS.
  HTTP32ABBA warm8K, sameoriginalsettings/allretainedflags1, cached0/32805prompt/16output:
  A1[44868.286,44601.357], B1[44004.291,43802.434], B2[44065.448,43842.913],
  A2[45161.981,46194.374]ms TTFT. Mean45206.500->43928.771ms, **-2.826%latency/+2.909%throughput**,
  pairedreductions1.859/3.774%, Adrift2.109%; allfourcorrespondingcasespositive/allguardsclean. Clocks and
  placementunfixed, do not promise exactpercentage. Nextcommit/pushthenretainedprofileandPACKqualification.
- 2026-11-22: empty-grid GPU cost diagnostic completed with identical cross-arm useful-outputhashes:
  +2304emptyCTAs at64x128 changedgate-16.31us/down+0.37us (noise); +9216 at64x512 gave+52.83/+23.49us;
  +32256 at16x1024 gave+72.83/+123.61us. Oldgenericdown is NOT the retained-tail earlyreturn kernel.
  Extraallocation/cache/clocks remainconfounders; these are descriptive deltas, not calibratedperCTA costs.
  Large empty-grid ratios do NOT imply large runtimecost. No compact-grid/early-shell rewrite from this.
  Padding inthissubset is only9.58%slots (6.65%for>128experts), not evidence a new dynamic-tail kernel wins;
  priorforcedJ32 lostreuse. Preserve currentkernel instead of inventinganother specialization from ratios.
- 2026-11-22: execution-only NULL-marker skipping was checked as a distinct possible remedy for APIshorttails,
  before implementation. It does not unlock the cold8229fixture: inside considers2048/4096/6144/8192plusatmost
  one lastturn opener, endpoint8228isaddedlater, atmost6states;8192isreal/protected, notpolicydiscarded.
  NULLmarks have no savedBlob but still determinebatcharithmetic/progress/cancelpolls; retention simulation
  itself mustkeepallcandidates. SkippingNULLandbalancing cannotborrowacrossreal8192without changingcache
  semantics or historicalstatecapture. No code changed or speedclaimed; oldread-balance remainsNOOP here.
- 2026-11-22: OWNED16 committed/pushed as**172e3620**. This is the next baseline, not a pending attention
  experiment. Actualbuild79ef is unchanged after the commit; no tag-only rebuild was attributed to a gain.
  Run its retained profile before PACK's full consistent Engine/derived-header build. PACK remains an opt-in
  hypothesis: byte-format evidence from the earlier nonvirtual prototype does not qualify this new wiring
  or establish performance. Preserve actual-source base serializer and test fallback/short calls separately.
- 2026-11-22: retained attentionprofile `attention-retained-trace` measured8/32K: span8779.98/42479.14ms,
  kernelunion8347.43/40843.29, overlap2091.39/8410.11, NVFP43894.97/15585.27,
  Flash1386.57/12948.04, F32small207.56/837.96. PriorF32profile Flash1456.31/13602.86: about4.8%lower
  dispatchtime, consistent with microbench; separateprofiles/notisolatedwallpairs. Checkpoint D2H->D2H
  idle136.99/222.93ms remains, so PACK is the remaining concrete candidate. Guardclean.
- 2026-11-22: common-input MMQ packing headroom measured from that retainedprofile, not inferred from all
  quantize time. At8K: dense2304 960calls/56.365ms, O3072 192/16.828, routedgate192/63.976,
  down1152 380/44.964; total182.132ms. At32K:3840/228.529,768/71.213,768/261.083,1504/183.772.
  Widthclassification uses actualblock128 andgrid=(rows,ceil(paddedwidth/512),1): CSVX/128isrows,Y3/5/6
  is1152/2304/3072 input. The templatefalse meansscatter=false, NOTids-null; no classesleftunknown.
  Even deletingALLdensequant includingnonreusableO givesonly73.193ms/0.834%span at8K and299.742ms/0.706%
  at32K. Reusingonlyidenticalinput+layout(Q/rel/gate, K/Vwhenone-stream, notO), keepingthefastestexisting
  packpergroup, hasoptimisticdispatchbudget**32.848ms/0.374%** and**132.981ms/0.313%**. Thisignoresnewreuse
  overhead/overlap, not a wall-speedup. Defercache/API/reorderingcomplexity below observedE2Enoise; earlier
  SiTU'slargerlocal savingfailedrepeatableactualqualification. No reusefeaturewasimplemented.
- 2026-11-22: provisionalactualPACK wiring is built, notpromoted: pure09304bb12aa01446d44892263ef7af837cb16dfb1b9cd834c6e7e8ef4ead00ae,
  separateDIAG4866f5d97d7acf1fdc0a0e209f00b41c7192936f43192d52231df3f92e5db519. Fourreviewedfilesonly;
  protectedvirtualdefaultisoldphysical-loop/save_last. All10headerusers recompiled,9vtables384->392,
  DIAGnewABIconsistent; purecontrolrelinkexact. No VERIFYflag/vectorinproduction.
  RawfatbinsareNOTequal: onlyInklingCUIDnames/loaderstrings/hash tables(diagnosticalsosymbolordering)
  differ. All29modules' code/constants/data/kernelnotes/resources/programheaders match; dynamic symbols
  matchaftername/ordernormalization. Reviewedmetadataexception, not a falsifiedraw-byte-equalityclaim.
  DIAGfullmemcmp runsinsideprivateFASTsuccessafterbulkD2H+sync, usingqualifiedactualbase serializer;
  fallbackcannotemitPASS. GPUbyte/wrap/F16/golden/fallbackqualificationrunning. No performanceorproduction
  deploymentclaim untilactual-rootE2E. PrivatewrapprofileoverridewasmovedAFTERdiagnosticargs toisolateit.
- 2026-11-22: actualPACK qualification passed (`guard-pack-production-qualification`, nofault): Q8full
  92875776Bmemcmp at93/2048/4096/6144/8192, FP16full171442176B at93/2048; allrecordsfast=1,
  stagecapacity700710912B. NewconsistentDIAG wrap atpivot4096, zero-resident/cpu-share0 validationonly,
  alsofast-verified6/4104/4199 andrestored/fullsixIDs[152717,13,198,33306,361,220]. Pure8K+8forcedtop10
  matchesoldgolden withallretainedflags1/PACK1. Purestage-mib32 fallsback(noactivation),94/2049requestshealthy.
  PurehasnoVERIFYcode; diagnosticresultsarenotperformancetimings. Actualsame-pure-binary unfrozenABBA
  94/2048/8192/32768, repeat2/warm8K, is next tocheckshortregressionsandrealgain beforeanycommit.
- 2026-11-22: PACKpure CPPABBA completed94/2K/8K/32K repeat2: meanthroughput**+.039/+.073/-.323/+.132%**,
  everylengthhasopposite-signpairs, so NOrepeatableCPPwin. Allguards clean; raw `pack-e2e-production-*`.
  Diagnostictrace0/1 warm8->8K gives span8771.04->8654.99ms, idle349.38->256.01, kernel8327.56->8309.55,
  overlap2084.65->2081.99. D2H->D2H idle143.81->57.22ms, newcopyBuffer->D2H38.26ms. That supports a
  copy-gap mechanism UNDERTHEPROFILER but doesnotqualifyproduction; oldmanyAPIcallbacksmayamplifyit.
  No newpinned/prefault/allocationfeaturefromthisguess. Spill'sfirstheap isnewnothrowuint8[], notzero-filled.
- 2026-11-22: PACKactualAPI8K ABBA, warm+2fresh8229prompt/16output,cached0: A1[10018.068,10018.068],
  B1[9892.468,9935.728], B2[9872.629,9967.431], A2[10442.234,10344.820]ms. Mean10205.797->9917.064,
  -2.829%latency, pairedreductions1.038/4.556%, Adrift3.748%; allfourcorrespondingcasespositive/guardsclean.
  Actualchatfixturehasmorecheckpoints thanbareCPP; native-tail/cachevarianceandAdrift makeexactmeanunsafe.
  Beforepromotion, prepareplainCPUserializerwalltimer(noROCprof, privateoneC++object/newconsistentABI),
  optionalrusagefaultdeltas, aroundonlyactualsave_card_stateafterinitialsync. Distinguishrealcopycostsaving
  fromprofiler-distortion instead of declaring the profiled116ms an unprofiled win. Pure source unchanged.
- 2026-11-22: plainserializer timer(noROCprof, privatehostobjectbb7fc199...; allGPUfatbinexactpure093)
  warm-excluded8K's4snapshots: old47.953ms/call vsPACK27.817ms/call, about80.54mssaved/request.
  Everyactualmeasured93Mimage alsoincurs22674minor faults/0major; thismatchespagecount butprocess-wide
  countersalonecannotattributeeachfault. Copycostimprovementisreal, notjustROCprofAPIcallbacks.
  ReverseAPI B3/A3/A4/B4: B3[9923.104,9937.717], A3[9998.250,9993.104],
  A4[10042.956,10025.048], B4[9905.107,9892.697]ms. Meanold10014.840->new9914.656ms,
  **-1.000%TTFT**, pairs0.653/1.346%, Adrift0.383%; allfourcorrespondingcasespositive/guardsclean.
  RetainPACKasAPIcheckpointwin, not a bare-engine throughputclaim. Nextcommit/pushthenbaseline.
  Remaining concrete hypothesis: per-current-output firsttouch in existingCPUworkers could parallelize
  thefaulting nowdominatingfastsave. Source-onlyprep; no newpin/allocation/thread/queue orprefaulted6Blob
  upfrontRAMincrease. TestsameolddataformatandactualE2E beforeadopting; pure093 currentlyunchanged.
- 2026-11-22: PACK committed/pushedas**c9bb85d9**, nextbaseline093pure. Existingpack-trace1 isalreadythe
  exactretainedcode/all6flags profile; commit/docs didnotrequireanunchanged rerun. No byte/timerinstrumentation
  shipped. Source-only firsttouch extension uses onlycurrentfuture-outputBlob; Spillallocatesnewheap and
  inside() allocatesnewuncommitteddue states. ExistingWorkers.run returnsafteralljobs and supportsnestedcalls;
  threads() canidentifyserialpool. CandidatewriteszeroonlybeforethealreadycompleteD2Hoverwrite, no unread
  byte loads, 4KiBsteps/2MiBjobspluslastbyte coverunalignedend. Source/availability/indexproof andlocalCPU
  timing precede anyadoption; no newworker/allocation/pin/queue or up-front6-stateRAMpolicy permitted.
- 2026-11-22: privatepage-touch flavors built outsideGit. PlainTIMER6fcb44a8... includesdispatch/faulting/join
  andsave: warm-excluded8K4calls mean27.433->13.546ms, same22674minor/0majorfaults; shortpos93
  27.246->11.960ms. Faultsarenotexcluded/reduced, workparallelized. ActualDIAG52c647ae... fullQ8/FP16
  comparison/controlledSWAwrap+restore andPURE2979b765... 8Kforcedgolden passed; guardsclean.
  APIandshort94/2K ABBA nowtestactualEE. Privateprefillfixturemaxsize nowincludeswarmup, fixingthe
  short-onlysweep'swarm8Kcapacity; earlierdatasetsallhadmax(lengths)>=warmup andarenotretimed/changed.
  NoRootcodechangeoradditionalpermanentflag. Prototypeonly; local2xsnapshotwinisnotanEEclaim.
- 2026-11-22: R2safety-gateblocker: page-touchAPI B2interruptedbyGuardmonitor/final-check `.lower()`onlist,
  notamdgpu fault. Two desktopVLCflatpak MESSAGErecordsare120-bytebinaryarrays; fullsavedkernel/desktop
  rereadwithUTF8replacementdecodedthem, nofault/reset/timeout/OOM/kwinmemoryevent, minfree1969.99MiB,
  noengineleftalive. A1/B1finishedbutthisABBAisnotqualified/spliced. Do notstoptheuser'sVLC.
  MinimalenvelopeadditionforR2/R3: `rdna2-rnd/scripts/run_guarded.py` mustdecodejournald'sknownbyte-array
  textform for MESSAGE/source fields, keepunknownformsfatal. Rawjournals/faultpatterns/floor/termination
  unchanged. CPUreplay+invalidtypechecks andguarded`true`verify beforefreshABBAunderexistingdesktopload.
- 2026-11-22: guardfix committed/pushed**6e52c12e**; inferencebinary093unchanged. Completefreshpage-touch
  v2ABBA passedallguards; nointerruptedB2datareused. API8229mean9888.078->9856.930ms(-0.315%),
  pairs0.543/0.089%, Adrift0.695%: inconclusive. Bare94mean462.125->450.475ms(-2.521%),
  pairs2.047/2.999%, Adrift-.830%;2Kmeanexactlyflat/pairsopposite. Warm8192actuallyread8192inallshort
  controls afterfixturefix. Do not extrapolate shortgain toalllengths; desktopVLCremaineduntouched.
- 2026-11-22: shortactualHTTPproto confirmsappropriateworkload: 131prompttokens(body94+template),16output,
  cached0, warm+2freshperprocess, sameoriginalsettings/all6baseflags, nofreeze. A1[1134.332,1168.727],
  B1[1119.171,1124.912], B2[1094.029,1108.659], A2[1136.210,1156.303]ms.
  Mean1148.893->1111.693ms(**-3.238%TTFT**), pairs2.561/3.918%, Adrift-.458%, allfourcasespositive,
  allguards clean. Proceedtoactual-root13lineadoptioninsidePACK1/threads>1; no newpermanentTOUCHflag.
  Build/native/actualshortAPIqualification pending. DefaultPACK0/fallback/serialpoolstayunchanged.
- 2026-11-22: actual13lineRootadoption**ebab4fad...** qualifies, oldPURE093vsnewPUREbothPACK1;
  noTOUCHflag/timer/verificationinproduction. WholeGPUfatbin/all29modulesrawbyte-identical093,
  other22linkinputs/vtables392unchanged. ActualDIAGfullQ8pos93/2048/4096/6144/8192andFP16pos93/2048
  pass; PURE8Kforcedgoldenmatches. ShortHTTP131+16cached0 ABBA:
  A1[1156.340,1161.189], B1[1107.027,1101.083], B2[1102.043,1116.330], A2[1127.800,1150.831]ms.
  Mean**1149.040->1106.621ms(-3.692%TTFT)**, pairs4.721/2.644%, Adrift-1.678%; allfourcasespositive,
  allguards clean. Keepconservativeworkload-specificclaim; long8Kextraeffectinconclusive, 2Kflat.
  Retainthislastsmallwin, then closehypothesismap and runfinaloriginal-vs-final/API/deployment. No new
  investigationbranchfromremaininglow-shareceilings; globaloptimum,200Kthroughputandotherhardwareuntested.

## Completion

R1 and R4 verified; R2 and R3 remain open. Final qualified commits, deployment and remaining limitations will be recorded here.
