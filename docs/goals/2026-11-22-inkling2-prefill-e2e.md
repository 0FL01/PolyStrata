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
- No unrelated refactoring, services or dependencies. Scratch artifacts stay outside Git.

## Change envelope

- `src/poly/engine.*`, `inkling2.*`, `inkling_attend.*`, `mmq*`, local kernels and their declarations/direct callers;
  CMake/tests only when a retained change requires a target or a concrete regression check. Docs for measured settings.
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
- Next: promote down-tail as the second baseline; reprofile it. Compare the surviving lane-owned attention and
  SiTU->MMQ fusion on the model. Replace the default-no-op read rebalance with a same-format checkpoint-copy
  packing experiment; do not disable prompt checkpoints merely to make the benchmark improve.
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

## Completion

Pending R1-R3; final qualified commits, deployment and remaining limitations will be recorded here.
