# GOAL: Architecture-aware Inkling 2 Mini Decode Optimization

## 1. Objective

Изучи актуальные коммиты PolyStrata, исходный код движка и историю экспериментов в `docs/goals/`.

Твоя задача — найти и практически доказать способы существенного увеличения decode tokens/s на текущем железе за счёт архитектурных особенностей Inkling 2 Mini.

Основные направления исследования:

- P0 — Loop-aware Critical Path Expert Placement.
- P1 — Cross-pass Route Prediction.
- P2 — Cross-pass Lossless KV Compression.
- P3 — Loop Message Hoisting.

Основной профит исследования должен получаться из практического RECON: низкоуровневого профилирования, анализа реальных данных, временных скриптов, симуляторов, микробенчмарков и экспериментальных реализаций.

Работай итеративно: RECON → hypothesis → measurement → reflection → refined hypothesis → prototype → verification → E2E benchmark.

Используй массово `@general` для параллельного исследования независимых гипотез, прототипирования, анализа результатов и независимого review.

Рассматривай предложенные направления как начальную карту гипотез. Актуализируй её по мере получения новых экспериментальных данных.

Целевой ориентир исследования — найти совокупность оптимизаций, способных дать существенный E2E прирост, желательно от 25%, при сохранении точности ответов модели.

## 2. Hardware & Production Context

Актуализируй состояние оборудования, движка, production-конфигурации и baseline непосредственно перед началом RECON.

Исходный контекст:

- AMD Radeon RX 6700 XT, gfx1031, RDNA2.
- 12 GiB VRAM.
- 96 MiB Infinity Cache.
- AMD Ryzen 7 5700X.
- 32 GiB DDR4 RAM.
- HIP/ROCm.
- Fedora Linux, KDE Plasma, GPU обслуживает desktop.
- Inkling 2 Mini Q4_0 /home/stfu/.lmstudio/models/unsloth/Inkling-2-Mini/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf
- Q8_0 K/V.
- 200K context capacity.
- 2048 MiB VRAM reserve.
- 8 CPU workers.
- Single-sequence inference.
- Serial decode, текущий production path.

Исторический baseline после Q4_0 оптимизаций:

- 2K: 51.325 t/s.
- 4K: 49.325 t/s.
- 16K: 47.990 t/s.
- 64K: 38.140 t/s.

Используй эти числа для первоначального sizing. Зафиксируй свежий baseline на текущем HEAD и фактической production-конфигурации.

Основная архитектура:

- 24 физических слоя.
- 48 логических visits.
- Prelude: слои 0–5.
- Core: слои 6–17, три прохода.
- Coda: слои 18–23.
- 128 experts на физический слой.
- Top-8 routing.
- 2304 hidden size.
- 1152 expert intermediate size.
- 40 sliding-window attention visits.
- 8 full-context attention visits.
- SWA window 1024.
- Независимые KV и convolution states для каждого логического visit.
- Общие физические веса между core passes.
- Previous-pass message banks.
- Rank-64 loop controllers.

Фокус исследования — эффективное использование этой геометрии при сохранении вычислительной семантики текущей модели.

## 3. Accuracy Contract

Сохранение точности ответов модели является обязательным условием принятия каждого оптимизационного изменения.

Для каждой гипотезы определи точную numerical boundary, относительно которой проверяется эквивалентность.

Сохраняй:

- Model weights и quantization.
- Формулы вычислений.
- Порядок FP32 accumulation и reduction.
- Операции FMA и промежуточную точность.
- Семантику quantization.
- Router top-8 expert IDs, порядок и веса.
- Семантику expert aggregation.
- Sampling distribution и RNG consumption.
- KV и convolution state.
- Семантику checkpoint, rollback, restore и prefix reuse.
- Текущую модель, число core passes и context capacity.

Используй следующие уровни проверки.

**Q0 — Leaf parity**

Побайтовое сравнение результатов baseline и candidate на реальных весах и активациях.

Проверяй применимые FP32 buffers, quantized activation bytes, partial reductions, attention outputs и packed Q8_0/Q8_1.

**Q1 — Operator chain parity**

Проверка полной изменённой вычислительной цепочки.

Для MoE:

`activation → quantization → gate/up → SiLU → intermediate quantization → down → weighted expert sum`

Для attention:

`Q/K/V → cache representation → attention partials → combine → output quantization`

Для loop controllers:

`previous-pass banks → messages → normalization → rank projection → gain/forget → controlled output`

**Q2 — Engine parity**

Сравнивай:

- Forced-token logits.
- Raw greedy token IDs.
- Полные FP32 logits, используемые sampling.
- Ordered routing records.
- Router weight bits.
- Committed logical state.
- KV и convolution state.

Forced-token и free-running greedy проверки выполняй как самостоятельные эксперименты.

**Q3 — Sampling parity**

Сравнивай fixed-seed sampling при одинаковых sampler settings, исходном состоянии и воспроизводимой numerical trajectory.

**Q4 — Stateful inference parity**

Проверяй длинный контекст, SWA wrap, checkpoint, rollback, disk save/restore, продолжение сессии после restart, cached prefix reuse.

Для оптимизированного внутреннего представления состояния сравнивай его декодированное каноническое представление.

**Q5 — Agent workload validation**

Проверяй реальные английские agent workloads: reasoning, code generation, debugging, JSON, tool calls и tool-result replay.

Используй существующие quality batteries из истории проекта.

Ожидаемый результат каждого применимого уровня — PASS.

При candidate-specific divergence локализуй первую отличающуюся numerical boundary, сформулируй причину, подготовь исправленный прототип и повтори проверки.

Учитывай историческую особенность PolyStrata: CPU и GPU expert arithmetic могут различаться, а live expert swaps способны создавать A/A divergence.

Для экспериментов с placement используй детерминированные replay fixtures, matched-placement controls, shadow reference placement и отдельно live A/A.

Различай физическое место исполнения эксперта и его reference numerical semantics.

Статус PASS требует доказательства эквивалентности изменённой вычислительной операции и соответствующих engine-level проверок. Наблюдаемую A/A вариативность документируй отдельно.

## 4. Authorized Execution Envelope

Разрешено самостоятельно выполнять следующие действия в рамках исследования:

- Читать репозиторий PolyStrata, историю Git, документацию, исходники и доступные локальные RECON-артефакты.
- Читать актуальную конфигурацию оборудования, ROCm/HIP, RAM, VRAM и локального inference service.
- Использовать `@general` для делегирования research, implementation, benchmarking, numerical analysis и review.
- Создавать изолированные Git worktrees для экспериментальных реализаций.
- Создавать временные C++, HIP, Python и shell-прототипы.
- Создавать CPU simulators, trace analyzers, benchmark harnesses и parity fixtures.
- Использовать существующие GGUF weights для низкоуровневых тестов.
- Читать существующие session snapshots и создавать отдельные исследовательские fixtures.
- Собирать HIP Release binaries и релевантные тесты.
- Использовать HIP events, ROCm API/kernel tracing, CPU profiling, дизассемблирование и анализ ISA.
- Выполнять standalone GPU benchmarks через актуальный RDNA2 guard.
- Управлять исследовательскими процессами и их временными файлами.
- При необходимости согласованно освобождать GPU от idle inference service, сохраняя исходную конфигурацию и восстанавливая service после измерений.
- Использовать SSH nc-lab для дополнительного внешнего RECON, изучения документации, исходников и metadata.
- Сохранять подтверждённые экспериментальные патчи, evidence manifests и отчёты в исследовательской рабочей директории.
- Выполнять cleanup созданных временных ресурсов после сохранения необходимых evidence artifacts.

GPU execution envelope наследуй из актуальной документации проекта.

Используй существующий guard, один GPU benchmark одновременно, актуальный VRAM safety threshold, production reserve 2048 MiB и подтверждённые параметры HIP runtime.

При guard stop reason зафиксируй состояние эксперимента и обработай его как отдельный checkpoint.

Каждому сабагенту явно передавай разрешённые действия, paths, commands и tools, необходимые для его конкретного Context Pack.

## 5. Context Pack Orchestration Protocol

Каждый Context Pack — атомарная исследовательская задача с собственной гипотезой, входами, разрешениями и измеримым результатом.

Формируй задания для `@general` по следующему контракту.

### Context Pack Input Port

- `PACK_ID`
- `PARENT_HYPOTHESIS`
- `OBJECTIVE`
- `BASELINE_COMMIT`
- `RELEVANT_SOURCE_FILES`
- `RELEVANT_PRIOR_EXPERIMENTS`
- `INPUT_ARTIFACTS`
- `AUTHORIZED_ACTIONS`
- `EXPERIMENT_MATRIX`
- `ACCURACY_GATE`
- `PERFORMANCE_GATE`
- `OUTPUT_ARTIFACTS`

Передавай сабагенту минимальный достаточный контекст для конкретной задачи.

### Context Pack Output Port

Каждый сабагент возвращает:

- `STATUS`: PASS / FAIL / INCONCLUSIVE / BLOCKED.
- `HYPOTHESIS`.
- `METHOD`.
- `REPRODUCTION_COMMANDS`.
- `MEASUREMENTS`.
- `BASELINE_COMPARISON`.
- `FIRST_DIVERGENCE`, если обнаружена.
- `ACCURACY_EVIDENCE`.
- `MEASURED_UPPER_BOUND`.
- `ARTIFACT_PATHS`.
- `INTERPRETATION`.
- `NEXT_ATOMIC_EXPERIMENT`.

Создай общий research manifest, через который оркестратор связывает результаты Context Packs.

Оркестратор управляет очередью GPU-экспериментов, разрешает зависимости, сопоставляет результаты сабагентов и принимает решения о следующей итерации.

CPU-only RECON, анализ документации, подготовку прототипов, обработку завершённых трасс и review выполняй параллельно.

GPU-измерения выполняй последовательно.

---

# PHASE 0 — Baseline & Research Infrastructure

## CP-00 — Historical Research Reconstruction

Цель: восстановить актуальное состояние R&D и карту уже проверенных гипотез.

Разрешения: чтение Git, исходного кода, документации и локальных research artifacts; создание исследовательского отчёта.

Особое внимание:

- `docs/goals/2026-11-21-inkling2-mini-rdna2.md`
- `docs/goals/2026-11-21-inkling2-decode-stalls.md`
- `docs/goals/2026-11-23-inkling2-decode-throughput.md`
- `docs/goals/2026-11-23-inkling2-decode-launch-bound.md`
- `docs/goals/2026-11-23-inkling2-next-levers.md`
- `docs/goals/2026-11-24-inkling2-q4-0-decode.md`
- `docs/goals/2026-11-24-inkling2-decode-implementation.md`
- `docs/goals/2026-11-24-inkling2-dspark-exact-throughput.md`
- Более свежие цели и коммиты.

Для каждого исторического эксперимента извлеки:

- Измеряемый bottleneck.
- Проверенную гипотезу.
- Конкретную реализацию.
- Изменение kernel latency.
- Изменение E2E throughput.
- Numerical parity.
- Причину принятия или отклонения.
- Условия, при которых гипотеза становится актуальной повторно.

Output: `historical-experiment-map.md`.

## CP-01 — Fresh Baseline

Цель: получить актуальный воспроизводимый baseline для всей исследовательской кампании.

Разрешения: сборка текущего HEAD, чтение production settings, guarded engine benchmarks, создание trace fixtures.

Измерь:

- 2K, 4K, 16K, 64K.
- 128K и 200K для long-context experiments.
- Decode t/s.
- Per-token wall time: median, p10, p90, p99.
- GPU busy time.
- GPU idle gaps.
- CPU expert join.
- Expert cache occupancy и hit rate.
- CPU expert counts per logical visit.
- PCIe transfers.
- Q4_0 expert kernels.
- Dense projections.
- Attention.
- Loop controller kernels.
- LM head.
- Свободную VRAM и изменения expert cache capacity.

Используй representative prose, code, reasoning и tool-call fixtures.

Раздели инструментированные трассы и untraced E2E benchmarks.

Собери A/A controls для оценки вариативности live placement.

Output: `baseline.json`, `baseline.md`, `trace-manifest.json`, воспроизводимые fixtures.

## CP-02 — Logical Visit Instrumentation

Цель: создать общий trace format для всех последующих архитектурных исследований.

Разрешения: чтение и экспериментальная модификация instrumentation движка в отдельном worktree, сборка и guarded validation.

Для каждого логического visit собери:

- Token position.
- Logical visit ID.
- Physical layer ID.
- Core pass ID.
- Selected expert IDs.
- Ordered router weights.
- CPU/GPU placement.
- Resident slot state.
- Pending swaps.
- CPU gate/up span.
- CPU down span.
- GPU expert kernel span.
- Router handoff.
- Exposed CPU join.
- Actual visit duration.

Дополнительно фиксируй события `cover()`, `shrink_cache()`, `follow()` и завершения expert uploads.

Создай compact trace format, позволяющий анализировать продолжительные decode trajectories на CPU.

Проверь overhead instrumentation на отдельном контрольном запуске.

Output: trace schema, instrumentation patch, trace collector, sample traces.

---

# PHASE 1 — P0: Loop-aware Critical Path Expert Placement

## P0 Objective

Оптимизировать размещение экспертов относительно стоимости критического пути, используя повторное посещение физических core-слоёв.

Измерять основной эффект через wall-clock latency каждого логического visit и суммарный decode t/s.

Исходная гипотеза:

Одинаковый expert cache hit rate может создавать существенно различающуюся latency из-за разного распределения CPU misses между visits.

Текущий `use_weight` уже учитывает повышенную стоимость once-visited layers. Исследуй более точную cost-aware модель для текущего Q4_0 baseline.

Relevant paths:

- `src/poly/engine.cpp`
- `src/poly/engine.hpp`
- `src/poly/inkling2.cu`
- `src/poly/cpu_experts.cpp`

Relevant symbols:

- `Engine::moe()`
- `Engine::follow()`
- `Engine::shrink_cache()`
- `Engine::cover()`
- `use_w_`
- `recent_`
- `L.slot`
- `Inkling2::run_visits()`

## CP-P0-01 — Per-visit Cost Surface

Цель: измерить реальную стоимость распределения восьми экспертов между CPU и GPU.

Разрешения: CPU/GPU microbenchmarks на реальных GGUF weights, temporary harnesses, guarded HIP tests.

Построй cost surface для `n_cpu = 0..8`, где `n_gpu = 8 - n_cpu`.

Измеряй одновременно выполняющиеся CPU и GPU branches.

Сравни:

- Prelude.
- Core pass 1.
- Core pass 2.
- Core pass 3.
- Coda.
- Разные физические слои и типы матриц.
- Hot и cold weights.
- Разные состояния CPU L3 и Infinity Cache.

Оцени:

`T_visit ≈ max(T_cpu, T_gpu) + T_handoff + T_sync`

Откалибруй эту модель по реальным engine traces, учитывая фактическое перекрытие фаз.

Output: `visit-cost-surface.json`, latency model, prediction error и критические области CPU miss count.

## CP-P0-02 — Trace-driven Placement Simulator

Depends on: CP-02, CP-P0-01.

Цель: определить достижимый потенциал cache policy посредством CPU-only simulation.

Разрешения: чтение трасс, создание Python/C++ simulator, анализ результатов.

Реализуй replay текущей cache policy с учётом:

- Physical layer ownership.
- Expert slot sizes.
- VRAM capacity.
- `use_weight`.
- Recency.
- Swap budget.
- Upload completion.
- Context-driven cache shrinking.
- Трёх visits каждого физического core-слоя.

Откалибруй simulator по фактическому hit rate, placement transitions и CPU joins из baseline.

Сравни следующие политики:

1. Current policy.
2. Frequency-based placement.
3. Static frequency oracle.
4. Cost-aware static placement.
5. Future-aware dynamic oracle.
6. Cost-aware online placement.

Для oracle выделяй optimistic upper bound и hardware-feasible estimate.

Разрешай oracle использовать future routing knowledge как диагностический инструмент.

Используй actual byte budget и измеренные transfer costs в hardware-feasible модели.

Output:

- `placement-simulator`.
- `placement-oracle.json`.
- Потенциальная экономия ms/token.
- Predicted E2E throughput.
- Распределение выигрыша по logical visits.
- Чувствительность к VRAM capacity.

## CP-P0-03 — Marginal Slot Utility

Depends on: CP-P0-02.

Цель: определить наиболее ценные expert slots с точки зрения критического пути.

Разрешения: CPU-only analysis, симуляторы, оптимизационные алгоритмы.

Оцени utility каждого эксперта:

`U(e) = marginal_saved_critical_path_time(e) / resident_bytes(e)`

Учитывай:

- Logical pass.
- Physical expert reuse.
- Co-occurrence других experts.
- Нелинейную стоимость нескольких CPU misses в одном visit.
- Изменение полезности эксперта при изменении размещения соседних experts.
- Стоимость загрузки и вытеснения.

Исследуй greedy marginal allocation, local search и trace-driven cost weighting.

Раздели trace segments на training и held-out evaluation.

Output: `slot-utility-analysis.json`, proposed allocation algorithm.

## CP-P0-04 — Placement-independent Numerical Semantics

Цель: проверить возможность изменения physical expert placement при сохранении reference arithmetic.

Разрешения: создание CPU/HIP parity harnesses, чтение и экспериментальная модификация CPU/GPU expert paths, guarded GPU tests.

Начни с одного реального Q4_0 expert.

Сравни:

- CPU activation quantization.
- GPU activation quantization.
- Gate/up FP32 outputs.
- SiLU outputs.
- Intermediate quantization.
- Down outputs.
- Weighted aggregation.

Локализуй первую численную разницу.

Исследуй:

- Воспроизведение reference CPU arithmetic на GPU.
- Воспроизведение reference GPU arithmetic на CPU.
- Разделение physical execution backend и numerical semantics.
- Shadow placement, определяющий reference numerical path.

Измерь стоимость воспроизведения каждого numerical path.

Output: first-divergence report, exactness prototype, measured overhead, feasibility decision.

## CP-P0-05 — Cost-aware Placement Prototype

Depends on: CP-P0-02, CP-P0-03, CP-P0-04.

Цель: реализовать минимальный opt-in cost-aware placement candidate.

Разрешения: isolated implementation, Release build, parity tests, guarded engine benchmarks.

Выбери алгоритм на основании simulator evidence.

Для выполнения candidate используй доказанный numerical contract.

Проведи staged validation:

- Matched routing replay.
- Placement transition replay.
- Frozen fixtures.
- Live swaps.
- Growing KV.
- Long-context restore.
- Greedy и fixed-seed generation.

Initial research gate: выявить потенциал порядка 2–3 ms/token на длинном контексте либо иной воспроизводимый эффект, оправдывающий стоимость интеграции.

Final gate: Q0–Q5 по применимости и воспроизводимый untraced E2E gain.

Output: experimental patch, parity report, benchmark matrix, implementation decision.

---

# PHASE 2 — P1: Cross-pass Route Prediction

## P1 Objective

Использовать маршруты предыдущих core passes текущего токена для предсказания дальнейших CPU expert misses.

Inkling повторно посещает один физический слой через 12 logical visits. Это создаёт потенциальное окно для подготовки весов.

Фактический router определяет expert selection и weights.

Prediction используется как сигнал для подготовки данных, scheduling и residency planning.

## CP-P1-01 — Cross-pass Routing Correlation

Depends on: CP-02.

Цель: измерить реальную корреляцию expert selection между core passes.

Разрешения: CPU-only trace analysis, статистические скрипты.

Для каждого физического core-слоя вычисли:

- Pass 1 → pass 2 overlap.
- Pass 2 → pass 3 overlap.
- Pass 1 → pass 3 overlap.
- Top-8 intersection.
- Jaccard similarity.
- Expert transition frequencies.
- Conditional probabilities.
- Route stability across tokens.
- Распределение по expert rank.
- Корреляцию CPU misses между проходами.

Раздели результаты по context length и workload.

Особенно исследуй условную вероятность:

`P(expert required at pass p+1 | routes at pass p, physical layer, current token)`

Отдельно оцени предсказуемость экспертов, фактически находящихся в RAM.

Output: `cross-pass-correlation.json`, per-layer heatmaps, correlation report.

## CP-P1-02 — Lightweight Predictors

Depends on: CP-P1-01.

Цель: определить точность и стоимость доступных предикторов.

Разрешения: создание CPU-only predictors и trace replay harnesses.

Сравни:

- Global expert frequency.
- Per-layer frequency.
- Previous-token routes.
- Same-token previous-pass routes.
- Conditional transition matrix.
- Pass-aware recency.
- Комбинацию previous-token и previous-pass signals.
- Predictor с учётом текущего residency.

Используй held-out traces.

Измеряй:

- Precision.
- Recall.
- CPU miss recall.
- Useful prediction rate.
- False-positive bytes.
- Predicted expert availability deadline.
- Predictor overhead.

Output: `predictor-evaluation.json`, наиболее перспективные predictors.

## CP-P1-03 — Lookahead & Bandwidth Budget

Depends on: CP-P1-02, CP-P0-01.

Цель: определить физическую реализуемость подготовки предсказанных weights.

Разрешения: CPU memory benchmarks, HIP transfer benchmarks, guarded concurrency measurements.

Измерь:

- Lead time между посещениями физического слоя.
- Effective PCIe H2D bandwidth.
- DMA setup cost.
- CPU DRAM bandwidth.
- CPU L3 working-set behavior.
- GPU Infinity Cache working-set behavior.
- Влияние transfers на concurrent GPU compute.
- Влияние prefetch на CPU expert bandwidth.

Историческая отправная точка:

CPU Q4_0 experts достигают около 44.7 GB/s против 46 GB/s при чтении того же payload.

Исследуй полезность каждого предсказания через фактическое завершение подготовки данных к deadline.

Output: `lookahead-budget.json`, feasible prefetch envelope.

## CP-P1-04 — Predictive Prefetch Simulator

Depends on: CP-P1-03.

Цель: измерить верхнюю границу эффекта prediction с реальными transfer costs.

Разрешения: CPU simulator, trace replay, сравнительный анализ.

Сравни:

- Existing cache policy.
- Predicted CPU-side prefetch.
- Predicted residency promotion.
- Bounded GPU staging.
- Perfect future-aware prefetch oracle.

Учитывай физический byte budget, staging capacity, сроки завершения transfers и interference.

Output: predicted useful bytes/token, wasted bytes/token, exposed join reduction, feasible latency savings.

## CP-P1-05 — Predictive Scheduling Prototype

Depends on: CP-P1-04, CP-P0-04.

Цель: реализовать наиболее перспективную архитектуру предиктивной подготовки весов.

Разрешения: isolated implementation, CPU/GPU staging prototypes, Release build, guarded A/B, exactness verification.

Используй:

- Ограниченный prefetch budget.
- Logical pass-aware prediction.
- Actual router verification.
- Reference numerical semantics.
- Измеренную стоимость transfer и residency changes.

Проверь применение prediction на live workload, включая изменяющиеся routing distributions.

Output: prediction patch, accuracy report, observed useful/wasted prefetch, E2E benchmark matrix.

---

# PHASE 3 — P2: Cross-pass Lossless KV Compression

## P2 Objective

Исследовать lossless compression между KV разных core passes одного физического слоя.

Цель — уменьшить реальный VRAM footprint Q8_0 KV и высвободить capacity для expert slots.

Сохраняй точные Q8_0 codes и FP16 scale bits каждого логического visit.

Relevant paths:

- `src/poly/inkling2.cu`
- `src/poly/inkling2.hpp`
- `src/poly/inkling_attend.cu`
- `src/poly/disk_sessions.cpp`
- `src/poly/engine.cpp`

Full-context logical visit groups:

- Independent: 5.
- Physical layer 11: visits 11, 23, 35.
- Physical layer 17: visits 17, 29, 41.
- Independent: 47.

Проверь актуальность mapping по текущему исходному коду.

Размер Q8_0 KV одного visit на token:

`2 × 8 × 128 × 34/32 = 2176 bytes`

При 200K восемь full-context caches занимают приблизительно 3.24 GiB payload.

## CP-P2-01 — Snapshot Extraction & Integrity

Цель: извлечь реальные per-pass KV данные для offline analysis.

Разрешения: чтение session snapshots, создание parser, CPU-only extraction, создание новых guarded fixtures при необходимости.

Подготовь datasets:

- 2K.
- 16K.
- 64K.
- 128K.
- 200K.

Для prose, code и agent trajectories извлеки отдельно:

- K codes.
- V codes.
- K scale bits.
- V scale bits.
- Logical visit.
- Physical layer.
- Token position.
- KV head.
- Quantization block.

Выполни проверку snapshot checksum и независимое повторное чтение данных.

Output: `kv-dataset-manifest.json`, extraction script, integrity evidence.

## CP-P2-02 — Cross-pass Entropy Analysis

Depends on: CP-P2-01.

Цель: определить информационную избыточность между проходами.

Разрешения: CPU-only statistical analysis и temporary compression scripts.

Исследуй:

- Byte equality.
- Signed code delta.
- XOR.
- Conditional entropy.
- Bitplane entropy.
- Delta distribution.
- Zero-run lengths.
- Scale bit similarity.
- Sparsity.
- Head-specific correlations.
- Position-specific correlations.
- Различия K и V.
- Различия между проходами.

Сравни cross-pass correlation с token-adjacent correlation и независимым хранением.

Оцени сжимаемость разных reference choices:

- Pass 2 относительно pass 1.
- Pass 3 относительно pass 1.
- Pass 3 относительно pass 2.
- Adaptive reference choice.

Output: `kv-cross-pass-entropy.json`, estimated compression ceiling.

## CP-P2-03 — Lossless Codec Prototypes

Depends on: CP-P2-02.

Цель: проверить несколько способов точного восстановления Q8_0 bytes.

Разрешения: CPU codec prototypes, existing compression tooling, temporary C++/Python benchmarks.

Исследуй:

- Delta + bit packing.
- XOR + bitplane packing.
- Small delta with outliers.
- RLE для подходящих областей.
- Block-local coding.
- Tile-level adaptive representation.
- Separate scale encoding.
- Raw-block fallback.
- Lightweight dictionary coding при подтверждённой повторяемости.

В качестве диагностического entropy ceiling используй доступные general-purpose lossless codecs.

Для каждого codec учитывай полную стоимость:

- Payload.
- Block headers.
- Indices.
- Offsets.
- Alignment.
- Reference storage.
- Raw fallback.
- Append overhead.
- Metadata allocations.

Проверь побайтовый round-trip на всех datasets.

Initial feasibility gate: ориентир от 20% сокращения совокупного full-context KV footprint.

Output: `codec-comparison.json`, exactness evidence, selected layouts.

## CP-P2-04 — GPU Random-access Decompression

Depends on: CP-P2-03.

Цель: определить стоимость точного доступа к compressed KV непосредственно из attention kernels.

Разрешения: HIP codec prototypes, attention microbenchmarks, guarded GPU tests, ISA analysis.

Для каждого перспективного codec исследуй:

- Tile-level independent decoding.
- Coalesced memory access.
- Vectorized reconstruction.
- Register pressure.
- LDS usage.
- VGPR occupancy.
- Reference reads.
- Actual memory traffic.
- Reconstruction cost.
- Cold и warm attention latency.

Используй исходный FP32 attention reduction order.

Сравни baseline Q8_0 и compressed representation на 2K, 16K, 64K и 190K–200K.

Проверяй exact attention partials, output FP32 и packed Q8_1.

Output: `gpu-codec-benchmarks.json`, exactness report, measured latency cost.

## CP-P2-05 — Append, Rollback & Lifetime Model

Depends on: CP-P2-04.

Цель: определить жизненный цикл compressed cache для последовательной looped архитектуры.

Разрешения: isolated state/cache implementation, CPU state tests, guarded integration tests.

Учти порядок записи KV:

- Pass 1 создаёт reference row.
- Pass 2 записывает собственный exact representation.
- Pass 3 записывает собственный exact representation.
- Committed token state содержит все логические visits.

Исследуй:

- Append-only compressed tiles.
- Raw representation для активных незавершённых tiles.
- Reference lifetime.
- Tile finalization.
- Exact random-access reconstruction.
- Restore.
- Rewind.
- Prefix branching.
- SWA wrap для соответствующих extensions.
- Каноническую сериализацию текущего snapshot format.

Измерь реальный allocated/mapped VRAM footprint.

Output: state model, lifetime tests, codec integration candidate.

## CP-P2-06 — VRAM-to-Throughput Conversion

Depends on: CP-P2-05, CP-P0-02.

Цель: определить суммарный E2E эффект compressed KV с учётом дополнительных resident experts.

Разрешения: CPU placement simulation, isolated allocator integration, guarded E2E benchmarks.

Выполни три сравнения:

A — исходный Q8_0 cache, исходная expert capacity.

B — compressed cache, matched expert capacity.

C — compressed cache, увеличенная expert capacity в рамках прежнего VRAM reserve и exact numerical contract.

Сравнение A/B изолирует вычислительную стоимость compressed representation.

Сравнение B/C показывает пользу высвобожденной VRAM.

Сравнение A/C определяет суммарный E2E результат.

Особое внимание удели 64K, 128K и 200K.

Output: actual saved VRAM, extra resident expert slots, CPU join reduction, attention latency delta, net decode throughput, parity evidence.

---

# PHASE 4 — P3: Loop Message Hoisting

## P3 Objective

Использовать неизменность previous-pass message banks внутри очередного core pass.

Вынести вычисления, зависящие исключительно от `bank0_`, `bank1_` и immutable weights, из последовательного visit critical path в grouped computation на границе прохода.

Relevant paths:

- `src/poly/inkling2.cu`
- `src/poly/inkling2.hpp`
- `src/poly/kernels.cu`

Relevant symbols:

- `Inkling2::run_visits()`
- `Inkling2::control()`
- `Inkling2::heads()`
- `Inkling2::message_query()`
- `control_in_row()`
- `control_in_kernel()`
- `control_in3_kernel()`
- `mix_row()`
- `bank0_`
- `bank1_`
- `control_out_body()`

## CP-P3-01 — Dependency Graph & Invariance Analysis

Цель: доказать зависимости и lifetime вычислений внутри core pass.

Разрешения: чтение исходников, построение dependency graph, temporary instrumentation.

Раздели операции на:

- Previous-pass bank-only.
- Current residual-dependent.
- Current stream-dependent.
- Immutable weight-dependent.
- Position-dependent.
- State-dependent.

Проверь lifetime `bank0_` и `bank1_`, включая transitions между visits 17→18 и 29→30.

Определи точные места, где значения banks становятся доступны для grouped computation.

Исследуй Q, K, V и FFN message streams всех 12 физических core-слоёв.

Output: `loop-dependency-graph.md`, список hoistable operations, numerical boundaries.

## CP-P3-02 — Baseline Controller Profiling

Depends on: CP-P3-01.

Цель: измерить реальную стоимость message computation и loop controllers.

Разрешения: HIP microbenchmarks, source instrumentation, guarded traces.

Измерь отдельно:

- Bank loading.
- Bank RMS reductions.
- Query dot products.
- Message softmax/mixing.
- Message normalization.
- Current residual normalization.
- Rank-64 projection.
- Gain/forget projection.
- Controller output.
- Launch overhead.
- GPU occupancy.

Определи суммарное время hoistable operations per token.

Output: `controller-cost-breakdown.json`, E2E upper bound.

## CP-P3-03 — Message Precomputation Prototype

Depends on: CP-P3-02.

Цель: вычислять bank-dependent messages для всех необходимых core visits заранее.

Разрешения: experimental HIP kernels, parity harnesses, guarded benchmarks.

Исследуй следующие arms:

A — исходный per-visit message computation.

B — предварительное вычисление только bank RMS terms.

C — предварительное вычисление полного message.

D — предварительное вычисление message и его normalization.

E — grouped message computation для всех Q/K/V/FFN streams.

Начальная оценка scratch capacity:

`12 physical layers × 4 streams × 2304 × sizeof(float) = 432 KiB`

Используй точную geometry выбранного implementation.

Сохраняй исходные per-thread FMA operations, reduction tree, normalization и промежуточную точность.

Проверяй реальные bank buffers из разных tokens, contexts и core passes.

Output: exactness-tested grouped kernels, scratch footprint, measured kernel latency.

## CP-P3-04 — Numerical Scheduling

Depends on: CP-P3-03.

Цель: найти оптимальный scheduling precomputed messages относительно текущего inference stream.

Разрешения: isolated scheduling prototypes, HIP event measurements, guarded engine tests.

Сравни:

- Grouped computation сразу после обновления banks.
- Grouped computation с отдельными batches физических слоёв.
- Lazy preparation блоков.
- Prefetch precomputed messages.
- Reuse общих bank-dependent reductions между streams.

Исследуй сохранение порядка арифметики rank projection при повторном использовании её invariant components.

Оцени cache interference, register pressure, дополнительный memory traffic и изменение launch count.

Output: scheduling benchmark matrix, selected candidate.

## CP-P3-05 — Engine Integration Prototype

Depends on: CP-P3-04.

Цель: интегрировать выбранную exact message-hoisting реализацию в экспериментальный engine.

Разрешения: isolated patch, HIP Release build, relevant unit tests, guarded E2E benchmarks.

Проверь:

- Pass transitions.
- Все logical visits.
- Полную controller chain.
- Greedy token parity.
- Full FP32 logits.
- Checkpoint/restore.
- Prompt prefill.
- Serial decode.

Измерь прирост на 2K, 4K, 16K и 64K.

Output: opt-in patch, numerical parity evidence, untraced benchmark matrix.

---

# PHASE 5 — Cross-vector Synthesis

## CP-X-01 — Architecture-aware Interaction Analysis

Цель: определить взаимодействие успешных P0–P3 candidates.

Разрешения: чтение evidence, CPU simulation, combined experimental builds и guarded benchmarks.

Особенно исследуй:

- P0 + P1: prediction как дополнительный сигнал marginal slot utility.
- P0 + P2: распределение дополнительных expert slots.
- P1 + P2: изменение CPU miss prediction при выросшем residency.
- P3 + P1: изменение lookahead window и доступного времени подготовки данных.
- P3 + P0: изменение overlap CPU/GPU branches.

Измеряй каждую комбинацию относительно единого актуального baseline.

Output: interaction report и proposed combined candidate.

## CP-X-02 — Combined Candidate

Цель: измерить совокупный эффект подтверждённых оптимизаций.

Разрешения: isolated integration, Release build, exactness tests, guarded engine benchmarks.

Используй fresh baseline и интерливированные paired comparisons.

Проведи полный Accuracy Contract для затронутых boundaries.

Измерь:

- Decode throughput.
- Prefill throughput.
- TTFT.
- CPU join.
- GPU busy/idle.
- VRAM footprint.
- Expert cache behavior.
- Stateful inference.
- Real agent task completion latency.

Output: combined patch, benchmark evidence, complete parity report.

---

# PHASE 6 — Independent Review

## CP-R-01 — Numerical Correctness Review

Передай независимому `@general` подтверждённые candidate patches, reference implementation, numerical contract и parity fixtures.

Разрешения: чтение исходников и патчей, самостоятельные parity tests, временные correctness harnesses.

Задача: проверить:

- Floating-point operation order.
- CPU/GPU numerical semantics.
- Quantization.
- Expert aggregation.
- State lifetime.
- KV reconstruction.
- Cache transitions.
- Rewind/restore.
- Boundary conditions.

При обнаружении отличий верни минимальный воспроизводимый counterexample и первую отличающуюся boundary.

Output: independent correctness verdict.

## CP-R-02 — Benchmark Methodology Review

Передай отдельному `@general` raw measurements, harnesses, baseline и candidate binaries metadata.

Разрешения: CPU analysis, статистическая проверка, подготовка повторных benchmark runs.

Проверь:

- Одинаковость workloads.
- Cache placement comparability.
- Warmup.
- Thermal state.
- Clock behavior.
- Instrumentation overhead.
- A/A variability.
- Paired measurement methodology.
- Outliers.
- Reproducibility.

Output: independent performance verdict.

## CP-R-03 — Architecture & Safety Review

Передай отдельному `@general` implementation design, memory lifecycle, scheduling и allocator changes.

Разрешения: чтение и review экспериментального кода, создание focused tests.

Проверь корректность HIP execution, buffer sizing, pointer lifetime, synchronization, memory ownership, VRAM reserve и совместимость с актуальным RDNA2 execution envelope.

Output: independent architecture verdict.

---

# 7. Benchmark Protocol

Для первичного screening используй минимальные воспроизводимые эксперименты, позволяющие оценить потенциал гипотезы.

Для перспективных candidates переходи к полноценным A/B.

Используй:

- Release builds.
- Один и тот же source-pinned baseline.
- Одинаковые prompts и token trajectories.
- Общие snapshots.
- Comparable placement.
- Warmup.
- ABBA и BAAB order.
- Отдельные traced и untraced measurements.
- Повторные серии с изменённым порядком запуска.
- Контроль GPU/CPU clocks, температуры и VRAM.
- 256–512 decode tokens для стандартных E2E runs.
- Более длинные traces для cache policy и route prediction.
- Четыре и более paired comparisons для финальной квалификации перспективного candidate.
- Confidence interval или другой обоснованный способ оценить статистическую устойчивость выигрыша.

Фиксируй median, paired deltas, dispersion и raw measurements.

Используй A/A controls для определения noise floor.

Считай отдельными категориями:

- Измеренный leaf improvement.
- Измеренный operator improvement.
- Measured exposed critical-path reduction.
- Симулированный upper bound.
- Измеренный E2E improvement.

Финальная performance qualification основывается на untraced E2E gain, превышающем измеренную вариативность baseline.

# 8. Adaptive Research Loop

После завершения каждого Context Pack:

1. Проверь полноту evidence.
2. Сопоставь измерения с initial hypothesis.
3. Определи реальный bottleneck.
4. Рассчитай измеренный или диагностический upper bound.
5. Оцени потенциальную стоимость реализации.
6. Определи numerical risk.
7. Сформулируй следующий атомарный эксперимент.
8. Делегируй его подходящему `@general`.
9. Обнови research manifest.

Для FAIL формируй объяснение механизма и новые условия применимости гипотезы.

Для INCONCLUSIVE уточняй экспериментальный дизайн, изолируй confounding variables и повторяй измерения.

Для BLOCKED фиксируй точный технический blocker и минимальный unlock.

Для PASS переходи к следующему уровню реализации и validation.

После существенного подтверждённого выигрыша выполняй fresh profiling соответствующего experimental baseline.

Продолжай исследование, пока evidence поддерживает конкретный falsifiable next experiment.

Порядок P0–P3 адаптируй по measured payoff, numerical exactness, стоимости реализации и риску.

# 9. Final Deliverables

Подготовь итоговый research report на русском языке. (Зафиксировать в отдельную цель и коммит пуш)

Он должен содержать:

- TL;DR.
- Актуальный baseline.
- Карту bottlenecks.
- Историю выполненных Context Packs.
- Результаты каждой гипотезы.
- Измеренный upper bound каждого направления.
- Подтверждённые и отклонённые варианты с объяснением механизма.
- Реальные E2E benchmarks.
- Accuracy evidence.
- Независимые review verdicts.
- Paths ко всем воспроизводимым artifacts.
- Experimental patches для квалифицированных candidates.
- Оставшиеся возможности оптимизации.
- Точные technical blockers.
- Приоритизированный implementation plan для последующего внедрения.

Для каждого пункта implementation plan укажи:

- Целевые файлы и symbols.
- Конкретное изменение.
- Почему оно должно работать согласно измерениям.
- Ожидаемый performance effect.
- Numerical boundaries.
- Acceptance criteria.
- Reproduction commands.
- Rollback strategy.
- Зависимости от других оптимизаций.

Сохрани актуальный research manifest, достаточный для продолжения исследования другим оркестратором или новой agent session.

# 10. Start

Приступай к практическому RECON.

Сначала запусти через `@general` параллельные Context Packs для восстановления истории экспериментов, анализа архитектуры, подготовки instrumentation, snapshot extraction и dependency analysis.

Самостоятельно организуй сбор свежего baseline и очередь guarded GPU benchmarks.

Далее итеративно углубляй каждую перспективную гипотезу на основании измерений, активно используя сабагентов для прототипирования, независимого анализа и проверки результатов.

Основной критерий успеха — практически доказанный прирост decode throughput при сохранении точности вычислений и ответов модели.
