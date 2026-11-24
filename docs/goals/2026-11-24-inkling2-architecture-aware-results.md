# Tideglass на RDNA2: результаты архитектурного RECON

**Итог: исследование завершено; целевой прирост throughput не достигнут.** [Контракт исследования](2026-11-24-inkling2-architecture-aware-decode-recon.md) · [manifest и условия возобновления](arch-decode-recon/research-manifest.md).

**На текущем этапе включать в production нечего.** Доказана потеря численной эквивалентности при прямой замене CPU expert на GPU expert. Lossless-кодеки корректны на проверенных данных, но их реальная упаковка далека от порога, который оправдывал бы ещё один путь в attention. Измеренный speedup кандидата отсутствует: это не равнозначно измеренному нулевому выигрышу.

## База сравнения

На RX 6700 XT, Ryzen 7 5700X, HIP Release `c0dd0e52`, production Tideglass и Q8_0 KV измерены **32.86 / 37.35 / 36.68 / 32.34 / 26.95 / 21.35 t/s** при 2048 / 4096 / 16384 / 65536 / 131072 / 199743 prompt IDs. Это шесть отдельных первых запросов по 256 шагов, не A/B и не medians повторных серий. Минимум свободной VRAM ≥1999 MiB, hard stop не было. [Времена, перцентили и ограничения](arch-decode-recon/baseline.md). Старые 51.325 t/s на 2K — другая модель, задача и placement, а не контрольная нога новой оптимизации.

Для повторного контроля на 2K использовать **42.61 t/s** (медиана восьми запусков после warmup, 41.96–43.02). В четырёх парах одной сборки максимальный модуль разницы составил 1.62%; это наблюдавшийся разброс, а не верхняя граница шума. На 64K отдельная ROCm-трасса дала 4.630 ms/token attention и 4.592 ms/token в группе прочих projections; wall=40.890 ms/token против 17.638 ms/token kernel busy union. Остаток нельзя полностью отнести к CPU expert work. На 17/17 показанных ID профиль совпал с untraced префиксом.

## Решения по направлениям

| Направление | Что проверено | Решение |
|---|---|---|
| P0 — размещение | На слое 3 и model-derived входе **реальный** HIP two-slot и AVX2 расходятся уже в `xn[1603]`: −24 и −25 при одном scale. На настоящем decode-входе (pos 2048, visit 6, CPU expert 116, weight `0x3f94809c`) все 2304 кода совпали, но 1152/1152 строк gate различались, первая `0x3e49071c` vs `0x3e4aa5cc`. | Нельзя менять backend, сохраняя только router IDs и weights. Нужны shadow placement, reference arithmetic и исходный порядок агрегации. Это Q0-контрпримеры, не результат engine Q2. |
| P1 — подготовка экспертов | Ядро Tideglass имеет 4 478 976 байтов Q4_0 на expert. PCIe 4.0 ×16. От постановки pinned H2D до возврата `hipStreamSynchronize` медиана 173.69 µs (~25.8 GB/s); `memcpy` из вращающегося рабочего набора в pinned + H2D: 322.65 µs. Контроль на четырёх байтах: 15.84 µs. | Прогноз без попадания в **CPU miss** и completion к deadline не имеет ценности. Нельзя считать время отправки команды временем доступности весов. Цифры без конкурирующего decode; это не E2E gain. |
| P2 — cross-pass KV | [Снимки и кодеки](arch-decode-recon/kv/README.md): на 2K/16K/64K/128K/near-cap лучшая практическая экономия всех 8 full caches 2.63/2.29/1.19/0.85/0.80%. На 199999 committed rows это около 27 MiB вместо целевых 664 MiB. | Ни один проверенный layout не достиг 20%. Не менять `inkling_attend` или disk format. |
| P3 — hoisting | [Lifetime](arch-decode-recon/loop/loop-dependency-graph.md) и [свежий профиль](arch-decode-recon/profiler/results-2k.md): 33 границы logits, все 33 greedy ID совпали с untraced prefix. Banks готовы на visits 18 и 30, но из 36.614 ms/token traced wall весь input slot занимает лишь 0.312 ms/token, включая неустранимый live RMS. | Не интегрировать отдельно. Даже бесплатное удаление всего input slot соответствует +0.86% на **traced** wall, не untraced gain. Вернуться при независимом способе убрать launches. |

Проверена также [группировка Q8_0 dense projections](arch-decode-recon/dense/README.md): в отличие от старого Bartowski, Tideglass уже делает один grouped front launch на каждый из 48 visits, без отдельной квантизации. Ещё 12 Q4_0 relative projections занимают вместе лишь 0.057 ms/token. Даже устранение их вызовов **вместе с двумя соседними паузами** за нулевую цену соответствует +0.56% traced wall; не делать ещё один mixed kernel на этих данных.

**Численный прототип для P0:** [HIP-реализация AVX2 quant/dot](arch-decode-recon/arithmetic/README.md) на двух fixtures совпала с CPU по Q8_0, gate/up и down при точном CPU SiTU. На реальном expert 116 с уже резидентными горячими весами pack + gate/up заняли 22.878 µs против 14.190 µs у обычного HIP; это цена отдельного leaf, без transfer и без aggregation. С существующей HIP SiTU разошлись 58/1152 FP32 значений, первое `0xbe122059` vs `0xbe12205a`. На этих двух входах переквантование скрыло разницу, но контракт требует совпадения самой операции. **Q1 не пройден.**

### Воспроизведение transfer leaf

`transfer.cpp` измеряет изолированные последовательные операции, не pipeline: один слот в VRAM, pinned host buffer, вращение 32 источников, 4 warmup + 64 измерения, прямой/обратный порядок, 128 результатов на arm. Последняя передача каждого arm проверяется D2H и `memcmp`. Буфер синтетический, размер соответствует реальному Q4_0 expert. Сырые времена — `$D/guard-transfer/command.log`.

```sh
H="$PWD/.venv/lib/python3.13/site-packages/_rocm_sdk_devel"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -I"$H/include" docs/goals/arch-decode-recon/transfer.cpp \
  -L"$H/lib" -Wl,-rpath,"$H/lib" -lamdhip64 -o "$D/transfer"
python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output "$D/guard-transfer-repeat" -- "$D/transfer"
```

## Условия внедрения

**Финальный gate P0 не пройден.** [Прогноз на 32 токена](arch-decode-recon/placement/causal-32-report.md) был зафиксирован до снятия независимой трассы. На mixed 64K получено **+0.36 ms/token** в модели после pack+copy (95% блочный интервал −0.02…+0.99), на независимом 64K code — **−1.02** (−1.82…−0.44). Даже при нулевой цене копий gross составил только +1.59 и +0.21. Для P1 [применение CPU-замеров к 64K](arch-decode-recon/prefetch/long-context.md) даёт 0.338–0.356 ms/token в replay, не E2E. Не объединять два не прошедших отбор варианта.

1. **P0:** [held-out replay](arch-decode-recon/route/tideglass-recon.md) и [same-budget simulator](arch-decode-recon/placement/README.md) различают возможность и реализуемую политику. Медиана proxy на 64K 8.22 ms/token; это не достижимая экономия. На отложенном участке cost-aware static проиграл 1.79 ms/token даже при бесплатной установке, causal past-16 проиграл 0.73 с учётом pack+copy. Oracle, знающий следующие 32 токена, **выиграл 2.64 ms/token в модели** (блочный 95% интервал 2.28–3.09). Это примерно 35.36 вместо 32.34 t/s только при аддитивности времени и неизменности траектории; фактический E2E не измерен. Сверять `Engine::follow/cover/shrink_cache`, reference assignment и ordered aggregation. Патч для движка оправдан только после hardware-feasible экономии порядка 2–3 ms/token на независимом длинном запросе без подглядывания, полной операторной parity и A/B. Без этих условий оставить действующий `use_weight` и placement.
2. **P1:** предыдущий pass даёт полезную информацию, но при одном прогнозе на слой/переход на 64K достигнуто лишь 14.4% recall реальных CPU misses, 83.6 MB speculative H2D/token и 1.52 ms/token gap на затронутых visits **до** платы за копирование и interference. На [четырёх независимых префиксах заданий](arch-decode-recon/route/task-prefixes.md) recall 25.1% при 29.8 MB ложных прогнозов на токен. Не менять GPU residency по такому сигналу. Проверять только более дешёвое прогревание данных уже назначенного CPU expert; CPU bytes и порядок сумм должны оставаться теми же.
   **CPU-only прототип:** [54–58 µs выигрыша на верном прогнозе](arch-decode-recon/prefetch/README.md), 4104/4104 операторных сравнений побитово совпали. Ложные прогнозы во время предыдущего CPU вызова замедляли весь путь в медиане на 41–47 µs. При учёте промахов и перекрытия модель даёт лишь 0.20–0.22 ms/token на 2K, а не измеренный выигрыш движка. Не отправлять этот вариант в engine A/B только на основании горячего microbench.
3. **P2/P3:** текущих данных для внедрения нет. P2 пересматривать после другой exact representation с ≥20% на aggregate footprint и независимой workload. P3 — при ином scheduling, которое действительно убирает launches, либо существенно большем свежем профиле. Не суммировать независимые потолки или дважды перекрывающиеся участки.

**Независимые проверки:** [бенчмарки](arch-decode-recon/review-baseline.md), [извлечение и round-trip KV](arch-decode-recon/review-kv.md), [HIP lifetime, ограничения join и trace replay](arch-decode-recon/review-architecture.md). В двух коротких контекстах raw allocation с разной гранулярностью нельзя выдавать за codec gain; для 2K при одинаковом arena grouping у raw и hybrid получилось 40 MiB. Ни одно ревью не является Q0–Q5 sign-off для изменения движка.

В симуляции P0 debiting pack+copy означает оплату полного изолированного p90 сервиса **кандидата**: не возвращается стоимость фоновых копий baseline и не засчитывается overlap. Поэтому знак или величина такого net — результат конкретного screening gate, а не измерение разницы двух планировщиков. Прямой A/B может дать другую величину. [Независимое ревью placement](arch-decode-recon/review-placement.md) подтвердило no-go для проверенных политик, но не для placement в целом.

Production-флаги, веса, backend и формат сессий не изменены. Для каждой следующей реализации нужны отдельный opt-in patch, сохранение Q0/Q1 границ, проверки logits/greedy/sampling/состояния на сопоставимом размещении и парный untraced E2E за пределами A/A. Для P0 это включает CPU/GPU grouping и merge; для P1 router остаётся авторитетным даже при ошибочном прогнозе. Без прошедших эти ворота candidates комбинированный A/B и deployment не проводятся. Для отката исследовательских вариантов достаточно выбрать исходный Release без экспериментального патча.
