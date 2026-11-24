# Tideglass на RDNA2: результаты архитектурного RECON

**Статус: active.** [Контракт исследования](2026-11-24-inkling2-architecture-aware-decode-recon.md) · [manifest с точкой продолжения](arch-decode-recon/research-manifest.md).

**На текущем этапе включать в production нечего.** Доказана потеря численной эквивалентности при прямой замене CPU expert на GPU expert. Lossless-кодеки корректны на проверенных данных, но их реальная упаковка далека от порога, который оправдывал бы ещё один путь в attention. Измеренный speedup кандидата отсутствует: это не равнозначно измеренному нулевому выигрышу.

## База сравнения

На RX 6700 XT, Ryzen 7 5700X, HIP Release `c0dd0e52`, production Tideglass и Q8_0 KV измерены **32.86 / 37.35 / 36.68 / 32.34 / 26.95 / 21.35 t/s** при 2048 / 4096 / 16384 / 65536 / 131072 / 199743 prompt IDs. Это шесть отдельных первых запросов по 256 шагов, не A/B и не medians повторных серий. Минимум свободной VRAM ≥1999 MiB, hard stop не было. [Времена, перцентили и ограничения](arch-decode-recon/baseline.md). Старые 51.325 t/s на 2K — другая модель, задача и placement, а не контрольная нога новой оптимизации.

## Решения по направлениям

| Направление | Что проверено | Решение |
|---|---|---|
| P0 — размещение | На слое 3 и model-derived входе **реальный** HIP two-slot и AVX2 расходятся уже в `xn[1603]`: −24 и −25 при одном scale. На настоящем decode-входе (pos 2048, visit 6, CPU expert 116, weight `0x3f94809c`) все 2304 кода совпали, но 1152/1152 строк gate различались, первая `0x3e49071c` vs `0x3e4aa5cc`. | Нельзя менять backend, сохраняя только router IDs и weights. Нужны shadow placement, reference arithmetic и исходный порядок агрегации. Это Q0-контрпримеры, не результат engine Q2. |
| P1 — подготовка экспертов | Ядро Tideglass имеет 4 478 976 байтов Q4_0 на expert. PCIe 4.0 ×16. От постановки pinned H2D до возврата `hipStreamSynchronize` медиана 173.69 µs (~25.8 GB/s); `memcpy` из вращающегося рабочего набора в pinned + H2D: 322.65 µs. Контроль на четырёх байтах: 15.84 µs. | Прогноз без попадания в **CPU miss** и completion к deadline не имеет ценности. Нельзя считать время отправки команды временем доступности весов. Цифры без конкурирующего decode; это не E2E gain. |
| P2 — cross-pass KV | [Снимки и кодеки](arch-decode-recon/kv/README.md): на 2K/16K/64K/128K/near-cap лучшая практическая экономия всех 8 full caches 2.63/2.29/1.19/0.85/0.80%. На 199999 committed rows это около 27 MiB вместо целевых 664 MiB. | Ни один проверенный layout не достиг 20%. Не менять `inkling_attend` или disk format. |
| P3 — hoisting | [Lifetime](arch-decode-recon/loop/loop-dependency-graph.md) и [свежий профиль](arch-decode-recon/profiler/results-2k.md): 33 границы logits, все 33 greedy ID совпали с untraced prefix. Banks готовы на visits 18 и 30, но из 36.614 ms/token traced wall весь input slot занимает лишь 0.312 ms/token, включая неустранимый live RMS. | Не интегрировать отдельно. Даже бесплатное удаление всего input slot соответствует +0.86% на **traced** wall, не untraced gain. Вернуться при независимом способе убрать launches. |

### Воспроизведение transfer leaf

`transfer.cpp` измеряет изолированные последовательные операции, не pipeline: один слот в VRAM, pinned host buffer, вращение 32 источников, 4 warmup + 64 измерения, прямой/обратный порядок, 128 результатов на arm. Последняя передача каждого arm проверяется D2H и `memcmp`. Буфер синтетический, размер соответствует реальному Q4_0 expert. Сырые времена — `$D/guard-transfer/command.log`.

```sh
H="$PWD/.venv/lib/python3.13/site-packages/_rocm_sdk_devel"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -I"$H/include" docs/goals/arch-decode-recon/transfer.cpp \
  -L"$H/lib" -Wl,-rpath,"$H/lib" -lamdhip64 -o "$D/transfer"
python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output "$D/guard-transfer-repeat" -- "$D/transfer"
```

## Условия внедрения

1. **P0:** сначала [held-out replay](arch-decode-recon/route/tideglass-recon.md), затем оценка policy при тех же байтах VRAM. Медиана proxy на 64K 8.22 ms/token; это не достижимая экономия. Сверять `Engine::follow/cover/shrink_cache`, reference assignment и ordered aggregation. Патч для движка оправдан только после hardware-feasible экономии порядка 2–3 ms/token, полной операторной parity и A/B. Без этих условий оставить действующий `use_weight` и placement.
2. **P1:** предыдущий pass даёт полезную информацию, но при одном прогнозе на слой/переход на 64K достигнуто лишь 14.4% recall реальных CPU misses, 83.6 MB speculative H2D/token и 1.52 ms/token gap на затронутых visits **до** платы за копирование и interference. Не менять GPU residency по такому сигналу. Проверять только более дешёвое прогревание данных уже назначенного CPU expert; CPU bytes и порядок сумм должны оставаться теми же.
3. **P2/P3:** текущих данных для внедрения нет. P2 пересматривать после другой exact representation с ≥20% на aggregate footprint и независимой workload. P3 — при ином scheduling, которое действительно убирает launches, либо существенно большем свежем профиле. Не суммировать независимые потолки или дважды перекрывающиеся участки.

**Независимые проверки:** [бенчмарки](arch-decode-recon/review-baseline.md), [извлечение и round-trip KV](arch-decode-recon/review-kv.md), [HIP lifetime, ограничения join и trace replay](arch-decode-recon/review-architecture.md). В двух коротких контекстах raw allocation с разной гранулярностью нельзя выдавать за codec gain; для 2K при одинаковом arena grouping у raw и hybrid получилось 40 MiB. Ни одно ревью не является Q0–Q5 sign-off для изменения движка.

Production-флаги, веса, backend и формат сессий не изменены. Для каждой следующей реализации нужны отдельный opt-in patch, сохранение Q0/Q1 границ, проверки logits/greedy/sampling/состояния на сопоставимом размещении и парный untraced E2E за пределами A/A. Для P0 это включает CPU/GPU grouping и merge; для P1 router остаётся авторитетным даже при ошибочном прогнозе. Без прошедших эти ворота candidates комбинированный A/B и deployment не проводятся. Для отката исследовательских вариантов достаточно выбрать исходный Release без экспериментального патча.
