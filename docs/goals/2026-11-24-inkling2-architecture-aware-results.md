# Tideglass на RDNA2: результаты архитектурного RECON

**Статус: active.** [Контракт исследования](2026-11-24-inkling2-architecture-aware-decode-recon.md) · [manifest с точкой продолжения](arch-decode-recon/research-manifest.md).

**На текущем этапе включать в production нечего.** Доказана потеря численной эквивалентности при прямой замене CPU expert на GPU expert. Lossless-кодеки корректны на проверенных данных, но их реальная упаковка далека от порога, который оправдывал бы ещё один путь в attention. Измеренный speedup кандидата отсутствует: это не равнозначно измеренному нулевому выигрышу.

## База сравнения

На RX 6700 XT, Ryzen 7 5700X, HIP Release `c0dd0e52`, production Tideglass и Q8_0 KV измерены **32.86 / 37.35 / 36.68 / 32.34 / 26.95 / 21.35 t/s** при 2048 / 4096 / 16384 / 65536 / 131072 / 199743 prompt IDs. Это шесть отдельных первых запросов по 256 шагов, не A/B и не medians повторных серий. Минимум свободной VRAM ≥1999 MiB, hard stop не было. [Времена, перцентили и ограничения](arch-decode-recon/baseline.md). Старые 51.325 t/s на 2K — другая модель, задача и placement, а не контрольная нога новой оптимизации.

## Решения по направлениям

| Направление | Что проверено | Решение |
|---|---|---|
| P0 — размещение | На модели Tideglass, слое 3 и model-derived входе, **реальный** HIP two-slot и AVX2 расходятся уже в `xn[1603]`: −24 и −25 при одном scale. Gate отличается в 1152/1152 строках. На том же входе обычный HIP и two-slot совпали на всех проверенных границах. | Нельзя менять backend, сохраняя только router IDs и weights. Нужны shadow placement, reference arithmetic и исходный порядок агрегации. Это Q0-контрпример, не результат engine Q2. |
| P1 — подготовка экспертов | Ядро Tideglass имеет 4 478 976 байтов Q4_0 на expert. PCIe 4.0 ×16. От постановки pinned H2D до возврата `hipStreamSynchronize` медиана 173.69 µs (~25.8 GB/s); `memcpy` из вращающегося рабочего набора в pinned + H2D: 322.65 µs. Контроль на четырёх байтах: 15.84 µs. | Прогноз без попадания в **CPU miss** и completion к deadline не имеет ценности. Нельзя считать время отправки команды временем доступности весов. Цифры без конкурирующего decode; это не E2E gain. |
| P2 — cross-pass KV | [Снимки и кодеки](arch-decode-recon/kv/README.md): на 2K/16K/64K/128K/near-cap лучшая практическая экономия всех 8 full caches 2.63/2.29/1.19/0.85/0.80%. На 199999 committed rows это около 27 MiB вместо целевых 664 MiB. | Ни один проверенный layout не достиг 20%. Не менять `inkling_attend` или disk format. |
| P3 — hoisting | [Lifetime и границы](arch-decode-recon/loop/loop-dependency-graph.md): banks готовы у входа в visits 18 и 30; 48 сообщений на проход можно подготовить, но не current RMS, rank или gain/forget. | Интеграция только при доказанном экономичном способе подачи этих сообщений; не приписывать входным операциям стоимость всего controller. |

### Воспроизведение transfer leaf

`transfer.cpp` измеряет изолированные последовательные операции, не pipeline: один слот в VRAM, pinned host buffer, вращение 32 источников, 4 warmup + 64 измерения, прямой/обратный порядок, 128 результатов на arm. Последняя передача каждого arm проверяется D2H и `memcmp`. Буфер синтетический, размер соответствует реальному Q4_0 expert. Сырые времена — `$D/guard-transfer/command.log`.

```sh
H="$PWD/.venv/lib/python3.13/site-packages/_rocm_sdk_devel"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -I"$H/include" docs/goals/arch-decode-recon/transfer.cpp \
  -L"$H/lib" -Wl,-rpath,"$H/lib" -lamdhip64 -o "$D/transfer"
python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output "$D/guard-transfer-repeat" -- "$D/transfer"
```

## Условия внедрения

Production-флаги, веса, backend и формат сессий не изменены. Для каждой следующей реализации нужны отдельный opt-in patch, сохранение Q0/Q1 границ, проверки logits/greedy/sampling/состояния на сопоставимом размещении и парный untraced E2E за пределами A/A. Для P0 это включает CPU/GPU grouping и merge; для P1 router остаётся авторитетным даже при ошибочном прогнозе. Без прошедших эти ворота candidates комбинированный A/B и deployment не проводятся. Для отката исследовательских вариантов достаточно выбрать исходный Release без экспериментального патча.
