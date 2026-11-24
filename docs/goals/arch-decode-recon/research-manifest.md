# Архитектурный RECON: журнал кампании

Статус: **unmet** для целевого ускорения; RECON и [итоговый отчёт](../2026-11-24-inkling2-architecture-aware-results.md) завершены. Контракт — [исходная цель](../2026-11-24-inkling2-architecture-aware-decode-recon.md), §§1–10. Цель — измеренный выигрыш serial decode без изменения численной траектории; 25% — ориентир, а не основание ослабить parity. Модель, production-конфигурация и формат состояния не изменены.

## Что считать результатом

| Этап | Решение и первичное доказательство |
|---|---|
| CP-00/01/02 | Восстановить историю, измерить HEAD на рабочей модели, отдельно получить трассу логических visits. Исторические t/s не подставлять вместо новых. |
| P0/P1 | Отделить цену CPU misses от числа cache hits. Проверять прогноз на отложенных токенах. Смена backend не сохраняет численную семантику автоматически. |
| P2 | Измерять на реальном snapshot. Включить reference, raw fallback, offsets и alignment в размер; подтвердить byte round-trip до GPU-прототипа. |
| P3 | Отделить неизменяемые bank terms от live stream. Для выигрыша нужна экономия на критическом пути, а не сумма длительностей всех controller kernels. |
| X/R | Квалифицировать только варианты с Q0–Q5 по применимости и ≥4 untraced paired comparisons выше шума A/A. Отдельный русский отчёт, независимые verdicts, план внедрения или точный blocker. |

## Точка воспроизведения

- Исходники движка `c0dd0e52f3975dbed3972deda2c2c75010b330c2`; `398027f8` меняет только инструкцию guard. HIP Release `build-poly-hip/strata-poly`, SHA-256 `877fbdfed4120abe7ee9f5e093a3af17f9c8220ebb28a8388a650cf338a775ef`. Сборка: `cmake --build build-poly-hip --target strata-poly -j4`.
- RX 6700 XT (gfx1031, 12 GiB), Ryzen 7 5700X, Fedora 45, `7.2.15-300.fc45.x86_64`. `strata.service` до запуска inactive; не останавливался. Рабочая конфигурация указывает на отдельный бинарник `strata-poly-c014d6ec`. Для измерений используется сборка HEAD.
- **Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf**, заявленный SHA-256 `7af63a2c9362f0ac6fc708443348ead5dbfcb7ecde8cd454ea3572c68864aa72`, не Bartowski/H0. Q8_0 K/V, 200000 context, prompt batch 2048, 8 CPU workers, 2048 MiB reserve, 64 swaps. Параметры и HIP env берутся из рабочей конфигурации; только pipe server, копия expert profile, без lookup/draft. Исходный profile SHA-256 `9293626a35e6fa2ecd34ba5078eb5c7add2f39fec1afe7fde425cd116bd2d946`.
- Закрытый каталог (`$D`): `~/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK`. `profile.seed`; `mixed-N/{config.json,prompt.csv,metrics.json,engine.log,session.bin}`; `guard-*/{warnings.json,stop-reasons.json,vram.jsonl}`. Содержимое журналов и снимки в Git не переносить. Скрипт заново строит prompt из файлов репозитория с рабочим токенизатором; `prompt_sha256` проверяет identity. Смешанный текст — синтетический workload, не quality battery.
- Первый strict preflight отказал **до GPU launch**. По явному указанию пользователя memory/page fault отдельно учитывается как warning. На `--warn-amdgpu-faults` первый запуск завершён, stop reasons пусты. Следующий запуск только после завершения предыдущего. Ни VRAM floor 1536 MiB, ни проверки reset/hang/timeout/OOM не снижены.

```sh
# D — приватный каталог вне репозитория; profile.seed — неизменная копия профиля.
python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output "$D/guard-repeat" -- \
  timeout 1500 .venv/bin/python docs/goals/arch-decode-recon/baseline_run.py \
  "$D/repeat" 2048 --profile-seed "$D/profile.seed" --save
```

## Текущий checkpoint

- **CP-01 — length sweep завершён**: [таблица](baseline.md), [машиночитаемая сводка](baseline.json). Полный SHA-256 production GGUF проверен независимо (`$D/verified-model.sha256`). Один length sweep не даёт noise floor.
- **P2-01→03**: [результаты на Tideglass](kv/README.md). На 2K/16K/64K/128K/near-cap все испытанные тайлы точно восстановлены, но лучшая практическая экономия полного KV 2.63/2.29/1.19/0.85/0.80% против gate 20%. Для этих вариантов не запускать GPU codec.
- **P3-01 PASS; P3-02 измерен весь input slot**: [dependency graph](loop/loop-dependency-graph.md) и [профили текущей модели](profiler/results-2k.md). На 2K/64K 0.312/0.323 ms/token для всего входа. Подфазы внутри kernel отдельно не измерены; даже их суммарный бюджет недостаточен. P3-03→05 не продвигать.
- **CP-02**: [разбор реального образца](trace/sample-validation.md), [хэши и поля](trace/trace-manifest.json). 12288/12288 visits согласуются с cache replay; 257/257 greedy ID совпали, одна пара ON/OFF дала −3.66% t/s. Это не оценка untraced E2E и не полная Q2 parity.
- **P0/P1**: [held-out replay](route/tideglass-recon.md). На 64K медианы join proxy при 1/2/3 misses: 4.5/110/310 µs. Full-weight H2D достигает малой доли misses дорогой ценой; дальше проверять только CPU-prefetch без смены backend и отдельный same-byte placement oracle. Q0 hardware-контрпример на настоящем decode-входе хранится в `~/.cache/opencode-tmp/opencode/arch-cp04-capture.tBbJzJ/{run,replay-guard}`.

- **CP-P0-02/03**: [same-byte oracle и marginal utility](placement/README.md). На 64K tested static / online policy не улучшили baseline даже в симуляции. Oracle с будущими 32 токенами и pack+copy charge даёт **+2.64 ms/token** (не E2E). Выбор для последней проверки: зафиксировать каузальный предиктор, затем применить его к независимому 64K логу на неповторяющемся коде и отсеять при net <2 ms/token. Новая фикстура в `$D/long-holdout/`, SHA-256 `07a8a845547371b5f98d60e790c2dcceaa464dd4689f3a0ca1da8092f4194baa`.

- **CP-00 PASS**: [историческая карта](historical-experiment-map.md). Для P0 сравнивать с действующей x2-политикой, а не с равномерными slots. Чтобы получить +25% от исторических 38.14 t/s на 64K, надо снять 5.24 ms/token **wall**, а CPU join и GPU spans частично перекрываются.
- **CP-R-02**: [границы и live A/A](review-baseline.md). Три длинных интервала составили 22.8% первого 2K окна. Их не отбрасывали. После warmup 42.61 t/s по восьми ногам; четыре пары одной сборки различались на 0.64–1.62%, ID совпали. В `mixed` первые 2K и 4K — только prose, а near-cap использует повторения; дополнительные task-prefix traces и неповторяющийся 64K код измерены отдельно. Для пар используется первое–последнее получение ID, обязательная полная длина и нулевые reuse/drafts.

**Закрытие:** `$D/trace-{on,off}`, `$D/roc{2k,64k}`, `$D/trace-code-2048`, `$D/trace-mixed-65536`, `$D/guard-{transfer,proxy}`. Сборка CP-02: `~/.cache/opencode-tmp/opencode/arch-cp02-trace-wt/build-cp02/strata-poly`. A/A: `$D/aa-2048-r1/`. Новый causal-32 predictor [дважды не прошёл gate](placement/causal-32-report.md), включая `$D/long-code-65536/run/visits.psv`. [CPU-reference HIP](arithmetic/cpu-ref-rx6700xt.md) восстановил отдельные quant/dot листья, но не FP32 SiTU. [CPU helper на 64K](prefetch/long-context.md) не оправдывает интеграцию. Полноценного engine candidate для matched-placement Q2–Q5, комбинации и четырёхпарного untraced A/B не осталось; для них нет результата. Не выкатывать исследовательские патчи. Очередь GPU закрыта; возобновлять только при изменении условий из итогового отчёта, не дообучать на раскрытых holdouts.
