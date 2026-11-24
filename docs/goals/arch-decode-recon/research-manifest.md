# Архитектурный RECON: журнал кампании

Статус: **active**. Контракт — [исходная цель](../2026-11-24-inkling2-architecture-aware-decode-recon.md), §§1–10. Цель — измеренный выигрыш serial decode без изменения численной траектории; 25% — ориентир, а не основание ослабить parity. Не менять модель, production-конфигурацию и формат состояния.

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

Первый untraced 2K leg: 256 интервалов greedy decode, 32.86 t/s; p10/median/p90/p99 по приходу ID 19.79/22.64/29.24/146.47 ms. Это одна серия в период параллельного CPU RECON, **не** сравнение с историческими 51.325 t/s и не оценка speedup. Измеренные ID и интервалы сохранены отдельно. Из этой же сессии сохранён реальный KV snapshot. Очередь: 4K → 16K → 64K → 128K → 199743 (оставлено место ответу). При hard stop сохранить завершённые ноги отдельно. Затем quiet A/A и отдельные instrumented runs.
