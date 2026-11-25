# Воспроизведение Adaptive KV / Hybrid Decode

Локальный каталог — `~/.cache/opencode-tmp/opencode/`; пути в таблице от него. Дамп KV, труды модели, tool payload, ELF и логи содержат локальные данные, в git не включены. Экспериментальные флаги — **не** опции установленного `strata-poly`. Сохранены исходники, входные хеши, команды сборки, нативный протокол, профили и guard receipts. Для переноса на другое железо сначала заново собрать и измерить; здесь проверен только gfx1031.

| Что воспроизводить | Первичный артефакт / как получить таблицу |
|---|---|
| Старые гипотезы и контроли | `global-kv-20261125/{PLAN.md,SCOREBOARD.md}` и [прежняя сводка](scoreboard.md). Не считать старые ноги повторами новых. |
| Общий ROC trace и первый untraced crossover | `kv-live-profile-20261125/{REPORT.md,build.py,run.py,analyze.py,cap4900/}`. `analyze.py cap4900` обрабатывает уже снятые ROC dispatch/API/copy/marker traces. `hybrid-20261125/late.py` читает оттуда же `trace-visits-ns.json`. |
| Новый критический путь | `kv-live-profile-20261125/funded-roc/REPORT.md` (4900/5412); `global-kv-20261125/critical/RESULTS.md`, `critical/code197-v3/visits.jsonl` (197K repair и fixed-map fingerprint); `k8v4-suffix.sLesOsDB/PROFILE.md` (tool return после H2). |
| Нативная факторная матрица | `hybrid-20261125/unified/build.py`; `global-kv-freeagent-confirm/{workloads.py,trial.py,report.py}`. `python -B report.py unified-confirm` в последнем каталоге. Замеры находятся в `TASK-SEED/ARM/trajectory.json` и `turn-N.{prompt.csv,output.csv,rows.tsv,metrics.json}`. Для нового прогона `trial.py run --output НОВЫЙ_КАТАЛОГ --elf-sha256 a65d0d117a894f7a51e813ce0bf263d5c8ead3921a07a99c0778bfd4e893e95a`. |
| Независимый reference и iso-process | `global-kv-freeagent-confirm/{official-hybrid42,official-hybrid79}/report.md`; команда `trial.py run --suite official --seed 42 --output НОВЫЙ_КАТАЛОГ` (повторить с 79). `hybrid-20261125/iso5412-mixed/` — Q8K/V4K ABBA при почти одинаковом измеренном resident. |
| Фиксированные Q и KV precision | `adaptive-mixed-kv-20261125/{REPORT.md,scan.py,pack_test.py,sweep.csv,isa-summary.txt}`. `adaptive-selector-recon.q6ny7r/{REPORT.md,run.sh,static_screen.py,fixed-frontier.txt}` для политик выбора. CPU-only, не пересчитывать matmul одновременно с untraced benchmark. |
| Physical affine V4-32 | `hybrid-20261125/unified-affine32/{RESULTS.md,gate.sh,greedy199/,board42/,original79/}`. Saved device-picked pages и независимый CPU oracle, VMM receipts и две реальные траектории в сравнении с symmetric в том же ELF. |
| Прочитать тот же **реальный выбранный** KV на RDNA2 | `kv-leaf-fractional.pp8DCx/{RESULTS.md,leaf.hip,check.py,isa.txt,guard-A/,guard-B/}`; сохранены обе таблицы страниц, результаты и независимый dequant/attention oracle. |
| Широкий compact suffix | `k8v4-suffix.sLesOsDB/{README.md,RESULTS.md,build_engine.py,agent_pair.py,guard-01/,agent-01/}`. `python -B report_agent.py agent-01`. |
| Fractional experts | `fractional-recon/{RECON.md,REPLAY197.md,screen.py,screen.json,phase.cpp,phase-mixed.csv,phase-code.csv}`. Документы содержат параметры, формулу стоимости, разделение train/held-out, равный бюджет на актуальном trace и команды сборки. |
| Speculation | `sparse-spec.tppF9v/{README.md,freeagent-final.txt,freeagent-final.json}` для причинного replay, `hybrid-20261125/{spec-build.py,spec/,spec-mixed/,spec-agent-complete/,spec-audit-mixed/}` для реального verifier. Сохранены исходная W4 нога `spec-agent/4/` и её ошибка post-check; terminal shown отличается от committed. `summary.py spec-mixed` показывает фактические, не максимальные ширины и acceptance. `freeagent_replay.py --hybrid ../hybrid-20261125/replay-view` использует обе измеренные стоимости. |
| AVX2 refit, бюджет, 129K | `hybrid-20261125/{pack.cpp,pack.txt,pack-build.py,trials.py,summary.py}`. Таблица: `python -B summary.py pack-mixed fund-mixed`. `pack-audit-mixed/` — отдельная проверка на frozen arena, не timing arm. |

**Контрольные входы.** GGUF `Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf`, ранее проверенный SHA-256 `7af63a2c9362f0ac6fc708443348ead5dbfcb7ecde8cd454ea3572c68864aa72`. Mixed: `swa-recon-20261125/seed200/session.bin`, 199104 ID, SHA `51d3a9bfbb05b0eea2d9640348b95a041b8bf580d297f9693da425f00d279181`. Agent: `global-kv-20261125/longagent/build-197/session.bin`, 197021 ID, SHA `742184fcbcc3fa9e92538cb5b976b113c07f4fd2e6bb624639b3c457d6b713c7`. Профиль каждому процессу копируется заново (`9293626a…`). Полная ёмкость 200000; для W1/TG считать интервалы между эмитированными ID **после** первого, а singleton prompt forward — в TTFT. Температура 0 для greedy; при оценке задач T=1, top_k=0, фиксированное семя, полная история и реальные tool responses.

### Матрица 197K без доступа к локальным логам

`dense/q8k` — Q8 с полным вниманием / K256; `refit/combo` — те же варианты на symmetric V4. `IDs` — весь выход трёх ходов, включая служебные ID. Три TTFT — для первого запроса и двух tool returns. В последней колонке три разные величины: время RESTORE; от восстановленной сессии до **любого** исхода; от начала подготовки процесса до того же исхода. Последнее **уже включает** RESTORE; значения не складывать. Класс `length` — незавершённый final turn; `wrong` — получен полный, неверный JSON. По два полезных чтения в каждой ноге, никаких ретраев.

| Задача/seed | Arm | Результат | IDs | TTFT T1/T2/T3, s | RESTORE / до исхода / c запуска, s |
|---|---|---|---:|---|---|
| coding-42 | dense | pass | 534 | 9.13 / 1.55 / 1.42 | 3.48 / 41.34 / 76.59 |
| coding-42 | q8k | pass | 516 | 1.78 / 1.25 / 1.25 | 3.30 / 27.00 / 32.39 |
| coding-42 | refit | pass | 417 | 2.05 / 1.74 / 1.82 | 4.03 / 26.24 / 32.31 |
| coding-42 | combo | pass | 449 | 2.07 / 1.75 / 1.82 | 4.02 / 23.11 / 29.17 |
| reasoning-42 | dense | pass | 588 | 1.51 / 1.55 / 1.11 | 3.08 / 32.43 / 37.59 |
| reasoning-42 | q8k | length | 607 | 1.66 / 2.60 / 2.08 | 3.14 / 59.26 / 64.52 |
| reasoning-42 | refit | length | 612 | 2.23 / 2.77 / 1.36 | 3.86 / 32.96 / 38.80 |
| reasoning-42 | combo | pass | 500 | 3.54 / 2.82 / 1.38 | 4.84 / 25.78 / 33.03 |
| coding-79 | dense | pass | 498 | 1.50 / 1.28 / 1.26 | 3.12 / 29.94 / 35.12 |
| coding-79 | q8k | pass | 572 | 1.48 / 1.23 / 1.26 | 3.14 / 28.88 / 33.98 |
| coding-79 | refit | length | 607 | 2.04 / 1.73 / 1.81 | 4.81 / 34.30 / 41.25 |
| coding-79 | combo | pass | 545 | 2.02 / 1.75 / 1.81 | 3.83 / 25.98 / 31.78 |
| reasoning-79 | dense | pass | 558 | 1.52 / 1.56 / 1.11 | 4.21 / 30.95 / 37.12 |
| reasoning-79 | q8k | wrong | 513 | 1.55 / 1.65 / 1.16 | 3.35 / 25.70 / 31.05 |
| reasoning-79 | refit | length | 615 | 2.20 / 2.75 / 1.36 | 3.92 / 33.85 / 39.76 |
| reasoning-79 | combo | pass | 563 | 2.22 / 2.74 / 1.33 | 4.42 / 25.89 / 32.56 |

**Порядок безопасного повторения.** Не перезаписывать каталоги решений: задать новый output label или копию локального harness, проверить пути и хеши. Передать владение GPU, остановить исходно активный локальный сервис, проверять `strata.service=inactive` под `global-kv-20261125/gpulock`. Только затем один процесс `python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output НОВЫЙ_GUARD -- …`. Оставлять reserve 2048 MiB и floor 1536 MiB; анализировать не только exit status, но и `stop-reasons.json`, новые kernel warnings, `vram.jsonl`. После последнего владельца вернуть исходный systemd unit и проверить loopback `/health`, max_context, SHA работающего engine и прежний автозапуск. Запускать shell-команды генератора или легализовать неуспешный ответ «исправлением» turn нельзя: задействован ограниченный интерпретатор/локальный grader, не `exec` ответа.

По окончании 37 закрытых guard receipts в каталогах этих кампаний: аппаратных остановок нет, warnings только из истории загрузки. Минимум sampled free 2534 MiB в исследовательских ногах и 2022 MiB у установленного ELF (выше floor 1536, но ниже запрошенной reserve 2048: это не жёсткая гарантия свободной памяти). Завершение guard не заменяет проверку успеха команды или grade. Исходный `strata.service` возвращён: локальный `/health` → `loaded=true`, `max_context=200000`, `inkling-2-mini`; работающий `strata-poly-50effa3a`, SHA `a8cfc09f…`, unit active/running, автозапуск по-прежнему disabled. `hybrid-20261125/{service-restored.json,verify-report.py,guards.py}`. Конфиг, исполняемый файл и defaults не менялись.
