# Free-running tasks at 129K and 197K

## Checkpoints
- 129K: 129,015 tokens / 2,339,298,148 bytes; restore 2.11 s, append 63,219 in 127.381 s, save 2.39 s, ready→verified total 136.25 s. Guard min 3107.08 MiB, {'boot': 22, 'kernel': 0}, stops=[].
- 197K: 197,021 tokens / 3,523,425,436 bytes; restore 2.99 s, append 68,006 in 172.368 s, save 2.49 s, ready→verified total 183.20 s. Guard min 3099.35 MiB, {'boot': 22, 'kernel': 0}, stops=[].

## matrix-129 — binary `b60fba7f1272519c9702e9d1dce4abcef415f29074839381e366941ead9e3a28`
| Context | Case | Seed | Arm | Grade | Calls/error/retry | Output IDs/turn | TTFT s/turn | TG tok/s | Restore s | Task s | Time to success s | First ID divergence vs first arm, shared input |
|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| 129K | board | 42 | dense | pass | 1/0/0 | 47/107 | 1.386/0.897 | 24.35 | 2.19 | 8.58 | 8.58 | — |
| 129K | board | 42 | window8192 | length (no natural EOS before native turn limit) | 0/0/0 | 256 | 2.830 | 28.51 | 2.17 | 11.80 | — | T1:2 (reasoning/unknown) |
| 129K | original | 42 | dense | pass | 1/0/0 | 47/89 | 1.360/0.944 | 23.59 | 2.33 | 8.03 | 8.03 | — |
| 129K | original | 42 | window8192 | length (no natural EOS before native turn limit) | 0/0/0 | 256 | 1.355 | 32.97 | 2.25 | 9.11 | — | T1:2 (reasoning/unknown) |
| 129K | current | 42 | dense | pass | 1/0/0 | 38/111 | 1.420/1.002 | 21.38 | 2.47 | 9.35 | 9.35 | — |
| 129K | current | 42 | window8192 | length (no natural EOS before native turn limit) | 0/0/0 | 256 | 1.475 | 30.14 | 2.39 | 9.96 | — | T1:1 (frame/unknown) |

Guard groups complete: 3/3. Lowest free VRAM: 3106.45 MiB.

First-turn cap failures were re-parsed without dispatching any calls. Below are native span classes, not successful tool actions.
- board: {'frame': 29, 'reasoning': 7, 'tool_json': 220}; accepted calls: 0.
- original: {'frame': 8, 'reasoning': 239, 'tool_json': 9}; accepted calls: 0.
- current: {'frame': 25, 'reasoning': 51, 'tool_json': 180}; accepted calls: 0.

## matrix-197 — binary `b60fba7f1272519c9702e9d1dce4abcef415f29074839381e366941ead9e3a28`
| Context | Case | Seed | Arm | Grade | Calls/error/retry | Output IDs/turn | TTFT s/turn | TG tok/s | Restore s | Task s | Time to success s | First ID divergence vs first arm, shared input |
|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| 197K | board | 42 | dense | pass | 1/0/0 | 47/99 | 1.524/1.029 | 18.81 | 3.54 | 10.28 | 10.28 | — |
| 197K | board | 42 | window8192 | length (no natural EOS before native turn limit) | 0/0/0 | 256 | 1.395 | 26.82 | 3.57 | 10.94 | — | T1:2 (reasoning/unknown) |
| 197K | original | 42 | dense | pass | 1/0/0 | 47/89 | 1.541/1.036 | 18.89 | 4.36 | 9.74 | 9.74 | — |
| 197K | original | 42 | window8192 | length (no natural EOS before native turn limit) | 0/0/0 | 256 | 1.466 | 26.66 | 3.19 | 11.07 | — | T1:2 (reasoning/reasoning) |
| 197K | current | 42 | dense | pass | 1/0/0 | 38/105 | 1.388/1.088 | 19.01 | 3.13 | 9.96 | 9.96 | — |
| 197K | current | 42 | window8192 | length (no natural EOS before native turn limit) | 0/0/0 | 256 | 1.378 | 25.05 | 3.16 | 11.60 | — | T1:1 (frame/frame) |

Guard groups complete: 3/3. Lowest free VRAM: 2978.11 MiB.

First-turn cap failures were re-parsed without dispatching any calls. Below are native span classes, not successful tool actions.
- board: {'frame': 32, 'reasoning': 7, 'tool_json': 217}; accepted calls: 0.
- original: {'frame': 2, 'reasoning': 254}; accepted calls: 0.
- current: {'frame': 2, 'reasoning': 254}; accepted calls: 0.

## matrix-codec197 — binary `82a479243d5363efe3ba23c60308073fb5f05bbc0b3371524aa6bb61c91b4424`
| Context | Case | Seed | Arm | Grade | Calls/error/retry | Output IDs/turn | TTFT s/turn | TG tok/s | Restore s | Task s | Time to success s | First ID divergence vs first arm, shared input |
|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| 197K | board | 42 | dense | pass | 1/0/0 | 47/102 | 1.406/1.049 | 13.92 | 3.24 | 13.08 | 13.08 | — |
| 197K | board | 42 | refit | pass | 1/0/0 | 47/100 | 2.253/1.188 | 18.63 | 4.19 | 11.31 | 11.31 | T1:6 (reasoning/reasoning) |
| 197K | current | 42 | dense | pass | 1/0/0 | 38/102 | 1.414/1.065 | 19.29 | 3.11 | 9.71 | 9.71 | — |
| 197K | current | 42 | refit | pass | 1/0/0 | 47/107 | 1.787/1.327 | 20.41 | 3.90 | 10.64 | 10.64 | T1:1 (frame/frame) |

Guard groups complete: 2/2. Lowest free VRAM: 3134.40 MiB.

| Context/case/seed | Arm | PID VRAM (ready/restored/after grade), MiB | Board free (ready/restored/after grade), MiB | Sealed rows | Mapped KV / native cover, MiB | Expert arena, MiB (slots) |
|---|---|---|---|---:|---|---|
| 197/board/42 | dense | 8018.17/7982.07/8005.59 | 3182.80/3132.56/3175.19 | — | —/— | — (—) |
| 197/board/42 | refit | 8018.16/7988.57/8012.09 | 3162.15/3198.85/3171.29 | 194944 | 2558.81/3320.31 | 2368.00 (551) |
| 197/current/42 | dense | 8018.16/7982.06/8005.58 | 3177.06/3219.20/3195.66 | — | —/— | — (—) |
| 197/current/42 | refit | 8018.15/7988.56/8012.08 | 3183.08/3212.68/3190.63 | 194944 | 2558.81/3320.31 | 2368.00 (551) |

## matrix-codec197-seed79 — binary `82a479243d5363efe3ba23c60308073fb5f05bbc0b3371524aa6bb61c91b4424`
| Context | Case | Seed | Arm | Grade | Calls/error/retry | Output IDs/turn | TTFT s/turn | TG tok/s | Restore s | Task s | Time to success s | First ID divergence vs first arm, shared input |
|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| 197K | board | 79 | dense | pass | 1/0/0 | 56/134 | 1.481/1.031 | 19.08 | 3.51 | 12.44 | 12.44 | — |
| 197K | board | 79 | refit | pass | 1/0/0 | 38/133 | 1.885/1.135 | 20.49 | 3.80 | 11.35 | 11.35 | T1:1 (frame/frame) |

Guard groups complete: 1/1. Lowest free VRAM: 3070.54 MiB.

| Context/case/seed | Arm | PID VRAM (ready/restored/after grade), MiB | Board free (ready/restored/after grade), MiB | Sealed rows | Mapped KV / native cover, MiB | Expert arena, MiB (slots) |
|---|---|---|---|---:|---|---|
| 197/board/79 | dense | 8018.17/7982.08/8005.59 | 3067.39/3131.22/3093.11 | — | —/— | — (—) |
| 197/board/79 | refit | 8018.15/7988.56/8012.09 | 3081.12/3115.68/3101.15 | 194944 | 2558.81/3320.31 | 2368.00 (551) |

## Separate 197K oracle capture (excluded from timing)
- Verification: `longagent/capture-197/validation.json`; replay grade=pass, 1 calls/0 failures/0 retries.
- `captures/longcode/set`: 96 sample pairs at predictor positions `197232-197243` inclusive, all eight visits, actual post-tool reasoning; tool `turn_at=197186`.
- Canonical state: `longagent/capture-197/dense/final.state.bin` (197,314 IDs; SHA256 `f6933ca9bba785b22732aa2937e1ee40518c43d75cd732cfcdf8a1ec98705a0a`). Restore seed: `longagent/build-197/session.bin` (197,021 IDs).
- All 2,176 packed current-row bytes per sample matched the canonical state. Replay-vs-timing native parity: False (see capture manifest). Guard min 3069.60 MiB, {'boot': 22, 'kernel': 0}, stops=[].
