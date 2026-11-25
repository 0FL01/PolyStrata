## Физически уменьшенный global KV, RX 6700 XT

Раздельные процессы, **не** проценты между разными сериями. Цена CPU miss в скобках: `384 × (1 − hits/lookups)`; относится к lookup основного GEN включая первый предиктор, только W1 серии. В таблице V4 held — frozen/385 experts; adaptive — live 64 swaps с 4900 MiB v2 cap; default — авто, возможна иная конкретная карта. Не принимать диагностический короткий TG за независимую timing-серию. Последний столбец — вес expert transfers, если измерен; не путать с PCIe трафиком KV.
| Серия / вариант | global resident backing MiB | expert slots / mapped MiB | process final MiB | hits / lookups (CPU на шаг) | TG IDs/s | p90 ms | swap MiB |
|---|---:|---:|---:|---|---:|---:|---:|
| residency/199104-live-exact | 3309.69 | 371 / 1590.0 | 7974.4 | 24825/49536 (191.6) | 18.36 | 61.19 | — |
| residency/199104-live-scan32 | 3309.69 | 371 / 1590.0 | 7974.4 | 25169/49536 (188.9) | 26.27 | 43.86 | — |
| residency/199104-live-ring32 | 544.00 | 1017 / 4356.0 | 7974.4 | 38225/49536 (87.7) | 34.63 | 33.12 | — |
| residency/199104-live-scan80 | 3309.69 | 371 / 1590.0 | 7974.4 | 25256/49536 (188.2) | 23.54 | 46.37 | — |
| residency/199104-live-ring80 | 1360.00 | 826 / 3540.0 | 7974.4 | 35681/49536 (107.4) | 29.02 | 37.65 | — |
| residency/199104-exact | 3309.69 | 371 / 1590.0 | 7974.4 | 15542/49536 (263.5) | 13.80 | 76.49 | — |
| residency/199104-scan | 3309.69 | 371 / 1590.0 | 7974.4 | 15513/49536 (263.7) | 22.44 | 47.45 | — |
| residency/199104-ring-held | 544.00 | 371 / 1590.0 | 5208.4 | 15513/49536 (263.7) | 20.72 | 52.14 | — |
| residency/199104-ring-mask-funded | 544.00 | 1017 / 4356.0 | 7974.4 | 28237/49536 (165.1) | 22.80 | 44.86 | — |
| codec/speed0-128-default-refit | 1739.00 | 1013 / 4352.0 | 9160.0 | 17935/24960 (108.1) | 27.36 | 42.64 | — |
| codec/speed0-128-held-off | 2244.00 | 385 / 1650.0 | 6962.9 | 7304/24960 (271.6) | 19.01 | 56.33 | — |
| codec/speed0-128-held-refit | 1739.00 | 385 / 1650.0 | 6457.9 | 7341/24960 (271.1) | 18.64 | 57.55 | — |
| codec/speed0-199-default-refit | 2550.81 | 819 / 3520.0 | 9139.9 | 16730/24960 (126.6) | 23.43 | 48.86 | — |
| codec/speed0-199-held-off | 3320.31 | 385 / 1650.0 | 8039.4 | 7879/24960 (262.8) | 15.78 | 67.91 | — |
| codec/speed0-199-held-refit | 2550.81 | 385 / 1650.0 | 7269.9 | 7881/24960 (262.8) | 17.10 | 64.53 | — |
| codec/speed0-64-default-refit | 907.00 | 1208 / 5184.0 | 9159.8 | 19154/24960 (89.3) | 32.40 | 37.54 | — |
| codec/speed0-64-held-off | 1156.00 | 385 / 1650.0 | 5874.7 | 7367/24960 (270.7) | 20.61 | 51.99 | — |
| codec/speed0-64-held-refit | 907.00 | 385 / 1650.0 | 5625.8 | 7388/24960 (270.3) | 19.13 | 58.41 | — |
| fidelity/speed-affine199 | 2839.91 | 385 / 1650.0 | 7558.8 | 7840/24960 (263.4) | 17.53 | 60.94 | — |
| fidelity/speed-dense199 | 3320.31 | 385 / 1650.0 | 8039.4 | 7879/24960 (262.8) | 16.48 | 65.42 | — |

### Повторное профилирование после изменения layout

HIP events на серии иных процессов; 8 global visits в миллисекундах (медиана суммы на один predictor), не wall latency:
* 131328-exact-events 7.333 ms; 24 postwarmup steps
* 131328-scan-events 2.092 ms; 24 postwarmup steps
* 199104-exact-events 10.792 ms; 24 postwarmup steps
* 199104-live-exact-events 10.784 ms; 32 postwarmup steps
* 199104-live-ring32-events 2.109 ms; 32 postwarmup steps
* 199104-live-scan32-events 2.054 ms; 32 postwarmup steps
* 199104-ring-events 2.134 ms; 24 postwarmup steps
* 199104-scan-events 2.047 ms; 24 postwarmup steps
* events0-128-held-off 7.43 ms; 16 postwarmup steps
* events0-128-held-refit 9.167 ms; 16 postwarmup steps
* events0-128-held-reuse 9.093 ms; 16 postwarmup steps
* events0-199-held-off 10.667 ms; 16 postwarmup steps
* events0-199-held-refit 13.259 ms; 16 postwarmup steps
* events0-199-held-reuse 13.265 ms; 16 postwarmup steps
* events0-64-held-off 3.878 ms; 16 postwarmup steps
* events0-64-held-refit 4.81 ms; 16 postwarmup steps
* events0-64-held-reuse 4.827 ms; 16 postwarmup steps
