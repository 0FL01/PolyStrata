# Near-197K authentic post-tool attention: bounded CPU handoff

Inputs: `longagent/capture-197/validation.json`, `captures/longcode/set`, canonical `final.state.bin` (197,314 rows, SHA-256 `f6933ca9bba785b22732aa2937e1ee40518c43d75cd732cfcdf8a1ec98705a0a`). The capture’s own graded turn-2 answer passed. Both prompts and turn-1 output IDs match the original dense reference, but turn-2 emitted IDs diverge after six. Positions **197,232 and 197,235** are the first and last captured reasoning predictors in the matching emitted prefix. The comparison uses their actual device `.out` vectors from the instrumented dense replay, not uncaptured vectors from the original run. STRSESS checksums and overlapping native Q8_0 rows were checked. This is one trajectory and two correlated queries; neither token-level sparse quality nor run-time performance is measured.

**Dense device parity:** 384 head samples (2 positions × 8 visits × 24 heads), relative L2 p50 `6.37e-07`, p99 `3e-06`, max `4.99e-06`; absolute L2 max `1.4e-05`. Current turn starts at 197,186, within the mandatory 1,024-row tail at both positions.

**Control:** mixed200’s held-out endpoints 199,112/199,119 (same count). Exact repeats of full aligned 128-row token strings at any earlier nonoverlapping token offset: **46/1541 (2.99%)** in the near-cap final snapshot, versus **1,491/1,555 (95.88%)** in the mixed seed. Substantial differences between these trajectories are expected; mixed is not equivalent.

| Set / visit | Old mass p50 / p95 | Last 1,024 mass p50 / p05 | Sink mass p50 / p95 |
| --- | ---: | ---: | ---: |
| near197 / all | 62.34% / 96.76% | 35.55% / 2.34% | 0.00% / 4.03% |
| near197 / 5 | 87.71% / 99.81% | 12.06% / 0.05% | 0.00% / 14.12% |
| near197 / 11 | 59.51% / 94.87% | 40.49% / 5.13% | 0.00% / 0.05% |
| near197 / 17 | 69.09% / 94.01% | 27.62% / 5.83% | 0.01% / 0.78% |
| near197 / 23 | 60.01% / 88.89% | 39.99% / 11.09% | 0.00% / 0.08% |
| near197 / 29 | 58.78% / 90.46% | 41.17% / 9.37% | 0.01% / 0.55% |
| near197 / 35 | 56.39% / 91.12% | 43.40% / 8.88% | 0.00% / 0.14% |
| near197 / 41 | 81.97% / 97.50% | 17.87% / 2.50% | 0.00% / 0.73% |
| near197 / 47 | 49.91% / 93.95% | 39.48% / 3.28% | 0.25% / 63.35% |
| mixed200 / all | 99.90% / 100.00% | 0.08% / 0.00% | 0.00% / 3.30% |

## Sparse sensitivity and calibrated transport floor

Pins are exact last-1,024-plus-current-turn, rounded outward to whole kept pages. Total budgets include pins. Context mass excludes the sink as a denominator component; old-kept is the fraction of each head’s normalized old mass actually retained. The sparse output renormalizes fetched scores with sink and uses the captured gate. Shared top mass is the ideal sum of the three heads’ exact masses; Quest shares a max bound, while the centroid is per-head. All transport rows charge the **real within-group union**, 272 bytes/row/KV group. Cross-group union in an all-head physical layout can only increase payload; it was not measured. `Cold` assumes all rounded pins are resident and all other selected rows must be fetched; it is not a cache-miss prediction.

The optimistic milliseconds divide cold bytes by the prior **27.0–28.3 GB/s** bulk pinned H2D medians on RX 6700 XT / PCIe 4.0 ×16 in [the transfer measurements](README.md#системная-цена-выгрузки-в-ram). These rates do not cover CPU gathering, fragmentation, index traversal, or attention. Earlier per-copy 34 KiB direct pages were about 3.8 GB/s; P256 group pages are 68 KiB and remain unmeasured. Selector metadata is a separately modeled FP16 centroid/minmax scan over eligible old pages; not charged to transfer.

Full dense reference: **3,274.4 MiB** logical read per query across eight visits, **3,257.4 MiB** cold excluding the 1,024 rows, or **120.7–126.5 ms** under bulk copy alone.

| Set / Page / Budget / Selector | Context mass p50 / p05 | Old kept p50 | Gated L2 p50 / p95 | Read / cold MiB | Bulk lower-model ms | Selector metadata MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| near197 / 128 / 1,024 / recent only | 35.55% / 2.69% | 0.00% | 48.03% / 103.97% | 18.9 / 0.0 | 0.00–0.00 | 0.0 |
| near197 / 128 / 8,192 / per-head centroid | 93.01% / 44.41% | 83.58% | 6.54% / 49.45% | 218.0 / 199.1 | 7.38–7.73 | 23.9 |
| near197 / 128 / 8,192 / per-head top mass | 95.59% / 48.40% | 89.16% | 4.49% / 40.46% | 225.7 / 206.8 | 7.66–8.03 | 0.0 |
| near197 / 128 / 8,192 / shared top mass | 95.30% / 47.73% | 88.23% | 4.74% / 43.82% | 135.8 / 116.9 | 4.33–4.54 | 0.0 |
| near197 / 128 / 8,192 / shared Quest | 67.10% / 14.70% | 32.60% | 21.78% / 71.19% | 135.8 / 116.9 | 4.33–4.54 | 47.9 |
| near197 / 128 / 32,768 / per-head centroid | 95.74% / 59.71% | 89.90% | 4.13% / 29.13% | 837.1 / 818.2 | 30.32–31.78 | 23.9 |
| near197 / 128 / 32,768 / per-head top mass | 97.96% / 68.33% | 95.01% | 1.85% / 20.53% | 897.4 / 878.5 | 32.55–34.12 | 0.0 |
| near197 / 128 / 32,768 / shared top mass | 97.66% / 64.68% | 94.08% | 2.27% / 22.19% | 543.8 / 524.9 | 19.45–20.38 | 0.0 |
| near197 / 128 / 32,768 / shared Quest | 91.24% / 42.13% | 81.59% | 6.88% / 40.51% | 543.8 / 524.9 | 19.45–20.38 | 47.9 |
| near197 / 256 / 1,024 / recent only | 35.55% / 2.69% | 0.00% | 48.03% / 103.97% | 18.9 / 0.0 | 0.00–0.00 | 0.0 |
| near197 / 256 / 8,192 / per-head centroid | 92.79% / 43.76% | 83.29% | 6.77% / 50.95% | 210.6 / 191.7 | 7.10–7.45 | 12.0 |
| near197 / 256 / 8,192 / per-head top mass | 95.29% / 45.46% | 88.01% | 5.00% / 46.04% | 209.8 / 190.9 | 7.07–7.41 | 0.0 |
| near197 / 256 / 8,192 / shared top mass | 95.04% / 44.24% | 87.30% | 5.12% / 49.11% | 133.7 / 114.8 | 4.25–4.46 | 0.0 |
| near197 / 256 / 8,192 / shared Quest | 64.20% / 11.19% | 24.61% | 22.37% / 73.74% | 133.7 / 114.8 | 4.25–4.46 | 23.9 |
| near197 / 256 / 32,768 / per-head centroid | 95.57% / 57.63% | 89.41% | 4.38% / 30.23% | 831.1 / 812.2 | 30.09–31.54 | 12.0 |
| near197 / 256 / 32,768 / per-head top mass | 97.56% / 63.76% | 93.75% | 2.45% / 22.45% | 898.2 / 879.3 | 32.58–34.15 | 0.0 |
| near197 / 256 / 32,768 / shared top mass | 97.27% / 61.36% | 92.89% | 2.79% / 23.96% | 541.7 / 522.8 | 19.37–20.30 | 0.0 |
| near197 / 256 / 32,768 / shared Quest | 90.97% / 37.47% | 81.08% | 7.69% / 43.10% | 541.7 / 522.8 | 19.37–20.30 | 23.9 |
| mixed200 / 128 / 1,024 / recent only | 0.09% / 0.00% | 0.00% | 76.03% / 131.00% | 18.3 / 0.0 | 0.00–0.00 | 0.0 |
| mixed200 / 128 / 8,192 / per-head centroid | 49.70% / 5.80% | 46.86% | 25.49% / 84.85% | 209.7 / 191.4 | 7.09–7.43 | 24.2 |
| mixed200 / 128 / 8,192 / per-head top mass | 79.06% / 23.73% | 78.34% | 11.96% / 62.60% | 222.2 / 204.0 | 7.56–7.92 | 0.0 |
| mixed200 / 128 / 8,192 / shared top mass | 75.11% / 18.05% | 74.42% | 13.93% / 65.08% | 135.1 / 116.9 | 4.33–4.54 | 0.0 |
| mixed200 / 128 / 8,192 / shared Quest | 12.83% / 0.73% | 11.37% | 42.46% / 88.71% | 135.1 / 116.9 | 4.33–4.54 | 48.3 |
| mixed200 / 128 / 32,768 / per-head centroid | 82.04% / 32.47% | 81.88% | 8.97% / 41.73% | 789.1 / 770.9 | 28.56–29.94 | 24.2 |
| mixed200 / 128 / 32,768 / per-head top mass | 94.43% / 52.49% | 94.19% | 3.54% / 25.92% | 856.0 / 837.8 | 31.04–32.54 | 0.0 |
| mixed200 / 128 / 32,768 / shared top mass | 93.34% / 49.00% | 93.22% | 4.09% / 29.10% | 543.1 / 524.9 | 19.45–20.38 | 0.0 |
| mixed200 / 128 / 32,768 / shared Quest | 43.82% / 16.61% | 42.72% | 21.47% / 54.68% | 543.1 / 524.9 | 19.45–20.38 | 48.3 |

Per-head data: `results/longcode-heads.csv`, `results/longcode-pages.csv`; scan/wiring provenance: `results/longcode-run.json`. CPU wall time is not a device timing. Existing GPU captures, validation, and other experiments were left untouched.
