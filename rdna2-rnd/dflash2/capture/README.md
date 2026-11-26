# Native replay at the same target placement

The paired capture records the four device tensors DSpark actually consumes in a W1 process.
It follows the same token IDs and records DSpark proposals at each anchor. A DFlash2 replay reads
those exact tensors and is scored against the same following IDs. Capturing in a process without
DSpark at the same arena cap is not equivalent: 701 vs 699 resident experts changed feature bytes
by row 40 on the smoke input. The aligned replay below used one 699-expert placement at zero swaps.

## RX 6700 XT (`gfx1031`), Tideglass v2.1, K=7

Values are **1 + mean consecutive draft matches** on full-width anchors, not generated tok/s.
Every profile uses the same 2,614 DFT1 rows and 1,356 scored anchors. `agent_traj` here is a
scripted prefix; the separately [graded billing replay](../trajectory/README.md) uses real reads.

| Workload | Anchors | DSpark | BF16 | Q8_0 | Q4_0 | Q4_0-mixed | Q4_K-mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English reasoning | 248 | 4.41 | 5.35 | 5.36 | 5.12 | 5.31 | 5.32 |
| Python | 248 | 3.23 | 4.04 | 4.05 | 4.04 | 4.09 | 3.99 |
| Rust | 248 | 3.36 | 3.83 | 3.85 | 3.74 | 3.91 | 3.87 |
| C++ | 248 | 3.03 | 3.71 | 3.75 | 3.67 | 3.74 | 3.64 |
| Tool calls | 59 | 6.46 | 6.05 | 6.05 | 5.92 | 6.02 | 6.05 |
| Structured JSON | 248 | 4.69 | 5.50 | 5.50 | 5.38 | 5.54 | 5.50 |
| Scripted agent turn | 57 | 6.16 | 6.09 | 6.09 | 6.00 | 6.14 | 6.09 |
| **Pooled anchors** | **1,356** | **3.963** | **4.622** | **4.635** | **4.524** | **4.650** | **4.601** |
| **Simulated closed loop** | — | **3.386** | **4.084** | **4.118** | **3.931** | **4.139** | **4.084** |

The simulated walk advances by the accepted prefix plus one from each chosen anchor. It does not
measure request latency or reconstruct target arithmetic under a different physical placement.
Small differences on overlapping anchors do not establish a general ranking.

Receipts: `$R/same-target-replay/{paired,native}`, with `$R` the private `dflash2-recon` directory.
`native/comparison.json` contains survival curves, accepted-prefix histograms and walk results.
`build.py`, `run.py` and `matrix.py` give the capture/build/replay controls (see `--help`). Both
runners revalidate matching complete receipts in `--resume` mode; do not copy over failed legs.
