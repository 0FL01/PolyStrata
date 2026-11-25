# K256 + physical V4-refit integration gate

Private scratch only. `make.py` hotlinks three translation units, reuses the codec's allocator and disk-session objects without rebuilding them, verifies the unmodified link against Experiment v2, and checks that its inputs did not change. No repository or sibling builds are modified.

`STRATA_INKLING_SELECT=256` requires refit, native Q8 K on gfx1031, and the supported context size. After the one native RESTORE, the index builds two 64-row centroids per complete 128-row K page. Index updates take only newly completed pages. Selection is restricted to W1 token generation in an append-only request; fresh prompt batches and the single-row prompt predictor use dense mixed attention. The restore transfers data and builds the index but does not run attention. With selection off, refit keeps dense mixed attention everywhere; with refit and selection both off the existing native path is used. The codec's SAVE/second RESTORE restrictions are unchanged.

The ordered page gather keeps page zero, 256 scored old pages for each K/V group, and the contiguous recent tail. Both passes into the mixed kernel use absolute positions for K, V scales and relative bias. Cold V codes take half the native stride; hot codes subtract the compacted cold half from the absolute byte offset. Each page is wholly hot or wholly cold because the seal is page-aligned. The original global sink, gate, chunk combine, and optional Q8_1 output are used. This is approximate: V4 refits values and the page selection skips most old key/value rows during W1. Physical VMM backing nevertheless retains **all K** and all V in their respective encodings.

One guarded greedy pair (`speed199`) and one guarded historical board pair (`board197`) use the same ELF, isolated fresh processes, original checkpoints and controls, a 2,048 MiB requested reserve and an actual 4,900 MiB expert+KV budget. The latter uses `STRATA_RECON_CACHE_MIB=4832` because the v2 startup knob adds 68 MiB already mapped for growing KV. Existing exact dense results are quality/ID references, not matched timing legs. No repeats. `dispatch.json` checks phase counts against the harness and checks actual mapped bytes. The one-visit CPU replay uses device-picked pages, the saved native prefix, and two captured appended rows to verify the selected FP32 attention. The 199K TG report can exclude the first inter-token interval from both arms because only one interval contains the probe.

## Results

Both pairs completed on one Radeon RX 6700 XT (`gfx1031`), Tideglass-Inkling-2-Mini-Unbound-v2.1, ELF `a59d7b404fbee4ebe5b14bedbd45d97f91c5ca8608065c45d866e1d3d5eb905a`. Both guards reported no stop reasons (prior boot memory-fault warnings only), and minimum sampled free VRAM was 3,103.9 / 3,116.2 MiB respectively.

| Check | Dense refit | K256 + refit |
|---|---:|---:|
| 199,104 greedy IDs | 65 emitted | 65/65 match |
| 199K TG, excluding the probe interval | 20.75 IDs/s | 26.01 IDs/s |
| 199K TG, all 64 steps; p90 | 20.72 IDs/s; 54.44 ms | 25.95 IDs/s; 45.61 ms |
| 199K actual global KV backing vs native cover | 2,550.81 vs 3,320.31 MiB | 2,550.81 vs 3,320.31 MiB |
| 199K post-restore process resident VRAM | 7,916.56 MiB | 7,964.86 MiB |
| 197,021 historical board | pass, 1 tool call | pass, 1 tool call |
| 197K weighted TG; time to correct grade, excluding restore | 20.20 IDs/s; 11.22 s | 22.04 IDs/s; 9.91 s |
| 197K actual global KV backing vs native cover | 2,558.81 vs 3,320.31 MiB | 2,558.81 vs 3,320.31 MiB |
| Expert arena on both pairs | 536 slots, 2,304 MiB | 536 slots, 2,304 MiB |

- The placements retain real codec VMM savings of **769.5 / 761.5 MiB** respectively. The selector adds a 48.8125 MiB index and 0.195 MiB scratch, not a further KV allocation saving. The existing independent native `residency/199104-live-exact` control had 371 slots under its own ~4,900 MiB combined cap (different binary and mapping geometry, not a same-arm throughput comparison).
- At 199K, every one of 512 W1 global visits hit the selected mixed kernel and none fell back to dense; the eight global visits of the singleton prompt went dense. Mean K/V rows touched per group and global visit were 35,037.5 vs 199,137.5 (82.4% fewer), with 256 of ~1,538 eligible old pages selected. The CPU replay at absolute row 199,105, visit 5, used the actual GPU-picked page lists for all eight groups, both cold and hot encodings, the unshifted scales, causal relative bias, sink and gate. Max absolute FP32 error was `3.10e-6`. Both greedy arms also match the 17 overlapping IDs of the independent adaptive native Q8 reference, and all 65 IDs of the separate held-arena Q8 reference.
- At 197K the selected arm issued 296 + 888 selected W1 global visits with no dense W1 fallback. The 101/45 fresh rows across the user and tool suffixes were all dense mixed (1,144 batched row-visits + 24 singleton global visits), exactly as in refit. Selected W1 used 82.3% fewer K/V rows. Both arms passed all four checks with the same correct answer, no tool errors, and no retries. The exact Q8 board reference also passed. Against both refit and exact, the selected trajectory first differed at zero-based output index 1 in turn 1 (`200008` vs `1055`, frame); the histories differ thereafter. Completion lengths were 47/117 vs 38/112, so lower time to grade combines a higher token rate with a shorter trajectory. This is one task and seed, not an accuracy-rate estimate.
