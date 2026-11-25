# Production Hybrid Decode — RX 6700 XT

Status: active. Source: user's 2026-11-25 implementation request; [measured candidates](hybrid-decode.md), [KV evidence](hybrid-kv.md), [SWA contract](../2026-11-25-exact-swa-memory-compaction.md). The preceding documents describe prototypes, not the installed engine.

## Finish line

| Outcome (source) | Acceptance and primary evidence | State |
|---|---|---|
| R1 — exact SWA including `input_[0]` reuse | Opt-in production path, smaller resident allocation including staging, same greedy IDs and canonical state on the same decisions. Compare to installed ELF and feature-off, including wrapped prefill and session continuation. | Verified (`4ec4bb35`) |
| R2 — K256/P128, Q8 K / symmetric V4 and whole experts | Independent opt-ins and dense Q8 reference. Guarded model runs exercise growth/sealing, long append, prefix overwrite/rollback, SAVE, RESTORE and repeated RESTORE. Account for actual backing, transient space, index and slots. Never export lossy data as a native Q8 session. | Verified as opt-in, not selected as the default |
| R3 — RESTORE pack and tool-return latency | Integrate exact AVX2 refit against scalar; measure compact suffix prefill including Flash16 H2 against the same codec, TTFT and success. | In progress |
| R4 — new critical path | Profile the integrated binary on real calls; distinguish quant, gate/up, SiTU, down, reduction, CPU scheduling, residency and upload overlap. Test improvements against A/A and keep only repeatable wins on the critical path. | In progress |
| R5 — qualified production | Compare free agent trajectories against the installed configuration and iso-policy controls: grade, reasoning IDs, tools/retries, per-turn TTFT, TG, p90 and wall to correct answer. Release HIP, main commit + push, versioned deployment with rollback; verify running SHA, configuration, loopback health, authenticated API and continued session. | Pending |

Envelope: Inkling production engine and its direct session/CLI consumers, documentation, local isolated benchmarks and service configuration. RDNA2 gfx1031, Ryzen 7 5700X, real Tideglass weights. Do not change Qwen, infer quality from greedy TG, replace dense defaults, lower safety thresholds, or publish model weights, prompts, sessions, credentials or raw dumps.

Work on the smallest unresolved outcome. Use the research order as hypotheses, not instructions to promote all candidates. Preserve separate evidence for exact layout, approximate policy and changed placement. Advance on measured results; do not promote a failed or untested candidate.

## Checkpoint

Start: main `2cd563f6`. `/proc/599683/exe` and the on-disk Release build both matched the configured `strata-poly-50effa3a`, SHA-256 `a8cfc09f1c14d3f17c64920683bce7ec15544fe65f060b5bd379b0fb6d322da3`. Loopback health before trials: loaded Inkling, 200000 context.

Fresh native official control on the 197021-ID session, T=1, no lookup/draft, same profile seed per process, 8 workers, 2048 MiB reserve, 64 swaps. All four made two useful tool calls without retries. Wall starts after RESTORE; the length failure has no time to solution.

| Task/seed | Grade | Reasoning IDs | Tool-return TTFT (s) | Decode ID/s | Step p90 (ms) | RESTORE / wall (s) |
|---|---|---:|---|---:|---:|---|
| coding/42 | 118/118 | 124 | 1.21 / 1.19 | 22.26 | 50.67 | 3.17 / 20.18 |
| reasoning/42 | 4/4 | 444 | 1.56 / 1.07 | 24.49 | 50.31 | 3.10 / 27.82 |
| coding/79 | 118/118 | 162 | 1.19 / 1.18 | 23.09 | 48.59 | 3.09 / 21.17 |
| reasoning/79 | length | 511 | 1.50 / 1.06 | 25.27 | 47.37 | 3.10 / 27.67 to limit |

Guard: lowest free 2004.9 MiB; no stop. Separate ROC kernel trace of that installed ELF at 199104 mixed + one new ID, 32 W1, greedy: p50 step 50.99 ms, 8 global attention visits 10.63 ms, 40 SWA visits 0.79 ms, routed GPU kernels 3.18 ms. Gap from last expert/router producer to tail kernel: 20.41 ms/token (includes host submission and other gaps; **not** a direct CPU-late measurement). This is traced cadence, not speed qualification. Local receipts: `global-kv-freeagent-confirm/production-start-2cd563f6/` and `production-hybrid-20261125/official-roc/` under the handoff's private artifacts root.

Release HIP SWA candidate `2fa372677761a78173e3c2fd895dae7678a6644737d0c50407000780ffa7f6a7`.
Host CTest `poly_inkling_swa_layout_test`: 1,908 scenarios. On gfx1031 the Release objects linked to the real-engine parity driver matched **111/111** records across the DSpark run with Q8/batch 2048 (same 285 resident experts, 21,008 short/miss expert calls). Compared IDs, proposals, all FP32 logits in the tape, canonical state, accepted full KV, canceled/replayed prefix, sliced restore and continuation. FP16/plain/batch 128, no drafter, dedicated stage, packed-state flag on: **35/35** records, same 274 slots; no guard stops in either pair. These are fixed-decision checks, not measurements of live scheduling. The short authenticated HTTP coding control on the installed server solved 118/118 after two real tool calls; turns 2 and 3 reused 325 and 469 tokens.

Fresh 65,536-ID prefill (no RESTORE): installed vs compact with funding off, 1,356 whole slots at READY in both. Per-process VRAM at READY / after generation was **169.19 / 170.01 MiB lower** with compaction. Both produced the same 17 greedy IDs, 4,483/6,528 expert hits. TTFT 76.95 → 76.43 s, inter-token p90 40.74 → 41.53 ms on 16 intervals. One pair; the established gain is memory, not cold TTFT or cadence.

CPU screen on the 5700X: real mixed and independent code call replay, 74,999,808 output words checked. For 3–8 host experts, gate/up including SiTU: 263 / 255 µs p50, down including ordered reduction: 142 / 132 µs. Input quantization ~0.59 µs; worker batch-end tails below 0.3 µs. Tile prefetch, affinity and reduced worker counts did not beat repeatability on both inputs. Causal whole-slot replay kept 99.63% of the measured baseline backend assignments: marginal-cost weighting buys <1 ms/token *modeled*, not an engine speedup. Prefer testing increased whole-slot capacity within the free-memory floor. Local probes `real-expert-phase.35ZU6C` and `residency-causal.MdkrPj`.

199104 mixed RESTORE, natural VMM, same SWA Release ELF, stock → compact+fund → compact+fund → stock: **23.03 / 23.48 / 23.59 / 23.28 ID/s**, p90 **51.06 / 50.19 / 49.16 / 50.27 ms**, 97/97 greedy IDs matched. READY slots 1356 → 1395, hits 22090 → 22679 out of 37248. Full RESTORE 3.15 / 3.71 / 3.16 / 3.21 s. The actual installed control measured 23.29 ID/s. A small positive cadence effect on this repeated prompt does not qualify funding for an agent.

On-device native disk interchange passed: installed, new ELF flag off, and compact layouts matched all 165 greedy IDs in each of two gates, including naturally grown append, repeat RESTORE and cross-ELF continuation; saved state bytes matched. Short / near-cap process VRAM fell by 171.0 / 169.3 MiB. On the live-map four-task screen the paired installed arm solved 1/4 vs compact-only 3/4; notably even the installed repeat differs from its earlier 3/4. Coding/42, the only shared success, took 27.95 → 20.76 s with 288 → 141 reasoning IDs but **22.55 → 22.20 ID/s**. Not a same-work speedup or an accuracy guarantee. Receipts: `hybrid-product-validation.lE5tEh/RESULTS.md`.

The exact host gate/up paired Q4_0 dot was not integrated: 76,032 single-core output words matched, but in the complete 8-worker calls, 22,201,344 output words matched and the median 0.21 / 0.52 ms per position savings on mixed / code were below 4.31 / 1.78 ms A/A drift. Neither a hot leaf nor a fitted residency replay establishes engine throughput.

Next checkpoint: qualify the disjoint V4 hot/cold storage, K256 selector, funded slots and tool-return reader on the integrated Release. The SWA code is in main; the full hybrid deployment is not qualified by the SWA results.

## Integrated hybrid, before the wide compact Flash reader

The combined Release with all approximation controls off matched all 111 DSpark/Q8 reference records again. The host reader passed 205 file checks and 98,560 block refits. On gfx1031, a synthetic allocation refusal **before calling HIP** on the second plane of a real 199104-ID native import proved the admission rollback: all 47 new handles (187.31 MiB) released, byte-identical compact SAVE of the previous 4422-ID session and continuation in the same worker. The arena stayed 2496 MiB smaller as intended. Physical VRAM after recovery agreed within 4 KiB; minimum free 2025.5 MiB. No GPU fault was injected. `production-hybrid-20261125/fault-injection.nqCfAZLA/report.json`.

The first live seal exposed a real difference: 117/32,768 FP16 scale words were one ULP lower on-device than in host capture. Making the FP32 intermediate round before converting to half fixed it. The old leaf fails the scale witness; the fixed one passes. On a natural seal across position 2176, all **524,288 V4 code bytes and 32,768 scales** matched. A canceled 12,000-token read resumed at 2177 with 8/8 matching greedy IDs, also 8/8 after another conversation overwrote a parked prefix. A real tool return reused 197021 rows and appended 211. Separate checks grew naturally from 35064 through a 2048-row append, imported legacy 131328/197021/199104 native sessions, saved compact, restored repeatedly in both directions and trimmed/regrew to exactly the 4900 MiB limit without a dense V mapping. At 199104 the compact file was **2,763,091,432 B** (native **3,559,694,824 B**); compact restore/resave matched SHA-256. Dense refused compact without damaging the old session. See `production-hybrid-20261125/lifecycle-report.md` for exact builds and legs; the fixed seal run did not repeat all disk sizes.

**A real prefill regression:** from empty to 65536 Tideglass mixed IDs, same 2048 batch and 2048 MiB reserve as the installed run, K256 + V4 + funded SWA with the bounded mixed reader took **152.43 s** to first ID vs installed **76.95 s**, though subsequent TG was **30.82 vs 28.99 ID/s** and 17/17 greedy IDs matched. This is why faster W1 does not qualify the wide reader. The natural-growth run went through seal and wrap, not a checkpoint import.

129015-ID archived receiver-board task with a genuine new tool result, seeds 79/42: installed and both integrated arms solved 4/4 fields with one read each, no retry. Installed → K256 + V4 (5412 MiB): TG 28.03 → 31.05 and 27.85 → 30.56 ID/s; wall after RESTORE 9.906 → 9.331 and **7.506 → 8.033 s**. On seed 42, reasoning grew 64 → 78 IDs; tool-return TTFT 0.875 → 1.094 s. Against V4 alone at the same cap the selector saved 0.84 / 1.04 s to grade. This does not qualify a blanket 128K policy. `production-hybrid-20261125/agent129-production/abba2/report.txt`.

Four matched-placement SWA-only trajectory pairs now match **1,955/1,955 sampled IDs**. Two pairs used the same nonzero map and two deliberately used zero GPU expert slots to isolate arithmetic. These are not four timing samples of the live placement policy.

### Native, selected and packed backings at near-cap

One mixed 199104-ID RESTORE, one new input and 96 inter-token intervals per untraced arm. RX 6700 XT, 8 CPU workers, 2048 MiB reserve, no draft/lookup. All 97 greedy IDs matched on this repeated text. The capped integrated arms use compact SWA and fund whole experts; the cap is not the installed memory policy.

| Arm | Mapped global KV incl. index/hot ring | Mapped arena / slots | W1 ID/s | Step p90 | RESTORE / first ID after RESTORE |
|---|---:|---:|---:|---:|---:|
| Installed `50effa3a`, native | — | automatic | 23.29 | 49.99 ms | 3.17 / 0.143 s |
| Integrated, 5412 MiB, dense native | 3320.31 MiB | 2048 MiB / 475 | 21.56 | 53.06 ms | 3.25 / 0.133 s |
| Integrated, 5412 MiB, K256 + V4 | 2622.96 MiB | 2752 MiB / 640 | 28.68 | 41.95 ms | 3.37 / 0.186 s |
| Integrated, 4900 MiB, K256 + V4 | 2622.96 MiB | 2240 MiB / 521 | 26.15 | 44.65 ms | 3.63 / 0.207 s |

In a separate 32-step ROC trace at 5412 MiB, sums per token (p50): eight global visits **3.52 ms** (installed **10.63**), 40 SWA visits **0.85 ms**, routed GPU kernels **3.12 ms**. GPU-producer-to-tail gap **19.97 ms**. That gap includes host scheduling and is not a measurement of the CPU dots alone. Receipts: `production-hybrid-20261125/{released-roc,w1-partial.md}`.

Union of all timestamped kernels and memory-copy intervals between vocabulary softcaps, original → hybrid: median kernel activity **23.72 → 16.61 ms**; copy activity outside kernels **1.05 → 0.97 ms** (p90 **3.55 → 4.11**); neither a kernel nor a copy running **25.27 → 24.71 ms**. These traces report H2D events, not their byte counts. Removing all non-overlapped DMA would not remove the whole quiet interval. The separate medians need not sum to median step latency. `production-hybrid-20261125/w1_transfers.py`.

The integrated K256 reader matched all **14,336 selected page IDs** from an independent oracle, including ties, the first eligible span, sort-bucket boundaries and real Tideglass Q/K at 199105. Max absolute output error vs FP64 over the *selected* rows was 3.56e−6; this does not compare selection to dense attention. Local receipt: `k256-oracle.t9g18n`.

On free 197K trajectories at the same 5412 MiB cap, both dense native and K256 + V4 passed **2/4**, vs 3/4 in the starting installed run. Hybrid lost reasoning/42 and recovered coding/79. In the sole common success (coding/42), rate rose 20.10 → 27.34 ID/s but output grew 336 → 422 IDs and time to correct rose 20.71 → 21.36 s. Both post-tool TTFTs regressed in **every** cell. All legs made two useful reads without retries. Do not use this capped configuration as the service default.

K256 **without V4**, on natural placement with funded SWA, passed 4/4 in the same 197K cells. Relative to the starting installed control: coding/42 **20.18 → 21.33 s** (124 → 185 reasoning IDs), coding/79 21.17 → 20.60 s, reasoning/42 27.82 → 22.96 s, reasoning/79 length failure → pass in 23.50 s. TG 24.22–30.84 ID/s, p90 42.03–46.90 ms. At 129K the same-ELF dense vs K256/seed 79 comparison doubled reasoning (62 → 124 IDs), with wall **19% longer** despite 27.99 → 31.63 ID/s. Keep native dense Q8 selectable. Receipts: `hybrid-product-validation.lE5tEh/RESULTS.md`, `production-hybrid-20261125/agent129-production/native2/report.txt`.

Held-out 197K seed 117 **rejects automatic near-cap K256-only**. With native Q8, compact SWA and funded slots in both arms, dense passed both; K256 passed coding in 25.76 vs 26.94 s but failed to stop naturally in reasoning (dense pass in 25.71 s). K256's final visible JSON passed the content check; hitting the same 512-ID limit is still a completion failure. Reasoning generated 402 → 474 reasoning IDs, 24.54 → 30.05 ID/s and p90 50.47 → 42.60 ms. All four legs made two successful tool calls. `production-hybrid-20261125/agent129-production/held117-r1/report.txt`.

Removing the cap changed the result. With the original compact mixed reader, 197K reasoning/42,79,117: same-ELF dense passed 2/3, K256 + V4 passed 3/3. Warm time to correct 25.00 → 24.06 s, 24.79 → 23.38 s, length limit → pass in 26.26 s. First post-tool TTFT worsened from 1.51–1.52 s to 2.83–2.87 s, with the same two useful reads and no retries. Minimum sampled free 1955 MiB. This is a reasoning-only screen, not evidence about uncapped coding or the separate H2 suffix reader.

Local reproduction: `production-hybrid-20261125/bench.sh LABEL ELF --steps 96` runs the 199104 native import plus one fresh ID. Pass `--fresh 65536` for cold prefill, `--roc` for a separate profiled leg, and repeat `--flag NAME=VALUE` for the controls in [INKLING2.md](../../INKLING2.md#optional-global-attention-and-storage). Each label must be new. The wrapper serializes across the GPU lock, saves and restores the service's entry activity, clones the same expert profile and runs `rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults`. `w1_report.py LABEL...` compares same-fixture untraced legs and reports first differing greedy ID, p90, hits, physical backing where logged and sampled free VRAM. These local helpers require the private model and snapshots listed in [hybrid-evidence.md](hybrid-evidence.md); the repository does not ship those files.
