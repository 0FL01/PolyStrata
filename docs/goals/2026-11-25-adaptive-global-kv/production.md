# Production Hybrid Decode — RX 6700 XT

Status: active. Source: user's 2026-11-25 implementation request; [measured candidates](hybrid-decode.md), [KV evidence](hybrid-kv.md), [SWA contract](../2026-11-25-exact-swa-memory-compaction.md). The preceding documents describe prototypes, not the installed engine.

## Finish line

| Outcome (source) | Acceptance and primary evidence | State |
|---|---|---|
| R1 — exact SWA including `input_[0]` reuse | Opt-in production path, smaller resident allocation including staging, same greedy IDs and canonical state on the same decisions. Compare to installed ELF and feature-off, including wrapped prefill and session continuation. | In progress |
| R2 — K256/P128, Q8 K / symmetric V4 and whole experts | Independent opt-ins and dense Q8 reference. Guarded model runs exercise growth/sealing, long append, prefix overwrite/rollback, SAVE, RESTORE and repeated RESTORE. Account for actual backing, transient space, index and slots. Never export lossy data as a native Q8 session. | In progress |
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

Next: native disk interchange and free agent A/B with the SWA candidate. The full deployment remains unqualified.
