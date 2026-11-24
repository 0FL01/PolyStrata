# Fresh 2K and 64K decode profiles — CP-01 / CP-P3-02

**Deprioritize P3 as a standalone throughput change.** On the RX 6700 XT, the entire fused controller-input stage took **0.312 ms/token at 2K** and **0.323 ms/token at 64K**, including live normalization that bank hoisting cannot remove. If this stage vanished at zero cost and every removed nanosecond shortened the traced decode window, throughput would improve by only **0.86% at 2K** or **0.80% at 64K**. These are ceilings for removing the measured dispatch work, **not** measured improvements or untraced E2E bounds.

## What was measured

Nov 24, 2026: RX 6700 XT (`gfx1031`, 40 CUs), Ryzen 7 5700X, HIP Release engine source `c0dd0e52` (binary SHA-256 `877fbdfed412…`), Tideglass Inkling-2-Mini Unbound v2.1, Q8_0 K/V, 200K capacity, 2,048 MiB reserve, 8 CPU workers and 64 expert swaps. One 2,048-token prose prefix of the `mixed` fixture and 32 serial greedy decode steps (`temperature=0`, `seed=42`; no drafts or lookup). rocprofv3 1.3.0 / ROCm 7.13.0 traced HIP runtime, kernels and memory copies, without performance counters. The guard recorded only boot-time memory-fault warnings, no stop reasons, and a sampled free-VRAM minimum of 1,999.4 MiB. The engine loaded 1,176 of 3,072 routed experts into VRAM.

**Window verification:** The pipe reported 33 IDs (`length` stop, 2,048 prompt tokens read, zero reuse/drafts). Raw CSV inspection found 33 same-stream `softcap_kernel` → `__amd_rocclr_copyBuffer` pairs, each with both dispatches correlated to the same `hipGraphLaunch`. The first pair ends prefill; the following 32 copy-to-copy intervals end at the last decode logits. Their total (1,171.658 ms) agrees with the pipe’s first-to-last ID span (1,171.678 ms) and `DONE` decode timer (1,171.7 ms). In the window: 30,612 complete kernel dispatches, all on compute stream/queue 1; 752 traced H2D copy records, which do not expose byte counts. Re-running the CPU analyzer on the raw CSVs reproduced the saved JSON byte-for-byte.

| Decode window, mean over all 32 steps | ms/token |
|---|---:|
| Elapsed | 36.614 |
| Kernel dispatch union | 19.065 |
| H2D copy union | 4.348 |
| Kernel/copy overlap | 2.077 |
| Kernel-or-copy union | 21.335 |
| Neither traced kernel nor copy active | 15.279 |

Largest kernel groups: specialized Q4_0 expert slots **8.643 ms/token**, other projections **4.558** (including generic `native_mmvq_segments_kernel<8>` at **3.123**), LM-head Q6_K projection **1.213**, and attention decode/combine **1.036**. The specialized grouped Q4_0 dense kernel did **not** dispatch in this window, despite its enabled flag. One Q4_0 expert dispatch was recorded at **154.671 ms** inside a 189.139 ms step; these measurements include it. Its cause is not identified. The uncovered interval is not a CPU-expert-join or launch-overhead measurement.

## Controller cost and bound

Every step had 24 grouped Q/K/V and 24 single FFN controller sequences, each with one fused input, one matched rank-64 projection, and one fused output. That is **96 logical stream rows/token**, or **3,072** total, with no unmatched controller or LM-head sequences.

| Measured dispatch stage | Q/K/V (ms/token) | FFN (ms/token) | Total (ms/token) |
|---|---:|---:|---:|
| Fused input: bank loads/RMS/query/mix, message RMS and live RMS | 0.149 | 0.163 | **0.312** |
| Rank-64 projection | 0.303 | 0.126 | **0.429** |
| Fused gain/forget and controlled output | 0.295 | 0.133 | **0.428** |
| All three stages, union | 0.747 | 0.422 | **1.169** |

The input dispatched 1 workgroup for FFN or 3 for Q/K/V, 128 threads per workgroup. That is the launch shape, **not achieved occupancy**. The fused bank-only operations have no separate timestamps. HIP launch calls correlated to controller input account for **0.145 ms/token** of host API duration; all controller stages account for **0.432 ms/token**. Neither is exposed launch cost, and neither can be added to the device budget.

- **Dispatch-only E2E ceiling on this trace:** `36.614 / (36.614 − 0.312) − 1 = +0.86%`. Restricting the removal to the input stage’s **0.295 ms/token** not overlapping another recorded kernel or copy gives **+0.81%** under the same assumption. As a sensitivity check only, omitting the 189 ms step leaves 31.694 ms/token wall and 0.312 ms/token input, or **+0.99%**; the main result includes all steps.
- **Scale check:** A +25% throughput increase at this traced rate would require removing **7.323 ms/token** from wall time. Even deleting the **entire** 1.169 ms/token controller chain free of charge, including rank and output work that the [dependency analysis](../loop/loop-dependency-graph.md) shows cannot be bank-only hoisted, would give just **+3.30%** in the same calculation. Hoisting still has to compute and store the messages, preserve live normalization and FP32 order, and schedule the new work. The trace does not rule out a separate launch/fusion change, but it does not support booking those gains for P3.

**Matched comparison:** Config, binary, prompt and profile-seed hashes agree with the untraced `mixed-2048` leg, and the actual prompt CSVs match. **All 33 greedy IDs match** its prefix; first divergence: none. Over the first 32 intervals, traced vs untraced arrivals were **36.615 vs 44.766 ms mean**, **30.451 vs 28.975 ms median**. Their largest intervals were 189.183 and 518.475 ms, respectively. This is one noisy comparison, not a measured profiling overhead, an optimization A/B, or candidate accuracy evidence. The full 256-step untraced leg was 32.86 t/s; it is not the denominator of the traced ceiling.

## §5 disposition and evidence

**PASS** for matched fresh dispatch attribution; **not material on the tested leg** for P3 as proposed. Fused sub-operation costs, occupancy, and exposed launch/join costs remain unmeasured. Do not queue standalone P3 GPU prototypes on this evidence. Revisit only if bank prep can fold into an independently justified change that removes launches without changing rank arithmetic, or if a separate matched trace shows a larger budget. No optimization parity claim.

Private evidence under the [campaign directory](../research-manifest.md#точка-воспроизведения): `roc2k/{trace/,config.json,metrics.json,engine.log,controller-cost-breakdown.json}`, `guard-roc2k/{stop-reasons.json,warnings.json,vram.jsonl}`, and the untraced `mixed-2048/`. Reproduction of the analysis is CPU-only:

```sh
D="$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK"
python -B docs/goals/arch-decode-recon/profiler/analyze.py \
  --run "$D/roc2k" --untraced "$D/mixed-2048" | diff -u "$D/roc2k/controller-cost-breakdown.json" -
```

## 64K follow-up

Same hardware, build, config, profile seed, and profiler settings. This was a fresh 65,536-token `mixed` prompt and **16 serial decode intervals**, versus 32 at 2K. The guard had **no stop reasons**, 22 boot-time memory-fault warning records and no AMDGPU warnings during the run; its sampled free-VRAM minimum was **2,040.4 MiB** (1,536 MiB floor). The engine again loaded 1,176 of 3,072 routed experts into VRAM.

The 17 logits-copy markers were each paired with their softcap on the same stream and `hipGraphLaunch` correlation. The decode window was **654.239 ms**, versus **654.150 ms** from the first to last pipe ID and **654.3 ms** in `DONE`. It contained 15,285 compute dispatches on stream/queue 1 and 491 H2D copy records (no byte field). Every step had 24 grouped Q/K/V and 24 single FFN controller sequences with matching rank and output dispatches: **96 logical rows/token**, **1,536 total**, no unmatched sequences. The analyzer reproduced the saved 64K JSON byte-for-byte.

| Traced attribution (mean ms/token) | 2K · 32 steps | 64K · 16 steps |
|---|---:|---:|
| Elapsed | 36.614 | 40.890 |
| Kernel dispatch union | 19.065 | 17.638 |
| H2D copy union | 4.348 | 5.523 |
| Kernel/copy overlap | 2.077 | 2.910 |
| Kernel-or-copy union | 21.335 | 20.251 |
| Neither traced kernel nor copy active | 15.279 | 20.639 |
| Attention decode + combine | 1.036 | **4.630** |
| Other projections (excludes LM head and controller rank) | 4.558 | **4.592** |
| Specialized Q4_0 expert slots | 8.643 | 3.345 |
| Q6_K LM-head projection | 1.213 | 1.392 |
| Fused controller input | 0.312 | 0.323 |
| Rank-64 projection | 0.429 | 0.428 |
| Fused gain/forget and controlled output | 0.428 | 0.426 |
| All controller stages, union | 1.169 | 1.176 |

At 64K, attention and other projections were the largest kernel groups. Attention split into **4.313 ms/token** in `decode_q8_kernel<8>` and **0.317 ms/token** in `decode_combine_kernel`. The apparent drop in expert time is not a context speedup: 2K includes a 154.671 ms expert dispatch, and the two windows have different lengths, prompts and live placement. The uncovered time and traced HIP API durations do not establish CPU join or launch cost and must not be added to the busy unions.

**P3 boundary at 64K:** Entire fused input **0.323 ms/token**, or **0.310 ms/token** after excluding overlap with other traced device work. The same zero-cost removal calculation gives **+0.80%** (exclusive-only: **+0.76%**); deleting the entire controller, including rank and output that are not bank-only, would give **+2.96%**. These are traced dispatch-only ceilings. The fused trace cannot price the bank-only subset separately, and real hoisting still pays compute, scratch traffic and live RMS. The 64K trace does not change the decision to park standalone P3.

**Free-running greedy comparison:** **17/17 IDs matched** the untraced `mixed-65536` prefix; no first divergence. The prompt CSV and the config, binary and profile-seed hashes match; the pipe verified no reuse or drafts. The same 16 arrival intervals averaged **40.884 ms traced vs 36.437 ms untraced** (medians **37.145 vs 34.384 ms**). One short matched prefix does not measure profiler overhead, bound untraced E2E performance, or establish candidate parity. The full untraced 256-step rate was **32.34 t/s**, not the denominator of the traced ceiling.

Private evidence: `roc64k/{trace/,config.json,metrics.json,engine.log,controller-cost-breakdown.json}`, `guard-roc64k/{stop-reasons.json,warnings.json,vram.jsonl}`, and `mixed-65536/`. To reproduce the CPU check, substitute `roc64k` and `mixed-65536` in the command above.
