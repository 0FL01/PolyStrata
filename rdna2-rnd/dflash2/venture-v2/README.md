# Venture decode: price a committed row

Tideglass Inkling 2 Mini, one RX 6700 XT (`gfx1031`, 12 GiB), Ryzen 7 5700X, 8 expert workers, HIP Release. Starting point: `1798843d`, including the transaction from `384251dc`. The installed W1 and the starting epoch worker are **different binaries**. W1 with a loaded drafter and the same residency policy is the verifier correctness control; W1 without a drafter at the same physical reserve is the product control.

## Where the time goes

The controlled greedy test restored 65,792 rows, evaluated 2,048 more to warm the Q4_0 drafter, and emitted 49 IDs (48 fed). Both arms used stable prefill, dense Q8 with compact SWA, the same profile seed and 5,200 MiB combined expert/KV cap. Epoch 4 with 64 swaps: **30.35 → 41.58 IDs/s** loaded W1 → K3. Wire ID-gap p90 **45.70 → 98.35 ms**. Both ended with the same 49 IDs, profile and **13,318 / 18,816** routed picks on GPU. No placement repair; 48 of 51 verifier rows kept. The last emitted ID need not be fed. The cap and staging are controls, not service settings.

On a separate K3 kernel trace, 13 rounds used **215.4 ms** of generic resident Q4_0 expert slots, **146.2 ms** of target Q8 attention leaves, **108.1 ms** of native multi-column GEMV and **29.9 ms** of Q6 head kernels (target and draft together). Median summed draft kernels per ask: **6.12 ms**; shared Q6 head 1.11, FFN projections 1.58 and draft attention scores/values 0.65/1.61 ms. These are dispatch durations, not mutually exclusive wall-time savings. In a separate timed worker the new admission join took **141.24 ms for 567 copies**, over 12 boundaries. The RAM packer, PCIe transfer, event waits and publication are included in the join; it is not isolated link time.

The same accounting on a real three-turn coding task at 64K, sampled seed 42, changes the ranking. Below, `charged rows` counts only speculative rounds, excluding separate W1 calls. Verification time ends when target logits are available. Round time also includes host sampling, keeping rows and appending committed taps. The CPU timer overlaps GPU work.

| Epoch / K / swap limit | Ask rounds | Charged rows kept/evaluated | Ask / verify / whole round, ms per kept row | Reasoning / total IDs | Decode IDs/s | Wire p90, ms | Time to correct, s |
|---|---:|---:|---|---|---:|---:|---:|
| 4 / 0 / 64 | 0 | — | — | 307 / 517 | 30.98 | 38.22 | 25.27 |
| 4 / 3 / 64 | 155 | 502 / 582 | 2.25 / 21.72 / 24.26 | 307 / 517 | 41.22 | 82.74 | 20.82 |
| 8 / 0 / 32 | 0 | — | — | 339 / 565 | 31.09 | 38.41 | 26.36 |
| 8 / 7 / 32 | 112 | 549 / 741 | 1.53 / 22.42 / 24.21 | 339 / 565 | 41.25 | 126.96 | 21.89 |

Each arm passed 118/118, made the two required tool calls and stopped naturally. Each loaded W1/spec pair matched **every ID on each turn**. Between policies the first difference was output 13 on turn 1; later prompts are not parity controls. Epoch 4 K3 resumed 65,796 / 68,019 / 68,151 rows and read 2,162 / 94 / 102 new rows. Its TTFTs were 5.85 / 1.22 / 1.18 s, after RESTORE. On the much more predictable 49-ID wire test, W6 K5 took 50.06 IDs/s (64 swaps), W8 K7 48.63–52.88 (64 swaps) and 51.08 (32 swaps). All 40 / 42 guesses at W6 / W8 matched. Optimizing this suffix alone would select wide windows for the wrong reason.

## Admission by bytes

One incoming expert costs **4,478,976 B** (three Q4_0 matrices), or **4,644,864 B** with the Q4_1 down. Admission unpublishes a same-size victim; it doesn't buy another slot. One scratch worker retained the recency half-life, size classes and `incoming >= 2 × victim + 2` threshold. It budgeted actual gate/up/down transfer bytes (including refills). The priced variant compared estimated net CPU time saved per byte among eligible pairs. It used an 80 µs marginal miss proxy, a 32-row discounted forecast and a fixed copy price. The price is an input, not an online measurement: W1 and verifier can make identical decisions from committed history.

| Epoch 4 K3, same 64K fixture | Slots copied | Payload MiB | Exposed batch join, ms | GPU picks / 18,816 | Decode IDs/s | Wire p90, ms |
|---|---:|---:|---:|---:|---:|---:|
| Stock, 64 entries | 567 | 2,430.32 | 163.70 | 13,318 | 39.40 | 102.73 |
| 144 MiB, stock ranking | 376 | 1,610.82 | 98.16 | 12,570 | 44.86 | 92.04 |
| 144 MiB, 32 µs/MiB | 286 | 1,224.65 | 84.96 | 11,995 | 41.37 | 94.76 |
| 144 MiB, 64 µs/MiB | 94 | 402.15 | 40.04 | 10,740 | 40.05 | 101.33 |

In this ledger run the cheap unpriced cap beat both modeled prices. The highest price avoided 2.0 GiB of transfer but sent 2,578 more picks to CPU. Do not minimize wait or maximize hit rate separately. At 0 and 64 µs/MiB, matched W1/K3 pairs also matched ordered admission fingerprints at every boundary, final profile, route counts and IDs. Separate A/B work on publication keeps ranking and membership fixed instead of charging fewer admissions.

**When does a miss actually stall verification?** A separate instrumented run put a HIP event after the last resident expert operation and a second on the same stream when `cpu_experts` returned. Elapsed time was read after the next router wait, not by inserting a new blocking sync. This measures an uncovered interval on the target stream, not all CPU compute or a hardware idle counter. Across the visits with misses, W4 summed to **519.2 ms** in the host call, **333.1 ms** between device timestamps (569 recorded visits); the W3 round added **35.5 / 24.1 ms**. On W4, with 1–4 host picks out of 32 the median gap stayed at ~5 µs. At 6/8/12/16 host picks it was **147/442/754/1,334 µs**. A hit at a CPU-limited layer is worth more than a hit where the GPU is already slower. Correlated picks, weight reuse and placement affect these conditional medians: they are not a per-miss invoice.

## What ports from four gfx906 cards

The following comparisons are against [Strata `563779dd`](https://github.com/0FL01/Strata/tree/563779dd), not against an earlier Qwen/MI50 benchmark. Donor timings are **the donor's**; only the right-hand evidence was measured on RDNA2.

| Strata commits | Mechanism and donor evidence | Local difference / test |
|---|---|---|
| `159abe3e`, `9ad07d5e` | Restore W1 partition/reduction order, share weights across repeated routed picks, stage Q8_1 activations. 64K T4 42.3 → 37.2 ms and T8 74.6 → 63.5 ms with the latter bundle. | Local verifier already uses W1 attention geometry and per-input FP32 router. Only **GPU** picks can share slot weights. In archived 64K W4 route tapes, 34.6% of mixed / 43.4% of code GPU picks repeated within the same visit. CPU/GPU merge order must not change. |
| `15f2d3d6`, `93f8387d` | Bidirectional draft, TP2 private head, fused K/V preparation, multiplicity dispatch, cold projection skip. Donor first ask 8.27 → 3.46 ms with its fast-path bundle. | We already share the resident Q6_K head and have FP32 draft rings and separate committed taps. Buying a second large head would evict experts. Cold skip is about unneeded old **prompt** rows, not skipping target verification. |
| `eb48e024` | LDS Q8 projection, shared FP32 GEMV, depth mix + control, fused draft quantization and separate verify pricing. Donor HTTP window 20.33 → 19.07 s. | Local W1 groups front projections and fuses RMS/quant; the W4 front has more launches. Target FP32 router arithmetic is not interchangeable with the donor's subgroup reducer. Control-in kernels together occupied only 4.49 ms of the ~1.24 s K3 trace. |
| `563779dd` | Cooperative combine exponentials and expert load hoisting; donor HTTP 19,309.3 → 19,309.0 ms, despite a better kernel suite. | Stock wave32 already scalar-loads uniform chunk metadata. Our probe extends beyond the donor's 128 chunks: 158-chunk W4 packed combine **24.33 → 13.85 µs**, 32-chunk **6.32 → 5.65 µs**, byte-exact FP32 and Q8_1. Stock combine was only **5.05 ms** in the whole trace. |

Four **output** rows share Q8 activations in the local W1 Q4 kernel; that is not sharing a weight across verifier **input** rows. Extending those four-row kernels alone to all verifier slots cut the same 1,207 Q4 dispatches from 215.4 to 147.8 ms. GU used 40 VGPR/1,536 B LDS and down 32/1,024, versus 24 VGPR and 512/1,024 B in the generic path; no scratch spills. The local four-row GU uses wave32 `v_dot4c_i32_i8`, with a fixed partial sum order. Both untraced pairs **lost**: 41.31 → 37.16 and 40.15 → 38.46 IDs/s. Do not ship that dispatch change on this evidence.

## Full-history compact KV

[LexiEngine's layout](https://github.com/W61k3r/LexiSuite-Releases/wiki/LexiEngine-Inkling-2) rotates Q/K with WHT128 and V by two WHT64s before q4_0 packing. Its [RX 7900 XTX results](https://github.com/W61k3r/LexiSuite-Releases/wiki/Benchmarks) use a different engine and GPU; they do not isolate the effect of rotation. We compared codecs without dropping global rows: eight global **logical visits** and an unchanged recent Q8 tail. The 40 exact local SWA rings are not counted as full history.

- **Accuracy, fixed inputs.** Six recorded predictors on three real 64K/197K code/reason traces, all eight visits and 24 query heads: 1,152 head observations per arm, not independent tasks. The native CPU replay reproduced captured Q8 GPU output to max gated relative L2 `3.62e-6`. With Q8 K, p95 relative L2 was **6.40%** for stock refit V4, **5.51%** raw q4_0 V and **4.57%** WHT64 q4_0 V. With K and V both compressed: raw/raw **17.12%**, rotated/rotated **13.63%**, the latter with max **54.93%**. K4 changes softmax: rotated K kept the same top attention row on 86.55% of these head observations. These are **attention** ranks, not predicted vocabulary tokens. No static head mask follows from six queries.
- **Bytes.** At 200,000 rows and 197,888 sealed cold positions, global payload Q8/Q8 = **3,320.31 MiB**, K8/V4 = **2,547.31**, K4/V4 = **1,774.31**. These include scales; they are not measured savings of physical VRAM or a promise of 1,546 MiB of available slots. Rotating a cold numerator also requires returning to the native basis before mixing hot rows and applying the gate.
- **RDNA2 kernel.** Synthetic full-history causal attention; same signed-nibble refit for the packed arms, **without WHT**. After a 256 MiB disjoint flush, medians of 15 graph replays including combine/Q8_1 output. At 65,536 positions W1 Q8/Q8 / K8/V4 / K4/V4: **0.497 / 0.427 / 0.409 ms**; W4 **1.242 / 1.266 / 1.294 ms**. At 199,070: W1 **1.380 / 1.186 / 1.109**, W4 **3.565 / 3.660 / 3.759 ms**. W4 times cover all four queries at fixed 128-chunk geometry; they are not four W1 timings. CPU oracle and future-row masking passed. Four-wave shaders had 175 / 170 / 166 VGPRs, no LDS or spills; packed K/V emitted 56 static signed bit extracts versus 28 with only V packed. Fewer bytes helped one query, not this four-query shader.
- **Share the unpack, not just the bytes.** Repeated at W1's actual partition (158 / 160 chunks at the two spans), with four waves cooperating to stage decoded K and V for four queries in 32 KiB LDS. One group × 32 keys gave full W4 **1.188 / 3.435 ms** vs dense **1.235 / 3.574** and nonsharing K4/V4 **1.286 / 3.780**. Two groups × 16 keys lost to dense at both spans despite identical bytes and LDS. One-group ISA: 164 VGPRs, no scratch spills, 1,200 static instructions vs 1,801 nonsharing. The shared leaf had max FP32 error `4.41e-8` against its **own codec** CPU oracle and passed future-V mutation. This does not qualify K4 model quality or include WHT or a service speedup. Receipts: `kv-leaf/share/{build,guard-s15,results.md}`.

K256 attention prunes pages and LoopCD changes either coda state or scoring. Neither is an exact replacement for this dense target verifier. See the [earlier real-task counterexamples](../../../docs/goals/2026-11-25-adaptive-global-kv/production.md) and [LoopCD state/quality experiment](../../../docs/goals/native-donor-recon/contrast.md). Neither is needed to test V-only precision.

## Physical-budget and agent controls

With no fixed expert/KV arena and the same 2,048 MiB physical reserve, the restored 128K wire continuation emitted **27.34 / 34.24 / 40.87 IDs/s** (installed W1 / E4 K3 64 swaps / E8 K7 32 swaps). ID-gap p90 was **44.42 / 116.64 / 187.84 ms**. Near 199K the ordering changed: **23.13 / 32.14 / 30.58 IDs/s**, p90 **50.79 / 134.59 / 239.14 ms**. All used stable prefill and unusually predictable suffixes. At 199K the guard observed ~2,022 MiB minimum free device-wide in each leg, worker peak RSS **18,234 / 18,482 / 18,482 MiB** and initial resident expert counts **1,424 / 1,321 / 1,321** *before* growth. Do not read RSS as private RAM or a sampled device-wide high-water mark as worker allocations.

In the near-200K **free-running coding** task we restored 194,973 rows and replayed exactly 2,048 archived rows before the original question at 197,135. At fixed 5,200 MiB, seed 42, installed W1 / E4 K3 / E8 K7 made two successful tool calls with zero retries. Installed W1 failed the final answer format (618 output IDs); the speculative policies passed in **30.87 / 33.65 s** with **443 / 444 IDs**, at **25.66 / 22.26 IDs/s**, ID-gap p90 **136.11 / 224.58 ms**. First TTFTs: 10.80 / 10.26 / 10.41 s, including 2,162 fresh prompt rows, not the separate 3–5 s RESTORE. The two later requests appended only 94 and 102 rows. A different trajectory is not acceleration of the same answer.

**Separate production-protocol billing task**, ordinary prefill at the physical reserve, T=1, no forced CPU reuse. Each cell is time to a correct answer (s) / pooled **delivered** SSE gap p90 (ms). Each arm made two real tool calls and scored 118/118.

| Seed | Installed W1 | E4 K3 / 64 | E8 K7 / 32 | DSpark |
|---|---:|---:|---:|---:|
| 42 | 11.58 / 25.79 | 9.56 / 60.76 | 8.53 / 97.67 | 8.49 / 59.32 |
| 79 | 14.24 / 24.32 | 11.39 / 57.13 | 12.10 / 93.93 | 15.11 / 55.60 |

These are not ID-matched work: for example E4 produced 365/514 output IDs across its two seeds and DSpark 307/696. E4 reduced time to correct versus installed W1 on these two runs, but not its delivery gap and not DSpark on both seeds. Alongside the [earlier same-reserve results](../production/README.md), this is insufficient evidence for a new interactive default.

## Receipts and controls

Raw logs, models, prompts, token IDs, checkpoints, temporary workers and disassemblies stay on the profiling host. `$V` below is `~/.cache/opencode-tmp/opencode/polystrata-runs/venture-v2`. No service token or private task data are required for the aggregate reader.

```sh
python -B rdna2-rnd/dflash2/venture-v2/wire_report.py --control "$V/control-w1" "$V/control-e4"
python -B "$V/admission/read-admission.py" "$V/admission/price-w1" --same-policy "$V/admission/price-k3"
python -B "$V/agent/report.py" "$V/agent/controlled64"
```

`trace4/trace`, `cost-e4`, `agent/controlled64`, `admission`, `donor/{REPORT.md,isa,guard-4x158,guard-4x32}`, `trace-slots`, `kv/{wiki-source,evidence}`, `kv-leaf/{build-v2,guard-s15}` contain the underlying receipts. `wire`, `agent/bench.py` and the component build scripts record the actual launch arguments. Each GPU leg takes the common `dflash2-recon/gpu.lock` and runs under `run_guarded.py --warn-amdgpu-faults`. A boot-history memory-fault warning is not a new failure. Use the [production harness](../production/README.md) with independent writable output/profile paths to repeat.
