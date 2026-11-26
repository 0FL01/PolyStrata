# DFlash2 selector codebooks on the Ryzen 7 5700X

**Use BF16 as the exact-storage control.** On one Zen 3 hardware thread, a prefaulted BF16 mmap needed **3.90 µs p50 / 5.84 µs p99** for the whole seven-position correction on repeated real selector inputs, before host/device handoff. It matched all **784** BF16 oracle choices. Q8_0 saves **91.58 MiB** per pair, but changed **5/784** choices; Q4_0 saves **140.42 MiB** and changed **66/784**. These are draft proposals, not emitted target tokens. See [native placement](NATIVE.md) for the separate GPU comparison and measured handoffs.

## What was measured

The selector looks up **7 predecessor rows** (the next read depends on the last pick) and **112 successor rows**, computes `dot(A[prev] * g[t], B[candidate])` in FP32, adds each unary score, applies `30*tanh(score/30)`, then finds the greedy pick. T1 additionally materializes all probabilities. This is the full correction on the chosen weight representation, not quantized activations, a reduced candidate set, or an approximate search. Timing excludes attribution of taps, the draft forward pass, LM head, top-16, gate projection, data transfers and target verification. There is no dense codebook expansion.

Main run: Ryzen 7 5700X, `taskset -c 7`, GCC 16.2.1 `-O3 -mavx2 -mfma -mf16c`, Linux 7.2.15 (4 KiB pages). 200 warmups then 2,000 measured walks per cell; no measured steady-state page faults in the main run. GGML's existing decoder serves the quantized formats; FP16/BF16 use vector expansion and each dot uses AVX2/FMA. Times below characterize these kernels, not the fastest possible kernel for each quant. Synthetic sets use 512 scattered hot IDs, Zipf(1.07) over permuted IDs, or uniform across 200,058 IDs. “Real” repeats 112 fixed blocks from seven private DFT1 captures (16 evenly spaced generated positions each from English reasoning, Python/Rust/C++, tool calls, JSON and agent trajectory). It is a cache-warm replay of those inputs, **not** live/model-wide throughput. Greedy, p90 and all raw timings are in the logs.

### All converter codebook types, holding the draft fixed

Times are µs per block, T1. Hot and Zipf are p50; random and real are p50 / p99. Source BF16 and supplied F16/Q8_0/Q4_0 tables below use their original GGUFs. Other rows are selector-only GGUFs obtained from the same BF16 donors with unweighted GGML quantization. Precision uses the same real BF16 candidates/gates/unaries for every row. The relative correction RMSE forces the oracle's previous ID so it isolates score error; picks and TVD use each arm's own sequential choices, so a differing predecessor can amplify a local error. TVD is on the fixed top-16 at T=1.

| A and B format | B/row | MiB / pair | Hot | Zipf | Random | Real | Picks / 784 | Corr. rel. RMSE | Mean TVD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 512 | 195.37 | 4.29 | 5.50 | 12.38 / 17.31 | 3.90 / 5.84 | 784 | 6.7e-8 | 1.1e-7 |
| F16 (`Q4_0-mixed`) | 512 | 195.37 | 4.32 | 5.87 | 14.67 / 29.01 | 3.90 / 5.87 | 784 | 6.7e-8 | 1.1e-7 |
| F32 | 1024 | 390.74 | 5.76 | 7.69 | 15.51 / 21.11 | 5.13 / 6.80 | 784 | 6.7e-8 | 1.1e-7 |
| Q8_0 | 272 | 103.79 | 7.06 | 7.82 | 13.91 / 19.84 | 6.64 / 9.28 | 779 | 0.44% | 0.00512 |
| Q6_K | 210 | 80.13 | 27.28 | 28.01 | 32.10 / 40.26 | 26.87 / 34.28 | 782 | 1.66% | 0.00676 |
| Q5_K | 176 | 67.16 | 8.78 | 9.69 | 15.37 / 22.62 | 8.30 / 10.91 | 740 | 2.84% | 0.04751 |
| Q5_0 | 176 | 67.16 | 24.61 | 25.21 | 28.76 / 35.86 | 24.32 / 31.35 | 764 | 3.51% | 0.02440 |
| Q5_1 | 192 | 73.26 | 25.27 | 26.01 | 29.78 / 36.38 | 24.87 / 30.07 | 764 | 3.00% | 0.02467 |
| Q4_K | 144 | 54.95 | 7.36 | 7.71 | 10.31 / 13.93 | 7.05 / 9.07 | 766 | 5.59% | 0.03186 |
| Q4_0 | 144 | 54.95 | 7.16 | 7.54 | 9.46 / 13.12 | 6.83 / 9.43 | 718 | 7.35% | 0.07563 |
| Q4_1 | 160 | 61.05 | 7.48 | 7.79 | 9.48 / 12.92 | 7.12 / 9.78 | 721 | 6.19% | 0.07387 |
| IQ4_NL | 144 | 54.95 | 49.46 | 49.91 | 53.83 / 65.27 | 48.98 / 56.41 | 740 | 5.89% | 0.05820 |
| IQ4_XS | 136 | 51.90 | 33.24 | 33.57 | 35.71 / 41.26 | 32.72 / 37.90 | 735 | 6.10% | 0.06336 |

- The independent scalar dot on decoded rows matched all 784 choices for **each stored pair**, with max raw correction error **7.63e-6**. Against the mmap row gathers in the Python/ggml bridge BF16 oracle, max post-cap score delta **8.58e-6**. BF16→F16 is not universally exact: expanding both to FP32 changed **159/51,214,848 predecessor** and **2,732/51,214,848 successor** words (max abs **2.98e-8**), despite matching all trace choices. The supplied and bench-generated F16/Q8_0/Q4_0 codebook bytes match; the entire Q8_0 and Q4_K-mixed codebook pairs also match. BF16's exact weight storage and sparse FP32 reduction do not imply bit-identical scores to a GPU reduction or identical stochastic draws.
- The legacy executable's AVX2 fused kernel is different. Pinned to SMT siblings `6,14`, its one-thread T1 hot/Zipf/random medians were F16 **3.40/4.64/10.72 µs** on the first pass and Q8_0 **6.14/7.02/14.54 µs**, Q4_0 **6.71/7.89/12.22 µs** on the complete pass. The complete pass also recorded **1.61 ms** F16 random median. The old program interleaves 2T/4T spin loops, does not record faults in the steady cells, and labels process-RSS growth (including query buffers) as its residency check. Do not use these timings as a direct A/B or infer an F16 steady-state regression. The replacement isolates single-thread work and reports faults and rounded table coverage separately.
- `MADV_DONTNEED` on the private mapping drops page-table entries, **not** the file's page cache. BF16's first 200 random bursts: **29.23 / 255.40 µs p50/p99**, 2,480 minor and zero major faults; supplied Q4_K-mixed: **27.15 µs / 2.80 ms**, with 37 major faults. Do not report this as a controlled cold-drive test. `MADV_WILLNEED` + touching each covered page took BF16 **3.07 ms**, F16 **2.69 ms**, Q8_0 **1.07 ms**, Q4_0 **0.41 ms**, *after* the first-touch bursts. Codebook `mincore` spans the table bytes rounded to pages (a common boundary may be counted twice); BF16/Q8_0/Q4_0 were **195.375/103.797/54.953 MiB**, matching approximate file-RSS growth. Full-pair `mlock` failed with `ENOMEM`: soft `RLIMIT_MEMLOCK` is **8 MiB**. Pinning all weights is not necessary for warm reads; prefault at load and measure p99 under real expert/LM-head contention and memory pressure before enabling this path.

### Extended selector mixes, predecessor / successor

There are 16 times as many successor as predecessor gathers. If compressing only one table, keeping the successor in an expansion-friendly format can save CPU time, but the accuracy comparison is not one-dimensional. Mixed runs are separate passes and share the host with other work; the BF16 bookend control was **7.18→4.41 µs** on the repeat. Ranges below span the two observed p50s where repeated. No draft backbone was changed in this test.

| A / B | Pair MiB | Real T1 p50 µs | Picks / 784 | Forced corr. rel. RMSE | Free-walk mean TVD |
|---|---:|---:|---:|---:|---:|
| Q8_0 / F16 | 149.58 | 4.04 | 780 | 0.35% | 0.00296 |
| F16 / Q8_0 | 149.58 | 6.50 | 779 | 0.27% | 0.00494 |
| Q6_K / BF16 | 137.75 | 9.41–11.53 | 783 | 1.43% | 0.00274 |
| BF16 / Q6_K | 137.75 | 46.74–49.10 | 781 | 0.82% | 0.00671 |
| Q4_K / BF16 | 125.16 | 7.18–7.52 | 770 | 4.63% | 0.02117 |
| BF16 / Q4_K | 125.16 | 7.58–11.75 | 759 | 3.14% | 0.03299 |
| Q4_K / Q8_0 | 79.37 | 7.71–11.41 | 762 | 4.64% | 0.02728 |
| Q8_0 / Q4_K | 79.37 | 8.15–11.80 | 761 | 3.15% | 0.03036 |

First differences (0-based trace block and draft position): supplied Q8_0 and Q4_0 first disagree at `1:4` (179518 vs BF16 167380); symmetric BF16/F16/F32 have none. Q6_K / BF16's single difference is at `30:6` (62 vs 7). The complete table, including first disagreements and the teacher-forced score checks, is in `matrix.log` and `mixed.log`. Candidate worth testing if host memory is constrained: Q8_0 predecessor / BF16 successor, same bytes as Q8_0/F16. It is not an exact-selector substitute. For complete-model mixed profiles, **Q4_0-mixed backbone + BF16 books** preserves the exact codebook values at the supplied mixed profile's byte cost; **Q4_K-mixed backbone + BF16 books** adds 91.58 MiB wherever placed relative to the supplied Q8_0 books. Neither change restores BF16 projections/gates or establishes whole-drafter quality.

## VRAM and transfers: estimates tied to measured logs

Raw tensor payloads from the five supplied GGUFs, excluding the GGUF metadata, any borrowed target weights, draft KV, workspace and allocation alignment. The slot column is core-Q4_0 equivalents at ~4.272 MiB each, rounded up; a Q4_1-down slot is ~4.430 MiB, so exact displacement depends on the mix of resident layers. Payload arithmetic is **not** a load/eviction test. `kTail` and block alignment add a few hundred bytes per expert, and current VMM arena backing moves in 64 MiB pieces.

| Supplied profile | Backbone only (MiB) | Tables (MiB) | Fully GPU weights (MiB) | Core-slot equiv: backbone only / fully GPU | Incremental equiv for tables alone |
|---|---:|---:|---:|---:|---:|
| BF16 | 748.02 | 195.37 | 943.39 | 176 / 221 | 46 |
| Q8_0 | 397.60 | 103.79 | 501.39 | 94 / 118 | 25 |
| Q4_0 | 210.71 | 54.95 | 265.66 | 50 / 63 | 13 |
| Q4_0-mixed | 227.45 | 195.37 | 422.82 | 54 / 99 | 46 |
| Q4_K-mixed | 259.15 | 103.79 | 362.94 | 61 / 85 | 25 |

Add **40 MiB FP16 or 21.25 MiB Q8_0** for 5 × 2048 rows of draft K+V at 8 × 128, about 10 or 5 more core-slot equivalents before rounding and workspace. Placement still needs a real HIP comparison at fixed capacity and fixed map. For scale, on the same RX 6700 XT/Tideglass, `$R/runs/{taps-en_reasoning,oracle-2k-w1-frozen,dspark-log-en_reasoning}/wire.log` reported **1228 slots / 5261 MiB**, **1244 / 5330**, and **1122 / 4806** at READY, respectively, with free VRAM **2033 / 2040 / 2033 MiB**. The historical DSpark arm used a different drafter and SWA allowance, so its 106-slot drop vs the tap arm is not a measured codebook cost. At the tap run's payload budget, Q4_0-mixed with FP16 KV costs about **63 core slots** with host codebooks or **109** with device codebooks; the ~46-slot delta is ~3.7% of 1228. At a tighter **475-slot** near-cap W1 control ([existing adaptive-KV run](../../../docs/goals/2026-11-25-adaptive-global-kv/production.md#native-selected-and-packed-backings-at-near-cap)), the same 46 slots are ~9.7%. None of these slot fractions predicts hit-rate or ID/s; startup packing, routing, live policy and reserve determine the realized change.

- **Host exact correction with GPU top-16/gate:** a packed handoff is **8,064 B D2H** (`7 × (16 × (4B ID + 4B unary) + 256 × 4B gate)`), assuming the CPU already knows the anchor, then **28 B H2D** for seven chosen IDs (or **448 B** for a full 7×16 probability payload if required). At the historical **27.0–28.3 GB/s** bulk pinned **H2D** rate, using that rate for both directions only as a payload-rate yardstick gives ~**0.29–0.30 µs** round trip. It omits synchronization, command launch and CPU overhead; no matched D2H or selector round trip was measured here.
- **Host gather, device correction:** host needs the candidate IDs and can fetch all potentially used rows before the device walk: at most **97 predecessor** (anchor + 6×16) and **112 successor** rows, **107,008 B** BF16 / **56,848 B** Q8_0. That is **3.78–3.96 / 2.01–2.11 µs** of bulk H2D payload, not including readback, packing, launch and synchronization. The smaller **119-row** count only applies after the path is known; requesting each dependent predecessor separately adds handoffs. Uploading both full codebooks each round at that bulk rate alone would cost **7.24–7.59 ms** BF16 or **3.85–4.03 ms** Q8_0, versus a one-time upload for resident tables.
- **Reality check:** existing same-host RX 6700 XT PCIe 4.0×16 isolated [HIP transfer receipts](../../../docs/goals/arch-decode-recon/placement/README.md) measured **15.87 µs p50** even for **4 B H2D** through `hipStreamSynchronize`. A pinned **4,478,976 B** expert took **173.72 µs**, rising to **322.65 µs** with separate CPU packing. Those are measured, but neither direction of the 8 KiB/ID selector handoff was timed at its actual size. If the CPU already receives shortlist data for scheduling/sampling, a separate copy may be avoidable; otherwise fixed handoff cost can dominate the 4 µs CPU walk. Compare real GPU resident, batched host handoff and no-extra-handoff placement before choosing a deployed default. Bulk-rate source: [adaptive-KV transfer measurements](../../../docs/goals/2026-11-25-adaptive-global-kv/README.md#системная-цена-выгрузки-в-ram).

## Reproduce

Run from the repo root. Build against a matching ggml-base static archive (the existing `build-poly-hip` archive is sufficient; no HIP runtime is called). The scratch reference and bridge can be replaced with the maintained [reference harness](../reference/). Use the original, unmodified captures and set BLAS/OMP to one thread. The binary and private fixtures/logs stay outside the repo. The selector-only GGUFs use ~1.3 GiB of disk and cannot be loaded as draft models.

```bash
R=$HOME/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon
C=$R/cpu-codebooks-repeat
M=$HOME/.lmstudio/models/unsloth/Inkling-2-Mini
D=$M/DFlash2
B=$D/Inkling-2-Mini-DFlash2-BF16.gguf
S=$D/Inkling-2-Mini-DFlash2-Q4_0-mixed.gguf
Q=$D/Inkling-2-Mini-DFlash2-Q8_0.gguf
mkdir -p "$C/generated"
taskset -c 7 g++ -std=c++20 -O3 -mavx2 -mfma -mf16c -Wall -Wextra -Werror \
  -Ithird_party/llama.cpp/ggml/include rdna2-rnd/dflash2/codebooks/bench_codebooks.cpp \
  build-poly-hip/ggml/src/libggml-base.a -pthread -o "$C/bench_codebooks"
taskset -c 7 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python rdna2-rnd/dflash2/codebooks/export_queries.py --ref "$R/dflash2_ref.py" \
  --bridge "$R/libggml_qbridge.so" --target "$M/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf" \
  --model "$B" --out "$C/bf16.dfq1" \
  "$R/taps/"{en_reasoning,code_python,code_rust,code_cpp,tool_calls,json_struct,agent_traj}.dft1 \
  > "$C/export.log"
taskset -c 7 "$C/bench_codebooks" pack "$B" "$C/generated" > "$C/pack.log"
taskset -c 7 python rdna2-rnd/dflash2/codebooks/audit_weights.py "$D" "$C/generated" > "$C/audit.log"
taskset -c 7 "$C/bench_codebooks" bench "$B" "$C/bf16.dfq1" "$D/"*.gguf \
  "$C/generated/"*.gguf "$S:$Q" "$Q:$S" > "$C/matrix.log"
P=$C/generated/q6_K.gguf
K=$C/generated/q4_K.gguf
taskset -c 7 "$C/bench_codebooks" bench "$B" "$C/bf16.dfq1" \
  "$P:$B" "$B:$P" "$K:$B" "$B:$K" "$K:$Q" "$Q:$K" > "$C/mixed.log"
taskset -c 7 "$C/bench_codebooks" bench "$B" "$C/bf16.dfq1" \
  "$B" "$P:$B" "$B:$P" "$K:$B" "$B:$K" "$K:$Q" "$Q:$K" "$B" > "$C/mixed-repeat.log"
```

Use `-` instead of `bf16.dfq1` for synthetic timing only. The logged run is `$R/cpu-codebooks/`, including `export.log`, `pack.log`, `audit.log`, `matrix.log`, `mixed*.log`, and the legacy 3-format benchmark logs. A two-block smoke with the maintained reference/bridge also matched 14/14 oracle choices (`maintained.log`). None of these checks measures full-drafter quality, target acceptance or the interaction with running HIP experts.
