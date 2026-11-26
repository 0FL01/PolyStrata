# DFlash2 exact HIP attention experiment (RDNA2)

This experiment changes the attention calculation's work distribution, not its arithmetic. The stock
kernel stays in the short-context path (absolute position below 512). On longer inputs, one score kernel
shares each K load across four grouped-query heads and schedules 128-key tiles independently. A second
kernel preserves the stock max, exp, denominator, V FMA stream, and final sum order, using aligned
four-element V loads. The extra score workspace is 2.125 MiB. The 2,048-token ring, absolute RoPE,
per-query sliding-window lower bound, and all eight noise rows are unchanged.

The DSO is a scratch build, not a production change. [`build.py`](build.py) builds a stock control and
an exact-attention variant from one snapshot. It requires a new output directory and never overwrites
the shared DSO. It leaves the selector, FFN/projection kernels, target Q6_K head and top-16 selection
alone. The experimental DSO exports the same four `df2_*` live ABI symbols; the measurements below
use the standalone replay and are **not live target verification or tokens/sec**.

## RX 6700 XT (`gfx1031`) results

Times in ms, stock → exact, medians of five process pairs. Both long captures exercise the full
sliding window. See the procedure below for what the first-proposal measurement includes.

| Capture and drafter | First proposal | Later proposals | All proposals | Round including append |
|---|---:|---:|---:|---:|
| 2k Q4_0 | 22.20 → 20.18 | 9.27 → 7.68 | 9.79 → 8.18 (−16.4%) | 10.07 → 8.46 |
| 2k Q4_K-mixed | 22.38 → 20.16 | 10.75 → 9.16 | 11.21 → 9.59 (−14.4%) | 11.52 → 9.91 |
| 64k Q4_0 | 21.83 → 19.13 | 9.26 → 7.67 | 9.77 → 8.12 (−16.8%) | 10.04 → 8.40 |
| 64k Q4_K-mixed | 22.73 → 20.56 | 10.74 → 9.16 | 11.22 → 9.62 (−14.3%) | 11.53 → 9.93 |
| Short Q4_0, stock short path | 13.73 → 13.84 | 5.57 → 5.56 | 5.89 → 5.89 | 6.17 → 6.17 |

**Parity:** 1,400/1,400 greedy IDs (1,225 at full SWA) and byte-identical JSONL including hidden
states, ordered top-16s, unary scores, and selector results. Tested all five GGUF profiles at 2k,
Q4_0 and Q4_K-mixed at 64k, and Q4_0 on the short control. The guard recorded no stop and only
historic fault warnings; the minimum sampled free VRAM was 9,938 MiB.

At full SWA, `rocprofv3` measured the following median GPU stage times over 15 proposals after dropping
the first proposal, each with seven head rows. Append timeline includes launch gaps, but not the upload
of DFT1 taps. The wall times above come from separate runs without the profiler.

| Stage | 2k Q4_0 | 2k Q4_K-mixed |
|---|---:|---:|
| Attention computation, 5 layers | 3.87 → 2.25 ms | 3.87 → 2.25 ms |
| └ Exact score / exact softmax + V | 0.64 / 1.61 ms | 0.65 / 1.60 ms |
| Attention projections | 0.84 → 0.84 ms | 1.34 → 1.34 ms |
| FFN gate + up / other FFN projections | 1.14 / 0.44 → 1.14 / 0.44 ms | 1.77 / 0.76 → 1.77 / 0.76 ms |
| Shared Q6_K target head + top-16 | 2.04 + 0.14 → 2.04 + 0.14 ms | 2.04 + 0.14 → 2.04 + 0.14 ms |
| One-row append, kernels / device timeline | 0.14 / 0.24 → 0.14 / 0.24 ms | 0.18 / 0.28 → 0.18 / 0.28 ms |

The roughly 1.6 ms reduction in GPU attention time accounts for the proposal gain. Attention is now
about the same cost as the target head plus top-16, rather than the dominant stage. Most remaining
attention time is in softmax and V accumulation, not score generation. FFN projections already cost
more than total attention in Q4_K-mixed. This points to the head, projections, or the V stage as
better next measurements than top-16 or append; it does not establish that any are bandwidth-bound.

**Artifacts:** `$O/exact-final-v1/exact/libstrata-dflash2.so` and matching `dflash2-replay`, with the
control in `$O/exact-final-v1/stock/`. Raw traces, guard logs, and `summary.json`, `stages.json`, and
`parity.sha256` are in `$O/exact-final-receipts-v1/`. The original `$R/libstrata-dflash2.so` was not
overwritten (SHA-256 starts `435f5e515e03f539`).

## Reproduce

From the repository root, with `build-poly-hip` already built for gfx1031 and `rocprofv3` on `PATH`:

```sh
R=$HOME/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon
M=$HOME/.lmstudio/models/unsloth/Inkling-2-Mini
O=$R/native-opt
B=$O/exact-new-build
D=$O/exact-new-receipts
python -B rdna2-rnd/dflash2/native/opt/build.py "$B"
flock -x -w 480 "$R/gpu.lock" python rdna2-rnd/scripts/run_guarded.py \
  --warn-amdgpu-faults --output "$D" -- \
  bash rdna2-rnd/dflash2/native/opt/bench.sh "$R" "$M" "$B" "$D"
python -B rdna2-rnd/dflash2/native/opt/summarize.py "$D"
python -B rdna2-rnd/dflash2/native/opt/trace_summary.py "$D"
```

The script uses the real `taps/code_python.dft1` and `long-taps/{2k,64k}/tap.dft1` captures. The latter
two start measuring at absolute positions 2,048 and 67,840, with a full 2,048-row real warmup in each arm.
It alternates the order of five fresh process pairs. Each timing process measures 25 seven-ID proposals;
the **first** measurement is immediately after real warmup (not a forced empty GPU cache), and **later**
is the mean of steps 2–25. Neither includes model upload, append, profiler overhead, or hidden readback.
Ordered IDs, candidates, unary scores, selector output and round-trip FP32 hidden-state dumps are checked
separately with `cmp`. The profiler also runs separately.

## Rejected variants

An earlier split-and-merge softmax reduced untraced Q4_0 proposal time at 2k from 9.74 to 7.12 ms,
but matched only 167 of 175 greedy IDs. The first difference was absolute position 2,055, proposal
7 (`13` → `153938`), and the largest relative hidden RMS error was 0.070. A fused, reduction-order
preserving two-head variant matched all bytes there but took 10.47 ms. Neither is in this build.
The receipts for those variants are under `$O/attention-ab-v1`.
