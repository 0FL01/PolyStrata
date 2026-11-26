# CPU expert reuse in the target verifier

The experiment shares Q4 weight decode across inputs of an already selected CPU expert. It does not select experts, change their placement, change the GPU path, or make the wide verifier exact. The runtime and defaults are untouched.

## What is compared

- `serial`: the linked Release `cpu_experts` once per row.
- `grouped`: the current Release multi-row `cpu_experts`. It already groups uses by expert pointer so the same weight row is nearby in cache. It still invokes the ggml dot once per input.
- `multi`: the same host stages, with one Q4 nibble decode and weight-scale conversion per block for two to four inputs, using AVX2/F16C. Each input keeps its own ggml AVX2 FP32 accumulators, block-order FMA chain and final lane reduction. The Q4_1 down path keeps the independent affine-term FMA chain. Activation quantization, SiTU, router weights, and the weighted sum in original per-row expert order are unchanged. Single-use groups still call ggml. This experiment is specific to the linked Zen 3 ggml arithmetic; it makes no cross-build or cross-architecture parity claim.

## Evidence

**Bitwise branch parity:** On Tideglass v2.1, both wide CPU implementations matched independent serial CPU calls for all 24 captured W1 rows across 48 visits on each of the 64K mixed and code workloads, using W8 partitions. Serial also matched all 2,062 captured nonempty CPU sums (4,750,848 FP32 words), plus the implicit zeros for empty visits. The timed W2/4/6/8 cases were each checked again against serial. The separate 2K archive has 64 consecutive real W1 rows (3,072 visits); both wide CPU implementations matched serial on all of them at W8, but that archive did not contain independently captured CPU sums. Neither comparison changes the captured placement within a case, even where the original serial trajectory moved an expert between rows.

**First actual wide divergence, not a synthetic FFN input:** Reuse the sibling divergence capture at absolute position 2305 with the same 1,143 slots and no swaps. W2 and W4's first row matched W1 in the captured CPU inputs, ordered route IDs, weight bits, and CPU outputs at all 48 visits. W8's first row first differs at visit 0 `router_scores` (78/128 words, max absolute difference `4.77e-7`), before the CPU expert code. At that visit the FP32 FFN input, ordered route IDs, and CPU/GPU mask match W1, but two of eight route weights and 1,501/2,304 CPU sum words differ. The source cutover is concrete: `gemv_f32_waves` uses its one-block-per-output reduction when `ceil(128/8) * n < 80` for this router, and a 32-lane reduction otherwise. W6/W8 cross that threshold; the cited capture is W8. Serial, grouped, and multi all match the actual W8 CPU sums on all 384 rows/visits. In a counterfactual replay that changes only the first row's visit-0 router weights to W1, all three reproduce the captured **W1** CPU sum there, and still reproduce captured **W8** CPU sums everywhere else. CPU reuse cannot remove the upstream difference.

**1-core microcost:** Ryzen 7 5700X, GCC 16.2, Tideglass Q4_0 gate/up, Q4_1 down at visits 0–2 and Q4_0 down otherwise. One reproducibly sampled window per logical visit and width, plus a separate first-window/visit-0 check. Two occurrences of each arm per `A B C C B A` block, with rotated assignment and two blocks per case in the repeat. Cold = CLFLUSHOPT on model payload before the timer; warm = repeat the same case/arm before the timer, not a claim that the whole working set fits in L3. These were concurrency-limited runs on a shared machine, not isolated hardware. Sum of per-case median *wall* durations in the repeat (ms, not one full verifier round):

| Width | Mixed cold serial / grouped / multi | Code cold serial / grouped / multi | Grouped/multi process CPU, mixed / code |
|---|---:|---:|---:|
| 2 | 190.02 / 197.75 / 192.78 | 114.65 / 113.69 / 108.88 | 0.986× / 1.040× |
| 4 | 291.69 / 288.81 / 262.79 | 330.26 / 308.34 / 291.24 | 1.068× / 1.075× |
| 6 | 560.43 / 505.71 / 475.04 | 391.11 / 393.64 / 351.69 | 1.074× / 1.115× |
| 8 | 622.35 / 610.16 / 548.53 | 538.75 / 524.12 / 462.74 | 1.113× / 1.137× |

W2 did not show a consistent win. W4–8 were faster in both warm and evicted repeat screens, and the earlier 2K control has the same positive direction at those widths. The two-core cold screen (CPUs 5 and 6, no timed major faults) gives grouped/multi wall ratios of **1.02/1.00× at W2**, **1.06/1.06× at W4**, **1.25/1.11× at W6**, and **1.03/1.11× at W8** (mixed/code). Do not interpret the difference between the one- and two-core series as scaling: they ran at different levels of contention. The timing files retain scheduler interruptions, wall/CPU discrepancies and major faults (including six in the one-core mixed W8 cold leg, one in code W4 cold). Process CPU time reduces preemption sensitivity but does not remove memory bandwidth, frequency, page-cache, or SMT interference. GPU work, offload, `meanwhile` launch callbacks, and state commit are not timed here; a CPU-branch speedup is only an optimistic all-cost-in-this-branch bound in that replay, not a projected decode throughput or a reason to widen a near-term verifier.

**Actual wide inputs:** Replaying the CPU parts of the captured W2/W4/W8 at position 2305, the one-core warm grouped/multi sum of case medians was **169.02/160.81**, **376.83/327.12**, and **595.95/507.85 ms**, respectively. These sum all 48 visits separately and do not measure a GPU round. At W8, 496/1,240 (40.0%) CPU uses are repeats, versus 22–26% in the W1 trajectory replays. The visit-0 case itself had 41 uses of 27 CPU experts and took **18.14/15.75 ms**. The W8 warm leg had no timed major faults; its cold leg had 21. A sensitivity check retaining only the 45 W8 cold visits with no timed major faults in any arm still gives 1.217× grouped/multi wall, but is subject to the same shared-host interference.

**Fixed-placement demand ceiling, not a cache or transfer measurement:** The short archive has real ordered routed IDs and per-visit VRAM masks. Hold the mask from the first row of each window fixed at that visit, replay actual choices, and count duplicate uses within the same backend. At W2/4/6/8 the removable fraction of independently decoded expert payload is **10.4/19.8/27.0/31.9% CPU** and **19.5/37.2/47.3/53.0% GPU**. With the real serial per-row masks instead, CPU repeats are **9.3/15.4/19.8/22.4%**; at 64K they are **6.5/15.1/23.3/24.9% mixed** and **9.6/19.7/20.1/26.0% code**. These ceilings assume each repeated payload would otherwise be loaded again and each distinct one only once; caches may already keep it. No dot or required activation can be skipped merely because the weight is shared. No PCIe or GPU expert implementation was measured.

## Reproduce (CPU only)

Paths below are local scratch inputs. Use a new output directory for repeats. Do not publish binary captures, prompts, or weights. Raw timing, branch checks and build hashes for the reported runs are in `~/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon/reuse/`. `ldd.txt` confirms no GPU libraries are linked. Replay needs the existing Release CPU object and static ggml libraries, not an active GPU.

```sh
T=$HOME/.cache/opencode-tmp/opencode
R=$T/polystrata-runs/dflash2-recon/reuse-repeat
H=rdna2-rnd/dflash2/reuse
M=$HOME/.lmstudio/models/unsloth/Inkling-2-Mini/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf
A=$T/polystrata-runs/native-donor-bc2b/router
D=$T/polystrata-runs/dflash2-recon/divergence
umask 077
taskset -c 6 python -B "$H/build.py" "$R"
taskset -c 6 python -B "$H/fixtures.py" archived-2k "$A/mixed2k.hidden.bin" \
  "$A/mixed2k.normalized.v2.jsonl" "$R/archive2k.bin" "$R/archive2k-dedup.json"
for w in 2 4 8; do
  taskset -c 6 python -B "$H/fixtures.py" dvt "$D/short-w$w/trace.dvt1" "$R/short-w$w.bin" \
    --baseline "$D/short-w1/trace.dvt1" > "$R/short-w$w-first.json"
done
taskset -c 6 python -B "$H/fixtures.py" dvt "$D/short-w8/trace.dvt1" "$R/short-w8-w1weights.bin" \
  --baseline "$D/short-w1/trace.dvt1" --restore-first-visit0-weights > "$R/short-w8-w1weights-first.json"
for work in mixed code; do
  for mode in check cold warm; do
    taskset -c 6 "$R/replay" "$M" "$T/tideglass-impl/capture/$work/run/cpu.bin" "$R/$work-$mode.csv" 1 2 "$mode"
  done
done
taskset -c 6 "$R/replay" "$M" "$R/short-w8-w1weights.bin" "$R/counterfactual-check.csv" 1 1 check
taskset -c 6 python -B "$H/summarize.py" "$R"/*.csv > "$R/summary.json"
```
