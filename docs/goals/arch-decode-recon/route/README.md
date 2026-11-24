# P0/P1: routing and exposed expert work

**Status: INCONCLUSIVE for optimization.** See the [fresh Tideglass §5 report](tideglass-recon.md) for the 2K mixed,
2K code and 64K mixed CP-02 comparisons. This page records historical evidence and the analyzer's data contract.

## Model and trace identity

| Evidence | Model and setup | What it can establish |
| --- | --- | --- |
| Historical ordered routes, November 23 | Inkling-2-Mini `UD-IQ4_XS` (NVFP4 core experts), scratch `route-capture`, RX 6700 XT, 2,048 synthetic engineering-notebook prompt + 96 forced output IDs, 95 computed decode rows, live swaps, 1,155 startup slots. Historical filename inferred from config backups and NVFP4 log; model SHA not recorded. | Ordered top-8, FP32 weight bits, and held-out route hit rates. **No** per-expert CPU/GPU placement or visit timing. |
| Historical ROCm kernel traces, November 21 | Same `UD-IQ4_XS` filename in captured command, older `fixed` binary, 8-token prompt and 24 forced IDs, four fully pinned physical layers (6, 7, 9, 13), live swaps, 1,303 startup slots. Two same-fixture legs: host route, and card route only on the pinned layers. | CPU miss *counts* inferred from branch launch shape and traced GPU gaps. **No** routed expert IDs or CPU timestamps; cannot join to the November 23 archive. |
| Prior Q4_0 goal | Bartowski `Inkling-2-Mini-Q4_0.gguf`, not Tideglass; docs report 4.86 ms median exposed join at restored 64K after four-row slots and 7.941 ms in a later, lower-hit arm. The raw routes/traces are no longer at the published scratch paths. | Historical sizing only. Same model family and GPU do not make joins, hit rates, or tokens/s comparable to the requested current weight file. |
| Current comparison | Campaign source `c0dd0e52`; current config `Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf`, manifest SHA-256 `7af63a2c9362f0ac6fc708443348ead5dbfcb7ecde8cd454ea3572c68864aa72`, separate from Bartowski/H0. Q8_0 KV, 200,000 cap, 2,048 MiB reserve, eight CPU workers and 64 swaps. | CP-02 PSVST1 comparison in the fresh report. Do not transfer historical rates. |

The Tideglass GGUF header identifies `inkling2_loop`. Every core expert's gate/up and down tensors are Q4_0:
**4,478,976 bytes per expert** (native weight payload, not rounded slot bytes or measured DMA traffic). Layers 0–2 have
Q4_1 down and Q4_0 gate/up (**4,644,864 B** per expert); all remaining layers are Q4_0 for both tensors. The model
manifest, not a basename or geometry match, is the binding identity for evaluation.

## P1: real held-out routes, historical diagnostic

`~/.cache/claude-fv/route/frames.bin` stores 128 FP32 router logits, 128 learned bias values, one expert scale,
eight IDs and eight FP32 weights per visit, in order. Its shape is 95 × 48 × 1,092 bytes. The parser verifies that
each visit's bias is byte-identical across tokens, that repeated core visits have the same physical bias, and that
all 24 physical bias vectors are distinct. Capture SHA-256 `215c506e256c72d4fa969031f06073636c33524ca747442eab7f24ff94b4aef8`;
capture binary SHA-256 `490ff31877a4cd161879346d372f4438ab74b61db56de79f25aed930f9c4e84d`.

The forced continuation has a **62-token period**. Main screen: first 62 rows only, fit on positions 2048–2087,
test on 2088–2109; 22 held-out tokens × 12 layers per transition, 21 tokens for all predictors (same denominator;
first eval token only initializes previous-token state). Overlap is mean shared expert IDs out of eight.
The null is overlap expected under independent pairing with the same per-layer held-out marginals; lag-1 preserves
local token correlation and tests source at `t-1` against target at `t`.

| Pass | Top-8 overlap | Jaccard | Marginal null | Lag-1 null | Previous-token same-target overlap | Across-layer overlap range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1→2 | 4.277 | .391 | 2.128 | 2.401 | 3.099 | 2.591–5.364 |
| 2→3 | 4.144 | .369 | 1.948 | 2.000 | 2.940 | 1.773–5.227 |
| 1→3 | 3.087 | .260 | 1.744 | 1.897 | 2.940 | 0.909–4.000 |

At four picks, **route precision / recall (%)**, micro-averaged across held-out token-layer pairs; no residency filters
or RAM-miss labels in this archive:

| Predictor | 1→2 | 2→3 | 1→3 |
| --- | ---: | ---: | ---: |
| Layer+target-pass frequency | 45.0 / 22.5 | 36.3 / 18.2 | 36.3 / 18.2 |
| Previous token (same target visit) | 47.3 / 23.7 | 44.4 / 22.2 | 44.4 / 22.2 |
| Same token, previous pass | **68.0 / 34.0** | **64.0 / 32.0** | 48.5 / 24.3 |
| Smoothed conditional transition | 49.1 / 24.6 | 42.2 / 21.1 | 39.9 / 19.9 |
| Previous token + previous pass | 62.0 / 31.0 | 59.6 / 29.8 | **51.3 / 25.6** |

With the same 40 training rows but all 55 available suffix rows (contains 33 repeated forced inputs), overlap is
4.055/3.979/2.935 and previous-pass precision@4 is 62.9%/60.8%/45.1%. Resilience to this repeat is not independent
generalization. Neither split samples code, reasoning, tool traffic, other contexts, or spontaneous model continuation.
The best overlap may be dominated by already-resident experts. A predictor trained on this model must **not** be reused for Tideglass.

## P0: historical P0 timing envelope, not attributable CPU savings

The preserved kernel CSVs under `~/.cache/opencode-tmp/opencode/dspark-speed-1795285637/` have 23 complete
head-to-head intervals of 48 visits per leg. `copy2` router markers (host) or `route_one` markers (pinned arm),
followed by disjoint Inkling tail `conv_add`, identify visits. The SiTU dispatch grid Y gives the number of GPU experts;
where no GPU expert kernel is present, all eight experts are inferred to have run on CPU. `add_scaled` or `moe_sum`
marks the last expert launch. The gap from its end to the tail's start includes host enqueue time and other causes;
with no expert launch it begins at router-copy end. Copy-engine work does not enter the kernel union. There is no
host/GPU timestamp subtraction or attribution to exact exposed CPU join.

| Measured diagnostic (medians unless noted) | Host route | Card route on pinned layers |
| --- | ---: | ---: |
| CPU expert fraction on measured visits | 20.99% | 20.88% |
| Head-to-head traced span | 38.768 ms | 38.676 ms |
| CPU-positive visit → tail gaps, summed per token | 8.934 ms | 9.743 ms |
| Same gap envelope, just core passes 2+3 (P1's target visits) | **1.220 ms** | **1.473 ms** |
| Per-visit gap for 0 / 1 / 2 / 3 CPU experts | 4.1 / 4.1 / 105.9 / 270.5 µs | 4.1 / 4.1 / 112.4 / 302.7 µs |

In these traces, per-layer CPU-miss count Pearson r averages .489/.494/.317 for 1→2/2→3/1→3 (host route) versus
.232/.183/.135 for source counts shifted back one token. Four physical layers have constant zero misses and are
excluded from the averages (8/12 eligible). The card-route arm gives .503/.510/.363 (lag null .253/.212/.158).
This is **count** association, not identity prediction. The arms have nearly identical hit rates yet different
tail gaps; traces are correlated and not evidence that card routing *caused* the difference. The 1.220/1.473 ms
envelope includes everything in those gaps. It is neither a measured reduction nor a current-model bound.

## CP-02 integration and reproduction

Run from the repository root, NumPy required. Use one thread for array math and keep raw captures and private metadata
in the campaign scratch directory. Upper/lower positions are absolute input positions, not emitted-token counts.

```sh
R=docs/goals/arch-decode-recon/route
D=$(mktemp -d "$HOME/.cache/opencode-tmp/opencode/arch-route.XXXXXX")
export OPENBLAS_NUM_THREADS=1
python -B "$R/test_routes.py"
python -B "$R/historical_routes.py" "$HOME/.cache/claude-fv" --output "$D/first-cycle.jsonl"
python -B "$R/analyze_routes.py" "$D/first-cycle.jsonl" --holdout-start 40 \
  --output "$D/first-cycle-analysis.json" --transition-output "$D/first-cycle-transitions.npz"
python -B "$R/historical_routes.py" "$HOME/.cache/claude-fv" --allow-repeated-continuation --output "$D/all.jsonl"
python -B "$R/analyze_routes.py" "$D/all.jsonl" --holdout-start 40 --output "$D/all-analysis.json"
python -B "$R/analyze_historical_rocm.py" "$HOME/.cache/opencode-tmp/opencode/dspark-speed-1795285637" \
  --output "$D/historical-rocm.json"

# CP-02, example for CP-01's 2048 prompt + 256 computed serial steps; confirm bounds from its run.
python -B "$R/cp02_to_routes.py" "$CAPTURE" --metadata "$META" --from-pos 2048 --to-pos 2304 \
  --output "$D/tideglass.jsonl"
python -B "$R/analyze_routes.py" "$D/tideglass.jsonl" --output "$D/tideglass-analysis.json"
# Generalization: use --train path/to/normalized-disjoint-trajectory.jsonl, repeatable.
```

`$META` is JSON with `model_sha256`, `model_basename`, `workload`, `trajectory_id` (use the prompt/trajectory identity,
not the run label; A/A repeats aren't train/test), prompt length and SHA-256, binary/patch and profile-seed pins,
sampling/forced/restore settings, and optionally `expert_bytes` (24 integer payload sizes in physical-layer order).
The adapter uses [`trace/collect.py`](../trace/collect.py) unchanged. It rejects missing/duplicate positions, window
visits, skipped experts, and mixed frozen/live decode. Two separate captures can be compared at the normalized
boundary over the same position slice: known same model/runtime plus matching ordered IDs, FP32 weight bits and
backend hash means that boundary matches, not that full logits, greedy IDs or committed state match.

Normalized `inkling-route-v1`: a JSON header (`phase=serial_decode`) and 48 ordered JSON rows per token; `pos`, `visit`,
`layer`, `pass` (0 prelude, 1/2/3 core, 4 coda), ordered `ids`, unsigned `weight_bits`, optional ordered `cpu` mask,
per-layer `resident_ids` and `pending_ids`, `route_ready_ns`, and named timing fields. Unknown placement or timing stays
unknown. CP-02 GPU branch markers are not silently promoted to exact joins: `gpu_to_cpu_ready_marker_ns` is the
GPU event span from expert/shared-done to a marker enqueued at host branch return. Command submission and marker
overhead remain in that span. `visit_host_enqueue_ns` is not visit completion. Separate host/GPU metrics are not
added or subtracted across clock domains. `held_out_observed_timing_fit` validates grouped timing models against
observed visits only; it never assigns a counterfactual cost to moving an expert.

P1 reports route precision/recall, actual target CPU miss recall, issued useful/misfire counts, optional payload
and false-positive bytes, hit-conditioned marker-gap exposure and observed route-ready lead. At prediction issue,
`+source_mask` filters both resident and pending IDs if those facts were captured; it never sees the target map or
future token. An expert made resident naturally before the target is not counted as a useful miss prediction even
if it was absent when issued. Selection for an observed miss is not proof its transfer could meet the deadline or
supplant the baseline CPU branch. For a fixed cap, all arms use the same test rows. Expert transition NPZ measures
held-out co-occurrences, `joint[p, l, e, f] / source_count[p, l, e]`; rank histograms and same-ID counts are in JSON.

## Gates and next inputs

`tools/replay_cache_policy.py` is not an Inkling replay: 48 layers × 512 experts, `--dump-routing` window format,
global STRP profile ranks, and periodic per-layer heuristic, not `Engine::follow()` on 24 physical slot owners.
Current source weights once-visited layers by 2, applies decaying recency plus prior ties, evicts immediately at
ask within matching size classes, and installs only on observed completion. `cover()` and `shrink_cache()` can
change capacity. Residency is keyed by `(physical layer, expert)`, not logical visit. Use CP-02 labels as observed,
then reconstruct actual maps on the same token sequence including upload completions before trusting any counterfactual.

The router runs on the actual activation and always owns ordered top-8 and weights. A P1 predictor may prepare
bytes only; no skipping, replacement, routing bias, RNG changes, different top-k order or arithmetic. P0 changing a
CPU miss into a GPU hit does not automatically preserve arithmetic: CPU dot/quantization/sums and GPU Q8_1/slots/tail
use different paths. You need a matched reference placement and Q0/Q1 leaf/chain checks before live hybrid moves,
then Q2–Q5 as applicable: full FP32 logits and greedy in separate experiments, sampling, KV/convolution and state,
restore/rollback/prefix, agent workload. Record A/A placement-dependent divergence separately. This pack has **no**
candidate A/B or first differing numerical boundary. It changes no engine behavior.

Next inputs beyond the fresh 2K/64K traces: matched-placement or restored serial traces and disjoint prompt
replicates for generalization (prose/reasoning/tool, ≥256 computed tokens each). Capture guarded metadata,
ordered IDs/bits, placement/slot offsets and resident/pending maps, upload events, branch spans and route timestamps;
combine with matched frozen and live A/A, traced/off controls, and the untraced CP-01 baseline. For P0 import concurrent
cost surfaces for 0–8 CPU experts and actual candidate byte budget; for P1 import isolated and concurrent DMA/DRAM
transfer costs and verify that useful bytes arrive in time. A single candidate of 4,478,976 B takes ~168 µs at the
historical isolated 26.6 GB/s DMA rate *before* setup or interference; one per 24 core target visits is **107.5 MB/token**
(~4.1 GB/s at the historical 38.14 t/s reference). This is arithmetic sizing, not a deployable bandwidth or speedup.
3→future-token preparation is outside this screen. Do not add 1→3 and 2→3 per-pair savings; their target visits coincide.

P0 advancement requires a reproducible ~2–3 ms/token long-context benefit or another measured payoff, then
untraced end-to-end A/B above A/A noise and exactness. P1 needs observed useful cold misses on exposed target visits
under a measured transfer/queue budget before a GPU scheduling prototype; trace marker sums alone cannot pass.
