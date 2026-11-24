# Fresh P0/P1 RECON — §5 output

Follow-up: [P1 short external task prefixes](task-prefixes.md) (English reasoning, Python, tool call and tool-result replay).

**STATUS:** P0 **INCONCLUSIVE** as an optimization; trace supports screening cache allocation by exposed cost, not by
hit count alone. P1 **INCONCLUSIVE** generally; **do not advance the tested full-weight previous-pass prefetch** on
these traces. There is no candidate or measured E2E speedup.

**HYPOTHESIS:** P0: routing misses have nonlinear critical-path costs; paying for a GPU hit where it reduces a
multi-miss exposed join may matter more than an equal increase in hit count elsewhere. P1: an earlier pass of the
same physical layer can predict a subsequent CPU miss early enough to stage its bytes without substituting for
the real router or changing the numerical path.

**METHOD:** CPU analysis of three guarded serial decode captures on RX 6700 XT (`gfx1031`), Ryzen 7 5700X:
2K mixed, 2K code, and 64K mixed. The model was `Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf` (campaign SHA-256
`7af63a2c9362f0ac6fc708443348ead5dbfcb7ecde8cd454ea3572c68864aa72`), instrumented `c0dd0e52` binary SHA-256
`93027e25d7f3723c69bec32e7efbaa0eab6c25344fa07bf3a3d644fb3ed1dc62`, Q8_0 KV, 200K cap, eight CPU workers,
2,048 MiB reserve, 64 swaps, temperature 0 seed 42; no draft/lookup. All three have 256 contiguous computed rows
with 48 visits each and live placement; 1,159/1,148/909 observed resident slots at the first traced row.
Each had no guard stop. The 64K leg directly prefills; it is not a restored conversation. The prompts use synthetic
repo text. The code prompt is a different trajectory, not a separate codebase or held-out code corpus: mixed includes
the same replay-script text. The 64K prompt uses the same mixed-text generator with a longer prefix; this is a
context-shift test, not a simultaneous-controlled model or context-size causal comparison.

Use a predeclared contiguous 166/90 split within the mixed captures, and all 256 mixed-2K rows as training for the
external code-2K and context-shift 64K tests. Predictors are scored on all test rows except the first (previous-token
warm-up). No target cache map or future route is available at issue. The target's actual CPU placement is the label, not
what a different policy would have routed or placed. Miss-count fit is a descriptive held-out median grouped by
observed CPU count with/without stage, not the allocation simulator.

**REPRODUCTION_COMMANDS:** From the repo root, `R=docs/goals/arch-decode-recon/route`; `D` and `S` are pinned in
**ARTIFACT_PATHS**. Metadata JSON is kept in `S`; it contains no service credentials. The commands below use a new
results subdirectory, or use the existing outputs in `S` to read the reported results.

```sh
export OPENBLAS_NUM_THREADS=1
O=$(mktemp -d "$HOME/.cache/opencode-tmp/opencode/arch-route.XXXXXX")
python -B "$R/test_routes.py"
for spec in 'trace-on 2k 2048' 'trace-code-2048 code-2k 2048' 'trace-mixed-65536 64k 65536'; do
  read -r run key pos <<EOF
$spec
EOF
  python -B "$R/cp02_to_routes.py" "$D/$run/visits.psv" --metadata "$S/tideglass-$key-meta.json" \
    --from-pos "$pos" --to-pos "$((pos + 256))" --output "$O/$key.jsonl"
done
for key in 2k 64k; do
  python -B "$R/analyze_routes.py" "$O/$key.jsonl" --holdout-start 166 \
    --output "$O/$key-analysis.json" --transition-output "$O/$key-transitions.npz"
  python -B "$R/screen_lookahead.py" "$O/$key.jsonl" "$D/guard-transfer/command.log" \
    --output "$O/$key-deadlines.json"
done
for key in code-2k 64k; do
  python -B "$R/analyze_routes.py" "$O/$key.jsonl" --train "$O/2k.jsonl" --output "$O/$key-transfer-analysis.json"
done
python -B "$R/analyze_routes.py" "$O/64k.jsonl" --train "$O/2k.jsonl" --eval-start 166 \
  --output "$O/64k-transfer-matched-suffix-analysis.json"
python -B "$R/screen_lookahead.py" "$O/code-2k.jsonl" "$D/guard-transfer/command.log" \
  --first-test-row 1 --output "$O/code-2k-deadlines.json"
```

## MEASUREMENTS

**P0: observed cost.** Medians over tokens, ms/token (CPU and GPU spans overlap; do not sum them). Marker gap is
the GPU stream time from expert/shared completion to enqueue of CPU consumption as observed with probes. It includes
exposed CPU work **and** host dispatch/submission; zero misses give zero by construction. Router wait is a host wait,
not independent GPU work or a separately removable budget.

| Held-out cells | CPU miss rate | GPU visit span | Router wait | CPU branch | GPU expert span | Marker gap, all visits | Marker gap, core 2+3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed 2K suffix (90) | 16.55% | 21.060 | 12.874 | 6.719 | 5.235 | 3.929 | 1.436 |
| Code 2K external (256) | 17.76% | 21.131 | 13.042 | 6.803 | 5.229 | 4.063 | 1.136 |
| Mixed 64K suffix (90) | 26.09% | 28.614 | 16.093 | 11.180 | 4.865 | 8.217 | 3.344 |
| Mixed 64K full, trained at 2K (256) | 29.18% | 29.623 | 16.105 | 12.146 | 4.743 | 9.283 | 3.733 |

| CPU experts per visit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed 2K marker gap, µs median | 0 | 4.5 | 109.7 | 309.4 | 405.7 | 533.0 | 674.5 | 783.9 | 939.8 |
| Code 2K marker gap, µs median | 0 | 4.5 | 111.0 | 309.9 | 417.4 | 533.5 | 671.5 | 785.3 | 1235.9 |
| Mixed 64K suffix marker gap, µs median | 0 | 4.5 | 110.0 | 309.5 | 406.4 | 522.5 | 651.6 | 758.0 | 902.2 |
| Mixed 64K suffix sample count | 661 | 1,024 | 1,087 | 814 | 433 | 198 | 71 | 27 | 5 |

For the 64K suffix, CPU branch medians for one/two/three misses are 111/214/319 µs; cached-GPU expert spans are
111/101/89 µs. Sequential host branch duration is not the exposed cost. Even same-stage count changes are not a
controlled `n→n−1` intervention. At 64K, held-out marker-sum MAE is **0.40 ms/token** with count medians trained
at either 64K or 2K on the **same** 64K suffix. Transferring 2K medians to the full 64K run (including the earlier,
less stable cache state) gives **0.76 ms/token**. At 2K it is 0.37 ms on mixed and 0.61 ms on external code.
Stage-specific medians improved whole-visit time MAE from
0.78 to 0.47 ms on mixed 2K, but neither reliably improved the exposed-gap fit nor transferred full visit timing to
64K (3.90 ms MAE from count-only vs 4.00 ms with stage). Full attention, context, stage, CPU/GPU overlap, model formats
and residency history have to remain separate in any cache-cost model. The eight-miss tail is sparse and some
samples include unrelated enqueue stalls.

**P1: route prediction.** Triples are `1→2 / 2→3 / 1→3`. Same-ID overlap is out of eight; marginal null preserves each
layer's observed prevalence but pairs routes independently. Pearson correlations are means of per-layer CPU miss
counts, not miss-identity predictions.

| Eval split | Same-token mean overlap | Marginal null | Lag-one cross-pass overlap | Miss count r (lag-one null) |
| --- | --- | --- | --- | --- |
| Mixed 2K suffix | 4.15 / 3.98 / 2.88 | 1.67 / 1.23 / 1.20 | 2.16 / 1.76 / 1.63 | .43 / .48 / .32 (.05 / .08 / .10) |
| Code 2K external | 3.97 / 4.22 / 2.85 | 1.51 / 1.30 / 1.19 | 2.29 / 2.05 / 1.74 | .49 / .59 / .40 (.19 / .25 / .21) |
| Mixed 64K suffix | 3.00 / 3.76 / 2.08 | 1.67 / 1.25 / 1.13 | 1.75 / 1.35 / 1.21 | .17 / .39 / .15 (−.06 / .04 / −.01) |
| Mixed 64K external from 2K | 2.83 / 3.71 / 2.03 | 1.51 / 1.15 / .98 | 1.66 / 1.33 / 1.13 | .31 / .46 / .24 (−.00 / .06 / .00) |

| Eval split | Previous-pass route precision@4 | Previous-token | Target-pass frequency | Conditional transition |
| --- | --- | --- | --- | --- |
| Mixed 2K suffix | 64.9 / 61.4 / 44.3% | 37.4 / 36.1 / 36.1% | 44.5 / 36.2 / 36.2% | 53.6 / 47.0 / 43.5% |
| Code 2K external | 62.7 / 64.0 / 43.3% | 42.2 / 36.5 / 36.5% | 22.4 / 11.4 / 11.4% | 25.9 / 16.2 / 13.5% |
| Mixed 64K suffix | 46.7 / 57.8 / 32.0% | 30.6 / 21.7 / 21.7% | 46.3 / 34.0 / 34.0% | 49.3 / 45.8 / 37.9% |
| Mixed 64K external from 2K | 43.7 / 57.2 / 30.1% | 30.4 / 21.5 / 21.5% | 35.6 / 20.4 / 20.4% | 40.0 / 29.3 / 23.9% |

At four picks, route recall is half the precision. More route overlap does not imply more opportunities: after masking
resident/pending IDs, one previous-pass pick gives **20.8%/19.6%** CPU-miss recall on mixed 2K, **19.3%/19.2%** on
code 2K, only **12.5%/15.6%** on the 64K suffix. At 64K, 1→3 recalls 7.4% at 21.6% miss precision and would compete
with 2→3; no additive claim. Median across-token source-ready→target-ready gaps by layer span 4.33–5.39 ms for
adjacent passes at mixed 2K, 4.50–5.18 ms for code and 6.15–7.44 ms for the 64K suffix. This is host-clock
lead, not DMA overlap measured during inference.

**P1: payload, deadlines, and exposed-gap envelope.** Actual isolated 4,478,976 B transfers, 128 samples each:
registered pinned completion including minimal pack + enqueue median/p90 **173.72/175.90 µs**, and separately packed
staging **322.65/351.18 µs**. Simulate one serial transfer server using observed host ready times and those durations;
not a GPU workload, no existing swaps or DRAM/PCIe contention, and no arena capacity/eviction effects. Count a target
visit once if at least one predicted observed miss arrives by route-ready. We report the *entire observed marker
gap* in that visit, not a reduction attributable to the prediction.

| Held-out | Cap/source visit | On-time useful picks, pinned median | Miss recall | Payload / wasted, MB/token | Peak payload in independent staging | Observed gaps touched, mean ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed 2K | 1 | 516 / 1,182 | 20.1% | 59.5 / 33.5 | 53.7 MB | 0.812 |
| Code 2K | 1 | 1,400 / 3,454 | 19.3% | 60.7 / 36.1 | 53.7 MB | 0.856 |
| Mixed 64K | 1 | 590 / 1,662 | 14.4% | 83.6 / 53.9 | 53.7 MB | 1.520 |
| Mixed 2K | 4 | 771 / 1,896 | 30.0% | 95.4 / 56.6 | 170.2 MB | 0.927 |
| Code 2K | 4 | 2,136 / 5,684 | 29.4% | 99.8 / 62.3 | 197.1 MB | 0.996 |
| Mixed 64K | 4 | 1,011 / 3,409 | 24.6% | 171.6 / 120.7 | 152.3 MB | 1.867 |

All jobs are on time at isolated pinned median, including false positives; at pack p90 cap 1 still meets all modeled
deadlines, but cap 4 misses 22 jobs on mixed 2K and 138 on code. Larger payloads lengthen the schedule and make the
result more sensitive to queue delay. The 64K cap-1 payload alone is ~2.7 GB/s at the matched untraced reference
32.34 token/s; normal swaps are additional. This calculation is a load estimate, not measured PCIe utilization.

**BASELINE_COMPARISON:** P1 predictor baselines are in the route table; masked frequency often labels few useful
misses despite good route coverage. Trace-on/off with the same instrumented binary at mixed 2K: 40.78/42.33 t/s
(single on/off pair, 3.7% lower with trace); matched-prompt 64K trace-on vs uninstrumented CP-01: 31.17/32.34 t/s
(single pair, confounded by binary/layout and runtime load). Neither comparison is a cache or prefetch candidate.

**FIRST_DIVERGENCE:** None in all 257 emitted greedy IDs for mixed 2K trace-on vs trace-off or vs its uninstrumented
CP-01 run, nor for 64K trace-on vs its same-prompt CP-01 run. No corresponding code trace-off leg and no P0/P1
candidate. There are no paired logits/state/ordered visit records across these comparisons, so no claim at those
boundaries.

**ACCURACY_EVIDENCE:** PSVST1 parser validated 12,288 visits per run, including ordered IDs/FP32 weight bits and
placement consistency. It is a single baseline trace, not Q0–Q5 parity. Sparse or moved residency can change
CPU/GPU reduction and arithmetic even with the same ordered route. Keep the real router authoritative; the host
prefetch arm must leave placement, bytes and summation unchanged. A GPU-consumed staged miss is a placement
change, not exact preparation by virtue of using the same expert ID.

**MEASURED_UPPER_BOUND:** For the specific mechanism of reclaiming the traced CPU-consumption gap while holding
other work fixed, the **whole measured marker-gap envelopes** are 3.929/4.063/8.217 ms/token median across all visits
and 1.436/1.136/3.344 ms/token median in core 2+3. These are not exact CPU-only times, bounds on every optimization
or E2E improvement. For tested P1 arms, on-time useful predictions are confined to visits with 0.812/0.856/1.520 ms
mean observed gap at cap 1; even cap 4 touches only 1.867 ms mean at 64K before transfer, contention and exactness
costs. One expert arriving cannot reclaim the complete visited gap in general. If prefetch changes only weight
readiness on the CPU, the reclaimable portion is smaller and was not separately timed.

**ARTIFACT_PATHS:** Repo: this report, [`README.md`](README.md), `analyze_routes.py`, `cp02_to_routes.py`,
`screen_lookahead.py`, `test_routes.py`, and archived-model parsers. Private: `D=~/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK`,
`S=~/.cache/opencode-tmp/opencode/arch-route.ldWtEw`. Raw traces and results:

| Run under `D` | `visits.psv` SHA-256 | Results under `S` |
| --- | --- | --- |
| `trace-on` | `894cadef4e5828faf0bebeff641f6f803579657dc9e7d1a35be01f4484d00364` | `tideglass-2k-{meta,analysis,deadlines}.json`, `tideglass-2k-transitions.npz` |
| `trace-code-2048` | `d2aab835d2f6d4a4e8769f7a8b9ce57a9daac4cf6a75e14084e5af335f8fc374` | `tideglass-code-2k-{meta,analysis,deadlines}.json` |
| `trace-mixed-65536` | `cb1b743fdb63e591ffb0b5d6e34d91f25b01ac375d172f085f2f80f4678a973f` | `tideglass-64k-{meta,analysis,transfer-analysis,deadlines}.json`, `tideglass-64k-transitions.npz` |

`S/tideglass-64k-transfer-matched-suffix-analysis.json` holds the 2K→64K comparison on the same 64K suffix.

**INTERPRETATION:** P0's exposed gap is large enough at 64K to justify the separate cost-aware simulator with
engine-faithful residency constraints. Prioritize many-miss visits over the fifth decimal of hit rate; validate with
matched placement and real holdouts. P1's previous-pass signal survives the code shift better than learned
frequency/transition rankings, but it weakens at long context and is expensive to exploit. Full-weight speculative
H2D on these paths does not justify a prototype now. The historical IQ4_XS and Bartowski results are diagnostic,
not substitutes for the fresh Tideglass baselines.

**NEXT_ATOMIC_EXPERIMENT:** P0: give the simulator `D/trace-{on,code-2048,mixed-65536}/visits.psv`,
`S/tideglass-*-analysis.json` and the transfer CSV. Compare under exact captured resident budgets and same-size
trades, with ask/completion timing, stratified marker costs and frozen control. If it clears the long-context
screen, prove Q0/Q1 at matched numerical placement before the Q2–Q5 and untraced paired gate. P1: no full-weight
DMA prototype. Revisit only if a cheaper preparation path preserves exact backend placement and a disjoint
restored/agent workload shows sufficient opportunity; require CPU memory stall and concurrent transfer/queue
measurements before asserting savings. No GPU, service, or engine changes were made in this pack.
