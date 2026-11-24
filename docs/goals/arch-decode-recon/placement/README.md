# CP-P0-02/03 — trace-driven expert placement

**Decision: retain `Engine::follow()`; do not build a placement prototype.** The 64K trace has headroom **if** future routes are known, but none of the tested causal policies crosses the screen. The [final frozen causal 32-token screen](causal-32-report.md) returned +0.36 ms/token net on mixed 64K and −1.02 on independently captured code 64K, below the ≥2 ms/token gate on both. CP-P0-04 arithmetic parity remains a separate prerequisite.

## Data and controls

RX 6700 XT (gfx1031), Ryzen 7 5700X, *Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf*, Q8_0 KV, 8 CPU workers, 64 swap queue entries, 2 GiB VRAM reserve, greedy serial decode. See the [campaign manifest](../research-manifest.md). The short `trace-on` mixed fixture is prose at 2K; `trace-code-2048` is a separate code fixture; `trace-mixed-65536` is the long mixed fixture. Each capture has 256 tokens × 48 visits. Positions starting at 2048 or 65536 through the next 127 tokens train the surface and policies; the last 128 tokens are held out. The oracles explicitly see that suffix; the other policies do not see future routes. These are three trajectories, not three trials of one intervention. All four input guards (three traces and transfer) had no stops; each recorded the same count of 22 boot warnings. The CPU replay ran with one math thread and `nice -n 17` while quiet A/A work was ongoing. This analysis ran no additional GPU commands and did not measure inference speed.

The observed replay reads [PSVST1](../trace/schema.md) through [`collect.py`](../trace/collect.py). It seeds the exact resident and pending maps, applies observed evictions, claims and shrink moves in host timestamp order, and checks **all 12,288 resident masks and selected offsets per trace**. No mismatch; there were no in-window shrink events in these captures. A 64K prefill left 910 funded slots, versus 1,176 at 2K. The replay uses realized transition decisions, not a reconstruction of prefill's unlogged scores. The separate weighted-recency proxy uses the original profile for ties, 128-token half-life, weights 2/1 for once-visited/core layers and the engine's admission rule. It starts recency at the capture boundary, so it is not exact. On 64K it matched 98.52% of held-out visit miss counts (71.84% hits vs 71.81% observed); it gains just 0.024 ms/token before modeled uploads.

| Trade class | Actual bytes/slot | 2K funded slots | 64K funded slots |
|---|---:|---:|---:|
| Physical layers 0–2 | 4,645,440 | 93 | 93 |
| Physical layers 3–23 | 4,479,264 | 1,083 | 817 |
| Total | | 5,283,068,832 B | 4,091,584,608 B |

An expert is keyed by **physical** layer and ID; core passes all use the same slot. Counterfactuals preserve the observed budget in each trade class, including reserved slots. A moving held-out capacity is rejected rather than silently creating slots. Arithmetic and routes are fixed to what the observed engine produced; this is not a claim that expert backend changes preserve the generated IDs.

## Timing and policy

The primary timing target is `gpu_dispatch_ms + gpu_experts_ms + gpu_join_ms`, sequential spans on the stream from after shared expert work to the routed branches' end. Using only `gpu_join_ms` would credit reductions in the wait without charging for added GPU work. The fitted value is a median for each logical visit and miss count, shrunk toward the stage and global medians when samples are sparse. All nine counts are supported by the training half, but not at every visit. The curve can be nonmonotone: zero misses need not be faster than one. No host CPU span is added to a GPU span. At 64K the primary fit has 21.6 µs held-out visit MAE and predicts the held-out token sum with +0.33 ms mean bias and 0.69 ms error SD. Join-only and full visit stream time are sensitivity targets; all three are observational, not causal wall-clock fits. Intervals resample both halves in 16-token blocks (120 draws; fixed chosen policies) and do not quantify uncertainty across requests, changed residency, or different transfer implementations.

- **Train-only frequency:** maximize individual route counts subject to the slot counts per trade class. **Static frequency oracle:** same algorithm fitted on the held-out routes. Neither table row charges installation; even a trained static map still needs to be installed.
- **Train-only cost static:** greedy marginal savings across entire routed visits, then same-class pair search from both greedy and frequency seeds; retain the lower *training* cost. Training permits a fixed map; evaluate its steady state and also replay installing it, eight requests at most per token.
- **Free dynamic diagnostic:** all routed experts for each token fit in the respective trade classes when given free swaps before every token. This is not a proposed policy. The separate relaxed timing bound allows the best miss count at each visit independently, even if no physical map realizes it.
- **Paid lookahead and causal replays:** choose disjoint same-class replacements using the modeled cost of future 16 or 32 tokens (oracle), or the last 16 tokens (causal). Incumbents are evicted at submission; a new expert is only eligible after serialized service and a subsequent observed row boundary. At the first held-out row all paid arms start from the same observed pre-route map, including pending uploads, which are claimed at their observed times. For the main 32/8 oracle there are at most 32 new copies every eight tokens, four per token averaged, under the observed 64-entry queue. No instant swaps, no credit for overlapped service or for uploads that the observed baseline already paid. The objective charges eventual eviction through the displaced expert's visits and requires a modeled gain over pack+copy service on its scoring window.

The campaign recorded `guard-transfer/command.log` on the same GPU with its [isolated transfer harness](../transfer.cpp). Per mode, 128 repeats: 4 B p50/p90 **15.87/17.31 µs**; pinned 4,478,976 B **173.72/175.90 µs**; pack+copy same size **322.65/351.18 µs**. Service uses p90, scaling its payload-dependent part to the allocation size. Measured service excludes concurrent decode contention, full RAM/pool/registration effects, and scheduler scoring cost. Observed token spacing is used only for transfer eligibility, not for claiming a throughput measurement. The paid oracle is a resource-constrained estimate, not evidence that the backend can execute this schedule or that it is optimal.

## Held-out results

Positive values save modeled ms/token vs the observed live placement. Paid rows debit full p90 service.

| Policy | 2K prose | 2K code | 64K mixed | 64K hits | 64K visits with ≥3 misses |
|---|---:|---:|---:|---:|---:|
| Observed live replay | 0 | 0 | 0 | 71.81% | 2,473 |
| Freeze observed map at held-out start | ≈0 | −0.41 | −2.78 | 65.67% | 3,237 |
| Train-only static frequency | +0.09 | −0.15 | −1.73 | 68.43% | 2,708 |
| Train-only cost static, ideal install | +0.38 | −0.05 | −1.79 | 67.88% | 2,795 |
| Train-only cost static, install/pack | −0.01 | −0.47 | −2.17 | 67.75% | 2,820 |
| Static frequency oracle, ideal install | +0.56 | +1.09 | +4.09 | 82.69% | 1,246 |
| Free per-token all-hit diagnostic | +4.08 | +4.39 | +8.61 | 100% | 0 |
| Relaxed per-visit timing upper bound | +4.19 | +4.48 | +8.66 | — | — |
| 16-token clairvoyant, every 4, pack | +0.06 | +0.26 | +1.76 | 77.77% | 1,396 |
| **32-token clairvoyant, every 8, pinned** | +1.22 | +1.43 | **+3.29** | 80.17% | 1,177 |
| **32-token clairvoyant, every 8, pack** | +0.60 | +0.78 | **+2.64** | 80.17% | 1,177 |
| Causal past-16, every 4, pinned | −1.62 | −1.43 | −0.02 | 72.98% | 2,216 |
| Causal past-16, every 4, pack | −2.31 | −2.14 | −0.73 | 72.98% | 2,216 |

At 64K the 32/8 oracle yields +3.93 ms/token gross, with 2,113,238,016 B of copies over 128 tokens. The 95% block intervals are [2.95, 3.77] pinned and [2.28, 3.09] pack; full-visit stream sensitivity is +3.38 and +2.73 ms/token. In the same-layer core pass 2 and 3 joint cells, 262/1,536 observed pairs had ≥3 misses **in both visits**, vs 216/1,536 cost static and 81/1,536 for the 32/8 oracle. The 64K cost static appears to help core passes 2+3 by +0.51 ms/token but loses −2.16 ms/token across prelude+coda. At 2K prose, cost static has a lower hit rate than frequency (83.28% vs 85.05%) but fewer visits with ≥3 misses (910 vs 1,012). Cost and hit rate are not interchangeable.

The [CP-01 untraced 64K leg](../baseline.md) was 32.34 t/s or 30.92 ms/token. If timing were additive and exact routes survived, the two oracle estimates would imply 36.19 and 35.36 t/s, not a measured speedup. The train-only cost static would imply 30.57 t/s even if preinstalled. Do not compare these projections to traced throughput or use them as a parity or A/B result.

## Marginal slot utility and next test

For a nonresident expert we score the per-visit reduction on adding it at the trained map, divided by actual resident GiB. For a resident expert we score the cost of removing it. Both hold the other slots fixed; adding an expert is **not** an admissible placement by itself. Pair scoring corrects the case where the two experts appear in the same visit: their net miss count change is zero. Training and held-out values can differ even at the same map, and a change of neighboring residents changes the marginal again. Static selection paid no upload fee during training; the installer simulation charges its 118 replacements at 64K (529,882,560 B, 20.80/41.54 ms serial p90 pinned/pack).

| Candidate slots | 2K prose train/test top-32 overlap | 2K code | 64K mixed |
|---|---:|---:|---:|
| Nonresident promotion | 8/32 | 1/32 | 3/32 |
| Resident removal loss | 25/32 | 21/32 | 18/32 |

At 64K the top hindsight promotion `(layer 8, expert 5)` has **2.12 → 26.31 ms/token/GiB** train→test at the same trained map; this is not a recommendation to pin that expert. Holding the suffix fixed and changing only the surrounding map also matters: `(layer 9, expert 34)` is absent from both maps and has utility **4.39** at the static frequency map vs **6.78 ms/token/GiB** at the cost-aware map. The slot-budget sensitivity is −3.53 / −1.79 / −0.40 ms/token at 782 / 910 / 1,038 slots, all ideal static installation. An extra 128 slots (575,506,080 B at the same class mix) recover 1.39 ms/token against the inferior static map, but still do not beat the current policy. This does not justify increasing the VRAM budget.

The final causal test applied the paid 32/8 schedule to past-only route estimates on a second long-context fixture. It failed the pack+copy gate on both requests. See the [final report](causal-32-report.md) for the frozen configuration, results and conditions to reopen. Do not use the suffix-fit scores or the free oracle as a deployable map. No numerical exactness work was performed here.

## Reproduce

```sh
R=docs/goals/arch-decode-recon/placement
D="$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK"
OPENBLAS_NUM_THREADS=1 python -B "$R/test_simulate.py" -v
bash "$R/reproduce.sh" "$D" "$D/placement-new"
```

[`placement-oracle.json`](placement-oracle.json) and [`slot-utility-analysis.json`](slot-utility-analysis.json) are compact summaries. The target scratch directory contains detailed calibration counts, 48 per-visit deltas, 9×9 same-layer cross-pass miss tables, selected maps, and all 3,072 physical slot utility rows per trace with stage breakdowns. No raw route captures, transfer logs, or generated token IDs are copied to the repository.
