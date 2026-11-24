# Causal 32-token placement screen: stop

**No-go. Do not prototype this policy in HIP or change `Engine::follow()`.** The frozen policy missed the **≥2 ms/token net** gate on both 64K traces. It found fewer costly joint misses, but the savings were too small even before charging any copies.

| 64K held-out suffix | Gross saved ms/token | Pack+copy charge ms/token | Net saved ms/token (95% block interval) | GPU hits, observed → causal | Pass 2→3 pairs with ≥3 misses in both visits |
|---|---:|---:|---:|---:|---:|
| Mixed (source fixture) | +1.59 | 1.23 | **+0.36 [−0.02, +0.99]** | 71.81% → 75.25% | 262 → 152 |
| Code (independent fixture) | +0.21 | 1.24 | **−1.02 [−1.82, −0.44]** | 78.57% → 78.76% | 107 → 60 |

These are savings in the fixed-route branch-stream proxy, not measured E2E wall time. Full-visit stream sensitivity is +0.38 ms/token mixed and −1.04 code. The measured live engine is the control, not an empty or frozen cache. In the earlier [placement screen](README.md), the clairvoyant 32-token pack+copy replay saved **+2.64 ms/token** on mixed; it was not run on the independent code trace. The causal policy submitted 445 and 449 uploads, respectively (about 2.00 GB per suffix). Simply removing copy charges from these decisions would not meet the gate on either fixture.

## Frozen method and fixtures

- The [configuration](causal-32-frozen.json) was pinned before the independent capture (`5f410ced2e7d931ac775304cb4a199cc2e4eb14522eb381eebb5094b857670c6`); its pre-capture pin is `$D/long-holdout/predictor-pin.txt`. One combined evaluation ran, mixed first then code. No settings or timing fits were selected using either suffix.
- The source fit uses only the first 128 tokens of the mixed 64K trace. That prefix provides the same nonlinear per-visit miss-count surface as the oracle and a fixed dictionary of whole routed token frames. On both trajectories, the forecast may also use the latest 64 **completed** token frames, with a 16-token frequency half-life. The independent prefix warms that rolling context; it does not refit weights or timing. Both score input positions `[65664, 65792)` (128 tokens, 6,144 visits).
- At submission row `t`, the query is the route of `t−1`. For each `h = 1..32`, the predictor estimates a distribution over whole route frames for `t+h`: mixture of recent frequency, the 20% fixed source prior, and nearest already-observed anchor→target examples at lag `h+1`. Anchor similarity includes both per-visit route matches and shared expert IDs across the repeated physical layer's three core passes. Conditional weight decays with forecast horizon. The expected nonlinear cost is integrated over these whole-route estimates, not computed from expert hit rates alone. Actual rows `t` and later are invisible to the scheduler.
- The replay starts each suffix with its actual pre-route residency and outstanding uploads. It verifies every mask and selected slot offset in both captures. The policy considers same-class evictions at most every eight tokens, up to 32 replacements at a time and within the 64-entry queue. Both captures funded 93 large plus 817 regular slots (4,091,584,608 B). It uses serialized pack+copy p90 and later boundary eligibility from the same CPU simulator as the oracle; no overlap credit or numerical parity is assumed. Uncertainty uses 16-token source and held-out blocks with the chosen policy fixed; the intervals do not cover request-to-request variation or arithmetic changes.
- Both fixtures use the same Tideglass model, RX 6700 XT, Q8_0 KV, 64K context, eight CPU workers and original profile seed. The code trace guard had no stop. Trace SHA-256: mixed `cb1b743fdb63e591ffb0b5d6e34d91f25b01ac375d172f085f2f80f4678a973f`, code `377076dbdb464641c55b1232f0350aefb7a56733ab7bccf5280d360c87fa7b58`. This screen ran on CPU with one math thread and `nice -n 17`.

## Conditions to reopen

Keep the existing weighted follower. Reopen only for a materially different causal signal or a resource change with separately measured benefit, not for another fit to these suffixes. Pin the new algorithm, resource charges and score window before capture; require ≥2 ms/token net on **new independent long-context fixtures** against live placement under pack+copy accounting. If it passes on CPU, require CP-P0-04 numerical parity before a guarded untraced HIP A/B test. A higher hit rate, an in-sample rank, or oracle headroom alone is not a pass.

## Reproduce

[`causal-32-result.json`](causal-32-result.json) contains the fixed-policy replay results, including joint misses, by-visit savings and input hashes. The raw traces stay in scratch. The command below reproduces the recorded run, not an additional independent holdout.

```sh
R=docs/goals/arch-decode-recon/placement
D="$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B "$R/test_causal_32.py" -v
nice -n 17 python -B "$R/causal_32.py" "$D/trace-mixed-65536/visits.psv" \
  "$D/guard-transfer/command.log" --independent "$D/long-code-65536/run/visits.psv" \
  --output "$D/long-holdout/reproduced.json"
```
