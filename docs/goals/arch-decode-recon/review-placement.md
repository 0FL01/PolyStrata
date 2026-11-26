# CP-R-02 — independent placement review

**Verdict: do not integrate the tested placement policies.** They did not clear the modeled gate against live
`Engine::follow()` on the tested routes. This is a screening decision, not a measured wall-time loss or evidence that
same-budget placement cannot work. Keep the follower and `use_weight` unchanged.

I checked the [placement report](placement/README.md), [`simulate.py`](placement/simulate.py), the
[oracle summary](placement/placement-oracle.json), and the saved [frozen 32/8](placement/causal_32.py) replay in campaign
scratch. Positive means saved time: fitted cost of the **observed** miss counts minus fitted cost of the alternative
miss counts, summed across the 48 visits, then minus the alternative's copy debit. The live zero is a reference, not
zero live upload work. These 64K values use the same funded budget of 93 + 817 physical slots, including reservations;
values are modeled ms/token.

| Policy | 64K mixed | Independent 64K code |
|---|---:|---:|
| Train-only static frequency, free install | −1.73 | — |
| Train-only cost static, free install / pack install | −1.79 / −2.17 | — |
| Causal past-16/4, gross / net pack | +0.68 / −0.73 | — |
| Clairvoyant 32/8, gross / net pack | +3.93 / +2.64 | — |
| Frozen causal 32/8, gross / net pack | +1.59 / +0.36 | +0.21 / −1.02 |

The 32/8 causal gate was **≥2.00 net pack on each request**; neither leg passes, even before debiting its copies under
this baseline convention. At 2K the best same-budget train-only static result was about +0.39 even with free
installation. The clairvoyant pack result already debits 1.29 ms/token for 2.113 GB of copies in 128 tokens. It
demonstrates headroom *in the model* with future routes, not a deployable policy or optimum. The free all-hit replay
(+8.61 at 64K) is a diagnostic; only the relaxed per-visit minimum (+8.66) is an upper bound, and only inside this
fitted timing model.

- **Timing.** Dispatch, GPU expert work and exposed join are successive HIP stream spans, so the primary target charges
  added GPU work and does not also sum overlapping host CPU spans. Visit-by-miss-count medians are observational and
  backed off for sparse cells. The 64K mixed held-out fit had 21.6 µs visit MAE and +0.33 ms/token mean error on
  *observed* placements; that does not validate cost after moving specific experts. Full-visit sensitivity gives +2.73
  for the pack oracle and +0.38/−1.04 for the frozen causal legs. Neither surface measures end-to-end time, and fixed
  routes do not test token parity.
- **Transfers versus live.** Paid replay evicts at submit, serializes new copies at isolated p90, limits the batch and
  queue, then credits a destination only at a subsequent recorded row boundary. Already pending uploads use the logged
  claims and are sunk. Live `follow_loop()` alternates two pinned buffers and can overlap packing, H2D and decode;
  [`settle_swaps()`](../../../src/poly/engine.cpp#L1405-L1417) can wait before the next row. The mixed/code traces still
  log 173/126 live asks at scored positions. The replay debits the candidate's entire modeled service but does not
  measure or credit the live follower's exposed transfer cost. Actual contention, file/pool effects, and scoring
  overhead are also not modeled. This is a deliberately asymmetric screen, not a wall-time estimate or bound in either
  direction.
- **Holdout.** Train-only maps and timing fits use the first 128 tokens; past-16 uses only completed tokens before each
  decision. The original last 128 are a chronological holdout from the *same* request, not a new request. The frozen
  32/8 replay fits the timing surface and source prior on the mixed prefix; on the distinct code capture it consumes
  only routes before each decision for local history, with no code-suffix fit. The config hash matches the pin recorded
  before the code capture; that pins settings, not the complete implementation history. The replay reuses live row
  clocks and inherited claims; it does not predict how they would change. The frequency oracle fits on the suffix and
  the dynamic oracle reads future routes by design.
- **Intervals.** The 95% percentile bands resample paired token totals in 16-token blocks, with 120 draws and
  timing-surface resampling. They do not refit the policies or rerun their schedules. The clairvoyant mixed band is
  [2.28, 3.09] pack and [2.95, 3.77] pinned; frozen causal pack is [−0.02, 0.99] mixed and [−1.82, −0.44] code. These
  bands describe within-trace model/resampling sensitivity, not independent request replication, causal uncertainty,
  or A/B speedup confidence.

CPU-only checks: four simulator unit tests passed; raw ledgers and input digests were spot-checked. No GPU run or full simulation rerun.
