# DFlash2 research on RDNA2

Inkling 2 Mini / Tideglass v2.1 on RX 6700 XT (`gfx1031`) and Ryzen 7 5700X. These are isolated
builds; they do not replace the installed `strata-poly` or start a new service. The production
control is the installed dense global Q8 + compact SWA engine with no drafter. K256 and V4 are
separate target arithmetic and are off in that control.

| Question | Reproducible experiment |
| --- | --- |
| What does the original model compute? | [CPU reference and literal donor graph](reference/README.md), including feature ordering, hidden states, candidate correction and softcap. |
| Can HIP draft it? | [Native replay and ABI](native/README.md); [exact draft attention and profiling](native/opt/README.md); [DP4A projection experiment](projection/README.md). |
| Which quants propose the right IDs? | [Paired native replay](capture/README.md), [FP32 reference and tensor-precision probes](reference/README.md), [real tool-trajectory replay](trajectory/README.md). |
| Can the selector stay in system RAM? | [Codebook benchmark](codebooks/README.md) and [live host/GPU placement](codebooks/NATIVE.md). |
| What would a perfect drafter buy? | [Oracle width/context sweep](verifier/README.md) with the same physical resident expert IDs. |
| Does batching keep the target's arithmetic/state? | [First divergence and exact live overlay](divergence/README.md); [CPU-expert reuse](reuse/README.md); [shared-query Q8 attention](multiq/README.md). |
| Is end-to-end work faster? | [Live guarded timing](live/README.md) and [grading with real tool returns](trajectory/README.md). |

Use the [active campaign record](../../docs/goals/2026-11-26-native-dflash2.md) to locate the build
and its matching receipts. Keep prompts, states, weights, raw scores, private seed profiles and
generated output files outside git. All GPU legs must hold the campaign's shared `gpu.lock` and
use `rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults`.

Interpret the measurements separately. **Replay** measures proposals on a fixed history and
has no verifier cost. The **oracle** measures an upper bound with perfect inputs, not the live
drafter. **Same-work speed** requires equal output IDs and, for isolating a kernel or layout, equal
physical expert placement and no swaps. **Natural placement** prices the VRAM taken by draft
weights, but can change the target's CPU/GPU choice and therefore the completion. A faster
different completion is not a same-work speedup. One-shot `TTFT` and per-token p90 here are
internal token-ready times, not the API's delivery latency.

For an experimental first choice, use Q4_0 and the exact target overlay, and compare K=3 with K=0
at the same zero-swap placement before widening. The held-out tool task shows the trade-off: same
643 sampled IDs and a shorter time to the correct answer, but more than doubled step p90. There is
no production default change. Wider verified windows, live swaps and context-growth boundaries
must have their own checks; matching greedy IDs alone does not establish sampling parity.

## Full-SWA quant replay

The table uses the original HIP DSO and holds each captured history fixed. Each context has 25
K=7 anchors and just 4–5 simulated closed-loop rounds. At 199,069 the archived prefix is different;
the change in acceptance cannot be attributed to context length alone. Proposal p50 pools all 125
anchors per profile and excludes the target, feature upload, and append.

| Draft | Mean accepted at 2,048 | 18,688 | 67,840 | 133,376 | 199,069 | Draft p50, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 6.16 | 6.96 | 6.76 | 6.96 | 5.44 | 16.38 |
| Q8_0 | 6.16 | 6.96 | 6.76 | 6.96 | 5.32 | 10.67 |
| Q4_0 | 6.72 | 6.72 | 6.20 | 6.88 | 5.08 | 9.29 |
| Q4_0-mixed | 6.48 | 6.64 | 6.24 | 6.92 | 5.44 | 9.45 |
| Q4_K-mixed | 6.24 | 7.00 | 6.84 | 6.96 | 5.68 | 10.79 |

```sh
python -B rdna2-rnd/dflash2/live/summarize.py "$R/native-long" \
  --resume "$R/native-long-resume" --domains 2k 16k 64k 128k 199k
```

Q4_0 was the quickest full-SWA proposer and had the smallest backbone. Q4_0-mixed led the
seven-domain common-target replay at **4.139 vs 3.931** closed-loop tokens/verify. That difference
does not, by itself, pay for a larger footprint or a longer proposal; the two need a live A/B at
the same headroom and at fixed placement. In the host F16 codebook test, putting 195.37 MiB in VRAM
displaced 40 experts on natural placement and did not win the fixed-placement live timing.
Keep mmap as the engineering default. Quantizing only those two F16 books to Q8_0 saved **91.58 MiB**
of file/RAM payload, but changed 16/2,149 native proposal IDs on Python/tool calls; Python closed-loop
prefix length fell from 2.701 to 2.647. Keep that as a memory option, not a quality-equivalent default.

Changing only the ten residual projections to IQ4_XS, IQ4_NL, Q5_K or Q6_K on a Q4_0 backbone
did not establish a new winner. The native Python/tool screen measured pooled draft p50 of
5.77 / 6.03 / 6.41 / 6.33 ms, with mean accepted prefixes in the simulated walk of
2.877 / 2.877 / 2.829 / 2.738. The stock Q4_0 has 2.783 on these two captures (83 walked rounds)
at 5.79 ms proposer p50.
Results and converter provenance are in `$R/{additional-quants,residual-screen,q8books-native}`.
