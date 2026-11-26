# Native DFlash2 production qualification on RDNA2

These are isolated tests of the production worker and protocol, not a rollout. The installed service and
its session files were not changed. Unless stated otherwise: RX 6700 XT (gfx1031), Ryzen 7 5700X,
Tideglass Inkling-2-Mini, dense Q8_0 and compact SWA, Release HIP, eight CPU workers.

`strata-poly --dflash2-model FILE` opts in. Use `--draft-max 0` for the same layout and tap updates but
no guesses, or `--draft-max 3` for K3. `--dflash2-budget-ms 65` opts into W1/lookup/DFlash2 selection
with an *estimated* round budget, not a per-token latency guarantee. DSpark remains a separate mode.
Use `STRATA_DFLASH2_CPU_REUSE=1` only on the checked Zen 3 build with matching ggml AVX2 arithmetic.
This mode does not combine DFlash2 with K256/V4 or with a second drafter. SAVE/RESTORE preserves the
target session; after restore, draft proposals resume after 2,048 newly evaluated rows.

## Wire quant screen

`measure.py` exercised native `GEN` and timestamped real `T` lines at the pipe reader. These are ID
gaps, not HTTP SSE event gaps. T=1/seed42, max 96 output IDs, stable prefill staging, 4,750 MiB fixed
expert+KV arena, no swaps, 1,092 initial expert slots in all DFlash2/W1 legs. All five DFlash2 files
matched every target ID and final route profile in both prompts. DSpark also matched the emitted IDs,
but started with 1,091 experts and did not match the final route profile. A higher ratio of DSpark
accepts does not imply faster total decode. BF16 is the drafter-weight reference.

| Worker | Reasoning tok/s | Accepted/offered | ID p90 ms | Tool prompt tok/s | Accepted/offered | ID p90 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Installed W1 | 30.60 | — | 36.76 | 33.10 | — | 38.14 |
| Installed DSpark | 37.52 | 46/47 | 76.40 | 38.81 | 25/27 | 67.76 |
| Q4_0 (two legs) | 44.27–44.63 | 66/84 | 75.73–75.82 | 52.76–52.92 | 49/54 | 70.68–72.45 |
| Q4_0-mixed | 40.48 | 65/87 | 81.76 | 52.85 | 49/54 | 70.37 |
| Q4_K-mixed | 40.95 | 65/87 | 80.10 | 52.64 | 49/54 | 71.38 |
| Q8_0 | 42.71 | 66/84 | 79.09 | 47.71 | 49/54 | 72.40 |
| BF16 | 39.13 | 66/84 | 87.77 | 47.49 | 49/54 | 78.12 |

First delivered ID took 1.50–1.62 s on reasoning and 1.95–2.03 s on the tool prompt, excluding
worker startup. Peak aggregate process RSS reported by `/usr/bin/time` was about 14.63 GiB for
installed W1, 14.89 GiB for Q4_0 and 15.42 GiB for BF16. It includes mapped file pages, not just
resident weights. The minimum device-wide sampled free VRAM was 4,331 MiB for installed W1,
3,965 MiB for Q4_0, 3,923 for Q4_0-mixed, 3,912 for Q4_K-mixed, 3,765 for Q8_0 and 3,410 for BF16.
This screen holds the arena size fixed. A second screen used the installed 2,048 MiB reserve,
64 live swaps and the same stable prefill controls, output limits, prompts and sampling. It did not
cap the expert arena. Both Q4_0 runs started with the same expert count; the other quants funded
fewer experts. Every arm matched all 96 reasoning and 67 tool-prompt IDs, but the expert placements
and route totals were not identical across quants.

| Worker | Initial experts | Reasoning tok/s | Reasoning accepted/offered | Tool tok/s | Tool accepted/offered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Installed W1 | 1,434 | 46.79 | — | 45.15 | — |
| Q4_0 K3 (two legs) | 1,332 | 30.88–31.52 | 53/124 | 26.93–27.61 | 35/93 |
| Q4_0-mixed K3 | 1,326 | 28.08 | 51/130 | 27.77 | 36/90 |
| Q4_K-mixed K3 | 1,319 | 29.29 | 53/124 | 27.68 | 37/90 |
| Q8_0 K3 | 1,284 | 29.50 | 54/122 | 26.27 | 35/93 |
| BF16 K3 | 1,202 | 25.29 | 52/127 | 24.26 | 37/90 |

Q4_0 stays the reasoning lead, has the lowest memory cost and is within the run-to-run spread of the
other Q4 quants on tool throughput. Neither Q4_0 nor any larger quant beats W1 with swaps here.
W1 delivered p90 gaps of 26.08–26.78 ms vs 59.29–60.26 ms for Q4_0 K3. On Q4_0 the verifier
replayed 33/42 reasoning rounds and 29/31 tool rounds. These prompts are not graded agent trajectories.

### Same-work comparison with the best RECON build

Reran the held-out billing seed 79 with the old exact combined binary and native Release, each with
loaded W1 and K3. All four arms matched the three prompts and all 643 sampled IDs (370 reasoning),
made the same two reads, and scored 118/118. K3 accepted 455 of 558 guesses in both builds. The
one-shot runner starts each turn in a fresh process with a private profile, 709 experts, a 3,072 MiB
fixed cap, zero swaps and `--top-probs 1` markers. Wall includes all three process startups; these
are sampling-marker gaps, not SSE delivery gaps.

| Worker | Task wall s | Decode tok/s | Sampling-marker p90 ms |
| --- | ---: | ---: | ---: |
| RECON loaded W1 | 40.65 | 24.23 | 45.98 |
| Native loaded W1 | 40.11 | 24.48 | 45.63 |
| RECON K3 | 33.01 | 34.02 | 101.45 |
| Native K3 | 31.95 | 35.67 | 97.01 |

Native K3 improved on the replicated RECON K3 run (about 4.8% higher throughput) and cut native
loaded-W1 time to correct by about 20%. It still more than doubled p90. This reproduces the fixed
placement gain, not a production gain with the installed settings.

Two follow-up arms on the same fixture also matched all 643 IDs and passed 118/118. K1 ran at
28.38 tok/s, 36.69 s wall and 71.50 ms p90 (310/331 accepted). An 85 ms budget with lookup enabled
ran at 24.98 tok/s, 39.54 s and 58.37 ms (80/113). This gives an actual latency/throughput trade
at fixed placement: K1 kept part of K3's speed; the adaptive arm kept much less. The budget is
an estimate and p90 here comes from the same CLI markers, not measured HTTP delivery.

`build_parity.py` produces a separate diagnostic ELF from the configured HIP Ninja build. Compare the
two diagnostic workers' logits, slot/recency/use fingerprints, saved profile and SAVE. Do not time that
binary or use it on production. Run each fresh leg with a private profile copy under
`rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults`; serialize GPU work. The snapshot fixtures and
raw logs belong only in the host-local run folder, not Git.

## Repeated RESTORE, live swaps

Same Q4_0 draft, W1/K3/115 ms policy, 64 swaps, 5,200 MiB fixed expert+KV arena. Each arm restored
the original target-only SAVE twice, appended 2,048 fresh rows and emitted 49 greedy IDs. Every ID
and the saved expert profile matched within each length. This repeats real history, but the suffixes
are controls rather than graded long-agent tasks. TTFT below excludes RESTORE and includes the append.

| Restored rows | W1 / K3 / adaptive tok/s | W1 / K3 / adaptive ID p90 ms | K3 replays/rounds | TTFT range |
| ---: | --- | --- | --- | --- |
| 65,792 | 30.95 / 19.14 / 30.97 | 38.09 / 93.44 / 37.59 | 21/21 | 4.88–5.28 s |
| 131,328 | 26.66 / 15.86 / 25.90 | 44.28 / 104.74 / 44.99 | 21/23 | 6.46–6.47 s |
| 197,021 | 20.39 / 12.98 / 20.20 | 58.16 / 134.31 / 58.63 | 21/22 | 8.38–8.44 s |

The policy speculated once per leg and replayed it in each case. Real placement changes are the
measured obstacle; the corresponding zero-swap RECON result (**26.18–28.11 tok/s** near 199K) is not
a competing live-placement baseline. Device-wide sampled free VRAM stayed above 2,454 MiB, with
about 18.05 GiB peak process RSS at the deepest context. No fresh guard stop.

The growth test separately restored 65,792 rows, then started generation just below the 69,632-row
mapped-KV boundary. During decode the engine grew to 73,728 rows and dropped from 940 to 925 experts.
Loaded W1 and Q4_0 K3 matched 25 greedy IDs, post-softcap FP32 logits, slot/recency/use fingerprints,
expert profile and complete SAVE. K3 made 11 draft rounds, 10 placement replays, and one W1 call.
Lower swap limits are not a performance remedy on the 96-ID Python fixture: at 1/4/16 slots in flight,
loaded W1 was 34.86/38.39/41.44 tok/s vs 25.63/26.46/29.78 K3, with 33/38/34 placement replays.
Every sampled ID and saved profile matched within each pair.

With 64 swaps, a separate two-turn cancellation control sent `STOP` after wire ID 9 or 18
and received 10 or 19 IDs respectively. K3 had already done placement replays. A W1 leg
limited to the same delivered length matched all IDs on that turn and on a 32-ID continuation,
every selected FP32 logit, the per-position fingerprints, both SAVEs and the full route profile.

A separate 64K diagnostic covered verifier endpoints K1 and K7 at T=1/seed42. Both matched
loaded W1's 45 sampled IDs, FP32 logits, per-row placement/recency fingerprints, profile and SAVE.
K1 matched **22/22** guesses but replayed 18/22 rounds. The fact that it lost even with perfect
token acceptance shows why acceptance alone cannot pick the width. Eliminating the measured
K1 wall gap in that instrumented run would require saving roughly 24 ms per placement replay;
that estimate is a scale check, not an uninstrumented timing comparison.

## Actual HTTP streaming and agent grade

`agent_api.py` launches an isolated authenticated loopback instance of `serve.server`, with its own
profile, slots and engine log. One free-running billing trajectory per arm, T=1/top-p=1/top-k=0/min-p=0,
seed 79, stock 2,048 MiB reserve and 64 swaps. Both bounded tool returns are used. The benchmark
starts when the first request is sent, not while the server loads. Delivery gaps exclude role-only
chunks and are SSE event gaps, not native token gaps. Reasoning/answer counts below re-tokenize text
delivered by SSE and need not sum to the exact native completion count.

| Worker | Initial experts | Grade | Wall s | Decode tok/s | SSE p90 ms | Completion IDs | Reasoning / answer (est.) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Installed dense W1 | 1,428 | 118/118 | 16.55 | 46.53 | 24.08 | 637 | 368 / 180 |
| Q4_0 loaded W1 | 1,313 | 118/118 | 12.13 | 43.84 | 26.07 | 404 | 144 / 171 |
| Q4_0 K3 | 1,314 | grader rejected `raise` | 19.56 (failed) | 38.17 | 56.82 | 634 | 349 / 196 |
| Q4_0, 85 ms adaptive | 1,311 | 118/118 | 14.48 | 43.12 | 48.99 | 494 | 234 / 171 |
| Installed DSpark | 1,309 | 118/118 | 13.88 | 58.58 | 53.76 | 628 | 368 / 171 |

All arms made two tool calls, with no retries. Cold first meaningful event took 1.32 s installed
W1, 1.31 s loaded W1, 1.40 s K3, 1.43 s adaptive, 1.51 s DSpark; warm TTFT was 0.80–0.92 s.
Grouping adjacent delivery events at gaps ≤2 ms, adaptive had 117/437 events in bursts, fixed K3
350/577 and installed W1 4/580. SSE tool parsing and host scheduling also affect the gaps; they
are not verifier round durations. Minimum sampled free VRAM was 1,889–2,003 MiB across arms.
The 85 ms policy used 55 DFlash2 and 31 lookup rounds over three turns, and replayed 48 times.

These are different free trajectories, visibly so in reasoning length and initial placement. The
adaptive arm beats installed W1 in one wall-to-correct sample, but not loaded W1 or DSpark, and
its p90 is about twice installed W1's. A failed grade here does not locate a target arithmetic
divergence, and cannot be excluded from performance qualification. Do not deploy on this result.

Second seed (42), same SSE gate: all arms scored 118/118. Installed W1 took 10.06 s (p90 27.80 ms),
loaded W1 11.56 s (26.88 ms), DSpark 9.63 s (56.26 ms), fixed K1 14.45 s (40.71 ms), adaptive
65 ms 12.89 s (26.70 ms) and adaptive 105 ms 11.51 s (27.49 ms). The two policies preserved a W1-like
gap p90 by running serial most of the time: 415/425 rounds at 65 ms and 303/332 at 105 ms. Fixed K1
replayed 72/244 draft rounds. Neither adaptive arm beats installed W1 or DSpark on this seed.

## Real long-history tasks over the native protocol

The private `global-kv-freeagent-confirm/dflash_long_ab.py` uses the same independent coding
and reasoning judges, followed by the model's real tool replies, with no forced output. Both
arms used the Release candidate and loaded Q4_0, T=1/seed42, 64 swaps, the same frozen profile,
5,200 MiB expert+KV cap and the production prefill settings. W1 and K3 both began with 1,197
experts. `RESUME` and `DONE` proved exact prefix reuse on every turn. At 64K and 128K, the
first request had 2,048 archived warm rows before the new question. At 197K, it did not: a
warm append would leave too little room for the unchanged task caps. The K3 arm offered no
guesses at 197K, making that group a cold-restore/session control, not a DFlash2 speed test.

| Saved rows and task | W1 / K3 grade | W1 / K3 wall s | W1 / K3 tok/s | W1 / K3 ID p90 ms | K3 accepted/offered; replays/rounds |
| --- | --- | ---: | ---: | ---: | --- |
| 65,796 coding | 118/118 / length | 23.02 / 30.17 | 32.60 / 26.88 | 35.83 / 80.87 | 390/722; 158/241 |
| 65,796 reasoning | length / 4/4 | 23.64 / 25.54 | 36.18 / 28.99 | 34.42 / 77.45 | 338/621; 127/207 |
| 129,015 coding | 118/118 / 118/118 | 22.78 / 26.76 | 26.37 / 19.51 | 43.84 / 102.45 | 214/426; 111/142 |
| 129,015 reasoning | 4/4 / 4/4 | 28.51 / 31.23 | 27.10 / 21.04 | 46.19 / 106.08 | 290/555; 122/185 |
| 197,021 coding, cold | 118/118 / 118/118 | 24.43 / 32.10 | 20.66 / 19.87 | 53.93 / 55.63 | — |
| 197,021 reasoning, cold | 4/4 / 4/4 | 31.50 / 28.76 | 21.99 / 22.27 | 54.84 / 54.11 | — |

Task wall starts after RESTORE. `length` has no time to correct. All runs made two successful
tool calls with no retries. First delivery took 4.59–4.71 s at 64K, 6.19–6.44 s at 128K and
1.47–1.50 s at 197K (no warm append). Trajectories differed within each pair, even with no
guesses at 197K. The first-turn common output prefixes were 16/2, 13/2 and 28/2 IDs for
coding/reasoning at the three respective lengths. Do not attribute those differences to verifier
arithmetic without a matched-prefill control. The four exercised K3 legs all had lower
decode throughput and higher ID gap p90 than the corresponding W1 arm.

Completion IDs, including EOS and framing, were W1/K3 538/632 and 612/546 at 64K,
379/357 and 532/475 at 128K, and 421/557 and 599/547 at 197K (coding then reasoning).
The corresponding reasoning-ID counts were 294/401 and 518/407, 164/116 and 396/344,
and 211/316 and 465/408, counted after the native reasoning marker. Different reasoning
lengths in the cold group cannot be a DFlash2 effect: neither arm proposed any guesses.

The matched-prefill repeat disabled CPU-share/stage-ahead/prefetch and kept 64 decode swaps.
At 64K coding, both arms produced the same three prompts and all 350 output IDs, scored 118/118,
and took 18.90 s W1 vs 24.32 s K3. ID gap p90 was 35.73 vs 86.21 ms; K3 replayed 113/141 windows.
At 128K reasoning, all three prompts and 612 output IDs matched, including the length failure in
both arms. W1/K3 took 31.41/35.03 s, p90 42.39/96.09 ms, with 132/223 K3 replays. These controls
reproduce the latency penalty on identical real-agent work; they do not qualify production prefill.
The matched coding outputs had 109 reasoning and 155 answer IDs in both arms; the matched
reasoning outputs had 528 reasoning IDs and no final answer.

## Proposer reprofile: final layer only

The private final-layer experiment kept all eight bidirectional K/V rows but projected only queries
and downstream rows `0..K` in the fifth layer. Row 0 is needed by the one-back convolution. Q4_0
replay matched hidden states, all top candidates, unary scores and selector output at K=1–7 on
short, 2K and 64K features. At 2K, proposal-only K3 went from 6.668 to 6.251 ms. In the live 64K
trace, the median sum of proposal kernels went from 6.109 to 5.968 ms, with FFN projections
1.578 → 1.409 ms, softmax/V 1.602 → 1.471 ms and the shared Q6_K head unchanged at about 1.12 ms.
Untraced 64K decode was 20.60 → 20.62 tok/s and 86.90 → 86.79 ms p90, with 21/21 placement replays
in both arms. An alternating short screen did not improve decode either: stock 27.79–28.64 tok/s
vs 25.94–26.63 with the change at 64 swaps. This is a kernel saving, not a qualified engine gain.
The head uses wave32 DP4A with LDS activation loads and no spills in the verified K3 gfx1031 ISA;
the V accumulation uses aligned vector loads, but still rereads V for each of the four GQA heads.
The full Q6_K head is about 378 MB, or about 338 GB/s of nominal weight reads for the 1.12 ms
projection; faster top-16 alone will not avoid that scan. In the same device trace, the median
single-row draft append had 0.144 ms of kernel time over a 0.240 ms device interval (38 appends,
from tap gather to the last V copy). Hiding appends can save latency, but not the measured decode gap.

At 64K with zero swaps, the native CPU weight-unpack reuse switch improved alternating K3 runs
from 37.25–37.54 to 38.52–38.60 tok/s (about 3.1% on the paired medians). Wire ID p90 fell from
97.64–98.88 to 93.88–94.50 ms. Same 49 greedy IDs, profile, 34/39 accepted/offered, and 14 windows.
Just over half of committed routes ran on the CPU. The host's `amd_df` DRAM events did not open
under `perf stat`, so this is an engine speed measurement, not a measured drop in DDR4 bytes.
An earlier 64-swap short pair only moved from 28.47 to 28.69 tok/s; the switch does not solve replay.

On the traced 64K run, ordinary anchor placement took a median 0.114 ms and the swap-stream wait
0.016–0.017 ms with tracing enabled. Uploads had finished at least 51.98 ms before every measured
replay settlement; their summed DMA duration is not removable waiting time. The much larger cost
is the verifier and the W1 replay body (~82 and ~35 ms with tracing). Parallel proposal/anchor
execution could hide at most the roughly 7 ms proposer in that trace without also changing the
target work. Use a contention-only W1/concurrent-proposal test before trying a suffix verifier.

### Grouped V accumulation experiment

The scratch prototype packed softmax weights by key and reused each V load across four GQA heads.
It kept the original FP32 reduction/FMA order. It matched all 88 synthetic attention cases and
matched proposals, hidden rows, candidates and selector scores in the short/2K/64K replay matrix.
The grouped V kernel used 16 VGPRs, 4 KiB LDS and no spills on gfx1031. It makes four times fewer V
byte requests within a group at the instruction level; that is not a measured fourfold drop in DRAM bytes.
At 64K, untraced replay K3 fell from 6.72 to 5.31 ms/proposal; K1 and K7 improved by about 1.39 ms too.

Live 64K ABBA with 64 swaps: stock 20.63–20.67 vs grouped 20.53–20.82 tok/s, with 21/21 placement
replays. Without swaps: stock 35.67–37.93 vs grouped 33.41–36.31 tok/s. In each pair all 49 IDs, the
saved expert profile and accepted/offered counts matched. Neither live screen reproduced the
proposal-only win in total decode. The candidate and its ISA stay in the private experiment folder.

### Where placement invalidates a verified row

The `replay-prefix-diag` worker records the first logical visit whose routed expert changes
CPU/GPU backend. Logical visits, not physical weight layers, are what a partial replay could
skip. Loaded W1 and K3 matched the selected FP32 logits, position-by-position placement/recency
fingerprints, generated IDs, route profile and full SAVE in all four diagnostic pairs.

| Context | Placement replays | First changed visit (zero-based min / median / max) | Potentially reusable visit heads |
| --- | ---: | ---: | ---: |
| 74-row Python prompt | 36 | 0 / 8.5 / 43 | 536/1,728 |
| 65,792 + 2,048 rows | 21 | 0 / 3 / 41 | 172/1,008 |
| 131,328 + 2,048 rows | 21 | 0 / 8 / 46 | 379/1,008 |
| 197,021 + 2,048 rows | 21 | 1 / 6 / 45 | 192/1,008 |

The upper bound credits the attention head of the first changed visit, but not its MoE or any
later visit. It ignores the cost of capturing and restoring the split recurrent state, residual
links and taps. At 64K, even removing the entire traced median replay body (~35 ms) would not
cover the ~47 ms per replay needed for fixed K3 to tie W1 in the separate wire run. With the first
conflict usually early, no partial-repair path was added.

### Concurrent proposal contention

A scratch build ran loaded W1 while a reusable worker proposed K3 from the pre-anchor committed
taps on the draft stream. It discarded the guesses and joined before updating draft KV. The paired
64K wire controls used 64 swaps and made 48 W1 steps per leg. All six arms matched the 49 greedy
IDs, route counts and final profile. Both active modes completed all 48 proposals.

| Execution | Median wall ms/W1 step | Wire ID p90 range, ms |
| --- | ---: | ---: |
| W1 alone | 31.38 | 36.81–37.32 |
| W1 + serialized proposal | 38.05 | 43.57–44.11 |
| W1 + concurrent proposal | 33.55 | 38.50–38.58 |

Overlap hid 4.50 of the 6.67 ms incremental serialized cost but still added 2.17 ms/step.
This experiment did not verify a suffix.

### W1 anchor followed by conditional suffix

A separate scratch worker proposed K3 from the same committed taps, ran W1 for the anchor,
sampled and emitted its target token, then entered exact W3 only on a first-guess match. At
65,792 restored rows plus 2,048 fresh rows, 64 swaps and a 5,200 MiB arena, four alternating
legs matched all 49 greedy IDs, route counts and the saved profile. In both suffix legs all 14
first guesses matched, but all 14 W3 windows needed a placement replay; only 18 of 42 suffix
rows were kept per leg.

| Execution | Median decode wall ms | Wire ID p90 range, ms |
| --- | ---: | ---: |
| Loaded W1 | 1,497 | 36.72–37.25 |
| W1 anchor + conditional W3 suffix, serialized | 1,903 | 62.61–63.10 |
| Same suffix, subtracting all measured proposal time | 1,809 | — |

Subtracting the entire proposer time is an optimistic overlap estimate, not another measured
arm. It still leaves the suffix about 21% slower than W1. This does not justify implementing
the concurrent suffix path. The private binary and receipt are under `anchor-first-suffix.*/`.
