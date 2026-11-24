# Goal: Raise Inkling-2-Mini Q4_0 decode throughput on RDNA2

Status: complete
Source: 2026-11-24 user request to implement the Q4_0 RECON, iterate on evidence, and ship verified wins.

## Objective and gates

- **R1: Ship repeatable end-to-end decode wins.** Compare untraced, interleaved runs on the same card, Q4_0 GGUF,
  token sequence, and comparable placement. For each verified win, push the opt-in change to `main`, deploy its
  Release HIP binary, and use it as the next baseline. Status: verified (`a3fe529b`, `7ce0ea6a`, direct A/B below).
- **R2: Preserve answer quality.** Keep quantization, formulas and accumulation order. Compare affected boundary
  bytes, raw greedy IDs and routes on the same device and matched CPU/GPU placement. Report the first difference
  if one occurs; do not ship a divergent candidate. Status: verified (real boundary dumps, route records and greedy IDs).
- **R3: Reprofile verified wins and follow measured bottlenecks.** Continue while there is a concrete, falsifiable
  experiment; close with the measured wins, rejected hypotheses, remaining limits and final binary. Status: verified
  (post-deploy profiles and follow-up trials below).

Scope: Inkling-2-Mini Q4_0, HIP gfx1031, Q8_0 KV, existing CPU fallback and 2 GiB VRAM reserve. Defaults,
model file, sampling, KV/session format, and no-DSpark deployment remain unchanged. Guard standalone GPU runs;
keep logs and experimental tooling outside Git. The order below is a map of hypotheses, not a fixed sequence.

## Checkpoint evidence

Starting baseline `d0b8082e`, deployed engine `6d7e00d3` (SHA-256
`4c36f3c427ce215fc7fbb77720cfd7f45c6ee81813cca36e9e2971a790f53f37`). RECON artifacts on the host:
`~/.cache/opencode-tmp/opencode/inkling-q40-recon.j3z78x3d/PLAN.md` and adjacent traces, fixtures and scripts.
RX 6700 XT / Ryzen 7 5700X, 32 GiB. The RECON geometry-only prototype gave 46.50 → 48.21 tok/s at 2K + 256
(four positive pairs); no repeatable gain at 16K. Real Q4_0 gate/up, SiTU Q8_1, and down dumps and three sets
of 96 greedy IDs matched byte for byte. At 64K, exposed CPU joins had a median of 5.86 ms/token and full
attention consumed 3.72 ms/token of kernel time.

- The RECON rejects simple Q4_0 repacking, CPU task-size tuning and sharing chunk weights/denominator via LDS
  in decode attention combine. These are not on the implementation path without materially new evidence.
- Four-row slots, Release `gfx1031`: medians, 256 forced tokens and four adjacent off/on pairs per workload.
  2,048 prose 45.695 → 47.660 tok/s (+4.30%, pairs +3.86..+7.62%, 88.9% hits);
  4,113 code 44.775 → 46.160 (+3.09%, pairs +2.44..+9.70%, 85.7–86.3% hits);
  16,384 code 44.070 → 44.845 (+1.76%, pairs +0.80..+3.18%, 86.8–87.3% hits). Median prompt rates
  893.65 → 896.35, 937.55 → 942.45, 1,052.40 → 1,057.00 tok/s (no measured slowdown).
  For each context, 96 greedy IDs, 4,608 ordered route/placement records, and the real Q4_0 gate/up FP32,
  post-SiTU Q8_1 and down FP32 bytes match (94,733,856 bytes total, at positions `[prompt+2,prompt+8)`).
  Restoring the same 65,536-token snapshot in fresh processes (73.13–73.17% hits), 96 greedy IDs also
  matched in all four runs. The ABBA median was 32.414 → 32.212 tok/s; the pairs were +0.78% and −2.02%,
  so the short-context gain does not carry over reliably to this long control.
  `poly_mmvq_segments_test` and `poly_inkling_attend_test` passed. Logs:
  `~/.cache/opencode-tmp/opencode/inkling-q40-impl.fFrvEXeU/` (`summary.py`, `greedy-*`, `test-*`).
- `a3fe529b` runs as `~/.config/strata/inkling2-mini/strata-poly-a3fe529b`, SHA-256
  `d9ddfca33a0bbc139fb892af670837dea40a712d6dca627b3382ac2333d438e0`. HTTP health, models, text chat and
  79-token disk save/restore passed. Rollback is `before-q4-0-a3fe529b.json.bak` in that directory.
- Reprofiled after four-row slots (55 complete decode intervals per prompt, same instrumentation as RECON).
  At 2K the slots consumed 4.26 ms/token, dense Q4_0 projections ~2.14, GPU busy 13.61. At 16K the slots
  took 4.58 ms/token and the CPU branch was exposed for median 3.32. At 64K the exposed CPU join was 4.86
  ms/token, full attention 3.66 and slots 3.98. These are not additive wall-clock attributions.
- At 2K, grouped four-row Q/gate/relative reduced 124 projection launches to 48 and their combined time
  from 1.44–1.47 to 1.14–1.15 ms/token; grouping alone took 1.33, four rows without grouping 1.32. Four or
  two rows for o-projection alone saved at most 0.04 ms/token. It is not part of the next candidate.
- Dense scratch relink, slots on: all four adjacent pairs at each context favored grouping, medians
  48.270 → 50.035 (2K), 47.495 → 48.620 (4K), 45.875 → 46.605 (16K), 320 forced tokens each.
  Restored 64K with 96 greedy IDs exact: 31.610 → 31.922 tok/s, pairs −1.49% and +3.50%; inconclusive.
  The Release binary `dense-release` is built (SHA-256
  `83823024347c67ffec89dfdadb90a485e98172aa27cb453304476600e4205820`). On its parity relink, at each of
  2K/4K/16K, 96 raw greedy IDs, 4,608 ordered route/placement records, and the actual FP32 projections plus
  expert chain dumps matched; 124,372,512 boundary bytes total. `poly_mmvq_segments_test` passed.
- Clean Release (`dense-clean-*`, slots on): 2K 48.830 → 50.125 tok/s (+2.65%, four pairs +2.36..+4.53%);
  4K 47.635 → 48.520 (+1.86%, two pairs +1.16..+2.56%); 16K 46.035 → 47.095 (+2.30%, two pairs +1.72..+2.89%).
  320 forced tokens per run. Prefill did not slow outside its spread. At restored 64K, 31.964 → 32.779 tok/s
  (+2.55%, two pairs +1.60% and +3.55%, 72.62% hits in every run), same 96 greedy IDs. Earlier timings from the
  series stopped at 1,092 MiB free VRAM were excluded; no GPU fault and no guard threshold was changed.
- `7ce0ea6a` was pushed to `origin/main` and installed as `~/.config/strata/inkling2-mini/strata-poly-7ce0ea6a`
  with both Q4_0 flags set. HTTP health, models, text chat and a 79-token disk save/restore passed. The binary
  is byte-identical to `dense-release`; private rollback is `before-q4-0-7ce0ea6a.json.bak`.
- Final Release traces (`final-trace-*`, before any scratch-only trials): 1,013 launches per full decode token
  versus 1,089 before grouping, about 1.24 ms/token in grouped dense Q4_0 at 2K and 16K. The Q4_0 expert
  gate/up+down still took 4.65 and 4.58 ms/token respectively. Restored 64K spent 3.70 ms/token in full
  attention, 3.84 in the Q4_0 expert products, and 1.03 in the Q6_K head. These are GPU times, not wall times.
- On real rotated CPU Q4_0 experts at eight workers, paired AVX2 gate/up yielded 225.39 → 219.40 µs
  for two experts (overlapping interquartile ranges); fusing with intermediate quantization did not help.
  Hoisting the activation work mattered at one worker (756.87 → 675.05 µs) but much less at eight
  (225.39 → 219.45 µs). No CPU changes are shipped on this evidence.
- Real paired CPU gate/up (`cpu-ab16k-*`, `cpu-restore64-*`): four 256-token pairs each, 16K medians
  47.475 → 47.735 and restored 64K 37.481 → 37.613 tok/s; two pairs lost at each length. At 64K the 256
  greedy IDs matched. In a separate 96-token 64K trace at 73.60% hits, 60 matched decode windows, exposed
  host join 7.941 → 7.581 ms/token median; gate/up 8.625 → 8.499. With eight workers the real two-expert
  work achieved 44.7 GB/s effective payload in a follow-up, close to 46.0 GB/s for just reading the same
  bytes with a checksum. Expert-major down loops did not give a reliable gain either. No CPU patch.
- Wider raw Q4_0 tiles (`wider-slots`): cold seven-expert gate/up 70.88 µs with four rows vs 71.44 / 72.16 /
  98.08 for 6/8/12; down 41.72 vs 42.16 / 42.72 / 43.72. Swapping the launch axes did not help. DPP cut
  at most ~0.5 µs in these cold calls and cost ~4.8 µs on hot seven-expert gate/up. All were bitwise.
  Seven cached experts in the real short decode take about 61/34 µs (gate/up/down), between the flushed
  and fully hot microbench. The marginal projected payload rate across 3–8 experts is 372–385 GB/s vs
  355 GB/s for the separate cold 256 MiB read, consistent with some cache reuse, not an estimate of DRAM traffic.
- Mixed Q4_0/Q8_0 grouping saved 48 launches a token (1,013 → 965) in four 2K traces. GPU time for the
  grouped projections barely changed (2.63–2.70 → 2.62–2.63 ms/token including the untouched K-quants).
  Untraced A/B (320 tokens, four pairs): 2K 51.045 → 51.860 tok/s, all pairs won; 4K 48.990 → 49.355,
  two pairs lost and hit rate also varied. In a fresh six-leg 2K trace (`mix-rows-*`, all 86.1% hits),
  four rows for Q8_0 K/V alone took 1.036 ms/token vs 0.938–1.004 without it. Mixed with four-row Q8_0
  took ~2.13 ms/token versus 2.05 for mixed with one row (same number of launches). Retain neither.
- Raw weight load/store (`weight-loads`): aligned dwords with `alignbyte` worsened cold seven-expert gate/up
  70.84 → 72.40 µs; only using dwords on naturally aligned blocks had no effect. One coalesced store for all
  four outputs saved <1 µs gate/up and <0.6 µs down, not enough to move decode. All boundary bytes matched.
- Attention with fixed chunk bounds: nine variants matched partial, FP32 and Q8_1 bytes at 1K SWA and 2K,
  16K, 64K and 190K full. At 64K, eight waves took 450.28 µs vs four waves at 722.45 and two at 1,096.09;
  DPP8 and prefetching one key per lane each saved about 4 µs per full visit, too little for a decode switch.

## Direct end-to-end comparison

Original deployed `6d7e00d3` vs final deployed `7ce0ea6a`, RX 6700 XT (`gfx1031`), Ryzen 7 5700X, Bartowski
Q4_0, Q8_0 KV, eight CPU workers, 2 GiB VRAM reserve and no lookup/DSpark. Fresh processes, same profile copied
for every run, untraced and interleaved. Decode-only, medians of measured tok/s; percentages are ratios of
medians, not sums of patch gains. The first three prompts use 320 fixed continuation IDs. The long run
restores the same 65,536-token prefix, reads one new prompt token and generates 256 greedy IDs.

| Prompt | Original | Final | Change | Adjacent pairs won |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 47.930 | 51.325 | +7.08% | 4/4 |
| 4,113 | 46.785 | 49.325 | +5.43% | 2/2 |
| 16,384 | 44.275 | 47.990 | +8.39% | 2/2 |
| 65,536 restored | 37.206 | 38.140 | +2.51% | 4/4 |

All 24 legs passed the guard (at least 1,900.9 MiB free). On the short runs, equal or comparable expert hit
rates (89.6%, 86.8%, 87.0–87.5%); long runs 81.11–81.57%. Raw IDs match in every run in their respective
fixtures; the forced runs verify equal decode work, while the independent exact quality checks above use
greedy outputs. Prefill medians at 2K/4K/16K were 920.3 → 888.8, 958.9 → 954.1, and 1,057.4 → 1,061.2
tok/s. The two slow 2K prefills did not reproduce in an independent prefill-only A/B with four pairs:
912.9 → 917.6 tok/s, paired changes −0.40% to +1.84%, same first output ID.

## Completion

- Final Release HIP binary: `build-poly-hip/strata-poly`, byte-identical to the running `strata-poly-7ce0ea6a`
  above, with `STRATA_INKLING_Q4_0_SLOTS=1` and `STRATA_INKLING_Q4_0_DENSE=1` in the service only. Both
  flags default off in the engine. Model, formulas, accumulation order, KV and session format did not change.
- `cmake --build build-poly-hip --target strata-poly poly_mmvq_segments_test poly_inkling_attend_test -j4`;
  both targeted tests passed on `gfx1031`. `/health` returned 200 with the model loaded after restoring the service.
  Comparing both flags on versus both off at matched placement also gave all 96 greedy IDs and all 4,608
  ordered route/placement records at each of 2K, 4K and 16K; 256 restored-64K greedy IDs matched the original binary.
- Remaining limits: expert weight reads dominate short decode. At 64K, exposed CPU misses depend on expert
  placement and leave several ms/token on the critical path; the full Q8_0 KV scan adds 3.70 ms/token of GPU
  work. The measured low-risk changes left no candidate with a repeatable further gain. No result is claimed for
  another GPU, model quantization, or the whole context range (190K was an attention microbenchmark only).
