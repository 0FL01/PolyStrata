# Goal: Raise Inkling-2-Mini Q4_0 decode throughput on RDNA2

Status: active
Source: 2026-11-24 user request to implement the Q4_0 RECON, iterate on evidence, and ship verified wins.

## Objective and gates

- **R1: Ship repeatable end-to-end decode wins.** Compare untraced, interleaved runs on the same card, Q4_0 GGUF,
  token sequence, and comparable placement. For each verified win, push the opt-in change to `main`, deploy its
  Release HIP binary, and use it as the next baseline. Status: in progress.
- **R2: Preserve answer quality.** Keep quantization, formulas and accumulation order. Compare affected boundary
  bytes, raw greedy IDs and routes on the same device and matched CPU/GPU placement. Report the first difference
  if one occurs; do not ship a divergent candidate. Status: in progress.
- **R3: Reprofile verified wins and follow measured bottlenecks.** Continue while there is a concrete, falsifiable
  experiment; close with the measured wins, rejected hypotheses, remaining limits and final binary. Status: pending.

Scope: Inkling-2-Mini Q4_0, HIP gfx1031, Q8_0 KV, existing CPU fallback and 2 GiB VRAM reserve. Defaults,
model file, sampling, KV/session format, and no-DSpark deployment remain unchanged. Guard standalone GPU runs;
keep logs and experimental tooling outside Git. The order below is a map of hypotheses, not a fixed sequence.

## Current checkpoint

The dense candidate passed the Release and exact-output gates on top of `a3fe529b`.
Promote that binary with `STRATA_INKLING_Q4_0_DENSE=1`, then trace the new baseline at 2K, 16K and 64K.
Test paired CPU dots in the 16K/64K engine, mixed-format projection launches, and whether wider expert
tiles improve enough over the shipped four rows to justify another in-engine trial.

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
- On real rotated CPU Q4_0 experts at eight workers, paired AVX2 gate/up yielded 225.39 → 219.40 µs
  for two experts (overlapping interquartile ranges); fusing with intermediate quantization did not help.
  Hoisting the activation work mattered at one worker (756.87 → 675.05 µs) but much less at eight
  (225.39 → 219.45 µs). No CPU changes are shipped on this evidence.
- Fixed chunk bounds attention: nine variants matched partial, FP32 and Q8_1 bytes at 1K SWA and 2K, 16K,
  64K, 190K full. At 64K, eight waves 450.28 µs vs four waves 722.45 and two 1,096.09; DPP8 and one key
  prefetched per lane saved ~4 µs per full visit. The extra workgroups spread K/V reads across fewer groups
  at once; the small gains from the instruction/prefetch changes do not justify a decode switch.
