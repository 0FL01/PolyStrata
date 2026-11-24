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

Baseline `d0b8082e`, deployed engine `6d7e00d3` (SHA-256
`4c36f3c427ce215fc7fbb77720cfd7f45c6ee81813cca36e9e2971a790f53f37`). RECON artifacts on the host:
`~/.cache/opencode-tmp/opencode/inkling-q40-recon.j3z78x3d/PLAN.md` and adjacent traces, fixtures and scripts.
RX 6700 XT / Ryzen 7 5700X, 32 GiB. The RECON geometry-only prototype gave 46.50 → 48.21 tok/s at 2K + 256
(four positive pairs); no repeatable gain at 16K. Real Q4_0 gate/up, SiTU Q8_1, and down dumps and three sets
of 96 greedy IDs matched byte for byte. At 64K, exposed CPU joins had a median of 5.86 ms/token and full
attention consumed 3.72 ms/token of kernel time.

1. Verify actual production integration of the raw four-row Q4_0 geometry: both short workloads, 16K/64K controls,
   HIP build, boundary and greedy checks. Reprofile after deployment if it passes.
2. Measure Q4_0 dense projection grouping and geometry independently. Drop a change if throughput is inside the spread.
3. For CPU multi-miss visits, try an exact paired gate/up dot on real Q4_0 weights. Test overlapping CPU/GPU work
   before adopting any isolated CPU win. Revisit long attention only if the traces justify it.

## Checkpoint evidence

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
