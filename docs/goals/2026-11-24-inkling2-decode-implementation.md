# Goal: Raise Inkling-2-Mini decode throughput on RDNA2

Status: complete
Source: 2026-11-24 user instructions to implement the decode RECON, iterate on evidence, and ship verified wins.
Last updated: 2026-11-24

## Objective

Increase end-to-end decode tokens/s for Inkling-2-Mini on the RX 6700 XT without losing answer quality.
Use the profiling plan as a map of hypotheses; change direction when a measurement disproves one.

## Frozen Contract

- R1: Ship repeatable end-to-end decode wins. Acceptance: same-card, same-model interleaved A/B with the same
  tokens and comparable placement; each accepted change builds, lands on `origin/main`, and passes a local
  production smoke check. Evidence: two positive short-context tests on the final path, a neutral old/new
  cross-check, `e72e40d7` on `origin/main`, and the live API check below. Status: verified.
- R2: Preserve answer quality. Acceptance: compare raw greedy IDs at the same settings and placement for each
  change; check changed arithmetic at its boundary and resolve candidate-specific divergence before shipping.
  Evidence: byte-identical FP32/Q8_1 boundary dumps and all 96 greedy IDs on each of four matched-placement
  fixtures; no first differing ID. Status: verified.
- R3: Reprofile after a substantial win and continue while there is a concrete prospective test. Acceptance:
  close tested hypotheses with measured outcomes, report remaining limits, and provide the final HIP binary.
  Evidence: final Release trace, correlated 16K/64K host traces, follow-up trials and installed binary hash.
  Status: verified.

Constraints: gfx1031 HIP and system RAM only; no change to the configured 2 GiB VRAM reserve, quantization,
attention chunking or floating-point reduction order for the initial experiments. New production paths are opt-in;
the old path remains available. Use the RDNA2 guard for standalone GPU runs and keep production on loopback.

## Change Envelope

Inkling decode kernels in `src/kernels/cuda/iq_kernels.cu` and `src/poly/`, their immediate dispatch and CPU
expert path, and the measured-results docs. Temporary profilers and microbenchmarks stay outside the tree unless
needed for a concrete regression. No new dependency, API, or service.

## Baseline and artifacts

Baseline `992ebbb0` (`cedd3bf3` production binary, SHA256 `607933d4190d067246f14ce6313bea0d05b7addf0287aa1435339277d40efe81`).
Only scale-select shipped, opt-in. RECON and implementation artifacts on the test host:
`~/.cache/opencode-tmp/opencode/inkling-decode-recon.Sy4nkYZ3/` and `~/.cache/opencode-tmp/opencode/inkling-decode-impl.C2bRhNAk/`.

## Checkpoint evidence

- The final Release binary at 2K + 512 forced tokens, stock settings (RX 6700 XT and Ryzen 7 5700X, Inkling-2-Mini
  UD-IQ4_XS, Q8_0 KV, 200K limit, 2,048 MiB reserve, batch 2,048, eight CPU threads, 64 swaps, no speculation),
  averaged **45.91 → 46.29 tok/s (+0.81%)** in untraced ABBA + BAAB. Individual rates were 46.53/46.17/45.38/45.57
  off and 46.31/46.34/46.39/46.10 on; 87.7–88.3% routed GPU hits. The candidate won three of four neighboring
  pairs. A separate test switched kernels on mirrored token positions at 87.8% hits: 46.27 → 46.57 tok/s
  (+0.65%, both pairs positive, 1,006 positions, block-bootstrap interval −0.09% to +1.51%). This supports a
  small short-context gain; the interval includes zero and neither test supports a fixed percentage across workloads.
  A separate two-start-per-build comparison against the previously installed `cedd3bf3` binary was essentially
  flat (45.280 → 45.275 tok/s at 87.8–88.0% hits), so a speedup is not visible in every short run.
- The three-arm test on RX 6700 XT, Ryzen 7 5700X, Inkling-2-Mini UD-IQ4_XS, Q8_0 KV, 2K + 256 forced,
  swaps on, averaged 43.95 tok/s off, 45.23 with scale select, and 44.35 with select and VOP3 (four starts
  each). The GPU cache ranged from 1,052 to 1,154 loaded experts across starts. VOP3 did not add a gain over
  select in this test and was removed. The starts measure production settings; the placement varies, so
  the exact size of the gain needs the matched checks.
- Against the same baseline, the select-only RDNA2 code keeps the signed dot products, FP32 contraction, lane
  accumulation, and shuffle tree. Every original NVFP4 slot instruction body was byte-identical in the
  three-arm build. For the new kernel the compiler emitted 341 vs. 401 static instructions, 4 vs. 15
  EXEC-conditional branches and 62 vs. 37 VGPRs, with zero scratch in both.
- Real 2K gate/up FP32, SiTU Q8_1 and down FP32 boundary dumps were byte-identical in all three arms. For each
  of 2K prose, 4,113-token code and 16K code, all 96 greedy IDs and ordered `(position, physical layer,
  expert, backend, router weight bits)` records matched. Inkling's visit order repeats physical layers but
  is fixed and the records preserve that order. These runs freeze a matched cache; timing them is not a
  substitute for production timing with swaps on.
- At 64K restored from the same session, both arms processed one new prompt token, then emitted the same 96
  greedy IDs with the same cache and all 4,608 visit records matching. A repeat with four starts per arm at
  54.6% GPU hits averaged 25.71 → 26.03 tok/s; pairs split two wins each. No long-context throughput gain or
  stable regression was established. All four Release 32K prefill runs also
  emitted the same 16 IDs. Prompt times were 34.23–34.34 s off and 34.45–34.79 s on (two starts per arm,
  unpinned cache). Bulk prefill does not dispatch to this kernel; the candidate can handle its small tails.
- On a second 2K fixture with a frozen 45.1% GPU hit rate, mirrored token-by-token timings at 1,006 matched
  forced positions were 26.08 → 26.14 tok/s (+0.22%, block-bootstrap interval −1.04% to +2.03%). This does not
  establish a throughput gain for low-hit workloads. In the selected-arm 16K trace, matching 1,920 tail launches
  to their preceding GPU work gave a median exposed host join of 4.74 ms/token. The per-visit median was zero
  with one CPU expert, 107 µs with two and 232 µs with three. These are traced host-branch waits, not CPU dot
  product times or untraced throughput.
- At 64K restored with a 3,700 MiB capped cache (not the stock hit rate), the analogous median exposed host
  join was 16.73 ms/token across 40 tokens, with all 1,920 visits correlated. Gate/up and down plus sum took
  14.66 and 7.53 ms/token respectively in the host branch; these phase medians do not add to the exposed join.
- The 2K exact Release trace at similar stock hit rates (84.8% and 84.6%) puts NVFP4 slots at 5.009 → 4.819
  ms/token and median total kernel-busy time at 15.554 → 15.368 ms/token. Tracing stretches the token intervals;
  the gain comes from the slot kernels but the traced throughput is not the end-to-end performance gate.
- CPU candidates on Ryzen 7 5700X, eight workers, real NVFP4 GGUF weights: converting activation scales once,
  packing the same activation codes in aligned rows, and quantizing each SiTU Q8_0 block in its producer task
  all kept the intermediate outputs and sums byte-identical. For one rotating expert all were 107.3–107.9 µs
  vs. 107.8 µs baseline; two rotating experts were 217.1–221.4 vs. 224.4 µs with overlapping quartiles. Packing
  four was slower (435.7 vs. 431.9 µs); task fusion was slower on eight hot experts (559.0 vs. 531.7 µs).
  None clears the current bar for integration without a larger measured contribution on the host join.
  A subsequent two-dot gate/up loop that reused the activation load and optionally prefetched weights stayed
  exact but did not change that conclusion: two rotating experts took 220.9 µs baseline, 221.3 µs paired,
  or 217.5 µs paired with prefetch; at eight rotating experts all were within the baseline quartiles.
- Packing adjacent K and V scale halves (raw bits) into one load kept attention FP32 partials, FP32 outputs and
  Q8_1 bytes exact, but cold kernel medians were 449.20 → 450.08 µs at 64K and 1244.25 → 1240.49 µs at 190K
  (synthetic Q8_0, 51 event samples per arm, 256 MiB cache flush). No KV layout change for this result.
- In the real 16K code and 64K prose fixtures, lossless per-32 5/6/7-bit outlier packing (raw fallback, 16-block
  tiles, 16-byte alignment and 4-byte offsets) **increased** the full KV bytes by 0.70% even with adaptive widths.
  An average seven-bit block still had 5–7 exceptional codes to store. This is a sizing test of that format,
  not a general upper bound on lossless compression.
- Specializing the existing select path for the two compile-time input widths kept the synthetic FP32 outputs
  exact. Seven rotating experts took 72.58 → 71.20 µs for gate/up and 48.34 → 46.66 µs for down. Unrolling by
  five was slower than select. The real gate/up → Q8_1 → down chain and IDs stayed exact with the fixed-width
  loop, but stock-config mirrored decode was 46.76 → 46.72 tok/s (−0.10%, block-bootstrap interval −0.76%
  to +0.47%, 1,006 forced positions). Do not add the fixed-width loop.
- Hoisting the SWA ring index calculation to the host kept all three output buffers exact before, across, and
  after wrap. It saved 0.36–0.56 µs hot and 0.40–1.04 µs read-flushed per visit. At 40 visits the measured
  savings amount to 0.014–0.042 ms/token before end-to-end overhead, below the observed timing spread.
- Non-temporal K/V code loads (`slc` in the emitted ISA) to preserve a 48 MiB resident payload kept outputs exact but did not make
  re-reading the resident bytes faster. After a read flush, the combined attention + resident-read median was
  590.44 → 589.92 µs at 64K and 1382.77 → 1384.21 µs at 190K; the input, work and returned bytes were the same.

## Completion

- Built `build-poly-hip/strata-poly` with `cmake --build build-poly-hip --target strata-poly poly_mmvq_segments_test -j 4`
  (HIP Release, gfx1031); `poly_mmvq_segments_test` passed on the RX 6700 XT. SHA256:
  `495923d467a4d88c47b993c84c97e1ec15fb58042cbcdafccca0a7860b096140`.
- Pushed `e72e40d7` to `origin/main` and deployed the matching binary locally as `strata-poly-e72e40d7`.
  `strata.service` is active on `127.0.0.1:8080` with `STRATA_HIP_NVFP4_SLOT_SELECT=1`; the real engine process
  was checked. `/health` and the authenticated chat request both returned HTTP 200. The previous config was
  backed up with mode 0600. The guard recorded no stop reason across 95 runs; the minimum free VRAM was 1,877 MiB.
- The measured short-context gains did not resolve CPU multi-miss exposure or long Q8_0 attention bandwidth.
  None of the tested follow-ups had a repeatable end-to-end win or enough isolated headroom to justify another
  implementation. The 2 GiB reserve, expert placement policy, attention math and session format are unchanged.
