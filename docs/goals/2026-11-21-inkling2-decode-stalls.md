# Goal: fewer Inkling decode stalls and launches on RDNA2

Status: active
Source: user requests of 2026-11-21: "Нужно подумать как поднять скорость decode, жду план после RECON ... гоняй тесты,
профилируй", then on the plan: "компромисс принимаем, не критично", "приступай к реализации", "делай копию плана в
цель, ревью правок своих через саб агентов делай", "и коммит пуш делать тоже", "в прод пока не деплоить, я тут играю
и мне vram нужен".
Last updated: 2026-11-21

## Objective

Inkling-2-Mini decode (`Inkling-2-Mini-UD-IQ4_XS.gguf`, 200,000 context, Q8_0 KV) on this RX 6700 XT / Ryzen 7 5700X
is faster than `32b3fc6f` with the production config, at short context and at 16K, with coherent output, reviewed,
committed and pushed. Not deployed in this goal.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest
unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text.
Finish when every required outcome is resolved and affected constraints remain satisfied. Stop on a fresh guard fault
or a proven blocker; record the evidence and smallest unlock.

## Frozen Contract

### Required Outcomes

- R1: Expert cache weighted by what a miss costs.
  - Source: plan stage 1; user accepted a small prefill loss ("компромисс принимаем").
  - Acceptance: under `STRATA_INKLING_DECODE=1` the uses of the once-visited layers' experts (0-5, 18-23) count
    twice for the cache fill and the swaps; paired forced decode above `32b3fc6f` beyond variation; 2K/8K prefill
    measured and recorded (a loss of a few percent is accepted).
  - Primary evidence: forced 94+96 fixture and 2K/8K prefill, interleaved runs.
  - Status: verified
  - Evidence: with R2 and R4's first part, in the stage-1 pair below (prototype alone: 39.07 -> 40.15 tok/s);
    prefill measured there. Review: weighted tie-breaks (follow(), shrink_cache()) accepted and fixed.
- R2: A 3-stream visit's streams in one set of launches.
  - Source: plan stage 2 (upper bound measured: 39.3 -> 42.8 tok/s).
  - Acceptance: for decode rows (`fused(n)`), the q/k/v streams' mix, control (control_in, rank product,
    control_out) and RMS norm run as one launch each for the three streams, and their Q8_1 quantization as one;
    outputs bit for bit those of the per-stream launches (unit test and parity fixture with the switch on); traced
    launches a token fewer; paired forced decode faster than R1's build.
  - Primary evidence: a unit test, the parity fixture, rocprofv3 trace, A/B runs.
  - Status: verified
  - Evidence: `poly_segments_f32_test` all passed (bitwise); parity fixture with CPU experts identical to `32b3fc6f`
    with the switch on and off; stage-1 pair below; traced launches 1,578 -> 1,166 a token (with stage 2).
- R3: Decode attention closer to the card's bandwidth.
  - Source: plan stage 3 (16K full-attention visit 355-658 us for ~35.6 MB of KV, ~95 GB/s).
  - Acceptance: with cold KV, a full-attention decode visit at 16K takes clearly less than now; results within the
    existing kernel test's tolerance; the short-context visit not slower; 16K decode end to end faster.
  - Primary evidence: `poly_inkling_attend_test` (extended with a cold-cache timing), trace, 16K A/B.
  - Status: verified
  - Evidence: `decode_q8_kernel` (stage 2 below): cold 16K 1.061 -> 0.438 ms, 190K 6.73 -> 2.20 ms, 200 keys
    0.037 -> 0.025 ms; test PASS (largest relative difference 2.3e-5, was 3.0e-5); 16K decode +13% over stage 1.
- R4: Smaller fusions on the decode path.
  - Source: plan stage 4.
  - Acceptance: any of conv2 + head norms + K/V store, decode attention + combine, expert activation + Q8_1 that is
    kept is bit for bit the separate launches' result (or within the attention test's tolerance for the attention)
    and measures faster in paired runs; one that does not pay is dropped and recorded.
  - Primary evidence: unit tests, parity fixture, A/B runs.
  - Status: open
- R5: The CPU experts' fixed costs.
  - Source: plan stage 5 (a cold CPU expert 155 us against a 107 us read floor for 8 threads).
  - Acceptance: per-expert CPU time measured lower with the same sums, or the stage stopped with the measurement.
  - Primary evidence: per-layer CPU expert timing, A/B runs.
  - Status: open
- R6: Reviewed, committed, pushed; not deployed.
  - Source: user ("ревью правок своих через саб агентов", "коммит пуш делать тоже", "в прод пока не деплоить").
  - Acceptance: each code stage reviewed by a general-purpose sub-agent with accepted findings fixed; default path
    (switch off) identical on the parity fixture; task commits on `origin/main`; the production service and config
    untouched by this goal.
  - Primary evidence: review results here, parity, `git log origin/main`.
  - Status: open

### Constraints

- C1: RDNA2/HIP only; same GGUF, Q8_0 KV, 200,000 context; no expert skipping, no reduced precision for speed.
- C2: New behavior under `STRATA_INKLING_DECODE=1`; with it off the path is byte-identical; other models unchanged.
- C3: GPU runs through `rdna2-rnd/scripts/run_guarded.py`; one engine at a time; benchmarks on copies of the frozen
  expert profile. No GPU runs while the user needs the card (a game holding VRAM).
- C4: Server stays on 127.0.0.1 with its key; artifacts, logs, profiles, binaries out of Git.
- C5: Docs keep measured numbers with what they were measured on.

### Non-goals

- Production deployment; requantizing dense weights; DSpark changes; other models.

## Change Envelope

- `src/poly/inkling2.*`, `src/poly/inkling_attend.*`, `src/poly/engine.*` (cache weight hook), `src/poly/kernels.cu`
  and `include/strata/poly/kernels.hpp` (multi-segment F32 product and RMS norm), `src/poly/cpu_experts.*` (R5),
  unit tests and CMake entries, docs (`docs/goals/`, DETAILS if a switch's meaning changes).
- Forbidden: new dependencies or services; request protocol changes; default-path changes.

## Plan (from RECON)

RECON, RX 6700 XT / Ryzen 7 5700X, `32b3fc6f` with the production flags, forced 94+96 fixture: 39.1 tok/s
(25.6 ms/token); 16K: 30.9 tok/s. Traced: GPU kernels ~18.5 ms/token (NVFP4 experts 5.6, Q8_0 projections 4.3 and
head 1.1 near bandwidth, small kernels ~5.5), 1,578 launches a token. Untraced a chain of tiny dependent kernels
costs 3.2 us a launch (hipGraph 2.6); rocprof adds ~3 us a launch, so its 4 us gaps are ~2 us without it. CPU-expert
waits 3.1 ms/token, 80% of them in the 12 once-visited layers (200-340 us a visit against 5-70 for core visits):
the cache counts uses per visit, so the core layers (three visits a token) hold ~1,200 of ~1,330 slots. A CPU
expert costs 110-250 us; reading one with 8 threads 107 us (mmap = anonymous = THP); 6/8/12/16 threads equal.
ROC_ACTIVE_WAIT_TIMEOUT and HSA_ENABLE_INTERRUPT=0 did not change the router round trip. 16K: a full-attention
decode visit 355-658 us for ~35.6 MB of KV (~95 GB/s), its combine 50-59 us.

Prototypes (outside Git, `~/.cache/claude-dec3/recon-prototypes.patch`): edge layers' uses x2 -> 39.07 -> 40.15
tok/s, 16K 30.87 -> 32.65 (16K prefill 679.7 -> 662.2, one run); stream 0 alone for all three streams (timing only)
-> 42.8, with x2 44.0, 16K +6.5%.

1. Cache weight for the once-visited layers (R1): +2.8% measured.
2. Batch the three streams' launches (R2): ~+7% expected (upper bound +8.9%).
3. Decode attention: coalesced KV reads at long context, more blocks at short spans (R3): ~-2.5 ms at 16K,
   ~-0.5 ms short.
4. Small fusions (R4): ~-0.6 to -0.9 ms.
5. CPU expert fixed costs (R5): ~-0.3 to -0.7 ms.

## Current State

- Resolved: R1, R2, R3; R4 in part (conv2 + head norms + K/V store fused).
- Next: R4's rest (attention + combine, expert activation + Q8_1), R5.

## Material Decisions

- 2026-11-21: plan approved; prefill trade-off of R1 accepted; no deployment in this goal.

## Checkpoint History

- 2026-11-21: RECON (numbers above); scratch `~/.cache/claude-dec3/`.
- 2026-11-21: stage 1 (R1, R2, R4 part, R3 step 1). Under `STRATA_INKLING_DECODE=1`: the once-visited layers'
  uses weigh 2 for the cache; `Inkling2::heads()` runs a 3-stream visit's mix, control and norms as one launch each
  for the three streams (`gemv_f32_waves_segments()`, `rms_norm_segments()`, `mix3_kernel`, `control_in3_kernel`,
  `control_out3_kernel`) and quantizes the three inputs at once; `heads_kv_kernel` does a decode row's K/V
  convolutions, head norms and Q8_0 store in one launch; `decode_kernel` loads 8 keys' values ahead and the combine
  is unrolled (same sums). Checks: segments test bitwise; parity fixture (1,064 prompt tokens, 8 forced steps, top-5,
  CPU experts) identical to `32b3fc6f` with the switch off and on; attention test differences unchanged.
  Paired, interleaved (RX 6700 XT, A = `32b3fc6f`, B = candidate, production flags):
  forced 94+96 decode A 38.86 / 38.95 / 38.88, B 44.37 / 44.24 / 43.97 tok/s (+13.7%);
  decode after 16,384 tokens A 31.99 / 30.77, B 36.21 / 36.00 (+15%);
  prefill 16K A 689.3 / 697.0, B 671.1 / 662.8 (-3.8%); 2K A 694.1 / 631.1, B 671.9 / 626.2; 8K A 720.4 / 647.9,
  B 692.2 / 659.3 (the second round slower for both).
  Attention test with cold caches (new): decode at 16K 0.990 -> 0.862 ms, 190K 6.67 -> 6.50 ms, at 200 keys
  0.043 -> 0.061 ms (one run; hot 0.018 -> 0.016): the value loads were not the limit.
  Reviews (general-purpose): stage R1/R2 - weighted tie-breaks accepted, no R2 defect; stage R4/R3 - an all-NaN block's
  scale (seed the maximum at 0 as store2_q8_kernel does) and the test's flush buffer accepted and fixed; an `llround`
  overflow at profile counts of 2^62 and an unchecked test stream creation rejected (not reachable in use).
- 2026-11-21: stage 2 (R3). `decode_q8_kernel` for Q8_0 K/V (FP16 keeps decode_kernel): a block a chunk of keys and
  every K/V group, a wave a group, eight lanes a key with 16 dimensions a lane, so a wave's loads are four keys'
  128-byte rows whole (decode_kernel read a key a thread, 1,024 bytes apart); up to 128 chunks of at least 32 keys;
  the combine unchanged. Attention test, cold caches, old -> new kernel: at 200 keys 0.037 -> 0.025 ms, window
  0.055 -> 0.041, 16K 1.061 -> 0.438, 190K 6.733 -> 2.197; hot 16K 0.245 -> 0.106. All cases PASS; added 31 rows at
  190K: 1.8e-6 of the largest, with a spike key 3.4e-5 (the flash path's spike case at that position: 4.1e-5).
  Parity fixture (CPU experts): switch off identical to `32b3fc6f`; on: 8/8 top-1 equal, largest top-1 probability
  change 0.018. Paired, interleaved (A = `32b3fc6f`, B = stage 1 `7d9de842`, C = stage 2; the machine slower than in
  stage 1's runs): forced A 35.82 / 36.87, B 41.66 / 41.51, C 41.74 / 41.86 tok/s; after 16,384 tokens A 29.94 /
  29.75, C 36.25 / 35.64, and in a second series B 33.47 / 34.99, C 39.32 / 38.32 (+13% over stage 1). Trace (forced):
  1,166 launches a token (1,578 in RECON), attention 12.4 us a visit. Review (general-purpose): rounding over very
  long chunks (31 rows at 200K, constructed inputs: 1.46e-4) rejected on the measured random and spike cases above;
  an overflow from +-2e38 relative-table entries rejected (the model's table is small).
