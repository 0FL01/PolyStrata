# Goal: faster dense projections and LM head in Inkling decode

Status: complete
Source: user requests of 2026-11-21: take "faster kernels for the dense Q8 projections and the LM head" into work,
profile (production may be stopped), plan, then "copy the plan into a goal and implement iteratively", review with
general-purpose sub-agents, commit, push (and deploy, as the previous goals).
Last updated: 2026-11-21

## Objective

The dense attention projections (q, gate, rel, k, v) and the LM head of Inkling-2-Mini decode take measurably less
GPU time on this RX 6700 XT, decode is faster than `5903fe61` with the production config, and the result runs in
`strata.service`.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest
unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text.
Finish when every required outcome is resolved and affected constraints remain satisfied. Stop on a fresh guard fault
or a proven blocker; record the evidence and smallest unlock.

## Frozen Contract

### Required Outcomes

- R1: Projections of a visit in fewer launches.
  - Source: plan stage 1 (one launch per weight type for q, gate, rel, k, v of a token).
  - Acceptance: traced decode shows at least 0.7 ms/token less in those launches (else stop and report); rows bit for
    bit those of `native_mmvq` (unit test and the parity fixture with the switch's other parts substituted).
  - Primary evidence: rocprofv3 trace of the forced decode fixture; segment test.
  - Status: verified
  - Evidence: `poly_mmvq_segments_test`: Q8_0, Q5_K, Q6_K segments 0 of 8,576 outputs differ; a step's five Q8_0
    projections 90.7–92.9 µs as five launches, 69.6–70.6 as one. Engine trace (all-Q8_0 visit): q 24.5 + gate 24.5
    + k 10.4 + v 10.4 + rel 5.4 µs and four gaps → one 64.0 µs launch; launches 1,755 → 1,578 a token. Parity with
    the switch's other parts substituted: identical to `90b7ec72`. End to end ≈1.2 ms/token (below).
- R2: LM head faster.
  - Source: plan stage 2 (Q6_K, a warp a row).
  - Acceptance: head launch time below the current kernel's on the same trace; logits within tolerance; top-1 IDs equal.
  - Primary evidence: trace; unit test; parity fixture.
  - Status: verified
  - Evidence: engine trace 1,277 → 1,017 µs a token (median of 96); unit test within 1.3e-7 of the largest logit
    (random weights: 1,472–1,514 → 1,259–1,349 µs); parity fixture 8/8 top-1, largest change 0.040 (GPU experts).
- R3: Decode faster end to end, coherent, deployed.
  - Source: user ("коммит пуш, ревью через саб агентов general", "Жду результат"); previous goals' qualification.
  - Acceptance: paired A/B (A = `5903fe61` + production config, B = candidate) on the forced decode and 16K fixtures,
    B above A beyond variation; default path byte-identical; decode-on top-1 equal; SWA ring-wrap rewind; review by
    a general-purpose sub-agent with accepted findings fixed; commits on `origin/main`; service runs it with a rollback
    copy; live agent round trip.
  - Primary evidence: A/B logs, parity, `test-agent.py`, `git log origin/main`.
  - Status: verified
  - Evidence: A/B and checks in the history; `ec320c9d` on `origin/main`; the service runs `strata-poly-ec320c9d`
    (rollback `before-ec320c9d.json.bak` + `strata-poly-5903fe61`); live agent round trip passed (tool call ID,
    reasoning/text, cancel then `MARBLE`), 401 without the key, guard minimum 1,987 MiB free; its requests decoded at
    39.0–43.6 tok/s (35.7–41.2 with `5903fe61`).

### Constraints

- C1: RDNA2/HIP only; same GGUF, Q8_0 KV, 200K context; no reduced precision.
- C2: Opt-in under `STRATA_INKLING_DECODE=1`; default path byte-identical; other models unchanged.
- C3: GPU runs through `rdna2-rnd/scripts/run_guarded.py`; one engine at a time; benchmarks on profile copies.
- C4: Server on 127.0.0.1 with its key; artifacts, logs, profiles, binaries out of Git.

### Non-goals

- Multi-token decoding; the o-projection (its input is the attention's output); other models' kernels.

## Change Envelope

- `src/kernels/cuda/native_mmvq.cu`, `include/strata/kernels/native_mmvq.hpp` (a segments entry point, a head kernel),
  `src/poly/inkling2.*` (callers), `src/poly/engine.*` only if the logits launch needs a hook, a unit test and its
  CMake entry, docs/goals, production config.

## Plan (from RECON, RX 6700 XT, `5903fe61`, production flags)

Per decode token (traced): Q8_0 projections 4.8 ms (1.29 GB), Q6_K/Q5_K layer projections 0.3 ms, LM head (Q6_K,
200,058 x 2304, 378 MB) 1.28 ms. Streaming reads: 361–379 GB/s at 60 MB–2 GiB, but ~6 µs fixed a launch (0.94 MB in
10.6 µs, 7.5 MB in 29.7 µs). The Q8_0 kernel is at the streaming speed of its sizes (7.5 MB in 30.8 µs); an
aligned-load LDS prototype was no faster (313–315 against 317 GB/s at 60 MB). A visit's five projections: five
launches 100.7 µs, two 84.1, one 72.1 (warm clocks). LM head: 1,295–1,315 µs; a warp a row 1,108–1,150. Decode
clocks: memory 1000 MHz, fabric 1941 MHz (both top), GPU 2400 MHz, busy 90–98%. Floor: ~1.75 GB/token of dense
weights at ~360 GB/s = ~4.9 ms against 6.4.

1. One launch per type for a visit's q, gate, rel, k, v (segments with their own input and output), each row the
   same code as `native_mmvq`: −1.0…1.3 ms/token expected.
2. Q6_K head a warp a row: −0.15…0.19 ms.
3. Checks, A/B, review, commit, push, deploy.

## Current State

- Resolved: R1–R3.

## Material Decisions

- 2026-11-21: plan approved by the user.

## Checkpoint History

- 2026-11-21: RECON (numbers above). The upstream packed layouts were measured too: packed Q8_0 26.3 against 30.5 µs a
  3072 x 2304 call (bit for bit) but a second copy of q/gate/o (+540 MB of VRAM, ~120 experts); the packed Q6_K head
  no faster (1,259 µs). Not used.
- 2026-11-21: stages 1–2 as R1/R2 evidence. Paired forced decode (94+96), A = `5903fe61`, B = candidate, same
  config: A 38.04 / 38.85 / 39.08 / 38.12 / 38.41 / 38.93 / 37.95 / 37.84 (median 38.27), B 40.38 / 40.37 / 40.55 /
  40.16 / 40.04 / 39.93 / 37.10 / 34.77 (median 40.10, +4.8%; the 34.77 run had no matching trace change and was
  not repeated). 16K context, 128 tokens: A 31.12 / 31.68 / 30.82 / 32.04 / 30.76 (median 31.12), B 30.20 / 30.20 /
  32.96 / 32.63 / 32.58 (median 32.58, +4.7%); traced 16K: GPU busy 21.26 → 20.08 ms/token. Default path identical
  to `90b7ec72`; decode on: 8/8 top-1 (0.040 GPU experts, 0.057 CPU experts); SWA ring-wrap rewind equal; segment
  and head test PASS. Review (general-purpose): accepted — eligibility now asks each matrix's packed registration,
  as native_mmvq does, not the environment; the test fails instead of passing when the head does not fit. The warp
  head also taking a prompt's last token's logits (one row) is as designed under the switch, now documented.

## Completion

- Resolved outcomes: R1–R3 verified.
- Commands and artifacts: `poly_mmvq_segments_test`, `poly_inkling_attend_test`; scratch benchmarks in
  `~/.cache/claude-inkling-dense`, `~/.cache/claude-inkling-dec`, `~/.cache/claude-inkling-pp` (outside Git).
- Constraint and diff-scope check: opt-in under `STRATA_INKLING_DECODE=1`; default path identical on the parity
  fixture; other models' kernels unchanged in results (bodies moved into device functions); no new VRAM copies.
- Final status: complete. What remains of the dense floor (~4.9 ms of weights a token at ~360 GB/s against ~5.5 now)
  is not reachable by launches; more decode speed needs fewer bytes a token.
