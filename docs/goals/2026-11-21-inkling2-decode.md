# Goal: faster Inkling-2-Mini decode on RDNA2

Status: active
Source: user requests of 2026-11-21: plan after RECON with profiling to raise decode substantially (an architecture
rewrite is welcome for a multiple), then "copy the plan into a goal and implement iteratively", lower the VRAM reserve
from 3 GiB to 2 GiB unless prefill drops significantly (Pareto), review with general-purpose sub-agents, commit, push
and deploy.
Last updated: 2026-11-21

## Objective

Decode of Inkling-2-Mini (`Inkling-2-Mini-UD-IQ4_XS.gguf`, 200,000 context, Q8_0 KV) on this RX 6700 XT / Ryzen 7
5700X is substantially faster than the installed `90b7ec72` configuration at short and long context, outputs stay
coherent, and the measured configuration runs in `strata.service`.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest
unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text.
Finish when every required outcome is resolved and affected constraints remain satisfied. Stop on a fresh guard fault
or a proven blocker; record the evidence and smallest unlock.

## Frozen Contract

### Required Outcomes

- R1: Decode is substantially faster than the installed configuration.
  - Source: "цель поднять скорость decode, значительно"; plan targets about 50 tok/s short context, over 40 tok/s at 16K.
  - Acceptance: paired cold A/B (A = installed `90b7ec72` + prod config, B = candidate) on the forced 94+96 fixture and
    the 16K-context decode fixture; B clearly above A beyond run-to-run variation. Targets are context, not a gate.
  - Primary evidence: `~/.cache/claude-inkling-dec` fixtures (`drun.sh`, `bench.py tg`) A1/B1/B2/A2.
  - Status: pending
  - Evidence:
- R2: Coherent output and unchanged default path.
  - Source: AGENTS.md (opt-in, default byte-identical); previous Inkling goals' qualification.
  - Acceptance: switches off = same IDs/bytes as `90b7ec72`; switches on = top-1 IDs match on the fixed 1,064-token
    prefix, probability changes within the engine's existing variation; one row vs chunk; SWA restore after ring wrap;
    new kernels pass their unit test against the existing path; live agent round trip (tool call ID, reasoning/text,
    cancel then `MARBLE`).
  - Primary evidence: bench `parity`, `poly_inkling_attend_test` (or a new decode test), `test-agent.py`.
  - Status: pending
  - Evidence:
- R3: VRAM reserve 3072 → 2048 MiB unless prefill drops significantly.
  - Source: user, 2026-11-21 ("снижай лимит с 3гб до 2гб если prefill значительно не падает ... по паррето").
  - Acceptance: 2K/8K cold prefill and decode measured at both reserves on the candidate; 2048 deployed unless prefill
    falls by a significant margin (more than a few percent beyond variation) without a matching decode gain.
  - Primary evidence: paired runs at both reserves.
  - Status: pending
  - Evidence:
- R4: Reviewed, committed, pushed and deployed.
  - Source: user ("коммит пуш, ревью через саб агентов general - делать, это важно").
  - Acceptance: each code stage reviewed by a general-purpose sub-agent, accepted findings fixed; task commits on
    `origin/main`; service runs the qualified binary/config with a rollback copy; authenticated single-model API, 401
    without key, one real request.
  - Primary evidence: review results in this file, `git log origin/main`, service log and API checks.
  - Status: pending
  - Evidence:

### Constraints

- C1: RDNA2/HIP only; one card, one sequence; same GGUF, Q8_0 KV, 200,000 context; no expert skipping or reduced
  precision for speed.
- C2: New behavior opt-in by environment switch; default path byte-identical. Other models' defaults unchanged.
- C3: GPU runs through `rdna2-rnd/scripts/run_guarded.py` (1536 MiB floor; stop on a fresh fault); one engine at a
  time; benchmarks use copies of the frozen expert profile, never the production profile.
- C4: Server stays on 127.0.0.1 with its API key. Benchmark artifacts, logs, profiles and binaries stay out of Git.
- C5: Docs keep measured numbers with what they were measured on.

### Non-goals

- Prefill speed beyond keeping it (R3 only guards it); other models; multi-card; speculative decoding unless stage 4's
  measurement gate passes.

## Change Envelope

- Target: Inkling decode path (rows below `kStageRows`): `src/poly/inkling2.*`, `src/poly/inkling_attend.*`, shared
  engine MoE/router decode code in `src/poly/engine.*` behind Inkling-only switches, the NVFP4 slot MMVQ only if stage 4
  runs, a decode attention unit test, CMake entries, docs (`docs/goals/`, results doc), production config.
- Forbidden: new dependencies or services; changes to the request protocol; default-path changes.

## Plan (from RECON)

RECON, RX 6700 XT / Ryzen 7 5700X, installed `90b7ec72` config: forced 94+96 decode 22.0/22.4 tok/s (45 ms/token);
4K 17.1, 16K 15.1–16.5 tok/s. 2,700 kernel launches/token; GPU busy ~25 ms/token: `mix_kernel` 5.1 ms (one block,
10 serial reductions), decode attention 2.9 ms short / 11.2 ms at 16K, F32 `gemv` 3.6 ms (400 launches), NVFP4 cached
experts 5.0 ms (~200 GB/s), small kernels ~3 ms, Q8_0 projections 6 ms and LM head 1.5 ms (near bandwidth). GPU idle:
~8 ms waiting for CPU experts (125–147 CPU experts/token = 560–660 MB at 43–47 GB/s DRAM), ~5 ms router round trips,
~5 ms launch gaps. A CPU expert was used at the same visit within the previous 16 tokens 75–82% of the time;
`--expert-swaps 64` raised VRAM hits 62 → 86% and prose decode 20.4 → 23.8 tok/s.

1. Decode kernels: one-pass `mix_kernel` (same reduction trees), split-key decode attention, F32 gemv one wave a row,
   fused small kernels.
2. Expert cache: swaps for Inkling, VRAM reserve (R3).
3. Async decode pipeline: GPU routing, cached experts launched from device-side ids, CPU sum through pinned memory and an
   in-stream flag, whole token enqueued ahead (or a graph). Probe the in-stream wait on gfx1031 first.
4. Gated: NVFP4 slot MMVQ tuning if experts are on the critical path; lookup speculation only if agent-workload
   acceptance exceeds ~1.3 tokens/step (reasoning measured 1.00–1.02).

## Current Checkpoint

- Closes: R1, R2, R4
- Smallest next action: final cold A/B pair (installed `90b7ec72` + prod config against the candidate + new config) on
  the forced decode fixture, 16K decode and 2K/8K prefill; live agent check; deploy with a rollback copy.
- Expected evidence: B above A beyond run-to-run variation; agent round trip passes.
- Stop or replan if: a guard fault, or B not faster.

## Current State

- Resolved: RECON, stages 1–4 (decode kernels; swaps and reserve as configuration; NVFP4 slot rows).
- Next: final A/B, deployment.

## Material Decisions

- 2026-11-21: user approved the plan; reserve 2048 MiB unless prefill drops significantly.
- 2026-11-21: stage 3 replanned on evidence. Routing on the card with cached experts launched from device ids
  (host waits only for the picks) measured 36.4/36.4 tok/s against 37.6 without it: decode is GPU-bound and the
  inter-kernel gap is GPU-side (median 4.2 µs in prefill, where the host is far ahead, as in decode). Reverted
  (patch kept outside Git). Stage 3 is now fewer launches on the one-row residual path; stage 4 the NVFP4 slot
  kernel (36/72 calls a row on 32 lanes: 56–75% lane use).

## Checkpoint History

- 2026-11-21: RECON (numbers above).
- 2026-11-21: stage 1. `mix_kernel` keeps a row in registers (always on): 21.9 → 4.8 µs a call, top-5 probabilities
  and IDs of the 1,064-token parity fixture with CPU experts identical to `90b7ec72` (explicit FMAs where the
  compiler had fused; with five inputs it stopped fusing and 1,129 of 6,912 values moved by an ulp). Under
  `STRATA_INKLING_DECODE=1` for fewer than 32 rows: split-key decode attention (`attend_decode`, test: within
  3e-5 of the plain path; 16K full layer 0.78 → 0.20 ms, 190K 11.8 → 5.2 ms, window 0.090 → 0.025 ms), F32
  products by a wave or a block a row (`gemv_f32_waves`), fused K/V convolutions, head norms, K/V stores,
  convolution + residual tails (with the CPU experts' sum) — those bit for bit (checked with the others off) — and
  the loop control in 3 launches instead of 7. A fused norm + Q8_1 quantization was not bit for bit and was dropped.
  Launches 2,703 → 1,830 a token, GPU busy 25 → 17.3 ms a token (traced). One-run screens, forced 94+96: 26.7–27.8
  tok/s against 22.0/22.4; 16K context 15.1 → 26.3 tok/s. Default path: parity fixture identical to `90b7ec72`;
  decode on: 8/8 top-1 IDs equal, largest top-1 probability change 0.057 (CPU experts). SWA ring-wrap rewind at
  3,072 with batch 2048: restored and replayed IDs equal. Review (general-purpose): accepted an all-`-inf` tile
  giving NaN and a clamped `-inf` sink gaining weight (both fixed, test passes); refuted a Q8 dot overflow (queries
  are clipped head norms) and an `-ffp-contract=off` mismatch (`inkling2.cu` is built with the default).
- 2026-11-21: stage 2 (configuration only), stage-1 binary, decode on, one run each: swaps 0 / reserve 3072 →
  forced 26.7, prose 27.7 tok/s (VRAM hits 58–60%); `--expert-swaps 64` → 35.9 / 36.7 (79–85%); plus reserve 2048 →
  37.6 / 38.6 (82–90%, 1,329 experts against 1,093). 16 and 32 swaps in flight: 36.3–37.4 / 38.6, no better than 64.
  R3: ABBA prefill at reserve 3072 / 2048 / 2048 / 3072: 2K 665.4 / 705.4 / 703.0 / 668.6 tok/s, 8K 701.5 / 730.1 /
  730.6 / 705.3 — prefill rises with the larger cache, so 2048; guard minimum free 1,997 MiB at 2048.
- 2026-11-21: stage 4 (NVFP4 slot rows, `iq_set_slot_lanes()`, set by Inkling for its decode calls only): a row to
  16 lanes. Traced expert launches 6.4 → 5.3 ms a token (8 lanes for 72-call rows and 4 for 36: 5.9; 2 / 8 / 16 / 32
  lanes swept). Forced decode 38.8–38.9 (one 36.7).
- 2026-11-21: stage 3 as replanned: a one-row kernel doing the residual's convolution, the mix, the norm and Q8_1 for
  a token (launches 1,851 → 1,415, bit for bit but for a 128-thread norm whose reference sums its four warps in a
  data-dependent order) measured 36.1–37.6: a single block waits on memory (18 µs reading the CPU experts' sum from
  host memory, 9–17 µs for a norm with Q8_1 that separate launches do in 4–5) — reverted (kept outside Git). Kept:
  the decode attention writes the output projection's Q8_1 blocks, and the fused MoE tail leaves fo_ 0 so the next
  layer skips its fill. Forced decode 39.31 / 39.50 / 39.52 tok/s; 16K context 31.4 tok/s (15.1 installed). Default
  path identical to `90b7ec72` on the parity fixture; decode on (GPU experts): 8/8 top-1, largest change 0.033.
  Attention unit test PASS; SWA ring-wrap rewind at 3,072 equal. Review: accepted the slot switch reaching prompt
  batches without MMQ and outliving the engine — now set for a decode call's duration only (RAII in run_visits).

## Completion

- Resolved outcomes:
- Commands and artifacts:
- Constraint and diff-scope check:
- Final status:
