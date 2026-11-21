# Goal: DSpark speculative decoding for Inkling-2-Mini

Status: unmet
Source: user requests of 2026-11-21: "провести профилирование и составить план по добавлению Dspark ускорителя для
decode (t/s)" with the files `dspark-Inkling-2-Mini-F16.gguf` and `dspark-Inkling-2-Mini-Q8_0-F16-head.gguf`,
"прототипируй, смотри метрики и далее создаём полноценную поддержку в наш форк", then "Делай копию плана в цель и
приступай к итеративно реализации", "убедись что у модели данные параметры заданы (у основной модели --temp 1.0
--top-p 1.0 --top-k 0 --min-p 0.0 - это надо зафиксировать) и --spec-draft-n-max 3 - вероятно сделаем, если от
больших окон профита нету (т.е паррето результат берём)", "коммит пуш, ревью через саб агентов general", "Жду
результат".
Last updated: 2026-11-21

## Objective

Inkling-2-Mini decode on this RX 6700 XT runs faster with the DSpark drafter (`dspark-Inkling-2-Mini-Q8_0-F16-head.gguf`)
guessing tokens that the model checks a window at a time, at the fixed production sampling (temperature 1.0, top_p
1.0, top_k 0, min_p 0.0), with the draft length chosen by measurement, and `strata.service` runs it.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest
unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text.
Finish when every required outcome is resolved and affected constraints remain satisfied. Stop on a fresh guard fault
or a proven blocker; record the evidence and smallest unlock.

## Frozen Contract

### Required Outcomes

- R1: Windows for Inkling.
  - Source: plan stage 1.
  - Acceptance: with the drafter loaded, Engine::window()/keep() work for Inkling (t <= 8): kept tokens' logits match
    one-by-one decode in top-1 (top-1 probability change within the decode switch's 0.06); keep(n < t) leaves the
    state of n tokens (short-conv histories restored from per-row snapshots, no replay); greedy text equals plain
    decode on the fixtures or the first difference is a near-tie; SWA ring wrap with windows works; without the
    drafter the engine is unchanged (window 1, no lookup).
  - Primary evidence: oracle-window runs; parity run vs plain decode; ring-wrap fixture.
  - Status: verified
  - Evidence: with CPU experts (deterministic: two plain runs identical) speculative greedy text equals plain decode to token 116 (prose) and 126 (code); both first differences are near-ties (top-2 0.519/0.447 and 0.485/0.443), from the window rows' other sums. Oracle windows run (t = 2/4/6: 49.9/72.6/80.1 tok/s); keep() never replays (all-prefix snapshots). Without --draft-model: tokens and top-5 probabilities identical to `ec320c9d` (prose, code; CPU experts), window 1.
- R2: Cheaper verification.
  - Source: plan stage 2 ("gate: T(4) at least 10% lower, else stop the stage").
  - Acceptance: a 4-token window's round time at least 10% below R1's, rows of each expert read once a window, or
    the stage stopped with the measurement that shows it does not reach that.
  - Primary evidence: oracle-window timings, trace.
  - Status: verified
  - Evidence: stage stopped by its gate: one read per expert of a window (column slots: 69.2/72.0 against 70.2/71.9 tok/s at t = 4, interleaved; sorting the slots 65.1 against 65.6) and multi-row Q8_0 for small matrices (14.9 against 5.6 ms a round) did not pay and were dropped. Kept: the Q6_K head a warp a row for 2-8 rows (bit for bit the one-row kernel's: 1.19 against 4.37 ms at 4 rows, 2.23 against 8.49 at 8; unit test) and Q8_0 for the drafter's 8-row 9216-tall matrices (132 against 185 us).
- R3: DSpark drafter on the card.
  - Source: plan stage 3; RECON choice of the Q8_0-F16-head file (same acceptance as F16, less VRAM).
  - Acceptance: `--draft-model <dspark gguf>` loads the Inkling2 DSpark drafter (checks its variant and taps);
    taps of every call are injected for kept rows only; the drafter's rows are part of the sequence state; on the
    same taps its drafts equal the numpy prototype's on >= 99% of anchors.
  - Primary evidence: comparison run against `dspark_sim.py`; conversation restore check.
  - Status: verified
  - Evidence: STRATA_DRAFT_LOG against dspark_sim.py on the same anchors: first guess equal on 127/127 (prose) and 95/95 (code); all seven equal on 85.0%/87.4% (later, less likely guesses move with the card's int8 activations in the Q8_0/Q6_K products, which numpy does not have); mean accepted length 1.80 vs 1.77 (prose), 4.67 vs 4.61 (code). The drafter's rows are in the rows and state (card_row_bytes, extra_bytes, hn_'s first row).
- R4: Fixed sampling of the main model.
  - Source: user ("--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 - это надо зафиксировать").
  - Acceptance: the production run config sets these four as the model's sampling; top_k 0 reaches strata-poly as
    "no limit" (not 64) and costs no more than a top_k 20 pick; requests that set their own still win (the server's
    contract).
  - Primary evidence: GEN line keys (server test), sampler timing, the running config.
  - Status: verified
  - Evidence: server: top_k 0 reaches strata-poly as 0 (whole_vocab), config and Settings accept it for strata-poly only (tests); engine: a full-vocabulary pick 0.28 ms against 0.14 ms for top_k 20 and 19.5 ms before (Ryzen 7 5700X); plain decode at the fixed sampling 36-38 tok/s against 21 tok/s with `ec320c9d`. The pick draws in token order now: same distribution, other tokens for a given seed (only top_k 0 requests, which the server used to send as 64).
- R5: Draft length by measurement.
  - Source: user ("--spec-draft-n-max 3 ... если от больших окон профита нету (т.е паррето результат берём)").
  - Acceptance: a maximum draft length option; production uses 3 unless a longer one measures faster on the agent
    fixtures; the measurement is recorded.
  - Primary evidence: A/B over draft lengths.
  - Status: verified
  - Evidence: --draft-max N (default 3). Temperature 1.0, two seeds, tok/s: tool 38.05 -> k1 41.37, k2 43.66, k3 44.39, k5 41.48; code 35.17 -> 43.31, 41.87, 43.97, 44.95 (39.0/50.9); json 37.51 -> 38.45, 38.52, 37.70, 39.69. k5 is no better than k3 within its spread: 3.
- R6: Faster end to end, reviewed, committed, deployed.
  - Source: plan stage 4-5; user ("коммит пуш, ревью через саб агентов general", "Жду результат").
  - Acceptance: paired A/B vs the production binary at the fixed sampling on code, tool, json, prose, ru (short)
    and 16K context; deploy gate: mean over tool/code/json >= +25%, no fixture worse than -3%; default path
    byte-identical without the drafter; review by general-purpose sub-agents with accepted findings fixed; commits on
    `origin/main`; service runs it with a rollback copy; live agent round trip.
  - Primary evidence: A/B logs, parity, `test-agent.py`, `git log origin/main`.
  - Status: unmet
  - Evidence: A/B at the fixed sampling, k3, three seeds (tok/s): tool 38.0 -> 44.8 (+18%), code 35.9 -> 44.8 (+25%), json 37.7 -> 39.7 (+5%), prose 37.4 -> 38.5 (+3%), ru 38.5 -> 35.6 (-7%). Mean of tool/code/json +16% against the gate's +25%, and ru below -3%: the drafter is not turned on in production. Review (general-purpose): accepted and fixed - q8_ too small for the drafter's block under a small --prompt-batch (widest(), and --prompt-batch >= 8), the full-vocabulary draw's edge cases (all -inf, a tiny temperature, run sums in doubles), the Settings validator, bare-engine test fixtures; the seeded top_k 0 change is documented (R4). Deployed: `32b3fc6f` on `origin/main`; the service runs `strata-poly-32b3fc6f` with the config's sampling block temperature 1.0, top_p 1.0, top_k 0, min_p 0.0 and no --draft-model (rollback `before-32b3fc6f.json.bak` + `strata-poly-ec320c9d`); live agent round trip passed (AGENT_LIVE_PASS; its requests decoded at 38.1-41.8 tok/s), 401 without the key.

### Constraints

- C1: RDNA2/HIP only; same target GGUF, Q8_0 KV, 200K context.
- C2: Opt-in: without `--draft-model` the engine path is byte-identical; other models unchanged.
- C3: GPU runs through `rdna2-rnd/scripts/run_guarded.py`; one engine at a time; benchmarks on profile copies.
- C4: Server on 127.0.0.1 with its key; artifacts, logs, profiles, binaries, prototypes out of Git.

### Non-goals

- Probabilistic drafting with rejection sampling (drafts are the drafter's argmax; a guess is kept when the sample
  equals it, which is lossless); the F16/BF16 drafter files; other models' speculation.

## Change Envelope

- `src/poly/inkling2.*` (windows, taps, drafter), `src/poly/engine.*` (hooks the drafter needs), `src/poly/main.cpp`
  (draft length option, full-vocabulary sampling), kernels in `src/poly/` or `src/kernels/cuda/` for the drafter and
  windows, `serve/server.py` (top_k 0 for strata-poly) and its tests, a unit test with its CMake entry, docs/goals,
  docs/DETAILS.md, the production config.

## Plan (RECON, RX 6700 XT, production config of `ec320c9d`)

Measured: acceptance (numpy drafter on our engine's taps, greedy, 511 anchors; tokens a round at draft length 1/3/5/7):
code 1.89/3.25/4.26/4.78, tool calls 1.84/2.92/3.62/3.93, JSON 1.68/2.41/2.57/2.61, prose EN 1.60/2.08/2.20/2.24,
prose RU 1.30/1.44/1.46/1.46; F16 = Q8_0-F16-head; full-attention draft layer capped at 1024 rows: code 4.63 -> 3.01
mean accepted. Verify (oracle windows t = 1..8): 25.1, 36.0, 41.9, 51.1, 60.8, 68.5, 82.7, 101 ms a round. Trace
t = 1/2/4: experts 5.7/10.75/20.6 ms (a read per token and expert), dense 4.3/6.5/9.5. Distinct experts / uses in a
window (prose): t=2 0.79, t=4 0.61, t=8 0.45. VRAM: 400 MB to the drafter -> decode -1.4%.

1. Windows for Inkling (R1): opt-in with the drafter; logits for t rows; per-row short-conv snapshots written by the
   conv kernels into the engine's snapshot slots (all prefixes, so keep() never replays); KV rings already hold
   kWindow + rows - 1.
2. Cheaper verification (R2): each distinct expert's rows together (GPU slots, CPU tasks), multi-row dense kernels.
3. Drafter (R3): taps (residual entering visits 5/23/41, final normed rows) of every call, staged and injected at
   keep() for kept rows; SWA rings of 1024 + 8 rows for draft layers 0/2, full layer 1 growing with the context; the
   drafter's rows in the sequence state; block rows apart from the committed rings; draft(token, 0) runs the block
   and recurrent step 1, draft(guess, i) the next step, so the Guesses/Gate policy picks the length; W' = W_head x
   W_hidden_out precomputed; markov_w1 rows from host memory.
4. Sampling (R4) and draft length (R5); end to end A/B, review, commit, push, deploy (R6).

## Current State

- Resolved: R1-R5; R6 unmet on its speed gate (see R6).
- Next: none in scope; turning the drafter on is the user's call (one flag, numbers in R6).

## Material Decisions

- 2026-11-21: plan approved by the user; Q8_0-F16-head drafter; sampling and draft length fixed by the user.

## Checkpoint History

- 2026-11-21: acceptance at temperature 1.0 (numpy drafter on sampled continuations, tokens a round at k = 3): code
  3.17, tool 2.78, json 2.50, prose 2.02, ru 1.40 - near the greedy ones. Plain decode at top_k 0 with the old sampler:
  21.5 tok/s (a 200K-token heap a pick). Drafter cost a round: first guess 4.3 ms (short context; 6.9 at 4.3K), the
  next 0.43 each; with the drafter loaded the cache holds ~130 fewer experts and plain decode is 2.5% slower
  (37.86 against 38.83 tok/s, oracle window 1). One GPU page fault (pinned host address) in a run without the drafter
  (code, seed 714); the same run did not repeat it twice with this build nor twice with `ec320c9d`, and the journal
  has such faults on 10, 11 and 14 November.

- 2026-11-21: RECON (numbers above). Prototypes outside Git: `~/.cache/claude-dspark` (oracle windows, tap dump,
  routing log, numpy drafter).

## Completion

- Resolved outcomes: R1-R5 verified; R6 unmet (speed gate), the rest of R6 done (review, commits, deploy of the build
  with the fixed sampling, drafter off).
- Commands and artifacts: `poly_mmvq_segments_test` (few-row kernels), `poly_inkling_attend_test`,
  `serve.test_server`, `serve.test_runconfig`; scratch in `~/.cache/claude-dspark` (outside Git).
- Constraint and diff-scope check: without --draft-model the engine path is the old one (tokens and top-5 identical
  to `ec320c9d`); other models: the new lin() kernels and the drafter hook only run with a guesser that sets them.
- Final status: unmet.
