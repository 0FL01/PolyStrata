# Goal: Native DSpark disk sessions for Inkling on RDNA2

Status: active
Source: user requests for native DSpark with disk sessions, a dynamic draft budget, and measured rollout after RECON.
Last updated: 2026-11-24

## Frozen contract

- R1: Save and restore a committed Inkling prefix with DSpark enabled, including across a process restart. Keep
  ordinary Inkling snapshots compatible and reject incompatible models/layouts before transfer.
  Evidence: a real restart, exact file recapture and continuation on the same schedule, plus a plain Inkling control.
  Status: verified in Release. Real HTTP restart and byte recapture passed. 2,095- and 65,555-token DSpark files
  recaptured exactly; each resumed from its full prefix with one new input token. 2K also restored with max 1 instead
  of max 3. A changed byte in the draft GGUF and both plain/DSpark format swaps were rejected before transfer. The
  old plain file recaptured exactly and the 32 greedy IDs matched.
- R2: Apply the configured draft maximum as a per-round budget, not a fixed window. Never exceed available context
  or generation room. Status: verified in Release. At a 2,055-token context limit with 2,048 prompt and seven output
  tokens, observed proposal counts under budgets 0/1/2/3 were none, 1/1/1/1/1, 2/2/2/2/1 and 3/3/3/2/1. All
  four budgets restored the same 2,095-token snapshot and read just the new suffix; zero proposed no guesses, and
  positive budgets used adaptive verification. Invalid integer budgets fail before loading the model.
- R3: Qualify speed changes on English tool calls, reasoning and code, without changing target results. Commit, push
  and deploy verified changes; reprofile and report the final binary and limits. Evidence: target parity and same-host
  A/B on the actual Release, followed by live smoke checks for what ships. Status: pending.

## Scope

HIP RDNA2 `strata-poly`, local Bartowski Q4_0 target, current DSpark GGUF. Preserve target/draft math and quantization,
existing slot API, guards, and ordinary decode. No transparent file migration, other backends, split/parallel sessions
or persistence of an in-flight request/RNG. Expected paths: `src/poly/disk_sessions.*`, `main.cpp`, `guesses.hpp`,
the existing Inkling/engine/HIP verification paths if profiling proves a cause, and their user docs.

## Current checkpoint

R3: isolate the remaining full-attention difference after using the single-row Q6_K reduction for verifier projections.
Then compare every row and rollback against serial with frozen expert placement; measure the cost and test the separate
placement issue before any production activation.

## Evidence and next

- RECON (local `inkling-dspark-disk.path`): scratch serializer round-tripped 2K/16K/64K; exact 48 FP32 logit rows
  and greedy IDs per fixed-schedule 2K/16K fixture. At 64K restore took 1.43 s vs 76.8 s original prefill once.
- Max 3 (2 seeds, RX 6700 XT/5700X) measured +9% tools, +10% code and +11% English reasoning decode. Not parity:
  same saved prefix diverged at greedy ID 43 on 16K when changing the target from serial to four-row verify.
  Shipping disk support does not authorize activating that verifier in production.
- Release budget checkpoint: uncorrected verifier, same restored 2K snapshot and 144 greedy tokens, budget 2/3 first
  diverged from 0 at ID 91. `eval(1)` and `window(1)` matched all 1,249 common boundaries, and changing future IDs
  did not change row zero with the same verifier width. Q6_K warp-columns projections were the first differing
  boundary (visit zero); using the ordinary projection reduction in a scratch build made row zero exact at 2K and
  in 16K two-row verify. At 16K four-row verify the first difference moved to full attention (visit five), whose
  chunk cap depends on the number of rows. Neither scratch change is a production fix yet.
- Release disk checkpoint: RX 6700 XT / 5700X, 65,555 committed tokens, 1,381,440,500 bytes: save 1.033 s,
  restore 1.498 s vs 76.675 s original prefill (filesystem cache, one run). Existing conversation-file, guesses,
  HIP attention and MMVQ checks pass. Evidence under the RECON directory in `r1-*`.

Complete the frozen outcomes with the smallest relevant experiment and diff. Reprofile after material changes;
stop when outcomes are evidenced or no remaining in-scope hypothesis has a falsifiable next action.
