# Goal: Native DSpark disk sessions for Inkling on RDNA2

Status: unmet — disk persistence and budget shipped; parity-preserving production speculation not qualified
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
  A/B on the actual Release, followed by live smoke checks for what ships. Unmet: the scratch verifier did not qualify
  for rollout with the live expert cache.

## Scope

HIP RDNA2 `strata-poly`, local Bartowski Q4_0 target, current DSpark GGUF. Preserve target/draft math and quantization,
existing slot API, guards, and ordinary decode. No transparent file migration, other backends, split/parallel sessions
or persistence of an in-flight request/RNG. Expected paths: `src/poly/disk_sessions.*`, `main.cpp`, `guesses.hpp`,
the existing Inkling/engine/HIP verification paths if profiling proves a cause, and their user docs.

## Evidence

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
- Release HTTP tool transcript (`r2-http-prefix`): after restart, the 820,711,244-byte file for 36,929 tokens
  restored in 1.079 s and recaptured byte-identically. Resending the actual assistant reply, including reasoning,
  preserved all 36,929 cached tokens and read 17 new ones in 1.145 s. The initial 36,865-token prefill took 39.590 s.
  Leaving out the saved assistant reply in a separate control made the client branch inside the saved tokens;
  the disk layout has no earlier checkpoints and read the prompt again. Saving just the stable 36,865-token prefix
  allowed the branch to reuse all of it and read 14 new tokens. These are cache reuse observations, not a decoder A/B.

## Verifier measurements

The scratch verifier uses ordinary Q6_K projection reduction and ordinary per-row attention chunking. From the same
snapshot and with fixed expert placement, all FP32 logits and recaptured snapshots matched serial after keeping
1, 3 or 4 of four rows, on 2K, 16K and 64K prefixes. Changing only the future guesses did not change row zero.

Untraced ABBA from common snapshots, max 0 versus max 3, 192 greedy tokens, RX 6700 XT / Ryzen 7 5700X,
Bartowski Q4_0, Q8_0 KV, 2 GiB reserve, 8 workers, no lookup. Rates are medians in tok/s:

| Prefix | Fixed placement | Live swaps | First differing greedy ID with live swaps |
|---|---:|---:|---|
| 2K | 32.166 → 37.739 | 43.577 → 46.729 | 47 in both max-3 runs; 83 between max-0 controls |
| 16K | 30.474 → 35.898 | 37.529 → 38.069 | 93 in one max-3 run; 146 between max-0 controls |
| 64K | 25.250 → 25.515 | 34.276 → 32.242 | None; both live pairs slower (−7.79%, −4.08%) |

All fixed-placement IDs matched. Freezing placement is slower than live ordinary decode and is not an equivalent
replacement for it. The live one-token path can move experts before each token; the window shares one placement for
all rows and counts rejected rows in the cache's use history. CPU and GPU expert paths need not yield identical
values. The A/A differences also show why restoring a snapshot alone does not control expert placement.

On three 320-token live serial traces, a conservative pre-verification bound for unchanged expert placement admitted
**no** two- or wider-row windows, even where the completed trace contained stretches with no pending uploads. More
aggressive admission would need a way to reproduce the serial cache transitions and apply only committed route
votes, including across KV growth. This is not a proof that every such design fails.

Other measurements did not justify a patch: separate first/next proposal cost estimates won only on the slower
fixed-placement 64K path (24.806 → 25.415 tok/s); splitting the initial Q6_K head `1+2` took 2.098 ms versus
1.104 ms grouped on real weights. Smaller DSpark attention tiles raised the 64K leaf from 13.041 ms to
15.386–28.983 ms. The exact inner-loop no-unroll variant saved 0.235 ms at 64K in isolation but neither fixes
placement nor offsets the live 64K regression on these fixtures.

Raw evidence and temporary harnesses are in `inkling-dspark-disk.path` (local scratch): `keep-*`, `test-fixed-*`,
`test-swaps-*`, `split-fixed-*`, `swap-proof-*`, `q6-base-head-v2` and `attention-*`.

## Rollout

`410545f6` and `9302cfa4` are in `origin/main`. The latter is built as `build-poly-hip/strata-poly` (Release,
`gfx1031`), SHA-256 `36bd806d6d483b4b13f3649e2497ba04b69236ccee2e6d4209754c7a3352146c`. The local text server
runs these bytes with the existing Q4_0 decode flags, disk sessions on and no drafter. Health, models, chat and
plain slot save/restore passed. The prior plain session format and the vision server are unchanged. The next
performance cycle needs a verifier whose cache transitions preserve serial decisions (or a separately approved
change to the placement policy), then fresh prefix/rollback parity and English agent A/B before activation.
