# Goal: Measure and deploy an Inkling-2-Mini weight variant

Status: active

Source: local `docs/goals/2026-11-24-inkling2-h0-h3-abliteration-agent-plan.md` (digest in the [lock](inkling2-weights/experiment-lock.json)), plus the user's request to iterate through the plan and deploy at the end.

Last updated: 2026-11-24

## Frozen outcomes

| ID | Source and acceptance | Primary evidence | Status |
| --- | --- | --- | --- |
| R1 | P0–P2: pin the experiment; reproduce the 32/1/36 precision transplant plan; verify a bounded streaming writer on small fixtures. | [Experiment lock](inkling2-weights/experiment-lock.json), [corpus](inkling2-weights/corpus-manifest.json), [inventory](inkling2-weights/transplant-plan.json), [fixture results](inkling2-weights/p2-report.md). | verified |
| R2 | P3–P5: prove no-op parity against H0 and establish a fresh RDNA2 baseline with an exact active-vocabulary NLL observer and timing noise. | [No-op parity](inkling2-weights/p3-report.md), [observer alignment](inkling2-weights/p4-report.md), [fresh H0 result](inkling2-weights/h0-result.json) on RX 6700 XT. | verified |
| R3 | H1-B through H3-E: build the fixed variants on the copy, verify every untouched payload and screen the incremental changes on one release binary/configuration. | [H1](inkling2-weights/h1-e-report.md), [H2](inkling2-weights/h2-build-report.md) and [H3](inkling2-weights/h3-build-report.md) are verified parents, not recommendations. | CPU builds verified; GPU screening pending |
| R4 | A0 through F0: use authorized pinned Mix-16G data, compare D0 and the fixed A1 transfer; open A2 only on the plan's evidence. Recommend the smallest demonstrated combination. | [Authorized A0 inventory](inkling2-weights/orca-inventory.json) and [compatibility report](inkling2-weights/a0-report.md); D0/A1 runtime results and final model SHA pending. | A1 offline build in progress |
| R5 | User: deploy the qualified modified weights next to the original, with an identifying prefix including `OrcaRouter` for an actual Orca transfer; preserve the original and clean up only this task's disposable files. | Final path and SHA, original SHA, production API smoke and cleanup record. | blocked (no qualified candidate) |

## Constraints and change envelope

- Preserve the user-specified original GGUF. The no-op writer's separate copy is the working base. No in-place changes to model sources. Use `Tideglass-OrcaRouter-` for an Orca-derived final file; do not label U-only weights as Orca-derived.
- Keep the exact H0/U/A pins and stage whitelists. No keep decision based on structural compatibility alone, no alternate unpinned donor, no gated access bypass, and no speculative MoE changes. Evaluate false refusals on benign, checkable tasks.
- RDNA2 only. Use bounded guarded runs, isolated expert profiles and sessions, no driver changes or performance counters. Do not pass a GPU fault override without separate user approval or restart the host as part of this task. Keep production configuration intact until a candidate passes; final deployment has its own smoke and rollback.
- Experiment scripts in `tools/inkling_weights/`, stage evidence in `docs/goals/inkling2-weights/`, and large outputs and logs outside git. Engine changes only if required for the observer or qualifying a listed variant; no changes to the release timing path for NLL.
- Record each build before advancing to the next. On 2026-11-24 the user asked to simplify tests and continue: the next **offline** build depends on its parent's verified file, not on a completed GPU matrix. Do not turn a failed or skipped check into a pass.
- Reuse completed H0/H1 quality evidence. For intermediate variants run guarded finite-logit and four frozen critical tasks, not the 32-task and 20-run matrices by default. Defer matched 2K/16K timing to the finalist; use a short screen first and expand only if a promotion decision needs more evidence. A short screen does not establish the original noise estimates or waive the final check. The real GPU fault stop and GGUF integrity checks still apply.

## Current checkpoint

- **A1-B (R4, offline)**: copy the 24 fixed Mix-16G attention output tensors onto the verified H3 workcopy. Use authorized bounded ranges and mark D0 control as pending, not skipped or failed. Neither a full donor download nor a behavioral experiment is part of this offline build.
- **GPU screening (R3/R4)**: H2/H3 load/finite and critical smoke are pending; H1 timing is incomplete. The `[gfxhub] page fault` in original H0 blocks new GPU work on this boot. D0 is structurally eligible for a guarded load attempt; its full donor file has not been obtained or hashed.

## Checkpoints recorded

- [P0 preflight and re-pin](inkling2-weights/p0-report.md): verified user-specified source, pinned Release ELF and live configuration separately from RECON. No historical timing imported.
- [P1](inkling2-weights/p1-report.md): both headers matched the RECON hashes; validated 32/1/36 replacements, byte deltas and retained tokenizer and head. [P2](inkling2-weights/p2-report.md): seven synthetic fixture checks passed. Only headers of source U have been read.
- Closure of the available offline work: 13 CPU checks passed and the verified no-op is under `~/.cache/opencode-tmp/opencode/inkling-weights/p3/`. Removed the superseded 15 GB no-op duplicate, corpus rebuild scratch, writer staging files and temporary P4 binaries. Kept the working copy, isolated seed, corpus, manifests, observer and logs for resumption. The running `strata.service` stayed active with its original process and byte-identical configuration. No H1–H3 or Orca weights were built or deployed.
- [A0 RECON](inkling2-weights/a0-report.md): resolved the earlier `403 GatedRepo` through the active Hugging Face login. Both decoders agreed on the 724 tensor descriptors, and the pinned build's CPU expert type check passed for every actual D0 pair. No inference, service stop, or guard override.
- Post-reboot: P3 matched all 32 greedy IDs on original versus no-op; P4 matched CLI top-probability NLL within `1.237e-6` nat and repeated byte-identically. Production was inactive after reboot and its configuration hash was unchanged.
- [P5](inkling2-weights/p5-report.md): 2,048 finite NLL rows, 32/32 tasks, four critical tasks frozen; 16 measured runs and four warm-ups matched all forced IDs and cleared the guard. Recorded the larger 2K decode noise before candidate runs.
- [H1-B](inkling2-weights/h1-build-report.md): 32 exact donor BF16 payloads, 672 unchanged and successful guarded load/finite smoke on the separate H1 file. Production was not changed.
- [H1-E](inkling2-weights/h1-e-report.md): quality difference was small and mixed, with one new noncritical failure at the token limit. Timing incomplete; no promotion or claim of a speed change. No guard bypass; the original model and inactive production configuration were left untouched.
- [H2-B](inkling2-weights/h2-build-report.md): CPU-verified full Q8_0 embedding and 703 unchanged parent payloads. Runtime smoke deferred.
- [H3-B](inkling2-weights/h3-build-report.md): CPU-verified 36 Q8_0 projections with 668 unchanged parent payloads. Runtime smoke deferred.
