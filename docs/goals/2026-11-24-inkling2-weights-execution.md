# Goal: Measure and deploy an Inkling-2-Mini weight variant

Status: active

Source: local `docs/goals/2026-11-24-inkling2-h0-h3-abliteration-agent-plan.md` (digest in the [lock](inkling2-weights/experiment-lock.json)), plus the user's request to iterate through the plan and deploy at the end.

Last updated: 2026-11-24

## Frozen outcomes

| ID | Source and acceptance | Primary evidence | Status |
| --- | --- | --- | --- |
| R1 | P0–P2: pin the experiment; reproduce the 32/1/36 precision transplant plan; verify a bounded streaming writer on small fixtures. | [Experiment lock](inkling2-weights/experiment-lock.json), [corpus](inkling2-weights/corpus-manifest.json), [inventory](inkling2-weights/transplant-plan.json), [fixture results](inkling2-weights/p2-report.md). | verified |
| R2 | P3–P5: prove no-op parity against H0 and establish a fresh RDNA2 baseline with an exact active-vocabulary NLL observer and timing noise. | No-op payload manifest and GPU raw IDs; observer alignment and H0 records. | in_progress |
| R3 | H1-B through H3-E: build the fixed variants on the copy, verify every untouched payload and evaluate the incremental changes on one release binary/configuration. | Build manifests and paired 2K/16K quality, timing and resource reports. | pending |
| R4 | A0 through F0: use authorized pinned Mix-16G data, compare D0 and the fixed A1 transfer; open A2 only on the plan's evidence. Recommend the smallest demonstrated combination. | Donor inventory, run results and final model SHA with stage verdicts. | blocked (`403 GatedRepo`) |
| R5 | User: deploy the qualified modified weights next to the original, with an identifying prefix including `OrcaRouter` for an actual Orca transfer; preserve the original and clean up only this task's disposable files. | Final path and SHA, original SHA, production API smoke and cleanup record. | pending |

## Constraints and change envelope

- Preserve the user-specified original GGUF. The no-op writer's separate copy is the working base. No in-place changes to model sources. Use `Tideglass-OrcaRouter-` for an Orca-derived final file; do not label U-only weights as Orca-derived.
- Keep the exact H0/U/A pins and stage whitelists. No keep decision based on structural compatibility alone, no alternate unpinned donor, no gated access bypass, and no speculative MoE changes. Evaluate false refusals on benign, checkable tasks.
- RDNA2 only. Use bounded guarded runs, isolated expert profiles and sessions, no driver changes or performance counters. Do not pass a GPU fault override without separate user approval or restart the host as part of this task. Keep production configuration intact until a candidate passes; final deployment has its own smoke and rollback.
- Experiment scripts in `tools/inkling_weights/`, stage evidence in `docs/goals/inkling2-weights/`, and large outputs and logs outside git. Engine changes only if required for the observer or qualifying a listed variant; no changes to the release timing path for NLL.
- Work one atomic stage at a time, recording a verdict before starting the next. The user's request to iterate authorizes proceeding to the next stage after each checkpoint, but not treating a failed gate as a pass. Stop when no independent in-scope checkpoint remains or the final deployment and cleanup pass.

## Current checkpoint

- **P3 (R2)**: [no-op writer](inkling2-weights/p3-report.md) produced a separate 15,064,977,088-byte working copy with all 704 payloads, typed metadata and the full file byte-identical to H0. GPU load and raw greedy ID parity remain blocked. The CPU/build portion of P4 does not depend on running inference and can be prepared independently.
- `Guard.boot_clean()` rejected a current-boot amdgpu page fault. The smallest unlock for GPU work is a clean boot that passes the existing guard. Until then do not start standalone inference or stop production.
- Authenticated HEAD against pinned Orca Mix-16G returned `403 GatedRepo`. This independently blocks R4's Orca branch, not H0–H3 once the GPU gate clears.

## Checkpoints recorded

- [P0 preflight and re-pin](inkling2-weights/p0-report.md): verified user-specified source, pinned Release ELF and live configuration separately from RECON. No historical timing imported.
- [P1](inkling2-weights/p1-report.md): both headers matched the RECON hashes; validated 32/1/36 replacements, byte deltas and retained tokenizer and head. [P2](inkling2-weights/p2-report.md): seven synthetic fixture checks passed. Only headers of source U have been read.
