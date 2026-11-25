# RDNA2 Native Donor RECON — Inkling 2 Mini

Status: **complete**, 2026-11-26. Source: request to profile `vllm-rdna/rdna_extras`, especially `bc2b207c`, and plan later production integration for the hybrid RX 6700 XT engine.

**Result:** [Prioritized production plan](native-donor-recon/integration.md). Start with opt-in, serial **LoopCD-Hidden** and its warm-state contract, not with speculative DMA. On a real three-turn task at two held-out seeds, Hidden kept both useful tools and the 118/118 grade with shorter time to completion. For **Skip-Coda Logits**, a new GPU-only weak-head implementation matched the full corrected logits and saved 46–138 µs against an already weight-sharing paired head in the leaf. Neither is a qualified default. Direct H2 tile addressing, Q4 DPP and read-only next-router warming had no repeatable untraced engine win. Leave production unchanged.

## Finish line

- R1 — Differential: map #195/196/206/207/209/210/211/213/214, including their native, Python, test and benchmark paths, to our layouts, arithmetic and execution order. Verified: [source-pinned comparison](native-donor-recon/differential.md).
- R2 — Experiments: verified on gfx1031 and the host. [Whole-expert DMA and cold recall](native-donor-recon/transfers.md), [packed CPU router, helper and registered-RAM W1 ABBA](native-donor-recon/router.md), [attention and MoE leaf → ELF, ISA and parity](native-donor-recon/kernels.md), [DSpark head, acceptance and verifier](native-donor-recon/dspark.md), [native LoopCD, graded tool trajectories and exact APC weak-head tests](native-donor-recon/contrast.md).
- R3 — Decision: verified. [Integration order and stop conditions](native-donor-recon/integration.md), [state, quality and time-to-correct qualification](native-donor-recon/qualification.md). The outcome is a plan for subsequent integration, not a deployment.

Complete these outcomes using the smallest experiments that can change the decision. Work in `~/.cache/opencode-tmp/opencode/polystrata-runs/native-donor-bc2b/`; guarded runs share one GPU lock. Use the committed tree and the donor as read-only sources, scratch prototypes and independent outputs. No change to the engine, weights, service, settings or donor. Keep measured throughput separate from trace budgets and simulation. Do not claim that output-ID equality on one trajectory establishes agent quality.

## Closure and artifacts

The original engine (`b14defe8`, last binary change `35c899cb`, ELF `aa3aab2d…`) gave 25/25 matching IDs traced/untraced. Experiments used scratch ELFs and independent controls, not a vLLM installation. Donor pin `bc2b207c2b`; local `rdna_extras` at `93c59de2b5`. No change to model, production executable, settings or service; no unrelated worktree files included. Separately requested guard change `78b2b6f7` exempts historical VCN decoder incidents, not fresh timeouts. 89 saved guard stop-reason arrays were empty; no fresh warning entries. This is **not** 89 successful commands: initial loader, fixture, parity and budget failures were retained and diagnosed in the respective reports. No inference result from an invalid arm was counted.

The host archive holds the raw private fixtures, code, commands, ISA, timings, native IDs and state checks. It is intentionally not a portable model download. After all benchmarks stopped, `polystrata-dedupe native-donor-bc2b` completed without sudo. To check selected receipts on this host, from the repository root:

```sh
manifest="$PWD/docs/goals/native-donor-recon/receipts.sha256"
(cd ~/.cache/opencode-tmp/opencode/polystrata-runs/native-donor-bc2b && sha256sum -c "$manifest")
```
