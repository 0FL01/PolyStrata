# Goal: faster Inkling-2-Mini prefill on RDNA2

Status: active
Source: user requests of 2026-11-21: raise `Inkling-2-Mini` prefill speed; follow the RECON plan below iteratively; review code through general-purpose subagents (stressed as very important); commit, push and deploy to production afterwards. Production downtime is allowed.
Last updated: 2026-11-21

## Objective

Cold prefill of the deployed `Inkling-2-Mini-UD-IQ4_XS.gguf` on this RX 6700 XT (gfx1031) / Ryzen 7 5700X rises clearly beyond run-to-run variation at 2,048 and 8,192 tokens, with coherent output, and the faster build serves production.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text. Finish when every required outcome is resolved and affected constraints remain satisfied. Stop on a fresh guard fault or a proven blocker; record the evidence and smallest unlock.

## Frozen Contract

### Required Outcomes

- R1: Prefill is faster.
  - Source: "оптимизировать инференс движок, цель поднять скорость prefill".
  - Acceptance: A/B/B/A on the same fixtures (`pp` 2,048 and `pp8192`), same binary family and production settings; B beats A in every pair at both lengths.
  - Primary evidence: `bench.py` runs under `rdna2-rnd/scripts/run_guarded.py`.
  - Status: pending
  - Evidence:
- R2: Output stays correct.
  - Source: previous Inkling contracts (coherent text, tool replay); RECON plan validation list.
  - Acceptance: kernel test against the existing attention kernels within 1e-4 relative and no NaN; at a fixed 1,064-token prefix all top-1 IDs match the current engine (largest top-probability change reported); 1-, 32- and 512-row reads give the same IDs; restore at 4,096 after SWA ring wrap matches a cold read; decode fixture IDs unchanged.
  - Primary evidence: `poly_inkling_attend_test`; `bench.py parity`-style forced runs.
  - Status: verified for the stage-1 diff (see history)
  - Evidence:
- R3: Code reviewed by general-purpose subagents.
  - Source: "не забывать ревью через саб агентов делать, это очень важно".
  - Acceptance: each engine diff reviewed before commit; findings with a reachable failure scenario fixed or refuted with evidence.
  - Primary evidence: review results recorded in Checkpoint History.
  - Status: pending
  - Evidence:
- R4: Commit, push, deploy.
  - Source: "коммит пуш деплой в прод потом".
  - Acceptance: task commits on `origin/main`; `strata.service` runs the qualified binary/config with a rollback copy; authenticated model listing and a live streamed tool round trip pass.
  - Primary evidence: `git ls-remote`, `systemctl --user is-active strata`, live HTTP checks.
  - Status: pending
  - Evidence:

### Constraints

- C1: One HIP RDNA2 card, one sequence, the same GGUF, Q8_0 K/V, 200,000 context, unchanged client limits and request protocol. No expert skipping, no reduced-precision attention operands (FP32 math; only summation order may change).
- C2: Decode path and other models' defaults unchanged. The new attention is opt-in (`STRATA_INKLING_FLASH=1`) and used only for prompt batches; production enables it in its config.
- C3: One engine at a time; GPU runs through `run_guarded.py` (1,536 MiB floor); stop on a fresh GPU/desktop fault. Production stays up between experiments when no GPU run is active.

### Non-goals

- Rewriting NVFP4 MMQ, changing weight/KV formats, `--vram-reserve-mib`, other GPU architectures or backends.

## Change Envelope

- Target: Inkling prompt-batch attention, then host stalls / small kernels only where a fresh profile shows them.
- Expected paths: `src/poly/inkling2.cu`, `src/poly/inkling2.hpp`, a new `src/poly/inkling_attend_test.cu`, `CMakeLists.txt`; for stage 2 only the Inkling-relevant parts of `src/poly/engine.cpp`; this document and `docs/DETAILS.md` numbers.
- Allowed artifacts: source, tests, docs. Forbidden in Git: binaries, traces, logs, model files, credentials. Scratch lives in `~/.cache/claude-inkling-pp/`.

## Plan (from RECON)

- Stage 0 (gate): fused FP32 attention kernel + standalone test/timer at real geometry. Gate: one SWA visit over 512 rows ≤ 3 ms (now ~18.5). Fallback if missed: rocBLAS SGEMM QK/PV + fused bias/softmax kernel.
- Stage 1: integrate for n ≥ 16 behind `STRATA_INKLING_FLASH=1`; validate R2; measure R1.
- Stage 2: re-profile; fix only confirmed host stalls (CPU-expert wait before `moe_sum`, serialized expert H2D, rocBLAS lazy load on first request).
- Stage 3: small kernels (conv, mix, control/router FP32 GEMMs, axpy/copy2/fill) only if > 10% of wall time after stages 1–2.

## Baseline (RECON, 2026-11-21)

RX 6700 XT, TheRock ROCm 7.13, deployed binary `3d6760d8` (sha256 `23294c89…`), production args, `STRATA_HIP_NVFP4_MMQ=1` (note: `bench.py` does not apply the config's `env`; without it PP is 97.7 tok/s).

- `pp` 2,048: 266.4 / 267.5 tok/s. Trace: 7.80 s wall, 6.43 s GPU busy, kernel sum 6.11 s; attention 3.56 s (scores 2.13, values 1.36), MMQ ~1.75 s, idle 1.37 s (572 ms CPU-expert wait before `moe_sum`, 242 ms serialized H2D, 142 ms one-time rocBLAS load).
- `pp8192`: 209.3 / 204.4 tok/s; kernel sum 34.4 s, attention 24.8 s (72%), MMQ 7.0 s.
- CLI screens (one run each, 2,048): batch 1024 286.3 (but ~65 fewer resident experts), `--cpu-share 0` 250.9, `--stage-ahead 512` 230.5.

## Current Checkpoint

- Closes: R1 (further), then R4.
- Smallest next action: stage 3 — the largest remaining items that are cheap to change: flash kernel at long context, the sequential short convolution, the control projections; then final pair, deploy.
- Stop or replan if: a stage-2 change does not beat repeated-run variation on the 2K/8K pair.

## Material Decisions

- 2026-11-21: user chose `--prompt-batch 2048` for production (measured +37% PP, −3% TG against 512); stage 2 targets batch 2048.

## Checkpoint History

- 2026-11-21: RECON, baseline above; plan approved by the user.
- 2026-11-21: stage 0 gate passed. `poly_inkling_attend_test` (RX 6700 XT): 14 cases within 6.8e-6 of the plain path's largest output (dominant-key case 4.1e-5; fails at 4.9e-4 without the compensated denominator), no NaN. Per 512-row call: window 23.6 → 1.85 ms, full at 4K 82.1 → 7.34 ms, full at 32K 835 → 56.4 ms. Kernel: 207 VGPRs, no scratch.
- 2026-11-21: stage 1 integrated (`STRATA_INKLING_FLASH=1`, n ≥ 32). Review (general-purpose, two rounds): accepted denominator rounding (keys summed apart from the sink, per-block sums, compensated sum) and the masked-block NaN with a −inf sink (m starts at max(sink, −FLT_MAX)); refuted a non-finite V of an unattended future key (its own and every later token's output is non-finite in both paths). Per-key selects spilled (256 VGPRs, 880 B scratch, 3.6× slower) and were replaced.
- 2026-11-21: R2 evidence. All experts on CPU (same placement): flash off is bit-identical to the deployed binary on the 1,064-token top-5 trace; flash on keeps all 8 top-1 IDs, largest top-1 probability change 0.059 (prompt batch 1 vs 512 on the plain path alone differs by 0.126; different expert placement 0.103). Batch 1/32/512 with flash: same IDs. SWA wrap restore at 4,096 = cold replay `[152717, 13, 198, 3256, 150130, 25]`. Full HIP build passes except `poly_fattn_test` (NVIDIA-only, pre-existing); `poly_workers_test` passes.
- 2026-11-21: R1 pair (frozen copy of the restored profile, batch 512): `pp` A 258.1/259.6 → B 443.9/448.6 tok/s (+72%); `pp8192` A 199.3/199.5 → B 445.6/447.2 (×2.24); TG A 23.77/24.41, B 24.11/23.87.
- 2026-11-21: side effect found and repaired: `bench.py` passed production's `--expert-profile`, so 13 benchmark runs added ~20M of 26.9M counts to the live profile. The file is additive (prior + use); one measured run each (786,432 counts at 2K, 3,145,728 at 8K) was subtracted (10 × 2K, 3 × 8K; 169 counts clamped), leaving 9.63M; the polluted file is kept in scratch. Benchmarks now use a fresh copy of a frozen profile per run.
- 2026-11-21: stage 2 profile (flash, batch 512, 2K): 4.61 s span, 1.40 s GPU idle — 554 ms waiting for CPU experts before `moe_sum`, 277 ms chained H2D staging, 139 ms rocBLAS first use; kernels: NVFP4 MMQ 1.32 s, Q8_0 MMQ 0.31 s, flash 0.29 s. A layer's ~68 uncached experts (~306 MB) are restaged for every 512-token call. Screens with flash: batch 1024 PP 515.1/475.1 (2K/8K), batch 2048 611.5/600.8; TG batch 512 23.99/23.87 vs 2048 23.23/23.21 (−3.0%, 1,216 vs 1,415 cached experts).
- 2026-11-21: stage 2 at batch 2048 (8K trace, flash): 13.57 s, idle 1.74 s, 1.31 s of it GPU waiting between staging copies; DMA busy only 2.67 s (~70 GB at ~26 GB/s) — the host copy from the mapping into the pinned buffers is the limit. `STRATA_POLY_STREAM_STAGE=1` screen: 602.7 → 632.2 (8K). Probe: `hipHostRegister(ReadOnly)` on the mmapped GGUF works, DMA 26.6 GB/s, bytes match, ~70 ms/GiB.
- 2026-11-21: `STRATA_POLY_DIRECT_STAGE=1` (opt-in, engine.cpp): mapper_ registers the uncached experts' page-merged spans (8.0 GiB here); stage() copies registered experts from the mapping and the rest through the pinned buffers, each to its place; h2d() splits copies at registered-span edges; padding of whole-tile steps zeroed. Review (general-purpose, three rounds): accepted partial-page registration breaking cache refills (fixed in h2d), default-path teardown change (removed), stale gap bytes (zeroed); found nothing else. Evidence: with equal expert placement the 1,064-token and 8K top-5 traces are bit-identical with the switch on and off (md5 follows placement only: 1168–1170 cached → `23e6f01d`, 1174–1178 → `1be7960e`). Warm `--serve` (second and third prompt of a process, batch 2048, flash): 8K 610.1/608.6 → 692.2/676.9 tok/s; 2K 641.7/599.2, 636.3/600.1 → 715.0/673.4, 729.4/674.9.
- 2026-11-21: warm 8K kernel split per prompt (direct staging): NVFP4 MMQ ~4.0 s, flash ~1.82 s, Q8_0 MMQ ~1.08 s, conv ~0.37 s, control `gemm_small` ~0.37 s, mix ~0.26 s; idle ~0.83 s.

## Completion

- Resolved outcomes:
- Commands and artifacts:
- Constraint and diff-scope check:
- Final status:
