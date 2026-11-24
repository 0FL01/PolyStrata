# Goal: faster exact Inkling DSpark decode on RDNA2

Status: active
Source: user request to raise DSpark decode t/s without divergence, then implement, reprofile, commit, push, and deploy wins.
Last updated: 2026-11-24

## Objective

Raise end-to-end decode throughput for Inkling-2-Mini Q4_0 with native DSpark on the RX 6700 XT without changing the target answer. Keep the default serial deployment until speculation matches it with live swaps and clears the prior rollout gate.

## Required outcomes

- R1: Keep only reproducible decode wins with exact relevant boundaries and greedy token IDs on gfx1031. Evidence: untraced paired engine runs, token IDs, and the closest kernel/state check; report losses and limitations. Status: in progress.
- R2: Push each qualified change to `origin/main`, build Release, deploy that binary without unqualified configuration changes, and use it as the next baseline. Evidence: commit, binary hash and loaded localhost health plus generation. Status: pending.
- R3: Qualify live speculative verification against serial, then measure adaptive decode against plain. Evidence: greedy IDs, fixed-seed sampler, committed state, and tool/code/JSON and 16K speed gate from the prior [DSpark goal](2026-11-24-inkling2-dspark-disk.md). Status: pending.

## Scope and checkpoint

`src/poly/inkling_dspark.cu`, the native Inkling verifier and its expert kernels, and affected documentation; use isolated prototypes outside the source tree. Keep model, precision, reduction order, routing/sampling, disk format, VRAM reserve and default no-drafter path unchanged unless separately proven exact. Serial GPU runs with the existing guard; no gfx1031 performance counters (`rocprofv3 --pmc`).

Complete the smallest unresolved outcome; let measured results choose the next experiment rather than following a fixed route. Stop a line of work that cannot meet exactness or end-to-end speed in its envelope; don't treat a leaf win as a rollout result.

- Initial baseline: `e116808d`. Live budget 3 vs attached budget 0 diverges at output 35 (2K) and 59 (16K); at 64K it is 31.34 vs 32.48 t/s on one restored fixture. In the probe, the first drafter attention variant preserved all FP32 intermediate/output bytes and reduced the 64K engine dispatch from 11.84 to 7.27 ms. The separate Q4_0 verifier tile leaf was exact; front-wait removal and commit fusion did not improve decode.
- Checkpoint 1 (R1): integrated gfx1031 value reuse behind `STRATA_INKLING_DSPARK_REUSE_V=1`. On the RX 6700 XT, fixed schedule and fixed 909 experts, restored 16K, four untraced pairs over 192 greedy IDs: median 30.014 → 30.754 t/s (+2.47%, four of four pairs). All IDs, proposed IDs and probability bits, verification decisions and the recaptured committed file matched; reported median first-proposal time was 7.49 → 6.13 ms. The earlier 64K live matched-route test measured 33.000 → 34.657 t/s (+5.02%) and exact IDs, decisions and snapshot with the same device instructions. Other live runs varied even in A/A; do not infer adaptive parity from these numbers. An isolated frozen 64K confirmation was stopped at 1,433 MiB free by the guard, with no GPU fault; use reduced arena and no direct staging in the next 64K control.
- Next (R2/R3): deploy the opt-in path in the Release binary, leaving the production configuration unchanged. Reprofile from it and measure the width-two live rollback/replay probe and the verifier tile independently.
