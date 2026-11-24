# H1-E — Quality measured; timing blocked by a GPU fault

**Verdict: blocked.** H1-B passed; the H0↔H1 timing comparison did not finish. No promotion, H2 build, or Orca run. Hardware: RX 6700 XT (`gfx1031`); model hashes in the [H1 build report](h1-build-report.md), observer and Release ELF in the [lock](experiment-lock.json). The timing run recorded lock SHA-256 `64e293dfc81b7a7cb326133dc9d8175b63d1dd2cf183c609311f92397a78af16` before the fault status update.

## Quality

The same eight frozen documents produced 2,048 finite scores on H1. Mean NLL (nat/token): **1.093273 H1 vs 1.093820 H0**, delta **−0.000547**, with **0.007194** sample standard deviation of the eight document differences. This is a mixed result, not a demonstrated quality benefit.

| Prefix | Russian Δ | English Δ | Python Δ | C++ Δ |
| --- | ---: | ---: | ---: | ---: |
| 2K | −0.007735 | +0.004331 | +0.002778 | +0.012751 |
| 16K | −0.008742 | −0.006170 | −0.000947 | −0.000639 |

The locked `high` greedy chat pilot passed **31/32** on H1 vs **32/32** on H0. All four critical cases passed and stopped with ID `200006`. Noncritical `2k-py-05` hit its 512-ID limit with no final answer; H0 passed with a native stop at 423 IDs. This is a length-limit failure, not an observed refusal. The `low` Orca diagnostic was not run.

Private quality evidence is under `$WORK/h1-e.1wjSaH/`, including `h1-reference.json` (SHA-256 `f2def4b0817c0b40db8e5ed382f0330693656f8e25868c81b8760a5eddfcc8d2`; incorporates the H1-B `2k-en` TSV once) and `chat/chat-result.json` (SHA-256 `18801569d6e979bc69461c9bf5618168bd447bf9444cbbabe114f20866ebddce`).

## Timing and stop condition

The [timing runner](../../../tools/inkling_weights/README.md#guarded-runtime-pilots) ran with the original H0 in A and the verified H1 in B. All eight measured 2K runs, the first five 16K runs, and all four warm-ups emitted exactly 512 expected IDs and cleared the guard. The 16K run `16k-06-A` loaded **H0**, then failed before emitting any completion IDs. It logged a ROCm memory access fault; the kernel logged `amdgpu 0000:2d:00.0: [gfxhub] page fault` at **2026-11-24T20:37:31.257+03:00** and named the H0 `strata-poly-c01` process (PID 56449, SDMA0 in the fault diagnostics). The minimum sampled free VRAM on that run was **2,036.4 MiB**, above the guard's 1,536 MiB floor. That sample does not establish the cause. A fault in H0 is not evidence that the H1 weights failed.

The unchanged guard stopped the run; there is no timing `summary.json`. Raw evidence is `$WORK/timing-2q9yjvnl/` (`failure.json`, `16k-06-A/stderr.log`, `16k-06-A/guard/`). A read-only `Guard.boot_clean()` recheck refused the current boot on that page fault. No runtime process remained; `strata.service` had already been inactive after reboot and remains inactive, with its original config SHA-256 unchanged. No further GPU work, guard override, driver change, or reboot was attempted. Keep the verified H1 and logs for resumption.

**Handoff:** the user subsequently shortened the test protocol; see the [execution status](../2026-11-24-inkling2-weights-execution.md). H1 can be an offline parent despite incomplete timing, but is not recommended for production. Do not join partial runs across boots or repeat the completed quality pilot without a change to its inputs. A fresh GPU run still needs a clean boot and a passing guard.
