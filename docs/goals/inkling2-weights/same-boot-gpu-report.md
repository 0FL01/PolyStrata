# Same-boot GPU screening on RX 6700 XT

No reboot, driver change, or production change. The Inkling NLL, chat, and timing pilots treat AMDGPU memory/page-fault journal entries as warnings, for both old boot entries and fresh ones, per the user's instruction. The warnings are recorded separately in `guard/warnings.json`. Only those diagnostics are downgraded: failed/timed-out commands, invalid results, low VRAM, GPU hangs/resets and OOM still fail. Other callers of the standalone RDNA2 guard remain strict unless they explicitly opt in. This policy is not a finding about the root cause of the H1-E fault.

## Results

Used the [verified reflink files](cow-report.md), pinned Release HIP `gfx1031` binary and observer, and the frozen greedy workloads. Each candidate loaded and produced 256 finite exact target scores on the same `2k-en` window; each critical chat case passed its checker and emitted the native stop ID. The comparators are the existing H0 results on the same boot.

| Variant | `2k-en` NLL (nat/token) | Delta vs H0 | Critical chat | Sampled minimum free VRAM |
| --- | ---: | ---: | ---: | ---: |
| H0 (P5) | 1.353743 | — | 4/4 | — |
| H2 | 1.344860 | −0.008882 | 4/4 | 2,009.6 MiB |
| H3 | 1.316686 | −0.037056 | 4/4 | 2,009.8 MiB |
| A1 | 1.304522 | −0.049221 | 4/4 | 2,007.5 MiB |

All 15 inference runs had boot warnings only, no fresh warnings and no stop reasons. The NLL numbers are from one document: they establish finite inference, not a broad quality advantage. Raw artifacts are in `~/.cache/opencode-tmp/opencode/inkling-weights/same-boot.qayuV0/`. Five guard tests and 15 weight-tool CPU tests passed. The exact H1-E kernel tail, including the GCVM protection status line, was replayed through the final warning classifier as both historical and live events; strict boot behavior was unchanged. A real no-GPU warning-mode probe propagated child exit 13 unchanged.

The follow-up [low-effort baseline](orca-low-baseline-report.md) did not reproduce a false refusal. No production promotion or timing conclusion is implied by the smoke results.
