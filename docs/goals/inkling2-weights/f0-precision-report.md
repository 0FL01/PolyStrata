# F0: Precision comparison on RX 6700 XT

These are exact NLL comparisons on the same eight frozen documents (256 targets each; 2,048 scores per model), not 2,048 independent quality observations. H2/H3/A1 use the [reflink artifacts](cow-report.md) and the locked `gfx1031` observer and settings. Reused the earlier same-boot `2k-en` smoke scores and the complete H0/H1 results. The other 21 H2/H3/A1 runs had only historical AMDGPU warnings, no fresh warnings and no stops. No reboot or full donor download.

| Variant | Aggregate NLL ↓ | Paired mean vs H0 | Better documents | Illustrative 95% t interval of mean paired delta |
| --- | ---: | ---: | ---: | ---: |
| H0 | 1.093820 | — | — | — |
| H1 | 1.093273 | −0.000547 | 5/8 | [−0.006561, +0.005468] |
| H2 | 1.091311 | −0.002508 | 4/8 | [−0.010433, +0.005417] |
| H3 | 1.089813 | −0.004006 | 4/8 | [−0.017757, +0.009744] |
| A1 | 1.102403 | +0.008583 | 2/8 | [−0.012796, +0.029962] |

Intervals describe the dispersion of paired document means in this small fixed pilot, not a general-population guarantee. H2 vs H1 was −0.001962 nat/token (document SD 0.007299, 4/8 improved); H3 vs H2 was −0.001498 (SD 0.014853, 3/8 improved); A1 vs H3 was +0.012589 (SD 0.018161, 2/8 improved). None shows a clear quality benefit over H0. H3 also truncated one [ordinary low-effort control](orca-low-baseline-report.md) that H0 passed.

Paired per-document changes vs H0 (nat/token; negative is better):

| Document | H1 | H2 | H3 | A1 |
| --- | ---: | ---: | ---: | ---: |
| 2k-ru | −0.007735 | −0.011028 | −0.009803 | +0.007001 |
| 2k-en | +0.004331 | −0.008882 | −0.037056 | −0.049221 |
| 2k-python | +0.002778 | +0.004892 | +0.005350 | +0.018995 |
| 2k-cpp | +0.012751 | +0.014180 | +0.017203 | +0.023985 |
| 16k-ru | −0.008742 | −0.009555 | +0.005068 | −0.000080 |
| 16k-en | −0.006170 | +0.001892 | −0.006475 | +0.021948 |
| 16k-python | −0.000947 | +0.001004 | −0.011550 | +0.033454 |
| 16k-cpp | −0.000639 | −0.012569 | +0.005213 | +0.012580 |

Raw evidence under `~/.cache/opencode-tmp/opencode/inkling-weights/same-boot.qayuV0/`: `h2-full-nll.json` SHA-256 `0f13a892f42b3e507d4b8daab069e1f96ef3b677438dfef3d1abdc322da6db41`; `h3-full-nll.json` SHA-256 `eb8f3805c875ae7ea230da66e295581fec21c5a4f224ec033a1cf6035ec00519`; `a1-full-nll.json` SHA-256 `7a15fac6383e2291fa4a42d077b7eae08b9f1fe539aba48e056f5873ebb38e82`.

## Decision and handoff

**Keep original H0** (`e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37`). H1–H3 have mixed quality results. A1 is rejected for promotion: it had worse NLL on six of eight documents and the low-effort pilot reproduced no target refusal to improve. It was not timed as a finalist; no speed claim is made. D0 and A2 were not opened. A larger test would need a representative reproduced workload for which a gain is plausible, not more repetitions of this negative refusal pilot.

Delivered A1 as an **experimental file, not the production model**, at `/home/stfu/.lmstudio/models/unsloth/Inkling-2-Mini/Tideglass-OrcaRouter-Inkling-2-Mini-A1.gguf`, SHA-256 `eb2a4fb30cd4fafdc92d42d25b09f52c97ad1d2189f89dc95933998cfa25da82`. Verified the full file and published the same CoW inode beside H0 without a second GGUF; the old scratch path is a symlink for replay. Btrfs error counters did not increase. The five old full copies were already removed as documented in the [CoW report](cow-report.md).

Restored `strata.service` with the unchanged config (`f38fe0d68292a419fea2bbb4bf3eb23df4dc1bf7da73df17e2eb3d920d3e792c`). It runs the pinned `strata-poly-c014d6ec` on the **original H0**, listening only at `127.0.0.1:8080`. Authenticated model listing and bounded chat returned 200 and the expected answer stopped normally; unauthenticated model listing returned 401. No fresh AMDGPU fault, reset, ring timeout or OOM during this smoke. No reboot or autostart change. Private evidence: `~/.cache/opencode-tmp/opencode/inkling-weights/restore-h0.AsAR29/final.json`, SHA-256 `902e0c364ff227056d3928e366bf7b386ce6ff2589e1013616d880c3f5156084`.
