# P5 — Fresh Inkling H0 baseline

**Status: passed.** [Baseline result](h0-result.json) records frozen quality and fresh H0↔H0 timing. Timing noise is larger at 2K and limits the conclusions below. See the [lock](experiment-lock.json) for H0 SHA-256, the Release HIP `gfx1031` binary, observer, GPU, corpus and runtime settings. Production was inactive for the standalone runs on RX 6700 XT.

## Exact teacher-forced NLL

Fresh process and seed-profile copy per frozen document; no chat template, no KV reuse. All 2,048 rows matched the exact target ID and expected position, and all scores were finite. Mean NLL in nats per token:

| Prefix | Russian | English | Python | C++ |
| --- | ---: | ---: | ---: | ---: |
| 2,048 IDs | 1.613798 | 1.353743 | 0.335666 | 0.683690 |
| 16,384 IDs | 2.194203 | 1.111470 | 0.697864 | 0.760122 |

Mean across the eight equally sized windows: **1.093820**. Cache capacity at load was 1,197–1,238 experts; lowest free VRAM sample was 1,973.6 MiB. Guard reason lists were empty. Use paired document differences for later NLL comparisons, not 2,048 independent observations. Raw TSVs and comparator input are under `$WORK/p5-nll.c97OLH/`; `h0-reference.json` has SHA-256 `b75af2f648a8a51e6022aca71948904ec8ff948bb8c4debf3ab93df426510e90`.

## Short tasks and frozen critical smoke

Using the locked `high` effort (template value `0.9`), greedy seed 42, and frozen per-task budgets, **32/32** answers passed; every case ended with ID `200006` rather than reaching its token budget. Verified against the frozen validators with the native Inkling reasoning blocks removed. Python is checked by a bounded AST interpreter and Rust by a token skeleton, not a compiler. No tool calls are dispatched. Private results and raw outputs are in `$WORK/p5-chat-h0/`; `chat-result.json` SHA-256 is `350ad99d954d8eb5586546b76a580dc6daf8d5be3722d75ab7b26cb5de95a844`.

Freeze critical smoke now, before H1: guarded load, finite observer output, and the four pre-marked tasks `2k-py-01`, `2k-rs-01`, `2k-json-01`, `2k-tool-01`, each requiring both validator pass and native stop. Record full task counts separately. This high-effort pilot did not measure the separate Orca low-effort refusal cases.

## Timing gate

The release-only [timing runner](../../../tools/inkling_weights/README.md#guarded-runtime-pilots) ran H0 in both arms with two warm-ups and `ABBABAAB` at each length. All 20 runs emitted the exact 512 frozen IDs (511 timed decode steps), with empty guard reason lists. No historical timing is used.

| Prefix | Decode median A / B (tok/s) | Adjacent B vs A, median and range | Prefill median A / B (tok/s) | Expert hit median A / B | Peak process RSS / lowest sampled free VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K | 44.70 / 44.97 | −0.18% [−4.10%, +7.92%] | 898.7 / 894.8 | 82.8% / 82.9% | 14.57 GiB / 1,983 MiB |
| 16K | 43.97 / 43.99 | +0.06% [−0.86%, +0.67%] | 901.9 / 904.4 | 84.9% / 84.9% | 15.04 GiB / 1,937 MiB |

Median load was 1.4 s in both arms at both lengths. Measured expert capacity ranged 1,226–1,240 at 2K and 1,233–1,250 at 16K. Sampled peak card VRAM usage was 10.05 and 10.09 GiB, respectively, including desktop use. Maximum major page faults per measured process were 25 and 1. The +7.92% pair at 2K paired 45.09 with 41.78 tok/s; the slower process had a lower sampled median core clock (2,202 vs 2,278 MHz), higher CLI CPU expert time (1.9 vs 1.6 ms/token), and no corresponding jump in major paging. These observations characterize run-to-run resource variation, not proof of a single cause. Exact work, finite NLL and task outcomes were stable; the two arm medians were close. Do not discard slow runs.

Keep the observed absolute adjacent-pair envelope (7.92% at 2K, 0.86% at 16K) as a warning against attributing small speed changes to the weights. Within it performance is inconclusive; a promotion decision near the plan's **5% maximum loss at either length** needs the allowed bounded follow-up rather than a claim from a noisy pair. No new critical failure and demonstrated benefit still apply. Raw per-run telemetry (temperatures, core/memory clocks and all samples, including warm-ups) is under `$WORK/timing-mme75147/`. Next: H1-B on the verified copy; a successful build is not a quality or speed result.
