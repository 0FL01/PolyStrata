# One-sequence live DFlash2 on gfx1031

`strata-poly-dflash2-live` is a **one-shot experimental target**, not a replacement server.
It uses the same `df2_open/append/propose` as [standalone native replay](../native/README.md).
Only the three committed intermediate residuals and the final target norm go to the draft.
Guesses never advance its ring. On a partial hit `engine.keep(fed)` and `df2_append(fed, …)`
consume the same prefix. The displayed correction/bonus or terminal token is not fed early.
The stock target sampler (including history and RNG) picks from its whole vocabulary;
no rejection ratio, draft softmax or candidate logits substitute for target probabilities.
This **does not** make the batched target's floating-point reductions equivalent to W1.

Build from a matching, already-built gfx1031 Release Ninja tree. The builder substitutes two
translation units in a separate executable; it does not overwrite production source, object
files, or the service. Examples below use the private campaign inputs:

```sh
R=$HOME/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon
M=$HOME/.lmstudio/models/unsloth/Inkling-2-Mini
CFG=$HOME/.config/strata/inkling2-mini/strata-inkling-2-mini.json
SEED=$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK/profile.seed
bash rdna2-rnd/dflash2/native/build.sh "$R" build-poly-hip
python3 rdna2-rnd/dflash2/live/build.py --out "$R/live-bin"
python3 rdna2-rnd/dflash2/live/measure.py --run "$R" --out "$R/new-fixed-q40" \
  --config "$CFG" --profile "$SEED" --domain code_python --mode live --cap 3072 \
  --weights "$M/DFlash2/Inkling-2-Mini-DFlash2-Q4_0.gguf" --steps 65 --k 3 \
  --library "$R/libstrata-dflash2.so" --binary "$R/live-bin/strata-poly-dflash2-live"
```

`--k 0` measures loaded W1, not production: it pays for appending committed features but does
not draft. `--serial-verify` drafts but feeds each new ID only after an ordinary W1 target pick.
`--adaptive` is a live policy pilot: chooses W1, suffix lookup or DFlash2 and the width up to
`--k`; uses observed round costs, measured whole-block draft cost and conditional prefix hits.
With `--dspark-weights` it also tries DSpark: both models are resident and ingest committed
rows even when not chosen. Changing a request's drafter is not free. `--adaptive-budget MS`
estimates a round-time cap, including width-specific proposal cost, and explores narrower
windows first. It cannot preempt a round or guarantee a delivery SLA; even W1 may miss a cap.
`--parity` writes all FP32
logits sampled from, then committed target convolution/ring and KV row bytes to a private
`target.tape`. Use `cmp` on matched-length legs; **do not** time diagnostic recordings.
`--warm-target` can import a target-only snapshot with 2,049 or more fresh suffix IDs; there
is no draft checkpoint import/export, server, vision, multi-sequence batching or separate target codec.

`measure.py` copies the seed profile for each fresh leg. No `--cap` means the installed reserve,
profile and live swaps; `--cap` fixes the joint expert/KV arena and disables swaps. Always check
actual slot counts (and placement fingerprint where available): equal MiB is not itself equal
placement. `min_free_mib` is sampled device-wide, `peak_rss_kib` is process RSS (includes mapped
file pages); neither is a tensor-payload measurement. `ttft_ms` starts with native prompt read,
not model load; `p90_ms` uses successive displayed-token times and includes inter-round stalls.

`long_taps.py` restores a real **target-only** snapshot, appends 2048 fresh prompt IDs and checks
the complete DFT1 file against committed input IDs and absolute positions. `replay.py` consumes
those rows at their original RoPE positions. Its `draft_ms` is proposer-only; no target, no
session restore, no host-to-device feature upload or verification is included. Its closed-loop
histogram is derived by walking `1 + accepted_prefix` over all-position predictions, dropping
anchors with less than seven remaining fed labels. `measure.py --mode dspark-log` is for forced
same-history replay **only**, not a delivered-token benchmark.

### First adaptive round budget screen

RX 6700 XT, Q4_0, exact target overlay, 74-ID Python prompt, 96 greedy output IDs, 701 experts
at the same placement fingerprint and zero swaps. Every leg matched all 96 IDs. In `$R/adaptive-budget-v1`,
loaded W1 before/after measured 22.83/25.37 tok/s (p90 44.8/42.2 ms); fixed W4 measured
35.89 tok/s (103.6 ms). The unconstrained adaptive pilot reached 32.25 tok/s (115.2 ms).
With a 75 ms budget the policy chose W1/W2 for 12/45 rounds, with no measured round over budget:
29.01 tok/s and 66.3 ms p90. In this one screen the cap lowers the adaptive stall cost, but is
still slower than fixed W4 and higher-p90 than W1. Do not infer service or interactive-agent latency.

The reviewed follow-up requires an in-engine `DF2_BUDGET` receipt and does not charge DSpark's
first head setup to its warm price. In `$R/adaptive-budget-v2`, the 75 ms pilot on the same Python
prompt matched all 96 IDs and improved loaded W1 from **25.44 to 29.32 tok/s**, with internal p90
**42.2 → 63.2 ms**. With both drafters loaded on the tool-call fixture (699 fixed slots), all 67 IDs
matched and W1 → pilot was **27.80 → 30.81 tok/s**, p90 **39.1 → 61.3 ms**. DSpark was chosen for
24 of 37 rounds, DFlash2 for one, lookup for nine, W1 for three. Both pilots exceeded the estimate
once. The available VRAM and placement were matched within each pair, not across the single/dual
models. The controller does not unload a draft or change its bits when memory gets tight.

The adjacent `--wave8` W4 leg, using the same DSO and token history, reduced mean draft time
5.81 → 5.44 ms (26 rounds), with measured throughput 35.89 → 36.58 tok/s. The two W1 controls
show much more drift than that throughput delta. This is a positive proposer result, not proof
of an end-to-end `--wave8` speedup.
