# Inkling2 target verifier probe (RX 6700 XT)

This is an isolated HIP probe, not a drafter or a production engine change. It builds against
`strata-poly` and only links scratch copies of the five instrumented translation units. All widths,
including the serial control, use the same target allocations and model. For each context, every
arm must preserve the serial run's exact set of resident `(physical layer, expert)` IDs at both
load and exit. Different contexts use different fixed arena sizes to leave room for their KV.

Three results are intentionally separate:

- **Serial:** an ordinary greedy `GEN` without lookups, forced IDs or a drafter.
- **Forced:** put the serial IDs through fixed-width windows and keep all rows. Record *every*
  real target argmax, including the one after the last row. The emitted IDs are forced and prove
  nothing about correctness by themselves. Window cost is a ceiling with perfect draft inputs,
  not an end-to-end generation rate.
- **Accepted:** offer the serial IDs as synthetic draft proposals, compare them with actual target
  argmaxes, use `keep()` for the matching prefix, and emit only target picks. On the first departure
  from the serial sequence, fall back to serial decode from the accepted state. This tests actual
  target acceptance and the resulting full sequence. It still does not measure any draft model,
  its cost, or its acceptance rate on new inputs.

The runner restores a native, no-draft Inkling2 `STRSESS` image and uses its exact saved token IDs
plus the same fresh token (ID 198 by default). `RESTORED`, `RESUME` and `DONE` must confirm all saved
rows were reused and exactly one prompt token was read. Do not infer the saved length from the original
prompt CSV: a disk session may include generated tokens. In particular, `mixed-199743/session.bin`
holds 199,999 positions, not 199,743; the default 199K measurement uses `seed200/session.bin` at
199,104 saved rows. The engine additionally verifies the snapshot's geometry, model and checksum.

```sh
R="$HOME/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon"
python rdna2-rnd/dflash2/verifier/build_probe.py --output "$R/probe-new"
python -B rdna2-rnd/dflash2/verifier/sweep.py \
  --binary "$R/probe-new/strata-poly-probe" \
  --config "$R/../native-donor-bc2b/baseline/fixed.json" \
  --profile "$R/../../arch-campaign.XWM1yK/profile.seed" \
  --contexts 2k 16k 64k 128k 199k --widths 1 2 4 6 8 --steps 48
```

Use `--snapshot-root` and `--long-snapshot` for other fixture locations. `--steps` is the
number of decode rows after the first generated ID, unless serial generation reaches EOS earlier.
Start with `--fixes 0` (the default). Only add diagnostic masks at an observed divergence:
bit 1 uses the serial GEMV launch rule for small-output F32 projections; bit 2 calls decode
attention separately per row, preserving each serial row's chunk sizing and combine order;
bit 4 suppresses the verifier's column kernels. These controls are not production fixes.
In particular, grouping the serial attention chunks for an entire W8 window does not fit the
existing split scratch at long spans: 160 chunks × 8 KV groups × 8 rows exceeds 4,096 blocks.
The per-row control pays separate launches to remove that limit, and does not price a future exact
batched implementation.
To compare device operation fingerprints, build a separate binary with `--trace` and add
`RECON_TRACE_POS` (an absolute *input* token position) to a copy of the fixed JSON's `env` object.
Trace runs add device synchronizations; do not use their timings.
To test partial-window rollback with a known wrong proposal, set `RECON_BAD_DRAFT_INDEX` (zero-based
output index, e.g. 1) in a fixed JSON copy and select only `--modes accepted` with a width above one.
Compare the whole emitted sequence with W1; a lower accepted/offered count alone is not sufficient.

Each engine process runs under `flock R/gpu.lock` and `run_guarded.py --warn-amdgpu-faults`.
Fresh timeout/reset/hang/OOM, insufficient free VRAM or a nonzero exit stops the sweep. A guard
warning for a memory fault or a past decoder timeout is not a failure. Rerun the same command to
reuse only matching validated `complete.json` checkpoints under `R/verifier/<series>/`; an interrupted
or failed attempt is kept and the retry gets a new directory. Snapshots are read-only inputs; each
engine gets its own reflink of the profile seed. Legacy `R/oracle_sweep.json` and
`R/runs/oracle-*` are not consulted: they mixed serial-only summaries with different raw verifier runs
and did not fix placement. The new probe does not rewrite the separate DFT1 tap artifacts.
Use `python -B rdna2-rnd/dflash2/verifier/summarize.py "$R/verifier/target-v2"` to report completed arms.

Equal greedy IDs do not imply equal arithmetic. At 65,793 input tokens, the stock W8 trace first
changes at the visit-0 router; with the W1 router reduction it first changes at visit-5 global
attention. With both controls all 816 traced buffers for the selected input row match W1. The
[short-context trace](../divergence/README.md#short-witness-first-query-at-position-2305-zero-based)
also compares full logits. Keep single-pass rates from a busy shared host separate from those
correctness results; neither measures a live drafter.

### Completed stock matrix

All **55** validated receipts match 49/49 actual target picks and emitted IDs against serial W1,
with identical physical resident experts at load and exit within each context. The measured spans
were 2,305 / 16,641 / 65,793 / 131,329 / 199,105 tokens at 1,090 / 1,016 / 847 / 607 / 358 slots.
No fresh GPU warnings or guard stops; minimum sampled free VRAM 2,203.7 MiB. The complete table is
`$R/verifier/target-v2-full.md`. At 199,105 the single pass measured serial W1 **16.10 tok/s** and
stock W8 forced/accepted **29.67/29.72 tok/s** with perfect guesses, before real draft cost.

At 65,793 and the same 847 slots, the quieter repeat measured **23.76 serial / 43.07 stock W8 /
44.58 W8 exact-router / 40.74 W8 exact-router and per-row attention**. The earlier 64K stock W8
sample was 13.26 tok/s. Do not use the mixed-load figures to rank width or infer a deployment
speedup. Full traces establish why same greedy IDs are insufficient; actual DFlash2 qualification
uses the separate [exact live verifier](../divergence/README.md#live-dflash2-exact-verifier).
