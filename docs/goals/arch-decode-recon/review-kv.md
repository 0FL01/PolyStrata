# CP-R-01 — Partial independent KV review

**PASS for byte extraction and CPU round-trip on the 2K spot check. FAIL the 20% screen for the tested layouts;
do not queue CP-P2-04.** Reviewed the `c0dd0e52` engine and the `kv/` code published in `4141ffaa`. No GPU,
engine restore, or full sweep was run for this review.

## Extraction and round-trip

`Inkling2::local`/`physical` and `save_card_rows` agree with the parser: eight full-context visits, including
distinct visits for each pass of physical layers 11 and 17. A record stores K codes, K scale bits, V codes,
and V scale bits in separate planes per visit, using that record's actual row count. STRSESS header and
payload checksums validate the container, not GPU state or external provenance.

With the existing `strsess-hash`, `mixed-2048/session.bin` matched both manifest checksums, both fingerprints,
and 2,304 committed IDs (the prompt plus the first 256 of 257 emitted IDs). `sample(..., prefix=1952, count=2)`
selected `[0,32)` and `[1920,1952)`, the latter split 6/26 rows across record boundary 1,926. A separate
row-by-row read at writer-derived record payload offsets 92,885,368 / 109,649,368 / 126,413,368 matched all
1,114,112 sampled bytes. All 704 practical trials returned exact code and opaque scale bytes; 64 pass-3
head-tiles used CPU-decoded pass-2 bytes as the reference. The three published CPU smoke tests passed.
No byte mismatch found in this spot check.

## Population and savings

The fixture label `mixed` is not representative coverage. Retokenizing the repeated 8,092-ID prose/code/tool
fixture reproduces the saved prompt IDs. At 2K the prompt is **only** the beginning of `AI_SETUP.md`; at 16K and
64K the published tiles contain 256 prose, 96 code, and 32 decode rows each, and **no** prompt tool-result rows.
The 128K and near-cap summary additions also sample zero prompt tool-result rows. There is one trajectory per
length and no independent prose/code/agent holdout. Decode rows account for 64/384 sampled at 2K and 32/384
at 16K and 64K, versus 11.11%, 1.54%, and 0.39% of their respective snapshots. Do not interpret a between-length
change as a controlled depth effect or the sample mean as an unbiased full-cache estimate.

The [comparison](kv/codec-comparison.json) agrees with the saved CPU summaries. The denominator is **17,408
bytes per committed row over all eight visits**, not the session file or GPU allocation. Four visits stay raw
in cross-only arms, so the other four must save at least 40% before overhead to reach 20% overall. The practical
projection charges tags, padding, offsets, and raw append staging as modeled; its best aggregate savings are
2.63% / 2.29% / 1.19% at 2K / 16K / 64K. The 128K and near-cap summaries project 0.85% / 0.80%; their snapshots
were not independently reread here. At 2K hybrid appears to save **58.33%** against separately rounded raw
planes (96 → 40 MiB), but **0%** against raw with the same arena grouping (40 → 40 MiB). That difference is not
codec savings or a measured gain in free VRAM.

Zstd-9's 11.81% / 10.18% / 8.50% are optimistic savings from the tested transforms on sampled visit/side
streams without random-access or append costs. Neither those results nor empirical single-byte conditional
entropy set a global compression ceiling. CPU byte equality does **not** establish Q1 attention-chain or Q2
engine parity, including logits, greedy IDs, or state transitions for a candidate. If revisiting the screen,
use distinct held-out trajectories and weight prompt/decode position strata by the rows they represent before
projecting footprint.
