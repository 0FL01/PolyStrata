# CP-02 — real 2K sample

**Pass for trace integrity and greedy IDs on this fixture.** On RX 6700 XT (`gfx1031`), the binary trace regenerated the
saved JSONL and summary byte-for-byte. All three runs emitted the same 257 greedy IDs, with 256 generated inputs fed back
to the model. The trace covers exactly positions `[2048, 2304)`, with all 48 visits per input: 12,288 visit records plus
4,032 cache events. Storage is `48 + 12,288 × 288 + 4,032 × 64 = 3,797,040` bytes (3.62 MiB versus 12.11 MiB JSONL).

The cache replay agreed with every visit’s resident mask, combined pending mask, pending count and selected slot offsets.
The initial snapshot has 1,159 visible experts plus 17 pending slots, out of 1,176 total. All 1,044 requested swaps
became visible at the next token, not necessarily when the DMA finished. During the trace, 79,808 of 98,304 ordered
selections hit VRAM (81.18%); 18,496 ran on CPU, including 524 selections of pending experts. The `DONE` counts have
384 extra lookups, consistent with the final prompt input computed singly outside the capture range. All 256 `cover()`
calls were no-ops at 4,096 already covered rows; no `shrink_cache()` occurred. These runs had no guard stops or new warnings;
all three guard logs carried the same 22 earlier boot warnings.

| Leg | Decode t/s | Mean ms | Median ms | p99 ms | Loaded slots |
|---|---:|---:|---:|---:|---:|
| `mixed-2048` (unmodified, untraced) | 32.859 | 30.433 | 22.643 | 146.466 | 1,171 |
| `trace-off` (patched, disabled) | 42.328 | 23.625 | 22.297 | 44.554 | 1,176 |
| `trace-on` (same binary, enabled) | 40.780 | 24.522 | 23.258 | 41.124 | 1,176 |

Throughput uses the sum of the 256 recorded arrival intervals. The single matched-binary pair observed **−3.66% t/s**,
or **+0.896 ms/token** mean with tracing enabled. Dropping the
first arrival interval on both legs gives 42.499 vs 40.915 t/s, so the observed difference is not explained by first-use
setup alone. Trace-on ran first; equal slot capacity and request hit counts are not proof of identical live placement.
The earlier baseline had five fewer slots and three arrival intervals above 270 ms; this is not evidence of a speedup
or a measurement of disabled-path cost. Repeat interleaved baseline/off/on legs at matched placement to estimate both
disabled-path cost and enabled overhead.

The traced visit spans sum to **22.812 ms/token** on the GPU stream; the exposed-join proxy sums to **5.752 ms/token**.
Prelude and coda account for 48.5% of that proxy in 12 of 48 visits. This identifies where to investigate next, not
recoverable time or GPU busy time. The off legs have no per-visit records: exact logits, routing weights, placement, and
committed state were not compared. The result establishes raw greedy-ID parity for this one short fixture only.

See the [manifest](trace-manifest.json) for input hashes, raw metrics and definitions. To repeat the CPU integrity check,
run [`collect.py`](collect.py) on `trace-on/visits.psv` with fresh `--jsonl` and `--summary` paths, then `cmp` both outputs
against the saved files. No GPU work was performed for this validation.
