# P2: Cross-pass lossless KV screening

**FAIL the 20% feasibility gate on the current codec family.** On the fresh mixed Tideglass snapshots, the best
practical layouts save 2.63%, 2.29%, and 1.19% of aggregate full-context Q8_0 KV at 2K, 16K, and 64K. The more
permissive Zstd diagnostic saves 11.81%, 10.18%, and 8.50%, respectively. Do not queue CP-P2-04 on this evidence.

## Inputs and exactness

CPU analysis on a Ryzen 7 5700X; capture supplied by the orchestrator on an RX 6700 XT (`gfx1031`). Source baseline
`c0dd0e52`; Tideglass Inkling 2 Mini Unbound v2.1 Q4_0, Q8_0 KV, plain serial decode, 200,000-token capacity,
2,048-row prompt batch, 2,048 MiB reserve, eight workers, no draft or lookup. One mixed trajectory per length;
each snapshot was saved after 256 decode steps. See [dataset manifest](kv-dataset-manifest.json).

The parser checks both STRSESS v1 checksums with the production `SessionHasher` (not standard XXH64). It checks
geometry, range, IDs, every record's size and the trailer. The primary `pread` offset calculation and an independent
`mmap` plane walk agreed on all bytes sampled, including a separate test of a tile crossing record boundary 1,926.
Regular sampling takes 12 evenly spaced 32-row tiles per snapshot, including first and last complete tiles: 384
rows and 6,684,672 bytes per capture. Each visit's K codes, K scale bits, V codes and V scale bits are hashed
separately. Scale bytes are opaque FP16 bits; the reader does not convert them to floats. Captured checksums and
fingerprints identify files, not the provenance of untrusted inputs. Use only immutable, trusted snapshots.

`Inkling2::local` and `physical` give eight full-context visits: two singletons (5, 47) and two triples
(11, 23, 35; 17, 29, 41) for physical layers 11 and 17. `save_card_rows` stores each record in visit order,
then in split planes of 1,024, 64, 1,024 and 64 bytes per row. This is 2,176 bytes per visit per token, or
3,481,600,000 bytes (3.24 GiB) at 200,000 tokens. The parser recognizes DSpark layout 2 but excludes the extra
drafter plane and SWA/convolution state from the experiment. All three real captures were layout 1.

## Measurements

Positive percentages are savings against all eight raw full-context visits, not just compressed visits. The
practical sizes project the measured sample average to each snapshot's committed length. This is not measured
VRAM allocation or decode throughput. Machine-readable summaries: [codec comparison](codec-comparison.json)
and [entropy](kv-cross-pass-entropy.json).

| Layout | 2,304 rows | 16,640 rows | 65,792 rows |
|---|---:|---:|---:|
| Raw payload (bytes) | 40,108,032 | 289,669,120 | 1,145,307,136 |
| Practical independent | −1.84% | −0.65% | −0.51% |
| Practical token-adjacent | −0.68% | 0.24% | 0.12% |
| Practical cross-pass, reference pass 1 | 1.51% | 1.13% | 0.45% |
| Practical cross-pass, reference preceding pass | 2.63% | 2.10% | 0.99% |
| Practical cross-pass, adaptive | 2.63% | 2.10% | 0.99% |
| Practical hybrid of independent, token-adjacent and cross-pass | 2.32% | 2.29% | 1.19% |
| Zstd-9 independent | 6.43% | 6.28% | 6.27% |
| Zstd-9 best of independent and token-adjacent | 8.32% | 7.69% | 7.09% |
| Zstd-9 cross-pass, anchors/singletons kept raw | 7.63% | 6.19% | 4.65% |
| Zstd-9 best of all measured references, all visits eligible | 11.81% | 10.18% | 8.50% |

K correlates more than V, and both weaken at depth. Across the six cross-pass reference choices, mean K code
equality falls from 3.86% at 2K to 2.60% at 64K; mean V equality falls from 2.61% to 1.81%. At 64K the mean
empirical conditional code entropy is 6.87 bits (K) and 7.20 bits (V) with a cross-pass reference, versus 7.11
and 7.33 with the previous token for core visits. Exact scale matches are under 0.2% on average for either
kind of reference. The raw payload is about half anchors/singletons, so a cross-only scheme must save at least
40% on its four dependent visits to reach 20% aggregate before overhead. The best observed hybrid at 64K would
save about 13 MiB of payload including format overhead, not counting any decompression cost.

The practical format indexes one K or V head per 32-row tile (four 32-code blocks/row). It chooses among signed
delta packing, XOR bitplanes, small signed deltas with raw outliers, zero-run/literal pairs and raw blocks. It
codes each scale byte plane separately by minimum + packed offset, reference XOR or raw; incompressible tiles
fall back to raw. The projection charges actual block and tile tags, 16-byte tile padding, an arena offset per
head/side/tile, active raw append staging and raw reference visits where applicable. The allocation sensitivity
model rounds the arena and metadata/staging allocations to an assumed 2 MiB grain and compares with raw using the
same grouping. Comparing against separately rounded engine planes can count allocator packing as compression,
especially at 2K; that comparison is emitted only as a diagnostic. It does not model VMM lookahead or actual
allocator mappings. The best savings in the matched allocation model are 0%, 2.16%, and 1.28%. Cross-pass
references never point forward; a pass-3-to-pass-2 choice can require two earlier visits. A token-adjacent chain
restarts at each tile and does not borrow the preceding tile's last row.

All 4,224 head-tile encode/decode trials per capture matched the original codes and scale bits byte for byte.
Every Zstd stream also round-tripped. The latter uses large per-visit/side streams over selected tiles with no
random-access, staging or offset cost, and takes the best observed size across raw, residual, and bitplane arms.
It is deliberately optimistic for the tested transforms, not an information-theoretic upper bound. No dictionary
arm was selected, and there was no engine integration, attention-output comparison, rollback, or token-ID A/B.

## Reproduce

From the repo root, with a C++20 compiler and Python with `numpy` and `zstandard`. Tested with Python 3.13.5,
NumPy 2.4.4, zstandard 0.23.0/libzstd 1.5.6. Full per-pass, K/V, head and tile statistics, distributions,
sample digests and codec cost breakdowns go to local scratch; no raw KV is written by these scripts.

```sh
kv=docs/goals/arch-decode-recon/kv
root="$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK"
out="$root/kv-cpu"
mkdir -p "$out"
c++ -std=c++20 -O2 -Wall -Wextra -Iinclude "$kv/snapshot_hash.cpp" \
  src/core/conversation_file.cpp -o "$out/strsess-hash"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
python3 "$kv/test_recon.py" "$out/strsess-hash"
for n in 2048 16384 65536; do
  python3 "$kv/recon.py" "$root/mixed-$n/session.bin" --checker "$out/strsess-hash" \
    --workload mixed > "$out/$n.json" || exit
done
```

128K/200K and separate prose/code/agent trajectories were not tested. If a long capture already exists from
another pack, the next useful check is the same bounded CPU screen there; do not create a GPU decompressor or
reserve another prefill solely for this codec family unless new offline evidence approaches the aggregate gate.
