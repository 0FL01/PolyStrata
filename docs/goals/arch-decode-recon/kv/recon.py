#!/usr/bin/env python3
"""CP-P2-01..03: validate a trusted snapshot and compare sampled lossless KV layouts (CPU only)."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import zstandard as zstd

import codec
from snapshot import HEADS, PASS, PHYSICAL, STRIDES, TILE, VISITS, VISIT_ROW, require, sample

REFS = {23: (11,), 29: (17,), 35: (11, 23), 41: (17, 29)}
ARMS = {
    "independent": (0,), "token_adjacent": (0, 1),
    "cross_from_pass1": (0, 2), "cross_from_previous_pass": (0, -1),
    "cross_adaptive": (0, 2, 3), "hybrid": (0, 1, 2, 3),
}
GATE = 0.20


def entropy(counts):
    p = np.asarray(counts, dtype=np.float64)
    p = p[p > 0]
    p /= p.sum()
    return float(-np.dot(p, np.log2(p)))


def rounded(x):
    return round(float(x), 6)


def byte_h(x):
    return entropy(np.bincount(x.ravel(), minlength=256))


def cond_h(x, ref):
    counts = np.bincount(((ref.ravel().astype(np.int32) << 8) | x.ravel()), minlength=65536)
    return entropy(counts) - entropy(counts.reshape(256, 256).sum(axis=1))


def plane_stats(q, s, r, rs):
    d = q.view(np.int8).astype(np.int16) - r.view(np.int8)
    x = q ^ r
    eq = x == 0
    runs = np.diff(eq.reshape(-1, 32).astype(np.int8), axis=1, prepend=0, append=0)
    lens = np.nonzero(runs == -1)[1] - np.nonzero(runs == 1)[1]
    se = np.all(s == rs, axis=-1)
    ds = np.bincount(d.ravel() + 255, minlength=511)
    # Keep signed delta and zero-run histograms sparse; axes for rates are [tile,row,head,block,lane].
    return dict(code_bytes=int(q.size), equal=rounded(eq.mean()), delta_abs_le_3=rounded((abs(d) <= 3).mean()),
                code_zero=rounded((q == 0).mean()), code_H=rounded(byte_h(q)), code_H_given_reference=rounded(cond_h(q, r)),
                code_delta_H=rounded(entropy(ds)), code_xor_H=rounded(byte_h(x)),
                delta_signed={int(i) - 255: int(ds[i]) for i in np.flatnonzero(ds)},
                xor_zero_runs_within_block={int(i): int(n) for i, n in enumerate(np.bincount(lens)) if n},
                xor_bitplane_H_lsb_first=[rounded(entropy(np.bincount(((x >> b) & 1).ravel(), minlength=2)))
                                          for b in range(8)],
                code_equal_by_head=[rounded(x) for x in eq.mean(axis=(0, 1, 3, 4))],
                code_equal_by_tile=[rounded(x) for x in eq.mean(axis=(1, 2, 3, 4))],
                mean_abs_delta_by_head=[rounded(x) for x in abs(d).mean(axis=(0, 1, 3, 4))],
                scale_bits_equal=rounded(se.mean()), scale_equal_by_head=[rounded(x) for x in se.mean(axis=(0, 1, 3))],
                scale_byte_H_low_high=[rounded(byte_h(s[..., b])) for b in range(2)],
                scale_byte_H_given_reference_low_high=[rounded(cond_h(s[..., b], rs[..., b])) for b in range(2)])


class Diagnostic:
    """Large per-visit streams over the selected tiles; not independently indexed or a formal bound."""

    def __init__(self):
        self.enc = zstd.ZstdCompressor(level=9)
        self.dec = zstd.ZstdDecompressor()

    def compressed_size(self, data):
        coded = self.enc.compress(data)
        require(self.dec.decompress(coded) == data, "zstd roundtrip failed")
        return len(coded)

    def size(self, q, s, r=None, rs=None):
        # Scales retain explicit low/high byte planes. Predictive streams are inverted as well.
        scales = s[..., 0].tobytes() + s[..., 1].tobytes()
        if r is None:
            return {"raw": self.compressed_size(q.tobytes() + scales)}
        x = q ^ r
        sx = s ^ rs
        sx_raw = sx[..., 0].tobytes() + sx[..., 1].tobytes()
        packed = np.packbits(((x.ravel() >> codec.SHIFTS[:, None]) & 1), axis=1, bitorder="little")
        recovered = np.bitwise_or.reduce(np.unpackbits(packed, axis=1, bitorder="little") * codec.WEIGHTS[:, None],
                                         axis=0).reshape(q.shape) ^ r
        require(np.array_equal(recovered, q) and np.array_equal(sx ^ rs, s), "diagnostic xor inverse failed")
        # The residual is modulo 256; unwrapped signed deltas are used in the reported histogram/local codec.
        d = q - r
        require(np.array_equal(r + d, q), "diagnostic modulo-delta inverse failed")
        return dict(xor=self.compressed_size(x.tobytes() + sx_raw),
                    xor_bitplanes=self.compressed_size(packed.tobytes() + sx_raw),
                    delta_mod256=self.compressed_size(d.tobytes() + sx_raw))


def statistical(samples):
    stats, diagnostic = {}, {}
    z = Diagnostic()
    for v, planes in samples.items():
        opts = ("previous_token", *REFS.get(v, ()))
        for side, q, s in (("K", *planes[:2]), ("V", *planes[2:])):
            diag = {"independent": z.size(q, s)}
            for ref in opts:
                if ref == "previous_token":
                    # Diagnostic includes the first rows relative to zeros; stats omit artificial references.
                    r, rs = [np.concatenate((np.zeros_like(p[:, :1]), p[:, :-1]), axis=1) for p in (q, s)]
                    diag[ref] = z.size(q, s, r, rs)
                    stats[f"{v}<-token:{side}"] = plane_stats(q[:, 1:], s[:, 1:], q[:, :-1], s[:, :-1])
                else:
                    i = 0 if side == "K" else 2
                    r, rs = samples[ref][i:i + 2]
                    diag[ref] = z.size(q, s, r, rs)
                    stats[f"{v}<-{ref}:{side}"] = plane_stats(q, s, r, rs)
            diagnostic[f"{v}:{side}"] = diag
    return stats, diagnostic


def practical(samples, meta, grain):
    count = len(meta["tile_starts"])
    payloads = {arm: Counter() for arm in ARMS}
    tile_modes = {arm: Counter() for arm in ARMS}
    attempted = 0
    for v, planes in samples.items():
        for side in (0, 2):
            for tile in range(count):
                for head in range(HEADS):
                    q, s = (p[tile, :, head, :, :] for p in planes[side:side + 2])
                    references = {2 + i: tuple(p[tile, :, head, :, :] for p in samples[r][side:side + 2])
                                  for i, r in enumerate(REFS.get(v, ()))}
                    candidates = {}
                    for tag in (0, 1, *references):
                        encoded = codec.encode(q, s, tag, references.get(tag))
                        cq, cs = codec.decode(encoded, references)
                        require(np.array_equal(q, cq) and np.array_equal(s, cs),
                                f"roundtrip differs at token {meta['tile_starts'][tile]}, visit {v}, side {side}, head {head}, tag {tag}")
                        candidates[tag] = encoded
                        attempted += 1
                    for arm, options in ARMS.items():
                        if arm.startswith("cross_") and not references:
                            continue  # anchors and non-core visits still pay their entire baseline footprint
                        options = tuple(max(references) if t == -1 else t for t in options)
                        chosen = min((candidates[t] for t in options if t in candidates), key=len)
                        payloads[arm][v] += len(chosen)
                        tile_modes[arm][{0: "independent", 1: "previous_token", 2: "pass1", 3: "pass2", 255: "raw"}[chosen[0]]] += 1

    rows = meta["prefix"]
    n_tiles = rows // TILE
    raw = rows * len(VISITS) * VISIT_ROW
    raw_mapped = len(VISITS) * sum(codec.aligned(rows * n, grain) for n in STRIDES)
    comparisons = {}
    for arm in ARMS:
        n_visits = len(payloads[arm])
        unencoded = len(VISITS) - n_visits
        projected = (sum(payloads[arm].values()) * n_tiles + count - 1) // count
        offset_width = 4 if projected < 2**32 else 8
        # One aligned append arena, one offset table (absolute byte offset from arena base per tile/side/head),
        # 32 bytes of shared layout metadata in that allocation, one tile of live raw staging per encoded visit.
        offsets = n_tiles * n_visits * 2 * HEADS * offset_width
        overhead = 32 + offsets + n_visits * TILE * VISIT_ROW
        anchor_mapped = unencoded * sum(codec.aligned(rows * n, grain) for n in STRIDES)
        mapped = anchor_mapped + codec.aligned(projected, grain) + codec.aligned(overhead, grain)
        # Same grouping as the candidate, with fixed-size indexed-by-multiplication raw rows in the arena.
        matched_raw_mapped = anchor_mapped + codec.aligned(n_visits * rows * VISIT_ROW, grain)
        total = unencoded * rows * VISIT_ROW + projected + overhead
        comparisons[arm] = dict(projected_total_bytes=total, projected_saving=rounded(1 - total / raw),
                                projected_mapped_bytes=mapped,
                                matched_raw_mapped_bytes=matched_raw_mapped,
                                projected_mapped_saving_vs_split_raw=rounded(1 - mapped / raw_mapped),
                                projected_mapped_saving_vs_matched_raw=rounded(1 - mapped / matched_raw_mapped),
                                passes_payload_gate=total * 5 <= raw * 4,
                                passes_matched_mapped_gate=mapped * 5 <= matched_raw_mapped * 4,
                                aligned_tile_bytes_on_sample_by_encoded_visit=dict(payloads[arm]),
                                sampled_modes=dict(tile_modes[arm]), projected_tile_arena_bytes=projected,
                                projected_offset_bytes=offsets, active_raw_bytes=n_visits * TILE * VISIT_ROW,
                                projected_unencoded_visits_bytes=unencoded * rows * VISIT_ROW)
    return dict(roundtrip_head_tiles=attempted, roundtrip=True, raw_bytes=raw,
                raw_mapped_split_bytes=raw_mapped,
                mapping_grain_assumed_bytes=grain, projected_from_sample=True, target_fraction=GATE,
                per_head_tile_rows=TILE, per_tile_alignment_bytes=codec.ALIGN, comparisons=comparisons)


def diagnostic_summary(metrics, count):
    denom = count * TILE * VISIT_ROW * len(VISITS)
    independent = adjacent = cross = cross_anchor_raw = hybrid = 0
    for key, diag in metrics.items():
        v = int(key.split(":")[0])
        raw = count * TILE * (VISIT_ROW // 2)
        i = min(raw, diag["independent"]["raw"])
        t = min(i, *diag["previous_token"].values())
        group = [min(diag[r].values()) for r in REFS.get(v, ())]
        c = min([i, *group])
        independent += i
        adjacent += t
        cross += c
        cross_anchor_raw += c if group else raw
        hybrid += min([t, *group])
    return {name: rounded(1 - n / denom) for name, n in (("independent", independent), ("token_adjacent", adjacent),
            ("cross_adaptive_anchors_raw", cross_anchor_raw), ("cross_adaptive_anchors_zstd", cross), ("hybrid", hybrid))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--workload", required=True, help="provenance label (e.g. code or mixed)")
    parser.add_argument("--prefix", type=int, help="only sample this many committed rows of the snapshot")
    parser.add_argument("--tiles", type=int, default=12, choices=range(2, 33), metavar="[2..32]")
    parser.add_argument("--mapping-grain", type=int, default=2 << 20,
                        help="assumed VRAM allocation grain for the footprint model (default: 2 MiB)")
    a = parser.parse_args()
    require(a.mapping_grain > 0 and (a.mapping_grain & (a.mapping_grain - 1)) == 0, "mapping grain must be a power of 2")
    meta, samples = sample(a.snapshot, a.checker.resolve(), a.prefix, a.tiles)
    stats, diag = statistical(samples)
    layout = practical(samples, meta, a.mapping_grain)
    print(json.dumps(dict(schema=1, workload=a.workload, snapshot=meta, physical_visit=PHYSICAL, core_pass=PASS,
                          practical=layout, bit_statistics=stats, diagnostic_zstd_level9=diag,
                          diagnostic_sample_saving=diagnostic_summary(diag, len(meta["tile_starts"]))),
                     indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
