#!/usr/bin/env python3
"""Read the two preserved 2026-11-21 NVFP4 fixed-geometry ROCm decode traces (not a generic rocprof parser)."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_routes import K, N_VISITS, check, critical_path, miss_counts, sha256, summary


def extract(csv_file, card_route):
    with csv_file.open(newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["Start_Timestamp"]))

    def name_is(row, key):
        return key + "(" in row["Kernel_Name"]

    def start(row):
        return int(row["Start_Timestamp"])

    def end(row):
        return int(row["End_Timestamp"])

    def router(row):
        return (name_is(row, "copy2_kernel") and row["Grid_Size_X"] == "2560"
                or name_is(row, "route_one_kernel"))

    heads = [i for i, r in enumerate(rows) if "native_q6_k_warps_kernel<" in r["Kernel_Name"]]
    check(len(heads) == 24, "historical fixture must have exactly 24 head markers")
    traced, busy, tail_gaps, counts = [], [], [], []
    expected_card = {6, 7, 9, 13, 18, 19, 21, 25, 30, 31, 33, 37} if card_route else set()
    for i, j in zip(heads, heads[1:]):
        window = rows[i + 1:j + 1]
        markers = [s for s, row in enumerate(window) if router(row)]
        check(len(markers) == N_VISITS, f"head {i}: {len(markers)} router markers, expected 48")
        check({v for v, s in enumerate(markers) if name_is(window[s], "route_one_kernel")} == expected_card,
              f"head {i}: card route visit layout changed")
        check(len({row["Stream_Id"] for row in window}) == 1, f"head {i}: heterogeneous kernel streams")
        cpu, gaps = [], []
        for v, (s, z) in enumerate(zip(markers, markers[1:] + [len(window)])):
            between = window[s + 1:z]
            conv = [k for k, row in enumerate(between) if name_is(row, "conv_add_kernel")]
            check(len(conv) == (2 if v != N_VISITS - 1 else 1), f"head {i}, visit {v}: no disjoint tail marker")
            expert = between[:conv[0]]
            situ = [r for r in expert if name_is(r, "situ_rows_kernel")]
            if expert:
                finish = "moe_sum_kernel" if v in expected_card else "add_scaled_kernel"
                check(len(situ) == 1 and name_is(expert[-1], finish), f"head {i}, visit {v}: unexpected expert span")
                gpu = int(situ[0]["Grid_Size_Y"])
                check(1 <= gpu <= K and (v not in expected_card or gpu == K), "invalid cached expert count")
            else:
                check(v not in expected_card, "no GPU experts on card-routing visit")
                gpu = 0
            cpu.append(K - gpu)
            gap = start(between[conv[0]]) - end(expert[-1] if expert else window[s])
            check(gap >= 0, "overlapping historical marker interval")
            gaps.append(gap)
        counts.append(cpu)
        tail_gaps.append(gaps)
        t0 = end(rows[i])
        last = t0
        total = 0
        for row in window:
            total += max(0, end(row) - max(start(row), last))
            last = max(last, end(row))
        traced.append(end(rows[j]) - t0)
        busy.append(total)
    return np.array(counts), np.array(tail_gaps), np.array(traced), np.array(busy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact_dir", type=Path, help="original dspark-speed-1795285637 artifact directory")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    runs = []
    for leg in (0, 1):
        card_route = bool(leg)
        path = args.artifact_dir / f"trace-{leg}/trace-TG_kernel_trace.csv"
        command = args.artifact_dir / f"moe-tr{leg}/command.log"
        check("Inkling-2-Mini-UD-IQ4_XS.gguf" in command.read_text(), "unexpected historical model")
        cpu, gaps, traced, busy = extract(path, card_route)
        exposed_envelope = np.where(cpu > 0, gaps, 0)
        runs.append(dict(csv=str(path), csv_sha256=sha256(path), command_sha256=sha256(command),
                         card_routes_for_pinned_layers=card_route, matched_tokens=len(traced), visits=int(cpu.size),
                         cpu_miss_fraction=float(cpu.mean() / K), miss_count_histogram=np.bincount(cpu.ravel(), minlength=K+1).tolist(),
                         token_head_to_head_us=summary(traced / 1000), token_kernel_union_us=summary(busy / 1000),
                         token_kernel_only_gap_us=summary((traced - busy) / 1000),
                         token_cpu_positive_tail_gap_envelope_us=summary(exposed_envelope.sum(axis=1) / 1000),
                         token_core2_3_cpu_positive_tail_gap_envelope_us=summary(exposed_envelope[:, 18:42].sum(axis=1) / 1000),
                         timing=critical_path(cpu, {"gpu_to_tail_gap_ns": gaps})["gpu_to_tail_gap_ns"],
                         miss_count_correlation=miss_counts(cpu)))
    result = dict(schema="inkling-historical-rocm-p0-v1", model_basename="Inkling-2-Mini-UD-IQ4_XS.gguf",
                  path_kind="legacy RX 6700 XT forced 8-token prompt + 24 new, pinned four layers, live swaps; correlated legs",
                  data_limits="No expert IDs, weight bits, CPU timestamps, or reliable attributable CPU-join cost; head spans and tail gaps are traced envelopes, not E2E estimates.",
                  runs=runs)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
