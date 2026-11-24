#!/usr/bin/env python3
"""Normalize a CP-02 PSVST1 capture restricted to a declared serial-decode interval."""

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

from analyze_routes import check, sha256, stage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trace"))
from collect import header, records


def mask(value):
    m = int(value, 16)
    return [i for i in range(128) if m & (1 << i)]


def normalize(row):
    check(not row["windowed"], f"position {row['pos']}: wanted plain eval, not window()")
    check("skipped" not in row["placement"], f"position {row['pos']}: skipped expert violates exactness contract")
    ms_to_ns = lambda key: round(row[key] * 1e6)
    return dict(pos=row["pos"], visit=row["visit"], layer=row["layer"],
                ids=row["ids"], weight_bits=row["weight_bits"], cpu=[p == "cpu" for p in row["placement"]],
                resident_ids=mask(row["resident_mask"]),
                pending_ids=mask(f"{int(row['asked_mask'], 16) | int(row['sent_mask'], 16):x}"),
                route_ready_ns=row["route_ready_ns"], router_wait_ns=row["router_wait_ns"],
                visit_host_enqueue_ns=row["host_end_ns"] - row["host_begin_ns"],
                cpu_branch_ns=row["cpu_ready_ns"] - row["cpu_begin_ns"],
                gate_up_host_ns=row["gate_end_ns"] - row["gate_begin_ns"],
                down_host_ns=row["down_end_ns"] - row["down_begin_ns"],
                visit_ns=ms_to_ns("gpu_visit_ms"), gpu_expert_ns=ms_to_ns("gpu_experts_ms"),
                gpu_to_cpu_ready_marker_ns=ms_to_ns("gpu_join_ms"), gpu_dispatch_gap_ns=ms_to_ns("gpu_dispatch_ms"),
                **{"pass": stage(row["visit"])})


def run(path, meta, start, end, output):
    cpu_timing = "PSVST1 events and host clocks; GPU marker gaps include enqueue/marker overhead, not isolated join"
    check(meta.get("model_sha256") and meta.get("workload") and meta.get("trajectory_id"),
          "metadata must pin model_sha256, workload, and trajectory_id")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as source:
        h = header(source)
        check(h["first"] <= start < end <= h["end"], "decode slice is outside PSVST1 bounds")
        out = []
        events = Counter()
        frozen = set()
        for row in records(source, h):
            if start <= row["pos"] < end:
                if row["kind"] == "visit":
                    x = normalize(row)
                    check(x["pos"] == start + len(out) // 48 and x["visit"] == len(out) % 48,
                          "incomplete or repeated serial-decode position")
                    out.append(x)
                    frozen.add(row["frozen"])
                else:
                    events[row["kind"]] += 1
    check(len(out) == 48 * (end - start), "requested decode slice was not fully captured")
    check(len(frozen) == 1, "frozen/live placement mixed within decode slice")
    with output.open("x") as f:
        h_out = dict(meta, schema="inkling-route-v1", phase="serial_decode", input=str(path), input_sha256=sha256(path),
                     input_header=h, capture_event_counts=events, frozen=frozen.pop(), decode_slice=[start, end],
                     timing_basis=cpu_timing)
        print(json.dumps(h_out, separators=(",", ":")), file=f)
        for row in out:
            print(json.dumps(row, separators=(",", ":")), file=f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--metadata", type=Path, required=True, help="JSON provenance for actual model, binary and workload")
    p.add_argument("--from-pos", type=int, required=True, help="absolute inclusive decode token position")
    p.add_argument("--to-pos", type=int, required=True, help="absolute exclusive decode token position")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.trace, json.loads(args.metadata.read_text()), args.from_pos, args.to_pos, args.output)


if __name__ == "__main__":
    main()
