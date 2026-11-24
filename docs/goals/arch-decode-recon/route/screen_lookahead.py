#!/usr/bin/env python3
"""Offline FIFO deadline screen for same-token previous-pass full-expert transfers."""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_routes import LAYERS, check, load, ratio, sha256, summary


def transfer_costs(log):
    rows = defaultdict(list)
    with log.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mode") in {"pinned", "pack"} and row.get("bytes"):
                rows[row["mode"], int(row["bytes"])].append(float(row["pack_us"]) + float(row["complete_after_submit_start_us"]))
    check(rows and {k[0] for k in rows} == {"pinned", "pack"}, "missing measured pinned/pack costs")
    return rows


def proposals(t, cap):
    # Use only pass 1→2 and 2→3: 1→3 would compete for the same targets as 2→3.
    check(np.all(t.ready_ns >= 0) and np.all(t.resident >= 0) and np.all(t.cpu >= 0), "missing CP-02 trace labels")
    sizes = t.metadata["expert_bytes"]
    total_miss = int(t.cpu[:, 18:42].sum())
    groups = {}
    jobs = []
    for i in range(len(t.pos)):
        for a in range(2):
            for l in LAYERS:
                va = l + 12 * a
                vb = va + 12
                src = int(t.ready_ns[i, va])
                due = int(t.ready_ns[i, vb])
                check(src <= due, "disagreeing source and target host clocks")
                miss = set(t.ids[i, vb][t.cpu[i, vb] == 1])
                picks = [e for e in t.ids[i, va] if t.resident[i, va, e] == 0 and t.pending[i, va, e] != 1][:cap]
                groups[i, vb] = int(t.timing["gpu_to_cpu_ready_marker_ns"][i, vb])
                for e in picks:
                    jobs.append((src, due, sizes[l], (i, vb), e in miss))
    # Stable order preserves router rank when several requests share the issue timestamp.
    return sorted(jobs, key=lambda r: r[0]), groups, total_miss


def fifo(jobs, groups, total_miss, duration_ns, tokens):
    end = 0
    on_time, on_time_useful, useful, payload, false_positive, slot_events = 0, 0, 0, 0, 0, []
    touched = set()
    queue_delays = []
    for issue, deadline, nbytes, target, hit in jobs:
        start = max(end, issue)
        end = start + duration_ns[nbytes]
        ready = end <= deadline
        queue_delays.append((start - issue) / 1000)
        payload += nbytes
        false_positive += nbytes * (not hit)
        useful += hit
        on_time += ready
        on_time_useful += ready and hit
        if ready and hit:
            touched.add(target)
        # Slot occupied from transfer start through arrival or target consumption, whichever is later.
        slot_events.extend([(start, nbytes), (max(end, deadline), -nbytes)])
    busy = peak = 0
    for _, delta in sorted(slot_events):
        busy += delta
        peak = max(peak, busy)
    known = [ns for ns in groups.values() if ns >= 0]
    touch_ns = sum(groups[g] for g in touched if groups[g] >= 0)
    return dict(tokens=tokens, target_cpu_misses=total_miss, issued=len(jobs), useful=useful, on_time=on_time,
                on_time_useful=on_time_useful, on_time_useful_rate=ratio(on_time_useful, len(jobs)),
                on_time_useful_miss_recall=ratio(on_time_useful, total_miss), queue_delay_us=summary(queue_delays),
                payload_bytes_per_token=payload / tokens, false_positive_bytes_per_token=false_positive / tokens,
                peak_independent_staging_payload_bytes=peak, measured_gap_visits=len(known),
                mean_traced_target_marker_us_per_token=sum(known) / 1000 / tokens,
                mean_marker_gap_on_visits_with_on_time_useful_pick_us_per_token=touch_ns / 1000 / tokens)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("transfer_log", type=Path)
    p.add_argument("--first-test-row", type=int, help="default skips floor(0.65 * tokens) training rows and first test row")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    t = load(args.trace)
    n = len(t.pos)
    start = args.first_test_row if args.first_test_row is not None else n * 65 // 100 + 1
    check(0 <= start < n, "no deadline-test positions")
    costs = transfer_costs(args.transfer_log)
    arms = {}
    for cap in (1, 2, 4):
        jobs, groups, miss = proposals(t[start:], cap)
        for mode in ("pinned", "pack"):
            for pctl in (50, 90):
                durations = {nbytes: round(np.percentile(measured, pctl) * 1000)
                             for (kind, nbytes), measured in costs.items() if kind == mode}
                fit = fifo(jobs, groups, miss, durations, n - start)
                arms[f"{mode}_p{pctl}_cap{cap}"] = dict(duration_ns_by_bytes=durations, **fit)
    result = dict(schema="inkling-route-lookahead-v1", trace_sha256=sha256(args.trace), cost_input_sha256=sha256(args.transfer_log),
                  first_position=int(t.pos[start]), last_position=int(t.pos[-1]),
                  assumptions="Static observed timestamps, serialized pack+transfer, no interference or baseline swap contention, infinite independent slots. Touching marker gap is not saving it or changing backend exactly.",
                  isolated_transfer_us={f"{mode}/{size}": summary(times) for (mode, size), times in costs.items()}, arms=arms)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
