#!/usr/bin/env python3
"""Keep compact, rerunnable summaries; detailed fits and all 3,072 slots stay in scratch."""

import argparse
import json
from pathlib import Path

import numpy as np


RUNS = ("trace-on", "trace-code-2048", "trace-mixed-65536")


def brief(result):
    out = {key: result[key] for key in (
        "trace", "capacity", "capacity_bytes", "min_observed_swap_queue_size", "heldout_calibration",
        "bootstrap", "train_cost_static_install", "heldout_relaxed_any_miss_per_visit_bound_branch_ms",
        "train_static_start", "train_local_swaps", "untraced_baseline_ms_token")}
    out["policies"] = {}
    for name, policy in result["heldout"].items():
        joint = policy["same_layer_cross_pass_joint_miss_histograms"]
        keys = ("hit_rate", "miss_histogram", "exact_miss_count_agreement_with_observed", "saved_ms_token",
                "net_ms_token", "net_interval", "H2D_service_ms", "H2D_bytes", "predicted_tokens_s_if_additive")
        out["policies"][name] = {key: policy[key] for key in keys if key in policy}
        out["policies"][name]["same_layer_both_visits_3plus_misses"] = {
            key: sum(sum(row[3:]) for row in matrix[3:]) for key, matrix in joint.items()}
    return out


def rankings(utility):
    out = {}
    for status, label in ((0, "promotion"), (1, "resident_removal_loss")):
        rows = [row for row in utility["rows"] if row[3] == status]
        tr = sorted(rows, key=lambda r: -sum(r[4]))
        te = sorted(rows, key=lambda r: -sum(r[5]))

        def top32(seq):
            return {tuple(row[:2]) for row in seq[:32]}

        corr = np.corrcoef([sum(r[4]) for r in rows], [sum(r[5]) for r in rows])[0, 1]
        out[label] = dict(top32_overlap=len(top32(tr) & top32(te)), pearson_across_all=float(corr),
                          top3_in_training=tr[:3], top3_in_heldout=te[:3])
    return out


def compact(value):
    if isinstance(value, float):
        return round(value, 7)
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


def formatted(obj, depth=0):
    if not isinstance(obj, dict) or depth >= 3:
        return json.dumps(compact(obj), allow_nan=False)
    prefix = "  " * depth
    entries = [prefix + "  " + json.dumps(k) + ": " + formatted(v, depth + 1) for k, v in obj.items()]
    return "{\n" + ",\n".join(entries) + "\n" + prefix + "}"


def write_object(path, obj):
    with path.open("x") as f:
        print(formatted(obj), file=f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path, help="replay output directory from reproduce.sh")
    p.add_argument("output", type=Path, help="new files in this directory")
    args = p.parse_args()
    result, slots = {}, {}
    transfer = seed = fields = stages = None
    for run in RUNS:
        r = json.loads((args.scratch / f"placement-{run}.json").read_text())
        u = json.loads((args.scratch / f"utility-{run}.json").read_text())
        transfer = dict(csv_sha256=r["transfer_csv_sha256"], timings_ms=r["transfer"])
        seed = r["profile_seed_sha256"]
        fields, stages = u["fields"], u["stages"]
        result[run] = brief(r)
        slots[run] = rankings(u)
    write_object(args.output / "placement-oracle.json", dict(transfer=transfer, profile_seed_sha256=seed, **result))
    write_object(args.output / "slot-utility-analysis.json", dict(fields=fields, stages=stages, **slots))


if __name__ == "__main__":
    main()
