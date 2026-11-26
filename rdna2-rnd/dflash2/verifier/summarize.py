#!/usr/bin/env python3
"""Read completed checkpoints or compare operation fingerprints for two trace arms."""
import argparse
import json
from pathlib import Path

from sweep import CONTEXTS, WIDTHS, tagged


def compare_trace(a, b):
    def read(path):
        rows = tagged(path.read_text(), "RECON_TRACE")
        if not rows:
            raise ValueError(f"missing traces: {path}")
        traces = {(int(pos), int(visit), stage): (int(size), hash_) for pos, visit, stage, size, hash_ in rows}
        if len(traces) != len(rows):
            raise ValueError(f"same input position was traced more than once (use a forced arm): {path}")
        return traces

    a, b = read(a), read(b)
    if list(a) != list(b):
        raise ValueError("trace selection or operation order differs")
    for key in a:
        if a[key] != b[key]:
            print(f"first tracked difference: input position {key[0]}, visit {key[1]}, {key[2]}: {a[key]} -> {b[key]}")
            return
    print(f"all {len(a)} traced operations match")


def summarize(series):
    print("| Context (actual) | Experts in VRAM | Mode | Fix | W | Tok/s | Full-window ms, p50 | "
          "Picks matching W1 | First difference (zero-based index; W1 → observed) | Drafts accepted / offered |")
    print("|---:|---:|:---|---:|---:|---:|---:|---:|:---|---:|")
    for ctx in CONTEXTS:
        results = {}
        for p in (series / ctx).glob("*/*/complete.json"):
            r = json.loads(p.read_text())["result"]
            key = (r["fix"], WIDTHS.index(r["width"]), ("serial", "forced", "accepted").index(r["mode"]))
            if key in results:
                raise ValueError(f"ambiguous checkpoints for {ctx}/{p.parent.parent.name}; report one series at a time")
            results[key] = r
        for _, r in sorted(results.items()):
            d = r.get("prediction_difference")
            diff = f"{d['index']}; {d['expected']} → {d['observed']}" if d else "—"
            n = len(r["tokens"])
            match = f"{r['picks_matching']}/{n}" if "picks_matching" in r else f"{n} reference IDs"
            p50 = r.get("full_window_p50_ms")
            checkpoint = "—" if p50 is None else f"{p50:.2f}"
            print(f"| {ctx} ({r['prompt_count']:,}) | {r['gpu_slots']} | {r['mode']} | {r['fix']} | {r['width']} | "
                  f"{r['wire_tps']:.2f} | {checkpoint} | {match} | {diff} | "
                  f"{r['draft_accepted']}/{r['draft_offered']} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("series", nargs="?", type=Path)
    ap.add_argument("--trace", nargs=2, type=Path, metavar="STDERR")
    a = ap.parse_args()
    if a.trace:
        compare_trace(*a.trace)
    elif a.series:
        summarize(a.series)
    else:
        ap.error("provide a series directory or two --trace stderr logs")


if __name__ == "__main__":
    main()
