#!/usr/bin/env python3
"""Frozen five-pair replay statistics; requests within a pass are not independent pairs."""
import argparse
import json
import math
from pathlib import Path
import statistics


def interval(ratios):
    logs = list(map(math.log, ratios))
    mean = statistics.mean(logs)
    radius = 2.776445105 * statistics.stdev(logs) / math.sqrt(5)  # t(4), two-sided 95%
    estimate, low, high = map(math.exp, (mean, mean - radius, mean + radius))
    precise = max(high / estimate - 1, 1 - low / estimate) <= 0.05
    return {"ratios": ratios, "geomean": estimate, "low": low, "high": high,
            "precise": precise, "decision": "insufficient data" if not precise else
            "confirmed gain" if low > 1 else "confirmed regression" if high < 1 else "not confirmed"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--pairs", nargs=5, required=True, help="baseline-id:candidate-id in pair order")
    ap.add_argument("--aa", required=True, help="one baseline-id:baseline-id control")
    ap.add_argument("--forced", action="store_true", help="read strict run_forced.py result records")
    a = ap.parse_args()

    def load(pair):
        ids = pair.split(":")
        if a.forced:
            records = [json.loads((a.root / name / "result.json").read_text()) for name in ids]
            if not all(record["valid"] for record in records):
                raise ValueError("invalid forced run")
            return [[record["metrics"]] for record in records]
        return [json.loads((a.root / name / "metrics.json").read_text()) for name in ids]

    pairs = [load(pair) for pair in a.pairs]
    count = 1 if a.forced else 3
    tg_by_request = [[b[i]["tg"] / base[i]["tg"] for base, b in pairs] for i in range(count)]
    tg = [math.prod(b[i]["tg"] / base[i]["tg"] for i in range(count)) ** (1 / count) for base, b in pairs]
    pp = [b[0]["pp"] / base[0]["pp"] for base, b in pairs]
    result = {"pairs": a.pairs, "aa": a.aa, "tg": interval(tg), "full_prefill": interval(pp),
              "tg_per_request": [interval(r) for r in tg_by_request],
              "prefill_time_ratio": interval([1 / r for r in pp]),
              "decode_time_ratio": interval([1 / r for r in tg])}
    # The gain/regression labels above are for throughput; time ratios are used only
    # for their upper-bound <=1.02 budget, where smaller is better.
    for metric in ("prefill_time_ratio", "decode_time_ratio"):
        result[metric].pop("decision")
    base, b = load(a.aa)
    result["aa_ratios"] = {"full_prefill": b[0]["pp"] / base[0]["pp"],
                            "tg": [b[i]["tg"] / base[i]["tg"] for i in range(count)]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
