#!/usr/bin/env python3
"""Compare the detailed native replay traces at the same prerecorded anchor positions."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(base: list, trial: list) -> dict:
    if not base or len(base) != len(trial):
        raise ValueError("expected the same nonempty all-position replay slice")
    stats = dict(steps=len(base), same_proposals=0, total_proposals=0, same_accepted=0,
                 first_proposal_diff=None, same_candidates=True)
    floats = {name: 0. for name in ("unary", "sel_scores", "sel_probs", "h_draft") if name in base[0]}
    for a, b in zip(base, trial, strict=True):
        if any(a[k] != b[k] for k in ("start", "anchor", "mode")):
            raise ValueError("replay is not aligned")
        stats["same_accepted"] += a["accepted"] == b["accepted"]
        stats["same_candidates"] &= a["cand"] == b["cand"]
        for i, (x, y) in enumerate(zip(a["draft_tokens"], b["draft_tokens"], strict=True)):
            stats["total_proposals"] += 1
            stats["same_proposals"] += x == y
            if x != y and stats["first_proposal_diff"] is None:
                stats["first_proposal_diff"] = [a["start"], i, x, y]
        for k in floats:
            err = np.abs(np.asarray(a[k], np.float32).astype(np.float64) - np.asarray(b[k], np.float32))
            if not np.isfinite(err).all():
                raise ValueError(f"nonfinite {k} at {a['start']}")
            floats[k] = max(floats[k], float(np.max(err)))
    return dict(stats, max_abs_error=floats)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("trial", nargs="+", type=Path)
    a = p.parse_args()

    def load(path: Path) -> list:
        return [json.loads(line) for line in path.read_text().splitlines()]

    base = load(a.base)
    print(json.dumps({str(trial): compare(base, load(trial)) for trial in a.trial}, indent=2))


if __name__ == "__main__":
    main()
