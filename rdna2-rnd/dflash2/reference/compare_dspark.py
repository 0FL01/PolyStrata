#!/usr/bin/env python3
"""Score same-history forced DSpark proposals against saved CPU DFlash2 replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dflash2_ref import DEFAULT_RECON_DIR
from replay_matrix import (DOMAINS, DRAFT_WIDTH, PROFILES, acceptance, closed_loop_rows,
                           read_dspark)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taps", type=Path, default=DEFAULT_RECON_DIR / "taps")
    ap.add_argument("--paired-dir", type=Path, default=DEFAULT_RECON_DIR / "paired-dspark")
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES)
    a = ap.parse_args()
    manifest = json.loads((a.taps / "manifest.json").read_text())
    report = {"domains": {}, "pooled": {}}
    sums = {}

    for dom in DOMAINS:
        m = manifest[dom]
        folder = a.paired_dir / dom
        evidence = json.loads((folder / "result.json").read_text())
        if evidence["ids"] != m["tokens"] or evidence["mode"] != "dspark-log" or evidence["k"] != DRAFT_WIDTH:
            raise ValueError(f"{dom}: no complete forced DSpark trajectory with width {DRAFT_WIDTH}")
        ids = np.asarray(m["prompt_ids"] + m["tokens"], dtype=np.int32)
        prompt = len(m["prompt_ids"])
        starts = np.arange(prompt, len(ids) - DRAFT_WIDTH)
        truth = ids[starts[:, None] + np.arange(1, DRAFT_WIDTH + 1)]
        ds, alignment = read_dspark(folder / "proposals.log", ids, prompt, len(starts))
        # The production policy cannot save a call by evaluating the final
        # answer token. The log therefore shortens its width one row before
        # the independent DFlash2 matrix runs out of next-token labels.
        if alignment["first_anchor_mismatch"] is not None or not len(truth) - 1 <= len(ds) <= len(truth):
            raise ValueError(f"{dom}: {alignment}")
        row = report["domains"][dom] = {"alignment": alignment, "draft_model": evidence["weights"],
                                         "binary_sha256": evidence["binary_sha256"], "methods": {},
                                         "anchors_lost_to_tail_width": len(truth) - len(ds)}
        preds = {"DSpark": ds}
        for profile in a.profiles:
            with np.load(a.replay / f"{profile}-{dom}.npz") as saved:
                if not np.array_equal(saved["positions"], starts) or not np.array_equal(saved["truth"], truth):
                    raise ValueError(f"{profile}/{dom}: not the same anchor set and labels")
                preds[profile] = saved["draft_tokens"][:len(ds)]
        truth = truth[:len(ds)]

        for name, pred in preds.items():
            loop = closed_loop_rows(pred, truth)
            stats = row["methods"][name] = {"all_positions": acceptance(pred, truth),
                                             "closed_loop": acceptance(pred[loop], truth[loop])}
            for mode, values in stats.items():
                sums.setdefault((name, mode), []).append(values)

    for (name, mode), per_domain in sums.items():
        n = sum(row["anchors"] for row in per_domain)
        k = sum(row["sum_accepted"] for row in per_domain)
        means = np.array([row["emitted_per_verify"] for row in per_domain])
        report["pooled"].setdefault(name, {})[mode] = {
            "anchors": n, "emitted_per_verify": 1 + k/n,
            "equal_domain_emitted_per_verify": float(means.mean()),
            "min_max_domain_emitted_per_verify": [float(means.min()), float(means.max())],
        }
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["pooled"], indent=2))


if __name__ == "__main__":
    main()
