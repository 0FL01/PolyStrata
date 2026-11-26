#!/usr/bin/env python3
"""Capture real, single-anchor CPU intermediate states and validate replay batching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dflash2_ref import (DEFAULT_BRIDGE, DEFAULT_DFLASH2_DIR, DEFAULT_RECON_DIR,
                         DEFAULT_TARGET_GGUF, DFlash2Model, DFlash2Weights,
                         QBridge, TargetSharedWeights)
from replay_matrix import DOMAINS, DRAFT_WIDTH, load_domain


def capture(model: DFlash2Model, tr, ids: np.ndarray, start: int) -> dict:
    model.reset_cache()
    ctx = model.inject(tr.taps[:start], tr.pos[:start])
    out = model.draft_block(int(ids[start]), start, return_intermediates=True)
    return {**out.pop("intermediates"), **out, "ctx_hidden": ctx,
            "anchor": np.int32(ids[start]), "truth": ids[start + 1:start + 1 + DRAFT_WIDTH]}


def parity(ref: dict, test: dict) -> dict:
    if set(ref) != set(test):
        raise ValueError("different intermediate-state keys")
    return {k: {"shape": list(a.shape), "dtype": str(a.dtype),
                "bit_identical": a.dtype == test[k].dtype and a.tobytes() == test[k].tobytes(),
                "max_abs": float(np.max(np.abs(np.asarray(a, dtype=np.float64) - test[k]), initial=0))}
            for k, a in ref.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taps", type=Path, default=DEFAULT_RECON_DIR / "taps")
    p.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET_GGUF)
    p.add_argument("--draft-dir", type=Path, default=DEFAULT_DFLASH2_DIR)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--domains", nargs="+", choices=DOMAINS, default=["code_python"])
    p.add_argument("--offsets", nargs="+", type=int, default=[0, 8])
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bridge = QBridge(args.bridge, args.threads)
    target = TargetSharedWeights(args.target, bridge)
    inputs = {d: load_domain(args.taps, d) for d in args.domains}
    report = {}

    models = {"source": args.draft_dir / "source/model.safetensors",
              "BF16": args.draft_dir / "Inkling-2-Mini-DFlash2-BF16.gguf"}
    for label, path in models.items():
        model = DFlash2Model(DFlash2Weights(path, bridge), target)
        for domain, (tr, ids, prompt) in inputs.items():
            for pos in (prompt + d for d in args.offsets):
                if pos < prompt or pos + DRAFT_WIDTH >= len(ids):
                    raise ValueError(f"{domain}: position {pos} is not a full generated anchor")
                state = capture(model, tr, ids, pos)
                np.savez_compressed(args.output / f"{domain}-{pos}-{label}.npz", **state)
                if label == "BF16":
                    with np.load(args.output / f"{domain}-{pos}-source.npz") as src:
                        row = parity({k: src[k] for k in src.files}, state)
                        report.setdefault(domain, {})[str(pos)] = {"source_vs_bf16": row}
        del model

    # Independent path check. The matrix path uses batched context projection,
    # block attention with future *true* rows excluded, and a dense FP32 head.
    # Use the original single-anchor/Q6_K implementation as a reference.
    model = DFlash2Model(DFlash2Weights(models["BF16"], bridge), target)
    for domain, (tr, ids, prompt) in inputs.items():
        starts = np.array([prompt + d for d in args.offsets], dtype=np.int64)
        model.reset_cache()
        hi = int(starts.max())
        model.inject(tr.taps[:hi], tr.pos[:hi])
        target.lm_head_mode = "blas"
        outs = model.draft_batch(ids[starts], starts)
        target.lm_head_mode = "f32"
        for pos, out in zip(starts, outs, strict=True):
            with np.load(args.output / f"{domain}-{pos}-BF16.npz") as single:
                report[domain][str(pos)]["batch_vs_single"] = {
                    "draft_match": int(np.count_nonzero(single["draft_tokens"] == out["draft_tokens"])),
                    "candidate_ids_match": int(np.count_nonzero(single["cand"] == out["cand"])),
                    "hidden": bridge.eval_error(single["h_draft"], out["h_draft"]),
                }

        # Independent tap-48 check: the active target head predicts the
        # next target token, not the token on the same record. Compare earlier
        # taps as controls. Close ties need not match HIP's accumulation.
        positions = np.linspace(prompt - 1, min(len(tr.pos) - 1, len(ids) - 2), 12, dtype=np.int64)
        target.lm_head_mode = "blas"
        checks = []
        for i in range(4):
            logits = target.lm_head(tr.taps[positions, i])
            pred = np.argmax(logits, axis=-1)
            top2 = np.partition(logits, -2, axis=-1)[:, -2:]
            checks.append({"visit": [5, 23, 41, 48][i],
                           "positions": positions.tolist(), "next_labels": ids[positions+1].tolist(),
                           "argmax": pred.tolist(), "next_match": int((pred == ids[positions+1]).sum()),
                           "same_row_match": int((pred == ids[positions]).sum()),
                           "best_minus_runner_up": (top2[:, 1] - top2[:, 0]).tolist()})
        target.lm_head_mode = "f32"
        report[domain]["target_head_tap_alignment"] = checks
    (args.output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(not state["bit_identical"] for rows in report.values() for pos, row in rows.items()
           if pos != "target_head_tap_alignment" for state in row["source_vs_bf16"].values()):
        raise SystemExit("BF16 GGUF vs source capture differs")


if __name__ == "__main__":
    main()
