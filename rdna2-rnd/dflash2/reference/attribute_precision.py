#!/usr/bin/env python3
"""Paired, single-family precision interventions on fixed real DFT1 anchors."""
from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path
import re
import time

import numpy as np

from dflash2_ref import (DEFAULT_BRIDGE, DEFAULT_DFLASH2_DIR, DEFAULT_RECON_DIR,
                         DEFAULT_TARGET_GGUF, DFlash2Model, DFlash2Weights, QBridge,
                         TargetSharedWeights, T)
from replay_matrix import (DOMAINS, DRAFT_WIDTH, acceptance, compare_selectors,
                           load_domain, prefix_len, run_domain)

FAMILIES = {
    "fc": r"^fc\.weight$",
    "conv_proj": r"_conv_proj\.weight$",
    "attn_qkv": r"\.attn_[qkv]\.weight$",
    "attn_out": r"\.attn_output\.weight$",
    "ffn_gate_up": r"\.ffn_(?:gate|up)\.weight$",
    "ffn_down": r"\.ffn_down\.weight$",
    "selector_hidden": r"^selector_hidden\.weight$",
    "selector_predecessor": r"^selector_predecessor\.weight$",
    "selector_successor": r"^selector_successor\.weight$",
    "selector_both_codebooks": r"^selector_(?:predecessor|successor)\.weight$",
}


def swap(base: DFlash2Weights, donor: DFlash2Weights, pattern: str, qtype: T | None = None) -> DFlash2Weights:
    """Replace a family without requantizing its already-low-precision base."""
    out = copy(base)
    out.tensors = base.tensors.copy()
    out.tensor_qtypes = base.tensor_qtypes.copy()
    names = [n for n in base.tensor_qtypes if re.search(pattern, n)]
    if not names:
        raise ValueError(f"no tensors for {pattern}")
    for name in names:
        if name in ("selector_predecessor.weight", "selector_successor.weight"):
            if qtype is not None:
                raise ValueError("specify a ready codebook donor")
            attr = "pred_cb" if "predecessor" in name else "succ_cb"
            setattr(out, attr, getattr(donor, attr))
        else:
            arr = donor.tensors[name]
            out.tensors[name] = arr if qtype is None else base.bridge.roundtrip(arr, int(qtype))
        out.tensor_qtypes[name] = donor.tensor_qtypes[name] if qtype is None else qtype.name
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taps", type=Path, default=DEFAULT_RECON_DIR / "taps")
    p.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET_GGUF)
    p.add_argument("--draft-dir", type=Path, default=DEFAULT_DFLASH2_DIR)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--anchors", type=int, default=12, help="equally spaced per domain")
    p.add_argument("--groups", nargs="+", choices=["kv", "q4_0", "q4_k", "codebooks"],
                   default=["kv", "q4_0", "q4_k", "codebooks"])
    a = p.parse_args()
    if a.anchors < 1:
        p.error("--anchors must be positive")
    a.output.mkdir(parents=True, exist_ok=True)
    bridge = QBridge(a.bridge, a.threads)
    target = TargetSharedWeights(a.target, bridge, "blas")
    data = {}
    report = {"positions": {}, "arms": {}, "pooled": {}}
    baselines = {}

    for domain in a.domains:
        tr, ids, prompt = load_domain(a.taps, domain)
        last = len(ids) - DRAFT_WIDTH - 1
        starts = np.linspace(prompt, last, min(a.anchors, last - prompt + 1), dtype=np.int64)
        truth = ids[starts[:, None] + np.arange(1, DRAFT_WIDTH + 1)]
        data[domain] = (tr, ids, starts, truth)
        report["positions"][domain] = starts.tolist()

    def load(profile: str) -> DFlash2Weights:
        return DFlash2Weights(a.draft_dir / f"Inkling-2-Mini-DFlash2-{profile}.gguf", bridge)

    def write_report():
        tmp = a.output / ".attribution.json.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(a.output / "attribution.json")

    def run(name: str, w: DFlash2Weights, kv: str = "fp32", base: str | None = None, store: bool = False):
        model = DFlash2Model(w, target, kv)
        arm = report["arms"][name] = {"kv": kv, "baseline": base, "domains": {}, "types": w.tensor_qtypes}
        if store:
            baselines[name] = {}
        for domain, (tr, ids, starts, truth) in data.items():
            t0 = time.perf_counter()
            pred = run_domain(model, tr, ids, starts, 32)
            d = pred["draft_tokens"]
            recall = (pred["cand"] == truth[..., None]).any(axis=-1)
            result = arm["domains"][domain] = {
                **acceptance(d, truth), "top16_recall": float(recall.mean()),
                "draft_tokens": d.tolist(), "truth": truth.tolist(),
            }
            if store:
                baselines[name][domain] = pred
            if "BF16" in baselines and name != "BF16":
                result["vs_bf16"] = compare_selectors(pred, baselines["BF16"][domain])
            if base:
                ref = baselines[base][domain]
                result["vs_base"] = compare_selectors(pred, ref)
                delta = prefix_len(d == truth) - prefix_len(ref["draft_tokens"] == truth)
                result["vs_base"].update(delta_sum_accepted=int(delta.sum()),
                                         longer_prefix=int((delta > 0).sum()),
                                         shorter_prefix=int((delta < 0).sum()))
            print(f"{name:31} {domain:13} {len(starts):2} anchors | {result['emitted_per_verify']:.4f} | {time.perf_counter() - t0:.1f}s", flush=True)
            write_report()
        rows = list(arm["domains"].values())
        n = sum(row["anchors"] for row in rows)
        pooled = report["pooled"][name] = {"anchors": n, "emitted_per_verify": 1 + sum(row["sum_accepted"] for row in rows) / n}
        if base:
            vs = [row["vs_base"] for row in rows]
            pooled.update(baseline=base,
                          delta_emitted_per_verify=sum(r["delta_sum_accepted"] for r in vs)/n,
                          longer_prefix=sum(r["longer_prefix"] for r in vs),
                          shorter_prefix=sum(r["shorter_prefix"] for r in vs),
                          domain_delta_count=[sum(r["delta_sum_accepted"] > 0 for r in vs),
                                              sum(r["delta_sum_accepted"] < 0 for r in vs)])
        write_report()

    if "kv" in a.groups:
        bf16 = load("BF16")
        run("BF16", bf16, store=True)
        for kv in ("f16", "q8_0", "q8_0_k"):
            run(f"BF16_kv_{kv}", bf16, kv, base="BF16")
        del bf16

    if "q4_0" in a.groups:
        q4 = load("Q4_0")
        q8 = load("Q8_0")
        run("Q4_0", q4, store=True)
        for role, expr in FAMILIES.items():
            run(f"Q4_0_upgrade_{role}", swap(q4, q8, expr), base="Q4_0")
        del q4, q8

    if "q4_k" in a.groups:
        mixed = load("Q4_K-mixed")
        bf16 = load("BF16")
        run("Q4_K-mixed", mixed, store=True)
        for label, pattern, dtype in (
            ("attn_out_to_Q4_K", FAMILIES["attn_out"], T.Q4_K),
            ("ffn_down_to_Q4_K", FAMILIES["ffn_down"], T.Q4_K),
            ("selector_hidden_to_Q8_0", FAMILIES["selector_hidden"], T.Q8_0),
        ):
            run(f"mixed_{label}", swap(mixed, bf16, pattern, dtype), base="Q4_K-mixed")
        del mixed, bf16

    if "codebooks" in a.groups:
        mixed = load("Q4_0-mixed")
        q8 = load("Q8_0")
        run("Q4_0-mixed", mixed, store=True)
        run("Q4_0-mixed_both_codebooks_to_Q8_0", swap(mixed, q8, FAMILIES["selector_both_codebooks"]), base="Q4_0-mixed")


if __name__ == "__main__":
    main()
