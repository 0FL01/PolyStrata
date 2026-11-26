#!/usr/bin/env python3
"""CPU-only greedy replay on the original Tideglass DFT1 domain captures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from dflash2_ref import (
    BLOCK_SIZE, DEFAULT_BRIDGE, DEFAULT_DFLASH2_DIR, DEFAULT_RECON_DIR,
    DEFAULT_TARGET_GGUF, SLIDING_WINDOW, VOCAB_VALID, DFlash2Model,
    DFlash2Weights, QBridge, TargetSharedWeights, load_dft1,
)

PROFILES = ("BF16", "Q8_0", "Q4_0", "Q4_0-mixed", "Q4_K-mixed")
DOMAINS = ("en_reasoning", "code_python", "code_rust", "code_cpp", "tool_calls", "json_struct", "agent_traj")
DRAFT_WIDTH = BLOCK_SIZE - 1


def load_domain(tap_dir: Path, domain: str) -> tuple:
    entry = json.loads((tap_dir / "manifest.json").read_text())[domain]
    path = tap_dir / f"{domain}.dft1"
    tr = load_dft1(path)
    ids = np.asarray(entry["prompt_ids"] + entry["tokens"], dtype=np.int32)
    prompt = int(entry["prompt_len"])
    n = len(tr.pos)
    # The last emitted token is not evaluated by the serial target; it has no
    # taps but is a valid next-token label. It can never be injected in replay.
    if (n not in (len(ids), len(ids) - 1) or len(ids) != prompt + entry["gen_len"]
            or not np.array_equal(tr.pos, np.arange(n))
            or not np.array_equal(tr.token_id, ids[:n])
            or not np.array_equal(tr.is_gen, np.arange(n) >= prompt)
            or np.any(ids < 0) or np.any(ids >= VOCAB_VALID) or not np.all(np.isfinite(tr.taps))):
        raise ValueError(f"misaligned trace/manifest or invalid tap values: {path}")
    return tr, ids, prompt


def prefix_len(match: np.ndarray) -> np.ndarray:
    return np.cumprod(match, axis=1).sum(axis=1)


def acceptance(proposals: np.ndarray, truth: np.ndarray) -> dict:
    if not len(truth):
        return {"anchors": 0}
    match = proposals == truth
    accepted = prefix_len(match)
    hist = np.bincount(accepted, minlength=BLOCK_SIZE)
    return {
        "anchors": len(truth), "emitted_per_verify": 1 + float(accepted.mean()),
        "accepted_per_verify": float(accepted.mean()), "sum_accepted": int(accepted.sum()),
        "hist_accepted_0_to_7": hist.tolist(),
        "prefix_survival": (np.cumsum(hist[:0:-1])[::-1] / len(truth)).tolist(),
        "per_offset_match": match.mean(axis=0).tolist(),
    }


def closed_loop_rows(proposals: np.ndarray, truth: np.ndarray) -> np.ndarray:
    row = 0
    rows = []
    accepted = prefix_len(proposals == truth)
    while row < len(accepted):
        rows.append(row)
        row += 1 + accepted[row]
    return np.asarray(rows, dtype=np.int64)


def compare_selectors(pred: dict, ref: dict) -> dict:
    a, b = pred["cand"], ref["cand"]
    # Compare support, scores and probabilities by token ID, not candidate rank.
    pairs = a[..., :, None] == b[..., None, :]
    shared = pairs.any(axis=-1)
    p, q = pred["sel_probs"], ref["sel_probs"]
    tvd = 0.5 * ((np.abs(p[..., :, None] - q[..., None, :]) * pairs).sum(axis=(-1, -2))
                 + (p * ~shared).sum(axis=-1) + (q * ~pairs.any(axis=-2)).sum(axis=-1))
    score_err = np.abs(pred["sel_scores"][..., :, None] - ref["sel_scores"][..., None, :])[pairs]
    h, r = pred["h_draft"], ref["h_draft"]
    same_prev = pred["predecessors"] == ref["predecessors"]
    return {
        "draft_token_agreement": float((pred["draft_tokens"] == ref["draft_tokens"]).mean()),
        "top16_overlap": float(shared.mean()),
        "h_draft_rel_rmse": float(np.sqrt(np.sum((h-r)**2, dtype=np.float64) / np.sum(r**2, dtype=np.float64))),
        "score_mae_on_shared_tokens": float(score_err.mean()),
        "realized_path_mean_tvd": float(tvd.mean()),
        "same_predecessor_fraction": float(same_prev.mean()),
        "same_predecessor_mean_tvd": float(tvd[same_prev].mean()),
    }


def read_dspark(path: Path, ids: np.ndarray, prompt: int, n_anchors: int) -> tuple[np.ndarray, dict]:
    logs = path.read_text().splitlines()
    matched = []
    first_mismatch = None
    first_short = None
    for i, line in enumerate(logs):
        fields = line.split()
        pos, anchor = (int(n) for n in fields[:2])
        if pos != prompt + i or pos >= len(ids) or anchor != int(ids[pos]):
            first_mismatch = {"position": pos, "logged": anchor,
                              "truth": int(ids[pos]) if pos < len(ids) else None}
            break
        if i < n_anchors and first_short is None:
            proposals = [int(x.split(":")[0]) for x in fields[2:]]
            if len(proposals) != DRAFT_WIDTH:
                first_short = {"position": pos, "proposals": len(proposals)}
            else:
                matched.append(proposals)
    return np.asarray(matched, dtype=np.int32).reshape(-1, DRAFT_WIDTH), {
        "log_rows": len(logs), "first_anchor_mismatch": first_mismatch,
        "same_history_full_width_anchors": len(matched), "first_short_block": first_short,
    }


def run_domain(model: DFlash2Model, tr, ids: np.ndarray, starts: np.ndarray,
               batch: int) -> dict[str, np.ndarray]:
    model.reset_cache()
    runs = []
    if starts[-1] > SLIDING_WINDOW and batch > 1:
        raise ValueError("use --batch 1 for captures extending past the ring capacity")
    end = 0
    for j in range(0, len(starts), batch):
        rows = starts[j:j + batch]
        hi = int(rows[-1])
        if hi > end:
            model.inject(tr.taps[end:hi], tr.pos[end:hi])
            end = hi
        runs.extend(model.draft_batch(ids[rows], rows))
    return {k: np.stack([r[k] for r in runs]) for k in runs[0]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taps", type=Path, default=DEFAULT_RECON_DIR / "taps")
    p.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET_GGUF)
    p.add_argument("--draft-dir", type=Path, default=DEFAULT_DFLASH2_DIR)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profiles", nargs="+", choices=(*PROFILES, "source"), default=PROFILES)
    p.add_argument("--domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    p.add_argument("--dspark-suffix", default=".dspark.log", help="alternative replay log suffix")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-anchors", type=int, default=0, help="0 = every full-width anchor")
    p.add_argument("--head", choices=["blas", "f32", "q8k"], default="blas")
    p.add_argument("--kv", choices=["fp32", "f16", "q8_0", "q8_0_k"], default="fp32")
    args = p.parse_args()
    if args.batch < 1 or args.max_anchors < 0:
        p.error("--batch must be positive and --max-anchors nonnegative")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    bridge = QBridge(args.bridge, args.threads)
    target = TargetSharedWeights(args.target, bridge, args.head)
    report = {"configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "domains": {}, "profiles": {}}
    inputs = {}
    baseline = {}
    for domain in args.domains:
        tr, ids, prompt = load_domain(args.taps, domain)
        starts = np.arange(prompt, len(ids) - DRAFT_WIDTH)
        if args.max_anchors:
            starts = starts[:args.max_anchors]
        if not len(starts):
            raise ValueError(f"{domain}: no complete draft block")
        truth = ids[starts[:, None] + 1 + np.arange(DRAFT_WIDTH)]
        ds, alignment = read_dspark(args.taps / f"{domain}{args.dspark_suffix}", ids, prompt, len(starts))
        loops = closed_loop_rows(ds, truth[:len(ds)])
        with (args.taps / f"{domain}.dft1").open("rb") as stream:
            trace_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        report["domains"][domain] = {
            "prompt_tokens": prompt, "target_emitted_tokens": len(ids) - prompt,
            "anchors": len(starts), "dft1_rows": len(tr.pos),
            "taps_sha256": trace_hash,
            "dspark": {**alignment, "all_positions": acceptance(ds, truth[:len(ds)]),
                        "closed_loop": acceptance(ds[loops], truth[loops])},
        }
        inputs[domain] = (tr, ids, starts, truth, len(ds))

    for profile in args.profiles:
        arm = report["profiles"][profile] = {}
        src = (args.draft_dir / "source/model.safetensors" if profile == "source"
               else args.draft_dir / f"Inkling-2-Mini-DFlash2-{profile}.gguf")
        model = DFlash2Model(DFlash2Weights(src, bridge), target, args.kv)
        for domain, (tr, ids, starts, truth, n_ds) in inputs.items():
            t0 = time.perf_counter()
            pred = run_domain(model, tr, ids, starts, args.batch)
            d = pred["draft_tokens"]
            u = pred["unary_argmax"]
            loops = closed_loop_rows(d, truth)
            common = closed_loop_rows(d[:n_ds], truth[:n_ds])
            uloop = closed_loop_rows(u, truth)
            recalled = (pred["cand"] == truth[..., None]).any(axis=-1)
            arm[domain] = {
                "all_positions": acceptance(d, truth), "closed_loop": acceptance(d[loops], truth[loops]),
                "unary_all_positions": acceptance(u, truth), "unary_closed_loop": acceptance(u[uloop], truth[uloop]),
                "dspark_common_all_positions": acceptance(d[:n_ds], truth[:n_ds]),
                "dspark_common_closed_loop": acceptance(d[common], truth[common]),
                "top16_recall_per_offset": recalled.mean(axis=0).tolist(),
                "top16_recall": float(recalled.mean()),
                "selector_fixed": int(((d == truth) & (u != truth)).sum()),
                "selector_broke": int(((d != truth) & (u == truth)).sum()),
                "elapsed_cpu_wall_s": time.perf_counter() - t0,
            }
            if profile == args.profiles[0]:
                baseline[domain] = {k: pred[k] for k in ("cand", "sel_scores", "sel_probs", "draft_tokens", "predecessors", "h_draft")}
            else:
                arm[domain]["vs_first_profile"] = compare_selectors(pred, baseline[domain])
            np.savez_compressed(out / f"{profile}-{domain}.npz", positions=starts, truth=truth,
                                draft_tokens=d, unary_argmax=u, candidate_ids=pred["cand"],
                                closed_loop_rows=loops)
            tmp = out / ".matrix.json.tmp"
            tmp.write_text(json.dumps(report, indent=2) + "\n")
            tmp.replace(out / "matrix.json")
            print(f"{profile:11s} {domain:13s} {len(starts):3d} anchors | "
                  f"all {arm[domain]['all_positions']['emitted_per_verify']:.4f} | "
                  f"closed {arm[domain]['closed_loop']['emitted_per_verify']:.4f} | "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
        del model


if __name__ == "__main__":
    main()
