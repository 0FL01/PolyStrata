#!/usr/bin/env python3
"""Run matched guarded native selector placement legs with the same live ELF/DSO."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

MODES = ("host", "gpu", "host-lru")
WEIGHTS = ("Q4_0-mixed", "Q4_0")


def read_leg(path: Path, mode: str, timed: bool) -> dict:
    if json.loads((path / "guard/stop-reasons.json").read_text()) != []:
        raise ValueError(f"{path}: guard did not pass")
    result = json.loads((path / "result.json").read_text())
    lines = (path / "guard/command.log").read_text().splitlines()
    for line in lines:
        if line.startswith("DF2_BOOKS "):
            book = dict(x.split("=", 1) for x in line.split()[1:])
            if book.pop("placement") != mode:
                raise ValueError(f"{path}: wrong selector placement")
            result.update({k: int(v) for k, v in book.items()})
    if (timed or mode != "host") and "device_table_bytes" not in result:
        raise ValueError(f"{path}: missing selector placement log")
    rounds = {}
    walks = {}
    for line in lines:
        if line.startswith("DF2_ROUND "):
            pos, k, accept, fed, draft_ms, full_ms = line.split()[1:]
            if int(k):
                rounds[int(pos), int(k)] = (int(accept), int(fed), float(draft_ms), float(full_ms))
        elif line.startswith("DF2_BOOK_WALK "):
            pos, anchor, k, nbytes, d2h, dev, cpu, pred, succ, *ids = line.split()[1:]
            key = int(pos), int(k)
            if key in walks or len(ids) != key[1]:
                raise ValueError(f"bad selector record: {line}")
            walks[key] = {"anchor": int(anchor), "ids": list(map(int, ids)), "bytes": int(nbytes),
                          "d2h_us": float(d2h), "kernel_us": float(dev), "cpu_us": float(cpu),
                          "pred_hits": int(pred), "succ_hits": int(succ)}
    if timed and (not walks or set(rounds) != set(walks)):
        raise ValueError(f"{path}: timed walks must cover every offered round")
    result.update(placement=mode, timed=timed)
    result["walks"] = [{"pos": p, "k": k, "accepted": rounds[p, k][0], **v} for (p, k), v in walks.items()]
    if walks:
        full = [w for w in result["walks"] if w["k"] == 7]
        if full:
            expected = 28 if mode == "gpu" else 8064
            assert {w["bytes"] for w in full} == {expected}, (path, full)
        result["width7"] = {field: {
            "p50": statistics.median(v := [w[field] for w in full]),
            "p90": statistics.quantiles(v, n=10, method="inclusive")[8] if len(v) > 1 else v[0],
        } for field in ("d2h_us", "kernel_us", "cpu_us")} if full else None
        if full:
            total = [w["d2h_us"] + w["kernel_us"] + w["cpu_us"] for w in full]
            result["width7"]["total_us"] = {
                "p50": statistics.median(total),
                "p90": statistics.quantiles(total, n=10, method="inclusive")[8] if len(total) > 1 else total[0],
            }
        if mode == "host-lru":
            result["cache_hit_fraction"] = {
                "pred": sum(w["pred_hits"] for w in walks.values()) / result["offered"],
                "succ": sum(w["succ_hits"] for w in walks.values()) / (16 * result["offered"]),
            }
    return result


def compare(a: dict, b: dict) -> dict:
    first_id = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), None)
    if first_id is None and len(a["ids"]) != len(b["ids"]):
        first_id = min(len(a["ids"]), len(b["ids"]))
    emitted = {"emitted_equal": a["ids"] == b["ids"], "first_emitted_disagreement": first_id}
    if not a["walks"] or not b["walks"]:
        return emitted
    aw = {(x["pos"], x["anchor"], x["k"]): x for x in a["walks"]}
    bw = {(x["pos"], x["anchor"], x["k"]): x for x in b["walks"]}
    shared = sorted(aw.keys() & bw.keys())
    n = sum(k[-1] for k in shared)
    same = sum(x == y for k in shared for x, y in zip(aw[k]["ids"], bw[k]["ids"]))
    first = next(([k[0], k[1], i, x, y] for k in shared
                  for i, (x, y) in enumerate(zip(aw[k]["ids"], bw[k]["ids"])) if x != y), None)
    return {"shared_proposals": n, "same_proposals": same, "shared_rounds": len(shared),
            "same_accepted_rounds": sum(aw[k]["accepted"] == bw[k]["accepted"] for k in shared),
            "host_only_rounds": len(aw) - len(shared), "trial_only_rounds": len(bw) - len(shared),
            **emitted, "first_proposal_disagreement": first}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "out", "config", "profile", "binary", "library", "model-dir"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--domain", default="code_python")
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--cap", type=int)
    p.add_argument("--weight", action="append", choices=WEIGHTS)
    p.add_argument("--placement", action="append", choices=MODES)
    p.add_argument("--serial-verify", action="store_true")
    p.add_argument("--clean", action="store_true", help="disable event timing and proposal logs")
    p.add_argument("--pinned", action="store_true", help="stage selector readback in persistent pinned memory")
    a = p.parse_args()

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    runner = Path(__file__).resolve().parent.parent / "live/measure.py"
    cfg = json.loads(a.config.read_text())
    digest = hashlib.sha256(a.library.read_bytes()).hexdigest()
    binary_digest = hashlib.sha256(a.binary.read_bytes()).hexdigest()
    profile_digest = hashlib.sha256(a.profile.read_bytes()).hexdigest()
    results = {}
    for weight in a.weight or WEIGHTS:
        for i, mode in enumerate(a.placement or MODES):
            leg = out / f"{weight}-{mode}-{i}"
            cli = dict(cfg, env=dict(cfg["env"], DF2_BOOKS=mode, DF2_BOOKS_TIMING="0" if a.clean else "1",
                                    DF2_BOOKS_PINNED=str(int(a.pinned))))
            config = leg.with_suffix(".json")
            config.write_text(json.dumps(cli))
            cmd = [sys.executable, str(runner), "--run", str(a.run), "--out", str(leg), "--config", str(config),
                   "--profile", str(a.profile), "--domain", a.domain, "--mode", "live", "--steps", str(a.steps),
                   "--binary", str(a.binary), "--library", str(a.library), "--weights",
                   str(a.model_dir / f"Inkling-2-Mini-DFlash2-{weight}.gguf")]
            if a.cap is not None:
                cmd += ["--cap", str(a.cap)]
            if a.serial_verify:
                cmd += ["--serial-verify"]
            with leg.with_suffix(".command").open("w") as log:
                # measure.py holds the shared GPU lock for the entire guarded leg.
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            if hashlib.sha256(a.library.read_bytes()).hexdigest() != digest:
                raise ValueError("DSO changed during the comparison")
            r = read_leg(leg, mode, not a.clean)
            if bool(r.get("pinned_bytes")) != a.pinned:
                raise ValueError(f"{leg}: wrong readback memory type")
            if r["binary_sha256"] != binary_digest or r["profile_seed_sha256"] != profile_digest:
                raise ValueError(f"{leg}: binary or profile seed changed during the comparison")
            results[leg.name] = r
            print(json.dumps({k: v for k, v in r.items() if k not in ("ids", "walks")}), flush=True)

    comparisons = {}
    for weight in a.weight or WEIGHTS:
        same_weight = [(label, r) for label, r in results.items()
                       if r["weights"] == f"Inkling-2-Mini-DFlash2-{weight}.gguf"]
        host = next(((label, r) for label, r in same_weight if r["placement"] == "host"), None)
        if host:
            for label, r in same_weight:
                if label != host[0]:
                    comparisons[label] = compare(host[1], r)
    (out / "books-results.json").write_text(json.dumps({"dso_sha256": digest, "legs": results,
                                                         "vs_host": comparisons}, indent=2) + "\n")
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
