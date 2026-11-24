#!/usr/bin/env python3
"""Normalize the archived 2026-11-23 D3 route-capture fixture, not a generic --dump-routing reader."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_routes import check, physical, sha256, stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact_dir", type=Path, help="historical .cache/claude-fv directory")
    p.add_argument("--allow-repeated-continuation", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = args.artifact_dir / "route/frames.bin"
    log = args.artifact_dir / "runs/d3-capture-213530/command.log"
    binary = args.artifact_dir / "bin/route-capture"
    log_lines = log.read_text().strip().splitlines()
    check("prompt=2048 new=96" in log_lines[0] and "INKLING_ROUTE_CAPTURE" in log_lines[0]
          and "NVFP4" in "\n".join(log_lines[1:]), "wrong capture provenance")
    forced = log_lines[-1].split(",")
    period = next(i for i in range(1, len(forced) + 1) if forced[i:] == forced[:-i])
    raw = np.frombuffer(source.read_bytes(), dtype="<u4")
    check(len(raw) == (len(forced) - 1) * 48 * (128 * 2 + 1 + 8 * 2), "unexpected capture shape")
    frames = raw.reshape(-1, 48, 273)
    bias = frames[:, :, 128:256]
    check(np.all(bias == bias[0]), "router bias changed within a logical visit")
    check(np.all(bias[0, 6:18] == bias[0, 18:30]) and np.all(bias[0, 6:18] == bias[0, 30:42]),
          "router bias does not match physical layer reuse")
    check(len({bias[0, v].tobytes() for v in (*range(18), *range(42, 48))}) == 24, "ambiguous router bias mapping")
    check(np.all(frames[:, :, 256] == frames[0, 0, 256]), "unexpected expert scales")
    n = len(frames) if args.allow_repeated_continuation else period
    meta = dict(schema="inkling-route-v1", phase="serial_decode", model_basename="Inkling-2-Mini-UD-IQ4_XS.gguf",
                model_sha256=None, identity_basis="historical config backups and NVFP4 capture log, no recorded model hash",
                workload="forced synthetic engineering notebook, 2048 prompt tokens", trajectory_id="fv-d3-capture-213530",
                input=str(source), input_sha256=sha256(source), binary_sha256=sha256(binary), log_sha256=sha256(log),
                frames_total=len(frames), frames_used=n, continuation_period_tokens=period,
                placement_basis="not recorded; no miss predictions or latency labels")
    with args.output.open("x") as f:
        def write(row):
            print(json.dumps(row, separators=(",", ":")), file=f)
        write(meta)
        for pos, token in enumerate(frames[:n], start=2048):
            for visit, row in enumerate(token):
                write({"pos": pos, "visit": visit, "layer": physical(visit), "pass": stage(visit),
                       "ids": row[257:265].tolist(), "weight_bits": row[265:273].tolist()})


if __name__ == "__main__":
    main()
