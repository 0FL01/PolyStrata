#!/usr/bin/env python3
"""Export fixed selector inputs and oracle outputs from CPU-only DFT1 replay."""

import argparse
import importlib.util
import pathlib
import struct
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ref", "bridge", "target", "model", "out"):
        parser.add_argument(f"--{name}", required=True, type=pathlib.Path)
    parser.add_argument("--per-trace", type=int, default=16)
    parser.add_argument("trace", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    if args.per_trace < 1:
        parser.error("--per-trace must be positive")

    spec = importlib.util.spec_from_file_location("dflash2_ref", args.ref)
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    bridge = ref.QBridge(args.bridge, nthreads=1)
    weights = ref.DFlash2Weights(args.model, bridge)
    target = ref.TargetSharedWeights(args.target, bridge, lm_head_mode="f32")
    model = ref.DFlash2Model(weights, target, kv_quant="fp32")
    records = []
    for path in args.trace:
        trace = ref.load_dft1(path)
        first = int(np.flatnonzero(trace.is_gen)[0])
        last = len(trace.pos) - ref.BLOCK_SIZE
        if not np.array_equal(trace.pos, np.arange(len(trace.pos))) or last < first:
            raise ValueError(f"{path}: need contiguous positions and at least one full generated block")
        starts = np.linspace(first, last, min(args.per_trace, last - first + 1), dtype=int)
        model.reset_cache()
        prev = 0
        for start in starts:
            model.inject(trace.taps[prev:start], trace.pos[prev:start])
            anchor = int(trace.token_id[start])
            out = model.draft_block(anchor, int(start), temperature=0.0)
            gate = out["h_draft"] @ weights.tensors["selector_hidden.weight"].T
            records.append(b"".join((
                struct.pack("<i", anchor),
                out["cand"].astype("<i4", copy=False).tobytes(),
                out["unary"].astype("<f4", copy=False).tobytes(),
                gate.astype("<f4", copy=False).tobytes(),
                out["draft_tokens"].astype("<i4", copy=False).tobytes(),
                out["sel_scores"].astype("<f4", copy=False).tobytes(),
                out["sel_probs"].astype("<f4", copy=False).tobytes(),
            )))
            prev = start
        print(f"{path}: {len(starts)} blocks, start positions {starts.tolist()}", flush=True)
    with args.out.open("xb") as stream:
        stream.write(struct.pack("<4sI", b"DFQ1", len(records)))
        stream.writelines(records)


if __name__ == "__main__":
    main()
