#!/usr/bin/env python3
"""Check the selector payloads in the five supplied GGUFs and a bench pack."""

import argparse
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "third_party/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

NAMES = ("selector_predecessor.weight", "selector_successor.weight")


def read_books(path: Path) -> dict:
    tensors = {t.name: t for t in GGUFReader(path).tensors if t.name in NAMES}
    if set(tensors) != set(NAMES):
        raise ValueError(f"{path}: missing codebook")
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("pack_dir", type=Path)
    args = parser.parse_args()
    profiles = {label: read_books(args.model_dir / f"Inkling-2-Mini-DFlash2-{label}.gguf")
                for label in ("BF16", "Q4_0-mixed", "Q4_K-mixed", "Q4_0", "Q8_0")}
    packed = {label: read_books(args.pack_dir / f"{label}.gguf")
              for label in ("f16", "q4_0", "q8_0")}

    for name in NAMES:
        bf16 = profiles["BF16"][name].data.view(np.uint16)
        f16 = profiles["Q4_0-mixed"][name].data
        if bf16.size != 51214848 or f16.size != 51214848:
            raise ValueError(f"{name}: unexpected element count")
        n_diff, max_diff = 0, 0.0
        for start in range(0, 200058, 1024):
            rows = slice(start, start + 1024)
            ref = (bf16[rows].astype(np.uint32) << 16).view(np.float32)
            got = f16[rows].astype(np.float32)
            n_diff += np.count_nonzero(ref.view(np.uint32) != got.view(np.uint32))
            max_diff = max(max_diff, float(np.max(np.abs(ref - got))))
        print(f"{name}: BF16/F16 differ in {n_diff}/{bf16.size} FP32 words, max abs={max_diff:.8g}")
        if not np.array_equal(profiles["Q4_K-mixed"][name].data, profiles["Q8_0"][name].data):
            raise ValueError(f"{name}: Q4_K-mixed and Q8_0 payloads differ")
        for fmt, source in (("f16", "Q4_0-mixed"), ("q4_0", "Q4_0"), ("q8_0", "Q8_0")):
            observed = profiles[source][name].data.view(np.uint8)
            if not np.array_equal(observed, packed[fmt][name].data.view(np.uint8)):
                raise ValueError(f"{name}: {source} and packed {fmt} payloads differ")
        print("  Q4_K-mixed/Q8_0 match; packed F16/Q8_0/Q4_0 match supplied tables")


if __name__ == "__main__":
    main()
