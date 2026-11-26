#!/usr/bin/env python3
"""Compare literal llama.cpp CPU graph functions to the saved real BF16-source states."""
from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path

import numpy as np

from dflash2_ref import (BLOCK_SIZE, DEFAULT_DFLASH2_DIR, DEFAULT_RECON_DIR,
                         DEFAULT_TARGET_GGUF, HEAD_DIM, HIDDEN_SIZE, MASK_TOKEN_ID,
                         NUM_KV_HEADS, NUM_LAYERS, SELECTOR_TOP_K, VOCAB_VALID,
                         DFlash2Weights, QBridge, TargetSharedWeights,
                         grouped_dynamic_causal_conv, rms_norm)
from replay_matrix import DOMAINS, load_domain

OUT = DEFAULT_RECON_DIR / "cpu-reference"
P, I = c.c_void_p, c.c_int
H, B, K = HIDDEN_SIZE, BLOCK_SIZE, SELECTOR_TOP_K


def ptr(a):
    if not a.flags.c_contiguous:
        raise ValueError("probe inputs and outputs must be C-contiguous")
    return a.ctypes.data


class LayerWeights(c.Structure):
    _fields_ = [(s, P) for s in (
        "attn_norm", "attn_conv_proj", "attn_conv_base", "attn_q", "attn_k", "attn_v",
        "attn_output", "attn_q_norm", "attn_k_norm", "ffn_norm", "ffn_conv_proj",
        "ffn_conv_base", "ffn_gate", "ffn_up", "ffn_down",
    )]


class LayerOutputs(c.Structure):
    _fields_ = [(s, P) for s in (
        "ctx_k", "ctx_v", "attn_delta", "attn_preconv", "block_q", "block_k", "block_v",
        "attn_out", "attn_residual", "ffn_delta", "ffn_preconv", "out",
    )]


class Donor:
    def __init__(self, path, threads):
        self.threads = threads
        self.lib = c.CDLL(str(path))
        self.lib.dflash_donor_hashes.restype = c.c_char_p
        self.hashes = dict(row.split() for row in self.lib.dflash_donor_hashes().decode().splitlines())
        for name, types in {
            "project": (P, P, I, I, I, I, P), "norm": (P, P, I, I, P),
            "fuse": (P, P, P, I, I, P), "conv": (P, P, P, I, I, P),
            "layer": (P, P, P, I, c.POINTER(LayerWeights), c.POINTER(LayerOutputs), I),
            "lattice": (P, P, P, P, P, P, I, I, P),
        }.items():
            fn = getattr(self.lib, f"dflash_donor_{name}")
            fn.argtypes = types
            fn.restype = I

    def call(self, name, *args):
        ret = getattr(self.lib, f"dflash_donor_{name}")(*args)
        if ret:
            raise RuntimeError(f"donor {name}: ggml status {ret}")

    def norm(self, x, w):
        out = np.empty_like(x)
        self.call("norm", ptr(x), ptr(w), x.shape[0], self.threads, ptr(out))
        return out

    def project(self, x, w):
        out = np.empty((len(x), len(w)), dtype=np.float32)
        self.call("project", ptr(w), ptr(x), x.shape[-1], len(w), len(x), self.threads, ptr(out))
        return out

    def lattice(self, x, logits, tokens, weights, postcap):
        out = np.empty((B, H), dtype=np.float32)
        self.call("lattice", ptr(x), ptr(logits), ptr(tokens), ptr(weights.tensors["selector_hidden.weight"]),
                  ptr(weights.pred_cb.raw_u8), ptr(weights.succ_cb.raw_u8), int(postcap), self.threads, ptr(out))
        return out


def diff(ref, got):
    a, b = ref.astype(np.float64), got.astype(np.float64)
    delta = a - b
    return {"rel_rmse": float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(a.ravel()), 1e-30)),
            "max_abs": float(np.max(np.abs(delta))), "mean_abs": float(np.mean(np.abs(delta))),
            "bit_identical": ref.tobytes() == got.tobytes()}


def walk(lattice):
    candidates = lattice[1:, :K].astype(np.int32)
    all_scores = lattice[1:, K:K + K*K].reshape(B-1, K, K)
    row = 0
    scores, tokens = [], []
    for cs, ls in zip(candidates, all_scores, strict=True):
        scores.append(ls[row])
        row = int(ls[row].argmax())
        tokens.append(cs[row])
    return candidates, np.stack(scores), np.asarray(tokens, dtype=np.int32)


def selector_diff(reference, operated):
    cand, _, tokens = walk(operated)
    ref_c = reference["cand"]
    idx = cand[:, None, :] == ref_c[:, :, None]
    # Read the donor lattice at the reference's predecessor, even when its
    # own greedy walk has diverged. Align scores by candidate ID, not slot.
    pred_match = cand[:-1] == reference["predecessors"][1:, None]
    pred_rows = np.concatenate(([0], pred_match.argmax(axis=-1)))
    pred_shared = np.concatenate(([True], pred_match.any(axis=-1)))
    all_scores = operated[1:, K:K + K*K].reshape(B-1, K, K)
    scores = all_scores[np.arange(B-1), pred_rows]
    aligned_scores = scores[np.arange(B-1)[:, None], idx.argmax(axis=-1)]
    shared = idx.any(axis=-1) & pred_shared[:, None]
    return {"ordered_candidates": int((cand == ref_c).sum()), "common_candidates": int(idx.sum()),
            "total_candidates": (B-1) * K, "proposal_ids": tokens.tolist(),
            "matched_proposals": int((tokens == reference["draft_tokens"]).sum()),
            "scores_compared_on_reference_path": int(shared.sum()),
            "score_on_shared_tokens": diff(reference["sel_scores"][shared], aligned_scores[shared]) if shared.any() else None}


def compare(donor, weights, target, trace, ref, start):
    t = weights.tensors
    nctx = len(ref["ctx_positions"])
    positions = np.concatenate((ref["ctx_positions"], ref["block_pos"])).astype(np.int32)
    if nctx == 0 or start != int(ref["block_pos"][0]) or int(trace.token_id[start]) != int(ref["anchor"]):
        raise ValueError("intermediate/DFT1 input mismatch")
    tokens = np.full(B, MASK_TOKEN_ID, dtype=np.int32)
    tokens[0] = ref["anchor"]
    x = donor.norm(target.bridge.dequantize_rows(target.embd_raw, target.embd_qtype, tokens, H), target.embd_norm)
    fused = np.empty((nctx, H), dtype=np.float32)
    taps = trace.taps[positions[:nctx]]
    donor.call("fuse", ptr(taps), ptr(t["fc.weight"]), ptr(t["enc_output_norm.weight"]), nctx, donor.threads, ptr(fused))
    checks = {"emb_normed": diff(ref["emb_normed"], x), "ctx_hidden": diff(ref["ctx_hidden"][positions[:nctx]], fused)}
    isolated = {}
    outputs = {"emb_normed": x, "ctx_hidden": fused}

    for i in range(NUM_LAYERS):
        layer = f"layer_{i}_"
        w = f"blk.{i}."
        names = [f"{w}{key}{'' if key.endswith('_base') else '.weight'}" for key, _ in LayerWeights._fields_]
        lw = LayerWeights(*(ptr(t[name]) for name in names))
        values = {key: np.empty_like(ref[key][i] if key.startswith("ctx_") else ref[layer+key])
                  for key, _ in LayerOutputs._fields_}
        lo = LayerOutputs(*(ptr(values[key]) for key, _ in LayerOutputs._fields_))
        donor.call("layer", ptr(x), ptr(fused), ptr(positions), nctx, c.byref(lw), c.byref(lo), donor.threads)
        x = values["out"]
        for key, value in values.items():
            expected = ref[key][i] if key.startswith("ctx_") else ref[layer+key]
            checks[layer+key] = diff(expected, value)
            outputs[layer+key] = value

        # Exact same-input regression for both sides of each real conv. The
        # expected value here is just the NumPy conv, not a full layer result.
        old_x = ref["emb_normed"] if i == 0 else ref[f"layer_{i-1}_out"]
        ffn_input = ref[layer+"ffn_preconv"]
        gate = ffn_input @ t[w+"ffn_gate.weight"].T
        up = ffn_input @ t[w+"ffn_up.weight"].T
        conv_inputs = {
            "attn": (rms_norm(old_x, t[w+"attn_norm.weight"]), ref[layer+"attn_out"] @ t[w+"attn_output.weight"].T),
            "ffn": (rms_norm(ref[layer+"attn_residual"], t[w+"ffn_norm.weight"]),
                    (gate * np.exp(-np.logaddexp(0, -gate)) * up) @ t[w+"ffn_down.weight"].T),
        }
        for key, pair in conv_inputs.items():
            delta, base = ref[layer+key+"_delta"], t[w+key+"_conv_base"]
            for side, inp in enumerate(pair):
                out = np.empty_like(inp)
                donor.call("conv", ptr(inp), ptr(delta), ptr(base), side, donor.threads, ptr(out))
                expected = grouped_dynamic_causal_conv(inp, delta[:, side], base[side])
                isolated[layer+key+f"_side_{side}"] = diff(expected, out)

    h = donor.norm(x, t["output_norm.weight"])
    if target.output_f32 is None:
        target.output_f32 = target.bridge.dequantize_2d(target.output_raw, target.output_qtype, VOCAB_VALID, H)
    logits = donor.project(h, target.output_f32)
    checks["h_out"] = diff(ref["h_out"], h)
    checks["raw_logits"] = diff(ref["raw_logits"], logits[1:])
    outputs.update(h_out=h, raw_logits=logits[1:])

    frozen = np.concatenate((np.zeros((1, VOCAB_VALID), dtype=np.float32), ref["raw_logits"]))
    after = donor.lattice(ref["h_out"], frozen, tokens, weights, True)
    uncapped = donor.lattice(ref["h_out"], frozen, tokens, weights, False)
    before = donor.lattice(ref["h_out"], 30*np.tanh(frozen/30), tokens, weights, False)
    end_to_end = donor.lattice(h, logits, tokens, weights, True)
    controls = {"after_correction_frozen_inputs": selector_diff(ref, after),
                "uncapped_frozen_inputs": selector_diff(ref, uncapped),
                "before_correction_frozen_inputs": selector_diff(ref, before),
                "after_correction_full_backbone": selector_diff(ref, end_to_end)}
    outputs.update(lattice_frozen_after=after, lattice_frozen_uncapped=uncapped,
                   lattice_frozen_before=before, lattice_full_after=end_to_end)
    return {"anchor": int(tokens[0]), "stages": checks, "same_input_conv": isolated, "selector": controls}, outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taps", type=Path, default=DEFAULT_RECON_DIR / "taps")
    p.add_argument("--states", type=Path, default=OUT / "states")
    p.add_argument("--bridge", type=Path, default=OUT / "bridge/libggml_qbridge.so")
    p.add_argument("--donor", type=Path, default=OUT / "bridge/libggml_dflash2_donor.so")
    p.add_argument("--source", type=Path, default=DEFAULT_DFLASH2_DIR / "source/model.safetensors")
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET_GGUF)
    p.add_argument("--output", type=Path, default=OUT / "donor")
    p.add_argument("--domains", nargs="+", choices=DOMAINS, default=["code_python", "tool_calls"])
    p.add_argument("--max-cases-per-domain", type=int, default=0, help="0 = all matching source-state files")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.threads < 1 or a.max_cases_per_domain < 0:
        p.error("invalid thread/case count")
    a.output.mkdir(parents=True, exist_ok=True)
    donor = Donor(a.donor, a.threads)
    bridge = QBridge(a.bridge, a.threads)
    target = TargetSharedWeights(a.target, bridge)
    weights = DFlash2Weights(a.source, bridge)
    with a.source.open("rb") as f:
        checkpoint_hash = hashlib.file_digest(f, "sha256").hexdigest()
    report = {"graph_fragment_sha256": donor.hashes, "config": {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
              "source_sha256": checkpoint_hash, "cases": {}}
    failed = False
    for domain in a.domains:
        trace, _, _ = load_domain(a.taps, domain)
        cases = sorted(a.states.glob(f"{domain}-*-source.npz"))
        if not cases:
            raise FileNotFoundError(f"no captured source states for {domain} in {a.states}")
        if a.max_cases_per_domain:
            cases = cases[:a.max_cases_per_domain]
        for path in cases:
            pos = int(path.stem.rsplit("-", 2)[1])
            tag = f"{domain}-{pos}"
            with np.load(path) as states:
                metrics, outputs = compare(donor, weights, target, trace, {k: states[k] for k in states.files}, pos)
            report["cases"][tag] = metrics
            np.savez_compressed(a.output / f"{tag}-donor.npz", **outputs)
            (a.output / "parity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            ids = metrics["selector"]["after_correction_full_backbone"]["matched_proposals"]
            hidden = metrics["stages"]["h_out"]["rel_rmse"]
            print(f"{tag}: final hidden rel-RMSE {hidden:.3g}, {ids}/7 proposals", flush=True)
            failed |= hidden > 1e-4 or any(e["max_abs"] != 0 for e in metrics["same_input_conv"].values())
            for key in ("after_correction_frozen_inputs", "after_correction_full_backbone"):
                s = metrics["selector"][key]
                failed |= (s["common_candidates"] != (B-1)*K or s["matched_proposals"] != B-1
                           or s["score_on_shared_tokens"]["max_abs"] > 2e-3)
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
