#!/usr/bin/env python3
"""Convert an Inkling 2 DFlash2 drafter (e.g. incoai/Inkling-2-Mini-DFlash2) to GGUF.

Adapted from upstream llama.cpp:
  - Converter: renoirvieir4/llama.cpp@0280b55a24727af3d2bb53dddb6faf1da358b44e
               (convert_inkling2_dflash2_to_gguf.py)
  - Runtime:   ggml-org/llama.cpp#33049 (dflash.target_visits,
               dflash.selector_softcap_after_correction, target tok_norm)

Enhancements for PolyStrata (AMD RDNA2 / gfx1031):
  - Self-contained: reads safetensors directly with numpy + bundled gguf-py
    (no PyTorch or HuggingFace safetensors wheel required).
  - Uses PolyStrata's own ggml C quantizer (libggml-base.a) when available for
    bit-exact Q8_0, Q6_K, Q5_K, Q5_0, Q4_K, IQ4_NL, IQ4_XS, Q4_0, F16, BF16
    quantization directly from the BF16 source checkpoint (zero double-quant).
  - Supports uniform (bf16, f16, q8_0, q4_0) and RDNA2 mixed-precision profiles
    (q4_0-mixed, q4_k-mixed) plus arbitrary --tensor-type REGEX=TYPE overrides.
  - Copies target tokenizer.ggml.* metadata verbatim from the active Tideglass
    GGUF, slices selector codebooks from padded vocab (201024) to real vocab
    (200058), maps logical target visits [5, 23, 41, 48] to physical layers
    [5, 11, 17, 24], and verifies all 81 tensors 1-to-1.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import os
import pathlib
import re
import struct
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

import gguf  # noqa: E402

logger = logging.getLogger("inkling2-dflash2")

T = gguf.GGMLQuantizationType

NAME_TO_QTYPE: dict[str, gguf.GGMLQuantizationType] = {
    "F32": T.F32,
    "F16": T.F16,
    "BF16": T.BF16,
    "Q8_0": T.Q8_0,
    "Q6_K": T.Q6_K,
    "Q5_K": T.Q5_K,
    "Q5_0": T.Q5_0,
    "Q5_1": T.Q5_1,
    "Q4_K": T.Q4_K,
    "Q4_0": T.Q4_0,
    "Q4_1": T.Q4_1,
    "IQ4_NL": T.IQ4_NL,
    "IQ4_XS": T.IQ4_XS,
}

PROFILE_FILETYPE: dict[str, gguf.LlamaFileType] = {
    "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    "f16": gguf.LlamaFileType.MOSTLY_F16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    "q4_0-mixed": gguf.LlamaFileType.MOSTLY_Q4_0,
    "q4_k-mixed": gguf.LlamaFileType.MOSTLY_Q4_K_M,
}

# Exact 1-to-1 tensor mapping from incoai/Inkling-2-Mini-DFlash2 safetensors to GGUF.
STATIC_TENSOR_MAP: dict[str, str] = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc_output_norm.weight",
    "norm.weight": "output_norm.weight",
    "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
    "candidate_selector.successor_codebook": "selector_successor.weight",
}

LAYER_TENSOR_MAP: dict[str, str] = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "attention_conv.base_kernel": "attn_conv_base",
    "attention_conv.kernel_projection.weight": "attn_conv_proj.weight",
    "mlp_conv.base_kernel": "ffn_conv_base",
    "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight",
}

CODEBOOK_SRC_NAMES = {
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
}


def target_layout(reader: gguf.GGUFReader) -> tuple[int, int, int, int, int]:
    """Return (physical_layers, loop_start, loop_end, loop_count, embedding_length)."""
    arch = reader.fields["general.architecture"].contents()
    if arch not in ("inkling2", "inkling2_loop"):
        raise ValueError(f"target is {arch!r}, not an Inkling 2 GGUF")
    n_phys = int(reader.fields[f"{arch}.block_count"].contents())
    start, end, count = 0, 0, 1
    if arch == "inkling2_loop":
        start = int(reader.fields[f"{arch}.loop.start"].contents())
        end = int(reader.fields[f"{arch}.loop.end"].contents())
        count = int(reader.fields[f"{arch}.loop.count"].contents())
    n_embd = int(reader.fields[f"{arch}.embedding_length"].contents())
    return n_phys, start, end, count, n_embd


def visit_to_layer(v: int, n_phys: int, start: int, end: int, count: int) -> int:
    """Map logical execution visit index to physical layer index (n_phys = final norm)."""
    core = (end - start) * count
    if v < start:
        return v
    if v < start + core:
        return start + (v - start) % (end - start)
    return v - core + (end - start)


def map_tensor_name(src_name: str) -> str:
    """Map a DFlash2 safetensors key to its canonical llama.cpp / PolyStrata GGUF name."""
    name = src_name[6:] if src_name.startswith("model.") else src_name
    if name in STATIC_TENSOR_MAP:
        return STATIC_TENSOR_MAP[name]
    m = re.match(r"^layers\.(\d+)\.(.+)$", name)
    if m:
        bid = int(m.group(1))
        rest = m.group(2)
        if rest in LAYER_TENSOR_MAP:
            return f"blk.{bid}.{LAYER_TENSOR_MAP[rest]}"
    raise KeyError(f"Unmapped DFlash2 source tensor: {src_name!r}")


class GgmlQuantBridge:
    """Thin ctypes wrapper around PolyStrata's libggml-base.a C quantizer."""

    def __init__(self, lib_path: pathlib.Path | None = None, nthreads: int = 8):
        self.nthreads = max(1, int(nthreads))
        self.lib = None
        if lib_path is not None and lib_path.is_file():
            self._load(lib_path)

    def _load(self, lib_path: pathlib.Path) -> None:
        lib = ctypes.CDLL(str(lib_path))
        lib.qbridge_quantize.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ]
        lib.qbridge_quantize.restype = ctypes.c_size_t
        lib.qbridge_dequantize.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ]
        lib.qbridge_dequantize.restype = None
        lib.qbridge_eval_error.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
        ]
        lib.qbridge_eval_error.restype = None
        lib.qbridge_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.qbridge_row_size.restype = ctypes.c_size_t
        self.lib = lib

    @property
    def available(self) -> bool:
        return self.lib is not None

    def quantize_2d(self, arr_f32: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
        """Quantize a 2D C-contiguous float32 array [nrows, ncols] into raw uint8 byte matrix."""
        assert self.lib is not None
        a = np.ascontiguousarray(arr_f32, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"Expected 2D array for ggml C quantization, got shape {a.shape}")
        nrows, ncols = a.shape
        row_bytes = int(self.lib.qbridge_row_size(int(qtype), ncols))
        out = np.empty((nrows, row_bytes), dtype=np.uint8)
        written = int(self.lib.qbridge_quantize(
            int(qtype), a.ctypes.data, out.ctypes.data, nrows, ncols, self.nthreads
        ))
        if written != nrows * row_bytes:
            raise RuntimeError(f"ggml_quantize_chunk wrote {written} bytes, expected {nrows * row_bytes}")
        return out

    def dequantize_2d(self, raw_u8: np.ndarray, qtype: gguf.GGMLQuantizationType, ncols: int) -> np.ndarray:
        assert self.lib is not None
        r = np.ascontiguousarray(raw_u8, dtype=np.uint8)
        nrows = r.shape[0]
        out = np.empty((nrows, ncols), dtype=np.float32)
        self.lib.qbridge_dequantize(int(qtype), r.ctypes.data, out.ctypes.data, nrows, ncols, self.nthreads)
        return out

    def eval_error(self, ref_f32: np.ndarray, test_f32: np.ndarray) -> dict[str, float]:
        assert self.lib is not None
        a = np.ascontiguousarray(ref_f32, dtype=np.float32)
        b = np.ascontiguousarray(test_f32, dtype=np.float32)
        stats = np.zeros(5, dtype=np.float64)
        self.lib.qbridge_eval_error(a.ctypes.data, b.ctypes.data, a.size, stats.ctypes.data)
        return {
            "rel_rmse": float(stats[0]),
            "sqnr_db": float(stats[1]),
            "max_abs": float(stats[2]),
            "mean_abs": float(stats[3]),
            "cos_sim": float(stats[4]),
        }


def choose_qtype(
    gguf_name: str,
    ndim: int,
    profile: str,
    overrides: list[tuple[re.Pattern[str], gguf.GGMLQuantizationType]],
) -> gguf.GGMLQuantizationType:
    """Determine the target GGMLQuantizationType for a tensor."""
    # Always keep 1D tensors, norms, and non-.weight tensors (e.g. *_conv_base) in F32,
    # matching llama.cpp's convert_hf_to_gguf.py rules.
    if ndim <= 1 or gguf_name.endswith("_norm.weight") or not gguf_name.endswith(".weight"):
        return T.F32

    for pat, qt in overrides:
        if pat.search(gguf_name):
            return qt

    if profile == "bf16":
        return T.BF16
    if profile == "f16":
        return T.F16
    if profile == "q8_0":
        return T.Q8_0
    if profile == "q4_0":
        return T.Q4_0

    if profile == "q4_0-mixed":
        # Speed-first RDNA2 (gfx1031) mixed profile:
        # - Codebooks in F16 (0.000% RMSE, sparse host-eligible lookup like DSpark markov_w1)
        # - Outlier/sensitive projections (fc, *_conv_proj, selector_hidden) in Q8_0
        # - Bulk attention (Q/K/V/O) & FFN (gate/up/down) in Q4_0 (fastest wave32 DP4A kernel)
        if gguf_name in ("selector_predecessor.weight", "selector_successor.weight"):
            return T.F16
        if gguf_name in ("fc.weight", "selector_hidden.weight") or gguf_name.endswith("_conv_proj.weight"):
            return T.Q8_0
        return T.Q4_0

    if profile == "q4_k-mixed":
        # Quality-first RDNA2 (gfx1031) mixed profile:
        # - selector_hidden in BF16 (1.1 MiB, zero load conversion in PolyStrata, 0.000% RMSE)
        # - fc, *_conv_proj, and selector codebooks in Q8_0 (compact 104 MiB codebooks, <0.8% RMSE)
        # - Residual-writing projections (attn_output, ffn_down) in Q6_K (RDNA2 warps_cols kernel)
        # - Remaining attention (Q/K/V) & FFN (gate/up) in Q4_K (same 4.5 bpw as Q4_0, -17% RMSE)
        if gguf_name == "selector_hidden.weight":
            return T.BF16
        if gguf_name in ("fc.weight", "selector_predecessor.weight", "selector_successor.weight") or gguf_name.endswith("_conv_proj.weight"):
            return T.Q8_0
        if gguf_name.endswith(("attn_output.weight", "ffn_down.weight")):
            return T.Q6_K
        return T.Q4_K

    raise ValueError(f"Unknown outtype/profile: {profile!r}")


def read_safetensors_header(path: pathlib.Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr, 8 + hlen


def bf16_u16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def eval_error_numpy(ref_f32: np.ndarray, test_f32: np.ndarray) -> dict[str, float]:
    a = ref_f32.ravel().astype(np.float64)
    b = test_f32.ravel().astype(np.float64)
    d = np.abs(a - b)
    ref_sq = float(np.dot(a, a))
    test_sq = float(np.dot(b, b))
    diff_sq = float(np.dot(d, d))
    dot = float(np.dot(a, b))
    rel_rmse = float(np.sqrt(diff_sq / ref_sq)) if ref_sq > 0 else 0.0
    sqnr_db = float(10.0 * np.log10(ref_sq / diff_sq)) if (diff_sq > 0 and ref_sq > 0) else 999.0
    cos_sim = float(dot / (np.sqrt(ref_sq) * np.sqrt(test_sq))) if (ref_sq > 0 and test_sq > 0) else 1.0
    return {
        "rel_rmse": rel_rmse,
        "sqnr_db": sqnr_db,
        "max_abs": float(d.max()) if d.size else 0.0,
        "mean_abs": float(d.mean()) if d.size else 0.0,
        "cos_sim": cos_sim,
    }


def convert(
    model_dir: pathlib.Path,
    target_path: pathlib.Path,
    outfile: pathlib.Path,
    outtype: str,
    overrides: list[tuple[re.Pattern[str], gguf.GGMLQuantizationType]],
    bridge: GgmlQuantBridge,
    report_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    cfg_path = model_dir / "config.json"
    st_path = model_dir / "model.safetensors"
    with cfg_path.open("r", encoding="utf-8") as f:
        hparams = json.load(f)

    if "DFlash2DraftModel" not in hparams.get("architectures", []):
        raise ValueError(f"{cfg_path} is not a DFlash2DraftModel")

    dflash_cfg = hparams["dflash_config"]
    target_gguf = gguf.GGUFReader(str(target_path))

    # Validate target architecture and visit->layer mapping
    n_phys, loop_start, loop_end, loop_count, target_embd = target_layout(target_gguf)
    n_visits = n_phys + (loop_count - 1) * (loop_end - loop_start)
    layer_ids = list(dflash_cfg["target_layer_ids"])
    if layer_ids != sorted(set(layer_ids)) or layer_ids[0] < 0 or layer_ids[-1] != n_visits:
        raise ValueError(f"target_layer_ids {layer_ids} must be sorted, unique, and end at {n_visits}")
    if hparams.get("num_target_layers", n_visits) != n_visits or int(hparams["hidden_size"]) != target_embd:
        raise ValueError(
            f"target mismatch: num_target_layers={hparams.get('num_target_layers')} vs n_visits={n_visits}, "
            f"hidden_size={hparams['hidden_size']} vs target_embd={target_embd}"
        )
    phys_layers = [visit_to_layer(v, n_phys, loop_start, loop_end, loop_count) for v in layer_ids]
    logger.info("Target visits %s -> physical layers %s", layer_ids, phys_layers)

    # Validate vocabulary and mask token
    n_vocab_real = len(target_gguf.fields["tokenizer.ggml.tokens"].data)
    n_vocab_padded = int(hparams["vocab_size"])
    mask_token_id = int(dflash_cfg["mask_token_id"])
    if not (0 <= mask_token_id < n_vocab_real <= n_vocab_padded):
        raise ValueError(
            f"Invalid vocab/mask geometry: mask={mask_token_id}, real={n_vocab_real}, padded={n_vocab_padded}"
        )

    softcap_mode = dflash_cfg.get("selector_softcap_mode", "before_correction")
    if softcap_mode not in ("before_correction", "after_correction"):
        raise ValueError(f"Unknown selector_softcap_mode: {softcap_mode!r}")

    ftype = PROFILE_FILETYPE[outtype]
    outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(outfile), arch="dflash")

    # General metadata
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("Inkling-2-Mini-DFlash2")
    writer.add_basename("Inkling-2-Mini-DFlash2")
    writer.add_license("apache-2.0")
    writer.add_base_model_count(1)
    writer.add_base_model_name(0, "Inkling 2 Mini")
    writer.add_base_model_organization(0, "thinkingmachines")
    writer.add_base_model_repo_url(0, "https://huggingface.co/thinkingmachines/Inkling-2-Mini")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(ftype)

    # Architecture metadata (exact match to upstream convert_inkling2_dflash2_to_gguf.py + PR #33049)
    block_count = int(hparams["num_hidden_layers"])
    head_dim = int(hparams["head_dim"])
    writer.add_block_count(block_count)
    writer.add_context_length(int(hparams["max_position_embeddings"]))
    writer.add_embedding_length(int(hparams["hidden_size"]))
    writer.add_feed_forward_length(int(hparams["intermediate_size"]))
    writer.add_head_count(int(hparams["num_attention_heads"]))
    writer.add_head_count_kv(int(hparams["num_key_value_heads"]))
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(float(hparams["rms_norm_eps"]))
    writer.add_causal_attention(bool(hparams.get("is_causal", False)))

    rope_theta = float((hparams.get("rope_parameters") or {}).get("rope_theta", hparams.get("rope_theta", 10000.0)))
    writer.add_rope_freq_base(rope_theta)

    # Sliding window
    use_swa = bool(hparams.get("use_sliding_window", False))
    sliding_window = hparams.get("sliding_window")
    layer_types = hparams.get("layer_types") or []
    if use_swa and sliding_window:
        is_swa = [lt == "sliding_attention" for lt in layer_types] if layer_types else [True] * block_count
        writer.add_sliding_window(int(sliding_window))
        writer.add_sliding_window_pattern(is_swa)

    # DFlash2-specific parameters
    writer.add_uint32("dflash.block_size", int(dflash_cfg["block_size"]))
    writer.add_uint32("dflash.conv_kernel_size", int(dflash_cfg["conv_kernel_size"]))
    writer.add_uint32("dflash.conv_group_size", int(dflash_cfg["conv_group_size"]))
    writer.add_uint32("dflash.selector_rank", int(dflash_cfg["selector_rank"]))
    writer.add_uint32("dflash.selector_top_k", int(dflash_cfg["selector_top_k"]))

    softcap = float(dflash_cfg.get("final_logit_softcapping", 0.0))
    if softcap > 0.0:
        writer.add_final_logit_softcapping(softcap)
    if softcap_mode == "after_correction":
        writer.add_bool("dflash.selector_softcap_after_correction", True)

    writer.add_array("dflash.target_layers", phys_layers)
    writer.add_array("dflash.target_visits", layer_ids)

    # Copy target tokenizer.ggml.* fields verbatim (preserving exact GGUFValueTypes)
    copied_tok_keys = []
    for key, field in target_gguf.fields.items():
        if not key.startswith("tokenizer.ggml.") or key == "tokenizer.ggml.mask_token_id":
            continue
        vtype = field.types[0]
        sub_type = field.types[1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type)
        copied_tok_keys.append(key)
    writer.add_uint32("tokenizer.ggml.mask_token_id", mask_token_id)

    # Process all safetensors tensors
    st_hdr, data_start = read_safetensors_header(st_path)
    tensor_reports: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    type_bytes: dict[str, int] = {}
    type_params: dict[str, int] = {}

    with st_path.open("rb") as sf:
        for src_name in sorted(st_hdr.keys()):
            info = st_hdr[src_name]
            if info["dtype"] != "BF16":
                raise ValueError(f"Expected BF16 in source safetensors, got {info['dtype']} for {src_name}")
            gguf_name = map_tensor_name(src_name)
            src_shape = list(info["shape"])
            off0, off1 = info["data_offsets"]
            sf.seek(data_start + off0)
            raw_u16 = np.frombuffer(sf.read(off1 - off0), dtype=np.uint16).reshape(src_shape)

            # Slice codebooks from padded vocab (201024) to real vocab (200058)
            sliced = False
            if src_name in CODEBOOK_SRC_NAMES:
                if src_shape[0] != n_vocab_padded:
                    raise ValueError(f"Codebook {src_name} dim0={src_shape[0]} != padded vocab {n_vocab_padded}")
                raw_u16 = np.ascontiguousarray(raw_u16[:n_vocab_real])
                sliced = True

            eff_shape = list(raw_u16.shape)
            ref_f32 = np.ascontiguousarray(bf16_u16_to_f32(raw_u16))
            qtype = choose_qtype(gguf_name, len(eff_shape), outtype, overrides)

            # Encode tensor payload & evaluate exact dequantization error vs source BF16
            if qtype == T.F32:
                out_arr = ref_f32
                err = {"rel_rmse": 0.0, "sqnr_db": 999.0, "max_abs": 0.0, "mean_abs": 0.0, "cos_sim": 1.0}
                nbytes = int(out_arr.nbytes)
                writer.add_tensor(gguf_name, out_arr, raw_dtype=T.F32)
            elif qtype == T.BF16:
                # Preserve exact source BF16 bits without float32 re-rounding
                out_u8 = np.ascontiguousarray(raw_u16).view(np.uint8)
                err = {"rel_rmse": 0.0, "sqnr_db": 999.0, "max_abs": 0.0, "mean_abs": 0.0, "cos_sim": 1.0}
                nbytes = int(out_u8.nbytes)
                writer.add_tensor(gguf_name, out_u8, raw_dtype=T.BF16)
            elif qtype == T.F16:
                out_f16 = ref_f32.astype(np.float16)
                deq = out_f16.astype(np.float32)
                err = bridge.eval_error(ref_f32, deq) if bridge.available else eval_error_numpy(ref_f32, deq)
                nbytes = int(out_f16.nbytes)
                writer.add_tensor(gguf_name, out_f16, raw_dtype=T.F16)
            else:
                # Quantized 2D matrix
                if len(eff_shape) != 2:
                    raise ValueError(f"Cannot quantize {len(eff_shape)}D tensor {gguf_name} to {qtype.name}")
                if bridge.available:
                    out_u8 = bridge.quantize_2d(ref_f32, qtype)
                    deq = bridge.dequantize_2d(out_u8, qtype, eff_shape[1])
                    err = bridge.eval_error(ref_f32, deq)
                else:
                    out_u8 = gguf.quants.quantize(ref_f32, qtype)
                    deq = gguf.quants.dequantize(out_u8, qtype)
                    err = eval_error_numpy(ref_f32, deq)
                nbytes = int(out_u8.nbytes)
                writer.add_tensor(gguf_name, out_u8, raw_dtype=qtype)

            gguf_shape = list(reversed(eff_shape))
            nelems = int(np.prod(eff_shape))
            qname = qtype.name
            type_counts[qname] = type_counts.get(qname, 0) + 1
            type_bytes[qname] = type_bytes.get(qname, 0) + nbytes
            type_params[qname] = type_params.get(qname, 0) + nelems

            tensor_reports.append({
                "src_name": src_name,
                "gguf_name": gguf_name,
                "src_shape": src_shape,
                "gguf_shape": gguf_shape,
                "sliced": sliced,
                "qtype": qname,
                "qtype_id": int(qtype),
                "elements": nelems,
                "bytes": nbytes,
                **err,
            })

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Compute SHA-256 of written GGUF
    h = hashlib.sha256()
    with outfile.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    sha256 = h.hexdigest()
    file_size = outfile.stat().st_size
    elapsed = time.time() - t0

    summary = {
        "outfile": str(outfile),
        "outtype": outtype,
        "file_type_id": int(ftype),
        "file_type_name": ftype.name,
        "size_bytes": file_size,
        "sha256": sha256,
        "elapsed_seconds": round(elapsed, 2),
        "quantizer_backend": "ggml-c-libggml-base" if bridge.available else "gguf-py",
        "target_gguf": str(target_path),
        "target_visits": layer_ids,
        "target_layers": phys_layers,
        "vocab_size_real": n_vocab_real,
        "vocab_size_padded": n_vocab_padded,
        "mask_token_id": mask_token_id,
        "final_logit_softcapping": softcap,
        "selector_softcap_after_correction": softcap_mode == "after_correction",
        "copied_tokenizer_keys": copied_tok_keys,
        "tensor_count": len(tensor_reports),
        "total_elements": sum(type_params.values()),
        "total_tensor_bytes": sum(type_bytes.values()),
        "type_distribution": {
            k: {"tensors": type_counts[k], "elements": type_params[k], "bytes": type_bytes[k]}
            for k in sorted(type_counts.keys())
        },
        "tensors": tensor_reports,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "Wrote %s (%0.2f MiB, %d tensors, %s) in %.1fs | sha256=%s",
        outfile.name, file_size / (1 << 20), len(tensor_reports), type_counts, elapsed, sha256[:16],
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Inkling 2 Mini DFlash2 safetensors to GGUF")
    ap.add_argument("model", type=pathlib.Path, help="drafter directory containing config.json and model.safetensors")
    ap.add_argument("--target", type=pathlib.Path, required=True, help="target Inkling 2 GGUF for tokenizer & layout")
    ap.add_argument("--outfile", type=pathlib.Path, required=True, help="output GGUF path")
    ap.add_argument(
        "--outtype",
        choices=sorted(PROFILE_FILETYPE.keys()),
        default="bf16",
        help="quantization profile (default: bf16)",
    )
    ap.add_argument(
        "--tensor-type",
        action="append",
        default=[],
        metavar="REGEX=TYPE",
        help="override ggml type for matching GGUF tensor names (repeatable)",
    )
    ap.add_argument(
        "--ggml-bridge",
        type=pathlib.Path,
        default=None,
        help="path to libggml_qbridge.so for exact C ggml_quantize_chunk (required for K/I quants)",
    )
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--report", type=pathlib.Path, default=None, help="optional per-tensor JSON report output")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    overrides: list[tuple[re.Pattern[str], gguf.GGMLQuantizationType]] = []
    for spec in args.tensor_type:
        if "=" not in spec:
            raise ValueError(f"--tensor-type must be REGEX=TYPE, got {spec!r}")
        pat_s, qt_s = spec.rsplit("=", 1)
        qt = NAME_TO_QTYPE.get(qt_s.upper())
        if qt is None:
            raise ValueError(f"Unsupported ggml type {qt_s!r}; supported: {sorted(NAME_TO_QTYPE)}")
        overrides.append((re.compile(pat_s), qt))

    bridge = GgmlQuantBridge(args.ggml_bridge, nthreads=args.threads)
    if args.outtype == "q4_k-mixed" and not bridge.available:
        raise RuntimeError("q4_k-mixed requires --ggml-bridge pointing to libggml_qbridge.so")

    convert(args.model, args.target, args.outfile, args.outtype, overrides, bridge, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
