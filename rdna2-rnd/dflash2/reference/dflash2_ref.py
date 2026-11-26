#!/usr/bin/env python3
"""Exact Inkling-2-Mini DFlash2 Reference Oracle & Quantization Replay Harness.

Implements the verified DFlash2 forward pass from donor-spec.md:
  - Target tap fusion: [t_5 || t_23 || t_41 || t_48] -> fc -> RMSNorm(enc_output_norm, 1e-6)
  - Per-layer (0..4) KV injection: ctx_hidden -> attn_k/v (NO attn_norm, NO attn_conv) ->
    k_norm(1e-6) -> NeoX half-split RoPE (theta=10000, dim=128) -> 2048-row SWA ring
    (supports exact FP32 KV, simulated Q8_0 KV, Q8_0 K-only, or F16 KV)
  - Target-shared token_embd + token_embd_norm (1e-6) for [anchor, 7 x mask(200056)]
  - 5 layers: input_layernorm -> attention_conv(prepare) -> q/k/v + q/k_norm + NeoX RoPE ->
    non-causal SWA-2048 GQA attention -> o_proj -> attention_conv(finish) -> residual ->
    post_attention_layernorm -> mlp_conv(prepare) -> SwiGLU -> mlp_conv(finish) -> residual
  - output_norm -> target output.weight[:200058] (NO softcap) -> top-16 unary candidates ->
    selector_hidden -> sequential sample_path with selector_predecessor/successor ->
    post-correction softcap(30.0)

Supports loading:
  (a) Original HF safetensors (source/model.safetensors, exact BF16 reference)
  (b) Any DFlash2 GGUF (BF16, Q8_0, Q4_0, Q4_0-mixed, Q4_K-mixed, custom) via libggml_qbridge.so
  (c) Optional per-tensor in-memory quantization overrides (--quant-override REGEX=TYPE)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import pathlib
import re
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

import gguf  # noqa: E402

DEFAULT_RECON_DIR = pathlib.Path.home() / ".cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon"
DEFAULT_BRIDGE = DEFAULT_RECON_DIR / "libggml_qbridge.so"
DEFAULT_DFLASH2_DIR = pathlib.Path.home() / ".lmstudio/models/unsloth/Inkling-2-Mini/DFlash2"
DEFAULT_TARGET_GGUF = DEFAULT_DFLASH2_DIR.parent / "Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf"

T = gguf.GGMLQuantizationType
NAME_TO_QTYPE: dict[str, int] = {
    "F32": int(T.F32),
    "F16": int(T.F16),
    "BF16": int(T.BF16),
    "Q8_0": int(T.Q8_0),
    "Q6_K": int(T.Q6_K),
    "Q5_K": int(T.Q5_K),
    "Q5_0": int(T.Q5_0),
    "Q5_1": int(T.Q5_1),
    "Q4_K": int(T.Q4_K),
    "Q4_0": int(T.Q4_0),
    "Q4_1": int(T.Q4_1),
    "IQ4_NL": int(T.IQ4_NL),
    "IQ4_XS": int(T.IQ4_XS),
}

STATIC_TENSOR_MAP = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc_output_norm.weight",
    "norm.weight": "output_norm.weight",
    "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
    "candidate_selector.successor_codebook": "selector_successor.weight",
}

LAYER_TENSOR_MAP = {
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

# Constants per donor-spec.md
HIDDEN_SIZE = 2304
NUM_LAYERS = 5
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS
FFN_SIZE = 6912
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
SLIDING_WINDOW = 2048
BLOCK_SIZE = 8
MASK_TOKEN_ID = 200056
VOCAB_VALID = 200058
CONV_GROUPS = 144
CONV_GROUP_SIZE = 16
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16
SOFTCAP = 30.0


def map_safetensors_name(src_name: str) -> str:
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


class QBridge:
    """ctypes wrapper around libggml_qbridge.so."""

    def __init__(self, lib_path: pathlib.Path = DEFAULT_BRIDGE, nthreads: int = 8):
        self.lib_path = pathlib.Path(lib_path)
        self.nthreads = max(1, int(nthreads))
        lib = ctypes.CDLL(str(self.lib_path))

        lib.qbridge_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.qbridge_row_size.restype = ctypes.c_size_t

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

        lib.qbridge_dequantize_rows.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
        ]
        lib.qbridge_dequantize_rows.restype = None

        lib.qbridge_eval_error.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
        ]
        lib.qbridge_eval_error.restype = None

        lib.qbridge_q6k_matvec_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ]
        lib.qbridge_q6k_matvec_f32.restype = None

        lib.qbridge_q6k_matvec_q8k.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ]
        lib.qbridge_q6k_matvec_q8k.restype = None

        lib.qbridge_matvec_f32.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ]
        lib.qbridge_matvec_f32.restype = None

        lib.qbridge_roundtrip.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ]
        lib.qbridge_roundtrip.restype = None

        self.lib = lib

    def row_size(self, qtype: int, ncols: int) -> int:
        block, _ = gguf.GGML_QUANT_SIZES[T(qtype)]
        if ncols <= 0 or ncols % block:
            raise ValueError(f"{T(qtype).name} requires a multiple of {block} columns, got {ncols}")
        return int(self.lib.qbridge_row_size(int(qtype), int(ncols)))

    def quantize_2d(self, arr_f32: np.ndarray, qtype: int) -> np.ndarray:
        a = np.ascontiguousarray(arr_f32, dtype=np.float32)
        nrows, ncols = a.shape
        rb = self.row_size(qtype, ncols)
        out = np.empty((nrows, rb), dtype=np.uint8)
        w = int(self.lib.qbridge_quantize(int(qtype), a.ctypes.data, out.ctypes.data, nrows, ncols, self.nthreads))
        if w != nrows * rb:
            raise RuntimeError(f"qbridge_quantize wrote {w} != {nrows * rb}")
        return out

    def dequantize_2d(self, raw_u8: np.ndarray, qtype: int, nrows: int, ncols: int) -> np.ndarray:
        r = np.ascontiguousarray(raw_u8.view(np.uint8))
        if nrows * self.row_size(qtype, ncols) > r.nbytes:
            raise ValueError("truncated packed matrix")
        out = np.empty((nrows, ncols), dtype=np.float32)
        self.lib.qbridge_dequantize(int(qtype), r.ctypes.data, out.ctypes.data, nrows, ncols, self.nthreads)
        return out

    def dequantize_rows(self, raw_u8: np.ndarray, qtype: int, row_ids: np.ndarray, ncols: int) -> np.ndarray:
        ids = np.ascontiguousarray(row_ids, dtype=np.int32)
        rows = raw_u8.nbytes // self.row_size(qtype, ncols)
        if np.any(ids < 0) or np.any(ids >= rows):
            raise ValueError("packed matrix row outside storage")
        out = np.empty((ids.size, ncols), dtype=np.float32)
        self.lib.qbridge_dequantize_rows(int(qtype), raw_u8.ctypes.data, ids.ctypes.data, ids.size, out.ctypes.data, ncols)
        return out

    def eval_error(self, ref_f32: np.ndarray, test_f32: np.ndarray) -> dict[str, float]:
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

    def matvec(self, qtype: int, raw_u8: np.ndarray, x_f32: np.ndarray, nrows: int, ncols: int, mode: str = "f32") -> np.ndarray:
        x = np.ascontiguousarray(x_f32, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        m = x.shape[0]
        if x.shape != (m, ncols) or nrows * self.row_size(qtype, ncols) > raw_u8.nbytes:
            raise ValueError("incorrect head matrix or activation dimensions")
        out = np.empty((m, nrows), dtype=np.float32)
        if int(qtype) == int(T.Q6_K) and mode == "q8k":
            self.lib.qbridge_q6k_matvec_q8k(raw_u8.ctypes.data, x.ctypes.data, out.ctypes.data, nrows, ncols, m, self.nthreads)
        elif int(qtype) == int(T.Q6_K):
            self.lib.qbridge_q6k_matvec_f32(raw_u8.ctypes.data, x.ctypes.data, out.ctypes.data, nrows, ncols, m, self.nthreads)
        else:
            self.lib.qbridge_matvec_f32(int(qtype), raw_u8.ctypes.data, x.ctypes.data, out.ctypes.data, nrows, ncols, m, self.nthreads)
        return out

    def roundtrip(self, arr_f32: np.ndarray, qtype: int, nthreads: int | None = None) -> np.ndarray:
        a = np.ascontiguousarray(arr_f32, dtype=np.float32)
        orig_shape = a.shape
        ncols = orig_shape[-1]
        nrows = a.size // ncols
        self.row_size(qtype, ncols)
        out = np.empty_like(a)
        nt = self.nthreads if nthreads is None else nthreads
        self.lib.qbridge_roundtrip(int(qtype), a.ctypes.data, out.ctypes.data, nrows, ncols, nt)
        return out.reshape(orig_shape)


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    """Exact FP32 RMSNorm along the last axis."""
    mean_sq = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    inv_rms = (1.0 / np.sqrt(mean_sq + np.float32(eps))).astype(np.float32)
    return (x * inv_rms * w).astype(np.float32)


def rope_neox(x: np.ndarray, positions: np.ndarray, theta: float = ROPE_THETA) -> np.ndarray:
    """NeoX half-split RoPE on x of shape [N, H, 128] at absolute positions [N]."""
    d = x.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / float(d)))
    angles = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)[:, None, :]
    sin = np.sin(angles).astype(np.float32)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return out


def grouped_dynamic_causal_conv(u: np.ndarray, delta_side: np.ndarray, base_side: np.ndarray) -> np.ndarray:
    """GroupedDynamicCausalConv per donor-spec.md §6.

    Args:
      u: [B, 2304]
      delta_side: [B, 2, 144]  (tap 0 = current t, tap 1 = lag t-1)
      base_side: [2, 2304]     (tap 0 = current t, tap 1 = lag t-1)
    """
    # Expand group dimension [B, 2, 144] -> [B, 2, 2304]
    dyn = np.repeat(delta_side, CONV_GROUP_SIZE, axis=-1)
    w0 = base_side[0] + dyn[..., 0, :]
    w1 = base_side[1] + dyn[..., 1, :]
    y = w0 * u
    y[..., 1:, :] += w1[..., 1:, :] * u[..., :-1, :]
    return y


def silu(x: np.ndarray) -> np.ndarray:
    return x * np.exp(-np.logaddexp(np.float32(0), -x))


def linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Flatten independent leading dimensions into one BLAS call."""
    return (x.reshape(-1, x.shape[-1]) @ w.T).reshape(*x.shape[:-1], w.shape[0])


class TargetSharedWeights:
    """Zero-copy mmap access to Inkling-2-Mini target token_embd, token_embd_norm, and output.weight."""

    def __init__(self, target_gguf: pathlib.Path, bridge: QBridge, lm_head_mode: str = "f32"):
        self.path = pathlib.Path(target_gguf)
        self.bridge = bridge
        self.lm_head_mode = lm_head_mode
        self.reader = gguf.GGUFReader(str(self.path))

        tmap = {t.name: t for t in self.reader.tensors}
        self.t_embd = tmap["token_embd.weight"]
        self.t_embd_norm = tmap["token_embd_norm.weight"]
        self.t_output = tmap["output.weight"]

        self.embd_qtype = int(self.t_embd.tensor_type)
        self.embd_raw = self.t_embd.data  # mmap uint8 view
        self.embd_norm = bridge.dequantize_2d(
            self.t_embd_norm.data, int(self.t_embd_norm.tensor_type), 1, HIDDEN_SIZE
        )[0]

        self.output_qtype = int(self.t_output.tensor_type)
        self.output_raw = self.t_output.data  # mmap uint8 view (first 200058 rows are contiguous at offset 0!)
        self.vocab_valid = VOCAB_VALID
        self.output_f32 = None

        # Precompute normalized mask embedding once
        self._mask_emb = self.embed_tokens(np.array([MASK_TOKEN_ID], dtype=np.int32))[0]

    def embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        if np.any(ids < 0) or np.any(ids >= VOCAB_VALID):
            raise ValueError("token ID outside Inkling's valid vocabulary")
        raw = self.bridge.dequantize_rows(self.embd_raw, self.embd_qtype, ids, HIDDEN_SIZE)
        return rms_norm(raw, self.embd_norm, RMS_EPS)

    def embed_block(self, anchor_id: int) -> np.ndarray:
        out = np.empty((BLOCK_SIZE, HIDDEN_SIZE), dtype=np.float32)
        out[0] = self.embed_tokens(np.array([anchor_id], dtype=np.int32))[0]
        out[1:] = self._mask_emb[None, :]
        return out

    def lm_head(self, h_draft: np.ndarray) -> np.ndarray:
        """Compute raw (pre-softcap) logits over valid vocab [0..200057]."""
        if self.lm_head_mode == "blas":
            if self.output_f32 is None:
                self.output_f32 = self.bridge.dequantize_2d(
                    self.output_raw, self.output_qtype, VOCAB_VALID, HIDDEN_SIZE
                )
            return linear(h_draft, self.output_f32)
        return self.bridge.matvec(
            self.output_qtype,
            self.output_raw,
            h_draft,
            nrows=self.vocab_valid,
            ncols=HIDDEN_SIZE,
            mode=self.lm_head_mode,
        )


class CodebookTable:
    """Sparse row-lookup wrapper for a [200058, 256] selector codebook (raw mmap or F32)."""

    def __init__(self, raw_u8: np.ndarray, qtype: int, bridge: QBridge):
        self.raw_u8 = raw_u8
        self.qtype = int(qtype)
        self.bridge = bridge

    def gather(self, token_ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        return self.bridge.dequantize_rows(self.raw_u8, self.qtype, ids, SELECTOR_RANK)


class DFlash2Weights:
    """Loads all 81 DFlash2 tensors from safetensors or GGUF, with optional per-tensor quant overrides."""

    def __init__(
        self,
        source_path: pathlib.Path,
        bridge: QBridge,
        quant_overrides: list[tuple[re.Pattern[str], int]] | None = None,
    ):
        self.source_path = pathlib.Path(source_path)
        self.bridge = bridge
        self.quant_overrides = quant_overrides or []
        self.tensors: dict[str, np.ndarray] = {}
        self.tensor_qtypes: dict[str, str] = {}
        self.pred_cb: CodebookTable
        self.succ_cb: CodebookTable

        if self.source_path.is_dir():
            self._load_safetensors(self.source_path / "model.safetensors")
        elif self.source_path.suffix == ".safetensors":
            self._load_safetensors(self.source_path)
        else:
            self._load_gguf(self.source_path)

    def _override_qtype(self, gguf_name: str) -> int | None:
        for pat, qt in self.quant_overrides:
            if pat.search(gguf_name):
                return qt
        return None

    def _apply_override_dense(self, gguf_name: str, arr_f32: np.ndarray) -> np.ndarray:
        qt = self._override_qtype(gguf_name)
        if qt is None:
            return arr_f32
        self.tensor_qtypes[gguf_name] = f"{self.tensor_qtypes[gguf_name]}->override({gguf.GGMLQuantizationType(qt).name})"
        return self.bridge.roundtrip(arr_f32, qt)

    def _apply_override_codebook(self, gguf_name: str, raw_u8: np.ndarray, qtype: int) -> CodebookTable:
        qt = self._override_qtype(gguf_name)
        if qt is None:
            return CodebookTable(raw_u8, qtype, self.bridge)
        self.tensor_qtypes[gguf_name] = f"{self.tensor_qtypes[gguf_name]}->override({gguf.GGMLQuantizationType(qt).name})"
        # Dequantize once, re-quantize to target qt, keep quantized!
        f32 = self.bridge.dequantize_2d(raw_u8, qtype, VOCAB_VALID, SELECTOR_RANK)
        new_raw = self.bridge.quantize_2d(f32, qt)
        return CodebookTable(new_raw, qt, self.bridge)

    def _load_safetensors(self, st_path: pathlib.Path) -> None:
        with st_path.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen).decode("utf-8"))
            hdr.pop("__metadata__", None)
            data_start = 8 + hlen

            for src_name in sorted(hdr.keys()):
                info = hdr[src_name]
                if info["dtype"] != "BF16":
                    raise ValueError(f"Expected BF16 in {st_path}, got {info['dtype']} for {src_name}")
                gguf_name = map_safetensors_name(src_name)
                shape = tuple(info["shape"])
                off0, off1 = info["data_offsets"]
                f.seek(data_start + off0)
                raw_u16 = np.frombuffer(f.read(off1 - off0), dtype=np.uint16).reshape(shape)
                self.tensor_qtypes[gguf_name] = "BF16"

                if gguf_name in ("selector_predecessor.weight", "selector_successor.weight"):
                    raw_u8 = np.ascontiguousarray(raw_u16[:VOCAB_VALID]).view(np.uint8)
                    cb = self._apply_override_codebook(gguf_name, raw_u8, int(T.BF16))
                    if gguf_name == "selector_predecessor.weight":
                        self.pred_cb = cb
                    else:
                        self.succ_cb = cb
                else:
                    arr_f32 = np.ascontiguousarray((raw_u16.astype(np.uint32) << 16).view(np.float32))
                    arr_f32 = self._apply_override_dense(gguf_name, arr_f32)
                    self.tensors[gguf_name] = arr_f32

    def _load_gguf(self, gguf_path: pathlib.Path) -> None:
        self._gguf_reader = gguf.GGUFReader(str(gguf_path))
        for t in self._gguf_reader.tensors:
            name = t.name
            qtype = int(t.tensor_type)
            py_shape = tuple(int(x) for x in reversed(t.shape))
            self.tensor_qtypes[name] = t.tensor_type.name

            if name in ("selector_predecessor.weight", "selector_successor.weight"):
                assert py_shape == (VOCAB_VALID, SELECTOR_RANK), f"Unexpected codebook shape {py_shape}"
                cb = self._apply_override_codebook(name, t.data, qtype)
                if name == "selector_predecessor.weight":
                    self.pred_cb = cb
                else:
                    self.succ_cb = cb
            else:
                if qtype == int(T.F32):
                    arr_f32 = np.ascontiguousarray(t.data.view(np.float32).reshape(py_shape))
                elif len(py_shape) == 2:
                    arr_f32 = self.bridge.dequantize_2d(t.data, qtype, py_shape[0], py_shape[1])
                else:
                    ncols = py_shape[-1]
                    nrows = int(np.prod(py_shape[:-1])) if len(py_shape) > 1 else 1
                    arr_f32 = self.bridge.dequantize_2d(t.data, qtype, nrows, ncols).reshape(py_shape)
                arr_f32 = self._apply_override_dense(name, arr_f32)
                self.tensors[name] = arr_f32


class DFlash2Model:
    """Exact stateful DFlash2 drafter with 2048-row SWA ring KV cache."""

    def __init__(
        self,
        weights: DFlash2Weights,
        target: TargetSharedWeights,
        kv_quant: str = "fp32",
    ):
        self.w = weights
        self.target = target
        self.bridge = weights.bridge
        self.kv_quant = kv_quant
        self.reset_cache()

    def reset_cache(self) -> None:
        self.k_ring = np.zeros((NUM_LAYERS, SLIDING_WINDOW, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32)
        self.v_ring = np.zeros((NUM_LAYERS, SLIDING_WINDOW, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32)
        self.pos_ring = np.full((SLIDING_WINDOW,), -1, dtype=np.int64)
        self.ctx_len = 0

    def _quantize_kv(self, arr: np.ndarray, is_k: bool) -> np.ndarray:
        if self.kv_quant == "fp32":
            return arr
        if self.kv_quant == "f16":
            return arr.astype(np.float16).astype(np.float32)
        if self.kv_quant == "q8_0":
            return self.bridge.roundtrip(arr, int(T.Q8_0), nthreads=1)
        if self.kv_quant == "q8_0_k":
            return self.bridge.roundtrip(arr, int(T.Q8_0), nthreads=1) if is_k else arr
        raise ValueError(f"Unknown kv_quant: {self.kv_quant!r}")

    def inject(self, taps: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Fuse target taps [N, 4, 2304] (or [N, 9216]) and inject K/V into 2048-row SWA ring."""
        t = self.w.tensors
        pos = np.asarray(positions, dtype=np.int64)
        n = pos.shape[0]
        f = np.ascontiguousarray(taps, dtype=np.float32).reshape(n, 4 * HIDDEN_SIZE)

        # c_p = RMSNorm(f_p @ fc^T, enc_output_norm, 1e-6)
        ctx_hidden = rms_norm(f @ t["fc.weight"].T, t["enc_output_norm.weight"], RMS_EPS)

        # Keep the last write to each slot explicitly (fancy assignment with
        # repeated indices has no defined write order in NumPy).
        pos = pos[-SLIDING_WINDOW:]
        stored = ctx_hidden[-SLIDING_WINDOW:]
        slots = pos % SLIDING_WINDOW
        self.pos_ring[slots] = pos

        for i in range(NUM_LAYERS):
            # NO attn_norm, NO attn_conv on ctx_hidden!
            k = (stored @ t[f"blk.{i}.attn_k.weight"].T).reshape(pos.size, NUM_KV_HEADS, HEAD_DIM)
            v = (stored @ t[f"blk.{i}.attn_v.weight"].T).reshape(pos.size, NUM_KV_HEADS, HEAD_DIM)
            k = rms_norm(k, t[f"blk.{i}.attn_k_norm.weight"], RMS_EPS)
            k = rope_neox(k, pos, ROPE_THETA)
            k = self._quantize_kv(k, is_k=True)
            v = self._quantize_kv(v, is_k=False)
            self.k_ring[i, slots] = k
            self.v_ring[i, slots] = v

        self.ctx_len = int(pos[-1]) + 1 if n > 0 else self.ctx_len
        return ctx_hidden

    def draft_batch(self, anchors: np.ndarray, starts: np.ndarray) -> list[dict[str, Any]]:
        """Greedy replay of independent anchors whose committed prefixes are in the ring.

        A batch may precompute true context KV beyond its first anchor. The
        per-anchor upper bound excludes all those rows from its attention.
        Each anchor has its own eight-row noise block and convolution history.
        """
        if not 0 < len(starts) == len(anchors):
            raise ValueError("one position per anchor required")
        count = len(starts)
        t = self.w.tensors
        x = np.broadcast_to(self.target._mask_emb, (count, BLOCK_SIZE, HIDDEN_SIZE)).copy()
        x[:, 0] = self.target.embed_tokens(anchors)
        block_pos = starts[:, None] + np.arange(BLOCK_SIZE)
        ctx_pos = np.arange(max(0, starts.min() - SLIDING_WINDOW + 1), starts.max())
        slots = ctx_pos % SLIDING_WINDOW
        if not np.array_equal(self.pos_ring[slots], ctx_pos):
            raise ValueError("draft_batch needs all context positions in the ring")
        ctx_mask = (ctx_pos[None, None, :] < block_pos[..., None] - SLIDING_WINDOW + 1) | (
            ctx_pos[None, None, :] >= starts[:, None, None]
        )
        ctx_mask = ctx_mask.repeat(GQA_RATIO, axis=1)[:, None]
        scale = np.float32(1.0 / math.sqrt(HEAD_DIM))

        def rotate(z: np.ndarray, w: np.ndarray, nheads: int) -> np.ndarray:
            z = rms_norm(z.reshape(-1, nheads, HEAD_DIM), w)
            return rope_neox(z, block_pos.ravel()).reshape(count, BLOCK_SIZE, nheads, HEAD_DIM)

        for i in range(NUM_LAYERS):
            p = f"blk.{i}."
            r = x
            h = rms_norm(x, t[p + "attn_norm.weight"])
            da = linear(h, t[p + "attn_conv_proj.weight"]).reshape(count, BLOCK_SIZE, 2, 2, CONV_GROUPS)
            h = grouped_dynamic_causal_conv(h, da[:, :, 0], t[p + "attn_conv_base"][0])
            q = rotate(linear(h, t[p + "attn_q.weight"]), t[p + "attn_q_norm.weight"], NUM_Q_HEADS)
            kb = rotate(linear(h, t[p + "attn_k.weight"]), t[p + "attn_k_norm.weight"], NUM_KV_HEADS)
            vb = linear(h, t[p + "attn_v.weight"]).reshape(count, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)

            q = q.reshape(count, BLOCK_SIZE, NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)
            q = q.transpose(0, 2, 1, 3, 4).reshape(count, NUM_KV_HEADS, BLOCK_SIZE * GQA_RATIO, HEAD_DIM)
            block_k, block_v = kb.swapaxes(1, 2), vb.swapaxes(1, 2)
            k_ctx = np.ascontiguousarray(self.k_ring[i, slots].transpose(1, 2, 0))
            v_ctx = np.ascontiguousarray(self.v_ring[i, slots].transpose(1, 0, 2))
            scores_ctx = q @ k_ctx * scale
            scores_ctx = np.where(ctx_mask, -np.inf, scores_ctx)
            scores_blk = q @ block_k.swapaxes(-1, -2) * scale
            scores = np.concatenate((scores_ctx, scores_blk), axis=-1)
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            attn = probs[..., :ctx_pos.size] @ v_ctx + probs[..., ctx_pos.size:] @ block_v
            attn = attn.reshape(count, NUM_KV_HEADS, BLOCK_SIZE, GQA_RATIO, HEAD_DIM).transpose(0, 2, 1, 3, 4)
            h = linear(attn.reshape(count, BLOCK_SIZE, -1), t[p + "attn_output.weight"])
            x = r + grouped_dynamic_causal_conv(h, da[:, :, 1], t[p + "attn_conv_base"][1])

            r = x
            h = rms_norm(x, t[p + "ffn_norm.weight"])
            dm = linear(h, t[p + "ffn_conv_proj.weight"]).reshape(count, BLOCK_SIZE, 2, 2, CONV_GROUPS)
            h = grouped_dynamic_causal_conv(h, dm[:, :, 0], t[p + "ffn_conv_base"][0])
            h = linear(silu(linear(h, t[p + "ffn_gate.weight"])) * linear(h, t[p + "ffn_up.weight"]),
                       t[p + "ffn_down.weight"])
            x = r + grouped_dynamic_causal_conv(h, dm[:, :, 1], t[p + "ffn_conv_base"][1])

        h = rms_norm(x, t["output_norm.weight"])[:, 1:]
        logits = self.target.lm_head(h.reshape(-1, HIDDEN_SIZE)).reshape(count, BLOCK_SIZE - 1, -1)
        return [self.select(int(anchor), hh, ll) for anchor, hh, ll in zip(anchors, h, logits, strict=True)]

    def draft_block(
        self,
        anchor_id: int,
        start_pos: int,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, Any]:
        """Run 1 DFlash2 draft block at absolute positions [start_pos .. start_pos + 7]."""
        t = self.w.tensors
        inter: dict[str, np.ndarray] = {}

        # Step 0: Anchor + 7x mask embedding with target token_embd_norm
        x = self.target.embed_block(anchor_id)  # [8, 2304]
        if return_intermediates:
            inter["emb_normed"] = x.copy()

        block_pos = np.arange(start_pos, start_pos + BLOCK_SIZE, dtype=np.int64)

        # Gather valid context positions in [max(0, start_pos - 2047), start_pos - 1]
        lo_min = max(0, start_pos - (SLIDING_WINDOW - 1))
        if start_pos > lo_min:
            ctx_positions = np.arange(lo_min, start_pos, dtype=np.int64)
            ctx_slots = ctx_positions % SLIDING_WINDOW
            # Sanity check ring integrity
            if not np.array_equal(self.pos_ring[ctx_slots], ctx_positions):
                raise RuntimeError(
                    f"SWA ring mismatch at start_pos={start_pos}: expected [{lo_min}..{start_pos-1}]"
                )
        else:
            ctx_positions = np.empty((0,), dtype=np.int64)
            ctx_slots = np.empty((0,), dtype=np.int64)

        pos_all = np.concatenate([ctx_positions, block_pos])  # [L_all]
        # Exact SWA-2048 non-causal visibility mask [8, L_all]:
        # visible(p_q, p_k) <=> p_k >= max(0, p_q - 2047)
        lo_per_q = np.maximum(0, block_pos - (SLIDING_WINDOW - 1))[:, None]  # [8, 1]
        mask_invalid = pos_all[None, :] < lo_per_q  # [8, L_all] bool
        if return_intermediates:
            inter["block_pos"] = block_pos
            inter["ctx_positions"] = ctx_positions
            inter["ctx_k"] = self.k_ring[:, ctx_slots]
            inter["ctx_v"] = self.v_ring[:, ctx_slots]
            inter["attn_visible"] = ~mask_invalid

        scale = np.float32(1.0 / math.sqrt(HEAD_DIM))

        for i in range(NUM_LAYERS):
            # --- Attention sublayer ---
            r = x
            h = rms_norm(x, t[f"blk.{i}.attn_norm.weight"], RMS_EPS)
            da = (h @ t[f"blk.{i}.attn_conv_proj.weight"].T).reshape(BLOCK_SIZE, 2, 2, CONV_GROUPS)
            h = grouped_dynamic_causal_conv(h, da[:, 0, :, :], t[f"blk.{i}.attn_conv_base"][0])
            if return_intermediates:
                inter[f"layer_{i}_attn_delta"] = da
                inter[f"layer_{i}_attn_preconv"] = h

            q = (h @ t[f"blk.{i}.attn_q.weight"].T).reshape(BLOCK_SIZE, NUM_Q_HEADS, HEAD_DIM)
            kb = (h @ t[f"blk.{i}.attn_k.weight"].T).reshape(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
            vb = (h @ t[f"blk.{i}.attn_v.weight"].T).reshape(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)

            q = rope_neox(rms_norm(q, t[f"blk.{i}.attn_q_norm.weight"], RMS_EPS), block_pos, ROPE_THETA)
            kb = rope_neox(rms_norm(kb, t[f"blk.{i}.attn_k_norm.weight"], RMS_EPS), block_pos, ROPE_THETA)
            if return_intermediates:
                inter[f"layer_{i}_block_q"] = q
                inter[f"layer_{i}_block_k"] = kb
                inter[f"layer_{i}_block_v"] = vb

            if ctx_slots.size > 0:
                k_all = np.concatenate([self.k_ring[i, ctx_slots], kb], axis=0)  # [L_all, 8, 128]
                v_all = np.concatenate([self.v_ring[i, ctx_slots], vb], axis=0)  # [L_all, 8, 128]
            else:
                k_all = kb
                v_all = vb

            # GQA attention: q is [8, 8_kv, 4_grp, 128], k_all is [L_all, 8_kv, 128]
            q_g = q.reshape(BLOCK_SIZE, NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)
            # scores: [8_q, 8_kv, 4_grp, L_all]
            scores = np.einsum("bhgd,khd->bhgk", q_g, k_all, optimize=True).astype(np.float32) * scale
            if np.any(mask_invalid):
                scores = np.where(mask_invalid[:, None, None, :], np.float32(-1e30), scores)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_s = np.exp(scores - scores_max, dtype=np.float32)
            probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True, dtype=np.float32)
            attn_out = np.einsum("bhgk,khd->bhgd", probs, v_all, optimize=True).astype(np.float32)
            attn_out = attn_out.reshape(BLOCK_SIZE, NUM_Q_HEADS * HEAD_DIM)
            if return_intermediates:
                inter[f"layer_{i}_attn_out"] = attn_out

            h = attn_out @ t[f"blk.{i}.attn_output.weight"].T
            h = grouped_dynamic_causal_conv(h, da[:, 1, :, :], t[f"blk.{i}.attn_conv_base"][1])
            x = r + h

            # --- MLP sublayer ---
            r = x
            h = rms_norm(x, t[f"blk.{i}.ffn_norm.weight"], RMS_EPS)
            dm = (h @ t[f"blk.{i}.ffn_conv_proj.weight"].T).reshape(BLOCK_SIZE, 2, 2, CONV_GROUPS)
            h = grouped_dynamic_causal_conv(h, dm[:, 0, :, :], t[f"blk.{i}.ffn_conv_base"][0])
            if return_intermediates:
                inter[f"layer_{i}_attn_residual"] = x
                inter[f"layer_{i}_ffn_delta"] = dm
                inter[f"layer_{i}_ffn_preconv"] = h
            gate = h @ t[f"blk.{i}.ffn_gate.weight"].T
            up = h @ t[f"blk.{i}.ffn_up.weight"].T
            h = (silu(gate) * up) @ t[f"blk.{i}.ffn_down.weight"].T
            h = grouped_dynamic_causal_conv(h, dm[:, 1, :, :], t[f"blk.{i}.ffn_conv_base"][1])
            x = r + h

            if return_intermediates:
                inter[f"layer_{i}_out"] = x.copy()

        # Final RMSNorm
        h_out = rms_norm(x, t["output_norm.weight"], RMS_EPS)  # [8, 2304]
        h_draft = h_out[1:BLOCK_SIZE]  # [7, 2304]
        if return_intermediates:
            inter["h_out"] = h_out.copy()

        # Shared target lm_head over real vocab [0..200057], NO softcap yet!
        u_raw = self.target.lm_head(h_draft)  # [7, 200058]
        res = self.select(anchor_id, h_draft, u_raw, temperature, rng)
        if return_intermediates:
            inter["raw_logits"] = u_raw
            res["intermediates"] = inter
        return res

    def select(self, anchor_id: int, h_draft: np.ndarray, u_raw: np.ndarray,
               temperature: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        # Top-K (K=16) unary candidates per draft position
        cand = np.empty((BLOCK_SIZE - 1, SELECTOR_TOP_K), dtype=np.int32)
        unary = np.empty((BLOCK_SIZE - 1, SELECTOR_TOP_K), dtype=np.float32)
        for pos_idx in range(BLOCK_SIZE - 1):
            row = u_raw[pos_idx]
            idx = np.argpartition(row, -SELECTOR_TOP_K)[-SELECTOR_TOP_K:]
            boundary = row[idx].min()
            if np.count_nonzero(row == boundary) > np.count_nonzero(row[idx] == boundary):
                high = idx[row[idx] > boundary]
                idx = np.concatenate((high, np.flatnonzero(row == boundary)[:SELECTOR_TOP_K - len(high)]))
            # Sort descending by score, tie-break ascending by token id
            vals = row[idx]
            order = np.lexsort((idx, -vals))
            idx_s = idx[order]
            cand[pos_idx] = idx_s.astype(np.int32)
            unary[pos_idx] = row[idx_s]

        # Context gate g_t = h_draft @ selector_hidden^T  [7, 256]
        g = linear(h_draft, self.w.tensors["selector_hidden.weight"])

        # Sequential path walk with post-correction softcap(30.0)
        draft_tokens = np.empty((BLOCK_SIZE - 1,), dtype=np.int32)
        unary_argmax = cand[:, 0].copy()
        sel_scores = np.empty((BLOCK_SIZE - 1, SELECTOR_TOP_K), dtype=np.float32)
        sel_probs = np.empty((BLOCK_SIZE - 1, SELECTOR_TOP_K), dtype=np.float32)
        corrections = np.empty_like(sel_scores)
        predecessors = np.empty(BLOCK_SIZE - 1, dtype=np.int32)

        prev = int(anchor_id)
        for pos_idx in range(BLOCK_SIZE - 1):
            predecessors[pos_idx] = prev
            a_prev = self.w.pred_cb.gather(np.array([prev], dtype=np.int32))[0]  # [256]
            b_cands = self.w.succ_cb.gather(cand[pos_idx])  # [16, 256]
            # corr = sum_r (A[prev, r] * g[pos_idx, r]) * B[c_k, r]  (NO 1/sqrt(R)!)
            ag = a_prev * g[pos_idx]
            corr = (b_cands @ ag).astype(np.float32)
            corrections[pos_idx] = corr
            sc = (SOFTCAP * np.tanh((unary[pos_idx] + corr) / SOFTCAP)).astype(np.float32)
            sel_scores[pos_idx] = sc

            if temperature <= 0.0:
                k_star = int(np.argmax(sc))
                # Record T=1 softmax probs for diagnostic confidence / comparison
                sm = np.exp(sc - np.max(sc))
                sel_probs[pos_idx] = sm / np.sum(sm)
            else:
                scaled = sc / np.float32(temperature)
                sm = np.exp(scaled - np.max(scaled))
                probs_t = sm / np.sum(sm)
                sel_probs[pos_idx] = probs_t
                if rng is None:
                    k_star = int(np.argmax(probs_t))
                else:
                    # Deterministic ascending-token-id CDF walk (matching llama.cpp)
                    id_ord = np.argsort(cand[pos_idx])
                    cdf = np.cumsum(probs_t[id_ord], dtype=np.float64)
                    u_rand = float(rng.random()) * float(cdf[-1])
                    pick = int(np.searchsorted(cdf, u_rand, side="right"))
                    pick = min(pick, SELECTOR_TOP_K - 1)
                    k_star = int(id_ord[pick])

            draft_tokens[pos_idx] = int(cand[pos_idx, k_star])
            prev = int(draft_tokens[pos_idx])

        res: dict[str, Any] = {
            "draft_tokens": draft_tokens,
            "unary_argmax": unary_argmax,
            "cand": cand,
            "unary": unary,
            "sel_scores": sel_scores,
            "sel_probs": sel_probs,
            "h_draft": h_draft,
            "selector_gate": g,
            "selector_corrections": corrections,
            "predecessors": predecessors,
        }
        return res


@dataclass
class DFT1Trace:
    pos: np.ndarray        # [N] int64
    token_id: np.ndarray   # [N] int32
    is_gen: np.ndarray     # [N] int32
    taps: np.ndarray       # [N, 4, 2304] float32


def load_dft1(path: pathlib.Path) -> DFT1Trace:
    raw = pathlib.Path(path).read_bytes()
    if len(raw) < 4 or raw[:4] != b"DFT1":
        raise ValueError(f"{path}: invalid DFT1 magic {raw[:4]!r}")
    rec_bytes = 8 + 4 + 4 + 4 * HIDDEN_SIZE * 4  # 36880 bytes
    payload = len(raw) - 4
    if payload % rec_bytes != 0:
        raise ValueError(f"{path}: payload {payload} not divisible by record size {rec_bytes}")
    n = payload // rec_bytes
    dt = np.dtype([
        ("pos", "<i8"),
        ("token_id", "<i4"),
        ("is_gen", "<i4"),
        ("taps", "<f4", (4, HIDDEN_SIZE)),
    ])
    arr = np.frombuffer(raw, dtype=dt, offset=4, count=n)
    return DFT1Trace(
        pos=arr["pos"].copy(),
        token_id=arr["token_id"].copy(),
        is_gen=arr["is_gen"].copy(),
        taps=arr["taps"].copy(),
    )


def replay_dft1(
    model: DFlash2Model,
    trace: DFT1Trace,
    mode: str = "both",
    temperature: float = 0.0,
    seed: int = 42,
    max_steps: int | None = None,
    ref_outputs: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay a DFT1 trace and compute comprehensive acceptance & agreement metrics.

    Alignment:
      - Prompt records have is_gen == 0 (indices 0 .. P-1).
      - Generated records have is_gen == 1 (indices P .. N-1).
      - At anchor index `start` (positions 0..start-1 already injected into draft KV):
        noise block = [token_id[start], MASK x 7]
        true targets for draft[0..6] are token_id[start+1 .. start+7].
    """
    n = len(trace.pos)
    gen_indices = np.where(trace.is_gen == 1)[0]
    if gen_indices.size == 0:
        raise ValueError("DFT1 trace has no generated tokens (is_gen == 1)")
    p_len = int(gen_indices[0])
    # Verify contiguous positions starting at 0
    if not np.array_equal(trace.pos, np.arange(n, dtype=np.int64)):
        raise ValueError("DFT1 trace positions are not contiguous 0..N-1")
    if not np.array_equal(trace.is_gen, np.arange(n) >= p_len):
        raise ValueError("DFT1 trace must contain a prompt followed by generated rows")

    results: dict[str, Any] = {
        "n_total": n,
        "n_prompt": p_len,
        "n_gen": n - p_len,
        "kv_quant": model.kv_quant,
        "temperature": temperature,
    }
    captured_all_pos: list[dict[str, Any]] = []

    def _run_pass(closed_loop: bool) -> dict[str, Any]:
        model.reset_cache()
        rng = np.random.default_rng(seed) if temperature > 0.0 else None

        # Prefill prompt taps [0 .. p_len - 1]
        if p_len > 0:
            model.inject(trace.taps[:p_len], trace.pos[:p_len])

        start = p_len
        steps = 0
        acc_hist = np.zeros(BLOCK_SIZE, dtype=np.int64)  # 0..7
        surv_hits = np.zeros(BLOCK_SIZE - 1, dtype=np.int64)
        pos_match = np.zeros(BLOCK_SIZE - 1, dtype=np.int64)
        unary_pos_match = np.zeros(BLOCK_SIZE - 1, dtype=np.int64)
        top16_hits = np.zeros(BLOCK_SIZE - 1, dtype=np.int64)

        sel_fixed = 0
        sel_broke = 0
        sel_both_right = 0
        sel_both_wrong = 0
        sel_changed = 0

        # Reference comparison accumulators (when ref_outputs provided on all_positions)
        ref_draft_match = 0
        ref_draft_total = 0
        ref_top16_overlap_sum = 0.0
        ref_h_rel_rmse_sum = 0.0
        ref_score_mae_sum = 0.0
        ref_score_total = 0
        ref_tvd_sum = 0.0

        t0 = time.perf_counter()
        while start + BLOCK_SIZE <= n:
            if max_steps is not None and steps >= max_steps:
                break
            anchor = int(trace.token_id[start])
            truth = trace.token_id[start + 1 : start + BLOCK_SIZE]

            out = model.draft_block(anchor, start, temperature=temperature, rng=rng)
            draft = out["draft_tokens"]
            u_arg = out["unary_argmax"]
            cand = out["cand"]

            # Consecutive acceptance count 0..7
            acc = 0
            for k in range(BLOCK_SIZE - 1):
                if draft[k] == truth[k]:
                    acc += 1
                else:
                    break
            acc_hist[acc] += 1
            for k in range(acc):
                surv_hits[k] += 1

            # Per-position stats
            for k in range(BLOCK_SIZE - 1):
                m_sel = (draft[k] == truth[k])
                m_uni = (u_arg[k] == truth[k])
                if m_sel:
                    pos_match[k] += 1
                if m_uni:
                    unary_pos_match[k] += 1
                if int(truth[k]) in cand[k]:
                    top16_hits[k] += 1
                if draft[k] != u_arg[k]:
                    sel_changed += 1
                if m_sel and not m_uni:
                    sel_fixed += 1
                elif not m_sel and m_uni:
                    sel_broke += 1
                elif m_sel and m_uni:
                    sel_both_right += 1
                else:
                    sel_both_wrong += 1

            if not closed_loop:
                if ref_outputs is not None and steps < len(ref_outputs):
                    ro = ref_outputs[steps]
                    ref_draft_match += int(np.sum(draft == ro["draft_tokens"]))
                    ref_draft_total += BLOCK_SIZE - 1
                    for k in range(BLOCK_SIZE - 1):
                        s1 = set(int(x) for x in cand[k])
                        s2 = set(int(x) for x in ro["cand"][k])
                        ref_top16_overlap_sum += len(s1 & s2) / float(SELECTOR_TOP_K)
                        # Proposal distribution TVD over union of candidates
                        p1 = {int(cand[k, j]): float(out["sel_probs"][k, j]) for j in range(SELECTOR_TOP_K)}
                        p2 = {int(ro["cand"][k, j]): float(ro["sel_probs"][k, j]) for j in range(SELECTOR_TOP_K)}
                        tvd = 0.5 * sum(abs(p1.get(tok, 0.0) - p2.get(tok, 0.0)) for tok in (s1 | s2))
                        ref_tvd_sum += tvd
                        common = cand[k, :, None] == ro["cand"][k, None, :]
                        ref_score_mae_sum += float(np.abs(out["sel_scores"][k, :, None] - ro["sel_scores"][k, None, :])[common].sum())
                        ref_score_total += int(common.sum())
                    err_h = model.bridge.eval_error(ro["h_draft"], out["h_draft"])
                    ref_h_rel_rmse_sum += err_h["rel_rmse"]
                else:
                    captured_all_pos.append({
                        "start": start,
                        "draft_tokens": draft.copy(),
                        "cand": cand.copy(),
                        "sel_scores": out["sel_scores"].copy(),
                        "sel_probs": out["sel_probs"].copy(),
                        "h_draft": out["h_draft"].copy(),
                    })

            advance = (acc + 1) if closed_loop else 1
            model.inject(trace.taps[start : start + advance], trace.pos[start : start + advance])
            start += advance
            steps += 1

        elapsed = time.perf_counter() - t0
        surv_rates = (surv_hits / max(1, steps)).tolist()
        mean_acc_len = 1.0 + sum(surv_rates)  # exact identity: E[1 + acc] = 1 + sum_{k=1..7} P(acc >= k)
        res_pass: dict[str, Any] = {
            "steps": steps,
            "elapsed_s": round(elapsed, 3),
            "ms_per_step": round(1000.0 * elapsed / max(1, steps), 2),
            "mean_acceptance_len": round(mean_acc_len, 4),
            "mean_draft_accepted": round(mean_acc_len - 1.0, 4),
            "survival_rates": [round(x, 4) for x in surv_rates],
            "pos_match_rates": [round(x, 4) for x in (pos_match / max(1, steps)).tolist()],
            "unary_pos_match_rates": [round(x, 4) for x in (unary_pos_match / max(1, steps)).tolist()],
            "top16_recall_by_pos": [round(x, 4) for x in (top16_hits / max(1, steps)).tolist()],
            "top16_recall_overall": round(float(np.sum(top16_hits)) / max(1, steps * (BLOCK_SIZE - 1)), 4),
            "acc_histogram_0_to_7": acc_hist.tolist(),
            "selector_stats": {
                "changed_from_unary": sel_changed,
                "fixed_unary_error": sel_fixed,
                "broke_unary_correct": sel_broke,
                "net_gain": sel_fixed - sel_broke,
                "win_rate_when_disagree": round(sel_fixed / max(1, sel_fixed + sel_broke), 4),
                "both_right": sel_both_right,
                "both_wrong": sel_both_wrong,
            },
        }
        if not closed_loop and ref_draft_total > 0:
            res_pass["vs_ref"] = {
                "draft_token_agreement": round(ref_draft_match / ref_draft_total, 4),
                "top16_overlap": round(ref_top16_overlap_sum / ref_draft_total, 4),
                "h_draft_mean_rel_rmse": round(ref_h_rel_rmse_sum / max(1, steps), 6),
                "sel_score_mae_on_shared_tokens": round(ref_score_mae_sum / max(1, ref_score_total), 5),
                "proposal_mean_tvd": round(ref_tvd_sum / ref_draft_total, 5),
            }
        return res_pass

    if mode in ("closed-loop", "both"):
        results["closed_loop"] = _run_pass(closed_loop=True)
    if mode in ("all-positions", "both"):
        results["all_positions"] = _run_pass(closed_loop=False)

    return results, captured_all_pos


def run_selfcheck(bridge: QBridge, target_path: pathlib.Path, dflash2_dir: pathlib.Path) -> dict[str, Any]:
    """Verify BF16 GGUF == source/model.safetensors bit-for-bit, verify Q6_K AVX2 matvec, and inspect target."""
    report: dict[str, Any] = {}

    # 1. Inspect target GGUF shared tensors
    target = TargetSharedWeights(target_path, bridge)
    report["target_shared_tensors"] = {
        "path": str(target_path),
        "token_embd.weight": {
            "shape": [int(x) for x in target.t_embd.shape],
            "type": target.t_embd.tensor_type.name,
            "type_id": target.embd_qtype,
            "n_bytes": int(target.t_embd.n_bytes),
        },
        "token_embd_norm.weight": {
            "shape": [int(x) for x in target.t_embd_norm.shape],
            "type": target.t_embd_norm.tensor_type.name,
            "n_bytes": int(target.t_embd_norm.n_bytes),
            "mean": float(np.mean(target.embd_norm)),
            "std": float(np.std(target.embd_norm)),
        },
        "output.weight": {
            "shape": [int(x) for x in target.t_output.shape],
            "type": target.t_output.tensor_type.name,
            "type_id": target.output_qtype,
            "n_bytes": int(target.t_output.n_bytes),
            "valid_vocab_rows": VOCAB_VALID,
            "valid_vocab_bytes": VOCAB_VALID * bridge.row_size(target.output_qtype, HIDDEN_SIZE),
        },
    }

    # 2. Verify AVX2 qbridge_q6k_matvec_f32 against exact ggml dequantize_row_q6_K @ X.T
    rng = np.random.default_rng(12345)
    test_x = rng.standard_normal((7, HIDDEN_SIZE)).astype(np.float32)
    test_rows = 4096
    w_deq = bridge.dequantize_2d(target.output_raw, target.output_qtype, test_rows, HIDDEN_SIZE)
    ref_logits = (test_x @ w_deq.T).astype(np.float32)
    fast_logits = bridge.matvec(target.output_qtype, target.output_raw, test_x, test_rows, HIDDEN_SIZE, mode="f32")
    q8k_logits = bridge.matvec(target.output_qtype, target.output_raw, test_x, test_rows, HIDDEN_SIZE, mode="q8k")

    # Benchmark full 200058 x 2304 Q6_K matvec (7 rows)
    _ = target.lm_head(test_x)  # warmup
    t0 = time.perf_counter()
    for _ in range(5):
        full_f32 = bridge.matvec(target.output_qtype, target.output_raw, test_x, VOCAB_VALID, HIDDEN_SIZE, mode="f32")
    t_f32_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

    t0 = time.perf_counter()
    for _ in range(5):
        full_q8k = bridge.matvec(target.output_qtype, target.output_raw, test_x, VOCAB_VALID, HIDDEN_SIZE, mode="q8k")
    t_q8k_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

    report["q6k_matvec_check"] = {
        "f32_vs_numpy_deq_4096rows": bridge.eval_error(ref_logits, fast_logits),
        "q8k_vs_numpy_deq_4096rows": bridge.eval_error(ref_logits, q8k_logits),
        "full_200058x2304_m7_f32_ms": round(t_f32_ms, 2),
        "full_200058x2304_m7_q8k_ms": round(t_q8k_ms, 2),
        "full_f32_vs_q8k_top1_match": int(np.sum(np.argmax(full_f32, axis=-1) == np.argmax(full_q8k, axis=-1))),
    }

    # 3. Load BOTH source/model.safetensors and Inkling-2-Mini-DFlash2-BF16.gguf
    w_st = DFlash2Weights(dflash2_dir / "source" / "model.safetensors", bridge)
    w_bf16 = DFlash2Weights(dflash2_dir / "Inkling-2-Mini-DFlash2-BF16.gguf", bridge)

    # Compare all 79 dense tensors and every valid row of both codebooks.
    max_dense_diff = 0.0
    mismatched_tensors: list[str] = []
    assert set(w_st.tensors.keys()) == set(w_bf16.tensors.keys())
    for name in sorted(w_st.tensors.keys()):
        a = w_st.tensors[name]
        b = w_bf16.tensors[name]
        if a.shape != b.shape or not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
            diff = float(np.max(np.abs(a - b)))
            max_dense_diff = max(max_dense_diff, diff)
            mismatched_tensors.append(f"{name} (max_abs={diff})")

    def check_codebooks(a: CodebookTable, b: CodebookTable) -> bool:
        return (a.qtype == b.qtype == int(T.BF16) and
                np.array_equal(a.raw_u8.view(np.uint8).ravel(), b.raw_u8.view(np.uint8).ravel()))

    pred_exact = check_codebooks(w_st.pred_cb, w_bf16.pred_cb)
    succ_exact = check_codebooks(w_st.succ_cb, w_bf16.succ_cb)

    report["weight_parity_bf16_gguf_vs_safetensors"] = {
        "dense_tensors_checked": len(w_st.tensors),
        "all_dense_bit_identical": len(mismatched_tensors) == 0,
        "max_dense_abs_diff": max_dense_diff,
        "mismatched_tensors": mismatched_tensors,
        "codebook_rows_checked_each": VOCAB_VALID,
        "pred_codebook_bit_identical": bool(pred_exact),
        "succ_codebook_bit_identical": bool(succ_exact),
    }

    # 4. Synthetic end-to-end forward pass, then a separate SWA wrap test.
    m_st = DFlash2Model(w_st, target, kv_quant="fp32")
    m_bf16 = DFlash2Model(w_bf16, target, kv_quant="fp32")

    syn_taps = rng.standard_normal((16, 4, HIDDEN_SIZE)).astype(np.float32) * 0.5
    syn_pos = np.arange(16, dtype=np.int64)
    ctx_st = m_st.inject(syn_taps, syn_pos)
    ctx_bf16 = m_bf16.inject(syn_taps, syn_pos)

    out_st = m_st.draft_block(anchor_id=1234, start_pos=16, temperature=0.0, return_intermediates=True)
    out_bf16 = m_bf16.draft_block(anchor_id=1234, start_pos=16, temperature=0.0, return_intermediates=True)

    inter_diffs = {}
    all_inter_exact = True
    for k in out_st["intermediates"]:
        a = out_st["intermediates"][k]
        b = out_bf16["intermediates"][k]
        exact = a.dtype == b.dtype and a.tobytes() == b.tobytes()
        if not exact:
            all_inter_exact = False
        inter_diffs[k] = {
            "bit_identical": exact,
            "max_abs": float(np.max(np.abs(a.astype(np.float64) - b), initial=0)),
        }

    # Also test Q8_0 KV cache impact on the same synthetic step
    m_q8kv = DFlash2Model(w_bf16, target, kv_quant="q8_0")
    m_q8kv.inject(syn_taps, syn_pos)
    out_q8kv = m_q8kv.draft_block(anchor_id=1234, start_pos=16, temperature=0.0)

    # Also test SWA-2048 ring wraparound (inject 2060 positions, draft at start_pos=2060)
    m_swa = DFlash2Model(w_bf16, target, kv_quant="fp32")
    swa_taps = rng.standard_normal((2060, 4, HIDDEN_SIZE)).astype(np.float32) * 0.25
    swa_pos = np.arange(2060, dtype=np.int64)
    m_swa.inject(swa_taps, swa_pos)
    out_swa = m_swa.draft_block(anchor_id=5678, start_pos=2060, temperature=0.0)

    # Also test DFT1 binary roundtrip
    DEFAULT_RECON_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DEFAULT_RECON_DIR, prefix="selfcheck-") as tmp:
        tmp_dft1 = pathlib.Path(tmp) / "synthetic.dft1"
        n_syn = 24
        with tmp_dft1.open("wb") as fp:
            fp.write(b"DFT1")
            for p in range(n_syn):
                tok = 100 + p
                is_gen = 0 if p < 12 else 1
                fp.write(struct.pack("<qii", p, tok, is_gen))
                fp.write(rng.standard_normal((4, HIDDEN_SIZE)).astype(np.float32).tobytes())
        tr = load_dft1(tmp_dft1)
        rep, _ = replay_dft1(m_bf16, tr, mode="both", temperature=0.0)
        dft1_ok = (tr.pos.shape == (n_syn,) and rep["all_positions"]["steps"] == 5 and rep["closed_loop"]["steps"] >= 1)

    report["forward_parity_bf16_gguf_vs_safetensors"] = {
        "ctx_hidden_bit_identical": bool(np.array_equal(ctx_st.view(np.uint32), ctx_bf16.view(np.uint32))),
        "all_intermediates_bit_identical": all_inter_exact,
        "intermediates": inter_diffs,
        "draft_tokens_st": out_st["draft_tokens"].tolist(),
        "draft_tokens_bf16": out_bf16["draft_tokens"].tolist(),
        "draft_tokens_exact_match": bool(np.array_equal(out_st["draft_tokens"], out_bf16["draft_tokens"])),
        "sel_scores_bit_identical": bool(np.array_equal(out_st["sel_scores"].view(np.uint32), out_bf16["sel_scores"].view(np.uint32))),
        "q8_0_kv_h_draft_err": bridge.eval_error(out_bf16["h_draft"], out_q8kv["h_draft"]),
        "q8_0_kv_draft_tokens_match": int(np.sum(out_bf16["draft_tokens"] == out_q8kv["draft_tokens"])),
        "swa_2060_wrap_ok": bool(out_swa["draft_tokens"].shape == (7,) and np.all(out_swa["draft_tokens"] < VOCAB_VALID)),
        "dft1_parser_and_replay_ok": dft1_ok,
    }
    return report


def parse_quant_overrides(specs: list[str]) -> list[tuple[re.Pattern[str], int]]:
    out = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--quant-override must be REGEX=TYPE, got {s!r}")
        pat_s, qt_s = s.rsplit("=", 1)
        qt = NAME_TO_QTYPE.get(qt_s.upper())
        if qt is None:
            raise ValueError(f"Unknown qtype {qt_s!r}; supported: {sorted(NAME_TO_QTYPE)}")
        out.append((re.compile(pat_s), qt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Inkling-2-Mini DFlash2 Reference Oracle & Quantization Replay Harness")
    ap.add_argument("--bridge", type=pathlib.Path, default=DEFAULT_BRIDGE)
    ap.add_argument("--target", type=pathlib.Path, default=DEFAULT_TARGET_GGUF)
    ap.add_argument("--dflash2-dir", type=pathlib.Path, default=DEFAULT_DFLASH2_DIR)
    ap.add_argument("--threads", type=int, default=4, help="bridge threads; set OPENBLAS_NUM_THREADS separately")
    ap.add_argument("--lm-head-mode", choices=["f32", "q8k", "blas"], default="f32")
    ap.add_argument("--selfcheck", action="store_true", help="Run BF16 GGUF == safetensors parity + Q6_K matvec + SWA checks")
    ap.add_argument("--replay", type=pathlib.Path, default=None, help="Path to DFT1 binary tap dump")
    ap.add_argument("--model", type=pathlib.Path, action="append", default=[], help="Draft safetensors or GGUF path(s) to replay")
    ap.add_argument("--quant-override", action="append", default=[], metavar="REGEX=TYPE", help="Per-tensor in-memory quant override")
    ap.add_argument("--kv-quant", choices=["fp32", "q8_0", "q8_0_k", "f16"], default="fp32")
    ap.add_argument("--mode", choices=["both", "closed-loop", "all-positions"], default="both")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="proposal sampling only; replay is not stochastic rejection sampling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--dump-json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    bridge = QBridge(args.bridge, nthreads=args.threads)

    if args.selfcheck:
        rep = run_selfcheck(bridge, args.target, args.dflash2_dir)
        txt = json.dumps(rep, indent=2)
        print(txt)
        if args.dump_json:
            args.dump_json.write_text(txt + "\n", encoding="utf-8")
        weights = rep["weight_parity_bf16_gguf_vs_safetensors"]
        forward = rep["forward_parity_bf16_gguf_vs_safetensors"]
        failed = (
            not all(weights[k] for k in ("all_dense_bit_identical", "pred_codebook_bit_identical",
                                        "succ_codebook_bit_identical"))
            or not all(forward[k] for k in ("ctx_hidden_bit_identical", "all_intermediates_bit_identical",
                                            "draft_tokens_exact_match", "sel_scores_bit_identical",
                                            "swa_2060_wrap_ok", "dft1_parser_and_replay_ok"))
            or rep["q6k_matvec_check"]["f32_vs_numpy_deq_4096rows"]["rel_rmse"] > 1e-5
        )
        return int(failed)

    if args.replay is not None:
        trace = load_dft1(args.replay)
        target = TargetSharedWeights(args.target, bridge, lm_head_mode=args.lm_head_mode)
        overrides = parse_quant_overrides(args.quant_override)
        models = args.model or [args.dflash2_dir / "Inkling-2-Mini-DFlash2-BF16.gguf"]
        all_results = []
        ref_outputs = None
        for idx, mpath in enumerate(models):
            w = DFlash2Weights(mpath, bridge, quant_overrides=overrides if idx > 0 or len(models) == 1 else [])
            dm = DFlash2Model(w, target, kv_quant=args.kv_quant)
            res, cap = replay_dft1(
                dm,
                trace,
                mode=args.mode,
                temperature=args.temperature,
                seed=args.seed,
                max_steps=args.max_steps,
                ref_outputs=ref_outputs,
            )
            res["model"] = str(mpath)
            if idx == 0 and cap:
                ref_outputs = cap
            all_results.append(res)
        txt = json.dumps(all_results, indent=2)
        print(txt)
        if args.dump_json:
            args.dump_json.write_text(txt + "\n", encoding="utf-8")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
