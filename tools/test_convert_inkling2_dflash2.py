#!/usr/bin/env python3
"""Unit tests for tools/convert_inkling2_dflash2_to_gguf.py (no GPU or downloads required)."""
from __future__ import annotations

import json
import pathlib
import re
import struct
import sys
import tempfile
import unittest

import numpy as np

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(REPO_ROOT / "third_party/llama.cpp/gguf-py"))

import convert_inkling2_dflash2_to_gguf as C  # noqa: E402
import gguf  # noqa: E402
import gguf_reader as G  # noqa: E402


def _write_safetensors(path: pathlib.Path, tensors: dict[str, np.ndarray]) -> None:
    hdr = {}
    chunks = []
    offset = 0
    for name, arr in tensors.items():
        # Convert float32 -> BF16 uint16 (round to nearest even / truncate top 16 bits)
        u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
        bf16 = ((u32 + np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)
        raw = bf16.tobytes()
        hdr[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(hdr).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for c in chunks:
            f.write(c)


class TestDFlash2Converter(unittest.TestCase):
    def test_visit_to_layer_inkling2_loop(self) -> None:
        # Inkling-2-Mini: 24 physical layers, loop [6, 18) x 3 -> 48 logical visits
        visits = [5, 23, 41, 48]
        phys = [C.visit_to_layer(v, n_phys=24, start=6, end=18, count=3) for v in visits]
        self.assertEqual(phys, [5, 11, 17, 24])
        # Verify loop wrap boundaries
        self.assertEqual(C.visit_to_layer(0, 24, 6, 18, 3), 0)
        self.assertEqual(C.visit_to_layer(6, 24, 6, 18, 3), 6)
        self.assertEqual(C.visit_to_layer(17, 24, 6, 18, 3), 17)
        self.assertEqual(C.visit_to_layer(18, 24, 6, 18, 3), 6)
        self.assertEqual(C.visit_to_layer(30, 24, 6, 18, 3), 6)
        self.assertEqual(C.visit_to_layer(42, 24, 6, 18, 3), 18)
        self.assertEqual(C.visit_to_layer(47, 24, 6, 18, 3), 23)

    def test_tensor_mapping_all_81(self) -> None:
        names = list(C.STATIC_TENSOR_MAP.keys())
        for bid in range(5):
            for rest in C.LAYER_TENSOR_MAP:
                names.append(f"layers.{bid}.{rest}")
        self.assertEqual(len(names), 81)
        mapped = [C.map_tensor_name(n) for n in names]
        self.assertEqual(len(set(mapped)), 81)
        self.assertEqual(C.map_tensor_name("model.fc.weight"), "fc.weight")
        self.assertEqual(C.map_tensor_name("candidate_selector.predecessor_codebook"), "selector_predecessor.weight")
        self.assertEqual(C.map_tensor_name("layers.3.attention_conv.base_kernel"), "blk.3.attn_conv_base")
        with self.assertRaises(KeyError):
            C.map_tensor_name("layers.0.unknown.weight")

    def test_choose_qtype_profiles(self) -> None:
        T = gguf.GGMLQuantizationType
        # Norms and conv bases always stay F32 in every profile
        for prof in C.PROFILE_FILETYPE:
            self.assertEqual(C.choose_qtype("blk.0.attn_norm.weight", 1, prof, []), T.F32)
            self.assertEqual(C.choose_qtype("blk.0.attn_conv_base", 3, prof, []), T.F32)
            self.assertEqual(C.choose_qtype("enc_output_norm.weight", 1, prof, []), T.F32)

        # q4_0-mixed policy
        self.assertEqual(C.choose_qtype("fc.weight", 2, "q4_0-mixed", []), T.Q8_0)
        self.assertEqual(C.choose_qtype("blk.0.attn_conv_proj.weight", 2, "q4_0-mixed", []), T.Q8_0)
        self.assertEqual(C.choose_qtype("selector_hidden.weight", 2, "q4_0-mixed", []), T.Q8_0)
        self.assertEqual(C.choose_qtype("selector_predecessor.weight", 2, "q4_0-mixed", []), T.F16)
        self.assertEqual(C.choose_qtype("blk.0.attn_q.weight", 2, "q4_0-mixed", []), T.Q4_0)
        self.assertEqual(C.choose_qtype("blk.4.ffn_down.weight", 2, "q4_0-mixed", []), T.Q4_0)

        # q4_k-mixed policy
        self.assertEqual(C.choose_qtype("selector_hidden.weight", 2, "q4_k-mixed", []), T.BF16)
        self.assertEqual(C.choose_qtype("fc.weight", 2, "q4_k-mixed", []), T.Q8_0)
        self.assertEqual(C.choose_qtype("selector_successor.weight", 2, "q4_k-mixed", []), T.Q8_0)
        self.assertEqual(C.choose_qtype("blk.2.attn_output.weight", 2, "q4_k-mixed", []), T.Q6_K)
        self.assertEqual(C.choose_qtype("blk.2.ffn_down.weight", 2, "q4_k-mixed", []), T.Q6_K)
        self.assertEqual(C.choose_qtype("blk.2.attn_q.weight", 2, "q4_k-mixed", []), T.Q4_K)
        self.assertEqual(C.choose_qtype("blk.2.ffn_gate.weight", 2, "q4_k-mixed", []), T.Q4_K)

        # Regex override takes precedence for 2D .weight tensors
        ov = [(re.compile(r"^blk\.0\.ffn_"), T.Q8_0)]
        self.assertEqual(C.choose_qtype("blk.0.ffn_gate.weight", 2, "q4_0", ov), T.Q8_0)
        self.assertEqual(C.choose_qtype("blk.1.ffn_gate.weight", 2, "q4_0", ov), T.Q4_0)

    def test_synthetic_end_to_end_bf16_and_q4_0_mixed(self) -> None:
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            # 1. Build a tiny synthetic inkling2_loop target GGUF (real vocab=64, padded=80)
            target_path = root / "target.gguf"
            tw = gguf.GGUFWriter(str(target_path), arch="inkling2_loop")
            tw.add_block_count(24)
            tw.add_embedding_length(64)
            tw.add_uint32("inkling2_loop.loop.start", 6)
            tw.add_uint32("inkling2_loop.loop.end", 18)
            tw.add_uint32("inkling2_loop.loop.count", 3)
            tw.add_tokenizer_model("gpt2")
            tw.add_tokenizer_pre("inkling2")
            tw.add_token_list([f"tok_{i}" for i in range(64)])
            tw.add_token_types([1] * 64)
            tw.add_token_merges(["a b"])
            tw.add_eos_token_id(60)
            tw.write_header_to_file()
            tw.write_kv_data_to_file()
            tw.close()

            # 2. Build a tiny synthetic DFlash2 checkpoint (1 layer, hidden=64, padded vocab=80)
            src_dir = root / "dflash2"
            src_dir.mkdir()
            cfg = {
                "architectures": ["DFlash2DraftModel"],
                "dflash_config": {
                    "block_size": 8,
                    "conv_group_size": 16,
                    "conv_kernel_size": 2,
                    "final_logit_softcapping": 30.0,
                    "mask_token_id": 62,
                    "selector_rank": 32,
                    "selector_softcap_mode": "after_correction",
                    "selector_top_k": 4,
                    "target_layer_ids": [5, 23, 41, 48],
                },
                "head_dim": 32,
                "hidden_size": 64,
                "intermediate_size": 128,
                "is_causal": False,
                "layer_types": ["sliding_attention"],
                "max_position_embeddings": 1024,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "num_target_layers": 48,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 10000.0},
                "sliding_window": 256,
                "use_sliding_window": True,
                "vocab_size": 80,
            }
            (src_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            tensors = {
                "fc.weight": rng.standard_normal((64, 256)).astype(np.float32),
                "hidden_norm.weight": rng.standard_normal((64,)).astype(np.float32),
                "norm.weight": rng.standard_normal((64,)).astype(np.float32),
                "candidate_selector.hidden_projection.weight": rng.standard_normal((32, 64)).astype(np.float32),
                "candidate_selector.predecessor_codebook": rng.standard_normal((80, 32)).astype(np.float32),
                "candidate_selector.successor_codebook": rng.standard_normal((80, 32)).astype(np.float32),
                "layers.0.input_layernorm.weight": rng.standard_normal((64,)).astype(np.float32),
                "layers.0.post_attention_layernorm.weight": rng.standard_normal((64,)).astype(np.float32),
                "layers.0.self_attn.q_norm.weight": rng.standard_normal((32,)).astype(np.float32),
                "layers.0.self_attn.k_norm.weight": rng.standard_normal((32,)).astype(np.float32),
                "layers.0.self_attn.q_proj.weight": rng.standard_normal((64, 64)).astype(np.float32),
                "layers.0.self_attn.k_proj.weight": rng.standard_normal((32, 64)).astype(np.float32),
                "layers.0.self_attn.v_proj.weight": rng.standard_normal((32, 64)).astype(np.float32),
                "layers.0.self_attn.o_proj.weight": rng.standard_normal((64, 64)).astype(np.float32),
                "layers.0.mlp.gate_proj.weight": rng.standard_normal((128, 64)).astype(np.float32),
                "layers.0.mlp.up_proj.weight": rng.standard_normal((128, 64)).astype(np.float32),
                "layers.0.mlp.down_proj.weight": rng.standard_normal((64, 128)).astype(np.float32),
                "layers.0.attention_conv.base_kernel": rng.standard_normal((2, 2, 64)).astype(np.float32),
                "layers.0.attention_conv.kernel_projection.weight": rng.standard_normal((16, 64)).astype(np.float32),
                "layers.0.mlp_conv.base_kernel": rng.standard_normal((2, 2, 64)).astype(np.float32),
                "layers.0.mlp_conv.kernel_projection.weight": rng.standard_normal((16, 64)).astype(np.float32),
            }
            _write_safetensors(src_dir / "model.safetensors", tensors)

            bridge = C.GgmlQuantBridge(None)  # pure gguf-py fallback
            for prof in ("bf16", "q4_0-mixed"):
                out = root / f"out-{prof}.gguf"
                C.convert(src_dir, target_path, out, prof, [], bridge)
                gf = G.GGUFFile(out)
                self.assertEqual(gf.metadata["general.architecture"], "dflash")
                self.assertEqual(gf.metadata["dflash.target_visits"], [5, 23, 41, 48])
                self.assertEqual(gf.metadata["dflash.target_layers"], [5, 11, 17, 24])
                self.assertEqual(gf.metadata["dflash.selector_softcap_after_correction"], True)
                self.assertEqual(gf.metadata["tokenizer.ggml.mask_token_id"], 62)
                self.assertEqual(len(gf.metadata["tokenizer.ggml.tokens"]), 64)
                by = {t.name: t for t in gf.tensors}
                self.assertEqual(by["selector_predecessor.weight"].shape, [32, 64])
                self.assertEqual(by["blk.0.attn_conv_base"].shape, [64, 2, 2])
                self.assertEqual(by["blk.0.attn_conv_base"].type_name, "F32")
                if prof == "q4_0-mixed":
                    self.assertEqual(by["fc.weight"].type_name, "Q8_0")
                    self.assertEqual(by["blk.0.attn_conv_proj.weight"].type_name, "Q8_0")
                    self.assertEqual(by["selector_predecessor.weight"].type_name, "F16")
                    self.assertEqual(by["blk.0.ffn_gate.weight"].type_name, "Q4_0")


if __name__ == "__main__":
    unittest.main()
