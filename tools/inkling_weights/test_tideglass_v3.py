"""Checks for the v3 whitelist and row codecs; no model or GPU required."""
from collections import Counter
from pathlib import Path
import os
import struct
from types import SimpleNamespace
import unittest

from tools.inkling_weights import tideglass_v3 as v3


class TideglassTest(unittest.TestCase):
    def test_map(self):
        parent, donor = {}, {}
        for layer in range(24):
            for proj in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_gate', 'attn_rel',
                         'ffn_gate_up_exps', 'ffn_down_exps'):
                name = f'blk.{layer}.{proj}.weight'
                shape = ((1152 if 'down' in proj else 2304), 2304, 128) if 'exps' in proj else (2304, 384)
                kind = (3 if 'down' in proj and layer < 3 else 2) if 'exps' in proj else 8
                parent[name] = v3.gg.Tensor(name, shape, kind, 0)
                donor[name] = v3.gg.Tensor(name, shape, 8, 0)
        for i in range(512):
            parent[str(i)] = donor[str(i)] = v3.gg.Tensor(str(i), (32,), 0, 0)
        pair = [SimpleNamespace(tensors=x) for x in (parent, donor)]
        edits = v3.plan(*pair)
        self.assertEqual(Counter(t.type_name for t in edits.values()), {'Q8_0': 144, 'Q4_0': 45, 'Q4_1': 3})
        name = 'blk.0.ffn_down_exps.weight'
        donor[name] = v3.gg.Tensor(name, (2304, 1152, 128), 8, 0)
        with self.assertRaisesRegex(ValueError, 'shapes differ'):
            v3.plan(*pair)

    @unittest.skipUnless(os.getenv('TIDEGLASS_GGML'), 'set TIDEGLASS_GGML to linked GGML library')
    def test_reference_blocks(self):
        # Two constant rows of opposite sign. An exact Q4_0 constant uses signed d and code zero;
        # Q4_1 represents a constant with zero delta and the row value in m.
        q = v3.Quantizer(Path(os.environ['TIDEGLASS_GGML']))
        data = b''.join(struct.pack('<e32b', 1., *([val]*32)) for val in (8, -8))
        for target, expected in [('q4_0', b''.join(struct.pack('<e', d)+bytes(16) for d in (-1, 1))),
                                 ('q4_1', b''.join(struct.pack('<ee', 0, m)+bytes(16) for m in (8, -8)))]:
            packed, peak, rms = q.expert(data, target, 64)
            self.assertEqual((bytes(packed), peak, rms), (expected, 0., 0.))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            q.expert(struct.pack('<e', float('inf'))+bytes([1])*32, 'q4_0', 32)
