#!/usr/bin/env python3
"""CPU-only smoke tests: known block modes and parser corruption/refusal cases."""

import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np

import codec
from snapshot import BLOCKS, TILE, VISIT_ROW, WIDTH, parse


class CodecTests(unittest.TestCase):
    def test_block_modes_and_exact_scale_bits(self):
        shape = (TILE, BLOCKS, WIDTH)
        s = np.full((TILE, BLOCKS, 2), [0x81, 0x7E], dtype=np.uint8)  # opaque bits, not floats
        rs = s ^ np.uint8(3)
        for mode, ref_value, value in ((2, 64, 63), (11, 126, 127), (19, 64, 67), (20, 64, 64)):
            with self.subTest(mode=mode):
                r = np.full(shape, ref_value, dtype=np.uint8)
                q = np.full(shape, value, dtype=np.uint8)
                if mode in (19, 20):
                    q[..., 11] = 200  # signed negative outlier
                encoded = codec.encode(q, s, 2, (r, rs))
                self.assertEqual(encoded[:2], bytes([2, mode]))
                dq, ds = codec.decode(encoded, {2: (r, rs)})
                self.assertEqual((dq.tobytes(), ds.tobytes()), (q.tobytes(), s.tobytes()))

    def test_temporal_chain_and_raw_tile(self):
        q = np.broadcast_to((np.arange(TILE, dtype=np.uint8)[:, None, None] + 105), (TILE, BLOCKS, WIDTH))
        s = np.broadcast_to(np.arange(TILE, dtype=np.uint8)[:, None, None], (TILE, BLOCKS, 2))
        encoded = codec.encode(q, s, 1, None)
        self.assertEqual(encoded[0], 1)
        dq, ds = codec.decode(encoded, {})
        self.assertEqual((dq.tobytes(), ds.tobytes()), (q.tobytes(), s.tobytes()))
        rng = np.random.default_rng(1234)
        q, s = (rng.integers(0, 256, size=a.shape, dtype=np.uint8) for a in (q, s))
        encoded = codec.encode(q, s, 0, None)
        self.assertEqual(encoded[0], 255)
        dq, ds = codec.decode(encoded, {})
        self.assertEqual((dq.tobytes(), ds.tobytes()), (q.tobytes(), s.tobytes()))


class SnapshotTests(unittest.TestCase):
    def test_layouts_and_corruption(self):
        # Tiny sparse disk-format fixtures in unlinked files. No inference data.
        for layout in (1, 2):
            with self.subTest(layout=layout):
                with tempfile.TemporaryFile(dir=Path(self.checker).resolve().parent) as fixture:
                    fd = fixture.fileno()
                    rows = 32
                    row_size = (7 + layout) * VISIT_ROW
                    state = 48 * 3 * (2 * 1024 + 2 * 2304) * 4 + 40 * rows * VISIT_ROW
                    if layout == 2:
                        state += 2304 * 4 + 2 * rows * VISIT_ROW
                    leading = struct.pack("<18q2q2Q", 0x494e4b4c494e4732, layout, rows, 100, 2304, 24, 1,
                                          state, row_size, (16 << 20) // row_size, *([0] * 8), 0, 24, 0, rows)
                    leading += struct.pack(f"<{rows}i2Q", *range(rows), 0, state)
                    record = struct.pack("<7Q7qQ", *([0] * 6), 1, 1, rows, 0, 0, 0, 0, 0, rows * row_size)
                    kv = 64 + len(leading) + state + len(record)
                    payload = kv - 64 + rows * row_size + 32
                    hdr = struct.pack("<8sII5Q", b"STRSESS\x01", 1, 64, 10, 11, payload, 0, 0)
                    os.pwrite(fd, hdr + bytes(8) + leading, 0)
                    os.pwrite(fd, record, 64 + len(leading) + state)
                    os.pwrite(fd, bytes(8) + b"STRSEND\x01", 64 + payload)
                    computed = subprocess.check_output([self.checker, str(fd), str(payload)], pass_fds=(fd,))
                    hh, ph = [int(h, 16) for h in computed.splitlines()]
                    os.pwrite(fd, struct.pack("<Q", hh), 56)
                    os.pwrite(fd, struct.pack("<Q", ph), 64 + payload)
                    meta, records = parse(fd, self.checker)
                    self.assertEqual((meta["inkling_layout"], meta["committed"], records), (layout, rows, [(rows, kv)]))
                    for at, why in ((16, "checksum mismatch"), (kv + 13, "checksum mismatch"),
                                    (64 + 6 * 8, "not Inkling Q8_0"), (64 + payload + 8, "bad end marker")):
                        original = os.pread(fd, 1, at)
                        os.pwrite(fd, bytes([original[0] ^ 1]), at)
                        with self.assertRaisesRegex(ValueError, why):
                            parse(fd, self.checker)
                        os.pwrite(fd, original, at)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checker")
    SnapshotTests.checker = p.parse_args().checker
    unittest.main(argv=[__file__])
