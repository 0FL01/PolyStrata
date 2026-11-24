#!/usr/bin/env python3
"""CPU-only smoke test with a deliberately synthetic PSVST1 capture; not performance evidence."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from analyze_routes import analyze, load, physical
from cp02_to_routes import run
from screen_lookahead import fifo


def fixture(path, last=8):
    with path.open("wb") as f:
        f.write(struct.pack("<8s3Q4I", b"PSVST1\r\n", 0, last, 1, 24, 48, 128, 8))
        for p in range(last):
            for v in range(48):
                part = (v - 6) // 12 if 6 <= v < 42 else -1
                ids = [((0 if p % 2 else 8) + i) for i in range(8)]
                # IDs 0-3 or 8-11 are on the card; every row executes the other four on CPU.
                t = 1 + 48_000 * p + v * 1000
                f.write(struct.pack("<HHI18Q8q8I4f2H16B2BbB", 1, 288, 0,
                                    p, p + 1, t, t + 850, 25, t + 100, t + 150, t + 500,
                                    t + 190, t + 300, t + 360, t + 450, 0x0f0f, 0, 1 << 63, 0, 0, 0,
                                    10, 20, 30, 40, -1, -1, -1, -1, *[0x3E000000] * 8, .0009, .0001, .0002,
                                    .000035, 1, 0, *ids, 1, 1, 1, 1, 0, 0, 0, 0, v, physical(v), part, 0))


class AnalyzerTest(unittest.TestCase):
    def test_held_out(self):
        with tempfile.TemporaryDirectory(dir=Path(sys.argv[0]).parent) as d:
            raw, normalized = Path(d) / "test.bin", Path(d) / "test.jsonl"
            fixture(raw)
            meta = dict(model_sha256="test-fixture", workload="synthetic", trajectory_id="test", expert_bytes=[20] * 24)
            run(raw, meta, 1, 8, normalized)
            t = load(normalized)
            self.assertEqual(t.pos.tolist(), [1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(int(t.bits[0, 0, 0]), 0x3E000000)
            r = analyze(t[3:], [t[:3]], [4])
            json.dumps(r, allow_nan=False)
            pair = "1->2"
            stats = r["held_out_cross_pass"][pair]
            self.assertEqual(stats["mean_intersection"], 8)
            self.assertEqual(stats["mean_independent_marginal_intersection"], 4)
            self.assertEqual(stats["by_layer"][6]["cross_pass_lag1_intersection"]["mean"], 0)
            self.assertEqual(stats["by_layer"][6]["source_to_target_route_ready_us"]["p50"], 12)
            arms = r["held_out_predictors"][pair]["micro"]
            self.assertEqual(arms["previous_pass@4"]["route_precision"], 1)
            self.assertEqual(arms["previous_pass@4"]["issued_nonresident"], 0)
            self.assertEqual(arms["previous_pass+source_mask@4"]["miss_recall"], 1)
            self.assertEqual(arms["previous_token+source_mask@4"]["false_positive_payload_bytes"], 3 * 12 * 4 * 20)
            self.assertEqual(r["timing"]["gpu_to_cpu_ready_marker_ns"]["per_token_us"]["p50"], 9.6)
            self.assertNotIn("exposed_join_ns", r["timing"])
            with self.assertRaisesRegex(ValueError, "outside PSVST1 bounds"):
                run(raw, meta, 1, 9, normalized)
            with raw.open("rb+") as f:
                f.seek(-1, 2)
                f.truncate()
            with self.assertRaisesRegex(ValueError, "truncated record"):
                run(raw, meta, 1, 8, normalized)

    def test_fifo_deadline_and_non_additive_visits(self):
        jobs = [(0, 10, 20, (0, 18), True), (0, 10, 20, (0, 18), True),
                (1, 10, 20, (0, 18), True), (20, 40, 20, (0, 19), False)]
        r = fifo(jobs, {(0, 18): 4000, (0, 19): 8000}, 4, {20: 5}, 1)
        self.assertEqual(r["on_time"], 3)
        self.assertEqual(r["on_time_useful"], 2)
        self.assertEqual(r["false_positive_bytes_per_token"], 20)
        self.assertEqual(r["mean_marker_gap_on_visits_with_on_time_useful_pick_us_per_token"], 4)


if __name__ == "__main__":
    unittest.main()
