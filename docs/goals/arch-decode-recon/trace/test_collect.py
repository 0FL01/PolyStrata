"""CPU-only PSVST1 fixture; optionally write it with `python test_collect.py sample.trace`."""

import io
import struct
import sys
import unittest
from pathlib import Path

import collect


def fixture():
    pos, tracks = 2048, []
    tracks.append(struct.pack("<8sQQQIIII", b"PSVST1\r\n", pos, pos + 2, 1, 24, 48, 128, 8))
    tracks.append(struct.pack("<HHIQQqqqqq", 2, 64, 0, 10, pos, pos + 1, 2048, 8192, 4096, 1))
    for visit in range(48):
        if visit < 6:
            layer, core_pass = visit, -1
        elif visit < 42:
            core_pass, layer = divmod(visit - 6, 12)
            layer += 6
        else:
            layer, core_pass = visit - 24, -1
        base = 20 + 10000 * visit
        host = (pos, 1, base, base + 9000, 300, base + 900)
        masks = (0, 2, 0x1000, 0, 0, 4)  # GPU expert 65; asked 12 and sent 66
        slots = (2048,) + (-1,) * 7
        weights = (0x3E000000,) * 7 + (0x80000000,)  # preserve the bit pattern of -0.0
        cpu = base + 1000, base + 8500, base + 2000, base + 5000, base + 6000, base + 8000
        tracks.append(struct.pack("<HHI18Q8q8I4f2H8B8B2BbB", 1, 288, 0, *host, *cpu, *masks, *slots, *weights,
                                  0.016, 0.007, 0.002, 0.003, 2, 3, 65, 0, 1, 2, 3, 4, 5, 6,
                                  1, 2, 0, 0, 0, 0, 0, 0, visit, layer, core_pass, 0))
    return b"".join(tracks)


class CollectorTest(unittest.TestCase):
    def parse(self, data):
        source = io.BytesIO(data)
        return list(collect.records(source, collect.header(source)))

    def test_complete(self):
        rows = self.parse(fixture())
        self.assertEqual(len(rows), 49)
        self.assertEqual(rows[0]["kind"], "cover")
        self.assertEqual(rows[0]["bytes_added"], 4096)
        self.assertEqual(rows[19]["layer"], 6)
        self.assertEqual(rows[19]["core_pass"], 1)
        self.assertEqual(rows[1]["resident_mask"], "00000000000000020000000000000000")
        self.assertEqual(rows[1]["weight_bits"][-1], 0x80000000)
        self.assertEqual(rows[1]["placement"][:3], ("gpu", "skipped", "cpu"))
        summary = collect.Summary()
        for row in rows:
            summary.take(row)
        self.assertEqual(summary.result(dict(visits=48))["by_cpu_count"][6]["visits"], 48)

    def test_truncated(self):
        raw = fixture()
        for n in (0, 10, len(raw) - 1, len(raw) - 288):
            with self.subTest(n=n), self.assertRaises(collect.TraceError):
                self.parse(raw[:n])

    def test_bad_mapping(self):
        raw = bytearray(fixture())
        raw[48 + 64 + 285] = 1  # first visit's physical layer
        with self.assertRaisesRegex(collect.TraceError, "bad mapping"):
            self.parse(raw)

    def test_placement_must_match_residency(self):
        raw = bytearray(fixture())
        raw[48 + 64 + 8 + 12 * 8 + 8] = 0  # clear GPU expert's residency bit
        with self.assertRaisesRegex(collect.TraceError, "inconsistent placement"):
            self.parse(raw)


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        with Path(sys.argv[1]).open("xb") as out:
            out.write(fixture())
    else:
        unittest.main()
