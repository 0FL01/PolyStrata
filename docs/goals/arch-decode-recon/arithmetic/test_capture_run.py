"""CPU-only tests for config isolation and exact router-weight handoff."""

import contextlib
import io
import json
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import capture_run as cp
from test_collect import fixture


def two_frames():
    raw = bytearray(fixture())
    start = cp.collect.HEADER.size + 64
    step = cp.collect.PREFIX.size + cp.collect.VISIT.size
    rank0_weight = start + cp.VISIT * step + cp.collect.PREFIX.size + struct.calcsize('<18Q8q')
    struct.pack_into('<I', raw, rank0_weight, 0x3de12345)
    frame = raw[start:]
    for offset in range(cp.collect.PREFIX.size, len(frame), step):
        struct.pack_into('<QQ', frame, offset, cp.POSITION + 1, 2)
    return raw + frame


class CaptureTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.run = self.dir / 'run'
        self.run.mkdir()
        (self.dir / 'guard').mkdir()

    def put(self, name, value):
        (self.dir / name).write_text(json.dumps(value))

    def run_fixture(self):
        trace = self.run / 'visits.psv'
        trace.write_bytes(two_frames())
        (self.run / 'input.bin').write_bytes(bytes(cp.BYTES))
        (self.run / 'engine.log').write_text('CP04 capture: pos=2048 visit=6 layer=6\n')
        model = self.dir / cp.MODEL
        self.put('run/config.json', dict(length=cp.POSITION, trace=True, cmd=['engine', '--model', str(model)],
                                        env=dict(cp.capture_env(self.run), STRATA_VISIT_TRACE=str(trace),
                                                 STRATA_INKLING_VISIT_TRACE_FIRST='2048',
                                                 STRATA_INKLING_VISIT_TRACE_END='2050')))
        self.put('run/metrics.json', dict(length=cp.POSITION, decode_steps=2))
        self.put('guard/stop-reasons.json', [])

    def test_private_config(self):
        self.put('source.json', {'env': {'KEEP': '1'}, 'args': ['--keep']})
        source = self.dir / 'source.json'
        old = source.read_bytes()
        result = cp.prepare(self.dir, source)
        self.assertEqual(source.read_bytes(), old)
        self.assertEqual(stat.S_IMODE(result.stat().st_mode), 0o600)
        self.assertEqual(cp.read_json(result), {'env': {'KEEP': '1', **cp.capture_env(self.run)}, 'args': ['--keep']})
        with self.assertRaises(FileExistsError):
            cp.prepare(self.dir, source)

    def test_exact_rank0_bits_in_command(self):
        self.run_fixture()
        evidence, argv = cp.replay(self.dir, 'hip')
        self.assertEqual((evidence['rank'], evidence['expert'], evidence['placement']), (0, 65, 'gpu'))
        self.assertEqual(evidence['weight_bits'], '0x3de12345')
        self.assertEqual(Path(argv[0]).name, 'parity-hip')
        self.assertEqual(argv[2:5], ['6', '65', '0'])
        self.assertEqual(struct.pack('<f', float.fromhex(argv[5])), struct.pack('<I', 0x3de12345))
        self.assertEqual(Path(argv[6]), self.run / 'input.bin')

    def test_ref_suite_dry_run_is_guarded_and_never_executes(self):
        self.run_fixture()
        argv = ['capture_run.py', 'cpu-ref', str(self.dir)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(cp.os, 'umask'), patch.object(cp.subprocess, 'run') as run:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(cp.main(), 0)
            run.assert_not_called()
        cmd = cp.shlex.split(stdout.getvalue())
        self.assertIn('--warn-amdgpu-faults', cmd)
        self.assertIn(str(self.dir / 'cpu-ref-guard'), cmd)
        self.assertEqual(Path(cmd[-8]).name, 'parity-hip-ref')
        self.assertEqual(cmd[-1], '--suite')
        self.assertEqual(json.loads(stderr.getvalue())['weight_bits'], '0x3de12345')

    def test_incomplete_trace_rejected(self):
        self.run_fixture()
        (self.run / 'visits.psv').write_bytes(fixture())
        with self.assertRaisesRegex(ValueError, 'one complete visit frame'):
            cp.replay(self.dir, 'hip')

    def test_marker_rejected(self):
        self.run_fixture()
        (self.run / 'engine.log').write_text('CP04 capture: pos=2048 visit=7 layer=7\n')
        with self.assertRaisesRegex(ValueError, 'exactly one capture marker'):
            cp.replay(self.dir, 'hip')


if __name__ == '__main__':
    unittest.main()
