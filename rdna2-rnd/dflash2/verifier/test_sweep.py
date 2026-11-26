import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

import sweep


class SweepTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_session_ids_and_context_limit(self):
        path = self.root / "session.bin"
        head = bytearray(240)
        struct.pack_into("<8sII", head, 0, b"STRSESS\x01", 1, 64)
        struct.pack_into("<5q", head, 64, 0x494E4B4C494E4732, 1, 8, 1000, 2304)
        struct.pack_into("<Q", head, 232, 3)
        path.write_bytes(head + struct.pack("<3i", 9, 12, 18))
        self.assertEqual(sweep.snapshot_prompt(path, 198, 3)[0], (9, 12, 18, 198))
        with self.assertRaisesRegex(ValueError, "no room"):
            sweep.snapshot_prompt(path, 198, 4)

    def test_first_difference_includes_length(self):
        self.assertIsNone(sweep.first_difference([1, 2], [1, 2]))
        self.assertEqual(sweep.first_difference([1, 2], [1, 3]), {"index": 1, "expected": 2, "observed": 3})
        self.assertEqual(sweep.first_difference([1, 2], [1]), {"index": 1, "expected": 2, "observed": None})

    def test_guarded_failure_preserves_attempt_and_does_not_publish(self):
        profile = self.root / "profile"
        profile.write_text("original")
        arm = self.root / "arm"
        req = dict(argv=["binary", "<profile>"], prompt="1,2")
        fixture = dict(tokens=[3, 4])
        copier = patch.object(sweep, "clone_profile", side_effect=shutil.copyfile)
        copier.start()
        self.addCleanup(copier.stop)
        with patch.object(sweep.subprocess, "Popen", return_value=Mock(wait=Mock(return_value=9))) as popen:
            with self.assertRaisesRegex(RuntimeError, "stopped sweep"):
                sweep.guarded_arm(arm, req, {}, fixture, profile, self.root / "gpu.lock", 30)
            args = popen.call_args.args[0]
            self.assertEqual(args[:3], ["flock", "--exclusive", str(self.root / "gpu.lock")])
            self.assertIn("--warn-amdgpu-faults", args)
        [partial] = list(arm.iterdir())
        self.assertEqual((partial / "profile.seed").read_text(), "original")
        self.assertFalse((partial / "complete.json").exists())

        clean = dict(tokens=[3, 4], width=1)
        with patch.object(sweep.subprocess, "Popen", return_value=Mock(wait=Mock(return_value=0))) as popen, \
             patch.object(sweep, "validate", return_value=clean):
            result = sweep.guarded_arm(arm, req, {}, fixture, profile, self.root / "gpu.lock", 30)
            self.assertEqual(result, clean)
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(sweep.guarded_arm(arm, req, {}, fixture, profile, self.root / "gpu.lock", 30), clean)
            self.assertEqual(popen.call_count, 1)
        [complete] = arm.glob("*/complete.json")
        self.assertNotEqual(partial, complete.parent)
        self.assertEqual(json.loads(complete.read_text())["result"], clean)

    def test_same_logical_request_reuses_legacy_footer(self):
        arm = self.root / "arm"
        attempt = arm / "attempt-old"
        attempt.mkdir(parents=True)
        req = dict(argv=["binary", "<profile>"], prompt="1,2", runner_sha256="old")
        old = dict(req, argv=["binary", str(attempt / "profile.seed")])
        (attempt / "request.json").write_text(json.dumps(old))
        result = dict(tokens=[3, 4])
        (attempt / "complete.json").write_text(json.dumps(dict(key=sweep.request_digest(req), result=result)))
        with patch.object(sweep, "validate", return_value=result), patch.object(sweep, "clone_profile") as copy:
            self.assertEqual(sweep.guarded_arm(arm, dict(req, receipt_version=1, runner_sha256="new"), {},
                                               None, None, None, 30), result)
            copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
