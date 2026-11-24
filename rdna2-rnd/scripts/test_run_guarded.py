"""Offline checks of guard classification and shutdown; no GPU commands."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from run_guarded import Guard


class CleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch('run_guarded.subprocess.run', return_value=Mock(stdout='-- cursor: test\n')):
            self.guard = Guard(Path(self.temp.name))
        self.guard.terminate = Mock()
        self.guard.watcher = Mock()
        self.guard.watcher.is_alive.return_value = False

    def test_join_before_final_check(self):
        calls = []
        self.guard.watcher.join.side_effect = lambda **_: calls.append("joined")
        self.guard.check = lambda: calls.append("final")
        self.guard.__exit__()
        self.assertEqual(calls, ["joined", "final"])
        self.assertTrue(self.guard.stop.is_set())

    def test_final_fault_fails_closed(self):
        self.guard.check = Mock(side_effect=RuntimeError("late fault"))
        with self.assertRaisesRegex(RuntimeError, "late fault"):
            self.guard.__exit__()
        self.assertTrue(self.guard.unsafe.is_set())

    def test_unfinished_monitor_fails_closed(self):
        self.guard.watcher.is_alive.return_value = True
        self.guard.check = Mock()
        with self.assertRaisesRegex(RuntimeError, "monitor did not finish"):
            self.guard.__exit__()
        self.guard.check.assert_not_called()

    def test_boot_memory_fault_is_only_a_warning_when_enabled(self):
        msg = 'amdgpu 0000:2d:00.0: [gfxhub] page fault (src_id:0 ring:24)'
        with patch('run_guarded.subprocess.run', return_value=Mock(stdout=json.dumps({'MESSAGE': msg}))):
            with self.assertRaisesRegex(RuntimeError, 'page fault'):
                self.guard.boot_clean()
            self.guard.warn_amdgpu_faults = True
            self.guard.boot_clean()
        self.assertEqual([x['message'] for x in self.guard.warnings.values()], [msg])

    def test_live_warning_does_not_hide_reset_or_count_twice(self):
        faults = ['amdgpu: [gfxhub] page fault', 'amdgpu: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801A31',
                  'amdgpu: MORE_FAULTS: 0x1', 'amdgpu: PERMISSION_FAULTS: 0x3', 'amdgpu: MAPPING_ERROR: 0x0']
        entries = [json.dumps({'MESSAGE': m, '__CURSOR': str(i)}) for i, m in enumerate(faults)]
        self.guard.warn_amdgpu_faults = True
        journal = Mock(stdout='\n'.join(entries))
        query = lambda argv, **_: journal if '-k' in argv else Mock(stdout='')
        with patch('run_guarded.subprocess.run', side_effect=query), patch.object(
            Path, 'read_text', autospec=True,
            side_effect=lambda p: str(4 << 30) if p.name == 'mem_info_vram_total' else '0',
        ):
            self.guard.check()
            self.guard.check()
            self.assertFalse(self.guard.unsafe.is_set())
            self.assertEqual(len(self.guard.warnings), len(faults))
            journal.stdout += '\n' + json.dumps({'MESSAGE': 'amdgpu: GPU reset after VM fault'})
            with self.assertRaisesRegex(RuntimeError, 'fresh kernel fault'):
                self.guard.check()
        self.assertTrue(self.guard.unsafe.is_set())


if __name__ == "__main__":
    unittest.main()
