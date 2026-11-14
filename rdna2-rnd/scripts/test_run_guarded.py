"""Offline checks of the mandatory fail-closed shutdown boundary; no GPU commands."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from run_guarded import Guard


class CleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
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


if __name__ == "__main__":
    unittest.main()
