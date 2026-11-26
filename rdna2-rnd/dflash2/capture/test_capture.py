import json
from pathlib import Path
import tempfile
import unittest

from matrix import native_predictions, stats
from run import FEATURE_BYTES, K, RECORD, capture_receipt, validate_dspark, validate_tape


class CaptureTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.prompt = [1, 2]
        self.tokens = list(range(10, 22))

    def test_tape_shape_and_tail(self):
        path = self.folder / 'tap.dft1'
        labels = self.prompt + self.tokens[:-1]
        rows = [(i, token, int(i >= len(self.prompt))) for i, token in enumerate(labels)]

        def dump(rows):
            path.write_bytes(b'DFT1' + b''.join(RECORD.pack(*row) + bytes(FEATURE_BYTES) for row in rows))

        dump(rows)
        validate_tape(path, self.prompt, self.tokens)
        dump(rows + [(len(labels), self.tokens[-1], 1)])
        with self.assertRaisesRegex(ValueError, 'extra/truncated'):
            validate_tape(path, self.prompt, self.tokens)
        for row in [(2, 10, 0), (2, 11, 1), (3, 10, 1)]:
            dump(rows[:2] + [row] + rows[3:])
            with self.assertRaisesRegex(ValueError, 'position, token ID or phase'):
                validate_tape(path, self.prompt, self.tokens)

    def test_dspark_requires_exact_anchor_and_width(self):
        path = self.folder / 'proposals.log'
        full = len(self.tokens) - K - 1
        lines = [f'{i + len(self.prompt)} {t}' + ' 7:1.0:1.0' * min(K, len(self.tokens) - i - 2)
                 for i, t in enumerate(self.tokens[:-2])]
        path.write_text('\n'.join(lines))
        self.assertEqual(validate_dspark(path, len(self.prompt), self.tokens), [[7] * K] * full)
        path.write_text('\n'.join(lines[:-1]))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            validate_dspark(path, len(self.prompt), self.tokens)
        path.write_text('\n'.join(lines) + ' 7:1.0:1.0')
        with self.assertRaisesRegex(ValueError, 'width'):
            validate_dspark(path, len(self.prompt), self.tokens)

    def test_uninstrumented_exit_is_not_a_capture(self):
        guard = self.folder / 'guard'
        guard.mkdir()
        (guard / 'stop-reasons.json').write_text('[]')
        (guard / 'command.log').write_text('prompt 2 tokens 12.0 tok/s, generated 12 tokens 3.0 tok/s (length)\n')
        with self.assertRaisesRegex(ValueError, 'DFT1 hook completion'):
            capture_receipt(self.folder, self.prompt, self.tokens, {}, len(self.tokens), True)

    def test_native_truth_and_prefix_walk(self):
        path = self.folder / 'steps.jsonl'
        truth = self.tokens[1:K + 1]
        row = dict(mode='all-positions', start=2, anchor=self.tokens[0], accepted=K, draft_tokens=truth)
        path.write_text(json.dumps(row) + '\n')
        self.assertEqual(native_predictions(path, 2, self.tokens, [truth]), [truth])
        path.write_text(json.dumps(row | {'accepted': 2}) + '\n')
        with self.assertRaisesRegex(ValueError, 'accepted prefix'):
            native_predictions(path, 2, self.tokens, [truth])
        result = stats([1, 7, 0, 2])
        self.assertEqual(result['all-positions']['sum_accepted'], 10)
        self.assertEqual(result['closed-loop']['sum_accepted'], 3)


if __name__ == '__main__':
    unittest.main()
