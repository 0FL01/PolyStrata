#!/usr/bin/env python3
"""CPU-only tests with hand-written answers, no model output or model file required."""

import json
import unittest

from corpus import Prose, checked_segments, check_ids, id_bytes
from validate_tasks import MANIFEST, cases, validate


PYTHON_REFERENCES = {
    "2k-py-01": "abs(x - y)",
    "2k-py-02": "10 < n < 30",
    "2k-py-03": "min(hi, max(lo, v))",
    "2k-py-04": "n // 4 * 4",
    "2k-py-05": "(pages + per_box - 1) // per_box",
    "2k-py-06": "(n % 2 == 0) != (n % 3 == 0)",
}


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_bytes())
        cls.cases = {c["id"]: c for c in cases(cls.manifest)}

    def test_shape(self):
        segments = list(checked_segments(self.manifest))
        self.assertEqual(len(self.manifest["tasks"]), 32)
        self.assertEqual(len(self.cases), 56)  # 32 short tasks + 24 Orca prompts
        self.assertEqual(len(segments), 74)  # 8 x (prefix + target), 2 forced, 56 prompts
        self.assertEqual(len({s["ids_path"] for s, _ in segments}), len(segments))

    def test_positive_and_negative_for_every_case(self):
        for key, case in self.cases.items():
            with self.subTest(case=key):
                check = case["check"]
                if check["kind"] == "python_expr":
                    answer = PYTHON_REFERENCES[key]
                elif check["kind"] == "json":
                    answer = json.dumps(check["value"])
                else:
                    answer = check["answers"][0]
                self.assertTrue(validate(case, answer)[0])
                self.assertFalse(validate(case, "That's not possible.")[0])

    def test_python_semantics_and_disallowed_forms(self):
        case = self.cases["2k-py-01"]
        self.assertTrue(validate(case, "```python\nx - y if x > y else y - x\n```")[0])
        for candidate in ("abs(y)", "abs(x - y) if True else print(1)", "abs(x-y) if True else x.__class__",
                          "x ** 2", "[x for x in (1, 2)]", "lambda: abs(x-y)", "abs(x-y); 0",
                          "1 // 0", "1" * 4097, "(" * 300 + "x" + ")" * 300):
            with self.subTest(candidate=candidate[:80]):
                self.assertFalse(validate(case, candidate)[0])

    def test_structural_rust(self):
        case = self.cases["2k-rs-06"]
        self.assertTrue(validate(case, "```rust\nb . saturating_add ( a )\n```")[0])
        self.assertFalse(validate(case, "b.wrapping_add(a)")[0])
        self.assertFalse(validate(case, "a.saturating_add(b); panic!()")[0])

    def test_json_strictness(self):
        case = self.cases["2k-json-01"]
        self.assertTrue(validate(case, '  {"items": [], "ready": true}  ')[0])
        for text in ('{"items":[],"ready":1}', '{"items":[],"ready":true,"ready":true}',
                     '{"items":[],"ready":NaN}', '{"items":[],"ready":true,"extra":0}',
                     '```json\n{"items":[],"ready":true}\n```'):
            with self.subTest(answer=text):
                self.assertFalse(validate(case, text)[0])

    def test_public_prose_and_ids(self):
        parser = Prose()
        parser.feed('<p>Лишнее</p><div class="mw-parser-output"><div><p>Русский&nbsp;текст<sup>[1]</sup>'
                    '</p></div><p>Дальше</p></div><p>Лишнее</p>')
        parser.close()
        self.assertEqual(parser.paragraphs, ["Русский текст", "Дальше"])
        self.assertEqual(id_bytes([0, 12, 199997]), b"0,12,199997")
        for ids in ([], [-1], [200006], [200058]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                check_ids(ids, {200006})


if __name__ == "__main__":
    unittest.main()
