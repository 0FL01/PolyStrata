"""Use hand-written model frames to verify the two tool returns and final grader, not model quality."""
import json
import re
import unittest

from tools.inkling_weights import tideglass_quality as q


class QualityTest(unittest.TestCase):
    def test_three_turn_debug(self):
        case = q.DEBUG[0]
        class Tokenizer:
            def __init__(self):
                self.tokens = {}

            def encode(self, text, **kwargs):
                parts = re.findall(r'<\|[^|]+\|>|.', text, re.S)
                return [self.tokens.setdefault(t, len(self.tokens)) for t in parts]

            def decode(self, ids):
                table = {i:t for t,i in self.tokens.items()}
                return ''.join(table[i] for i in ids)

        class Template:
            def render(self, messages, **kwargs):
                return json.dumps(messages)

        tokenizer, template = Tokenizer(), Template()
        replies = [q.INKLING_MODEL+'read_fixture'+q.INKLING_INVOKE+json.dumps(
            {'name':'read_fixture', 'args':{'name':key}}) for key in case['files']]
        replies.append(q.INKLING_MODEL+q.INKLING_TEXT+json.dumps(case['check']['value']))

        class Engine:
            def gen(self, ids, cap, **kwargs):
                answer = tokenizer.encode(replies.pop(0)+q.INKLING_END+q.INKLING_STOP, parse_special=True)
                return answer, dict(produced=len(answer), prompt=len(ids), finish='stop', resume=0, read=len(ids))

        r = q.task(Engine(), case, tokenizer, template, 42, 'low')
        self.assertTrue(r['success'])
        self.assertEqual((r['iterations'], r['calls'], r['dispatched']), (3, 2, list(case['files'])))
