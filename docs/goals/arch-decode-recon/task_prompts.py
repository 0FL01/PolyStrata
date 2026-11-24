"""Render short English task fixtures with the production template; never execute model output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate
from tools.strata_tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--tokenizer', type=Path, default=Path.home()/'.config/strata/inkling2-mini/tokenizer')
    p.add_argument('--long-code', action='store_true', help='also generate an independent 65536-ID source-review request')
    a = p.parse_args()
    a.out.mkdir(mode=0o700, exist_ok=False)
    corpus = json.loads((ROOT/'docs/goals/inkling2-weights/corpus-manifest.json').read_text())
    tasks = {}
    for category in ('logic','tool','python'):
        case = next(x for x in corpus['tasks'] if x['category']==category)
        tasks[case['id']] = [dict(role='user', content=case['prompt'])]
    tasks['tool-result-replay'] = [dict(role='user', content='Which records matched blue vase? Give their IDs as a JSON array.'),
        dict(role='assistant',content='',tool_calls=[dict(id='call_1',type='function',function=dict(name='catalog.find',arguments={'query':'blue vase'}))]),
        dict(role='tool', name='catalog.find', tool_call_id='call_1', content='{"records":[{"id":17},{"id":29}]}')]
    template = ChatTemplate(a.tokenizer/'chat_template.jinja')
    tok = Tokenizer.from_dir(a.tokenizer)
    summary = []
    for key,messages in tasks.items():
        text = template.render(messages, reasoning_effort='high')
        ids = tok.encode(text, parse_special=True)
        csv = ','.join(map(str,ids)).encode()
        (a.out/f'{key}.csv').write_bytes(csv)
        summary.append(dict(id=key,length=len(ids),sha256=hashlib.sha256(csv).hexdigest()))
    if a.long_code:
        # Keep the real chat envelope and request tail intact. Unlike the length sweep, no repeated filler.
        stub = 'REPLACE_SOURCE_HERE'
        text = template.render([dict(role='user',content='Review this inference engine source:\n'+stub+
            '\nExplain when expert slots change and how an asynchronous upload becomes usable. Cite the functions involved.')],
            reasoning_effort='high')
        left,right = text.split(stub)
        prefix,tail = tok.encode(left,parse_special=True),tok.encode(right,parse_special=True)
        code = tok.encode((ROOT/'src/poly/engine.cpp').read_text()+'\n'+(ROOT/'src/poly/inkling2.cu').read_text())
        needed = 65536-len(prefix)-len(tail)
        if len(code)<needed:
            raise ValueError('not enough independent source tokens')
        ids = prefix+code[:needed]+tail
        csv = ','.join(map(str,ids)).encode()
        (a.out/'long-code-review.csv').write_bytes(csv)
        summary.append(dict(id='long-code-review',length=len(ids),sha256=hashlib.sha256(csv).hexdigest()))
    (a.out/'index.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
