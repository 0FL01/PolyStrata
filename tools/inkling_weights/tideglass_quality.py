"""Natural completions on frozen short tasks, a two-read debugging task, and a restored repository history.

Run under run_guarded.py. No generated expression, command or tool argument is executed.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

from serve.frontend import (ChatTemplate, INKLING_END, INKLING_INVOKE, INKLING_MODEL, INKLING_STOP,
                            INKLING_TEXT, INKLING_THINK, InklingOutputParser, openai_to_messages)
from tools.strata_tokenizer import Tokenizer
from .validate_tasks import MANIFEST, validate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rdna2-rnd/kvq-v3'))
from worker import CONFIG, SAMPLING, Worker, read_ids  # noqa: E402

TOOL = dict(name='read_fixture', description='Read a named diagnostic artifact from the test bundle.',
            parameters=dict(type='object', properties=dict(name=dict(type='string')), required=['name']))
DEBUG = [
    dict(id='cpp-debug', prompt='Debug the C++ batch runner. Read test.log with read_fixture, then the file it names. '
         'Return JSON with key replacement: the corrected loop condition only (no spaces).',
         files={'test.log': 'last entry skipped: count=9, visits=8; source: batch.cc',
                'batch.cc': 'for (size_t row = 0; row + 1 < count; ++row) { process(rows[row]); }'},
         check={'kind': 'json', 'value': {'replacement': 'row<count'}}),
    dict(id='devops-debug', prompt='Сервис не находит свой сокет. Прочитай status.log, затем указанный там конфиг '
         'через read_fixture. Верни только JSON с полем socket: абсолютный путь, который должен использовать клиент.',
         files={'status.log': 'client ENOENT /run/atlas.sock; daemon ready, loaded /etc/atlas.conf',
                '/etc/atlas.conf': '[server]\nruntime_dir=/run/atlas-17\nsocket_name=local.sock'},
         check={'kind': 'json', 'value': {'socket': '/run/atlas-17/local.sock'}})]


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def task(w, case, tok, template, seed, effort, prefix=()):
    msgs = [dict(role='user', content=case['prompt'])]
    files = case.get('files', {})
    phases = {tok.encode(s, parse_special=True)[0]: s for s in
              (INKLING_END, INKLING_INVOKE, INKLING_MODEL, INKLING_STOP, INKLING_TEXT, INKLING_THINK)}
    seen, calls_n, repeated, thinking, generated, ngrams = [], 0, 0, 0, 0, 0
    start = time.monotonic()
    turns = []
    passed = False
    for i in range(4 if files else 1):
        norm, tools, kw = openai_to_messages(dict(messages=msgs, tools=[TOOL] if files else [], reasoning_effort=effort))
        rendered = template.render(norm, tools=tools, **kw)
        ids = tok.encode(rendered, parse_special=True)
        if prefix:
            # The source prefix already includes the history's own system/tool/effort declaration.
            ids = list(prefix) + ids[ids.index(tok.encode('<|message_user|>', parse_special=True)[0]):]
        ids_out, meta = w.gen(ids, case.get('max_new_tokens', 512), sampling=SAMPLING, seed=seed)
        if not ids_out or len(ids_out) != meta['produced'] or len(ids) != meta['prompt']:
            raise ValueError('invalid native completion counts')
        phase, span = None, []
        for t in ids_out:
            if t in phases:
                phase = phases[t]
            elif phase == INKLING_THINK:
                span.append(t)
        grams = Counter(tuple(span[j:j+16]) for j in range(0, len(span)-15))
        ngrams += sum(v-1 for v in grams.values())
        thinking += len(span)
        generated += len(ids_out)
        parser = InklingOutputParser(tools)
        eos = phases.get(ids_out[-1]) == INKLING_STOP and meta['finish'] == 'stop'
        events = parser.feed(tok.decode(ids_out[:-1] if eos else ids_out)) + parser.finish()
        answer = ''.join(e.text for e in events if e.kind == 'content')
        reasoning = ''.join(e.text for e in events if e.kind == 'reasoning')
        calls = [e.call for e in events if e.kind == 'tool_call']
        calls_n += len(calls)
        turns.append(dict(prompt_ids_sha256=sha256(','.join(map(str,ids)).encode()).hexdigest(),
                          generated_ids=ids_out, finish=meta['finish'], reasoning=len(span),
                          cached=meta['resume'], read=meta['read'], answer=answer))
        if not eos:
            break
        if not calls:
            passed = validate(case, answer)[0] and set(seen) == files.keys()
            break
        if not files or len(calls) != 1:
            break
        c = calls[0]
        if c.name != TOOL['name'] or not isinstance(c.arguments, dict) or set(c.arguments) != {'name'}:
            break
        key = c.arguments['name']
        if not isinstance(key, str) or key not in files:
            break
        repeated += int(key in seen)
        seen.append(key)
        cid = f'fixture_{i}'
        msgs += [dict(role='assistant', content=answer, reasoning_content=reasoning,
                      tool_calls=[dict(id=cid, type='function', function=dict(name=c.name, arguments=c.arguments))]),
                 dict(role='tool', tool_call_id=cid, content=files[key])]
    wall = time.monotonic()-start
    return dict(id=case['id'], seed=seed, effort=effort, success=passed, task_s=wall,
                time_to_correct_s=wall if passed else None, generated_tokens=generated, reasoning_tokens=thinking,
                repeated_reasoning_16grams=ngrams, calls=calls_n, dispatched=seen, repeated_calls=repeated,
                iterations=len(turns), natural_eos=eos, exhausted=meta['finish']=='length', turns=turns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'exe', 'profile', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--history', type=Path, help='197K CPython turn-1.prompt.csv: build and test a distinct SAVE')
    args = p.parse_args()
    os.umask(0o077)
    cfg = json.loads(CONFIG.read_bytes())
    tok = Tokenizer.from_dir(Path(cfg['tokenizer']))
    template = ChatTemplate(Path(cfg['tokenizer'])/'chat_template.jinja')
    corpus = json.loads(MANIFEST.read_bytes())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    w = Worker(out, binary=args.exe.resolve(), profile=args.profile, stable=False,
               extra_args=['--model', args.model.resolve(strict=True)])
    try:
        with (out/'scores.jsonl').open('x', buffering=1) as scores:
            def run(c, seed, effort, prefix=()):
                r = task(w, c, tok, template, seed, effort, prefix)
                dump(out/f'{c["id"]}-{seed}-{effort}.json', r)
                del r['turns']
                scores.write(json.dumps(r)+'\n')
            if args.history:
                original = read_ids(args.history)
                prefix = original[:197021]
                if len(original) != 197135 or tok.decode(prefix[-1:]) != INKLING_STOP:
                    raise ValueError('unexpected CPython history')
                w.gen(prefix, 1)
                session = out/'history.session'
                saved = w.save(session)
                # Switch to an unrelated short request before restoring.
                w.gen(tok.encode('Different history', parse_special=False), 1)
                restored = w.restore(session)
                if saved['pos'] != len(prefix) or restored['pos'] != saved['pos'] or restored['bytes'] != saved['bytes']:
                    raise ValueError('SAVE/RESTORE mismatch')
                dump(out/'session.json', dict(saved=saved, restored=restored,
                     source_sha256=sha256(args.history.read_bytes()).hexdigest()))
                case = dict(id='197k-repository', prompt='In the provided CPython source bundle, how does '
                            'TarFile.gettarinfo determine a symbolic link target and the size of a non-regular '
                            'entry? Return only JSON with keys link_reader (qualified function name) and size (integer).',
                            check={'kind':'json', 'value':{'link_reader':'os.readlink','size':0}})
                for seed in (42,79):
                    if seed != 42:
                        w.restore(session)
                    run(case, seed, 'low', prefix)
            else:
                for seed, effort in ((42,'low'), (79,'low'), (42,'high'), (79,'high')):
                    cases = corpus['tasks'] if (seed,effort)==(42,'low') else [c for c in corpus['tasks'] if c.get('smoke_candidate')]
                    for c in [*cases, *DEBUG]:
                        run(c, seed, effort)
    finally:
        if w.close() != 0:
            raise RuntimeError('engine failed to shut down cleanly')


if __name__ == '__main__':
    main()
