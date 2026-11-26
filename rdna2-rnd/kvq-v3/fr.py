#!/usr/bin/env python3
"""Free-running graded agent trajectories on restored long histories (KVQ V3).

Same task contract as the archived native harness: recorded turn-1 prompt (restored history + warm rows + question),
sampled T=1/top_p=1/top_k=0/min_p=0 with a seed, CAPS per turn, allowlisted read_case fixtures, original graders.
Each (arm, task, seed) is one fresh worker at fixed cap and zero swaps unless --swaps is given.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import Worker, read_ids, csv, KVQ, SAMPLING

RUNS = KVQ.parent
RESCUE = RUNS / 'venture-v2/fixture-rescue'
sys.path.insert(0, str(RESCUE))
import workloads as w   # noqa: E402

ROOT = Path.home() / 'ai/agent-frameworks/PolyStrata'
TOKENIZER = Path.home() / '.config/strata/inkling2-mini/tokenizer'
DECL = RUNS / 'dflash2-production-transaction/api-v2/installed-w1/turn-1-request.json'
FIX = RESCUE / 'fixtures'
CAPS = (224, 320, 512, 384, 384)
LIMIT = 6
PROMPTS = {
    ('64k', 'coding'): RUNS / 'dflash-long-warmed-production-42/64k-coding-42/native-w1/turn-1.prompt.csv',
    ('64k', 'reasoning'): RUNS / 'dflash-long-warmed-production-42/64k-reasoning-42/native-w1/turn-1.prompt.csv',
    ('128k', 'coding'): RUNS / 'venture-v2/lookup-epoch/qualify-128k-coding-79/w1/turn-1.prompt.csv',
    ('128k', 'reasoning'): RUNS / 'venture-v2/lookup-epoch/qualify-128k-reasoning-79/w1/turn-1.prompt.csv',
    ('197k', 'coding'): RUNS / 'dflash-long-cold-production-42/197k-coding-42/native-w1/turn-1.prompt.csv',
    ('197k', 'reasoning'): RUNS / 'dflash-long-cold-production-42/197k-reasoning-42/native-w1/turn-1.prompt.csv',
}


class Frontend:
    def __init__(self):
        sys.path.insert(0, str(ROOT))
        from serve import frontend
        from tools.strata_tokenizer import Tokenizer
        self.api = frontend
        self.tok = Tokenizer.from_dir(TOKENIZER)
        self.template = frontend.ChatTemplate(TOKENIZER / 'chat_template.jinja')
        sp = lambda m: self.encode(m)[0]
        self.user, self.tool, self.model = sp('<|message_user|>'), sp('<|message_tool|>'), sp(frontend.INKLING_MODEL)
        self.turn_ids = {self.user, self.tool, self.model, sp('<|message_system|>')}
        self.stop = sp(frontend.INKLING_STOP)
        self.invoke = sp(frontend.INKLING_INVOKE)
        self.phases = {sp(frontend.INKLING_THINK): 'reasoning', sp(frontend.INKLING_TEXT): 'answer',
                       self.invoke: 'tool', self.model: 'framing', sp(frontend.INKLING_END): 'framing',
                       self.stop: 'framing'}
        self.declarations = json.loads(DECL.read_text())['tools']
        _, self.tools, _ = frontend.openai_to_messages({'messages': [], 'tools': self.declarations})
        assert self.stop == 200006

    def encode(self, text):
        return self.tok.encode(text, parse_special=True)

    def segment(self, messages, marker):
        msgs, tools, opts = self.api.openai_to_messages({'messages': messages, 'tools': self.declarations})
        ids = self.encode(self.template.render(msgs, tools, **opts))
        return ids[ids.index(marker):]

    def parse(self, ids):
        raw = self.tok.decode(ids[:-1] if ids and ids[-1] == self.stop else ids)
        parser = self.api.InklingOutputParser(self.tools)
        events = parser.feed(raw) + parser.finish()
        answer = ''.join(e.text for e in events if e.kind == 'content')
        reasoning = ''.join(e.text for e in events if e.kind == 'reasoning')
        calls = [e.call for e in events if e.kind == 'tool_call']
        counts, phase = {}, 'framing'
        for t in ids:
            if t in self.phases:
                phase = self.phases[t]
                counts['framing'] = counts.get('framing', 0) + 1
            else:
                counts[phase] = counts.get(phase, 0) + 1
        return answer, reasoning, calls, counts

    def reply(self, call, data):
        text = json.dumps(data, separators=(',', ':'), ensure_ascii=True)
        assistant = dict(role='assistant', tool_calls=[dict(id=call.id, type='function',
                                                            function=dict(name=call.name, arguments='{}'))])
        msg = dict(role='tool', tool_call_id=call.id, content=text)
        return self.segment([assistant, msg], self.tool), msg


def run(front, where, save, size, task, seed, cap, swaps, env=None, binary=None):
    where.mkdir(parents=True, exist_ok=True)
    r = dict(task=task, seed=seed, size=size, save=str(save), turns=[], grade=None)
    kw = dict(cap_mib=cap, swaps=swaps, env=env)
    if binary:
        kw['binary'] = binary
    wk = Worker(where / 'home', **kw)
    try:
        r['restore'] = wk.restore(save)
        prompt = read_ids(PROMPTS[(size, task)])
        held = prompt[:r['restore']['pos']]
        given, total_calls = set(), 0
        keys = list(w.CASES[task]['fixtures'])
        t0 = time.time()
        for i, tcap in enumerate(CAPS, 1):
            assert prompt[:len(held)] == held
            at = next(j for j in range(len(prompt) - 1, 0, -1) if prompt[j] in front.turn_ids)
            ids, meta = wk.gen(prompt, tcap, SAMPLING, seed=seed, turn_at=at)
            (where / f'turn-{i}.output.csv').write_text(csv(ids))
            turn = dict(turn=i, prompt=len(prompt), out=len(ids), meta=meta, calls=[])
            r['turns'].append(turn)
            held = prompt + ids[:-1]
            length = meta['finish'] != 'stop' or not ids or ids[-1] != front.stop
            try:
                answer, reasoning, calls, turn['phases'] = front.parse(ids)
            except (ValueError, TypeError, RecursionError) as exc:
                r['grade'] = {'pass': False, 'class': 'length' if length else 'malformed-tool', 'reason': str(exc)}
                break
            turn['answer'] = answer[-400:]
            if length:
                r['grade'] = {'pass': False, 'class': 'length'}
                break
            if len(calls) != ids.count(front.invoke):
                r['grade'] = {'pass': False, 'class': 'malformed-tool'}
                break
            if not calls:
                r['grade'] = (w.grade(task, answer) if all(k in given for k in keys) else
                              {'pass': False, 'class': 'tool-evidence'})
                break
            if i == len(CAPS) or len(calls) > 2 or total_calls + len(calls) > LIMIT:
                r['grade'] = {'pass': False, 'class': 'retry-cap'}
                break
            prior, linked = set(given), []
            for c in calls:
                key, data = w.call_result(task, prior, c, FIX)
                tail, _ = front.reply(c, data)
                linked.extend(tail)
                turn['calls'].append(dict(name=c.name, arguments=c.arguments, fixture=key))
                total_calls += 1
                if key is not None:
                    given.add(key)
            prompt = held + ids[-1:] + linked
        r['task_wall_s'] = time.time() - t0
        r['decode_ms'] = sum(t['meta']['decode_ms'] for t in r['turns'])
        r['out_ids'] = sum(t['out'] for t in r['turns'])
        r['reasoning_ids'] = sum(t.get('phases', {}).get('reasoning', 0) for t in r['turns'])
        r['tool_calls'] = sum(len(t['calls']) for t in r['turns'])
        r['bad_calls'] = sum(1 for t in r['turns'] for c in t['calls'] if c['fixture'] is None)
    except Exception as exc:
        r['grade'] = {'pass': False, 'class': 'infrastructure', 'reason': repr(exc)}
    finally:
        r['rc'] = wk.close()
    (where / 'result.json').write_text(json.dumps(r, indent=1, default=str))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', required=True)
    ap.add_argument('--task', action='append', required=True)
    ap.add_argument('--seeds', required=True, help='e.g. 1-8')
    ap.add_argument('--arm', action='append', required=True, help='name=SAVE')
    ap.add_argument('--cap', type=int, default=5200)
    ap.add_argument('--swaps', type=int, default=0)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split('-'))
    front = Frontend()
    arms = [x.split('=', 1) for x in a.arm]
    out = Path(a.out)
    for seed in range(lo, hi + 1):
        for task in a.task:
            order = arms if (seed + len(task)) % 2 == 0 else arms[::-1]
            for name, save in order:
                where = out / f'{a.size}-{task}-{seed}' / name
                if (where / 'result.json').exists():
                    old = json.loads((where / 'result.json').read_text())
                    if (old.get('grade') or {}).get('class') != 'infrastructure':
                        continue
                    (where / 'result.json').rename(where / f'result.infra.{int(time.time())}.json')
                    where = where.parent / (name + '.retry')
                    if (where / 'result.json').exists():
                        continue
                r = run(front, where, save, a.size, task, seed, a.cap, a.swaps)
                g = r['grade'] or {}
                print(f"{a.size} {task} seed {seed} {name:8s} pass={g.get('pass')} class={g.get('class')} "
                      f"out={r.get('out_ids')} reason={r.get('reasoning_ids')} calls={r.get('tool_calls')} "
                      f"bad={r.get('bad_calls')} wall={r.get('task_wall_s', 0):.1f}s", flush=True)


if __name__ == '__main__':
    main()
