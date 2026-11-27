"""Export measured rows only; keep raw model text, private prompts and journal records outside Git."""
import argparse
import json
from pathlib import Path
from statistics import median


def load(path):
    return json.loads(path.read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    perf, quality, sessions = [], {}, {}
    for arm in 'AB':
        for n in (2048, 16384, 65536, 197000):
            for rep in range(1, 3 if n <= 16384 else 2):
                d = args.work/'pairs'/f'{arm}-{n}-{rep}'
                assert load(d/'guard/stop-reasons.json') == []
                perf.append(dict(arm=arm, rep=rep, **load(d/'result.json')))
        records = []
        for suite in ('short', 'long', 'repair'):
            d = args.work/f'quality-{arm}-{suite}'
            assert load(d.with_name(d.name+'-guard')/'stop-reasons.json') == []
            rows = [json.loads(line) for line in (d/'scores.jsonl').read_text().splitlines()]
            assert len(rows) == {'short': 52, 'long': 2, 'repair': 4}[suite]
            records.extend(rows)
            if suite == 'long':
                sessions[arm] = load(d/'session.json')
        quality[arm] = records
    assert len({r['exe_sha256'] for r in perf}) == 1
    for n in {r['prompt_tokens'] for r in perf}:
        assert len({(r['prompt_sha256'], r['forced_sha256']) for r in perf if r['prompt_tokens'] == n}) == 1
    def key(r):
        return (r['id'], r['effort'], r['seed'])
    assert list(map(key, quality['A'])) == list(map(key, quality['B']))
    matched = [i for i, (a,b) in enumerate(zip(quality['A'], quality['B'], strict=True)) if a['success'] and b['success']]
    summary = {}
    for arm, rows in quality.items():
        summary[arm] = dict(passed=sum(r['success'] for r in rows), trials=len(rows),
                            matched_successes=len(matched), matched_median_time_to_correct_s=median(rows[i]['time_to_correct_s'] for i in matched),
                            total_task_wall_s=sum(r['task_s'] for r in rows), generated=sum(r['generated_tokens'] for r in rows),
                            reasoning=sum(r['reasoning_tokens'] for r in rows), exhausted=sum(r['exhausted'] for r in rows),
                            repeated_16grams=sum(r['repeated_reasoning_16grams'] for r in rows))
    with args.output.open('x') as f:
        json.dump(dict(performance=perf, quality=quality, quality_summary=summary, sessions=sessions), f, indent=2)
        f.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
