#!/usr/bin/env python3
"""Summarize private trajectory receipts and show prompt divergence beside output divergence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run import R, dump, p90


def first_diff(a, b):
    at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return None if a == b else dict(index=at+1, reference=a[at] if at < len(a) else None,
                                    candidate=b[at] if at < len(b) else None)


def summarize(label):
    d = R / 'trajectory' / label
    r = json.loads((d / 'trajectory.json').read_text())
    plan = json.loads((d / 'plan.json').read_text())
    reasons = json.loads((d / 'guard/stop-reasons.json').read_text())
    warnings = json.loads((d / 'guard/warnings.json').read_text())
    vram = [json.loads(s)['free_mib'] for s in (d / 'guard/vram.jsonl').read_text().splitlines()]
    turns = r['turns']
    intervals = [x for t in turns for x in t['step_ms']]
    for i, event in enumerate(r['tools']):
        if i + 1 >= len(turns):
            break
        t = turns[i+1]['messages']
        assert t[-1]['role'] == 'tool' and json.loads(t[-1]['content']) == event['result']
        assert t[-1]['tool_call_id'] == t[-2]['tool_calls'][0]['id']
        assert json.loads(t[-2]['tool_calls'][0]['function']['arguments']) == event['arguments']
    if r['grade']['pass']:
        assert len(turns) == 3 and len(r['tools']) == 2 and r['grade']['checks_passed'] == r['grade']['checks'] == 118
    summary = dict(label=label, seed=r['seed'], mode=r['mode'], cap_mib=plan['fixed_expert_kv_mib'],
                   k=plan['k'] if r['mode'] in ('live', 'dspark') else 0, serial_verify=plan['serial_verify'],
                   grade=r['grade'], reads=len({t['case_id'] for t in r['tools'] if t['case_id'] is not None}),
                   dispatch_attempts=len(r['tools']),
                   outputs=[len(t['output_ids']) for t in turns], finishes=[t['finish'] for t in turns],
                   reasoning=[len(t['ids_by_phase']['reasoning']) for t in turns],
                   prompts=[len(t['prompt_ids']) for t in turns], slots=[t['slots'] for t in turns],
                   ttft_s=[t['ttft_s'] for t in turns], ttft_p90_s=p90([t['ttft_s'] for t in turns]),
                   cold_ttft_s=[t['cold_ttft_s'] for t in turns], step_p90_ms=p90(intervals),
                   pooled_decode_ids_per_s=1000*len(intervals)/sum(intervals) if intervals else None,
                   startup_s=r['startup_s'], compute_s=r['compute_s'], task_s=r['task_s'],
                   wall_to_correct_s=r['task_s'] if r['grade']['pass'] else None,
                   offered=sum(t['offered'] for t in turns), accepted=sum(t['accepted'] for t in turns),
                   guard_stops=reasons, fresh_warnings=sum(w['phase'] != 'boot' for w in warnings),
                   min_free_mib=min(vram), binary_sha256=plan['exe_sha256'],
                   tap_files=[t['tap_file'] for t in turns if 'tap_file' in t])
    return r, summary


def compare(ref, cand):
    rows = []
    for x, y in zip(ref['turns'], cand['turns']):
        p = first_diff(x['prompt_ids'], y['prompt_ids'])
        rows.append(dict(turn=x['turn'], same_prompt=p is None, prompt_first_diff=p,
                         output_first_diff=first_diff(x['output_ids'], y['output_ids']),
                         reasoning_first_diff=first_diff(x['ids_by_phase']['reasoning'], y['ids_by_phase']['reasoning']),
                         output_identical=x['output_ids'] == y['output_ids'],
                         reasoning_identical=x['ids_by_phase']['reasoning'] == y['ids_by_phase']['reasoning'],
                         same_slots=x['slots'] == y['slots'],
                         same_layout=x['layout'] == y['layout'] if 'layout' in x and 'layout' in y else None))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('labels', nargs='+', help='put the reference first, use one report per sampled seed')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    legs = [summarize(label) for label in a.labels]
    ref = legs[0][0]
    report = []
    for r, summary in legs:
        assert r['seed'] == ref['seed'], 'compare only the same seed'
        summary['vs_reference'] = compare(ref, r)
        report.append(summary)
    if a.out:
        dump(a.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
