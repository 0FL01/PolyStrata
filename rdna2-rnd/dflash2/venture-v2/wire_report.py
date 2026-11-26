#!/usr/bin/env python3
"""Summarize guarded production/measure.py receipts without printing model inputs or IDs."""
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b), 1) if x != y),
                None if len(a) == len(b) else min(len(a), len(b)) + 1)


def guarded_result(run):
    guard = run.with_name(run.name + '.guard')
    if load(guard / 'stop-reasons.json'):
        raise ValueError(f'{run.name}: guard stopped')
    return load(run / 'result.json')


def report(run, control):
    result = guarded_result(run)
    guard = run.with_name(run.name + '.guard')
    free = [json.loads(line)['free_mib'] for line in (guard / 'vram.jsonl').read_text().splitlines()]
    if control and len(control['turns']) != len(result['turns']):
        raise ValueError('control has a different number of turns')
    for i, t in enumerate(result['turns']):
        row = dict(run=run.name, turn=i + 1, ids=len(t['ids']), finish=t['finish'],
                   ttft_ms=t['ttft_ms'], decode_tps=t['decode_tps'], gap_p90_ms=t['p90_ms'],
                   accepted=t['accepted'], offered=t['offered'], reused=t['reused'],
                   retained=t.get('kept_verified_rows'), verified=t.get('verified_rows'),
                   gpu_routes=t['gpu_routes'], all_routes=t['all_routes'], initial_slots=result['slots'],
                   peak_worker_rss_mib=result['peak_rss_kib'] / 1024,
                   min_device_free_mib=min(free), rounds=result['rounds_w1_lookup_draft_replayed'][i]
                   if result['rounds_w1_lookup_draft_replayed'] else None)
        if control:
            # The caller must supply a same-input/policy control; ID agreement alone is not a layout proof.
            c = control['turns'][i]
            row.update(first_different_output_1_based=first_difference(c['ids'], t['ids']),
                       same_route_counts=(c['gpu_routes'], c['all_routes']) == (t['gpu_routes'], t['all_routes']),
                       same_final_profile=control['saved_profile_sha256'] == result['saved_profile_sha256'])
        print(json.dumps(row, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--control', type=Path, help='compare against this wire output directory')
    args = parser.parse_args()
    reference = guarded_result(args.control) if args.control else None
    for directory in args.runs:
        report(directory, reference)
