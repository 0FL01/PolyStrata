#!/usr/bin/env python3
"""Summarize paired CPU replay measurements; positive deltas mean savings."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
from statistics import mean, median


def report(path, tokens):
    log = path.with_suffix('.log').read_text()
    if 'timed_major_faults=0 ' not in log or 'parity=bitwise' not in log:
        raise ValueError(f'no successful bitwise/no-major-fault finish in {path}')
    cal = [tuple(map(int, fields)) for fields in re.findall(r'read_cal_ns cold=(\d+) warm=(\d+)', log)]
    with path.open() as f:
        groups = defaultdict(list)
        for v in csv.DictReader(f):
            groups[v['pair'], v['pos'], v['visit'], v['round']].append(v)
    if not groups:
        raise ValueError(f'no trials in {path}')
    per_round, prep = defaultdict(list), defaultdict(list)
    keys = ('whole', 'prev', 'compute', 'after_prev', 'after_prev_join_clipped')
    for (arm, pos, visit, _), rows in sorted(groups.items()):
        if [r['part'] for r in rows] != ['0', '1', '2', '3']:
            raise ValueError(f'incomplete pair {path}: {arm}/{pos}/{visit}')
        if arm == 'cold':
            a, b = (rows[0], rows[3]), (rows[1], rows[2])
        else:
            a, b = ([r for r in rows if r['arm'] == kind] for kind in ('cold', arm))
            if len(a) != 2 or len(b) != 2:
                raise ValueError(f'unbalanced pair {path}: {arm}/{pos}/{visit}')
        whole, prev, compute = (mean(int(r[f'{key}_ns']) for r in a) - mean(int(r[f'{key}_ns']) for r in b)
                                for key in keys[:3])
        x = rows[0]
        after = whole - prev
        clipped = after - max(0, compute - int(x['trace_join_ns']))
        per_round[arm, pos, visit].append(((whole, prev, compute, after, clipped), x))
        for y in b:
            start, end = int(y['prep_start_ns']), int(y['prep_end_ns'])
            if start >= 0:
                filled = y['prep_lines'] == y['expected_lines']
                prep[arm].append((end - start, filled and end <= int(y['deadline_ns'])))
    cases = defaultdict(list)
    for (arm, _, _), data in per_round.items():
        cases[arm].append((tuple(map(median, zip(*(d for d, _ in data)))), data[0][1]))
    result = {}
    for arm, items in cases.items():
        by_hit = {}
        for subset in ('all', 'hit', 'false', 'false_with_predecessor'):
            selected = [(d, x) for d, x in items if subset == 'all' or
                        subset == 'hit' and x['hit'] == '1' or subset == 'false' and x['hit'] == '0' or
                        subset == 'false_with_predecessor' and x['hit'] == '0' and int(x['prev_n'])]
            if not selected:
                continue
            stats = dict(cases=len(selected), median_case_saved_us={k: round(median(v) / 1000, 3)
                         for k, v in zip(keys, zip(*(d for d, _ in selected)))})
            if subset != 'false_with_predecessor':  # we did not stratify sampling on predecessor occupancy
                strata = defaultdict(list)
                for d, x in selected:
                    strata[x['hit'], x['n'], int(x['pop'])].append(d)
                stats['stratified_saved_us_per_token'] = {k: round(sum(
                    pop * median(d[i] for d in ds) for (_, _, pop), ds in strata.items()) / (1000 * tokens), 3)
                    for i, k in enumerate(keys)}
            by_hit[subset] = stats
        if rows := prep[arm]:
            by_hit['prep'] = dict(median_us=round(median(ns for ns, _ in rows) / 1000, 3), attempts=len(rows),
                                  finished_in_window=sum(ontime for _, ontime in rows))
        result[arm] = by_hit
    return dict(calibration_us=dict(zip(('cold', 'warm'), (round(median(v) / 1000, 3) for v in zip(*cal)))),
                logfile=log.splitlines(), arms=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=int, required=True, help='trace suffix token count, not benchmark trials')
    p.add_argument('csv', type=Path, nargs='+', help='a completed run with same-name .log')
    a = p.parse_args()
    if a.tokens < 1:
        p.error('tokens must be positive')
    print(json.dumps({f.stem: report(f, a.tokens) for f in a.csv}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
