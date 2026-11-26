#!/usr/bin/env python3
"""Summarize exact expert-reuse replay receipts without loading any activation or model bytes."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median, quantiles


ARMS = ('serial', 'grouped', 'multi')


def geometry(rows):
    by_multiplicity = [sum(a[f'n{k}'] for a in rows) for k in range(1, 9)]
    uses = sum(a['uses'] for a in rows)
    assert uses == sum(k * n for k, n in enumerate(by_multiplicity, 1))
    unique = sum(by_multiplicity)
    naive = sum(a['uses'] * (4644864 if a['down'] == 3 else 4478976) for a in rows)
    shared = sum(a['unique_bytes'] for a in rows)
    potential_decodes_removed = sum((k - (k + 3) // 4) * n for k, n in enumerate(by_multiplicity, 1))
    return dict(cases=len(rows), uses=uses, unique=unique, repeat_frac=(uses - unique) / uses if uses else 0,
                multiplicity=by_multiplicity, nominal_bytes=naive, distinct_bytes=shared,
                ideal_avoided_payload=(naive - shared), weight_decodes_removed=potential_decodes_removed)


def score(cases):
    assert cases
    pairs = []
    controls = dict(major=0, minor=0, invol=0, wall_gt_thread_1p10=0, trials=0)
    for rows in cases.values():
        by = {name: [r for r in rows if r['arm'] == name] for name in ARMS}
        rounds = max(r['round'] for r in rows) + 1
        assert len(rows) == rounds * 6 and all(len(x) == rounds * 2 for x in by.values())
        assert len({(r['round'], r['part']) for r in rows}) == len(rows)
        fold = lambda field: {name: median(r[field] / 1000 for r in rows) for name, rows in by.items()}
        pairs.append((fold('wall_ns'), fold('thread_ns')))
        for k in ('major', 'minor', 'invol'):
            controls[k] += sum(r[k] for r in rows)
        controls['wall_gt_thread_1p10'] += sum(r['wall_ns'] > 1.1 * r['thread_ns'] for r in rows)
        controls['trials'] += len(rows)

    def compare(axis):
        values = [a[axis] for a in pairs]
        sums = {name: sum(a[name] for a in values) for name in ARMS}
        delta = [a['grouped'] - a['multi'] for a in values]
        return dict(sum_of_case_medians_us=sums,
                    case_p50_us={name: median(a[name] for a in values) for name in ARMS},
                    case_p90_us={name: quantiles([a[name] for a in values], n=10, method='inclusive')[-1]
                                 if len(values) > 1 else values[0][name] for name in ARMS},
                    serial_over_grouped=sums['serial'] / sums['grouped'],
                    grouped_over_multi=sums['grouped'] / sums['multi'],
                    serial_over_multi=sums['serial'] / sums['multi'],
                    median_paired_grouped_minus_multi_us=median(delta), positive_cases=sum(x > 0 for x in delta))

    return dict(cases=len(cases), controls=controls, wall=compare(0), process_cpu=compare(1))


def summarize(path):
    by_width = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = {k: (v if k == 'arm' else int(v)) for k, v in row.items()}
            by_width[a['width']].append(a)
    out = {}
    for w, rows in sorted(by_width.items()):
        if 'arm' not in rows[0]:
            out[w] = {'all': geometry(rows), 'visit0': geometry([a for a in rows if a['visit'] == 0])}
            continue
        cases = defaultdict(list)
        for a in rows:
            cases[a['pos'], a['visit']].append(a)
        # The deliberately extra first visit at the first row is kept out of population screen if present.
        first_pos = min([p for p, v in cases if v == 0], default=None)
        first = (first_pos, 0)
        sample = dict(cases)
        if sum(v == 0 for p, v in cases) == 2:
            del sample[first]
        out[w] = {'screen': score(sample), 'first_visit': score({first: cases[first]}) if first in cases else None}
        fault_free = {k: v for k, v in sample.items() if not any(a['major'] for a in v)}
        if fault_free and len(fault_free) < len(sample):
            out[w]['major_fault_free'] = score(fault_free)
        for name, pred in [('has_repeat', lambda a: a['uses'] > sum(a[f'n{k}'] for k in range(1, 9))),
                           ('no_repeat', lambda a: a['uses'] == sum(a[f'n{k}'] for k in range(1, 9)))]:
            group = {k: v for k, v in sample.items() if pred(v[0])}
            if group:
                out[w][name] = score(group)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('paths', type=Path, nargs='+')
    args = ap.parse_args()
    print(json.dumps({p.stem: summarize(p) for p in args.paths}, indent=2))


if __name__ == '__main__':
    main()
