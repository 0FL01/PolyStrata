#!/usr/bin/env python3
"""Aggregate free-running results: pass rate, failure classes, output/reasoning IDs, wall time per arm (and task)."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
prefix = sys.argv[2] if len(sys.argv) > 2 else ''
rows = []
for f in sorted(root.glob(f'{prefix}*/*/result.json')):
    r = json.loads(f.read_text())
    arm = f.parent.name.replace('.retry', '')
    g = r.get('grade') or {}
    if g.get('class') == 'infrastructure':
        continue
    rows.append(dict(arm=arm, task=r['task'], seed=r['seed'], size=r['size'], ok=bool(g.get('pass')),
                     cls=g.get('class') or 'pass', out=r.get('out_ids') or 0, rea=r.get('reasoning_ids') or 0,
                     wall=r.get('task_wall_s') or 0, calls=r.get('tool_calls') or 0, bad=r.get('bad_calls') or 0,
                     turns=len(r.get('turns', []))))
arms = sorted({r['arm'] for r in rows}, key=lambda a: (a != 'ctrl', a != 'jit3', a))
for scope in ['all'] + sorted({r['task'] for r in rows}):
    print(f'== {scope}')
    print(f"{'arm':9s} {'n':>3s} {'pass':>6s} {'reason_ids':>11s} {'out_ids':>8s} {'wall_s':>7s} {'calls':>5s} {'bad':>4s} classes")
    for a in arms:
        R = [r for r in rows if r['arm'] == a and (scope == 'all' or r['task'] == scope)]
        if not R:
            continue
        c = collections.Counter(r['cls'] for r in R if not r['ok'])
        ok = [r for r in R if r['ok']]
        print(f"{a:9s} {len(R):3d} {np.mean([r['ok'] for r in R]):6.2f} {np.mean([r['rea'] for r in R]):11.1f} "
              f"{np.mean([r['out'] for r in R]):8.1f} {np.mean([r['wall'] for r in ok]) if ok else 0:7.2f} "
              f"{np.mean([r['calls'] for r in R]):5.2f} {sum(r['bad'] for r in R):4d} {dict(c)}")
