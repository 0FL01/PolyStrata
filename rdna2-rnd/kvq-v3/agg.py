#!/usr/bin/env python3
"""Aggregate screen JSONs: per codec bytes, reconstruction, attention error pooled over visits."""
import json
import sys
from pathlib import Path

import numpy as np

files = sys.argv[1:]
data = {}
floor = {}
for f in files:
    d = json.loads(Path(f).read_text())
    for v, vr in d['visits'].items():
        floor[v] = vr['q8_vs_fp32']['p95']
        for name, c in vr['codecs'].items():
            data.setdefault(name, {})[v] = c
rows = []
for name, per in data.items():
    vs = sorted(per, key=int)
    b = per[vs[0]]['bytes']
    rows.append((b, name, np.mean([per[v]['rec_rel'] for v in vs]), np.mean([per[v]['norm_ratio_mean'] for v in vs]),
                 np.mean([per[v]['att_p50'] for v in vs]), np.mean([per[v]['att_p95'] for v in vs]),
                 max(per[v]['att_p95'] for v in vs), max(per[v]['att_max'] for v in vs),
                 max(vs, key=lambda v: per[v]['att_p95']), len(vs)))
rows.sort()
print(f"visits={len(floor)}  Q8-vs-FP32 att p95 by visit: " + ' '.join(f'{v}:{floor[v]:.4f}' for v in sorted(floor, key=int)))
print(f"{'codec':18s} {'B/row':>5s} {'bpv':>5s} {'rec':>7s} {'norm':>7s} {'att50':>8s} {'att95':>8s} {'w-p95':>8s} {'w-max':>7s} worst")
for b, name, rec, nr, a50, a95, w95, wmax, wv, n in rows:
    print(f"{name:18s} {b:5d} {b * 8 / 128:5.2f} {rec:7.4f} {nr:7.4f} {a50:8.5f} {a95:8.5f} {w95:8.5f} {wmax:7.4f} v{wv} ({n})")
