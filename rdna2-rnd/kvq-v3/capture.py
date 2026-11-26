#!/usr/bin/env python3
"""Prefill the nested 197K coding history with KV/Q capture; SAVE dense Q8 snapshots at the historical cuts."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import Worker, read_ids, KVQ

RUNS = KVQ.parent
PROMPT197 = RUNS / 'dflash-long-cold-production-42/197k-coding-42/native-w1/turn-1.prompt.csv'
CUTS = (65796, 129015, 194973, 197021)

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--cuts', default=','.join(map(str, CUTS)))
ap.add_argument('--qstride', type=int, default=64)
ap.add_argument('--decode', type=int, default=0, help='greedy decode steps after the last cut (captured n=1)')
ap.add_argument('--no-capture', action='store_true')
a = ap.parse_args()
out = Path(a.out)
cap = out / 'capture'
cap.mkdir(parents=True, exist_ok=False)
ids = read_ids(PROMPT197)
cuts = [int(x) for x in a.cuts.split(',')]
env = {} if a.no_capture else {'STRATA_KVQ_CAPTURE': str(cap), 'STRATA_KVQ_QSTRIDE': a.qstride}
w = Worker(out / 'home', env=env)
rec = dict(startup_s=w.startup_s, info=w.info, cuts=[])
for c in cuts:
    t0 = time.time()
    got, meta = w.gen(ids[:c], 1)
    s = w.save(out / f'q8-{c}.strsess')
    rec['cuts'].append(dict(cut=c, gen=meta, save=s, wall_s=time.time() - t0, picked=got))
    print(json.dumps(rec['cuts'][-1]), flush=True)
if a.decode:
    got, meta = w.gen(ids[:cuts[-1]], a.decode)
    rec['decode'] = dict(ids=got, meta=meta)
rec['rc'] = w.close()
(out / 'capture.json').write_text(json.dumps(rec, indent=1))
print('done', rec['rc'])
