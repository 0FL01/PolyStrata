#!/usr/bin/env python3
"""Teacher-forced logit comparison of KV fixtures on recorded three-turn agent trajectories.

Each arm: fresh worker (fixed cap, zero swaps, stable prefill), RESTORE its SAVE, then for each recorded turn GEN the
recorded prompt with the recorded output forced (kvq_follow) and every step's top-K logits + lse logged.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import Worker, read_ids, csv, KVQ, GREEDY

RUNS = KVQ.parent
TRAJ = {
    '64k-coding': RUNS / 'dflash-long-warmed-production-42/64k-coding-42/native-w1',
    '64k-reasoning': RUNS / 'dflash-long-warmed-production-42/64k-reasoning-42/native-w1',
    '128k-coding': RUNS / 'venture-v2/lookup-epoch/qualify-128k-coding-79/w1',
    '128k-reasoning': RUNS / 'venture-v2/lookup-epoch/qualify-128k-reasoning-79/w1',
    '197k-coding': RUNS / 'dflash-long-cold-production-42/197k-coding-42/native-w1',
    '197k-reasoning': RUNS / 'dflash-long-cold-production-42/197k-reasoning-42/native-w1',
}


def read_log(path):
    raw = Path(path).read_bytes()
    k = int(np.frombuffer(raw[:4], '<i4')[0])
    rec = 8 + 8 + 4 + 4 + 8 * k
    n = (len(raw) - 4) // rec
    out = []
    for i in range(n):
        b = raw[4 + i * rec: 4 + (i + 1) * rec]
        step, forced = np.frombuffer(b[:16], '<i8')
        lse, fl = np.frombuffer(b[16:24], '<f4')
        pairs = np.frombuffer(b[24:], dtype=[('id', '<i4'), ('lg', '<f4')])
        out.append(dict(step=int(step), forced=int(forced), lse=float(lse), flogit=float(fl), ids=pairs['id'].copy(),
                        lg=pairs['lg'].astype(np.float64)))
    return out


def compare(ctrl, arm):
    """Per-position KL(ctrl||arm) over ctrl's top-K, NLL delta of the forced token, top-1 agreement."""
    kl, dnll, top1, tvs = [], [], [], []
    for a, b in zip(ctrl, arm):
        assert a['forced'] == b['forced']
        lp = a['lg'] - a['lse']
        p = np.exp(lp)
        bmap = dict(zip(b['ids'].tolist(), b['lg'].tolist()))
        floor = b['lg'].min()
        lq = np.array([bmap.get(i, floor) for i in a['ids'].tolist()]) - b['lse']
        kl.append(float(np.sum(p * (lp - lq))))
        tvs.append(float(0.5 * np.sum(np.abs(p - np.exp(lq)))))
        dnll.append((b['lse'] - b['flogit']) - (a['lse'] - a['flogit']))
        top1.append(int(a['ids'][0] == b['ids'][0]))
    kl = np.array(kl)
    return dict(n=len(kl), kl_mean=float(kl.mean()), kl_p50=float(np.median(kl)), kl_p95=float(np.percentile(kl, 95)),
                kl_max=float(kl.max()), tv_mean=float(np.mean(tvs)), dnll_mean=float(np.mean(dnll)),
                top1=float(np.mean(top1)), top1_flips=int(len(top1) - sum(top1)))


def run_arm(where, save, traj, cap, top):
    where.mkdir(parents=True, exist_ok=True)
    w = Worker(where / 'home', cap_mib=cap, swaps=0)
    t0 = time.time()
    r = w.restore(save)
    info = dict(restore=r, startup_s=w.startup_s, slots=[l for l in w.info if l.startswith('INFO')], turns=[])
    for k in (1, 2, 3):
        p = read_ids(traj / f'turn-{k}.prompt.csv')
        o = read_ids(traj / f'turn-{k}.output.csv')
        f = where / f'turn-{k}.follow.csv'
        f.write_text(csv(o))
        lg = where / f'turn-{k}.logits.bin'
        got, meta = w.gen(p, len(o), GREEDY, extra=f'kvq_follow={f} kvq_logits={lg} kvq_top={top}')
        assert got == o[:len(got)], 'forced output mismatch'
        info['turns'].append(dict(prompt=len(p), forced=len(o), meta=meta))
    info['rc'] = w.close()
    info['wall_s'] = time.time() - t0
    (where / 'arm.json').write_text(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--traj', required=True, choices=list(TRAJ))
    ap.add_argument('--arm', action='append', required=True, help='name=SAVE path; first is the control')
    ap.add_argument('--cap', type=int, default=5200)
    ap.add_argument('--top', type=int, default=256)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [x.split('=', 1) for x in a.arm]
    for name, save in arms:
        if (out / name / 'arm.json').exists():
            continue
        info = run_arm(out / name, save, TRAJ[a.traj], a.cap, a.top)
        print(name, 'restore', info['restore'], 'wall %.1f' % info['wall_s'], flush=True)
    ctrl = sum((read_log(out / arms[0][0] / f'turn-{k}.logits.bin') for k in (1, 2, 3)), [])
    summary = {}
    for name, _ in arms[1:]:
        arm = sum((read_log(out / name / f'turn-{k}.logits.bin') for k in (1, 2, 3)), [])
        summary[name] = compare(ctrl, arm)
        print(name, json.dumps(summary[name]), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
