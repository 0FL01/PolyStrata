#!/usr/bin/env python3
"""Private lossy dense-Q8 SAVE: replace cold global V rows (< seal) at all 8 visits by Q8(codec(V)).

Source V is the captured FP32 (direct) or the SAVE's own Q8 (requant). K, the hot V tail, SWA rings, recurrent
state and tokens stay byte-exact; the payload checksum is refreshed. Optional --protect FILE lists absolute rows to
keep native (per visit: "visit row" lines, or "* row" for all visits). --codec-map visit=codec,... overrides.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / 'ai/agent-frameworks/PolyStrata'
sys.path.insert(0, str(ROOT / 'docs/goals/arch-decode-recon/kv'))
sys.path.insert(0, str(HERE))
from snapshot import parse, read_at, VISITS, STRIDES, VISIT_ROW   # noqa: E402
import kvq   # noqa: E402
import screen   # noqa: E402

CHECK = HERE / 'strsess-hash'


def q8_encode(x):
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.max(np.abs(b), axis=1, keepdims=True) / np.float32(127)
    inv = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d > 0)
    q = np.clip(np.rint(b * inv), -127, 127).astype(np.int8)
    return q.reshape(-1), d.astype('<f2').reshape(-1)


def q8_decode(c, s):
    return (c.reshape(-1, 32).astype(np.float32) * s.astype(np.float32).reshape(-1, 1)).reshape(-1, 8, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', required=True)
    ap.add_argument('--cap', required=True)
    ap.add_argument('--codec', required=True)
    ap.add_argument('--codec-map', default='')
    ap.add_argument('--source', choices=('fp32', 'q8'), default='fp32')
    ap.add_argument('--hot', type=int, default=2048)
    ap.add_argument('--protect', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--verify-q8', action='store_true', help='check numpy Q8 of captured FP32 == SAVE codes')
    a = ap.parse_args()
    out = Path(a.out)
    assert not out.exists()
    table = screen.codec_table()
    cmap = {v: a.codec for v in VISITS}
    for kv in filter(None, a.codec_map.split(',')):
        v, c = kv.split('=')
        cmap[int(v)] = c
    protect = {v: set() for v in VISITS}
    if a.protect:
        for line in Path(a.protect).read_text().split('\n'):
            if line.strip():
                v, r = line.split()
                for vv in (VISITS if v == '*' else [int(v)]):
                    protect[vv].add(int(r))
    tmp = out.with_suffix('.tmp')
    subprocess.run(['cp', '--reflink=always', a.save, str(tmp)], check=True)
    fd = os.open(tmp, os.O_RDWR)
    meta, records = parse(fd, str(CHECK))
    committed = meta['committed']
    seal = (committed - a.hot) // 128 * 128
    cap = kvq.Capture(a.cap)
    stats = {v: dict(rows=0, mismatch_codes=0, energy=0.0, delta=0.0) for v in VISITS}
    per = meta['kv_record_rows']
    for vi, v in enumerate(VISITS):
        _, Vall = cap.kv(v, seal)
        Vall = np.asarray(Vall, np.float32)
        name = cmap[v]
        fn, mean = table[name]
        for ri, (rows, off) in enumerate(records):
            start = ri * per
            cold = min(rows, seal - start)
            if cold <= 0:
                break
            code = off + rows * (vi * VISIT_ROW + sum(STRIDES[:2]))
            scale = code + rows * STRIDES[2]
            c0 = np.frombuffer(read_at(fd, cold * 1024, code), np.int8)
            s0 = np.frombuffer(read_at(fd, cold * 64, scale), '<f2')
            native = q8_decode(c0, s0)
            src = Vall[start:start + cold]
            if a.verify_q8:
                cc, ss = q8_encode(src)
                stats[v]['mismatch_codes'] += int(np.count_nonzero(cc != c0)) + int(np.count_nonzero(ss != s0))
            x = src if a.source == 'fp32' else native
            if name == 'q8_0':
                y = native
            else:
                y, _ = screen.with_mean(fn, x, mean) if mean else (fn(x.reshape(-1, 128))[0].reshape(x.shape), 0)
            y = np.asarray(y, np.float32).reshape(cold, 8, 128)
            keep = [r - start for r in protect[v] if start <= r < start + cold]
            if keep:
                y[keep] = native[keep]
            nc, ns = q8_encode(y)
            assert os.pwrite(fd, nc.tobytes(), code) == cold * 1024
            assert os.pwrite(fd, ns.tobytes(), scale) == cold * 64
            st = stats[v]
            st['rows'] += cold
            st['energy'] += float(np.sum(src.astype(np.float64) ** 2))
            st['delta'] += float(np.sum((q8_decode(nc, ns) - src).astype(np.float64) ** 2))
        print(v, name, stats[v]['rows'], 'mismatch', stats[v]['mismatch_codes'],
              'rel %.4f' % np.sqrt(stats[v]['delta'] / stats[v]['energy']), flush=True)
    fields = subprocess.check_output([str(CHECK), str(fd), str(meta['bytes'] - 80)], pass_fds=(fd,)).decode().split()
    assert fields[0] == meta['header_xxh_style']
    assert os.pwrite(fd, struct.pack('<Q', int(fields[1], 16)), meta['bytes'] - 16) == 8
    os.fsync(fd)
    meta2, _ = parse(fd, str(CHECK))
    os.close(fd)
    tmp.rename(out)
    Path(str(out) + '.json').write_text(json.dumps(dict(args=vars(a), seal=seal, committed=committed, stats=stats,
                                                        codec_map=cmap), indent=1))
    print('wrote', out, 'seal', seal)


if __name__ == '__main__':
    main()
