"""Join CPU hint outputs to archived PSW visits; export cap-one CPU-helper cases.

Mask at issue comes only from the previous pass, never the target visit. Use
the target rows to check the join and score the pick, not to select it.
"""
import csv
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics as st
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'arch-decode-recon/trace'))
from collect import HEADER, PREFIX, read_visit


def visits(path):
    with path.open('rb') as f:
        magic, lo, hi, epoch, *shape = HEADER.unpack(f.read(HEADER.size))
        assert magic == b'PSW642\r\n' and shape == [24, 48, 128, 8]
        result = {}
        while prefix := f.read(PREFIX.size):
            kind, size, reserved = PREFIX.unpack(prefix)
            assert not reserved and size > PREFIX.size
            raw = f.read(size - PREFIX.size)
            assert len(raw) == size - PREFIX.size
            if kind == 1:
                v = read_visit(raw)
                result[v['pos'], v['visit']] = v
        assert len(result) == (hi - lo) * 48
        return result


def cpu(v):
    return [(e, w) for e, w, b in zip(v['ids'], v['weight_bits'], v['placement']) if b == 'cpu']


def capture(path):
    raw = path.read_bytes()
    magic, n, width, pos, tok = struct.unpack_from('<8s4I', raw)
    assert magic == b'PSCPU01\n' and width == 2304 and len(raw) == 24 + n*18528
    rows = {}
    for i in range(n):
        at = 24 + i*18528
        p, v, l, count, g, _, _, *ew = struct.unpack_from('<q6i8i8I', raw, at)
        assert count > 0 and count + g == 8 and (p,v) not in rows
        rows[p,v] = list(zip(ew[:count], ew[8:8+count]))
    return pos, tok, rows


def main():
    csv_path, cap_path, trace_path, out_path = map(Path, sys.argv[1:])
    start, tokens, cap = capture(cap_path)
    trace = visits(trace_path)
    # Captures from different processes: require all assigned CPU IDs in
    # order and their weight bits, including the empty branches, to agree.
    for p in range(start, start+tokens):
        for v in range(48):
            assert cap.get((p,v), []) == cpu(trace[p,v])
    with csv_path.open() as f:
        pred = {(int(r['pos']),int(r['visit'])): [int(r['h'+str(i)]) for i in range(4)]
                for r in csv.DictReader(f) if r['arm']=='1' and r['round']=='0'}
    groups = defaultdict(list)
    misses = 0
    stale = 0
    for p in range(start, start+tokens):
        for v in range(18,42):
            prev, old, target = (trace[p,j] for j in (v-1,v-12,v))
            actual = cpu(target)
            misses += len(actual)
            mask = lambda s: int(s['resident_mask'],16) | int(s['asked_mask'],16) | int(s['sent_mask'],16)
            pick = next((i for i in pred.get((p,v-1),[]) if not mask(old) >> i & 1),None)
            if pick is None:
                continue
            stale += bool(mask(target) >> pick & 1)
            # Spend 8 us inside the wait on the hint (model, not live). Both
            # arms replay the same remaining slack. A zero-slack case must not
            # silently disappear; not seen on these two traces.
            near = target['router_wait_ns'] - 8000
            assert near > 0
            hit = pick in dict(actual)
            groups[hit, len(actual)].append((p,v,pick,near,round(1e6*target['gpu_join_ms']),cpu(prev),actual))
    rng = random.Random(4242)
    sample = []
    for _, rows in sorted(groups.items()):
        k = min(3,len(rows))
        sample.extend((r,len(rows),k) for r in rng.sample(rows,k))
    rng.shuffle(sample)
    with out_path.open('x') as f:
        print('PREFETCH1',tokens,len(sample),file=f)
        for (p,v,e,near,join,prev,actual),pop,k in sample:
            print(p,v,e,near,join,pop,k,len(prev),len(actual),*(i for pair in prev+actual for i in pair),file=f)
    issued = sum(map(len,groups.values()))
    hits = sum(len(rows) for (h,_),rows in groups.items() if h)
    print(json.dumps(dict(tokens=tokens, captured=len(cap), targets=tokens*24, misses=misses,
                          issued=issued, hits=hits, stale=stale, cases=len(sample),
                          precision=hits/issued, recall=hits/misses, false_MB_per_token=(issued-hits)*4478976/tokens/1e6,
                          slack_us=st.median(r[3]/1000 for rows in groups.values() for r in rows)), indent=2))


if __name__ == '__main__':
    main()
