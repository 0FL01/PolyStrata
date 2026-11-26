#!/usr/bin/env python3
"""Join previously captured real tensors and ordered routes; never run the engine."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import struct


WIDTH = 2304
VISITS = 48
FRAME = struct.Struct('<q6i8i8I')
STAGES = {'ffn_norm': (0, 9216), 'route_ids': (2, 32), 'route_weights': (0, 32),
          'route_on_gpu': (2, 32), 'moe_cpu_sum': (0, 9216)}


def physical(v):
    return v if v < 6 else 6 + (v - 6) % 12 if v < 42 else v - 24


def expert_bytes(visit):
    return 4644864 if physical(visit) < 3 else 4478976


def write_frame(dst, p, v, ids, bits, on_gpu, x, sums):
    assert len(set(ids)) == len(bits) == len(ids) == len(on_gpu) == 8
    assert set(on_gpu) <= {0, 1} and min(ids) >= 0 and max(ids) < 128
    n = on_gpu.count(0)
    order = [i for i, gpu in enumerate(on_gpu) if not gpu]
    l = physical(v)
    dst.write(FRAME.pack(p, v, l, n, 8 - n, 3 if l < 3 else 2, 0,
                         *([ids[i] for i in order] + [0] * (8 - n)),
                         *([bits[i] for i in order] + [0] * (8 - n))))
    dst.write(x)
    dst.write(sums if n else bytes(4 * WIDTH))


def dedup(rows, first, count):
    result = {}
    for w in (2, 4, 6, 8):
        caps = {k: Counter() for k in ('live_cpu', 'live_gpu', 'fixed_cpu', 'fixed_gpu')}
        first_visit = {k: Counter() for k in caps}
        for p in range(first, first + count - w + 1, w):
            for v in range(VISITS):
                buffers = {k: [] for k in caps}
                fixed = set(rows[p, v]['resident_ids'])
                for r in range(w):
                    label = rows[p + r, v]
                    live = set(label['resident_ids'])
                    for i, cpu in zip(label['ids'], label['cpu']):
                        assert bool(cpu) == (i not in live)
                        buffers['live_cpu' if cpu else 'live_gpu'].append(i)
                        buffers['fixed_gpu' if i in fixed else 'fixed_cpu'].append(i)
                for k, ids in buffers.items():
                    repeat = len(ids) - len(set(ids))
                    cap = dict(uses=len(ids), repeats=repeat, payload=len(ids) * expert_bytes(v),
                               avoided_payload=repeat * expert_bytes(v))
                    caps[k].update(cap)
                    if v == 0:
                        first_visit[k].update(cap)
        result[w] = {'all_visits': caps, 'visit0': first_visit, 'rows': count // w * w}
    return result


def convert_archived(tape, labels, output, stats):
    with labels.open() as f:
        meta = json.loads(next(f))
        labels = [json.loads(l) for l in f]
    first, end = meta['decode_slice']
    assert len(labels) == (end - first) * VISITS
    rows = {}
    with tape.open('rb') as t, output.open('xb') as dest:
        dest.write(struct.pack('<8s4I', b'PSCPUNS\n', len(labels), WIDTH, first, end - first))
        for i, label in enumerate(labels):
            step, p, v, _ = struct.unpack('<4Q', t.read(32))
            x = t.read(4 * WIDTH)
            assert len(x) == 4 * WIDTH and len(t.read(4 * 128)) == 4 * 128
            ids = struct.unpack('<8i', t.read(32))
            assert (step, p, v) == (i // VISITS + 1, first + i // VISITS, i % VISITS)
            assert (p, v, physical(v), list(ids)) == (label['pos'], label['visit'], label['layer'], label['ids'])
            rows[p, v] = label
            write_frame(dest, p, v, ids, label['weight_bits'], tuple(not a for a in label['cpu']), x, bytes(4 * WIDTH))
        assert not t.read(1)
    with stats.open('x') as f:
        json.dump(dedup(rows, first, end - first), f, indent=2)
        f.write('\n')


def read_dvt(path):
    grouped = {}
    with path.open('rb') as f:
        assert f.read(4) == b'DVT1'
        while hdr := f.read(52):
            p, v, kind, size, name = struct.unpack('<qiii32s', hdr)
            assert 0 < size < 8_000_000
            x = f.read(size)
            assert len(x) == size
            s = name.split(b'\0', 1)[0].decode('ascii')
            if s not in STAGES:
                continue
            assert (kind, size) == STAGES[s] and 0 <= v < VISITS
            entry = grouped.setdefault((p, v), {})
            assert s not in entry
            entry[s] = x
    return grouped


def convert_dvt(tape, output, baseline=None, restore_visit0_weights=False):
    rows = read_dvt(tape)
    first = min(p for p, _ in rows)
    count = max(p for p, _ in rows) - first + 1
    assert count in (2, 4, 6, 8) and len(rows) == count * VISITS
    ref = read_dvt(baseline) if baseline else None
    if ref:
        first_diffs = {}
        for s, (_, size) in STAGES.items():
            found = None
            for v in range(VISITS):
                a, b = rows[first, v][s], ref[first, v][s]
                if a == b:
                    continue
                unequal = sum(a[i:i + 4] != b[i:i + 4] for i in range(0, size, 4))
                found = {'pos': first, 'visit': v, 'unequal_words': unequal, 'words': size // 4}
                if STAGES[s][0] == 0:
                    found['max_abs'] = max(abs(x - y) for (x,), (y,) in zip(struct.iter_unpack('<f', a),
                                                                           struct.iter_unpack('<f', b)))
                break
            first_diffs[s] = found
        print(json.dumps({'width': count, 'pos': first, 'first_row_first_diff_by_stage': first_diffs,
                          'restored_first_visit0_weights': restore_visit0_weights}, indent=2))
    if restore_visit0_weights:
        assert ref
        old, wide = ref[first, 0], rows[first, 0]
        for s in ('ffn_norm', 'route_ids', 'route_on_gpu'):
            assert old[s] == wide[s], s
        for s in ('route_weights', 'moe_cpu_sum'):
            assert old[s] != wide[s], s
            # The branch output is only a comparator in the replay, never an operand.
            wide[s] = old[s]
    with output.open('xb') as dest:
        dest.write(struct.pack('<8s4I', b'PSCPU01\n', len(rows), WIDTH, first, count))
        # PSCPU01 encodes empty CPU branches by omitted records.
        n = 0
        for (p, v), entry in sorted(rows.items()):
            assert entry.keys() == STAGES.keys()
            placed = struct.unpack('<8i', entry['route_on_gpu'])
            if all(placed):
                assert entry['moe_cpu_sum'] == bytes(4 * WIDTH)
                continue
            write_frame(dest, p, v, struct.unpack('<8i', entry['route_ids']),
                        struct.unpack('<8I', entry['route_weights']), placed, entry['ffn_norm'], entry['moe_cpu_sum'])
            n += 1
        dest.seek(8)
        dest.write(struct.pack('<I', n))


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode', required=True)
    arch = sub.add_parser('archived-2k')
    for name in ('tape', 'labels', 'output', 'stats'):
        arch.add_argument(name, type=Path)
    dvt = sub.add_parser('dvt')
    dvt.add_argument('tape', type=Path)
    dvt.add_argument('output', type=Path)
    dvt.add_argument('--baseline', type=Path)
    dvt.add_argument('--restore-first-visit0-weights', action='store_true')
    args = ap.parse_args()
    if args.mode == 'dvt':
        convert_dvt(args.tape, args.output, args.baseline, args.restore_first_visit0_weights)
    else:
        convert_archived(args.tape, args.labels, args.output, args.stats)


if __name__ == '__main__':
    main()
