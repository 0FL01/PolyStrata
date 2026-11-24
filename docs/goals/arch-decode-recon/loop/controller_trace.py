#!/usr/bin/env python3
"""Attribute the serial Inkling controller chain in rocprofv3 kernel CSVs (CPU only)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
from collections import Counter, defaultdict


CHAIN = {
    'qkv': ('mix3_kernel', 'control_in3_kernel', 'gemv_rows_segments_kernel', 'control_out3_kernel'),
    'ffn': ('mix_kernel', 'control_in_kernel', 'gemv_rows_kernel', 'control_out_kernel'),
}
HEAD = 'native_q6_k_warps_kernel'
NAME = re.compile(r'::(\w+)(?:<[^>]*>)?\(')


def read_csv(path):
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['Kind'] != 'KERNEL_DISPATCH':
                continue
            m = NAME.search(r['Kernel_Name'])
            name = m[1] if m else r['Kernel_Name']
            start, end = int(r['Start_Timestamp']), int(r['End_Timestamp'])
            wg = tuple(int(r[f'Workgroup_Size_{d}']) for d in 'XYZ')
            grid = tuple(int(r[f'Grid_Size_{d}']) for d in 'XYZ')
            res = tuple(int(r[k]) for k in ('VGPR_Count', 'LDS_Block_Size', 'Scratch_Size', 'SGPR_Count'))
            yield (r['Agent_Id'], r['Queue_Id'], r['Stream_Id']), (name, start, end, wg, grid, res)


def percentile(samples, p):
    xs = sorted(samples)
    i = (len(xs) - 1) * p
    return xs[int(i)] + (xs[min(int(i) + 1, len(xs) - 1)] - xs[int(i)]) * (i % 1)


def describe(samples, divisor=1):
    if not samples:
        raise ValueError('no samples')
    return {k: round(v / divisor, 6) for k, v in (
        ('min', min(samples)), ('p10', percentile(samples, .10)),
        ('median', statistics.median(samples)), ('p90', percentile(samples, .90)),
        ('mean', statistics.mean(samples)), ('max', max(samples)),
    )}


def window(rows, prev_lm_end):
    events = [i for i, r in enumerate(rows) if r[0] in (CHAIN['qkv'][1], CHAIN['ffn'][1])]
    expected = [name for _ in range(24) for name in (CHAIN['qkv'][1], CHAIN['ffn'][1])]
    if [rows[i][0] for i in events] != expected:
        return 'not 24 alternating serial QKV/FFN controllers'
    if sum(r[0] == 'heads_kv_kernel' for r in rows) != 48:
        return 'not 48 serial Q8 KV visits'
    ns = Counter()
    configs = []
    for i, side in zip(events, ('qkv', 'ffn') * 24):
        if tuple(r[0] for r in rows[i - 1:i + 3]) != CHAIN[side]:
            return f'{side}: not a baseline mix/input/rank/output chain'
        expected_grid = (128, 3 if side == 'qkv' else 1, 1)
        if rows[i][3:5] != ((128, 1, 1), expected_grid):
            return f'{side}: input is not a one-token dispatch'
        expected_rank_grid = (16384, 1, 3 if side == 'qkv' else 1)
        if rows[i + 1][3:5] != ((256, 1, 1), expected_rank_grid):
            return f'{side}: rank is not the one-block-per-output RDNA2 path'
        for j, stage in enumerate(('depth_mix', 'input', 'rank', 'output'), i - 1):
            name, start, end, wg, grid, res = rows[j]
            ns[f'{side}_{stage}'] += end - start
            configs.append((side, stage, name, wg, grid, res, end - start))
        ns['input_pre_gap'] += max(0, rows[i][1] - rows[i - 1][2])
        ns['input_post_gap'] += max(0, rows[i + 1][1] - rows[i][2])
    ns['input'] = ns['qkv_input'] + ns['ffn_input']
    ns['controller'] = ns['input'] + sum(ns[f'{side}_{k}'] for side in CHAIN for k in ('rank', 'output'))
    ns['input_local_span'] = ns['input'] + ns['input_pre_gap'] + ns['input_post_gap']
    ns['kernel_busy'] = sum(end - start for _, start, end, *_ in rows)
    ns['traced_head_to_head'] = rows[-1][2] - prev_lm_end
    return ns, configs


def analyze(label, path):
    queues = defaultdict(list)
    for queue, row in read_csv(path):
        queues[queue].append(row)
    used = [q for q, rows in queues.items() if any(r[0] == HEAD for r in rows)]
    if len(used) != 1:
        raise ValueError(f'{path}: expected one serial LM-head stream; found {used}')
    queue = used[0]
    rows = sorted(queues[queue], key=lambda r: r[1])
    heads = [i for i, r in enumerate(rows) if r[0] == HEAD]
    totals, calls, configs, reject = defaultdict(list), defaultdict(list), defaultdict(Counter), Counter()
    for a, b in zip(heads, heads[1:]):
        result = window(rows[a + 1:b + 1], rows[a][2])
        if isinstance(result, str):
            reject[result] += 1
            continue
        ns, stages = result
        for key, value in ns.items():
            totals[key].append(value)
        for side, stage, name, wg, grid, res, duration in stages:
            key = f'{side}_{stage}'
            calls[key].append(duration)
            configs[key][name, wg, grid, res] += 1
    if not totals:
        raise ValueError(f'{path}: no complete serial token windows, rejects={reject}')
    with path.open('rb') as f:
        sha256 = hashlib.file_digest(f, 'sha256').hexdigest()
    return {
        'label': label,
        'path': str(path),
        'sha256': sha256,
        'head_stream': queue,
        'head_dispatches': len(heads),
        'complete_windows': len(totals['input']),
        'rejected_windows': dict(reject),
        'per_token_ms': {key: describe(xs, 1e6) for key, xs in sorted(totals.items())},
        'per_call_us': {key: describe(xs, 1e3) for key, xs in sorted(calls.items())},
        'launch_shapes': {
            key: [dict(kernel=name, workgroup=wg, grid_global=grid, vgpr=vgpr, lds_bytes=lds,
                       scratch_bytes=scratch, sgpr=sgpr, calls=count)
                  for (name, wg, grid, (vgpr, lds, scratch, sgpr)), count in sorted(c.items())]
            for key, c in sorted(configs.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', action='append', required=True, metavar='LABEL=CSV')
    args = parser.parse_args()
    result = []
    for spec in args.trace:
        label, separator, path = spec.partition('=')
        if not label or not separator or not path:
            parser.error('--trace needs LABEL=CSV')
        result.append(analyze(label, Path(path).expanduser()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
