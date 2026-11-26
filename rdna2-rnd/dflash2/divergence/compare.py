#!/usr/bin/env python3
"""Compare aligned DVT1 rows, emit only parity and error aggregates (no raw activations)."""

import argparse
import json
import os
import struct

import numpy as np

from build import R


def load(path):
    header = struct.Struct('<qiii32s')
    rows = {}
    with path.open('rb') as f:
        if f.read(4) != b'DVT1':
            raise ValueError(f'bad trace magic: {path}')
        while raw := f.read(header.size):
            pos, visit, kind, size, stage = header.unpack(raw)
            stage = stage.rstrip(b'\0').decode('ascii')
            key = (visit, stage)
            data = f.read(size)
            row = rows.setdefault(pos, {})
            if size < 1 or len(data) != size or kind not in (0, 1, 2) or key in row:
                raise ValueError(f'bad record: {path}, {pos}, {key}')
            row[key] = (kind, data)
    return rows


def stat(visit, stage, kind, a, b):
    dtype = ('<u4', 'u1', '<i4')[kind]
    wa, wb = np.frombuffer(a, dtype=dtype), np.frombuffer(b, dtype=dtype)
    unequal = wa != wb
    out = dict(visit=visit, stage=stage, kind=('f32', 'bytes', 'i32')[kind],
               different=int(unequal.sum()), count=wa.size)
    if kind == 0:
        x, y = wa.view('<f4').astype('f8'), wb.view('<f4').astype('f8')
        if not np.all(np.isfinite(x) & np.isfinite(y)):
            raise ValueError(f'nonfinite FP32 trace: {visit}, {stage}')
        d = y - x
        dn = np.sqrt(np.sum(d * d))
        ref = np.sqrt(np.sum(x * x))
        out.update(max_abs=float(np.max(np.abs(d))), rms=float(dn / np.sqrt(wa.size)),
                   rel_l2=float(dn / ref) if ref else None)
        if stage == 'softcapped_logits':
            gap = lambda z: float(np.max(z) - np.partition(z, -2)[-2])
            out.update(top1_match=bool(x.argmax() == y.argmax()), ref_top1_margin=gap(x), test_top1_margin=gap(y))
    return out


def compare(ref_dir, test_dir):
    ref_info, test_info = [json.loads((p / 'receipt.json').read_text()) for p in (ref_dir, test_dir)]
    for key in ('prompt_sha256', 'profile_sha256', 'snapshot', 'resume'):
        if ref_info[key] != test_info[key]:
            raise ValueError(f'mismatch: {key}')
    for key in ('slots', 'layout', 'swaps'):
        if ref_info['placement'][key] != test_info['placement'][key]:
            raise ValueError(f'mismatch: placement {key}')
    a, b = [load(p / 'trace.dvt1') for p in (ref_dir, test_dir)]
    start = int(ref_info['result']['pos'])
    width = int(test_info['result']['rows'])
    if sorted(a) != list(range(start - 1, start + 8)) or sorted(b) != list(range(start - 1, start + width)):
        raise ValueError('incomplete trace rows')
    if a[start - 1] != b[start - 1]:
        raise ValueError('prefill logits mismatch, dual runs are incomparable')
    guards = []
    for p in (ref_dir, test_dir):
        guard = p / 'guard'
        stops = json.loads((guard / 'stop-reasons.json').read_text())
        warnings = json.loads((guard / 'warnings.json').read_text())
        vram = [json.loads(x)['free_mib'] for x in (guard / 'vram.jsonl').read_text().splitlines()]
        if stops:
            raise ValueError(f'guard stopped: {p}')
        guards.append(dict(name=p.name, boot_warnings=sum(w['phase'] == 'boot' for w in warnings),
                           fresh_warnings=sum(w['phase'] != 'boot' for w in warnings), min_free_mib=min(vram)))
    report = dict(ref=ref_dir.name, test=test_dir.name, first_absolute_position=start,
                  slots=int(ref_info['placement']['slots']), layout=ref_info['placement']['layout'], guards=guards, rows=[])
    for pos in sorted(b)[1:]:
        x, y = a[pos], b[pos]
        if x[(-1, 'input_ids')] != y[(-1, 'input_ids')]:
            raise ValueError(f'different input IDs at absolute position {pos}')
        if (49, 'softcapped_logits') not in x or (49, 'softcapped_logits') not in y:
            raise ValueError('missing logits')
        common = [k for k in x if k in y]
        if common != [k for k in y if k in x]:
            raise ValueError(f'out-of-order common trace stages at {pos}')
        required = {(v, label) for v in range(48) for label in ('cache_k', 'cache_v', 'attend_f32', 'ffn_norm',
                                                                'router_scores', 'route_ids', 'visit_out')}
        if not required <= set(common):
            raise ValueError(f'missing visit records at {pos}')
        differences = []
        for key in common:
            kind, bx = x[key]
            ky, by = y[key]
            if kind != ky or len(bx) != len(by):
                raise ValueError(f'different encoding or shape at {pos}, {key}')
            if bx != by:
                differences.append(stat(*key, kind, bx, by))
        first = differences[0] if differences else None
        first_fp32 = next((d for d in differences if d['kind'] == 'f32'), None)
        report['rows'].append(dict(offset=pos - start, compared=len(common),
                                   first_difference=first, first_fp32_difference=first_fp32,
                                   top1_match=(x[(49, 'softcapped_logits')] == y[(49, 'softcapped_logits')]) or
                                   next(d['top1_match'] for d in differences if d['stage'] == 'softcapped_logits'),
                                   only_ref=[k for k in x if k not in y], only_test=[k for k in y if k not in x],
                                   differences=differences))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ref')
    ap.add_argument('test')
    args = ap.parse_args()
    os.umask(0o077)
    result = compare(R / args.ref, R / args.test)
    (R / args.test / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{k: row[k] for k in ('offset', 'first_difference', 'first_fp32_difference', 'top1_match')}
                      for row in result['rows']], indent=2))
