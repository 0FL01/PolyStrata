#!/usr/bin/env python3
"""Separate append and seven-row proposal stages in the bench.sh rocprofv3 traces."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def end(row):
    return int(row['End_Timestamp'])


def start(row):
    return int(row['Start_Timestamp'])


def duration(row):
    return end(row) - start(row)


def summarize(path):
    with path.open() as f:
        rows = sorted(csv.DictReader(f), key=lambda row: int(row['Dispatch_Id']))
    names = [r['Kernel_Name'].replace('(anonymous namespace)::', '').split('(')[0] for r in rows]
    proposal_ends = [i for i, n in enumerate(names) if '::topk_merge' in n]
    append_starts = [i for i, n in enumerate(names) if '::gather_taps' in n]
    append_layers = [i for i, n in enumerate(names) if '::copy_v_ring' in n]
    warmup = [i for i in append_starts if i < proposal_ends[0]]
    proposal_starts = [max(i for i in append_layers if i < j) for j in proposal_ends]
    rounds = []
    for a, b in zip(proposal_starts, proposal_ends, strict=True):
        result = defaultdict(int)
        norms = 0
        for r, n in zip(rows[a + 1:b + 1], names[a + 1:b + 1], strict=True):
            if '::rms_norm' in n:
                norms += 1
                group = 'norm'
            elif '::attend_exact_scores' in n:
                group = 'attention_scores'
            elif '::attend_exact_values' in n:
                group = 'attention_softmax_values'
            elif '::attend' in n:
                group = 'attention_stock'
            elif 'topk_' in n:
                group = 'top16'
            elif 'q6_k_warps_cols_kernel<7>' in n:
                group = 'target_head7'
            elif any(s in n for s in ('mmvq', 'warps_cols_kernel', 'mmvf_multi_kernel')):
                if norms == 12:
                    group = 'selector_hidden'
                elif norms % 2 == 1:
                    grid = int(r['Grid_Size_X']) // int(r['Workgroup_Size_X'])
                    group = 'ffn_gate_up' if grid in (3456, 1728) else 'ffn_other'
                else:
                    group = 'attn_projections'
            elif '::conv' in n:
                group = 'dynamic_conv'
            elif 'norm_rope' in n:
                group = 'norm_rope'
            elif 'swiglu' in n:
                group = 'swiglu'
            elif 'quantize' in n:
                group = 'activation_quantize'
            else:
                group = 'other_kernel'
            result[group] += duration(r)
        result['propose_kernels'] = sum(result.values())
        result['attention_total'] = sum(v for k, v in result.items() if k.startswith('attention_'))
        rounds.append(result)
    appends = []
    for a in append_starts[len(warmup):]:
        b = [i for i in append_layers if i > a][4]  # Five layers per one-row append.
        appends.append({'append_timeline': end(rows[b]) - start(rows[a]),
                        'append_kernels': sum(map(duration, rows[a:b + 1]))})
    out = {'rounds': len(rounds)}
    for data in (rounds, appends):
        steady = data[1:]
        out.update({key + '_ms': median(r.get(key, 0) for r in steady) / 1e6
                    for key in sorted(set().union(*(r.keys() for r in steady)))})
    out['warmup_device_timeline_ms'] = (end(rows[proposal_starts[0]]) - start(rows[warmup[0]])) / 1e6
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    print(json.dumps({p.parent.name: summarize(p) for p in sorted(args.receipt.glob('*/trace_kernel_trace.csv'))},
                     indent=2))
