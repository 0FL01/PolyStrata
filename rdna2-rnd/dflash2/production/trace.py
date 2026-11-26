#!/usr/bin/env python3
"""Group native DFlash2 proposal kernels from a separate live rocprofv3 trace."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median


def summarize(path):
    with path.open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r['Dispatch_Id']))
    rounds, costs, norms = [], None, 0
    for row in rows:
        name = row['Kernel_Name'].replace('(anonymous namespace)::', '').split('(')[0]
        if '::copy_v_ring' in name:
            costs, norms = defaultdict(int), 0
            continue
        if costs is None:
            continue
        duration = int(row['End_Timestamp']) - int(row['Start_Timestamp'])
        if 'dflash2::rms_norm' in name:
            norms += 1
            group = 'norm'
        elif not norms:
            continue
        elif '::attend_exact_scores' in name:
            group = 'scores'
        elif '::attend_exact_values' in name:
            group = 'softmax_V'
        elif 'dflash2::attend' in name:
            group = 'short_attention'
        elif '::topk_' in name:
            group = 'top16'
        elif 'q6_k_warps_cols_kernel' in name and norms == 12:
            group = 'head_Q6_K'
        elif any(s in name for s in ('mmvq', 'warps_cols_kernel', 'mmvf_multi_kernel', 'wave8<')):
            group = 'selector_projection' if norms == 12 else 'FFN_projection' if norms % 2 else 'attention_projection'
        elif any(s in name for s in ('::conv', 'norm_rope', 'swiglu', 'quantize')):
            group = 'other_draft_kernels'
        else:
            continue  # target W1/window work can occur before the next draft
        costs[group] += duration
        if '::topk_merge' in name:
            if norms != 12:
                raise ValueError(f'expected 12 DFlash2 norms per proposal, got {norms}')
            costs['total_draft_kernels'] = sum(costs.values())
            rounds.append(costs)
            costs = None
    steady = rounds[1:]
    if not steady:
        raise ValueError('need two or more proposals')
    return dict(proposals=len(rounds), median_ms={k: median(r[k] for r in steady) / 1e6
                                                 for k in sorted({k for r in steady for k in r})})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kernel_csv', type=Path)
    print(json.dumps(summarize(p.parse_args().kernel_csv), indent=2))
