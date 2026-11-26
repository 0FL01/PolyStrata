#!/usr/bin/env python3
"""Union of traced kernel and H2D intervals between the first and last target logits.

For a single restored-turn native production/measure.py --rocprof run. Counts recorded
activities, not hardware engine utilization. In particular, a rocCLR copy implemented
as a compute blit counts as a dispatch, not a memory-copy record.
"""
import argparse
import csv
import json
from pathlib import Path


def rows(path):
    with path.open(newline='') as f:
        yield from csv.DictReader(f)


def intervals(records, lo, hi):
    for r in records:
        a, b = max(lo, int(r['Start_Timestamp'])), min(hi, int(r['End_Timestamp']))
        if a < b:
            yield a, b


def union(spans):
    total = 0
    end = 0
    for a, b in sorted(spans):
        total += max(0, b - max(a, end))
        end = max(b, end)
    return total


def summarize(run):
    if json.loads(run.with_name(run.name + '.guard').joinpath('stop-reasons.json').read_text()):
        raise ValueError(f'{run.name}: guard stopped')
    result = json.loads((run / 'result.json').read_text())
    turn, = result['turns']
    kernels = list(rows(run / 'trace/live_kernel_trace.csv'))
    ends = sorted(int(r['End_Timestamp']) for r in kernels if 'softcap_kernel' in r['Kernel_Name'])
    if len(ends) < 2:
        raise ValueError('need the prompt and at least one decode softcap')
    # Excludes the prompt, last pick and final tap append; not the wire interval.
    lo, hi = ends[0], ends[-1]
    k = list(intervals(kernels, lo, hi))
    copies = list(rows(run / 'trace/live_memory_copy_trace.csv'))
    h = list(intervals((r for r in copies if r['Direction'] == 'MEMORY_COPY_HOST_TO_DEVICE'), lo, hi))
    c = list(intervals(copies, lo, hi))
    kernel, h2d = union(k), union(h)
    return dict(run=run.name, softcap_intervals=len(ends) - 1, kept_spec_rows=turn.get('kept_verified_rows'),
                interval_ms=(hi - lo) / 1e6, dispatch_ms=kernel / 1e6, h2d_ms=h2d / 1e6,
                h2d_over_dispatch_ms=(kernel + h2d - union(k + h)) / 1e6,
                unaccounted_ms=(hi - lo - union(k + c)) / 1e6)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, nargs='+')
    for run in p.parse_args().run:
        print(json.dumps(summarize(run)))
