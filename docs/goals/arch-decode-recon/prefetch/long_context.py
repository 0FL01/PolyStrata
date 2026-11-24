#!/usr/bin/env python3
"""Replay the fixed CPU-prefetch policy on a CP-02 binary trace; no benchmark runs."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median, quantiles
import sys

from report import report
from trace_screen import mask, screen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trace'))
from collect import header, records


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def calibrate(csv_path, arm):
    stats = report(csv_path, 89)['arms'][arm]
    groups, finish = defaultdict(lambda: defaultdict(list)), []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r['pair'] != arm:
                continue
            if r['hit'] == '1':
                groups[r['pos'], r['visit'], r['round']][r['arm']].append(int(r['compute_ns']))
            if arm == 'helper_read' and r['arm'] == arm and r['prep_lines'] == r['expected_lines']:
                # Include the helper wake-up, not just its time executing the read loop.
                finish.append((int(r['prep_end_ns']) - int(r['deadline_ns']) + int(r['window_ns'])) / 1000)
    visits = defaultdict(list)
    for (pos, visit, _), times in groups.items():
        visits[pos, visit].append((mean(times['cold']) - mean(times[arm])) / 1000)
    benefits = [median(ds) for ds in visits.values()]
    result = dict(csv_sha256=digest(csv_path), hit_cases=len(benefits),
                  hit_compute_median_us=median(benefits), hit_compute_p90_us=quantiles(benefits, n=10, method='inclusive')[-1],
                  hit_compute_max_us=max(benefits))
    if finish:
        result.update(complete_reads=len(finish), prep_post_to_finish_median_us=median(finish),
                      prep_post_to_finish_p90_us=quantiles(finish, n=10, method='inclusive')[-1],
                      false_after_prev_median_us=stats['false']['median_case_saved_us']['after_prev'])
    return result


def rows(frames, first, end):
    for pos in range(first, end):
        for vi in range(18, 42):
            src, dst = frames[pos, vi - 12], frames[pos, vi]
            source_mask, target_mask = mask(src), mask(dst)
            accepted = {e for e, dev in zip(dst['ids'], dst['placement']) if dev == 'cpu'}
            predictions = [e for e, dev in zip(src['ids'], src['placement']) if dev == 'cpu' and not ((source_mask >> e) & 1)]
            if any((target_mask >> e) & 1 for e in predictions):
                raise ValueError('stale source residency; replay needs a route-independent live mask recheck')
            yield [e in accepted for e in predictions], dst['gpu_join_ms'] * 1000, (
                dst['route_ready_ns'] - dst['host_begin_ns']) / 1000


def project(visits, tokens, cap, hit_us, read_us=None, false_us=0):
    hits, total, false, on_time = 0, 0.0, 0, 0
    for picks, join_us, window_us in visits:
        planned = picks[:cap]
        delivered = planned if read_us is None else planned[:int(window_us // read_us)]
        n_hit = sum(delivered)
        hits += n_hit
        on_time += len(delivered)
        false += len(planned) - sum(planned)
        total += min(join_us, n_hit * hit_us)
    return dict(full_reads_on_time=on_time, useful_full_reads_on_time=hits,
                gross_saved_us_per_token=total / tokens, after_false_cost_us_per_token=(total + false * false_us) / tokens)


def analyze(frames, first, end, calibration):
    caps = (1, 2, 4, 8)
    trace, _ = screen(frames, first, end, caps)
    trace['fixed_savings'] = {}
    trace['near_helper_replay'] = {}
    tokens = end - first
    visits = list(rows(frames, first, end))
    for cap in caps:
        trace['fixed_savings'][cap] = {f'{us}_us_per_hit': project(visits, tokens, cap, us)
                                      for us in (100, 150, 200)}
    trace['untimed_one_hot_projection'] = project(visits, tokens, 1, calibration['ceiling']['hit_compute_median_us'])
    for key in ('schedule', 'schedule-repeat'):
        fit = calibration[key]
        trace['near_helper_replay'][key] = {band: project(visits, tokens, 1, fit['hit_compute_median_us'],
            fit[f'prep_post_to_finish_{band}_us'], fit['false_after_prev_median_us']) for band in ('median', 'p90')}
    return trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path, help='complete uncompressed CP-02 binary trace')
    p.add_argument('calibration_dir', type=Path, help='existing ceiling, schedule and schedule-repeat CSV/logs')
    a = p.parse_args()
    with a.trace.open('rb') as f:
        h = header(f)
        frames = {}
        for row in records(f, h):
            if row['kind'] != 'visit':
                continue
            key = row['pos'], row['visit']
            if key in frames or row['windowed'] or 'skipped' in row['placement']:
                p.error(f'not a unique serial decode frame at {key}')
            frames[key] = row
    first, end = h['first'], h['end']
    if len(frames) != 48 * (end - first) or end - first < 8:
        p.error('need at least eight complete frames spanning the declared capture')
    calibration = {name: calibrate(a.calibration_dir / f'{name}.csv', arm)
                   for name, arm in [('ceiling', 'oracle'), ('schedule', 'helper_read'), ('schedule-repeat', 'helper_read')]}
    print(json.dumps(dict(trace_sha256=digest(a.trace), calibration=calibration,
                         full=analyze(frames, first, end, calibration),
                         suffix=analyze(frames, first + (end - first) * 65 // 100 + 1, end, calibration)),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
