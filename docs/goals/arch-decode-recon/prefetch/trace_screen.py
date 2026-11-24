#!/usr/bin/env python3
"""Screen a CPU-only previous-pass policy and export bounded CP-02 bench cases."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics


EXPERT_BYTES = 4_478_976


def read(path):
    frames = {}
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for i, line in enumerate(stream):
            digest.update(line)
            row = json.loads(line)
            if i == 0:
                header = row
                if row['kind'] != 'header' or [row[k] for k in ('layers', 'visits', 'experts', 'top')] != [24, 48, 128, 8]:
                    raise ValueError('expected CP-02 Inkling serial trace header')
            if row['kind'] != 'visit':
                continue
            key = (row['pos'], row['visit'])
            visit = row['visit']
            layer = visit if visit < 6 else 6 + (visit - 6) % 12 if visit < 42 else visit - 24
            if (key in frames or row['windowed'] or row['layer'] != layer or
                len(row['ids']) != 8 or len(set(row['ids'])) != 8 or not all(0 <= e < 128 for e in row['ids']) or
                len(row['weight_bits']) != 8 or len(row['placement']) != 8 or
                not set(row['placement']) <= {'cpu', 'gpu'}):
                raise ValueError(f'invalid serial visit {key}')
            frames[key] = row
    if not frames or set(frames) != {(p, v) for p in range(header['first'], header['end']) for v in range(48)}:
        raise ValueError('missing or extra frame')
    return header, frames, digest.hexdigest()


def mask(row):
    return int(row['resident_mask'], 16) | int(row['asked_mask'], 16) | int(row['sent_mask'], 16)


def cpu(row):
    return [(e, w) for e, w, b in zip(row['ids'], row['weight_bits'], row['placement']) if b == 'cpu']


def bands(xs):
    return [round(xs[0], 3), round(statistics.median(xs), 3), round(xs[-1], 3)] if len(xs) < 10 else [
        round(statistics.quantiles(xs, n=10, method='inclusive')[i], 3) for i in (0, 4, 8)]


def screen(frames, first, end, caps):
    grouped = defaultdict(list)
    joins = Counter()
    tally = {k: Counter() for k in caps}
    times = defaultdict(list)
    for pos in range(first, end):
        for visit in range(18, 42):
            source, prev, dst = (frames[pos, v] for v in (visit - 12, visit - 1, visit))
            selected = dict(zip(dst['ids'], dst['placement']))
            actual = cpu(dst)
            join = dst['gpu_join_ms']
            near = dst['route_ready_ns'] - dst['host_begin_ns']
            lead = dst['route_ready_ns'] - source['route_ready_ns']
            if not 0 < near <= lead or join < 0:
                raise ValueError(f'invalid timing at {pos}/{visit}')
            joins['cpu_misses'] += len(actual)
            joins['join_proxy_ms'] += join
            picks = [e for e, b in zip(source['ids'], source['placement']) if b == 'cpu' and not ((mask(source) >> e) & 1)]
            for cap, count in tally.items():
                arm = picks[:cap]
                hits = sum(selected.get(e) == 'cpu' for e in arm)
                count['issued'] += len(arm)
                count['hits'] += hits
                count['hit_visits'] += bool(hits)
                count['false'] += len(arm) - hits
                count['target_gpu_selected'] += sum(selected.get(e) == 'gpu' for e in arm)
                count['became_resident_or_pending'] += sum((mask(dst) >> e) & 1 for e in arm)
                count['hit_visit_join_proxy_ms'] += join * bool(hits)
                # Diagnostic only: assumes equal share, not an upper bound or a scheduling result.
                count['proportional_join_proxy_ms'] += join * hits / max(1, len(actual))
            if not picks:
                continue
            expert = picks[0]
            hit = selected.get(expert) == 'cpu'
            prev_cpu = cpu(prev)
            times['near_us'].append(near / 1000)
            times['router_wait_us'].append(dst['router_wait_ns'] / 1000)
            times['same_layer_pass_lead_us'].append(lead / 1000)
            if hit:
                reads = sum(len(cpu(frames[pos, v])) for v in range(visit - 11, visit))
                times['intervening_cpu_uses_on_hit'].append(reads)
            grouped[hit, len(actual)].append((pos, visit, expert, near, round(join * 1e6), prev_cpu, actual))
    n = end - first
    arms = {}
    for cap, counts in tally.items():
        arms[cap] = dict(counts, precision=counts['hits'] / max(1, counts['issued']),
                         recall=counts['hits'] / max(1, joins['cpu_misses']),
                         issue_MB_per_token=counts['issued'] * EXPERT_BYTES / n / 1e6,
                         false_MB_per_token=counts['false'] * EXPERT_BYTES / n / 1e6,
                         hit_visit_join_proxy_us_per_token=counts['hit_visit_join_proxy_ms'] * 1000 / n,
                         proportional_join_proxy_us_per_token=counts['proportional_join_proxy_ms'] * 1000 / n)
    result = dict(tokens=n, first=first, end=end, **joins, arms=arms,
                  timing_p10_p50_p90={k: bands(v) for k, v in times.items()},
                  strata={f'{int(h)}/{k}': len(v) for (h, k), v in sorted(grouped.items())},
                  issued_with_cpu_predecessor=sum(bool(row[5]) for rows in grouped.values() for row in rows))
    return result, grouped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--cases', type=Path, help='exclusive create: stratified benchmark cases for last 35%%')
    args = p.parse_args()
    header, frames, digest = read(args.trace)
    start, end = header['first'], header['end']
    if end - start < 8:
        p.error('need at least eight frames')
    split = start + (end - start) * 65 // 100 + 1  # align with route/screen_lookahead.py
    all_rows, _ = screen(frames, start, end, (1, 2, 4))
    suffix, groups = screen(frames, split, end, (1, 2, 4))
    if args.cases:
        # The selected CP-02 trace has no within-token map transitions on picks. Do not use
        # target-route information to simulate an online residency lookup on another trace.
        if suffix['arms'][1]['became_resident_or_pending'] or suffix['arms'][1]['target_gpu_selected']:
            p.error('source mapping became stale: need a route-independent live mask recheck')
        rng = random.Random(4242)
        sample = []
        for _, rows in sorted(groups.items()):
            count = min(3, len(rows))
            sample.extend((*row, len(rows), count) for row in rng.sample(rows, count))
        rng.shuffle(sample)
        with args.cases.open('x') as out:
            print('PREFETCH1', suffix['tokens'], len(sample), file=out)
            for pos, visit, expert, near, join, prev, actual, pop, size in sample:
                print(pos, visit, expert, near, join, pop, size, len(prev), len(actual),
                      *(value for pair in prev + actual for value in pair), file=out)
    print(json.dumps(dict(trace_sha256=digest, full=all_rows, suffix=suffix), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
