#!/usr/bin/env python3
"""Report paired baseline/exact DFlash2 proposal latencies and byte parity."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    path = parser.parse_args().receipt
    result = {}
    metrics = ('first_proposal_ms', 'subsequent_proposal_ms', 'proposal_ms_per_step',
               'append_ms_per_step', 'ms_per_step', 'prompt_append_ms')
    for stock in sorted(path.glob('*-stock.steps.jsonl')):
        name = stock.name.removesuffix('-stock.steps.jsonl')
        a = stock.read_bytes()
        b = (path / (name + '-exact.steps.jsonl')).read_bytes()
        count = a.count(b'\n')
        row = {'positions': count, 'greedy_ids': count * 7, 'byte_identical': a == b,
               'stock_sha256': hashlib.sha256(a).hexdigest(), 'exact_sha256': hashlib.sha256(b).hexdigest()}
        result[name] = row
        for arm in ('stock', 'exact'):
            timings = sorted(path.glob(f'{name}-{arm}-t[1-5].json'))
            if not timings:
                continue
            data = [json.loads(t.read_text())[0]['all_positions'] for t in timings]
            row[arm] = {key: median([r[key] for r in data]) for key in metrics}
        if 'stock' in row and 'exact' in row:
            row['proposal_reduction_pct'] = {key: 100 * (1 - row['exact'][key] / row['stock'][key])
                                             for key in metrics[:3]}
    print(json.dumps(result, indent=2))
    if not result or not all(row['byte_identical'] and row['positions'] == 25 for row in result.values()):
        raise SystemExit('incomplete parity check or mismatch')


if __name__ == '__main__':
    main()
