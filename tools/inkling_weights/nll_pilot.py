"""Measure exact target NLL on the frozen 2K/16K documents, one guarded process per document."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from .chat_pilot import LOCK, MANIFEST, ROOT, WORK, checked_bytes, write_json


def read_document(path, prefix_tokens, targets):
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if len(rows) != len(targets):
        raise ValueError(f'{path}: wrong NLL row count')
    values = []
    for i, (row, target) in enumerate(zip(rows, targets, strict=True)):
        if [int(row[k]) for k in ('step', 'position', 'target_id')] != [i, prefix_tokens + i, target]:
            raise ValueError(f'{path}: incorrect NLL row identity at step {i}')
        nll = float(row['nll'])
        if not math.isfinite(nll) or nll < 0:
            raise ValueError(f'{path}: invalid NLL at step {i}')
        values.append(nll)
    return {'targets': len(values), 'mean_nll': statistics.mean(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory under the private experiment root')
    parser.add_argument('--observer', type=Path, required=True)
    parser.add_argument('--case', action='append', help='run only the specified frozen NLL documents')
    parser.add_argument('--reference', type=Path, help='prior nll-result.json for a paired document comparison')
    args = parser.parse_args()
    os.umask(0o077)
    lock = json.loads(LOCK.read_bytes())
    corpus = json.loads(checked_bytes(MANIFEST, lock['corpus_sha256']))
    observer = args.observer.resolve(strict=True)
    observer_sha = lock['standalone']['observer_sha256']
    checked_bytes(observer, observer_sha)
    seed = checked_bytes(Path(lock['standalone']['profile_seed_path']), lock['standalone']['profile_seed_sha256'])
    model = args.model.resolve(strict=True)
    out = args.output.resolve()
    if not model.is_file() or not out.is_relative_to(WORK.resolve()):
        parser.error('need a model file and an output under the experiment root')
    cases = [c for c in corpus['nll'] if not args.case or c['id'] in args.case]
    ids = {c['id'] for c in cases}
    if args.case and ids != set(args.case):
        parser.error('unknown NLL case')
    reference = None
    if args.reference:
        ref = json.loads(args.reference.read_bytes())
        if (ref['corpus_sha256'], ref['observer_sha256'], set(ref['documents'])) != (
                lock['corpus_sha256'], observer_sha, ids):
            parser.error('reference did not use the same frozen inputs and observer')
        reference = ref['documents']
    active = subprocess.run(['systemctl', '--user', 'show', '-p', 'ActiveState', '--value', lock['production']['unit']],
                            check=True, capture_output=True, text=True).stdout.strip()
    if active != 'inactive':
        raise RuntimeError('production must be inactive for standalone runs')
    out.mkdir(mode=0o700)
    env = {'HOME': str(Path.home()), 'PATH': os.environ['PATH'], **lock['engine_env']}
    docs = {}
    for c in cases:
        prefix, target = (WORK / 'corpus' / c[k]['ids_path'] for k in ('prefix', 'target'))
        checked_bytes(prefix, c['prefix']['ids_sha256'])
        targets = list(map(int, checked_bytes(target, c['target']['ids_sha256']).decode('ascii').split(',')))
        if len(targets) != c['target']['tokens']:
            raise ValueError(f'{target}: wrong target count')
        d = out / c['id']
        d.mkdir(mode=0o700)
        profile = d / 'profile.txt'
        profile.write_bytes(seed)
        subprocess.run([
            sys.executable, '-B', str(ROOT / 'rdna2-rnd/scripts/run_guarded.py'), '--output', 'guard',
            '--warn-amdgpu-faults', '--',
            'timeout', '--foreground', '--kill-after=10s', f'{lock["benchmark"]["per_standalone_run_timeout_seconds"]}s',
            'sh', '-c', 'exec "$@" > nll.tsv 2> stderr.log', 'sh', str(observer), str(model), str(profile),
            str(prefix), str(target),
        ], cwd=d, env={**env, 'TMPDIR': str(d)}, check=True)
        if json.loads((d / 'guard/stop-reasons.json').read_bytes()) != []:
            raise RuntimeError(f'{d}: guard did not clear')
        docs[c['id']] = read_document(d / 'nll.tsv', c['prefix']['tokens'], targets)
    result = {'model': str(model), 'observer_sha256': observer_sha, 'corpus_sha256': lock['corpus_sha256'],
              'documents': docs, 'mean_nll': statistics.mean(d['mean_nll'] for d in docs.values())}
    if reference is not None:
        delta = {i: d['mean_nll'] - reference[i]['mean_nll'] for i, d in docs.items()}
        result['paired_delta_nats'] = {'by_document': delta, 'mean': statistics.mean(delta.values()),
                                      'document_stdev': statistics.stdev(delta.values()) if len(delta) > 1 else None}
    write_json(out / 'nll-result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
