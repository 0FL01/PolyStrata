#!/usr/bin/env python3
"""Replay the five native DFlash2 shapes on the same W1 tensors consumed by DSpark."""
import argparse
import json
import os
from pathlib import Path

from run import DOMAINS, K, checked_guard, digest, guarded, ids, target_identity, validate_dspark, validate_tape, write

PROFILES = ('BF16', 'Q8_0', 'Q4_0', 'Q4_0-mixed', 'Q4_K-mixed')
ALL_POSITIONS = 'all-positions'


def prefix_matches(guesses, truth):
    return next((i for i, (a, b) in enumerate(zip(guesses, truth)) if a != b), K)


def stats(accepted):
    all_rows = list(accepted)
    closed = []
    i = 0
    while i < len(all_rows):
        closed.append(all_rows[i])
        i += 1 + all_rows[i]
    return {mode: summarize(rows) for mode, rows in [(ALL_POSITIONS, all_rows), ('closed-loop', closed)]}


def summarize(accepted):
    counts = [accepted.count(i) for i in range(K + 1)]
    return dict(anchors=len(accepted), sum_accepted=sum(accepted), accepted_histogram=counts,
                emitted_per_verify=1 + sum(accepted) / len(accepted),
                prefix_survival=[sum(counts[i:]) / len(accepted) for i in range(1, K + 1)])


def native_predictions(path, base, tokens, truth):
    guesses = []
    with path.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if i >= len(truth) or (row['mode'], row['start'], row['anchor']) != (ALL_POSITIONS, base + i, tokens[i]):
                raise ValueError(f'{path}: mismatched native anchor at row {i}')
            pred = ids(row['draft_tokens'])
            if len(pred) != K or row['accepted'] != prefix_matches(pred, truth[i]):
                raise ValueError(f'{path}: mismatched width or native accepted prefix at row {i}')
            guesses.append(pred)
    if len(guesses) != len(truth):
        raise ValueError(f'{path}: incomplete native replay')
    return guesses


def one(a, leg, capture, path, identity, truth):
    tape, receipt = capture
    steps = leg / 'steps.jsonl'
    result = leg / 'result.json'
    request = identity | dict(tape_sha256=receipt['tape_sha256'])
    if a.resume and leg.exists():
        checked_guard(leg / 'guard')
        old = json.loads(result.read_text())
        if old != request | dict(steps_sha256=digest(steps), summary_sha256=digest(leg / 'summary.json')):
            raise ValueError(f'{leg}: replay inputs or results changed since receipt')
    else:
        leg.mkdir(parents=True, exist_ok=False)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'DF2_', 'DFT1_', 'RECON_'))}
        env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (str(a.binary.parent), env.get('LD_LIBRARY_PATH'))))
        cmd = [str(a.binary), '--target', str(a.target), '--model', str(path), '--replay', str(tape),
               '--mode', ALL_POSITIONS, '--dump-json', str(leg / 'summary.json'), '--dump-steps', str(steps)]
        guarded(a.run, leg, cmd, env, a.timeout)
    [run] = json.loads((leg / 'summary.json').read_text())
    if (run['n_prompt'], run['n_gen'], run['first_pos'], run['all_positions']['steps']) != (
            receipt['prompt_len'], receipt['gen_len'] - 1, 0, len(truth)):
        raise ValueError(f'{leg}: native summary coverage mismatch')
    preds = native_predictions(steps, receipt['prompt_len'], receipt['tokens'], truth)
    if not result.exists():
        write(result, request | dict(steps_sha256=digest(steps), summary_sha256=digest(leg / 'summary.json')))
    return preds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'capture', 'out', 'target', 'models', 'binary'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--domain', action='append', choices=DOMAINS, help='default all seven')
    p.add_argument('--profile', action='append', choices=PROFILES, help='default all five')
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--resume', action='store_true', help='reuse only complete receipts matching the inputs')
    a = p.parse_args()
    if a.timeout < 1:
        p.error('timeout must be positive')
    os.umask(0o077)
    for key in ('run', 'capture', 'target', 'models', 'binary'):
        setattr(a, key, getattr(a, key).resolve(strict=True))
    a.out = a.out.resolve()
    domains = a.domain or DOMAINS
    profiles = a.profile or PROFILES
    target = target_identity(a.target)
    binary_id = dict(target=target, binary_sha256=digest(a.binary),
                     runtime_sha256=digest(a.binary.with_name('libstrata-dflash2.so')))
    manifest_path = a.capture / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    a.out.mkdir(parents=True, exist_ok=a.resume)
    report = {'capture_manifest_sha256': digest(manifest_path), 'draft_width': K, 'domains': {}, 'pooled': {}}
    fixtures = {}
    for name in domains:
        source = a.capture / name
        receipt = manifest[name]
        checked_guard(source / 'guard')
        if (receipt != json.loads((source / 'result.json').read_text()) or not receipt['forced'] or receipt['swaps'] or
                not receipt['dspark_sha256'] or receipt['target'] != target):
            raise ValueError(f'{source}: need a same-process forced W1 DSpark capture on the requested target')
        prompt, outputs = receipt['prompt_ids'], receipt['tokens']
        tape, ds = source / 'tap.dft1', source / 'proposals.log'
        if digest(tape) != receipt['tape_sha256'] or digest(ds) != receipt['proposals_sha256']:
            raise ValueError(f'{source}: capture changed since receipt')
        validate_tape(tape, prompt, outputs)
        dspark = validate_dspark(ds, len(prompt), outputs)
        if len(dspark) != len(outputs) - K - 1 or len(dspark) != receipt['full_width_anchors']:
            raise ValueError(f'{source}: missing common full-width cohort')
        truth = [outputs[pos + 1:pos + K + 1] for pos in range(len(dspark))]
        fixtures[name] = ((tape, receipt), truth)
        report['domains'][name] = dict(anchors=len(truth), slots=receipt['slots'], layout=receipt['layout'],
                                       methods={'DSpark': stats(prefix_matches(p, t) for p, t in zip(dspark, truth))})

    for profile in profiles:
        path = (a.models / f'Inkling-2-Mini-DFlash2-{profile}.gguf').resolve(strict=True)
        identity = binary_id | dict(model_sha256=digest(path), model=str(path))
        for domain, (capture, truth) in fixtures.items():
            out = a.out / profile / domain
            preds = one(a, out, capture, path, identity, truth)
            scores = stats(prefix_matches(p, t) for p, t in zip(preds, truth))
            report['domains'][domain]['methods'][profile] = scores
            print(f'{profile}/{domain}: {len(truth)} anchors, '
                  f'{scores[ALL_POSITIONS]["emitted_per_verify"]:.3f} tokens per verify (offline)', flush=True)

    for method in ('DSpark', *profiles):
        report['pooled'][method] = {}
        for mode in (ALL_POSITIONS, 'closed-loop'):
            rows = [row['methods'][method][mode] for row in report['domains'].values()]
            n = sum(x['anchors'] for x in rows)
            accepted = sum(x['sum_accepted'] for x in rows)
            report['pooled'][method][mode] = dict(anchors=n, sum_accepted=accepted,
                emitted_per_verify=1 + accepted / n,
                equal_domain_emitted_per_verify=sum(x['emitted_per_verify'] for x in rows) / len(rows))
    dest = a.out / 'comparison.json'
    if not dest.exists():
        write(dest, report)
    elif not a.resume or json.loads(dest.read_text()) != report:
        raise ValueError(f'{dest}: existing comparison differs')
    print(json.dumps(report['pooled'], indent=2))


if __name__ == '__main__':
    main()
