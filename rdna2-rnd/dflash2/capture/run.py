#!/usr/bin/env python3
"""Record stock W1 DFT1 features from a private CSV, or pair them with stock DSpark in one process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DOMAINS = ('en_reasoning', 'code_python', 'code_rust', 'code_cpp', 'tool_calls', 'json_struct', 'agent_traj')
VOCAB, EOS, K = 200058, 200006, 7
RECORD = struct.Struct('<qii')
FEATURE_BYTES = 4 * 2304 * 4


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def target_identity(path):
    # Identify the same local target inode without scanning a multi-GiB GGUF for every arm.
    path = path.resolve(strict=True)
    st = path.stat()
    return dict(path=str(path), device=st.st_dev, inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns)


def ids(values):
    if not isinstance(values, list) or not values or any(type(x) is not int or not 0 <= x < VOCAB for x in values):
        raise ValueError('expected nonempty in-vocabulary token IDs')
    return values


def csv(path):
    return ids(list(map(int, path.read_text().strip().split(','))))


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def checked_guard(guard):
    if not (guard / 'stop-reasons.json').is_file() or json.loads((guard / 'stop-reasons.json').read_text()) != []:
        raise RuntimeError(f'guard did not complete cleanly: {guard}')


def guarded(run, output, command, env, timeout):
    guard = output / 'guard'
    lock = run / 'gpu.lock'
    # The detached flock process holds this through GPU teardown even if the coordinator is killed.
    argv = ['flock', '--exclusive', str(lock), sys.executable, str(ROOT / 'rdna2-rnd/scripts/run_guarded.py'),
            '--warn-amdgpu-faults', '--output', str(guard), '--', 'timeout', '--kill-after=15s', str(timeout), *command]
    rc = subprocess.Popen(argv, env=env, cwd=ROOT, start_new_session=True).wait()
    if rc:
        raise RuntimeError(f'guard did not complete cleanly (exit {rc}): {guard}')
    checked_guard(guard)
    return guard


def validate_tape(path, prompt, out):
    expected = prompt + out[:-1]  # the last sampled or followed ID is never fed
    with path.open('rb') as f:
        if f.read(4) != b'DFT1':
            raise ValueError(f'{path}: invalid DFT1 header')
        if os.fstat(f.fileno()).st_size != 4 + len(expected) * (RECORD.size + FEATURE_BYTES):
            raise ValueError(f'{path}: missing/extra/truncated DFT1 rows (look for an accidentally fed final token)')
        for pos, token in enumerate(expected):
            if RECORD.unpack(f.read(RECORD.size)) != (pos, token, int(pos >= len(prompt))):
                raise ValueError(f'{path}: wrong position, token ID or phase at row {pos}')
            f.seek(FEATURE_BYTES, os.SEEK_CUR)


def validate_dspark(path, prompt_count, generated):
    lines = path.read_text().splitlines()
    if len(lines) != len(generated) - 2:
        raise ValueError(f'{path}: incomplete DSpark anchor coverage')
    full = []
    for i, line in enumerate(lines):
        pos, anchor, *fields = line.split()
        width = min(K, len(generated) - i - 2)
        if (int(pos), int(anchor), len(fields)) != (prompt_count + i, generated[i], width):
            raise ValueError(f'{path}: wrong DSpark position, anchor ID, or shortened tail width in row {i}')
        proposals = ids([int(x.split(':')[0]) for x in fields if x.count(':') == 2])
        if len(proposals) != width:
            raise ValueError(f'{path}: malformed DSpark proposal in row {i}')
        if width == K:
            full.append(proposals)
    return full


def capture(a, cfg, leg, prompt, follow, identities):
    leg.mkdir(exist_ok=False)
    args = [x for x in cfg['args'] if x != '--disk-sessions']
    if set(args) & {'--serve', '--tokens', '--follow', '--draft-model', '--no-draft', '--images'}:
        raise ValueError('need base target config without any request or drafter flags')
    profile = leg / 'profile'
    shutil.copyfile(a.profile, profile)
    args[args.index('--expert-profile') + 1] = str(profile)
    args[args.index('--expert-swaps') + 1] = '0'
    tape = leg / 'tap.dft1'
    proposals = leg / 'proposals.log'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'DFT1_', 'DF2_', 'RECON_'))}
    env.update({k: v for k, v in cfg['env'].items() if not k.startswith(('DFT1_', 'DF2_', 'RECON_'))})
    env.pop('STRATA_DRAFT_LOG', None)
    env.update(STRATA_INKLING_EXPERT_KV_MIB=str(a.cap), STRATA_FREEZE_EXPERTS='1', DFT1_CAPTURE=str(tape),
               LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']), LC_ALL='C')
    count = len(follow) if follow else (a.steps or 65)
    args += ['--lookup', '0', '--temperature', '0', '--seed', '42', '--eos-ids', str(EOS), '--max-new', str(count),
             '--tokens', ','.join(map(str, prompt))]
    if follow:
        args += ['--follow', ','.join(map(str, follow))]
    if a.dspark:
        args += ['--draft-model', str(a.dspark), '--draft-max', str(K), '--window', '8']
        env.update(STRATA_DRAFT_LOG=str(proposals), STRATA_INKLING_DSPARK_REUSE_V='1')
    else:
        args += ['--no-draft', '--window', '1']
    req = [str(a.binary), *args]
    write(leg / 'request.json', identities | {'argv': req, 'env': {k: v for k, v in env.items()
              if k.startswith(('DFT1_', 'STRATA_')) or k in ('LD_LIBRARY_PATH', 'LC_ALL')}})
    guarded(a.run, leg, req, env, a.timeout)
    return capture_receipt(leg, prompt, follow, identities, count, bool(a.dspark))


def one(pattern, text, leg, expected):
    matches = re.findall(pattern, text, re.M)
    if len(matches) != 1:
        raise ValueError(f'{leg}: expected exactly one {expected} in guard/command.log, found {len(matches)}')
    return matches[0]


def capture_receipt(leg, prompt, follow, identities, count, dspark):
    guard = leg / 'guard'
    checked_guard(guard)
    tape, proposals = leg / 'tap.dft1', leg / 'proposals.log'
    text = (guard / 'command.log').read_text()
    engine = one(r'^prompt (\d+) tokens [^\n]*?generated (\d+) tokens [^\n]*?\((length|stop)\)', text, leg, 'CLI completion')
    summary = one(r'^DFT1_COMPLETE rows=(\d+) generated=(\d+) layout=([0-9a-f]{16}) swaps=0$', text, leg, 'DFT1 hook completion')
    slot_count = one(r'; (\d+) of \d+ routed experts in VRAM', text, leg, 'placement report')
    output = one(r'^(\d+(?:,\d+)*)$', text, leg, 'output token CSV')
    output = ids(list(map(int, output.split(','))))
    if (int(engine[0]), int(engine[1])) != (len(prompt), len(output)) or (int(summary[0]), int(summary[1])) != (
            len(prompt) + len(output) - 1, len(output) - 1):
        raise ValueError(f'{leg}: wrong number of read, emitted, or tapped tokens')
    if len(output) > count or EOS in output[:-1] or (engine[2] == 'stop') != (output[-1] == EOS) or (
            len(output) < count and engine[2] != 'stop') or (follow and output != follow):
        raise ValueError(f'{leg}: wrong output length, EOS tail or forced IDs')
    validate_tape(tape, prompt, output)
    full = len(validate_dspark(proposals, len(prompt), output)) if dspark else None
    if dspark and full != len(output) - K - 1:
        raise ValueError(f'{leg}: no common full-width DSpark/DFlash2 coverage')
    receipt = identities | dict(prompt_ids=prompt, tokens=output, prompt_len=len(prompt), gen_len=len(output),
                                 tape_sha256=digest(tape), rows=int(summary[0]), slots=int(slot_count), layout=summary[2],
                                 proposals_sha256=digest(proposals) if dspark else None,
                                 swaps=0, forced=follow is not None, full_width_anchors=full, finish=engine[2])
    if (leg / 'result.json').exists():
        if receipt != json.loads((leg / 'result.json').read_text()):
            raise ValueError(f'{leg}: inputs or inspected artifacts changed since receipt')
    else:
        write(leg / 'result.json', receipt)
    print(f'{leg.name}: {len(prompt)} prompt, {len(output)} emitted, {summary[0]} tapped, '
          f'{slot_count} slots, {full if full is not None else 0} paired full-width anchors', flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'out', 'config', 'profile', 'binary'):
        p.add_argument('--' + key, type=Path, required=True)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prompt', type=Path, help='private comma-separated prompt IDs')
    modes.add_argument('--manifest', type=Path, help='old manifest: only reads prompt_ids and tokens')
    p.add_argument('--follow', type=Path, help='private comma-separated output IDS (single prompt only)')
    p.add_argument('--domain', action='append', choices=DOMAINS, help='subset of manifest (default all seven)')
    p.add_argument('--dspark', type=Path, help='pair capture with this DSpark GGUF, on forced W1 history')
    p.add_argument('--cap', type=int, default=3072, help='fixed expert+KV arena MiB')
    p.add_argument('--steps', type=int, help='output limit; clip forced inputs to this length')
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--resume', action='store_true', help='reuse only complete receipts matching the inputs')
    a = p.parse_args()
    if (a.domain and not a.manifest) or (a.follow and not a.prompt) or (a.dspark and not (a.manifest or a.follow)):
        p.error('DSpark pairing needs forced IDs; --domain requires --manifest, --follow requires --prompt')
    if min(a.cap, a.timeout, a.steps if a.steps is not None else 1) < 1:
        p.error('cap, timeout and steps must be positive')
    os.umask(0o077)
    a.run = a.run.resolve(strict=True)
    a.out = a.out.resolve()
    a.binary = a.binary.resolve(strict=True)
    a.dspark = a.dspark.resolve(strict=True) if a.dspark else None
    cfg = json.loads(a.config.read_text())
    if a.manifest:
        old = json.loads(a.manifest.read_text())
        legs = {name: (ids(old[name]['prompt_ids']), ids(old[name]['tokens']))
                for name in (a.domain or DOMAINS)}
    else:
        legs = {'prompt': (csv(a.prompt), csv(a.follow) if a.follow else None)}
    target = Path(cfg['args'][cfg['args'].index('--model') + 1])
    identity = dict(target=target_identity(target), binary_sha256=digest(a.binary),
                    config_sha256=digest(a.config), profile_sha256=digest(a.profile),
                    dspark_sha256=digest(a.dspark) if a.dspark else None, cap_mib=a.cap)
    a.out.mkdir(parents=True, exist_ok=a.resume)
    results = {}
    for name, (prompt, follow) in legs.items():
        if follow and a.steps:
            follow = follow[:a.steps]
        if follow and (EOS in follow[:-1] or (a.dspark and len(follow) < K + 2)):
            p.error('forced IDs must not continue past EOS; pairing requires at least 9 outputs')
        leg = a.out / name
        if a.resume and leg.exists():
            if not (leg / 'result.json').is_file():
                raise ValueError(f'{leg}: incomplete leg; preserve it separately before resuming')
            results[name] = capture_receipt(leg, prompt, follow, identity, len(follow) if follow else (a.steps or 65), bool(a.dspark))
        else:
            results[name] = capture(a, cfg, leg, prompt, follow, identity)
        if a.manifest and len({(row['layout'], row['slots']) for row in results.values()}) != 1:
            raise ValueError(f'{leg}: different expert placement across campaign')
    manifest = a.out / 'manifest.json'
    if not manifest.exists():
        write(manifest, results)
    elif not a.resume or json.loads(manifest.read_text()) != results:
        raise ValueError(f'{manifest}: existing campaign differs')


if __name__ == '__main__':
    main()
