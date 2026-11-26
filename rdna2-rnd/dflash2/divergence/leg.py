#!/usr/bin/env python3
"""Run one guarded instrumented W1/2/4/8 call on the shared RDNA2 host."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess as sp
import sys

from build import REPO, R

CACHE = R.parents[2]
ARCHIVE = CACHE / 'arch-campaign.XWM1yK'
FIXED = CACHE / 'polystrata-runs/native-donor-bc2b/baseline/fixed.json'
PY = REPO / '.venv/bin/python'
EXE = R / 'bin/strata-poly-divtrace'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(label, width, snapshot, baseline, settings):
    if not re.fullmatch(r'[a-zA-Z0-9-]+', label):
        raise ValueError('invalid leg name')
    if baseline and not re.fullmatch(r'[a-zA-Z0-9-]+', baseline):
        raise ValueError('invalid baseline name')
    force = (R / baseline / 'force.txt').resolve(strict=True) if baseline else None
    if not force and width != 1:
        raise ValueError('verifier leg requires a serial baseline')
    os.umask(0o077)
    out = R / label
    out.mkdir(parents=True)
    args = json.loads(FIXED.read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'RECON_', 'DIV_'))}
    env.update(args['env'], LD_LIBRARY_PATH=':'.join(args['lib_dirs']), LC_ALL='C.UTF-8')
    if env.get('STRATA_INKLING_EXPERT_KV_MIB') != '4968':
        raise ValueError('fixed budget changed')
    env.update(STRATA_FREEZE_EXPERTS='1', DIV_TRACE=str(out / 'trace.dvt1'), DIV_WIDTH=str(width))
    for setting in settings:
        name, value = setting.split('=', 1)
        if not name.startswith('DIV_FIX_'):
            raise ValueError('only DIV_FIX_* flags may change')
        env[name] = value
    opt = args['args'][:]
    opt[opt.index('--expert-swaps') + 1] = '0'
    prof = out / 'profile.seed'
    shutil.copyfile(ARCHIVE / 'profile.seed', prof)
    opt[opt.index('--expert-profile') + 1] = str(prof)
    if force:
        env['DIV_FORCE'] = str(force)
    with snapshot.open('rb') as f:
        f.seek(232)
        saved, = struct.unpack('<Q', f.read(8))
        if not 0 < saved < 199980:
            raise ValueError('invalid saved ID count')
        prompt = list(struct.unpack(f'<{saved}i', f.read(4 * saved)))
    if saved > 4096:
        # The fixed cap needs to release expert arena pages at long RESTORE admission. Still no swaps;
        # the subsequent trace requires exact matching slot layouts, not just the same slot count.
        env.pop('STRATA_FREEZE_EXPERTS')
    prompt.append(prompt[-1])  # same single fresh input after RESTORE in all legs; no retokenization
    request = {'args': [str(EXE), '--serve', *opt, '--no-draft', '--lookup', '0'],
               'snapshot': str(snapshot.resolve()), 'prompt': ','.join(map(str, prompt)), 'saved': saved,
               'binary_sha256': sha(EXE.read_bytes()), 'profile_sha256': sha(prof.read_bytes()),
               'env_overrides': {k: v for k, v in env.items() if k.startswith(('STRATA_', 'DIV_'))}}
    (out / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    with (R.parent / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sp.run([str(PY), str(REPO / 'rdna2-rnd/scripts/run_guarded.py'), '--warn-amdgpu-faults',
                '--output', str(out / 'guard'), '--', 'timeout', '--kill-after=15s', '180s',
                sys.executable, '-B', __file__, '--drive', str(out)], env=env, cwd=REPO, check=True)
    print(out / 'receipt.json')


def drive(out):
    req = json.loads((out / 'request.json').read_text())
    with (out / 'engine.log').open('w') as err, (out / 'wire.log').open('w') as wire, \
            sp.Popen(req['args'], text=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=err) as p:
        def rd():
            line = p.stdout.readline()
            if not line or line.startswith(('ERR ', 'FATAL ', 'SERR ')):
                raise RuntimeError(line.strip() or 'EOF from engine')
            wire.write(line)
            wire.flush()
            return line.strip()

        def until(prefix):
            while True:
                line = rd()
                if line.startswith(prefix):
                    return line

        def send(line):
            p.stdin.write(line + '\n')
            p.stdin.flush()

        try:
            ready = until('READY ')
            send('RESTORE ' + req['snapshot'])
            restored = until('RESTORED ')
            send('GEN 9 temperature=0 top_p=1 top_k=0 min_p=0 seed=42 ' + req['prompt'])
            tokens, resume = [], None
            while True:
                line = rd()
                if line.startswith('T '):
                    tokens.append(int(line.split()[1]))
                elif line.startswith('RESUME '):
                    resume = int(line.split()[1])
                elif line.startswith('DONE '):
                    break
            send('QUIT')
            p.stdin.close()
            if p.wait(timeout=30):
                raise RuntimeError('engine reported nonzero exit')
        except BaseException:
            p.kill()
            raise
    width = int(os.environ['DIV_WIDTH'])
    want = 8 if width == 1 else width
    log = (out / 'engine.log').read_text()
    result, = re.findall(r'^DIV_RESULT (.*)', log, re.M)
    placement, = re.findall(r'^DIV_PLACEMENT (.*)', log, re.M)
    fields = dict(x.split('=') for x in result.split())
    place = dict(x.split('=') for x in placement.split())
    assert resume == req['saved'] == int(restored.split()[1]) and int(fields['pos']) == resume + 1
    assert int(fields['rows']) == want and len(tokens) == want + 1 and place['swaps'] == '0'
    assert place['mode'] == ('decode' if width == 1 else 'window') and int(place['n']) == width
    if width == 1:
        if 'DIV_FORCE' in os.environ:
            shutil.copyfile(os.environ['DIV_FORCE'], out / 'force.txt')
        else:
            (out / 'force.txt').write_text('\n'.join(map(str, tokens[:8])) + '\n')
    (out / 'receipt.json').write_text(json.dumps({
        'prompt_sha256': sha(req['prompt'].encode()), 'snapshot': req['snapshot'],
        'binary_sha256': req['binary_sha256'], 'profile_sha256': req['profile_sha256'],
        'ready': ready, 'resume': resume, 'result': fields, 'placement': place, 'done': line,
    }, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label', nargs='?')
    p.add_argument('--width', type=int, choices=[1, 2, 4, 8])
    p.add_argument('--snapshot', type=Path, default=ARCHIVE / 'mixed-2048/session.bin')
    p.add_argument('--baseline')
    p.add_argument('--setting', action='append', default=[])
    p.add_argument('--drive', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.drive:
        drive(a.drive)
    elif a.label and a.width:
        prepare(a.label, a.width, a.snapshot, a.baseline, a.setting)
    else:
        p.error('provide one label and width')
