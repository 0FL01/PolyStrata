#!/usr/bin/env python3
"""Prepare a private CP-04 config, or resolve rank 0 from a finished guarded CP-02 capture."""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / 'trace'))
import collect

POSITION, VISIT, LAYER = 2048, 6, 6
BYTES = 2304 * 4 + 72 * 36
MODEL = 'Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf'
PRODUCTION = Path.home() / '.config/strata/inkling2-mini/strata-inkling-2-mini.json'


def read_json(path):
    return json.loads(path.read_bytes())


def capture_env(run):
    return dict(STRATA_CP04_CAPTURE=str(run / 'input.bin'), STRATA_CP04_POSITION=str(POSITION),
                STRATA_CP04_VISIT=str(VISIT))


def prepare(directory, config):
    if directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077:
        raise ValueError('capture directory must be owned by the caller and private (mode 0700)')
    src = read_json(config)
    values = capture_env(directory / 'run')
    if any(k in src['env'] for k in values):
        raise ValueError('source config already sets a CP-04 capture variable')
    src['env'].update(values)
    out = directory / 'capture-config.json'
    with os.fdopen(os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as f:
        json.dump(src, f, indent=2)
        f.write('\n')
    return out


def select_route(trace):
    reader = gzip.open if trace.suffix == '.gz' else open
    with reader(trace, 'rb') as f:
        h = collect.header(f)
        if h['first'] != POSITION:
            raise ValueError(f'trace must start at input position {POSITION}')
        rows = [r for r in collect.records(f, h) if r['kind'] == 'visit' and r['visit'] == VISIT]
    if sorted(r['pos'] for r in rows) != list(range(h['first'], h['end'])):
        raise ValueError('trace must contain exactly one complete visit frame per captured position')
    row = next(r for r in rows if r['pos'] == POSITION)
    if row['layer'] != LAYER or row['windowed'] or row['placement'][0] == 'skipped':
        raise ValueError('target visit must be serial eval of layer 6, with rank 0 executed')
    bits = row['weight_bits'][0]
    weight, = struct.unpack('<f', struct.pack('<I', bits))
    if not math.isfinite(weight):
        raise ValueError('rank-0 route weight is not finite')
    return row, bits, weight.hex()


def replay(directory, backend):
    run = directory / 'run'
    if read_json(directory / 'guard/stop-reasons.json') != []:
        raise ValueError('capture guard reported a hard stop')
    config, metrics = read_json(run / 'config.json'), read_json(run / 'metrics.json')
    if config['length'] != POSITION or metrics['length'] != POSITION or not config['trace']:
        raise ValueError('not a traced run from a 2048-token prompt')
    env = config['env']
    trace = run / 'visits.psv'
    for name, expected in dict(capture_env(run), STRATA_VISIT_TRACE=str(trace),
                               STRATA_INKLING_VISIT_TRACE_FIRST=str(POSITION),
                               STRATA_INKLING_VISIT_TRACE_END=str(POSITION + metrics['decode_steps'])).items():
        if env.get(name) != expected:
            raise ValueError(f'capture run has unexpected {name}')
    with trace.open('rb') as f:
        if collect.header(f)['end'] != POSITION + metrics['decode_steps']:
            raise ValueError('trace filter end does not match completed decode steps')
    marks = re.findall(r'^CP04 capture: pos=(\d+) visit=(\d+) layer=(\d+)$',
                       (run / 'engine.log').read_text(errors='replace'), re.MULTILINE)
    if marks != [(str(POSITION), str(VISIT), str(LAYER))]:
        raise ValueError('expected exactly one capture marker for position 2048 visit 6 layer 6')
    raw = run / 'input.bin'
    data = raw.read_bytes()
    if len(data) != BYTES:
        raise ValueError(f'capture should be {BYTES} bytes (xn then xq)')
    model = Path(config['cmd'][config['cmd'].index('--model') + 1])
    if not model.is_absolute() or model.name != MODEL:
        raise ValueError('capture used the wrong GGUF')

    row, bits, weight = select_route(trace)
    expert = row['ids'][0]
    evidence = dict(position=POSITION, visit=VISIT, layer=LAYER, rank=0, serial=row['serial'], expert=expert,
                    placement=row['placement'][0], weight_bits=f'0x{bits:08x}', weight_hex=weight,
                    input_sha256=hashlib.sha256(data).hexdigest())
    probe = HERE / 'out' / f'parity-{backend}'
    return evidence, [str(probe), str(model), str(LAYER), str(expert), '0', weight, str(raw)]


def cpu_ref_suite(directory, backend):
    evidence, cmd = replay(directory, backend)
    cmd[0] = str(HERE / 'out' / ('parity-hip-ref' if backend == 'hip' else 'parity-cpu-ref-model'))
    return evidence, [*cmd, '--suite']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True, dest='action')
    setup = sub.add_parser('prepare', help='copy production settings into an existing private run directory')
    setup.add_argument('directory', type=Path)
    setup.add_argument('--config', type=Path, default=PRODUCTION)
    go = sub.add_parser('replay', help='print rank-0 evidence and a replay command; do not run it by default')
    ref = sub.add_parser('cpu-ref', help='verify the CPU-reference prototype on both fixtures, then time HIP leaves')
    for cmd in (go, ref):
        cmd.add_argument('directory', type=Path)
        cmd.add_argument('--backend', choices=['cpu', 'hip'], default='hip')
        cmd.add_argument('--guard-output', type=Path, help='fresh HIP guard folder (default under DIRECTORY)')
        cmd.add_argument('--execute', action='store_true', help='execute the printed command; HIP always uses the guard')
    a = p.parse_args()
    os.umask(0o077)
    try:
        directory = a.directory.resolve(strict=True)
        if a.action == 'prepare':
            print(prepare(directory, a.config))
            return 0
        is_ref = a.action == 'cpu-ref'
        evidence, command = (cpu_ref_suite if is_ref else replay)(directory, a.backend)
        if a.backend == 'hip':
            out = a.guard_output.resolve() if a.guard_output else directory / ('cpu-ref-guard' if is_ref else 'replay-guard')
            command = [sys.executable, str(ROOT / 'rdna2-rnd/scripts/run_guarded.py'), '--warn-amdgpu-faults',
                       '--output', str(out), '--', 'timeout', '120' if is_ref else '60', *command]
        elif a.guard_output:
            p.error('--guard-output is only valid for --backend hip')
        print(json.dumps(evidence), file=sys.stderr)
        print(shlex.join(command), flush=True)
        return subprocess.run(command, cwd=ROOT).returncode if a.execute else 0
    except (ValueError, OSError, KeyError, IndexError) as exc:
        p.exit(1, f'{p.prog}: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
