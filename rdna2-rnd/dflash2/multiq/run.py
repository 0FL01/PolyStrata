#!/usr/bin/env python3
"""Build the isolated gfx1031 test with Release flags and take the shared GPU lock."""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = Path.home() / '.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon'
TARGET = 'poly_inkling_attend_test'


def build(directory, out):
    commands = [shlex.split(c) for c in subprocess.check_output(
        ['ninja', '-t', 'commands', TARGET], cwd=directory, text=True).splitlines()]
    compile_stock, = (cmd for cmd in commands if cmd[-1].endswith('/inkling_attend.cu'))
    cmd = compile_stock.copy()
    old = cmd[cmd.index('-o') + 1]
    for opt in ('-MT', '-MF'):
        i = cmd.index(opt)
        del cmd[i:i + 2]
    cmd.remove('-MD')
    if '--offload-arch=gfx1031' not in cmd or '-O3' not in cmd:
        raise ValueError('expected an existing gfx1031 Release build')
    cmd[-1] = str(HERE / 'bench.cu')
    cmd[cmd.index('-o') + 1] = str(out / 'bench.o')
    link = commands[-1][2:-2]  # omit Ninja's shell wrappers : && / && :
    link.remove(next(x for x in link if x.endswith('/inkling_attend_test.cu.o')))
    link[link.index(old)] = str(out / 'bench.o')
    link[link.index('-o') + 1] = str(out / 'bench')
    asm = cmd.copy() + ['--cuda-device-only']
    asm[asm.index('-c')] = '-S'
    asm[asm.index('-o') + 1] = str(out / 'bench-gfx1031.s')
    (out / 'build-commands.txt').write_text('\n'.join(shlex.join(c) for c in [cmd, link, asm]) + '\n')
    with (out / 'build.log').open('w') as log:
        for c in [cmd, link, asm]:
            subprocess.run(c, cwd=directory, check=True, stdout=log, stderr=subprocess.STDOUT)
    files = [HERE / 'bench.cu', HERE / 'shared.cuh', ROOT / 'src/poly/inkling_attend.cu',
             *(directory / x for x in link if x.endswith('.cu.o')), out / 'bench', out / 'bench-gfx1031.s']
    (out / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f}\n' for f in files))
    sdk = Path(next(x.split('=', 1)[1] for x in cmd if x.startswith('--rocm-path=')))
    return sdk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=RUN, help='campaign root that owns gpu.lock')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--build', type=Path, default=ROOT / 'build-poly-hip')
    p.add_argument('--samples', type=int, default=21)
    p.add_argument('--lock-wait', type=int, default=480, help='seconds to wait for other campaign jobs')
    p.add_argument('--build-only', action='store_true')
    a = p.parse_args()
    os.umask(0o077)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sdk = build(a.build.resolve(), out)
    if a.build_only:
        return
    libs = [sdk / 'lib', *sorted(sdk.parent.glob('_rocm_sdk_libraries_gfx103*/lib'))]
    env = dict(os.environ, STRATA_INKLING_DECODE_CHUNKS='160',
               LD_LIBRARY_PATH=':'.join(str(p) for p in libs) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    print(f'Built {out / "bench"}; waiting for {a.run / "gpu.lock"}', flush=True)
    proc = subprocess.run(['flock', '-x', '-w', str(a.lock_wait), str(a.run / 'gpu.lock'),
                           sys.executable, str(ROOT / 'rdna2-rnd/scripts/run_guarded.py'),
                           '--warn-amdgpu-faults', '--output', str(out / 'guard'), '--',
                           'timeout', '--kill-after=10s', '120', str(out / 'bench'), str(a.samples)], env=env)
    log = out / 'guard/command.log'
    if log.is_file():
        print(log.read_text(), end='')
    proc.check_returncode()


if __name__ == '__main__':
    main()
