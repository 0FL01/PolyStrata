#!/usr/bin/env python3
"""Build one probe, with no HIP runtime calls even in --hip / --isa modes."""
import argparse
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'out'
PIN = 'c0dd0e52'


def run(args, cwd=ROOT):
    subprocess.run(args, cwd=cwd, check=True)


def commands(build, target):
    return subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', target], text=True).splitlines()


def compile_cmd(lines, source):
    return shlex.split(next(l for l in lines if l.endswith(' -c ' + str(ROOT / source))))


def compile_from(lines, source, replacement, out, build, extra=(), isa=False):
    old = compile_cmd(lines, source)
    launcher = 1 if Path(old[0]).name == 'ccache' else 0
    args = old[launcher:old.index('-MD')] + list(extra)
    if isa:
        args += ['--offload-device-only', '-S', '-x', 'hip']
    else:
        args += ['-c']
        if '-x' in old:
            args += ['-x', old[old.index('-x') + 1]]
    run(args + [str(replacement), '-o', str(out)], build)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build-poly-hip', help='existing Release gfx1031 Ninja build')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--hip', action='store_true', help='compile and link HIP probe; do not run it')
    mode.add_argument('--isa', action='store_true', help='emit production device assembly only')
    p.add_argument('--cpu-ref', action='store_true', help='build the separate CPU-reference prototype or its ISA')
    a = p.parse_args()
    build = a.build.resolve()
    base = commands(build, 'poly_expert_parity')
    probe, host, poly, mmvq = ('src/poly/expert_parity.cpp', 'src/poly/cpu_experts.cpp',
                               'src/poly/kernels.cu', 'src/kernels/cuda/native_mmvq.cu')
    run(['git', 'diff', '--exit-code', PIN, '--', host, 'src/poly/cpu_experts.hpp', poly, mmvq,
         'src/poly/inkling2.cu', 'src/poly/engine.cpp', 'include/strata/kernels/q8_1_finite.hpp',
         'CMakeLists.txt', 'third_party/llama.cpp'])
    OUT.mkdir(exist_ok=True)
    if a.isa:
        if a.cpu_ref:
            compile_from(base, poly, HERE / 'cpu_ref.cu', OUT / 'cpu_ref.s', build,
                         ['-DCP_HIP=1', '-ffp-contract=off'], isa=True)
        else:
            for f in (poly, mmvq):
                compile_from(base, f, ROOT / f, OUT / (Path(f).stem + '.s'), build, ['-w'], isa=True)
        return

    cpu = OUT / 'cpu_experts.o'
    compile_from(base, host, ROOT / host, cpu, build)
    ggml = ROOT / 'third_party/llama.cpp/ggml/include'
    libs = [str(build / 'ggml/src' / f) for f in ('libggml-cpu.a', 'libggml-base.a')]
    source = HERE / 'parity.cpp'
    extra = ['-DCP_CPU_REF=1'] if a.cpu_ref else []
    if not a.hip:
        run(['c++', '-O3', '-DNDEBUG', '-std=c++20', '-ffp-contract=off', '-Wall', '-Wextra', '-pthread',
             f'-I{ROOT / "src/poly"}', f'-I{ROOT / "include"}', f'-I{ggml}', *extra, str(source), str(cpu),
             *libs, '-ldl', '-o', str(OUT / ('parity-cpu-ref-model' if a.cpu_ref else 'parity-cpu'))])
        return

    name = 'parity-hip-ref' if a.cpu_ref else 'parity-hip'
    h = OUT / (name + '.o')
    # Recompile both relevant device sources at the pinned source contents; use the
    # configured build's support archives and exact link flags for the rest.
    dk, mk = OUT / 'kernels.o', OUT / 'native_mmvq.o'
    compile_from(base, probe, source, h, build, ['-DCP_HIP=1', '-ffp-contract=off', f'-I{ROOT / "src/poly"}', *extra])
    variant = []
    if a.cpu_ref:
        ref = OUT / 'cpu_ref.o'
        compile_from(base, poly, HERE / 'cpu_ref.cu', ref, build, ['-DCP_HIP=1', '-ffp-contract=off'])
        variant = [str(ref)]
    compile_from(base, poly, ROOT / poly, dk, build, ['-w'])
    compile_from(base, mmvq, ROOT / mmvq, mk, build, ['-w'])
    cmd = shlex.split(next(l for l in base if ' -o poly_expert_parity ' in l))[2:-2]
    where = next(i for i, x in enumerate(cmd) if x.endswith('/src/poly/expert_parity.cpp.o'))
    cmd[where:where + 3] = [str(h), str(cpu), str(dk), str(mk), *variant]
    cmd[cmd.index('-o') + 1] = str(OUT / name)
    runtime = next(Path(x).parent for x in cmd if 'libamdhip64.so.' in x)
    run(cmd + [f'-Wl,-rpath,{runtime}'], build)


if __name__ == '__main__':
    main()
