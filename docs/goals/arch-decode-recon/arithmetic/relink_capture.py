#!/usr/bin/env python3
"""Link a CP-02 Release engine with one capture hook; do not modify or run the CP-02 build."""

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'capture'
PIN = 'c0dd0e52f3975dbed3972deda2c2c75010b330c2'


def capture(args, cwd):
    return subprocess.check_output(args, cwd=cwd)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_objects(link, build, original):
    # Dry-run Ninja always marks CMake's VerifyGlobs rule dirty. Read its
    # recorded dependency lists instead, without re-running CMake or Ninja.
    for arg in link:
        if not arg.endswith(('.o', '.a')):
            continue
        obj = build / arg
        mtime = obj.stat().st_mtime_ns
        if mtime > original.stat().st_mtime_ns:
            raise ValueError(f'{obj} is newer than the original CP-02 executable')
        if obj.suffix == '.o':
            info = capture(['ninja', '-t', 'deps', arg], build).decode().splitlines()
            if not info[0].endswith('(VALID)'):
                raise ValueError(f'invalid Ninja dependency record for {obj}')
            for dep in (build / x.strip() for x in info[1:] if x.strip()):
                if dep.stat().st_mtime_ns > mtime:
                    raise ValueError(f'{obj} is stale relative to {dep}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('worktree', type=Path, help='pinned worktree with exactly the CP-02 patch applied')
    p.add_argument('--build', type=Path, help='existing CP-02 gfx1031 Release Ninja build (default: WORKTREE/build-cp02)')
    a = p.parse_args()
    work = a.worktree.resolve(strict=True)
    build = a.build.resolve(strict=True) if a.build else (work / 'build-cp02').resolve(strict=True)
    cp02 = HERE.parent / 'trace' / 'cp02-instrumentation.patch'
    hook = HERE / 'capture-xn-xq.patch'
    if (capture(['git', 'rev-parse', 'HEAD'], work).decode().strip() != PIN or
            capture(['git', 'diff', '--binary', PIN, '--'], work) != cp02.read_bytes()):
        p.error('the CP-02 worktree must be exactly c0dd0e52 plus the published instrumentation diff')
    cache = (build / 'CMakeCache.txt').read_text().splitlines()
    if ('CMAKE_BUILD_TYPE:STRING=Release' not in cache or
            f'CMAKE_HOME_DIRECTORY:INTERNAL={work}' not in cache or
            not any(v.startswith('CMAKE_HIP_ARCHITECTURES:') and v.endswith('=gfx1031') for v in cache)):
        p.error('need a Release gfx1031 build configured against the given CP-02 worktree')
    subprocess.run(['git', 'apply', '--check', str(hook)], cwd=work, check=True)
    source = work / 'src/poly/inkling2.cu'
    original = build / 'strata-poly'
    original_sha = sha(original)
    lines = capture(['ninja', '-t', 'commands', 'strata-poly'], build).decode().splitlines()
    cmds = [shlex.split(line) for line in lines if line.endswith(f' -c {source}') or ' -o strata-poly ' in line]
    cc, link = cmds
    # CMake can emit multiple ccache launchers and writes a dependency file next to its object.
    while Path(cc[0]).name == 'ccache':
        cc.pop(0)
    cc = cc[:cc.index('-MD')]
    assert '--offload-arch=gfx1031' in cc
    assert link[:2] == [':', '&&'] and link[-2:] == ['&&', ':']
    link = link[2:-2]
    try:
        check_objects(link, build, original)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    # Passing the original dependency-file flag would overwrite the CP-02 link.d.
    dep = next((i for i, arg in enumerate(link) if arg.startswith('--dependency-file=')), None)
    if dep is not None:
        assert link[dep - 1] == '-Xlinker'
        del link[dep - 1:dep + 1]
    OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    combined, obj, engine = OUT / 'src/poly/inkling2.cu', OUT / 'inkling2.cu.o', OUT / 'strata-poly'
    combined.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, combined)
    with (OUT / 'build.log').open('w') as log:
        def run(args, cwd, env=None):
            print(shlex.join(args), file=log, flush=True)
            subprocess.run(args, cwd=cwd, check=True, stdout=log, stderr=log, env=env)

        # Stop Git at the scratch directory's parent so no checkout is discovered.
        run(['git', 'apply', '--no-index', str(hook)], OUT, dict(os.environ, GIT_CEILING_DIRECTORIES=str(OUT.parent)))
        run(cc + [f'-I{source.parent}', '-x', 'hip', '-c', str(combined), '-o', str(obj)], build)
        j = next(i for i, arg in enumerate(link) if arg.endswith('/src/poly/inkling2.cu.o'))
        link[j] = str(obj)
        new = engine.with_suffix('.new')
        link[link.index('-o') + 1] = str(new)
        run(link, build)
    if sha(original) != original_sha:
        raise RuntimeError('original CP-02 engine changed during relink')
    new.replace(engine)
    print(f'{engine}\n  SHA-256 {sha(engine)}\n  CP-02 original unchanged: {original_sha}')
    print(f'  CP-02 diff: {sha(cp02)}\n  Capture diff: {sha(hook)}\n  Combined source: {sha(combined)}')


if __name__ == '__main__':
    main()
