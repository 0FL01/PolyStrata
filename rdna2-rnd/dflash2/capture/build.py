#!/usr/bin/env python3
"""Relink only Inkling2 with the DFT1 recording hook, using a Release gfx1031 HIP build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace(text, marker, replacement):
    if text.count(marker) != 1:
        raise ValueError(f'expected exactly one Inkling2 hook: {marker!r}')
    return text.replace(marker, replacement)


def instrument(text):
    # Keep the paired download on the exact arrays consumed by mix_taps(), not a second norm pass.
    ds_inputs = 'dspark::Taps t{{ds_taps_, ds_taps_ + plane, ds_taps_ + 2 * plane, hn_ + n_embd_}, {}};'
    if text.count(ds_inputs) != 1:
        raise ValueError('DSpark input tap geometry changed; review the paired hook')
    text = '#include "capture.hpp"\n' + text
    text = replace(text, 'Inkling2::~Inkling2() {', '''Inkling2::~Inkling2() {
    if (dft1::tape.active()) dft1::tape.finish(layers_, stats_.swapped_experts);''')
    old = '    if (!draft_) return;\n    ds_taps_ = fbuf(3 * width);'
    text = replace(text, old, '''    if (!draft_ && dft1::enabled()) {
        ds_taps_ = fbuf(3 * width);
        dft1::norm = fbuf(width);
    }
''' + old)
    text = replace(text, '        if (draft_)   // the drafter\'s taps: the residual entering visits 5, 23 and 41 (drafter_kept())',
                   '        if (draft_ || dft1::enabled())   // enter visits 5, 23, 41')
    checks = '''
    if (windowed_ || block_pass_ || opt_.expert_swaps || !freeze_experts_)
        throw std::logic_error("DFT1 requires frozen W1 target execution with zero swaps");
    dft1::tape.write(pos_, n, tok_, serial_decode_, ds_taps_, NORM, rows_, layers_, st_);'''
    marker = '        layer(physical(visit_), n);\n    }\n}'
    text = replace(text, marker, marker[:-1] + '''    if (!draft_ && dft1::enabled()) {
        output_rows(0, n, dft1::norm);''' + checks.replace('NORM', 'dft1::norm').replace('\n    ', '\n        ') + '\n    }\n}')
    marker = 'void Inkling2::drafter_kept(int n) {\n    if (n <= 0) return;'
    return replace(text, marker, marker + '\n    if (dft1::enabled()) {' +
                   checks.replace('NORM', 'hn_ + n_embd_').replace('\n    ', '\n        ') + '\n    }')


def build(folder, out):
    out.mkdir(parents=True, exist_ok=False)
    src = ROOT / 'src/poly/inkling2.cu'
    lines = subprocess.check_output(['ninja', '-t', 'commands', 'strata-poly'], cwd=folder, text=True).splitlines()
    cmd, = (shlex.split(line) for line in lines if line.endswith(' -c ' + str(src)))
    link, = (shlex.split(line)[2:-2] for line in lines if ' --hip-link ' in line and ' -o strata-poly ' in line)
    if '--offload-arch=gfx1031' not in link or '-DNDEBUG' not in cmd:
        raise ValueError('need an existing Release gfx1031 HIP build')
    source = src.read_text()
    generated = out / src.name
    generated.write_text(instrument(source))
    old_obj = cmd[cmd.index('-o') + 1]
    obj = str(out / 'inkling2.cu.o')
    cmd[cmd.index('-o') + 1] = obj
    cmd[cmd.index('-MF') + 1] = str(out / 'inkling2.cu.d')
    cmd[-1] = str(generated)
    cmd.extend(['-I' + str(src.parent), '-I' + str(HERE)])
    link[link.index(old_obj)] = obj
    link[link.index('-o') + 1] = str(out / 'strata-poly-dft1')
    for i, word in enumerate(link):
        if word.startswith('--dependency-file='):
            if link[i - 1] != '-Xlinker':
                raise ValueError('unexpected linker dependency flag')
            del link[i - 1:i + 1]
            break
    with (out / 'build.log').open('x') as log:
        for command in (cmd, link):
            subprocess.run(command, cwd=folder, check=True, stdout=log, stderr=subprocess.STDOUT)
    (out / 'build.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'compile': cmd, 'link': link,
    }, indent=2) + '\n')
    print(out / 'strata-poly-dft1', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build-poly-hip')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    os.umask(0o077)
    build(a.build.resolve(strict=True), a.out.resolve())
