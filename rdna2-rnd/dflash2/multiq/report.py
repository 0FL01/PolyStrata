#!/usr/bin/env python3
"""Print run completeness, median timings and gfx1031 kernel resource counts."""
import argparse
from pathlib import Path
import re


ARMS = ('serial', 'batch8', 'group4', 'lds2g32', 'lds1g64')


def isa(path):
    text = path.read_text().split('.amdgpu_metadata', 1)[1]
    fields = dict(wavesize='wavefront_size', threads='max_flat_workgroup_size',
                  vgpr='vgpr_count', sgpr='sgpr_count', vgpr_spill='vgpr_spill_count',
                  sgpr_spill='sgpr_spill_count', private_B='private_segment_fixed_size', lds_B='group_segment_fixed_size')
    print('shader,' + ','.join(fields))
    for block in text.split('  - .args:'):
        props = dict(re.findall(r'^    \.(\w+):\s+(\S+)$', block, re.M))
        mangled = props.get('name', '')
        native = re.search(r'decode_q8_kernelILi([48])ELb0ELNS\w+ValuesE0E', mangled)
        sharing = re.search(r'shared_q8_kernelILi([24])ELi([12])ELi(8|16)E', mangled)
        if native:
            shader = f'stock_{native[1]}waves'
        elif sharing:
            shader = f'lds{sharing[2]}g{4 * int(sharing[3])}_w{sharing[1]}'
        elif 'decode_combine_kernel' in mangled:
            shader = 'combine'
        else:
            continue
        print(f'{shader},' + ','.join(props[key] for key in fields.values()))


def results(path):
    text = path.read_text()
    for line in text.splitlines():
        if line.startswith(('device=', 'hip_runtime=')):
            print(line)
    nchecks = 0
    counts = [0, 0, 0]
    times = {}
    for line in text.splitlines():
        if line.startswith('check '):
            comparisons = re.findall(r'(?:part_f32|out_f32|q8_bytes)=(\d+)/(\d+)', line)
            if len(comparisons) != 3 or any(int(diff) for diff, _ in comparisons):
                raise ValueError(line)
            for i, (_, n) in enumerate(comparisons):
                counts[i] += int(n)
            nchecks += 1
        if line.startswith('time '):
            data = dict(word.split('=') for word in line.split() if '=' in word)
            key = (int(data['pos']), int(data['width']), data['cache'], data['path'])
            times.setdefault(key, {})[data['arm']] = float(data['median'])
    print(f'bitwise: {nchecks} checks; {counts[0]} partial words, {counts[1]} output words, {counts[2]} Q8_1 bytes')
    if nchecks != 48 or (times and len(times) != 16):
        raise ValueError(f'incomplete run: {nchecks=} and {len(times)=}')
    print('pos,width,cache,path,' + ','.join(ARMS) + ',lds_best_vs_serial_pct,lds_best_vs_group4_pct')
    for (pos, width, cache, path), arms in sorted(times.items()):
        best = min(arms['lds2g32'], arms['lds1g64'])
        print(f'{pos},{width},{cache},{path},' + ','.join(f'{arms[a]:.5f}' for a in ARMS)
              + f',{100 * (best / arms["serial"] - 1):.2f},{100 * (best / arms["group4"] - 1):.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--isa-only', action='store_true')
    a = p.parse_args()
    isa(a.run / 'bench-gfx1031.s')
    if not a.isa_only:
        results(a.run / 'guard/command.log')


if __name__ == '__main__':
    main()
