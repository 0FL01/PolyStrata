#!/usr/bin/env python3
"""Build the exact-attention experiment from a private snapshot of the native DFlash2 inputs."""
import argparse
import re
from pathlib import Path
import shutil
import subprocess


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one occurrence of {old!r}; check the native runtime/build layout')
    return text.replace(old, new)


def main():
    script = Path(__file__).resolve()
    repo = script.parents[4]
    draft = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new isolated output directory')
    parser.add_argument('--hip-build', type=Path, default=repo / 'build-poly-hip')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a DSO, including the original R/ DSO.
    staged_repo = out / 'source'
    staged = staged_repo / 'rdna2-rnd' / 'dflash2'
    staged.mkdir(parents=True)
    # Snapshot the three relative-include siblings used by the native runtime; leave all repo files alone.
    for folder, patterns in [('native', ('*.hip', '*.hpp', '*.cuh')), ('live', ('*.h',)),
                             ('codebooks', ('*.cuh',)), ('projection', ('*.cuh',))]:
        dest = staged / folder
        dest.mkdir()
        for pattern in patterns:
            for src in (draft / folder).glob(pattern):
                shutil.copyfile(src, dest / src.name)
    native = staged / 'native'
    (native / 'opt').mkdir()
    shutil.copyfile(script.with_name('attention_exact.cuh'), native / 'opt' / 'attention_exact.cuh')
    (staged_repo / 'include').symlink_to(repo / 'include', target_is_directory=True)
    shutil.copyfile(draft / 'native' / 'build.sh', native / 'build.sh')
    replay = (draft / 'native' / 'replay.cpp').read_text()
    # One host timestamp sample, shared between the two arms. No profiler/hidden dumps in timing runs.
    replay = replace(replay, '    double elapsed = 0,', '    double first_propose = 0;\n    double elapsed = 0,')
    replay = replace(replay, '            result.propose += ms_since(now);',
                     '            const double dt = ms_since(now);\n'
                     '            if (result.steps == 0) result.first_propose = dt;\n'
                     '            result.propose += dt;')
    replay = replace(replay, '<< ",\\n    \\"append_ms_per_step\\":" << r.append / count',
                     '<< ",\\n    \\"first_proposal_ms\\":" << r.first_propose\n'
                     '        << ",\\n    \\"subsequent_proposal_ms\\":"'
                     ' << (r.propose - r.first_propose) / std::max(r.steps - 1, 1)\n'
                     '        << ",\\n    \\"append_ms_per_step\\":" << r.append / count')
    (native / 'replay.cpp').write_text(replay)
    build = ['bash', str(native / 'build.sh')]
    runtime = (native / 'runtime.hip').read_text()
    subprocess.run([*build, str(out / 'stock'), str(args.hip_build.resolve())], check=True)
    (native / 'runtime.stock.hip').write_text(runtime)
    runtime = replace(runtime, '#include "kernels.cuh"', '#include "kernels.cuh"\n#include "opt/attention_exact.cuh"')
    runtime = replace(runtime, '    DeviceBuffer<float> concat,',
                      '    DeviceBuffer<float> exact_scores;\n    DeviceBuffer<float> concat,')
    runtime = replace(runtime, '        gates.allocate(proposals * rank);', '        gates.allocate(proposals * rank);\n'
                      '        exact_scores.allocate(block * q_heads * exact_score_stride);')
    pattern = r'attend<<<block \* q_heads, head_dim, 0, stream>>>\(q_rot.get\(\), l.k_ring.get\(\), l.v_ring.get\(\),\s+k_rot.get\(\), v.get\(\), attended.get\(\), int\(pos\)\);'
    runtime, count = re.subn(pattern, 'attend_exact(q_rot.get(), l.k_ring.get(), l.v_ring.get(), k_rot.get(), v.get(),\n'
                            '                             exact_scores.get(), attended.get(), int(pos), stream);', runtime)
    if count != 1:
        raise ValueError('stock attention call changed; check the overlay')
    (native / 'runtime.hip').write_text(runtime)
    subprocess.run([*build, str(out / 'exact'), str(args.hip_build.resolve())], check=True)


if __name__ == '__main__':
    main()
