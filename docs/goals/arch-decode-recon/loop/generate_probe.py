#!/usr/bin/env python3
"""Emit a standalone HIP A/D controller-input probe; optionally extract real F32 queries."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
PIN = 'c0dd0e52'


def part(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected exactly one {old!r} in pinned baseline')
    return text.replace(old, new)


def emit():
    text = subprocess.check_output(['git', 'show', f'{PIN}:src/poly/inkling2.cu'], cwd=ROOT, text=True)
    wave = part(text, 'template <bool Maximum> __device__ float wave_reduce',
                'template <bool Maximum = false> __device__ float block_reduce')
    baseline = part(text, '__device__ __forceinline__ void control_in_row', '// The third:')
    bank = part(baseline, '__device__ __forceinline__ void control_in_row', '__global__ void control_in_kernel')
    bank = change(bank, 'control_in_row', 'bank_only_row')
    bank = change(bank, 'const float* in, float* msg, float* pair', 'float* msg, float* msg_norm')
    bank = change(bank, 'b1[kPer], x[kPer], q[kPer]', 'b1[kPer], q[kPer]')
    bank = change(bank, '        x[k] = in[row + i];\n', '')
    bank = change(bank, '        sums[4] += x[k] * x[k];\n', '')
    bank = change(bank, 'reduce(0, 5);', 'reduce(0, 4);')
    bank = change(bank, 'const float sx = 1.0f / sqrtf(sums[4] / float(kRowWidth) + 1.e-6f), sm', 'const float sm')
    bank = change(bank, '    float* out = pair + 2 * row;\n', '')
    bank = change(bank, '        out[i] = x[k] * sx;\n        out[kRowWidth + i] = m[k] * sm;',
                  '        msg_norm[row + i] = m[k] * sm;')
    template = Path(__file__).with_name('controller_probe.cu.in').read_text()
    print(change(template, '// @PINNED_DEVICE_CODE@', f'// {PIN}:src/poly/inkling2.cu\n{wave}{baseline}{bank}'), end='')


def queries(path):
    # Import only for the CPU-side GGUF command; no torch/ROCm/model load.
    sys.path.insert(0, str(ROOT))
    from tools.gguf_reader import GGUFFile

    g = GGUFFile(path)
    if g.metadata['general.architecture'] != 'inkling2_loop':
        raise ValueError('query extractor handles fixed-loop GGUF only')
    tensors = {t.name: t for t in g.tensors}
    with path.open('rb') as f:
        for i in range(6, 18):
            t = tensors[f'blk.{i}.loop_msg_query.weight']
            if t.type_id != 0 or t.shape != [2304, 4]:
                raise ValueError(f'bad query tensor shape or type: {t}')
            f.seek(g.data_start + t.offset)
            data = f.read(2304 * 4 * 4)
            if len(data) != 2304 * 4 * 4:
                raise ValueError(f'short query tensor: {t}')
            sys.stdout.buffer.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries-from', type=Path, help='emit 48 F32 query rows to stdout instead of HIP source')
    args = parser.parse_args()
    if args.queries_from:
        queries(args.queries_from)
    else:
        emit()
