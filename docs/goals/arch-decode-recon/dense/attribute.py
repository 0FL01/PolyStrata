#!/usr/bin/env python3
"""Attribute Tideglass serial dense dispatches from an existing roc2k capture (CPU only)."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent / 'profiler'))
from analyze import decode_windows, read_events
from tools.inkling_weights.transplant import HEADER_BYTES, parse_header


def physical(visit):
    return visit if visit < 6 else 6 + (visit - 6) % 12 if visit < 42 else visit - 24


def blocks(e):
    return math.prod(e.grid) // math.prod(e.workgroup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    config = json.loads((run / 'config.json').read_text())
    steps = json.loads((run / 'metrics.json').read_text())['decode_steps']
    model = Path(config['cmd'][config['cmd'].index('--model') + 1])
    with model.open('rb') as f:
        header = parse_header(f.read(HEADER_BYTES), model.stat().st_size)
    names = ('q', 'gate', 'rel', 'k', 'v', 'output')
    layers = [{p: header.tensors[f'blk.{i}.attn_{p}.weight'] for p in names} for i in range(24)]
    for p in names:
        groups = defaultdict(list)
        for i, layer in enumerate(layers):
            t = layer[p]
            groups[t.type_name, t.shape].append(i)
        for (typ, shape), ids in groups.items():
            print(f'{p:>6} {typ:>4} {str(shape):>13}: physical {",".join(map(str, ids))}')

    trace = run / 'trace'
    kernels = read_events(trace / 'decode_kernel_trace.csv', 'kernel')
    copies = read_events(trace / 'decode_memory_copy_trace.csv', 'copy')
    windows, _, method = decode_windows(kernels, copies, steps)
    assert method == 'logits_copy_end'
    acc = defaultdict(list)
    dispatches = 0

    def record(name, e):
        acc[name].append(e.end - e.start)

    def check(e, symbol, nblocks, ty=None):
        assert e.symbol == symbol and blocks(e) == nblocks, (e, symbol, nblocks)
        assert ty is None or ty in e.name, e.name

    q8_bytes = 0
    output_bytes = 0
    for start, end in windows:
        ks = [e for e in kernels if start <= e.start < e.end <= end]
        assert len({e.stream for e in ks}) == 1
        dispatches += len(ks)
        segs = [i for i, e in enumerate(ks) if e.symbol == 'native_mmvq_segments_kernel']
        assert len(segs) == 48 and not any('inkling_q4_0_segments' in e.symbol for e in ks)
        for visit, i in enumerate(segs):
            weights = layers[physical(visit)]
            assert weights['output'].type_name == 'Q8_0'
            proj = [weights[p] for p in names[:-1]]
            q8 = [t for t in proj if t.type_name == 'Q8_0']
            mixed = weights['rel'].type_name == 'Q4_0'
            assert (len(q8), mixed) in {(5, False), (4, True)}
            e = ks[i]
            rows = sum(t.shape[1] for t in q8)
            check(e, 'native_mmvq_segments_kernel', rows, '<8>')
            assert e.workgroup == (32, 4, 1)
            q8_bytes += sum(t.nbytes for t in q8)
            record(f'Q8_0 segments ({rows} rows)', e)
            streams = 3 if visit % 6 == 5 or 18 <= visit < 42 else 1
            check(ks[i - 1], 'native_rms_norm_q8_1_rows_kernel', streams)
            record(f'RMSNorm + Q8_1 ({streams} streams)', ks[i - 1])

            i += 1
            if mixed:
                rel = ks[i]
                check(rel, 'native_small_mmvq_kernel', weights['rel'].shape[1], 'Q40Block, 4, false')
                assert ks[i - 1].end <= rel.start <= rel.end <= ks[i + 1].start
                record('Q4_0 rel singleton', rel)
                acc['Q8 end -> sink start (Q4 + both gaps)'].append(ks[i + 1].start - ks[i - 1].end)
                i += 1
            else:
                acc['Q8 end -> sink start (no Q4)'].append(ks[i].start - e.end)
            check(ks[i], 'gemv_rows_kernel', 24)
            record('F32 sink', ks[i])
            if visit % 6 == 5:
                for e in ks[i + 1:i + 3]:
                    check(e, 'gemv_rows_kernel', 32)
                    record('F32 K/V taps', e)
                i += 2
            assert [e.symbol for e in ks[i + 1:i + 4]] == [
                'heads_kv_kernel', 'decode_q8_kernel', 'decode_combine_kernel']
            o = ks[i + 4]
            check(o, 'native_small_mmvq_kernel', weights['output'].shape[1], 'Q80Block, 8, false')
            output_bytes += weights['output'].nbytes
            record('Q8_0 output', o)

        quants = [i for i, e in enumerate(ks) if e.symbol == 'native_quantize_q8_1_kernel']
        assert len(quants) == 1
        i, = quants
        assert ks[i - 1].symbol == 'rms_norm_kernel' and ks[i + 1].symbol == 'native_q6_k_warps_kernel'
        record('Separate Q8_1 quant (LM head only)', ks[i])

    wall = sum(b - a for a, b in windows) / steps / 1e6
    print(f'\n{steps} steps, {dispatches:,} complete dispatches, {wall:.6f} ms/token wall ({method})')
    for name, ns in sorted(acc.items()):
        print(f'{name:>42}  {len(ns):5,d}  {len(ns)/steps:5.1f}/token'
              f'  {sum(ns)/steps/1e6:.6f} ms/token  p50 {statistics.median(ns)/1e3:.3f} us')
    print(f'Q8_0 logical weight payload: {q8_bytes//steps:,} bytes/token in project, '
          f'{output_bytes//steps:,} bytes/token in output')


if __name__ == '__main__':
    main()
