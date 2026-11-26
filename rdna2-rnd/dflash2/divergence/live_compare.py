#!/usr/bin/env python3
"""Compare every live target.tape byte; report first affected region without dumping private data."""

import argparse
from array import array
import hashlib
import json
import math
import mmap
from pathlib import Path
import re


def load(leg, *, need_placement=True):
    result = json.loads((leg / 'result.json').read_text())
    text = (leg / 'guard/command.log').read_text()
    dims, = re.findall(r'^DF2_PARITY vocab=(\d+) outputs=(\d+) state=(\d+) row_bytes=(\d+) committed=(\d+)$', text, re.M)
    placement = re.search(r'^DF2_TARGET_PLACEMENT slots=(\d+) layout=([0-9a-f]+) swaps=(\d+)$', text, re.M)
    cover = re.findall(r'^strata-poly: Inkling cover rows=\d+ mapped_kv=\d+ charged_kv=\d+ arena=\d+ slots=(\d+)$',
                       text[:placement.start() if placement else len(text)], re.M)
    # A long cold prompt may map its final KV reserve in one step before any token;
    # measure.py records the loaded slots, while the placement tap sees the final layout.
    slots = int(cover[-1]) if cover and int(cover[-1]) > 0 else result['slots']
    rounds = [tuple(map(int, r)) for r in re.findall(r'^DF2_ROUND (\d+) (\d+) (\d+) (\d+) ', text, re.M)]
    if sum(fed for _, _, _, fed in rounds) != result['n'] - 1 or (placement and int(placement[1]) != slots):
        raise ValueError(f'missing rounds or inconsistent placement: {leg}')
    if need_placement and not placement:
        raise ValueError(f'missing placement: {leg}')
    return result, tuple(map(int, dims)), placement.groups() if placement else (str(slots), None, None), rounds


def first_diff(a, b, start, stop, step):
    for off in range(start, stop, step):
        end = min(off + step, stop)
        aa, bb = a[off:end], b[off:end]
        if aa != bb:
            return off + next(i for i, (x, y) in enumerate(zip(aa, bb)) if x != y)
    return None


def comparison(ref, test):
    a, dims, placement, _ = load(ref)
    b, tdims, tplacement, rounds = load(test)
    for key in ('domain', 'binary_sha256', 'cap_mib', 'profile_seed_sha256', 'weights'):
        if a[key] != b[key]:
            raise ValueError(f'{key} differs')
    for key in ('temperature', 'seed', 'library_sha256', 'config_sha256'):
        if key in a and key in b and a[key] != b[key]:
            raise ValueError(f'{key} differs')
    if a['k'] != 0 or b['k'] == 0 or b['serial_verify'] or placement != tplacement or placement[2] != '0':
        raise ValueError('requires W1 vs real live windows with identical zero-swap slots')
    vok, outputs, state, stride, committed = dims
    tvok, toutputs, tstate, tstride, tcommitted = tdims
    discarded = sum(1 + k - fed for _, k, _, fed in rounds)
    # This separate side effect is absent from the canonical target tape. Both one-shot
    # legs start with an independent copy of exactly the same prior usage profile.
    profile_delta = profile_total(test) - profile_total(ref)
    result = dict(reference=ref.name, leg=test.name, slots=int(placement[0]), layout=placement[1],
                  offered=b['offered'], accepted=b['accepted'],
                  window_rounds=sum(k > 0 for _, k, _, _ in rounds),
                  rollback_rounds=sum(fed < k + 1 for _, k, _, fed in rounds),
                  partial_accept_rounds=sum(0 < right and fed < k + 1 for _, k, right, fed in rounds),
                  discarded_rows=discarded, profile_count_delta=profile_delta,
                  widths=sorted(set(1 + k for _, k, _, _ in rounds)),
                  first_different_id=next((i for i, (x, y) in enumerate(zip(a['ids'], b['ids'])) if x != y),
                                          None if outputs == toutputs else min(outputs, toutputs)))
    with (ref / 'target.tape').open('rb') as fa, (test / 'target.tape').open('rb') as fb, \
         mmap.mmap(fa.fileno(), 0, access=mmap.ACCESS_READ) as ma, mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ) as mb:
        nbytes = 4 * vok * outputs + state + committed * stride
        if len(ma) != nbytes or len(mb) != 4 * tvok * toutputs + tstate + tcommitted * tstride:
            raise ValueError('tape lengths do not match headers')
        result.update(tape_bytes=[len(ma), len(mb)], tape_sha256=[hashlib.sha256(m).hexdigest() for m in (ma, mb)],
                      exact=False, logits_compared=min(outputs, toutputs))
        if vok != tvok:
            raise ValueError('logit dimensions differ')
        logits_end = 4 * vok * min(outputs, toutputs)
        first = first_diff(ma, mb, 0, logits_end, 4 * vok)
        result['first_different_logit'] = None if first is None else dict(output=first // (4 * vok), index=first // 4 % vok)
        if dims != tdims:
            result['different_dimensions'] = [dims, tdims]
            return result
        # Native Tideglass layout: 48 convolution visits, then 40 SWA ring snapshots,
        # then global KV in 64-token groups, visit order 5/11/.../47, K/Kscale/V/Vscale planes.
        conv_visit = 4 * 3 * (2 * 1024 + 2 * 2304)
        planes = [1024, 64, 1024, 64]
        kv_visit = sum(planes)
        conv_end = logits_end + 48 * conv_visit
        state_end = logits_end + state
        if stride != 8 * kv_visit or state_end - conv_end != 40 * 1023 * kv_visit:
            raise ValueError('not the expected dense Q8, no-draft Tideglass state layout')
        conv = first_diff(ma, mb, logits_end, conv_end, conv_visit)
        ring = first_diff(ma, mb, conv_end, state_end, 1023 * kv_visit)
        glob = first_diff(ma, mb, state_end, nbytes, 64 * stride)
        result['first_different_conv'] = None
        result['first_different_swa_kv'] = None
        result['first_different_global_kv'] = None
        if conv is not None:
            visit, off = divmod(conv - logits_end, conv_visit)
            result['first_different_conv'] = dict(visit=visit, word=off // 4)
        if ring is not None:
            visit, off = divmod(ring - conv_end, 1023 * kv_visit)
            result['first_different_swa_kv'] = dict(visit=visit + visit // 5,
                                                   **kv_location(off, planes, 1023, committed - 1023))
        if glob is not None:
            chunk, off = divmod(glob - state_end, 64 * stride)
            rows = min(64, committed - chunk * 64)
            visit, off = divmod(off, rows * kv_visit)
            result['first_different_global_kv'] = dict(visit=5 + 6 * visit,
                                                      **kv_location(off, planes, rows, chunk * 64))
        result['exact'] = (result['first_different_id'] is None and first is None and conv is None and ring is None and glob is None)
        return result


def profile_total(leg):
    return sum(int(line.split()[2]) for line in (leg / 'profile').read_text().splitlines())


def prefill(ref, test):
    a, dims, placement, _ = load(ref)
    b, tdims, tplacement, _ = load(test, need_placement=False)
    for key in ('domain', 'seed', 'temperature', 'profile_seed_sha256', 'config_sha256'):
        if a[key] != b[key]:
            raise ValueError(f'{key} differs')
    if a['k'] or b['k'] or dims[0] != tdims[0] or dims[4] - dims[1] != tdims[4] - tdims[1]:
        raise ValueError('the prefill comparison requires identical prompt dimensions and K0')
    if tplacement[0] != placement[0] or (tplacement[1] is not None and placement != tplacement):
        raise ValueError(f'different prefill placement: {placement} / {tplacement}')
    width = 4 * dims[0]
    comparison_rows = min(dims[1], tdims[1])
    prompt_rows = dims[4] - dims[1] + 1
    with (ref / 'target.tape').open('rb') as fa, (test / 'target.tape').open('rb') as fb:
        a0, b0 = fa.read(width), fb.read(width)
        same = a0 == b0
        av, bv = array('f'), array('f')
        av.frombytes(a0)
        bv.frombytes(b0)
        delta = [float(x) - float(y) for x, y in zip(av, bv)]
        if len(delta) != dims[0]:
            raise ValueError('short prefill logit row')
        common = same and fa.read(width * (comparison_rows - 1)) == fb.read(width * (comparison_rows - 1))
        with mmap.mmap(fa.fileno(), 0, access=mmap.ACCESS_READ) as ma, mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ) as mb:
            dense_equal, swa_equal, swa_rows = prefill_kv(ma, mb, dims, tdims, prompt_rows)
        return dict(reference=ref.name, leg=test.name, prompt_rows=prompt_rows,
                    slots=int(placement[0]), layout_match=None if tplacement[1] is None else True,
                    prefill_exact=same, compared_words=dims[0],
                    different_words=sum(a0[i:i+4] != b0[i:i+4] for i in range(0, width, 4)),
                    maximum_absolute=max(map(abs, delta)), rms=math.sqrt(sum(x*x for x in delta) / len(delta)),
                    compared_rows=comparison_rows, all_common_logit_rows_exact=common,
                    global_prefill_kv_exact=dense_equal, swa_prefill_kv_exact=swa_equal,
                    swa_prefill_rows_compared=swa_rows)


def prefill_kv(a, b, dims, tdims, prompt):
    planes = [1024, 64, 1024, 64]
    kv_visit = sum(planes)
    size = dims[3]
    base = [4 * n[0] * n[1] for n in (dims, tdims)]
    if size != 8 * kv_visit or dims[2:4] != tdims[2:4]:
        raise ValueError('not the expected dense Q8 Tideglass layout')
    dense_equal = True
    for pos in range(0, prompt, 64):
        at, bt = (off + n[2] + pos * size for off, n in zip(base, (dims, tdims)))
        arows, brows = (min(64, n[4] - pos) for n in (dims, tdims))
        keep = min(64, prompt - pos)
        for stride in planes * 8:
            dense_equal &= a[at:at+stride*keep] == b[bt:bt+stride*keep]
            at, bt = at + stride * arows, bt + stride * brows
    conv = dims[2] - 40 * 1023 * kv_visit
    at, bt = (off + conv for off in base)
    earliest = [n[4] - 1023 for n in (dims, tdims)]
    start = max(0, *earliest)
    swa_rows = max(0, prompt - start)
    swa_equal = True
    for stride in planes * 40:
        af, bf = at + stride * (start - earliest[0]), bt + stride * (start - earliest[1])
        swa_equal &= a[af:af+stride*swa_rows] == b[bf:bf+stride*swa_rows]
        at, bt = at + 1023 * stride, bt + 1023 * stride
    return dense_equal, swa_equal, swa_rows


def kv_location(off, strides, n, first):
    for name, stride in zip(('K', 'Kscale', 'V', 'Vscale'), strides):
        if off < stride * n:
            row, at = divmod(off, stride)
            return dict(plane=name, token=first + row, byte_in_row=at)
        off -= stride * n
    raise ValueError('outside KV region')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', action='store_true', help='compare K0 prompt logit rows across ELFs')
    p.add_argument('reference', type=Path)
    p.add_argument('test', nargs='+', type=Path)
    args = p.parse_args()
    results = [(prefill if args.prefill else comparison)(args.reference, t) for t in args.test]
    print(json.dumps(results, indent=2))
    raise SystemExit(not all((r['prefill_exact'] and r['global_prefill_kv_exact'] and r['swa_prefill_kv_exact'])
                             if args.prefill else r['exact'] and r['rollback_rounds'] > 0
                             for r in results))
