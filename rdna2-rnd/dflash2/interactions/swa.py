#!/usr/bin/env python3
"""Run a matched-placement compact SWA pair on a separate real long prefix with live DFlash2."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
H = Path.home()
T = H / '.cache/opencode-tmp/opencode'
R = T / 'polystrata-runs/dflash2-recon'
CONFIG = H / '.config/strata/inkling2-mini/strata-inkling-2-mini.json'
WEIGHTS = H / '.lmstudio/models/unsloth/Inkling-2-Mini/DFlash2/Inkling-2-Mini-DFlash2-Q4_0.gguf'
SOURCE = T / 'global-kv-20261125/longagent/build-197/session.bin'
PROFILE = T / 'arch-campaign.XWM1yK/profile.seed'


def sha(p, n=None):
    with p.open('rb') as f:
        return (hashlib.file_digest(f, 'sha256') if n is None else hashlib.sha256(f.read(n))).hexdigest()


def put(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def tagged(text, label):
    return re.findall(r'^' + label + r' (.*)$', text, re.M)


def digest_sections(path, sizes):
    hashes = []
    with path.open('rb') as f:
        for n in sizes:
            d = hashlib.sha256()
            while n:
                b = f.read(min(n, 1 << 20))
                assert b
                d.update(b)
                n -= len(b)
            hashes.append(d.hexdigest())
        assert not f.read(1)
    return dict(zip(('logits', 'canonical_conv_swa', 'committed_global_kv'), hashes))


def parse(out, a, compact):
    g = out/'guard'
    assert json.loads((g/'stop-reasons.json').read_text()) == []
    text = (g/'command.log').read_text()
    start, end = [s.split() for s in tagged(text, 'IX_PLACEMENT')]
    assert start[0] == 'start' and end[0] == 'exit' and start[1:] == end[1:] and int(start[1]) > 0
    [counts] = tagged(text, 'IX_SWA')
    counters = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', counts)}
    if compact:
        assert counters['staged_visits'] > 0 and counters['stage'] > 0 and counters['borrow'] == 1
        saved = int(re.search(r'(\d+) bytes saved', text)[1])
    else:
        assert not counters['staged_visits'] and not counters['stage']
        saved = 0
    seq, = re.findall(r'^\d+(?:,\d+)+$', text, re.M)
    ids = list(map(int, seq.split(',')))
    assert len(ids) == a.steps, 'not a completed matched-length run'
    [tick] = tagged(text, 'DF2_TIMES')
    t = list(map(float, tick.split()))
    rounds = [x.split() for x in tagged(text, 'DF2_ROUND')]
    assert len(t) == len(ids) + 1 and sum(int(x[3]) for x in rounds) == len(ids) - 1
    assert int(rounds[0][0]) == a.rows
    assert sum(int(x[1]) for x in rounds) > 0 and counters['windows'] > 0
    assert re.search(rf'^prompt {a.rows} tokens ', text, re.M)
    info = dict(
        ids=ids, compact=compact, ring=counters, resident=int(start[1]), placement=start[2], saved_bytes=saved,
        ttft_ms=t[0]+t[1], rate=(len(ids)-1)*1000/(t[-1]-t[1]),
        p90_ms=statistics.quantiles([b-a for a,b in zip(t[1:], t[2:])], n=10, method='inclusive')[8],
        round_prefixes=[[int(y) for y in x[:4]] for x in rounds],
        offered=sum(int(x[1]) for x in rounds), accepted=sum(int(x[2]) for x in rounds),
        free_mib=min(json.loads(l)['free_mib'] for l in (g/'vram.jsonl').read_text().splitlines()),
        warnings_fresh=sum(w['phase'] != 'boot' for w in json.loads((g/'warnings.json').read_text())),
    )
    if a.audit:
        [meta] = tagged(text, 'DF2_PARITY')
        v, n, s, stride, committed = [int(x.split('=')[1]) for x in meta.split()]
        assert n == len(ids) and committed == a.rows + len(ids)-1
        info['canonical'] = digest_sections(out/'parity.bin', (v * n * 4, s, stride * committed))
        info['canonical_lengths'] = [v*n*4, s, stride*committed]
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path, help='new directory for this pair')
    p.add_argument('--binary', type=Path, default=R/'interactions/live-bin/strata-poly-dflash2-live')
    p.add_argument('--library', type=Path, default=R/'libstrata-dflash2.so')
    p.add_argument('--weights', type=Path, default=WEIGHTS)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--profile', type=Path, default=PROFILE)
    p.add_argument('--rows', type=int, default=16384)
    p.add_argument('--steps', type=int, default=33)
    p.add_argument('--arena', type=int, default=4096)
    p.add_argument('--audit', action='store_true', help='compare full logits and canonical state; do not price it')
    a = p.parse_args()
    os.umask(0o077)
    cfg = json.loads(CONFIG.read_text())
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with a.source.open('rb') as f:
        h = f.read(240)
        assert h[:8] == b'STRSESS\1'
        source_rows, = struct.unpack_from('<Q', h, 232)
        assert source_rows >= a.rows > 2048
        ids = struct.unpack(f'<{a.rows}i', f.read(4*a.rows))
        assert all(0 <= i < 200058 for i in ids)
    prompt = out/'prompt.csv'
    prompt.write_text(','.join(map(str, ids)) + '\n')
    context = a.rows + a.steps + 128
    put(out/'plan.json', dict(source=str(a.source), source_rows=source_rows, prompt_rows=a.rows, context=context,
        prompt_sha256=sha(prompt), steps=a.steps, arena_mib=a.arena, k=3, seed=42, audit=a.audit,
        profile_sha256=sha(a.profile), binary_sha256=sha(a.binary), draft_library_sha256=sha(a.library),
        weights=str(a.weights), weights_sha256=sha(a.weights)))

    results = []
    for compact in (0, 1):
        d = out/('compact' if compact else 'stock')
        d.mkdir()
        args = list(cfg['args'])
        args.remove('--disk-sessions')
        shutil.copyfile(a.profile, d/'profile')
        args[args.index('--expert-profile')+1] = str(d/'profile')
        args[args.index('--expert-swaps')+1] = '0'
        args[args.index('--max-context')+1] = str(context)
        # Price all target rows before freezing: a startup-full combined budget cannot later grow KV.
        args += ['--kv-whole','--cpu-share','0','--stage-ahead','0','--window','8','--no-draft','--lookup','0','--max-new',str(a.steps),
                 '--temperature','0','--seed','42','--top-k','0','--top-p','1','--eos-ids','200006']
        env = dict(cfg['env'], LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']), LC_ALL='C', IX_TOKEN_FILE=str(prompt),
                   STRATA_INKLING_K256='0', STRATA_INKLING_V4='0', STRATA_INKLING_SUFFIX_H2='0',
                   STRATA_INKLING_COMPACT_SWA=str(compact), STRATA_INKLING_FUND_EXPERTS='0',
                   STRATA_FREEZE_EXPERTS='1', IX_ARENA_MIB=str(a.arena), STRATA_INKLING_PREFETCH='0',
                   DF2_LIBRARY=str(a.library.resolve()), DF2_WEIGHTS=str(a.weights.resolve()), DF2_K='3', DF2_SERIAL_VERIFY='0')
        if a.audit:
            env['DF2_PARITY_TAPE'] = str(d/'parity.bin')
        command = [sys.executable, str(ROOT/'rdna2-rnd/scripts/run_guarded.py'), '--warn-amdgpu-faults', '--output',
                   str(d/'guard'), '--', 'timeout', '--kill-after=10s', '600', str(a.binary.resolve()), *args]
        put(d/'launch.json', dict(argv=command, env=env))
        clean = {k: v for k,v in os.environ.items() if not k.startswith(('DF2_', 'STRATA_', 'RECON_', 'IX_'))}
        with (R/'gpu.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run(command, env=clean | env, check=True)
        result = parse(d, a, compact)
        put(d/'result.json', result)
        results.append(result)
    x, y = results
    checks = {k: x[k] == y[k] for k in ('resident', 'placement', 'ids', 'round_prefixes')}
    if a.audit:
        checks.update({k: x['canonical'][k] == y['canonical'][k] for k in x['canonical']})
        checks['canonical_lengths'] = x['canonical_lengths'] == y['canonical_lengths']
    put(out/'compare.json', checks)
    print(checks, flush=True)
    if not all(checks.values()):
        raise ValueError('matched placement or exact SWA parity gate failed')


if __name__ == '__main__':
    main()
