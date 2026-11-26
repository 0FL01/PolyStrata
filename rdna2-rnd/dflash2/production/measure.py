#!/usr/bin/env python3
"""Measure a native worker via the production wire protocol, with private state/profile outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def probe(a):
    cfg = json.loads(a.config.read_text())
    args = list(cfg['args'])
    args[args.index('--expert-profile') + 1] = str(a.out / 'profile')
    shutil.copyfile(a.profile, a.out / 'profile')
    if a.batch is not None:
        args[args.index('--prompt-batch') + 1] = str(a.batch)
    args[args.index('--expert-swaps') + 1] = str(a.swaps)
    args += ['--lookup', str(a.lookup), '--eos-ids', '200006', '--serve']
    if '--disk-sessions' not in args:
        args.append('--disk-sessions')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'DF2_', 'RECON_'))}
    env.update(cfg['env'], LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']))
    if a.cap:
        env['STRATA_INKLING_EXPERT_KV_MIB'] = str(a.cap)
    if a.stable_prefill:
        args += ['--cpu-share', '0', '--stage-ahead', '0']
        env['STRATA_INKLING_PREFETCH'] = '0'
    if a.cpu_reuse:
        env['STRATA_DFLASH2_CPU_REUSE'] = '1'
    if a.tape:
        env['DF2_VERIFY_TAPE'] = str(a.out / 'logits.f32')
    if a.budget:
        args += ['--dflash2-budget-ms', str(a.budget)]
    if a.weights:
        args += ['--draft-model' if a.dspark else '--dflash2-model', str(a.weights), '--draft-max', str(a.k)]
        if a.dspark:
            env['STRATA_INKLING_DSPARK_REUSE_V'] = '1'
    cmd = [str(a.binary), *args]
    if a.rocprof:
        cmd = [str(a.rocprof), '--kernel-trace', '--hip-runtime-trace', '--memory-copy-trace', '-f', 'csv',
               '-d', str(a.out / 'trace'), '-o', 'live', '--', *cmd]
    raw = a.prompt.read_text().strip()
    with (a.out / 'stderr.log').open('w') as err, (a.out / 'wire.log').open('w') as log, \
            subprocess.Popen(['/usr/bin/time', '-f', 'PEAK_RSS_KiB %M', *cmd], env=env, text=True, bufsize=1,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err) as p:
        def send(line):
            p.stdin.write(line + '\n')
            p.stdin.flush()

        def read_until(tag, on_line=lambda *_: None):
            for line in p.stdout:
                now = time.monotonic_ns()
                log.write(line)
                if line.startswith(('ERR ', 'SERR ', 'FATAL ')):
                    raise RuntimeError(line.strip())
                fields = line.split()
                if not fields:
                    continue
                on_line(fields, now)
                if fields[0] == tag:
                    log.flush()
                    return fields
            raise RuntimeError(f'worker exited before {tag}; see {a.out / "stderr.log"}')

        read_until('READY')
        prefix = ''
        if a.restore:
            if not a.prefix:
                raise ValueError('RESTORE needs the original prompt IDs (--prefix)')
            prefix = a.prefix.read_text().strip() + ','
            for _ in range(a.restore_count):
                send(f'RESTORE {a.restore}')
                restored = read_until('RESTORED')
                if int(restored[1]) != prefix.count(','):
                    raise RuntimeError('RESTORE position does not match prefix')
        result = []
        feed = prefix + raw
        for turn in range(1 + len(a.append)):
            start = time.monotonic_ns()
            steps = a.first_steps if turn == 0 and a.first_steps else a.steps
            send(f'GEN {steps} temperature={a.temperature} top_p=1 top_k=0 min_p=0 seed={a.seed} {feed}')
            ids, ms = [], []
            def observe(fields, now):
                if fields[0] == 'T':
                    ids.append(int(fields[1]))
                    ms.append((now - start) / 1e6)
                    if turn == 0 and len(ids) == a.cancel_after:
                        send('STOP')
            done = read_until('DONE', observe)
            assert int(done[1]) == len(ids) and len(ids) > 0
            gaps = sorted(y - x for x, y in zip(ms, ms[1:]))
            result.append(dict(ids=ids, delivery_ms=ms, ttft_ms=ms[0],
                               p90_ms=gaps[min(len(gaps) - 1, int(.9 * len(gaps)))] if gaps else None,
                               decode_tps=1000 * len(ids) / float(done[4]),
                               finish=done[5], accepted=int(done[6]), offered=int(done[7]),
                               reused=int(done[8]), gpu_routes=int(done[9]), all_routes=int(done[10])))
            if a.save:
                saved = a.out / f'turn-{turn + 1}.bin'
                send(f'SAVE {saved}')
                read_until('SAVED')
                if a.restore_saved and turn < len(a.append):
                    for _ in range(a.restore_count):
                        send(f'RESTORE {saved}')
                        if int(read_until('RESTORED')[1]) != feed.count(',') + len(ids):
                            raise RuntimeError('RESTORE did not return to the committed prefix')
            if turn < len(a.append):
                feed += ',' + ','.join(map(str, ids)) + ',' + a.append[turn].read_text().strip()
        p.stdin.close()
        if p.wait(timeout=30) != 0:
            raise RuntimeError('worker did not exit cleanly')
    err = (a.out / 'stderr.log').read_text()
    slots = re.search(r'; (\d+) of \d+ routed experts in VRAM', err)
    rss = re.search(r'PEAK_RSS_KiB (\d+)', err)
    rounds = [list(map(int, m)) for m in re.findall(r'DFlash2 rounds W1=(\d+) lookup=(\d+) draft=(\d+) placement_replays=(\d+)', err)]
    report = dict(turns=result, slots=int(slots[1]) if slots else None, binary_sha256=digest(a.binary),
                  seed_profile_sha256=digest(a.profile), saved_profile_sha256=digest(a.out / 'profile'),
                  k=a.k, temperature=a.temperature, seed=a.seed, swaps=a.swaps,
                  quant=a.weights.name if a.weights else None, cap_mib=a.cap, budget_ms=a.budget,
                  peak_rss_kib=int(rss[1]) if rss else None, rounds_w1_lookup_draft_replayed=rounds)
    (a.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({**report, 'turns': [{k: v for k, v in t.items() if k not in ('ids', 'delivery_ms')}
                                         for t in result]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'config', 'profile', 'prompt', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('weights', 'restore', 'prefix'):
        p.add_argument('--' + key, type=Path)
    p.add_argument('--append', type=Path, action='append', default=[])
    p.add_argument('--restore-count', type=int, default=1)
    p.add_argument('--cancel-after', type=int, help='send STOP after observing this many IDs in turn 1')
    p.add_argument('--save', action='store_true')
    p.add_argument('--restore-saved', action='store_true', help='restore our own target SAVE between turns')
    p.add_argument('--stable-prefill', action='store_true', help='fix prefill staging; does not freeze decode swaps')
    p.add_argument('--cpu-reuse', action='store_true')
    p.add_argument('--tape', action='store_true', help='for the private build_parity.py binary only')
    p.add_argument('--rocprof', type=Path, help='rocprofv3 executable; do not use traced timings for A/B')
    p.add_argument('--dspark', action='store_true')
    p.add_argument('--cap', type=int)
    p.add_argument('--batch', type=int)
    p.add_argument('--steps', type=int, default=96)
    p.add_argument('--first-steps', type=int, help='override max output only in the first turn')
    p.add_argument('--k', type=int, default=3, choices=range(8))
    p.add_argument('--budget', type=float)
    p.add_argument('--lookup', type=int, default=0)
    p.add_argument('--swaps', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    os.umask(0o077)
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    probe(a)


if __name__ == '__main__':
    main()
