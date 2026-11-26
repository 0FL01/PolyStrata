#!/usr/bin/env python3
"""Free-running short billing agent; private, guarded, one fresh --tokens process per turn."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path.home() / '.cache/opencode-tmp/opencode'
R = SCRATCH / 'polystrata-runs/dflash2-recon'
ARCHIVE = SCRATCH / 'global-kv-freeagent-confirm'
CONFIG = Path.home() / '.config/strata/inkling2-mini/strata-inkling-2-mini.json'
TOKENIZER = CONFIG.parent / 'tokenizer'
PROFILE = SCRATCH / 'hybrid-20261125/spec-audit-mixed/serial/profile.seed'
DSPARK = Path.home() / '.lmstudio/models/unsloth/Inkling-2-Mini/dspark-Inkling-2-Mini-Q8_0-F16-head.gguf'
CAP, CONTEXT, RESERVE, EOS = 768, 2048, 2048, 200006
SEEDS = (42, 79)
TOOLS = [{'type': 'function', 'function': {'name': 'read_case',
          'description': 'Read a live billing specification or tariff by case ID.',
          'parameters': {'type': 'object', 'properties': {'case_id': {'type': 'string'}}, 'required': ['case_id']}}}]


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def replace_opt(args, key, value):
    args[args.index(key) + 1] = str(value)


def p90(vals):
    if not vals:
        return None
    p = (len(vals) - 1) * 0.9
    vals = sorted(vals)
    i = math.floor(p)
    return vals[i] + (p - i) * (vals[math.ceil(p)] - vals[i])


def base_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(('DF2_', 'STRATA_', 'RECON_'))}


def plan(a):
    cfg = json.loads(CONFIG.read_text())
    args = list(cfg['args'])
    args.remove('--disk-sessions')
    for k, v in {'--max-context': CONTEXT, '--prompt-batch': 256, '--vram-reserve-mib': RESERVE}.items():
        replace_opt(args, k, v)
    args += ['--prompt-cache', '0', '--lookup', '0', '--eos-ids', str(EOS), '--top-probs', '1',
             '--max-new', str(CAP), '--temperature', '1', '--top-k', '0', '--top-p', '1', '--min-p', '0',
             '--seed', str(a.seed)]
    env = dict(cfg['env'], LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']), LC_ALL='C')
    if a.cap is not None:
        replace_opt(args, '--expert-swaps', 0)
        args += ['--cpu-share', '0', '--stage-ahead', '0']
        env.update(STRATA_INKLING_EXPERT_KV_MIB=str(a.cap), STRATA_FREEZE_EXPERTS='1', STRATA_INKLING_PREFETCH='0')
    exe = Path(cfg['exe'])
    payload = {}
    if a.mode == 'dspark':
        args += ['--draft-model', str(DSPARK), '--window', '8', '--draft-max', str(a.k)]
        env['STRATA_INKLING_DSPARK_REUSE_V'] = '1'
        payload.update(weights=str(DSPARK), weights_sha256=sha(DSPARK))
    elif a.mode == 'live':
        if not a.weights:
            raise ValueError('live requires --weights')
        exe = a.binary or R / 'live-bin/strata-poly-dflash2-live'
        lib = a.library or R / 'libstrata-dflash2.so'
        args += ['--window', '8']
        env.update(DF2_LIBRARY=str(lib.resolve()), DF2_WEIGHTS=str(a.weights.resolve()), DF2_K=str(a.k),
                   DF2_SERIAL_VERIFY=str(int(a.serial_verify)))
        payload.update(library_sha256=sha(lib), weights_sha256=sha(a.weights))
    else:
        args += ['--no-draft', '--window', '1']
        if a.mode == 'taps':
            exe = a.binary or R / 'probe-bin/strata-poly-probe'
            env['RECON_DUMP_FROM'] = '0'
        elif a.binary:
            raise ValueError('production must use the installed executable')
    return dict(mode=a.mode, args=args, env=env, installed_exe=cfg['exe'], exe=str(exe.resolve()),
                exe_sha256=sha(exe), seed=a.seed, limit=CAP, turns=3, reasoning_effort='low',
                fixed_expert_kv_mib=a.cap, k=a.k, serial_verify=a.serial_verify,
                profile=str(PROFILE), profile_sha256=sha(PROFILE), tokenizer=str(TOKENIZER),
                **payload)


def run_once(d, p, ids):
    d.mkdir()
    shutil.copyfile(p['profile'], d / 'profile')
    args = list(p['args'])
    replace_opt(args, '--expert-profile', d / 'profile')
    assert len(ids) + p['limit'] <= CONTEXT
    cmd = [p['exe'], *args, '--tokens', ','.join(map(str, ids))]
    env = dict(base_env(), **p['env'])
    if p['mode'] == 'taps':
        env['RECON_DUMP_TAPS'] = str(d / 'trace.dft1')
    dump(d / 'launch.json', dict(argv=cmd, env={k: env[k] for k in p['env']}, prompt_ids=ids))
    markers, slots, loaded = [], None, None
    start = time.monotonic_ns()
    with (d / 'stderr.log').open('w') as err, (d / 'stdout.log').open('w+') as stdout:
        with subprocess.Popen(cmd, env=env, cwd=d, stdout=stdout, stderr=subprocess.PIPE, text=True) as proc:
            for line in proc.stderr:
                now = (time.monotonic_ns() - start) / 1e9
                if line.startswith('top '):
                    markers.append(now)
                if m := re.match(r'^strata-poly: loaded in ([.\d]+) s; (\d+) of \d+ routed experts in VRAM', line):
                    loaded, slots = now, int(m[2])
                err.write(line)
            rc = proc.wait()
        end = (time.monotonic_ns() - start) / 1e9
        stdout.seek(0)
        seq = stdout.read().strip()
    if rc:
        raise RuntimeError(f'{p["exe"]} exited {rc}, see {d / "stderr.log"}')
    stderr = (d / 'stderr.log').read_text()
    assert re.fullmatch(r'\d+(?:,\d+)*', seq), f'invalid IDs: {d}'
    emitted = list(map(int, seq.split(',')))
    m = re.search(r'prompt (\d+) tokens ([.\d]+) tok/s, generated (\d+) tokens ([.\d]+) tok/s \((\w+)\).*?(\d+) of (\d+) guessed tokens right', stderr)
    assert m is not None and int(m[1]) == len(ids) and int(m[3]) == len(emitted), d
    assert len(markers) == len(emitted) and loaded is not None and loaded < markers[0], d
    intervals = [(b - a) * 1000 for a, b in zip(markers, markers[1:])]
    row = dict(prompt_ids=ids, output_ids=emitted, finish=m[5], startup_s=loaded,
               ttft_s=markers[0]-loaded, cold_ttft_s=markers[0], step_p90_ms=p90(intervals), step_ms=intervals,
               process_wall_s=end, compute_s=markers[-1]-loaded, teardown_s=end-markers[-1], slots=slots,
               prompt_ids_per_s=float(m[2]), decode_ids_per_s=float(m[4]), offered=int(m[7]), accepted=int(m[6]))
    if layout := re.search(r'^DF2_TARGET_PLACEMENT slots=(\d+) layout=([0-9a-f]+) swaps=(\d+)$', stderr, re.M):
        assert int(layout[1]) == slots, d
        row.update(layout=layout[2], active_swap_slots=int(layout[3]))
    if p['mode'] == 'live':
        m = re.search(r'^DF2_TIMES (.*)$', stderr, re.M)
        assert m is not None, 'one-shot live hook did not run'
        ticks = list(map(float, m[1].split()))
        assert len(ticks) == 1 + len(emitted), d
        rounds = [list(map(float, s.split())) for s in re.findall(r'^DF2_ROUND (.*)$', stderr, re.M)]
        assert sum(r[3] for r in rounds) == len(emitted)-1
        row.update(live_exact_ttft_ms=ticks[0]+ticks[1], live_rounds=len(rounds),
                   offered=int(sum(r[1] for r in rounds)), accepted=int(sum(r[2] for r in rounds)))
    if p['mode'] == 'taps':
        f = d / 'trace.dft1'
        raw = f.read_bytes()
        width = 16 + 4 * 4 * 2304
        assert raw[:4] == b'DFT1' and (len(raw) - 4) % width == 0, d
        rows = [struct.unpack_from('<qii', raw, i) for i in range(4, len(raw), width)]
        expected = ids + emitted[:-1]
        assert [(pos, t) for pos, t, _ in rows] == list(enumerate(expected)), (d, len(rows), len(expected))
        assert all(gen == int(pos >= len(ids)) for pos, _, gen in rows), d
        row.update(tap_file=str(f), tap_sha256=sha(f), tap_rows=len(rows))
    return row


def classify(ids, ids_for):
    phase = 'framing'
    grouped = {p: [] for p in ('reasoning', 'answer', 'tool', 'framing')}
    for t in ids:
        if t in ids_for:
            phase = ids_for[t]
            grouped['framing'].append(t)
        else:
            grouped[phase].append(t)
    return grouped


def worker(out):
    sys.path[:0] = [str(ROOT), str(ARCHIVE)]
    from serve.frontend import (ChatTemplate, INKLING_END, INKLING_INVOKE, INKLING_MODEL, INKLING_TEXT,
                                INKLING_THINK, InklingOutputParser, openai_to_messages)
    from tools.strata_tokenizer import Tokenizer
    import workloads

    p = json.loads((out / 'plan.json').read_text())
    task = workloads.CASES['coding']
    keys = list(task['fixtures'])
    fixture_dir = ARCHIVE / 'fixtures'
    workloads.selfcheck()
    assert all(sha(fixture_dir / (k + '.json')) == p['fixture_sha256'][k] for k in keys)
    template = ChatTemplate(TOKENIZER / 'chat_template.jinja')
    tok = Tokenizer.from_dir(TOKENIZER)
    def special(txt):
        id_, = tok.encode(txt, parse_special=True)
        return id_
    invoke = special(INKLING_INVOKE)
    ids_for = {special(INKLING_THINK): 'reasoning', special(INKLING_TEXT): 'answer', invoke: 'tool',
               special(INKLING_MODEL): 'framing', special(INKLING_END): 'framing', EOS: 'framing'}
    msgs = [{'role': 'user', 'content': task['question']}]
    given = set()
    result = dict(seed=p['seed'], mode=p['mode'], turns=[], tools=[], grade={'pass': False, 'class': 'infrastructure'})
    task_at = time.monotonic()
    try:
        for i in range(1, 4):
            norm, tools, kw = openai_to_messages(dict(messages=msgs, tools=TOOLS, reasoning_effort='low'))
            rendered = template.render(norm, tools=tools, **kw)
            prompt_ids = tok.encode(rendered, parse_special=True)
            row = run_once(out / f'turn-{i}', p, prompt_ids)
            emitted = row['output_ids']
            result['turns'].append(row)
            raw = tok.decode(emitted[:-1] if emitted[-1] == EOS else emitted)
            parser = InklingOutputParser(tools)
            events = parser.feed(raw) + parser.finish()
            answer = ''.join(e.text for e in events if e.kind == 'content')
            reasoning = ''.join(e.text for e in events if e.kind == 'reasoning')
            calls = [e.call for e in events if e.kind == 'tool_call']
            row.update(turn=i, messages=msgs.copy(), raw=raw, answer=answer, reasoning=reasoning,
                       calls=[dict(name=c.name, arguments=c.arguments) for c in calls], ids_by_phase=classify(emitted, ids_for))
            if row['finish'] != 'stop' or emitted[-1] != EOS:
                result['grade'] = {'pass': False, 'class': 'length', 'reason': 'no natural EOS'}
                break
            if i == 3:
                result['grade'] = (workloads.grade('coding', answer) if not calls and invoke not in emitted
                                   else {'pass': False, 'class': 'tool-format', 'reason': 'unexpected final tool call'})
                break
            if len(calls) != 1 or emitted.count(invoke) != 1:
                result['grade'] = {'pass': False, 'class': 'tool-format', 'reason': 'expected one complete call'}
                break
            call, = calls
            before = time.monotonic()
            key, value = workloads.call_result('coding', given, call, fixture_dir)
            result['tools'].append(dict(turn=i, case_id=key, name=call.name, arguments=call.arguments, result=value,
                                        dispatch_s=time.monotonic()-before))
            if key != keys[i-1]:
                result['grade'] = {'pass': False, 'class': 'tool-order', 'reason': 'unexpected or disallowed file read'}
                break
            given.add(key)
            call_id = f'local_billing_{i}'
            msgs.extend([{'role': 'assistant', 'content': answer or None, 'reasoning_content': reasoning,
                          'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': call.name,
                                           'arguments': json.dumps(call.arguments, ensure_ascii=False)}}]},
                         {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(value)}])
            dump(out / 'trajectory.json', result)
    finally:
        result.update(task_s=time.monotonic()-task_at,
                      startup_s=sum(t['startup_s'] for t in result['turns']),
                      compute_s=sum(t['compute_s'] for t in result['turns']),
                      step_p90_ms=p90([x for t in result['turns'] for x in t['step_ms']]))
        dump(out / 'trajectory.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('tools', 'turns')}), flush=True)


def main():
    os.umask(0o077)
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('production', 'dspark', 'taps', 'live'))
    ap.add_argument('label')
    ap.add_argument('--seed', choices=SEEDS, type=int, default=42)
    ap.add_argument('--cap', type=int, help='use this combined target expert/KV cap and freeze residency within each turn')
    ap.add_argument('--binary', type=Path)
    ap.add_argument('--library', type=Path)
    ap.add_argument('--weights', type=Path)
    ap.add_argument('--k', type=int, choices=range(8), default=7)
    ap.add_argument('--serial-verify', action='store_true')
    a = ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]+', a.label):
        ap.error('label must be a single experiment name')
    p = plan(a)
    out = R / 'trajectory' / a.label
    out.mkdir(parents=True)
    p['fixture_sha256'] = {f.stem: sha(f) for f in (ARCHIVE / 'fixtures').glob('*.json')}
    p['workloads_sha256'] = sha(ARCHIVE / 'workloads.py')
    p['runner_sha256'] = sha(Path(__file__))
    p['template_sha256'] = sha(TOKENIZER / 'chat_template.jinja')
    dump(out / 'plan.json', p)
    with (R / 'gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        service = subprocess.check_output(['systemctl', '--user', 'show', 'strata.service', '-p', 'ActiveState', '--value'], text=True).strip()
        if service != 'inactive':
            raise RuntimeError(f'strata.service is {service}; refuse to launch')
        subprocess.run([sys.executable, str(ROOT / 'rdna2-rnd/scripts/run_guarded.py'), '--warn-amdgpu-faults',
                        '--output', str(out / 'guard'), '--', 'timeout', '--kill-after=15s', '900',
                        sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(out)],
                       cwd=ROOT, env=base_env(), check=True)


if __name__ == '__main__':
    main()
