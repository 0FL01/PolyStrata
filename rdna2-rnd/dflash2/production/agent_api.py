#!/usr/bin/env python3
"""Guarded, free-running billing agent through a private production SSE server.

One fresh resident server per arm; three requests share its conversation cache.
Output and inputs contain private fixtures, model text, and a private API key.
Run one arm per unique --out under the Btrfs run directory. See --help.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('trajectory_run', ROOT / 'rdna2-rnd/dflash2/trajectory/run.py')
trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trajectory)
sys.path.insert(0, str(trajectory.ARCHIVE))
import workloads  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def set_arg(args, flag, value):
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args.extend([flag, str(value)])


def local_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def make_config(a, out):
    cfg = json.loads(a.config.read_text())
    cfg['args'] = list(cfg['args'])
    args = cfg['args']
    if a.expert_epoch is not None:
        cfg.setdefault('env', {})['STRATA_EXPERT_EPOCH'] = a.expert_epoch
    # Ambiguous inherited draft settings would invalidate comparisons.
    for flag in ('--window', '--lookup', '--draft-max', '--draft-model', '--dflash2-model',
                 '--dflash2-budget-ms', '--no-draft', '--serve'):
        if flag in args:
            raise ValueError(f'remove inherited {flag} from the source config')
    shutil.copyfile(a.profile, out / 'profile')
    set_arg(args, '--expert-profile', out / 'profile')
    set_arg(args, '--lookup', 3 if a.mode == 'budget' else 0)
    set_arg(args, '--window', 1 if a.mode == 'installed-w1' else 8)
    if a.mode == 'installed-w1':
        args.append('--no-draft')
    else:
        # Server poly_engine() keys off the executable basename; retain the
        # exact ELF but give arbitrarily named private builds the right name.
        (out / 'strata-poly').symlink_to(a.binary.resolve())
        cfg['exe'] = str(out / 'strata-poly')
        if a.mode == 'dspark':
            args.extend(['--draft-model', str(a.dspark_weights.resolve()), '--draft-max', str(a.k)])
            cfg.setdefault('env', {})['STRATA_INKLING_DSPARK_REUSE_V'] = '1'
        else:
            args.extend(['--dflash2-model', str(a.weights.resolve()), '--draft-max',
                         str(0 if a.mode == 'native-w1' else a.k)])
            if a.mode == 'budget':
                args.extend(['--dflash2-budget-ms', str(a.budget)])
    (out / 'slots').mkdir()
    # No service side effects, model switching, external tools, or production logs.
    for key in ('vision', 'images', 'mcp_servers', 'before_load', 'before_load_ignore_failure',
                'before_load_strict'):
        cfg.pop(key, None)
    cfg.update(host='127.0.0.1', port=local_port(), api_key=secrets.token_hex(24),
               slot_save_path=str(out / 'slots'), log=str(out / 'engine.log'),
               open_browser=False, switch_on_request=False, lazy_load=False, idle_unload_s=0)
    trajectory.dump(out / 'config.json', cfg)
    return cfg


def stream_turn(cfg, body, out, turn):
    req = urllib.request.Request(f'http://127.0.0.1:{cfg["port"]}/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + cfg['api_key']})
    chunks, events, deliveries = [], [], []
    start = time.monotonic_ns()
    done = False
    # JSONL is flushed as events arrive, including incomplete/failed streams.
    with (out / f'turn-{turn}-sse.jsonl').open('w') as log:
        def record(kind, value):
            ms = (time.monotonic_ns() - start) / 1e6
            row = {'ms': ms, 'kind': kind, 'data': value}
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            events.append(row)
            return ms

        with OPENER.open(req, timeout=300) as response:
            if response.headers.get_content_type() != 'text/event-stream':
                raise RuntimeError('server did not return SSE')
            data = []
            for raw in response:
                line = raw.decode('utf-8').rstrip('\r\n')
                if line.startswith(':'):
                    record('comment', line)
                elif line.startswith('data:'):
                    data.append(line[5:].lstrip(' '))
                elif not line and data:
                    text = '\n'.join(data)
                    data.clear()
                    if text == '[DONE]':
                        record('done', text)
                        done = True
                        break
                    c = json.loads(text)
                    ms = record('chunk', c)
                    if 'error' in c:
                        raise RuntimeError(f'SSE error: {c["error"]}')
                    chunks.append(c)
                    delta = c['choices'][0]['delta']
                    if delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'):
                        deliveries.append(ms)
        if not done:
            raise RuntimeError('SSE ended without [DONE]')
    if not chunks or chunks[-1]['choices'][0].get('finish_reason') is None:
        raise RuntimeError('SSE ended without finish reason')
    # Production server's own collector handles incremental tool arguments.
    from serve.server import openai_collect
    answer = openai_collect(chunks)
    gaps = [b - a for a, b in zip(deliveries, deliveries[1:])]
    return dict(response=answer, ttft_delivery_ms=deliveries[0] if deliveries else None,
                delivery_ms=deliveries, delivery_gap_ms=gaps, delivery_gap_p90_ms=trajectory.p90(gaps),
                wall_ms=(time.monotonic_ns() - start) / 1e6,
                sse_event_count=len(events), native_output_ids=None,
                native_ids_note='Not exposed by production SSE; no instrumentation added.')


def run_agent(cfg, out, seed, temperature=1):
    workloads.selfcheck()
    task = workloads.CASES['coding']
    fixtures = trajectory.ARCHIVE / 'fixtures'
    keys = list(task['fixtures'])
    given, msgs = set(), [{'role': 'user', 'content': task['question']}]
    result = dict(turns=[], tools=[], grade={'pass': False, 'class': 'infrastructure'},
                  seed=seed, sampling=dict(temperature=temperature, top_p=1, top_k=0, min_p=0),
                  native_output_ids=None,
                  timing_note='SSE delivery estimates, not native model-token timings')
    start = time.monotonic()
    try:
        for i in range(1, 4):
            body = dict(model=cfg['model_name'], messages=msgs, tools=trajectory.TOOLS,
                        temperature=temperature, top_p=1, top_k=0, min_p=0,
                        seed=seed, max_tokens=trajectory.CAP,
                        reasoning_effort='low', stream=True)
            trajectory.dump(out / f'turn-{i}-request.json', body)
            row = stream_turn(cfg, body, out, i)
            result['turns'].append(row)
            choice = row['response']['choices'][0]
            msg = choice['message']
            calls = msg.get('tool_calls') or []
            trajectory.dump(out / 'trajectory.json', result)
            if i == 3:
                result['grade'] = (workloads.grade('coding', msg.get('content') or '')
                                   if not calls and choice['finish_reason'] == 'stop' else
                                   {'pass': False, 'class': 'tool-format', 'reason': 'expected final answer and stop'})
                break
            if len(calls) != 1 or choice['finish_reason'] != 'tool_calls':
                result['grade'] = {'pass': False, 'class': 'tool-format', 'reason': 'expected one complete call'}
                break
            call = calls[0]
            try:
                parsed = json.loads(call['function']['arguments'])
            except (ValueError, KeyError, TypeError):
                result['grade'] = {'pass': False, 'class': 'tool-format', 'reason': 'invalid arguments'}
                break
            before = time.monotonic()
            key, value = workloads.call_result('coding', given,
                SimpleNamespace(name=call['function'].get('name'), arguments=parsed), fixtures)
            result['tools'].append(dict(turn=i, call=call, case_id=key, result=value,
                                        dispatch_s=time.monotonic()-before))
            if key != keys[i-1]:
                result['grade'] = {'pass': False, 'class': 'tool-order', 'reason': 'unexpected or disallowed file read'}
                break
            given.add(key)
            msgs.extend([msg, {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(value)}])
            trajectory.dump(out / 'trajectory.json', result)
    except BaseException as e:
        result['error'] = repr(e)
        raise
    finally:
        result['task_s'] = time.monotonic() - start
        result['delivery_gap_p90_ms'] = trajectory.p90([x for t in result['turns'] for x in t['delivery_gap_ms']])
        trajectory.dump(out / 'trajectory.json', result)
    return result


def worker(out):
    cfg = json.loads((out / 'config.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    cmd = [sys.executable, '-B', '-m', 'serve.server', '--engine', 'strata', '--config', str(out / 'config.json'),
           '--host', '127.0.0.1', '--port', str(cfg['port'])]
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with (out / 'server.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=trajectory.base_env(), stdout=log, stderr=subprocess.STDOUT)
        try:
            start = time.monotonic()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited {proc.returncode}; see server.log and engine.log')
                try:
                    with OPENER.open(f'http://127.0.0.1:{cfg["port"]}/health', timeout=2) as r:
                        health = json.load(r)
                    if health.get('loaded') and health.get('model') == cfg['model_name']:
                        # A port race must not send the private workload to another server.
                        req = urllib.request.Request(f'http://127.0.0.1:{cfg["port"]}/v1/models',
                            headers={'Authorization': 'Bearer ' + cfg['api_key']})
                        with OPENER.open(req, timeout=2) as r:
                            models = json.load(r)
                        if not any(m['id'] == cfg['model_name'] for m in models['data']):
                            raise RuntimeError('private server did not list the configured model')
                        break
                except OSError:
                    pass
                if time.monotonic() - start > 600:
                    raise TimeoutError('server did not load in 600 seconds')
                time.sleep(.25)
            trajectory.dump(out / 'ready.json', dict(health=health, startup_s=time.monotonic()-start, server_pid=proc.pid))
            result = run_agent(cfg, out, plan['seed'], plan['temperature'])
            print(json.dumps({'mode': plan['mode'], 'grade': result['grade'], 'task_s': result['task_s'],
                              'delivery_gap_p90_ms': result['delivery_gap_p90_ms']}), flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def main():
    os.umask(0o077)
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('installed-w1', 'native-w1', 'fixed-k', 'budget', 'dspark'))
    ap.add_argument('--out', type=Path, required=True, help='new private run directory')
    ap.add_argument('--config', type=Path, default=trajectory.CONFIG)
    ap.add_argument('--profile', type=Path, default=trajectory.PROFILE)
    ap.add_argument('--binary', type=Path, help='native production DFlash2 executable (all but installed-w1)')
    ap.add_argument('--weights', type=Path, help='native DFlash2 GGUF')
    ap.add_argument('--dspark-weights', type=Path, default=trajectory.DSPARK)
    ap.add_argument('--k', type=int, choices=range(1, 8), default=3, help='maximum number of guesses')
    ap.add_argument('--budget', type=float, default=100, help='budget mode estimated round cost in ms, not a deadline')
    ap.add_argument('--seed', type=int, choices=trajectory.SEEDS, default=42)
    ap.add_argument('--temperature', type=float, default=1, help='sampling temperature (default 1)')
    ap.add_argument('--expert-epoch', help='decode residency for a native W1/DFlash2 worker')
    a = ap.parse_args()
    if a.mode != 'installed-w1' and not a.binary:
        ap.error('--binary is required for native and DSpark arms')
    if a.expert_epoch is not None and a.mode not in ('native-w1', 'fixed-k', 'budget'):
        ap.error('--expert-epoch requires a native W1/DFlash2 arm')
    if a.mode in ('native-w1', 'fixed-k', 'budget') and not a.weights:
        ap.error('--weights is required for native DFlash2 arms')
    if not math.isfinite(a.budget) or a.budget <= 0:
        ap.error('--budget must be positive and finite')
    if not math.isfinite(a.temperature) or a.temperature < 0:
        ap.error('--temperature must be nonnegative and finite')
    out = a.out.resolve()
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    cfg = make_config(a, out)
    inputs = [a.config, a.profile, Path(cfg['exe']), Path(__file__), ROOT / 'serve/server.py',
              ROOT / 'serve/frontend.py', trajectory.ARCHIVE / 'workloads.py',
              Path(cfg['tokenizer']) / 'chat_template.jinja']
    if '--model' in cfg['args']:
        inputs.append(Path(cfg['args'][cfg['args'].index('--model') + 1]))
    inputs += list((trajectory.ARCHIVE / 'fixtures').glob('*.json'))
    if a.mode in ('native-w1', 'fixed-k', 'budget'):
        inputs.append(a.weights)
    if a.mode == 'dspark':
        inputs.append(a.dspark_weights)
    trajectory.dump(out / 'plan.json', dict(mode=a.mode, seed=a.seed, temperature=a.temperature,
        top_p=1, top_k=0, min_p=0, lookup=3 if a.mode == 'budget' else 0,
        k=a.k, expert_epoch=a.expert_epoch, budget_ms=a.budget if a.mode == 'budget' else None,
        inputs_sha256={str(p.resolve()): trajectory.sha(p) for p in inputs},
        note='SSE delivery gaps are not model-token gaps; each arm has its own free-running prompts.'))
    import fcntl
    trajectory.R.mkdir(parents=True, exist_ok=True)
    with (trajectory.R / 'gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        service = subprocess.check_output(['systemctl', '--user', 'show', 'strata.service', '-p', 'ActiveState', '--value'], text=True).strip()
        if service != 'inactive':
            raise RuntimeError(f'strata.service is {service}; stop it before running')
        try:
            subprocess.run([sys.executable, str(ROOT / 'rdna2-rnd/scripts/run_guarded.py'), '--warn-amdgpu-faults',
                            '--output', str(out / 'guard'), '--', 'timeout', '--kill-after=45s', '2400',
                            sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(out)],
                           cwd=ROOT, env=trajectory.base_env(), check=True)
        finally:
            samples = out / 'guard/vram.jsonl'
            if samples.exists():
                free = [json.loads(line)['free_mib'] for line in samples.read_text().splitlines() if line.strip()]
                trajectory.dump(out / 'guard-summary.json', dict(
                    min_free_vram_mib=min(free) if free else None, samples=len(free),
                    note='Guard samples of free VRAM; not peak process VRAM or RSS.'))


if __name__ == '__main__':
    main()
