"""One private pipe-server run. Invoke through run_guarded.py, not directly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from strata_tokenizer import Tokenizer


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('length', type=int)
    p.add_argument('--save', action='store_true')
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--binary', type=Path, default=ROOT / 'build-poly-hip/strata-poly')
    p.add_argument('--config', type=Path, default=Path.home() / '.config/strata/inkling2-mini/strata-inkling-2-mini.json')
    p.add_argument('--profile-seed', type=Path, required=True)
    p.add_argument('--workload', choices=['mixed','code','prose'], default='mixed')
    p.add_argument('--prompt-csv', type=Path, help='reuse the exact CSV from a preceding run')
    p.add_argument('--visit-trace', type=Path, help='instrumented binary only; not an E2E timing leg')
    p.add_argument('--profiled', action='store_true', help='binary is an explicit profiler wrapper')
    a = p.parse_args()
    if subprocess.run(['systemctl','--user','is-active','--quiet','strata.service']).returncode == 0:
        p.error('stop the service before standalone benchmarks')
    if not 1 <= a.length <= 200000-a.steps-1 or a.steps < 2:
        p.error('prompt and response must fit in the production context')
    os.umask(0o077)
    a.run.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_bytes())
    args = cfg['args'].copy()
    local = a.run.resolve() / 'profile.txt'
    shutil.copyfile(a.profile_seed, local)
    args[args.index('--expert-profile')+1] = str(local)
    tok = Tokenizer.from_dir(cfg['tokenizer'])
    prose = (ROOT/'docs/AI_SETUP.md').read_text()
    code = (ROOT/'tools/replay_cache_policy.py').read_text()
    tool = '\nTool result: {"passed": 47, "failed": 2}. Explain which tests to rerun and why.\n'
    corpus = tok.encode({'mixed':prose+code+tool,'code':code,'prose':prose}[a.workload])
    ids = (corpus*((a.length+len(corpus)-1)//len(corpus)))[:a.length]
    csv = a.prompt_csv.read_text() if a.prompt_csv else ','.join(map(str,ids))
    if len(csv.split(',')) != a.length or not all(s.isascii() and s.isdigit() for s in csv.split(',')):
        p.error('wrong length or invalid IDs in prompt CSV')
    env = dict(PATH='/usr/bin:/bin', HOME=str(a.run.resolve()), LC_ALL='C', **cfg['env'],
               LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']))
    # Explicit opt-in diagnostics only, never inherited on a baseline timing leg.
    if any('TRACE' in k or k.startswith(('ROCP','LD_PRELOAD')) for k in env):
        p.error('unexpected trace or profiler setting in production configuration')
    if a.visit_trace:
        env['STRATA_VISIT_TRACE'] = str(a.visit_trace.resolve())
        env['STRATA_INKLING_VISIT_TRACE_FIRST'] = str(a.length)
        env['STRATA_INKLING_VISIT_TRACE_END'] = str(a.length + a.steps)
    cmd = [str(a.binary.resolve()),*args,'--no-draft','--lookup','0','--serve']
    meta = dict(config_sha256=sha(a.config.read_bytes()), binary_sha256=sha(a.binary.read_bytes()),
                seed_sha256=sha(local.read_bytes()), prompt_sha256=sha(csv.encode()), length=a.length,
                workload=a.workload, cmd=cmd, env=env, trace=bool(a.visit_trace), profiled=a.profiled)
    (a.run/'config.json').write_text(json.dumps(meta,indent=2)+'\n')
    (a.run/'prompt.csv').write_text(csv)
    with (a.run/'engine.log').open('w') as log, subprocess.Popen(cmd, cwd=a.run, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1) as proc:
        def send(msg):
            proc.stdin.write(msg+'\n')
            proc.stdin.flush()

        def line():
            msg = proc.stdout.readline().strip()
            if not msg or msg.startswith(('ERR','SERR','FATAL')):
                raise RuntimeError(msg or 'engine pipe closed; see engine.log')
            return msg

        try:
            ready = line()
            while not ready.startswith('READY '):
                ready = line()
            if int(ready.split()[1]) != 200000:
                raise RuntimeError(ready)
            send(f'GEN {a.steps+1} temperature=0 seed=42 '+csv)
            ts, output, resume = [], [], None
            while True:
                msg = line()
                if msg.startswith('T '):
                    ts.append(time.monotonic_ns()); output.append(int(msg.split()[1]))
                if msg.startswith('RESUME '):
                    resume = int(msg.split()[1])
                if msg.startswith('DONE '):
                    break
            f = msg.split(); dt = [(b-c)/1e6 for c,b in zip(ts,ts[1:])]
            if (len(dt)!=a.steps or len(output)!=int(f[1]) or int(f[2])!=a.length or
                    f[5:9]!=['length','0','0','0'] or int(f[14])!=a.length or resume!=0 or ts[-1]<=ts[0]):
                raise RuntimeError(msg)
            q = statistics.quantiles(dt,n=100,method='inclusive')
            result=dict(length=a.length,done=msg,decode_steps=len(dt),decode_t_s=1e9*len(dt)/(ts[-1]-ts[0]),
                        arrivals_ns=ts, engine_decode_ms=float(f[4]),
                        arrival_ms=dict(p10=q[9],median=statistics.median(dt),p90=q[89],p99=q[98]),
                        step_ms=dt,greedy_ids=output)
            (a.run/'metrics.json').write_text(json.dumps(result)+'\n')
            if a.save:
                send('SAVE '+str(a.run.resolve()/'session.bin'))
                while not line().startswith('SAVED '):
                    pass
        finally:
            proc.stdin.close()
        if proc.wait()!=0:
            raise RuntimeError('engine exit; see engine.log')


if __name__=='__main__':
    main()
