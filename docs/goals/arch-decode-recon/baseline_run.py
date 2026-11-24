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
    a = p.parse_args()
    if subprocess.run(['systemctl','--user','is-active','--quiet','strata.service']).returncode == 0:
        p.error('stop the service before standalone benchmarks')
    if not 1 <= a.length <= 200000-a.steps-1 or a.steps < 1:
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
    csv = ','.join(map(str,ids))
    env = dict(PATH='/usr/bin:/bin', HOME=str(a.run.resolve()), LC_ALL='C', **cfg['env'],
               LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']))
    # Explicit opt-in diagnostics only, never inherited on a baseline timing leg.
    for key in ('STRATA_VISIT_TRACE',):
        if key in os.environ:
            env[key] = os.environ[key]
    cmd = [str(a.binary.resolve()),*args,'--no-draft','--lookup','0','--serve']
    meta = dict(config_sha256=sha(a.config.read_bytes()), binary_sha256=sha(a.binary.read_bytes()),
                seed_sha256=sha(local.read_bytes()), prompt_sha256=sha(csv.encode()), length=a.length,
                workload=a.workload, cmd=cmd, env=env)
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
            while not line().startswith('READY '):
                pass
            send(f'GEN {a.steps+1} temperature=0 seed=42 '+csv)
            ts, output = [], []
            while True:
                msg = line()
                if msg.startswith('T '):
                    ts.append(time.monotonic_ns()); output.append(int(msg.split()[1]))
                if msg.startswith('DONE '):
                    break
            f = msg.split(); dt = [(b-c)/1e6 for c,b in zip(ts,ts[1:])]
            if len(dt)<2 or len(output)!=int(f[1]) or int(f[2])!=a.length or f[6:8]!=['0','0']:
                raise RuntimeError(msg)
            q = statistics.quantiles(dt,n=100,method='inclusive')
            result=dict(length=a.length,done=msg,decode_steps=len(dt),decode_t_s=1000*len(dt)/float(f[4]),
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
