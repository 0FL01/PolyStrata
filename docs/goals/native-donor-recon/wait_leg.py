"""One untraced, private W1 engine process. Run via flock + run_guarded.py only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import threading
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('out','exe','config','seed','prompt'):
        p.add_argument(key,type=Path)
    p.add_argument('--restore',type=Path)
    p.add_argument('--mode',type=int,choices=range(4),default=0)
    p.add_argument('--steps',type=int,default=160)
    p.add_argument('--require-registered',type=float,metavar='GIB',help='wait for the background mapper before GEN')
    a=p.parse_args()
    assert subprocess.run(['systemctl','--user','is-active','--quiet','strata.service']).returncode!=0
    a.out.mkdir(exist_ok=False); cfg=json.loads(a.config.read_text()); args=cfg['args'].copy()
    seed=a.out.resolve()/'profile.seed'; shutil.copyfile(a.seed,seed)
    args[args.index('--expert-profile')+1]=str(seed)
    resume=0; ids=a.prompt.read_text().strip()
    if a.restore:
        # The saved rows are immutable, including the visited conv and KV state.
        # Append one token (e.g. last emitted, not yet eval'd in the SAVE).
        assert ids.isdigit()
        with a.restore.open('rb') as f:
            assert f.read(8)==b'STRSESS\1'; f.seek(232); resume,=struct.unpack('<Q',f.read(8))
            assert 1<=resume<200000; committed=struct.unpack(f'<{resume}i',f.read(4*resume))
        ids=','.join(map(str,committed))+','+ids
    n=ids.count(',')+1
    env=dict(PATH='/usr/bin:/bin',HOME=str(a.out.resolve()),LC_ALL='C',**cfg['env'],
             LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']),RECON_WAIT=str(a.mode))
    assert not any('TRACE' in k or k.startswith(('ROCP','LD_PRELOAD')) for k in env)
    cmd=[str(a.exe.resolve()),*args,'--no-draft','--lookup','0','--serve']
    ready=threading.Event(); registered=[]; fault_span=None
    with (a.out/'stderr').open('w') as log, (a.out/'wire').open('w') as wire, subprocess.Popen(cmd,
            env=env, cwd=a.out, stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=log if a.require_registered is None else subprocess.PIPE,text=True,bufsize=1) as proc:
        def record():
            for text in proc.stderr:
                log.write(text); log.flush()
                m=re.search(r'([\d.]+) GiB of experts outside VRAM registered for direct uploads',text)
                if m:
                    registered.append(float(m[1])); ready.set()
            ready.set()
        def faults(): # whole process, not per-thread or child counts
            fields=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()
            return int(fields[7]),int(fields[9]) # minflt / majflt, stat fields 10 / 12
        if a.require_registered is not None:
            logger=threading.Thread(target=record); logger.start()
        def line():
            s=proc.stdout.readline().strip(); print(s,file=wire,flush=True)
            if not s or s.startswith(('ERR','FATAL','SERR')): raise RuntimeError(s or 'pipe closed')
            return s
        def until(prefix):
            while not (s:=line()).startswith(prefix): pass
            return s
        def send(msg): proc.stdin.write(msg+'\n'); proc.stdin.flush()
        try:
            until('READY ')
            if a.restore:
                send('RESTORE '+str(a.restore.resolve())); assert int(until('RESTORED ').split()[1])==resume
            if a.require_registered is not None:
                assert ready.wait(100) and registered==[a.require_registered],('not fully registered',registered)
                start=faults()
            send(f'GEN {1+a.steps} temperature=0 seed=42 '+ids)
            times=[]; out=[]; reuse=None
            while True:
                s=line()
                if s.startswith('T '): times.append(time.monotonic_ns()); out.append(int(s.split()[1]))
                elif s.startswith('RESUME '): reuse=int(s.split()[1])
                elif s.startswith('DONE '): break
            assert len(out)==1+a.steps and reuse==resume and s.split()[5]=='length',s
            if a.require_registered is not None: fault_span=[b-c for c,b in zip(start,faults())]
        finally:
            proc.stdin.close()
        assert proc.wait()==0
        if a.require_registered is not None: logger.join()
    (a.out/'metrics.json').write_text(json.dumps(dict(steps=a.steps,n=n,resume=reuse,ids=out,done=s,mode=a.mode,
        elf_sha=hashlib.sha256(a.exe.read_bytes()).hexdigest(),prompt_sha=hashlib.sha256(ids.encode()).hexdigest(),
        registered_gib=registered,gen_minflt_majflt=fault_span,dt_ms=[(b-c)/1e6 for c,b in zip(times,times[1:])]),indent=2)+'\n')


if __name__=='__main__': main()
