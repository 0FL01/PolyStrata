#!/usr/bin/env python3
"""Capture a *full* DFlash2 SWA history at an absolute long position, without tapping an entire prefill.

Optional native target RESTORE, then append the same 2048 fresh prompt rows and
generate greedily. The suffix CSV should contain exactly 2048 IDs. Verify every
saved absolute position, prompt/generated marker and input ID before using a tape.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
RECORD=16+4*2304*4


def worker(req):
    args=json.loads(req.read_text()); base=args['base']; suffix=args['suffix']; d=req.parent
    with (d/'engine.log').open('w') as err, (d/'wire.log').open('w') as wire, subprocess.Popen(args['cmd'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True) as p:
        def send(line): p.stdin.write(line+'\n'); p.stdin.flush()
        def until(prefix):
            for line in p.stdout:
                wire.write(line)
                if line.startswith(('ERR','SERR','FATAL')): raise RuntimeError(line)
                if line.startswith(prefix): return line.strip()
            raise RuntimeError('unexpected EOF')
        until('READY ')
        if args['snapshot']:
            send('RESTORE '+args['snapshot']); until('RESTORED ')
        send('GEN 33 temperature=0 top_k=0 seed=42 '+','.join(map(str,base+suffix)))
        if int(until('RESUME ').split()[1])!=len(base): raise RuntimeError('unexpected cached prefix')
        ids=[]
        for line in p.stdout:
            wire.write(line)
            if line.startswith('T '): ids.append(int(line.split()[1]))
            if line.startswith('DONE '): break
            if line.startswith('ERR'): raise RuntimeError(line)
        else: raise RuntimeError('no DONE')
        send('QUIT'); p.stdin.close()
        if p.wait(timeout=60): raise RuntimeError('engine failed')
    with (d/'tap.dft1').open('rb') as f:
        if f.read(4)!=b'DFT1': raise ValueError('bad magic')
        for i,t in enumerate(suffix+ids[:-1]):
            rec=f.read(RECORD)
            if len(rec)!=RECORD or struct.unpack_from('<qii',rec)!=(len(base)+i,t,int(i>=2048)):
                raise ValueError(f'bad DFT1 record {i}')
        if f.read(1) or len(ids)<9: raise ValueError('insufficient generation or extra records')
    (d/'result.json').write_text(json.dumps(dict(prefix=len(base),anchor=len(base)+2048,ids=ids))+'\n')


def main():
    if len(sys.argv)==3 and sys.argv[1]=='worker': return worker(Path(sys.argv[2]))
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','out','binary','config','profile','suffix'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--snapshot',type=Path); a=p.parse_args(); os.umask(0o077)
    cfg=json.loads(a.config.read_text()); suffix=list(map(int,a.suffix.read_text().strip().split(',')))
    if len(suffix)!=2048: p.error('need exactly 2048 fresh rows')
    base=[]
    if a.snapshot:
        # Native STRATA disk header: sequence length then int32 IDs. RESTORE itself validates the model and payload.
        with a.snapshot.open('rb') as f:
            f.seek(232); n=struct.unpack('<Q',f.read(8))[0]
            if not 0<n<=200000-2048-33: p.error('wrong snapshot length')
            base=list(struct.unpack(f'<{n}i',f.read(4*n)))
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False); args=list(cfg['args'])
    shutil.copyfile(a.profile,out/'profile'); args[args.index('--expert-profile')+1]=str(out/'profile')
    env={k:v for k,v in os.environ.items() if not k.startswith(('RECON_','STRATA_'))}
    env.update(cfg['env'],LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']),RECON_DUMP_FROM=str(len(base)),RECON_DUMP_TAPS=str(out/'tap.dft1'))
    req=out/'request.json'; req.write_text(json.dumps(dict(base=base,suffix=suffix,snapshot=str(a.snapshot.resolve()) if a.snapshot else None,
        cmd=[str(a.binary.resolve()),*args,'--serve','--no-draft','--lookup','0','--eos-ids','200006'])))
    with (a.run/'gpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run([sys.executable,str(ROOT/'rdna2-rnd/scripts/run_guarded.py'),'--warn-amdgpu-faults','--output',str(out/'guard'),'--',
                        'timeout','--kill-after=15s','400',sys.executable,__file__,'worker',str(req)],env=env,check=True)
    print((out/'result.json').read_text(),flush=True)


if __name__=='__main__': main()
