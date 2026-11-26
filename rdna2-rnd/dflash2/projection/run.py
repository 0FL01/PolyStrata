#!/usr/bin/env python3
"""Build and run the projection screen with the Release target's gfx1031 flags."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'third_party/llama.cpp/gguf-py'))
import gguf


def build(b, d):
    lines = [shlex.split(c) for c in subprocess.check_output(['ninja','-t','commands','poly_mmvq_segments_test'],cwd=b,text=True).splitlines()]
    c, = (a for a in lines if a[-1].endswith('/mmvq_segments_test.cu'))
    obj=c[c.index('-o')+1]
    for opt in ['-MF','-MT']:
        i=c.index(opt); del c[i:i+2]
    c.remove('-MD'); c[-1]=str(HERE/'bench.cu'); c[c.index('-o')+1]=str(d/'bench.o')
    subprocess.run(c,cwd=b,check=True)
    l=lines[-1][2:-2]; l[l.index(obj)]=str(d/'bench.o'); l[l.index('-o')+1]=str(d/'bench')
    for j,arg in enumerate(l):
        if arg.startswith('--dependency-file=') and j and l[j-1]=='-Xlinker':
            del l[j-1:j+1]; break
    subprocess.run(l,cwd=b,check=True)
    c[c.index('-o')+1]=str(d/'bench-gfx1031.s'); c[c.index('-c')]='-S'
    subprocess.run(c+['--cuda-device-only'],cwd=b,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'build-poly-hip'); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--weights',type=Path,nargs='+',required=True)
    p.add_argument('--tensors',nargs='+',default=['blk.0.ffn_gate.weight','blk.0.attn_conv_proj.weight','blk.0.ffn_down.weight'])
    a=p.parse_args(); os.umask(0o077); d=a.out.resolve(); d.mkdir(parents=True,exist_ok=False)
    build(a.build.resolve(),d)
    # These are real tap values, NOT claims about the hidden distribution in a draft FFN.
    dt=np.dtype([('p','<i8'),('id','<i4'),('gen','<i4'),('t','<f4',(4,2304))]); tape=a.run/'taps/json_struct.dft1'
    with tape.open('rb') as f:
        if f.read(4)!=b'DFT1': raise ValueError('bad DFT1')
    trace=np.memmap(tape,mode='r',dtype=dt,offset=4); rows=trace['t'][trace['gen']==1][:8]
    if len(rows)!=8: raise ValueError('no decode rows')
    env=dict(os.environ); c=a.build.resolve()/'CMakeCache.txt'
    # TheRock builds need its loader paths even when the bench links by absolute path.
    import re
    hip=Path(re.search('^CMAKE_HIP_COMPILER:[^=]+=(.*)$',c.read_text(),re.M)[1]); sdk=hip.parents[2]
    env['LD_LIBRARY_PATH']=':'.join(map(str,[sdk/'lib',*(sdk.parent.glob('_rocm_sdk_libraries_gfx103*/lib'))]))+':'+env.get('LD_LIBRARY_PATH','')
    for j,g in enumerate(a.weights):
        tensors={t.name:t for t in gguf.GGUFReader(str(g)).tensors}
        for name in a.tensors:
            t=tensors[name]; width, nr=map(int,t.shape); q=int(t.tensor_type)
            if not (q in (2,8,12,13,14,20,23) and width<=9216): raise ValueError((name,q,width))
            path=d/f'x-{width}.f32'; rows.reshape(8,-1)[:,:width].copy().tofile(path)
            out=d/f'{j}-{name}'; print(g.name,name,flush=True)
            subprocess.run(['flock','-x',str(a.run/'gpu.lock'),sys.executable,str(ROOT/'rdna2-rnd/scripts/run_guarded.py'),'--warn-amdgpu-faults','--output',str(out),
                            '--',str(d/'bench'),str(g),str(t.data_offset),str(q),str(nr),str(width),str(path)],check=True,env=env,start_new_session=True)
            print((out/'command.log').read_text(),flush=True)


if __name__=='__main__': main()
