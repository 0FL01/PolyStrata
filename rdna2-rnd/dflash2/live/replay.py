#!/usr/bin/env python3
"""Aligned native DFlash2 replay using live's ABI, not the target sampler. One guarded tape per invocation."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DT=np.dtype([('pos','<i8'),('id','<i4'),('gen','<i4'),('tap','<f4',(4,2304))])


def worker(a):
    with a.tape.open('rb') as f:
        if f.read(4)!=b'DFT1': raise ValueError('not DFT1')
    rows=np.memmap(a.tape,dtype=DT,mode='r',offset=4)
    first=int(rows[0]['pos']); p=int(np.flatnonzero(rows['gen'])[0]); n=len(rows)
    if (p<1 or first<0 or (first and p<2048) or not np.array_equal(rows['pos'],np.arange(first,first+n))
        or np.any(rows['gen'][:p]) or not np.all(rows['gen'][p:]==1) or n-p<8):
        raise ValueError('unaligned or insufficient history')
    ids=rows['id'].tolist()
    if not all(0<=v<200058 for v in ids): raise ValueError('invalid captured ID')
    so=ct.CDLL(str(a.library.resolve())); hip=ct.CDLL('libamdhip64.so'); ptr=ct.c_void_p; sz=ct.c_size_t; i=ct.c_int; i64=ct.c_int64
    def fn(lib,name,ret,*args):
        f=getattr(lib,name); f.argtypes=args; f.restype=ret; return f
    def ck(status):
        if status: raise RuntimeError(f'native call failed: {status}')
    alloc=fn(hip,'hipMalloc',i,ct.POINTER(ptr),sz); free=fn(hip,'hipFree',i,ptr); copy=fn(hip,'hipMemcpy',i,ptr,ptr,sz,i)
    op=fn(so,'df2_open',ptr,ct.c_char_p,ct.c_char_p,ptr); close=fn(so,'df2_close',None,ptr)
    append=fn(so,'df2_append',i,ptr,i64,i,ptr,ptr,ptr,ptr); propose=fn(so,'df2_propose',i,ptr,i64,i,i,ct.POINTER(i))
    model=op(os.fsencode(a.weights),os.fsencode(a.target),None)
    if not model: raise RuntimeError('df2_open failed')
    buf=ptr(); stride=2048*2304*4; ck(alloc(ct.byref(buf),stride*4)); planes=[buf.value+j*stride for j in range(4)]
    def inject(j,k):
        for t,plane in enumerate(planes):
            x=np.ascontiguousarray(rows['tap'][j:j+k,t,:]); ck(copy(plane,x.ctypes.data,x.nbytes,1))
        start=time.perf_counter_ns(); ck(append(model,first+j,k,*planes)); return (time.perf_counter_ns()-start)/1e6
    records=[]; pred=(i*7)()
    try:
        # All five attention layers condition on at most the last 2048 committed taps.
        j=max(0,p-2048); inject(j,p-j)
        for j in range(p,n-7):
            t=time.perf_counter_ns(); ck(propose(model,first+j,ids[j],7,pred)); dt=(time.perf_counter_ns()-t)/1e6
            guess=list(pred)
            if not all(0<=v<200058 for v in guess): raise ValueError('invalid proposal')
            acc=next((k for k,g in enumerate(guess) if ids[j+k+1]!=g),7)
            append_ms=inject(j,1)
            records.append(dict(pos=first+j,anchor=ids[j],pred=guess,accepted=acc,draft_ms=dt,append_ms=append_ms))
    finally:
        close(model); ck(free(buf))
    h=[0]*8; ch=[0]*8; j=0
    for z in records: h[z['accepted']]+=1
    while j<len(records):
        accepted=records[j]['accepted']; ch[accepted]+=1; j+=1+accepted
    result=dict(tape=a.tape.name, weights=a.weights.name, base=first, anchor=first+p, rounds=len(records),
        all_anchor_hist=h,closed_loop_hist=ch,mean_accepted=sum(j*f for j,f in enumerate(h))/sum(h),
        mean_closed_accepted=sum(j*f for j,f in enumerate(ch))/sum(ch),
        draft_p50_ms=statistics.median(x['draft_ms'] for x in records),append_p50_ms=statistics.median(x['append_ms'] for x in records))
    (a.out/'proposals.json').write_text(json.dumps(records)+'\n'); (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('library','target','weights','tape','out','run','config'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS); a=p.parse_args()
    if a.worker: worker(a)
    else:
        os.umask(0o077); a.out.mkdir(parents=True,exist_ok=False)
        cfg=json.loads(a.config.read_text()); env=dict(os.environ,LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']))
        subprocess.run(['flock','-x',str(a.run/'gpu.lock'),sys.executable,str(ROOT/'rdna2-rnd/scripts/run_guarded.py'),'--warn-amdgpu-faults','--output',str(a.out/'guard'),'--',
                        'timeout','--kill-after=10s','900',sys.executable,*sys.argv,'--worker'],env=env,check=True,start_new_session=True)
        print((a.out/'result.json').read_text(),flush=True)
