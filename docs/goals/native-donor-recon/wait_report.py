"""Summarize time at the same serial position under opposite eight-position blocks."""
import json
from pathlib import Path
import re
from statistics import mean, median
import sys


def report(root, label):
    legs={k:json.loads((root/(label+k)/'metrics.json').read_text()) for k in '012A'}
    ctrl=legs['0']; n=ctrl['n']; steps=ctrl['steps']; blocks={}; receipt={}
    for k,v in legs.items():
        assert all(v[f]==ctrl[f] for f in ('prompt_sha','steps','resume','n'))
        assert v['mode']==(int(k) if k in '12' else 0) and len(v['dt_ms'])==steps and len(v['ids'])==steps+1
        assert json.loads((root/('guard-'+label+k)/'stop-reasons.json').read_text())==[]
        receipt[k]=dict(rate=1000/mean(v['dt_ms']),first_diff_1based=next(
            (i for i,(a,b) in enumerate(zip(ctrl['ids'],v['ids']),1) if a!=b),None))
        if k in '12':
            m=re.search(r'RECON scored=(\d+) issued=(\d+) cpu_hits=(\d+) bytes=(\d+)',
                        (root/(label+k)/'stderr').read_text())
            receipt[k].update(zip(('scored','issued','hits','touched_B'),map(int,m.groups())))
    assert legs['0']['elf_sha']==legs['A']['elf_sha'] and legs['1']['elf_sha']==legs['2']['elf_sha']
    if any(leg['first_diff_1based'] for leg in receipt.values()): return dict(legs=receipt,comparable=False)
    for pos in range(n,n+steps):
        if pos%8<2: continue
        i=pos-n; phase=(pos//8)%2; on,off=(('1','2'),('2','1'))[phase]
        # Compare equivalent positions, never an early token to a late one. Only use
        # complete blocks. Synthetic "treatment" 0/A is a noise diagnostic, not a CI.
        ab=legs[on]['dt_ms'][i]-legs[off]['dt_ms'][i]
        aa=(legs['0']['dt_ms'][i]-legs['A']['dt_ms'][i])*(1-2*phase)
        blocks.setdefault(pos//8,[]).append((ab,aa))
    diffs=[tuple(map(mean,zip(*v))) for v in blocks.values() if len(v)==6]
    assert diffs
    med=[median(v) for v in zip(*diffs)]; avg=[mean(v) for v in zip(*diffs)]
    return dict(legs=receipt,comparable=True,blocks=len(diffs),median_block_delta_ms=dict(on_minus_off=med[0],aa=med[1]),
                mean_block_delta_ms=dict(on_minus_off=avg[0],aa=avg[1]),faster_blocks=sum(x<0 for x,y in diffs))


if __name__=='__main__':
    print(json.dumps({k:report(Path(sys.argv[1]),k) for k in ('s','l')},indent=2))
