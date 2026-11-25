"""Count exact strong-logit APC survivors on archived native post-softcap rows.

Rows outside the set get -inf independently of their weak logit. This is not a
draft shortlist, top-k approximation, or a measurement of contrastive quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def screen(path, beta, export):
    logb=np.log(np.float32(beta)); digest=hashlib.sha256(); sizes=[]; mass=[]
    with path.open('rb') as f:
        header=f.read(24); digest.update(header); magic,vocab,n,first=struct.unpack('<8sIIQ',header)
        assert magic==b'PSLG64\r\n' and vocab==200058 and n>0
        for j in range(n):
            raw=f.read(4*(1+vocab)); assert len(raw)==4*(1+vocab); digest.update(raw)
            z=np.frombuffer(raw,'<f4',offset=4); assert np.isfinite(z).all()
            m=z.max(); ids=np.flatnonzero(z>=m+logb); sizes.append(len(ids))
            p=np.exp(z.astype(np.float64)-m); mass.append(float(p[ids].sum()/p.sum()))
            if export: print(json.dumps(dict(file=path.name,first=first,pos=first+j,ids=ids.tolist())))
        assert not f.read(1)
    q=lambda a,p:float(np.quantile(a,p))
    return dict(sha256=digest.hexdigest(),pos=first,rows=n,sizes=sizes,
                p50=q(sizes,.5),p95=q(sizes,.95),max=max(sizes),mass_p50=q(mass,.5),mass_min=min(mass))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('paths',type=Path,nargs='+')
    parser.add_argument('--beta',type=float,default=.1);parser.add_argument('--export',action='store_true')
    a=parser.parse_args(); assert 0<a.beta<=1
    result={str(p):screen(p,a.beta,a.export) for p in a.paths};
    if not a.export: print(json.dumps(result,indent=2))
