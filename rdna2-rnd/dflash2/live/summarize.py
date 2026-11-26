#!/usr/bin/env python3
"""Report acceptance and proposer-only costs from complete native replay receipts."""
import argparse
import json
import statistics
from pathlib import Path

PROFILES=('BF16','Q8_0','Q4_0','Q4_0-mixed','Q4_K-mixed')
DOMAINS=('en_reasoning','code_python','code_rust','code_cpp','tool_calls','json_struct','agent_traj')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--resume',type=Path,action='append',default=[],help='additional fresh roots for completed retry legs')
    p.add_argument('--domains',nargs='+',default=DOMAINS)
    p.add_argument('--profiles',nargs='+',default=PROFILES)
    args=p.parse_args(); pooled={}
    print('domain,profile,anchors,mean_hits,walk_rounds,mean_walk_hits,draft_p50_ms,append_p50_ms')
    for d in args.domains:
        anchors=None
        for q in args.profiles:
            candidates = [root/d/q for root in [args.directory,*args.resume]
                          if (root/d/q/'result.json').is_file()
                          and (root/d/q/'guard/stop-reasons.json').is_file()
                          and not json.loads((root/d/q/'guard/stop-reasons.json').read_text())]
            if len(candidates) != 1:
                raise ValueError(f'{d}/{q}: expected one completed guarded leg, got {candidates}')
            folder, = candidates
            r=json.loads((folder/'result.json').read_text()); ps=json.loads((folder/'proposals.json').read_text())
            keys=[(x['pos'],x['anchor']) for x in ps]
            if anchors is not None and keys!=anchors: raise ValueError(f'{folder}: anchors differ')
            anchors=keys
            h=r['all_anchor_hist']; ch=r['closed_loop_hist']; times=[x['draft_ms'] for x in ps]
            inject=[x['append_ms'] for x in ps]; acc=[x['accepted'] for x in ps]; walk=[]; j=0
            while j<len(ps):
                n=ps[j]['accepted']; walk.append(n); j+=1+n
            if h!=[acc.count(k) for k in range(8)] or ch!=[walk.count(k) for k in range(8)]:
                raise ValueError(f'{folder}: prefix counts differ')
            terms=(acc,walk,times,inject)
            for buf,values in zip(pooled.setdefault(q,([],[],[],[])),terms): buf.extend(values)
            report(d,q,*terms)
    for q,terms in pooled.items(): report('pooled',q,*terms)


def report(d,q,acc,walk,times,inject):
    print(f'{d},{q},{len(acc)},{statistics.mean(acc):.3f},{len(walk)},{statistics.mean(walk):.3f},'
          f'{statistics.median(times):.3f},{statistics.median(inject):.3f}')


if __name__=='__main__': main()
