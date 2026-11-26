#!/usr/bin/env python3
"""Serial policy legs with actual DFT1 features and same-policy greedy truth; never a live codec window."""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

from swa import CONFIG, R, ROOT, T, WEIGHTS, put, sha, tagged


PREFIX = T/'global-kv-20261125/longagent/build-129'
SUFFIX = T/'arch-campaign.XWM1yK/long-code-65536/run/prompt.csv'
WARMUP = 2048
PLANE = 2304 * 4
RECORD = 16 + 4*PLANE


def worker(path):
    a = json.loads(path.read_text())
    d = path.parent
    with (d/'target.log').open('x') as err, (d/'wire.log').open('x') as wire, \
         subprocess.Popen(a['cmd'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True) as p:
        def line():
            s = p.stdout.readline()
            wire.write(s)
            if not s or s.startswith(('ERR', 'SERR', 'FATAL')):
                raise RuntimeError(f'bad target reply: {s!r}')
            return s.split()

        def send(s):
            p.stdin.write(s+'\n')
            p.stdin.flush()

        def until(tag):
            while (r := line())[0] != tag:
                pass
            return r

        until('READY')
        send('RESTORE '+a['snapshot'])
        assert int(until('RESTORED')[1]) == a['base']
        send(f"GEN {a['steps']} temperature=0 top_p=1 top_k=0 min_p=0 seed=42 " + Path(a['prompt']).read_text().strip())
        tok, res = [], []
        while True:
            x = line()
            if x[0] == 'RESUME':
                res.append(int(x[1]))
            if x[0] == 'T':
                tok.append(int(x[1]))
            if x[0] == 'DONE':
                break
        assert res == [a['base']] and 16 <= len(tok) <= a['steps']
        assert int(x[1]) == len(tok) and list(map(int, x[6:8])) == [0, 0], 'not W1'
        send('QUIT')
        p.stdin.close()
        if p.wait(timeout=60):
            raise RuntimeError('target exit failed')
    put(d/'wire.json', dict(ids=tok, done=x, prefix=res[0], anchor=res[0]+WARMUP))


def checked_trace(d, suffix, result):
    p = d/'tap.dft1'
    with p.open('rb') as f:
        assert f.read(4) == b'DFT1'
        for i, t in enumerate(suffix+result['ids'][:-1]):
            rec = f.read(RECORD)
            assert len(rec) == RECORD and struct.unpack_from('<qii', rec) == (result['prefix']+i, t, int(i >= WARMUP))
        assert not f.read(1)
    return sha(p)


def checked_replay(d, result):
    truth = result['ids'][:-1]  # Replay consumes only rows actually fed, no synthetic final feature.
    out = {}
    for mode in ('all-positions', 'closed-loop'):
        rows = [x for s in (d/'steps.jsonl').read_text().splitlines() if (x := json.loads(s))['mode'] == mode]
        i = 0
        for x in rows:
            assert x['start'] == result['anchor']+i and x['anchor'] == truth[i] and len(x['draft_tokens']) == 7
            acc = next((j for j, (p, q) in enumerate(zip(x['draft_tokens'], truth[i+1:i+8], strict=True)) if p != q), 7)
            assert x['accepted'] == acc
            i += 1+acc if mode == 'closed-loop' else 1
        assert rows and i+8 > len(truth)
        out[mode] = dict(rounds=len(rows), accepted=sum(x['accepted'] for x in rows), offered=7*len(rows),
                         hist=[sum(x['accepted'] == k for x in rows) for k in range(8)],
                         k3_accepted=sum(min(x['accepted'], 3) for x in rows), k3_offered=3*len(rows))
    return out


def native_state_difference(x, y):
    # Tideglass Q8, context > 1024: conv visits in logical order, then SWA visits in logical order.
    # Each SWA visit contributes 1023 chronological rows per plane, K/V with one 64-byte scale row each.
    conv_stride = 3*(2*1024+2*2304)*4
    swa_stride = 1023*(1024+64)*2
    with x.open('rb') as r, y.open('rb') as s:
        def changed(visits, size):
            out = []
            for v in visits:
                a, b = r.read(size), s.read(size)
                assert len(a) == len(b) == size
                if a != b:
                    out.append(v)
            return out
        out = dict(conv_visits=changed(range(48), conv_stride),
                   swa_visits=changed((i for i in range(48) if i % 6 != 5), swa_stride))
        assert not r.read(1) and not s.read(1)
    return out


def compare(d0, d, a, b):
    common = sum(1 for _ in itertools.takewhile(lambda p: p[0] == p[1], zip(a['ids'], b['ids'])))
    c = min(len(a['ids']), len(b['ids']))
    h = dict(placement=a['placement']==b['placement'], equal_greedy_prefix=common,
             first_differing_id=None if common == c else [common, a['ids'][common], b['ids'][common]],
             same_history_state_comparable=a['committed_ids_sha256']==b['committed_ids_sha256'],
             same_state_geometry=a['state_bytes']==b['state_bytes'],
             equal_native_state_prefix=a['native_state_prefix_sha256']==b['native_state_prefix_sha256'])
    if h['same_history_state_comparable'] and h['same_state_geometry']:
        h['native_state_changes'] = native_state_difference(d0/'state.bin', d/'state.bin')
    first, counts = [None]*4, [0]*4
    with (d0/'tap.dft1').open('rb') as r, (d/'tap.dft1').open('rb') as s:
        assert r.read(4) == s.read(4) == b'DFT1'
        for i in range(WARMUP+min(common, c-1)):
            x, y = r.read(RECORD), s.read(RECORD)
            assert len(x)==len(y)==RECORD and x[:16] == y[:16]
            for j in range(4):
                at = 16+j*PLANE
                if x[at:at+PLANE] != y[at:at+PLANE]:
                    counts[j] += 1
                    if first[j] is None:
                        first[j] = a['prefix']+i
    h.update(tap_labels=[5,23,41,48], first_differing_tap_row=first, differing_tap_rows=counts,
             shared_input_tap_rows=WARMUP+min(common,c-1))
    def proposals(path):
        return {x['start']: x for s in path.read_text().splitlines() if (x:=json.loads(s))['mode']=='all-positions'}
    x, y = (proposals(p/'steps.jsonl') for p in (d0, d))
    shared = sorted(i for i in x.keys() & y.keys() if i < a['anchor']+common)
    h.update(common_history_proposals=len(shared), changed_proposal_tuples=sum(x[i]['draft_tokens'] != y[i]['draft_tokens'] for i in shared))
    return h


def main():
    if len(sys.argv)==3 and sys.argv[1]=='worker':
        return worker(Path(sys.argv[2]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--binary', type=Path, default=R/'interactions/w1-bin/strata-poly-w1-dft1')
    p.add_argument('--snapshot', type=Path, default=PREFIX/'session.bin')
    p.add_argument('--profile', type=Path, default=PREFIX/'profile.seed')
    p.add_argument('--suffix-source', type=Path, default=SUFFIX)
    p.add_argument('--offset', type=int, default=10000)
    p.add_argument('--arena', type=int, default=2560)
    p.add_argument('--draft', type=Path, default=WEIGHTS)
    p.add_argument('--replay', type=Path, default=R/'dflash2-replay')
    p.add_argument('--steps', type=int, default=33)
    a = p.parse_args()
    os.umask(0o077)
    out=a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with a.snapshot.open('rb') as f:
        hdr=f.read(240)
        assert hdr[:8]==b'STRSESS\1' and struct.unpack_from('<q',hdr,72)[0]==1
        n,=struct.unpack_from('<Q',hdr,232)
        assert n > 0 and 35072 <= n+WARMUP < 200000-a.steps
        base=struct.unpack(f'<{n}i',f.read(4*n))
    suffix=list(map(int,a.suffix_source.read_text().strip().split(',')))[a.offset:a.offset+WARMUP]
    assert len(suffix)==WARMUP and all(0 <= t < 200058 for t in suffix)
    prompt=out/'prompt.csv'
    prompt.write_text(','.join(map(str, [*base,*suffix]))+'\n')
    cfg=json.loads(CONFIG.read_text())
    target=cfg['args'][cfg['args'].index('--model')+1]
    put(out/'plan.json', dict(prompt_sha256=sha(prompt), base=n, offset=a.offset, steps=a.steps, arena_mib=a.arena,
        snapshot=str(a.snapshot.resolve()), snapshot_header_sha256=sha(a.snapshot,240), snapshot_bytes=a.snapshot.stat().st_size,
        suffix_source=str(a.suffix_source), profile_sha256=sha(a.profile), serial_elf_sha256=sha(a.binary),
        replay_elf_sha256=sha(a.replay), draft_sha256=sha(a.draft)))

    clean={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_', 'DF2_', 'RECON_', 'IX_'))}
    results = {}
    def guarded(d, env, command):
        argv=[sys.executable,str(ROOT/'rdna2-rnd/scripts/run_guarded.py'),'--warn-amdgpu-faults','--output',str(d),'--',
              'timeout','--kill-after=10s','240',*command]
        put(d.parent/(d.name+'-launch.json'), dict(argv=argv, env=env))
        with (R/'gpu.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            subprocess.run(argv,env=clean|env,check=True)
        assert json.loads((d/'stop-reasons.json').read_text())==[]
        return min(json.loads(l)['free_mib'] for l in (d/'vram.jsonl').read_text().splitlines())

    for mode, expected in [('dense',(False,False)),('k256',(True,False)),('v4',(False,True))]:
        d=out/mode
        d.mkdir()
        args=list(cfg['args'])
        shutil.copyfile(a.profile,d/'profile')
        args[args.index('--expert-profile')+1]=str(d/'profile')
        args[args.index('--expert-swaps')+1]='0'
        args+=['--serve','--no-draft','--window','1','--lookup','0','--eos-ids','200006','--cpu-share','0','--stage-ahead','0']
        env=dict(cfg['env'],LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']),LC_ALL='C',STRATA_FREEZE_EXPERTS='1',
            STRATA_INKLING_COMPACT_SWA='1',STRATA_INKLING_PREFETCH='0',STRATA_INKLING_FUND_EXPERTS='0',
            STRATA_INKLING_K256=str(int(expected[0])),STRATA_INKLING_V4=str(int(expected[1])),STRATA_INKLING_SUFFIX_H2='0',
            IX_ARENA_MIB=str(a.arena),IX_STATE=str(d/'state.bin'),RECON_DUMP_FROM=str(n),RECON_DUMP_TAPS=str(d/'tap.dft1'))
        req=d/'request.json'
        put(req,dict(cmd=[str(a.binary.resolve()),*args], prompt=str(prompt), snapshot=str(a.snapshot.resolve()),base=n,steps=a.steps))
        vram=guarded(d/'target-guard',env,[sys.executable,str(Path(__file__).resolve()),'worker',str(req)])
        text=(d/'target.log').read_text()
        start,end=[s.split() for s in tagged(text,'IX_PLACEMENT')]
        assert start[0]=='start' and end[0]=='exit' and start[1:]==end[1:] and int(start[1])>0
        if results:
            assert start[2]==results['dense']['placement'], 'not the dense arm physical placement'
        [policy]=tagged(text,'IX_POLICY')
        k,v=map(int,re.fullmatch(r'k256=(\d+) v4=(\d+)',policy).groups())
        assert (bool(k),bool(v))==expected, 'the requested policy did not actually dispatch'
        result=json.loads((d/'wire.json').read_text())
        if k:
            assert k==8*(len(result['ids'])-1), 'not exactly all W1 global visits'
        tap_sha=checked_trace(d,suffix,result)
        state=d/'state.bin'
        size=state.stat().st_size
        native=size if not results else results['dense']['state_bytes']
        assert (size>native) == expected[1]
        ids=[*base,*suffix,*result['ids'][:-1]]
        result.update(placement=start[2], slots=int(start[1]), k256_dispatches=k, v4_dispatches=v,
                      state_bytes=size, state_sha256=sha(state), native_state_prefix_sha256=sha(state,native),
                      committed_ids_sha256=hashlib.sha256(struct.pack(f'<{len(ids)}i',*ids)).hexdigest(),
                      tap_sha256=tap_sha,target_free_min_mib=vram)
        put(d/'target-result.json',result)
        result['replay_free_min_mib']=guarded(d/'replay-guard',{'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']},
            [str(a.replay.resolve()),'--target',target,'--model',str(a.draft.resolve()),'--replay',str(d/'tap.dft1'),
             '--mode','both','--dump-json',str(d/'draft.json'),'--dump-steps',str(d/'steps.jsonl')])
        result['acceptance']=checked_replay(d,result)
        if results:
            result['vs_dense']=compare(out/'dense',d,results['dense'],result)
        results[mode]=result
        put(out/'result.json',results)
        print(mode,result['acceptance'],result.get('vs_dense'),flush=True)


if __name__=='__main__':
    main()
