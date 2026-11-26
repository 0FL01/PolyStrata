#!/usr/bin/env python3
"""One guarded cold-prompt leg. No service changes, no reuse of a writer's output.

Config is an installed strata-poly JSON (exe, args, lib_dirs, env). Manifest is
the private DFT1 capture index (prompt_ids and the serial generated tokens).
--follow is used ONLY when collecting aligned DSpark proposal IDs, never for E2E.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run','out','config','profile'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--domain', required=True)
    p.add_argument('--binary', type=Path)
    p.add_argument('--mode', choices=['production','dspark','dspark-log','live'], required=True)
    p.add_argument('--weights', type=Path)
    p.add_argument('--dspark-weights', type=Path, help='optional second resident draft, for dual-source live pilot')
    p.add_argument('--library', type=Path)
    p.add_argument('--steps', type=int, default=65)
    p.add_argument('--k', type=int, choices=range(8), default=7)
    p.add_argument('--serial-verify', action='store_true')
    p.add_argument('--adaptive', action='store_true',help='live DFlash2/W1/lookup cost pilot')
    p.add_argument('--adaptive-budget', type=float, help='estimated maximum work per speculative round (ms); not an SLA')
    p.add_argument('--wave8', action='store_true', help='opt in to the exact Q4_0 wave8 projection experiment in the DSO')
    p.add_argument('--parity', action='store_true', help='live only: private full target-logit and committed-state diagnostic')
    p.add_argument('--warm-target',type=Path,help='restore a real target-only snapshot, with --suffix of >=2049 IDs')
    p.add_argument('--suffix',type=Path,help='replace manifest prompt with fresh token CSV')
    p.add_argument('--cap', type=int, help='fixed expert+KV arena MiB, freeze and disable swaps; otherwise production placement')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--temperature', type=float, default=0)
    a = p.parse_args()
    os.umask(0o077)
    cfg = json.loads(a.config.read_text())
    exe = (a.binary or Path(cfg['exe'])).resolve(strict=True)
    m = json.loads((a.run/'taps/manifest.json').read_text())[a.domain]
    if a.mode != 'production' and (not a.weights or not a.weights.is_file()): p.error('need --weights')
    if a.mode == 'live' and not (a.binary and a.library and a.library.is_file()): p.error('need --binary and --library')
    if a.adaptive_budget is not None and (a.mode != 'live' or not a.adaptive or a.serial_verify or
                                          not math.isfinite(a.adaptive_budget) or a.adaptive_budget <= 0):
        p.error('--adaptive-budget needs adaptive live batched mode and a positive finite number')
    if a.wave8 and a.mode != 'live': p.error('--wave8 is live only')
    if a.parity and a.mode != 'live': p.error('parity tape requires live probe')
    if a.dspark_weights and (a.mode!='live' or not a.dspark_weights.is_file()): p.error('dual mode needs live + valid DSpark GGUF')
    if a.warm_target and (not a.suffix or a.mode!='live' or a.dspark_weights): p.error('target-only restore needs live and suffix, without DSpark')
    if a.suffix and a.mode=='dspark-log': p.error('follow requires manifest trajectory')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args = list(cfg['args'])
    if '--disk-sessions' in args: args.remove('--disk-sessions')
    shutil.copyfile(a.profile, out/'profile')
    args[args.index('--expert-profile')+1] = str(out/'profile')
    # Reproduce the installed switches, not caller-specific profiling flags.
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DF2_', 'STRATA_', 'RECON_'))}
    env.update(cfg['env'], LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']), LC_ALL='C')
    if a.cap is not None:
        env.update(STRATA_INKLING_EXPERT_KV_MIB=str(a.cap), STRATA_FREEZE_EXPERTS='1')
        args[args.index('--expert-swaps')+1] = '0'
    prompt=','.join(map(str,m['prompt_ids'])) if not a.suffix else a.suffix.read_text().strip()
    args += ['--tokens',prompt, '--max-new',str(a.steps), '--seed',str(a.seed),
             '--eos-ids','200006','--lookup','0','--temperature',str(a.temperature),'--top-k','0','--top-p','1']
    if a.mode == 'live':
        args += ['--window','8']
        env.update(DF2_LIBRARY=str(a.library.resolve()), DF2_WEIGHTS=str(a.weights.resolve()), DF2_K=str(a.k),
                   DF2_SERIAL_VERIFY=str(int(a.serial_verify)))
        if a.parity: env['DF2_PARITY_TAPE']=str(out/'target.tape')
        if a.warm_target:
            env['DF2_WARM_TARGET']=str(a.warm_target.resolve(strict=True)); args+=['--prompt-cache','0']
        if a.adaptive: env['DF2_ADAPTIVE']='1'
        if a.adaptive_budget is not None: env['DF2_ADAPTIVE_BUDGET_MS']=str(a.adaptive_budget)
        if a.wave8: env['DF2_WAVE8']='1'
        if a.dspark_weights:
            args+=['--draft-model',str(a.dspark_weights.resolve())]
            env['STRATA_INKLING_DSPARK_REUSE_V']='1'
    elif a.mode.startswith('dspark'):
        args += ['--draft-model',str(a.weights),'--window','8','--draft-max',str(a.k)]
        env.update(STRATA_INKLING_DSPARK_REUSE_V='1')
        if a.mode == 'dspark-log':
            args += ['--follow',','.join(map(str,m['tokens']))]
            env.update(STRATA_DRAFT_LOG=str(out/'proposals.log'))
    else:
        args.append('--no-draft')
    guard = out/'guard'
    cmd = [sys.executable, str(ROOT/'rdna2-rnd/scripts/run_guarded.py'), '--warn-amdgpu-faults', '--output', str(guard),
           '--', 'timeout','--kill-after=10s','900', '/usr/bin/time','-f','PEAK_RSS_KiB %M', str(exe), *args]
    subprocess.run(['flock','-x',str(a.run/'gpu.lock'),*cmd],env=env,check=True,start_new_session=True)
    text = (guard/'command.log').read_text()
    seq, = re.findall(r'^([0-9]+(?:,[0-9]+)*)$', text, re.M)
    ids = list(map(int,seq.split(',')))
    if a.mode == 'dspark-log':
        # Reject the earlier false positive: a valid file path and a DONE line don't prove --follow was used.
        assert ids == m['tokens'][:a.steps], 'followed outputs do not equal the target trajectory'
        anchors = [list(map(int,line.split()[:2])) for line in (out/'proposals.log').read_text().splitlines()]
        expected = [[len(m['prompt_ids'])+i,tok] for i,tok in enumerate(ids[:-1]) if len(ids)-i >= 3]
        assert anchors == expected, 'logged proposals do not match serial anchor positions/IDs'
    vram = [json.loads(line)['free_mib'] for line in (guard/'vram.jsonl').read_text().splitlines()]
    speed = re.search(r'generated (\d+) tokens ([.\d]+) tok/s \((\w+)\)',text)
    report = dict(domain=a.domain, mode=a.mode, k=a.k, serial_verify=a.serial_verify, cap_mib=a.cap,
                  temperature=a.temperature, seed=a.seed, parity=a.parity, adaptive=a.adaptive,
                  dspark_weights=a.dspark_weights.name if a.dspark_weights else None,
                  library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest() if a.library else None,
                  config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
                  weights=a.weights.name if a.weights else None, binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  profile_seed_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
                  ids=ids, tps=float(speed[2]), finish=speed[3], n=len(ids), min_free_mib=min(vram),
                  peak_rss_kib=int(re.search(r'^PEAK_RSS_KiB (\d+)$',text,re.M)[1]),
                   slots=int(re.search(r'; (\d+) of \d+ routed experts in VRAM',text)[1]))
    assert int(speed[1])==len(ids)
    report['swaps']=int(re.search(r'(\d+) experts swapped into VRAM',text)[1])
    placement = re.search(r'^DF2_TARGET_PLACEMENT slots=(\d+) layout=([a-f0-9]{16}) swaps=(\d+)$', text, re.M)
    if placement:
        report['first_decode_placement'] = dict(slots=int(placement[1]), layout=placement[2], swaps=int(placement[3]))
    if a.mode == 'live':
        if a.warm_target:
            stamp=re.search(r'^DF2_RESTORED (\d+) fresh=(\d+)$',text,re.M)
            start=re.search(r'^DF2_ROUND (\d+) ',text,re.M)
            assert stamp and start and int(stamp[1])+int(stamp[2])==int(start[1]) and int(stamp[2])>=2049
            report.update(restored_target_rows=int(stamp[1]),fresh_prompt=int(stamp[2]))
        times = list(map(float,re.search(r'^DF2_TIMES (.*)$',text,re.M)[1].split()))
        assert len(times)==1+len(ids)
        intervals = [b-a for a,b in zip(times[1:],times[2:])]
        rounds = [list(map(float,s.split())) for s in re.findall(r'^DF2_ROUND (.*)$',text,re.M)]
        assert sum(r[3] for r in rounds)==len(ids)-1
        report.update(adaptive_budget_ms=a.adaptive_budget, wave8=a.wave8,
                      verify_widths=[sum(r[1] == k for r in rounds) for k in range(8)])
        if a.adaptive_budget is not None:
            budgets = re.findall(r'^DF2_BUDGET ([^\s]+)$', text, re.M)
            if len(budgets) != 1 or float(budgets[0]) != a.adaptive_budget:
                raise ValueError('missing or mismatched budget receipt from live ELF')
            report['over_budget_rounds'] = sum(r[-1] > a.adaptive_budget for r in rounds)
        drafts = [r[4] for r in rounds if r[1]]
        if a.adaptive:
            fields=re.search(r'^DF2_POLICY serial=(\d+) lookup=(\d+) dflash2=(\d+)(?: dspark=(\d+))?$',text,re.M)
            report['policy_rounds']=dict(zip(('serial','lookup','dflash2','dspark'),map(int,fields.groups(default='0'))))
            assert sum(report['policy_rounds'].values())==len(rounds)
            drafts=[dt for dt in drafts if dt>0]
        report.update(ttft_ms=times[0]+times[1], p90_ms=statistics.quantiles(intervals,n=10,method='inclusive')[8] if len(intervals)>1 else None,
                      offered=int(sum(r[1] for r in rounds)), accepted=int(sum(r[2] for r in rounds)), rounds=len(rounds),
                      mean_draft_ms=statistics.mean(drafts) if drafts else None,
                      mean_round_ms=statistics.mean(r[-1] for r in rounds) if rounds else None)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='ids'}),flush=True)


if __name__ == '__main__': main()
