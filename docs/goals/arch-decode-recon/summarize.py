"""Public summaries only of completed, guarded runs. Raw IDs and journals stay private."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(p):
    return json.loads(p.read_bytes())


def row(run, guard):
    m, c = read(run/'metrics.json'), read(run/'config.json')
    done = m['done'].split()
    if read(guard/'stop-reasons.json') or done[5] != 'length':
        raise ValueError(f'{run.name}: incomplete or guard stop')
    free = [json.loads(s)['free_mib'] for s in (guard/'vram.jsonl').read_text().splitlines()]
    load = re.search(r'(\d+) of (\d+) routed experts in VRAM', (run/'engine.log').read_text())
    assert min(free)>=1536 and load and done[6:8]==['0','0']
    ids = m['greedy_ids']
    digest = hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()
    return dict(run=run.name,guard=guard.name,context=m['length'],workload=c['workload'],
                prompt_sha256=c['prompt_sha256'],binary_sha256=c['binary_sha256'],seed_sha256=c['seed_sha256'],
                decode_steps=m['decode_steps'],ids_sha256=digest,prompt_ms=float(done[3]),
                engine_decode_ms=float(done[4]),arrival_span_ms=sum(m['step_ms']),
                decode_t_s=1000*m['decode_steps']/sum(m['step_ms']),arrival_ms=m['arrival_ms'],
                request_expert_hits=int(done[9]),request_expert_lookups=int(done[10]),
                resident_at_load=int(load[1]),total_experts=int(load[2]),min_free_mib=min(free),
                memory_warnings=len(read(guard/'warnings.json')),stop_reasons=[])


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work',type=Path)
    p.add_argument('pairs',nargs='+',help='run-name:guard-name')
    a = p.parse_args()
    rows = [row(*(a.work/s for s in pair.split(':'))) for pair in a.pairs]
    print(json.dumps(rows,indent=2))
