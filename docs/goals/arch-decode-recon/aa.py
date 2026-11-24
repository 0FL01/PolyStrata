"""Quiet 2K live A/A: run only when the GPU queue is yours and the host is idle."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LENGTH, STEPS = 2048, 256
ORDER = ('AB', 'BA', 'BA', 'AB')


def read_text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def sensors():
    # Same DRM card as run_guarded.py. Keep sysfs values in their native units.
    gpu = Path('/sys/class/drm/card1/device')
    hwmon = Path('/sys/class/hwmon').glob('hwmon*')
    cpu_temp = [p for h in hwmon if read_text(h/'name') in ('k10temp', 'zenpower', 'coretemp')
                for p in h.glob('temp*_input')]
    return [Path('/proc/loadavg'), *(Path('/proc/pressure')/k for k in ('cpu', 'io', 'memory')),
            *sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')),
            *sorted(cpu_temp), *sorted(gpu.glob('hwmon/hwmon*/temp*_input')),
            *(gpu/k for k in ('pp_dpm_sclk', 'pp_dpm_mclk', 'gpu_busy_percent'))]


def guarded_leg(command, telemetry, paths):
    def sample(phase):
        log.write(json.dumps(dict(phase=phase, unix_ns=time.time_ns(),
                                  monotonic_ns=time.monotonic_ns(),
                                  readings={str(p): read_text(p) for p in paths}))+'\n')
        log.flush()

    with telemetry.open('x') as log:
        sample('before')
        with subprocess.Popen(command) as proc:
            while True:
                try:
                    code = proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    sample('during')
        sample('after')
    if code:
        raise subprocess.CalledProcessError(code, command)


def command(a, name):
    return [sys.executable, str(ROOT/'rdna2-rnd/scripts/run_guarded.py'),
            '--warn-amdgpu-faults', '--output', str(a.run/f'guard-{name}'), '--',
            'timeout', '1500', sys.executable, str(HERE/'baseline_run.py'), str(a.run/name),
            str(LENGTH), '--steps', str(STEPS), '--binary', str(a.binary), '--config', str(a.config),
            '--profile-seed', str(a.run/'profile.seed'), '--prompt-csv', str(a.run/'prompt.csv')]


def read_json(path):
    return json.loads(path.read_bytes())


def load_leg(run, name, hashes):
    if read_json(run/f'guard-{name}'/'stop-reasons.json') != []:
        raise ValueError(f'{name}: guard stop')
    c, m = (read_json(run/name/f) for f in ('config.json', 'metrics.json'))
    if c['trace'] is not False or c['profiled'] is not False or any(c[k] != v for k, v in hashes.items()):
        raise ValueError(f'{name}: traced or inputs changed')
    ts, ids = m['arrivals_ns'], m['greedy_ids']
    rate = m['decode_t_s']
    if (m['length'] != LENGTH or m['decode_steps'] != STEPS or len(ids) != len(ts) or
            len(ts) != STEPS+1 or ts[-1] <= ts[0] or
            not math.isclose(rate, 1e9*STEPS/(ts[-1]-ts[0]))):
        raise ValueError(f'{name}: invalid decode wall window')
    return dict(wall_ms=(ts[-1]-ts[0])/1e6, decode_t_s=rate, arrival_ms=m['arrival_ms']), ids


def paired(a, b, order):
    a_metrics, a_ids = a
    b_metrics, b_ids = b
    a_rate, b_rate = a_metrics['decode_t_s'], b_metrics['decode_t_s']
    first = next((dict(token_index_1based=i+1, output_position_0based=LENGTH+i, a=x, b=y)
                  for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y), None)
    return dict(order=order, a=a_metrics, b=b_metrics, b_minus_a_ms=b_metrics['wall_ms']-a_metrics['wall_ms'],
                b_minus_a_t_s=b_rate-a_rate, b_vs_a_pct=100*(b_rate/a_rate-1), first_greedy_difference=first)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='new private output directory')
    p.add_argument('--profile-seed', type=Path, required=True)
    p.add_argument('--prompt-csv', type=Path, required=True)
    p.add_argument('--binary', type=Path, default=ROOT/'build-poly-hip/strata-poly')
    p.add_argument('--config', type=Path, default=Path.home()/'.config/strata/inkling2-mini/strata-inkling-2-mini.json')
    p.add_argument('--dry-run', action='store_true', help='validate inputs and print the plan without running or writing')
    a = p.parse_args()
    a.run, a.binary, a.config = (path.resolve() for path in (a.run, a.binary, a.config))
    if a.run.exists() or not a.binary.is_file() or not os.access(a.binary, os.X_OK):
        p.error('need a new output directory and an executable engine binary')
    seed, csv = a.profile_seed.read_bytes(), a.prompt_csv.read_bytes()
    ids = csv.split(b',')
    if not seed or len(ids) != LENGTH or not all(t.isdigit() for t in ids):
        p.error('need a nonempty profile seed and exactly 2048 comma-separated token IDs (no newline)')
    config = a.config.read_bytes()
    cfg = json.loads(config)
    args = cfg.get('args', [])
    if ('STRATA_FREEZE_EXPERTS' in cfg.get('env', {}) or
            any(int(args[i+1]) <= 0 for i, flag in enumerate(args[:-1]) if flag == '--expert-swaps')):
        p.error('live A/A requires expert swaps and no freeze setting')
    hashes = {k+'_sha256': hashlib.sha256(v).hexdigest() for k, v in
              (('seed', seed), ('prompt', csv), ('config', config), ('binary', a.binary.read_bytes()))}
    pairs = [{label: f'p{i}-{label.lower()}' for label in order} for i, order in enumerate(ORDER, 1)]
    plan = {name: command(a, name) for name in ('warmup', *(n for pair in pairs for n in pair.values()))}
    if a.dry_run:
        print(json.dumps(dict(inputs=hashes, commands=plan), indent=2))
        return

    os.umask(0o077)
    a.run.mkdir(parents=True, exist_ok=False)
    for name, data in (('profile.seed', seed), ('prompt.csv', csv)):
        path = a.run/name
        path.write_bytes(data)
        path.chmod(0o400)
    paths = sensors()

    def leg(name):
        guarded_leg(plan[name], a.run/f'{name}-telemetry.jsonl', paths)
        return load_leg(a.run, name, hashes)

    warmup, _ = leg('warmup')
    results = []
    for order, pair in zip(ORDER, pairs):
        legs = {label: leg(name) for label, name in pair.items()}
        results.append(paired(legs['A'], legs['B'], order))
        deltas = [r['b_vs_a_pct'] for r in results]
        summary = dict(inputs=hashes, warmup=warmup, pairs=results, complete=len(results) == len(ORDER),
                       median_paired_delta_pct=statistics.median(deltas),
                       max_abs_paired_delta_pct=max(map(abs, deltas)))
        tmp = a.run/'summary.tmp'
        tmp.write_text(json.dumps(summary, indent=2)+'\n')
        tmp.replace(a.run/'summary.json')


if __name__ == '__main__':
    main()
