"""One guarded Tideglass forced run. Alternate arms externally; each seed is trained on its own GGUF."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from tools.strata_tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'exe', 'model', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--length', type=int, choices=[2048, 16384, 65536, 197000], required=True)
    p.add_argument('--history', type=Path, help='original comma-separated prompt IDs for long contexts')
    p.add_argument('--seed-profile', type=Path, help='omit only on a model-specific warm-up')
    args = p.parse_args()
    cfg = json.loads(args.config.read_bytes())
    flags = cfg['args'].copy()
    flags.remove('--disk-sessions')
    n, m = args.length, (256 if args.length == 2048 else 128)
    if args.history:
        ids = list(map(int, args.history.read_text().strip().split(',')))
    else:
        source = subprocess.check_output(['git', 'show', '35c899cb:src/poly/main.cpp']).decode()
        ids = Tokenizer.from_dir(Path(cfg['tokenizer'])).encode(source, parse_special=False)
    if len(ids) < n+m:
        raise ValueError(f'need {n+m} token IDs, got {len(ids)}')
    prefix, follow = (','.join(map(str, part)) for part in (ids[:n], ids[n:n+m]))
    out = args.output.resolve()
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    profile = out / 'profile'
    if args.seed_profile:
        shutil.copyfile(args.seed_profile, profile)
    flags[flags.index('--model')+1] = str(args.model.resolve(strict=True))
    flags[flags.index('--expert-profile')+1] = str(profile)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'HIP_VISIBLE', 'ROCR_VISIBLE',
                                                                     'HSA_OVERRIDE', 'GGML_', 'LD_PRELOAD'))}
    env.update(cfg['env'], LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']), STRATA_TOKEN_TIMING='1')
    cmd = [sys.executable, 'rdna2-rnd/scripts/run_guarded.py', '--warn-amdgpu-faults', '--output', str(out/'guard'),
           '--', '/usr/bin/time', '-f', 'peak_rss_kib=%M', 'timeout', '--foreground', '3600', str(args.exe.resolve()),
           *flags, '--tokens', '-', '--follow', follow, '--temperature', '0', '--max-new', str(m)]
    subprocess.run(cmd, input=prefix, env=env, text=True, check=True)
    log = (out/'guard/command.log').read_text()
    if (re.findall(r'^\d+(?:,\d+)+$', log, re.M) != [follow] or
            json.loads((out/'guard/stop-reasons.json').read_bytes())):
        raise ValueError('forced IDs or guard failed')
    metrics = re.search(fr'^prompt {n} tokens ([\d.]+) tok/s, generated {m} tokens ([\d.]+) tok/s \(length\); '
                        r'experts from VRAM ([\d.]+)%, CPU experts ([\d.]+) ms/token', log, re.M)
    if not metrics:
        raise ValueError('unexpected engine metrics or token counts')
    times = re.search(r'token timing: first ([\d.]+) ms, interval p50 ([\d.]+) ms, p90 ([\d.]+) ms', log)
    result = dict(zip(('prefill_tps', 'decode_tps', 'gpu_hit_percent', 'cpu_expert_ms_per_token', 'ttft_ms',
                       'p50_interval_ms', 'p90_interval_ms'), map(float, (*metrics.groups(), *times.groups()))))
    samples = [json.loads(s)['free_mib'] for s in (out/'guard/vram.jsonl').read_text().splitlines()]
    total = int(Path('/sys/class/drm/card1/device/mem_info_vram_total').read_text()) / 2**20
    result.update(model=args.model.name, exe_sha256=sha256(args.exe.read_bytes()).hexdigest(),
                  prompt_tokens=n, forced_tokens=m, prompt_sha256=sha256(prefix.encode()).hexdigest(),
                  forced_sha256=sha256(follow.encode()).hexdigest(), peak_device_used_mib=total-min(samples),
                  peak_rss_kib=int(re.search(r'^peak_rss_kib=(\d+)$', log, re.M)[1]),
                  resident_experts_at_load=int(re.search(r'(\d+) of 3072 routed experts in VRAM', log)[1]),
                  profile_seed_sha256=sha256(args.seed_profile.read_bytes()).hexdigest() if args.seed_profile else None)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result)


if __name__ == '__main__':
    main()
