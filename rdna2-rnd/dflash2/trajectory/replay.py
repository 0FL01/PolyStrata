#!/usr/bin/env python3
"""Compare native K=7 and aligned DSpark on the private, genuine billing trajectory."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

from run import CONFIG, CONTEXT, DSPARK, EOS, PROFILE, R, ROOT, TOKENIZER, base_env, dump, replace_opt, sha


K = 7
PROFILES = ('BF16', 'Q8_0', 'Q4_0', 'Q4_0-mixed', 'Q4_K-mixed')
SOURCE = R / 'trajectory/taps-natural-42'


def trace_rows(entry):
    f = Path(entry['tap_file'])
    raw = f.read_bytes()
    record = 16 + 4 * 2304 * 4
    assert sha(f) == entry['tap_sha256'] and raw[:4] == b'DFT1' and (len(raw) - 4) % record == 0
    expected = entry['prompt_ids'] + entry['tokens'][:-1]
    n_prompt = entry['prompt_len']
    assert [struct.unpack_from('<qii', raw, at) for at in range(4, len(raw), record)] == [
        (i, tok, int(i >= n_prompt)) for i, tok in enumerate(expected)]
    assert len(expected) == entry['tap_rows'] and entry['tokens'][-1] == EOS
    return len(expected)


def phases(output):
    sys.path.insert(0, str(ROOT))
    from serve.frontend import INKLING_END, INKLING_INVOKE, INKLING_MODEL, INKLING_TEXT, INKLING_THINK
    from tools.strata_tokenizer import Tokenizer
    tok = Tokenizer.from_dir(TOKENIZER)
    pairs = {INKLING_THINK: 'reasoning', INKLING_TEXT: 'answer', INKLING_INVOKE: 'tool',
             INKLING_MODEL: 'framing', INKLING_END: 'framing'}
    phase_by_id = {tuple(tok.encode(t, parse_special=True))[0]: phase for t, phase in pairs.items()}
    phase_by_id[EOS] = 'framing'
    result, state = [], 'framing'
    for t in output:
        if t in phase_by_id:
            state = phase_by_id[t]
            result.append('framing')
        else:
            result.append(state)
    return result


def replay_row(start, anchor, guesses, entry):
    prompt = entry['prompt_len']
    offset = start - prompt
    assert 0 <= offset < entry['tap_rows'] - prompt - K and anchor == entry['tokens'][offset]
    assert len(guesses) == K
    truth = entry['tokens'][offset+1:offset+1+K]
    return start, next((i for i, (g, t) in enumerate(zip(guesses, truth)) if g != t), K)


def hist(rows):
    if not rows:
        return {'anchors': 0, 'mean_accepted_draft': None, 'mean_emitted_including_bonus': None}
    n = len(rows)
    counts = Counter(acc for _, acc in rows)
    mean = sum(acc * count for acc, count in counts.items()) / n
    return dict(anchors=n, mean_accepted_draft=mean, mean_emitted_including_bonus=1+mean,
                accepted_prefix_histogram=[counts[i] for i in range(K+1)],
                prefix_survival=[sum(counts[i] for i in range(k, K+1))/n for k in range(1, K+1)])


def roll(rows):
    end = -1
    for p, a in rows:
        if p >= end:
            yield p, a
            end = p + a + 1


def compare_phases(rows, phase, prompt):
    return {name: hist([row for row in rows if phase[row[0] - prompt + 1] == name])
            for name in ('reasoning', 'tool', 'answer', 'framing')}


def launch(directory, cmd, changes):
    dump(directory / 'launch.json', dict(argv=cmd, environment=changes))
    with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
        subprocess.run(cmd, env=dict(base_env(), **changes), cwd=directory, stdout=stdout, stderr=stderr, check=True)


def native(directory, model, entry, cfg, replay):
    directory.mkdir()
    details, summary = directory / 'steps.jsonl', directory / 'summary.json'
    target = cfg['args'][cfg['args'].index('--model')+1]
    launch(directory, [replay, '--target', target, '--model', model, '--replay', entry['tap_file'], '--mode', 'both',
                       '--dump-json', str(summary), '--dump-steps', str(details)],
           dict(LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']), LC_ALL='C'))
    all_rows, closed_rows = [], []
    for line in details.read_text().splitlines():
        row = json.loads(line)
        name = row['mode']
        assert name in ('all-positions', 'closed-loop')
        found = replay_row(row['start'], row['anchor'], row['draft_tokens'], entry)
        assert found[1] == row['accepted']
        (all_rows if name == 'all-positions' else closed_rows).append(found)
    s, = json.loads(summary.read_text())
    assert s['n_total'] == entry['tap_rows'] and s['n_prompt'] == entry['prompt_len']
    for name, rows in [('all_positions', all_rows), ('closed_loop', closed_rows)]:
        assert s[name]['steps'] == len(rows) and s[name]['acc_histogram_0_to_7'] == hist(rows)['accepted_prefix_histogram']
    p = entry['prompt_len']
    assert [x for x, _ in all_rows] == list(range(p, entry['tap_rows']-K))
    assert closed_rows[0][0] == p and all(b[0] == a[0]+a[1]+1 for a,b in zip(closed_rows,closed_rows[1:]))
    assert sum(closed_rows[-1]) + 1 >= entry['tap_rows'] - K
    return all_rows, closed_rows


def dspark(directory, entry, cfg):
    directory.mkdir()
    profile = directory / 'profile'
    shutil.copyfile(PROFILE, profile)
    args = list(cfg['args'])
    args.remove('--disk-sessions')
    for name, value in {'--expert-profile': profile, '--max-context': CONTEXT, '--prompt-batch': 256}.items():
        replace_opt(args, name, value)
    prompt, expected = entry['prompt_ids'], entry['tokens']
    args += ['--prompt-cache', '0', '--lookup', '0', '--eos-ids', str(EOS), '--temperature', '0',
             '--max-new', str(len(expected)), '--tokens', ','.join(map(str, prompt)), '--follow', ','.join(map(str, expected)),
             '--draft-model', str(DSPARK), '--window', '8', '--draft-max', str(K)]
    log = directory / 'proposals.log'
    env = dict(cfg['env'], LD_LIBRARY_PATH=':'.join(cfg['lib_dirs']), LC_ALL='C',
               STRATA_INKLING_DSPARK_REUSE_V='1', STRATA_DRAFT_LOG=str(log))
    launch(directory, [cfg['exe'], *args], env)
    raw = (directory / 'stdout.log').read_text().strip()
    assert re.fullmatch(r'\d+(?:,\d+)*', raw) and list(map(int, raw.split(','))) == expected
    all_rows = []
    last = len(prompt) - 1
    stop = entry['tap_rows'] - K
    for line in log.read_text().splitlines():
        start, anchor, *guesses = line.split()
        start, anchor = int(start), int(anchor)
        assert start == last+1 and anchor == expected[start-len(prompt)]
        last = start
        if start < stop:
            all_rows.append(replay_row(start, anchor, [int(g.split(':', 1)[0]) for g in guesses], entry))
    assert [x for x, _ in all_rows] == list(range(len(prompt), stop))
    return all_rows, list(roll(all_rows))


def worker(out, idx):
    plan = json.loads((out / 'plan.json').read_text())
    entry = json.loads((SOURCE / 'manifest.json').read_text())[f'billing_turn_{idx}']
    available = trace_rows(entry) - entry['prompt_len'] - K
    here = out / f'turn-{idx}'
    here.mkdir()
    result = dict(turn=idx, kind='tool-read' if idx < 3 else 'reasoning-and-answer',
                  prompt_rows=entry['prompt_len'], captured_output_rows=entry['gen_len']-1,
                  eligible_full_width_anchors=max(0, available), unconsumed_eos_excluded=True)
    if available <= 0:
        result.update(status='skipped', reason='no full seven-proposal window inside recorded DFT1 rows')
    else:
        phase = phases(entry['tokens'])
        assert Counter(phase) == Counter(plan['phase_counts'][str(idx)]), 'capture phase counts changed'
        result.update(status='complete', methods={})
        for name, weights in {**plan['native_models'], 'DSpark': plan['dspark']}.items():
            if name == 'DSpark':
                a, b = dspark(here / name, entry, plan['config'])
            else:
                a, b = native(here / name, weights, entry, plan['config'], plan['replay'])
            result['methods'][name] = dict(all_positions=hist(a), closed_loop=hist(b),
                                            all_positions_by_next_id_phase=compare_phases(a, phase, entry['prompt_len']),
                                            closed_walk_source='forced W1 proposal tape' if name == 'DSpark' else 'native replay')
    dump(here / 'result.json', result)


def combine(result, indices, names):
    pooled = {}
    for name in names:
        pooled[name] = {}
        for mode in ('all_positions', 'closed_loop'):
            counts = [0] * (K+1)
            for r in result:
                if r['turn'] in indices and r['status'] == 'complete':
                    for i, n in enumerate(r['methods'][name][mode]['accepted_prefix_histogram']):
                        counts[i] += n
            rows = [(0, i) for i, n in enumerate(counts) for _ in range(n)]
            pooled[name][mode] = hist(rows)
    return pooled


def main():
    os.umask(0o077)
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]), int(sys.argv[3]))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    a = p.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]+', a.label)
    replay = R / 'dflash2-replay'
    capture = json.loads((SOURCE / 'trajectory.json').read_text())
    assert capture['grade']['pass'] and len(capture['tools']) == 2
    assert all(t['case_id'] for t in capture['tools'])
    for idx, tool in enumerate(capture['tools'], start=1):
        msgs = capture['turns'][idx]['messages']
        assert json.loads(msgs[-1]['content']) == tool['result'] and msgs[-1]['role'] == 'tool'
        assert msgs[-1]['tool_call_id'] == msgs[-2]['tool_calls'][0]['id']
    models = {q: DSPARK.parent / f'DFlash2/Inkling-2-Mini-DFlash2-{q}.gguf' for q in PROFILES}
    out = R / 'trajectory' / a.label
    report = R / f'report-{a.label}.json'
    assert not report.exists()
    out.mkdir()
    plan = dict(replay=str(replay), replay_sha256=sha(replay), library_sha256=sha(R / 'libstrata-dflash2.so'),
                manifest_sha256=sha(SOURCE / 'manifest.json'), capture_sha256=sha(SOURCE / 'trajectory.json'),
                native_models={k:str(v) for k,v in models.items()}, dspark=str(DSPARK),
                weight_sha256={k:sha(v) for k,v in {**models,'DSpark':DSPARK}.items()},
                config=json.loads(CONFIG.read_text()), profile_sha256=sha(PROFILE),
                phase_counts={str(t['turn']):{k:len(v) for k,v in t['ids_by_phase'].items()} for t in capture['turns']})
    dump(out / 'plan.json', plan)
    turns = []
    for i in (1, 2, 3):
        with (R / 'gpu.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert subprocess.check_output(['systemctl', '--user', 'show', 'strata.service', '-p', 'ActiveState', '--value'], text=True).strip() == 'inactive'
            subprocess.run([sys.executable, str(ROOT / 'rdna2-rnd/scripts/run_guarded.py'), '--warn-amdgpu-faults',
                            '--output', str(out / f'guard-{i}'), '--', 'timeout', '--kill-after=15s', '900',
                            sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(out), str(i)],
                           cwd=ROOT, env=base_env(), check=True)
        turns.append(json.loads((out / f'turn-{i}/result.json').read_text()))
        names = (*PROFILES, 'DSpark')
        dump(report, dict(k=K, target_temperature=1, replay_only=True, turns=turns,
                          tool_turns=combine(turns, {1, 2}, names), final_turn=combine(turns, {3}, names)))
    print(f'Complete: {report.name}; full-width anchors per turn {[t["eligible_full_width_anchors"] for t in turns]}')


if __name__ == '__main__':
    main()
