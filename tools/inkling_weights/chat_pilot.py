"""Run the frozen Inkling chat cases with the pinned standalone CLI and GPU guard."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from serve.frontend import INKLING_STOP, ChatTemplate, InklingOutputParser, literal_tags, mark_think_literals
from tools.strata_tokenizer import Tokenizer

from .validate_tasks import MANIFEST, validate


ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home() / '.cache/opencode-tmp/opencode/inkling-weights'
LOCK = MANIFEST.with_name('experiment-lock.json')


def checked_bytes(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f'stale input hash: {path}')
    return raw


def write_json(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2, ensure_ascii=False)
        out.write('\n')


def answer_only(tokenizer, ids):
    """Use the shipped parser; never execute code or route tool calls from the output."""
    raw = tokenizer.decode(ids)
    parser = InklingOutputParser()
    events = parser.feed(raw) + parser.finish()
    return raw, ''.join(event.text for event in events if event.kind == 'content')


def run_case(case, model, directory, cli, fixed_env, seed_bytes, lock, tok, template, effort, stop):
    messages = [{'role': 'user', 'content': case['prompt']}]
    # The frozen corpus has no literal control markers. Fail, rather than interpret any as special IDs.
    _, _, changed = mark_think_literals(messages, None, literal_tags(tok.control_tokens))
    if changed:
        raise ValueError(f'{case["id"]}: literal control tag in frozen input')
    prompt = template.render(messages, reasoning_effort=effort)
    prefix = tok.encode(prompt, parse_special=True)
    vocab = lock['benchmark']['active_vocab']
    if not prefix or stop in prefix or any(not 0 <= t < vocab for t in prefix):
        raise ValueError(f'{case["id"]}: invalid rendered prompt IDs')
    directory.mkdir(mode=0o700)
    (directory / 'prompt.txt').write_text(prompt, encoding='utf-8')
    prefix_csv = ','.join(map(str, prefix))
    (directory / 'prompt.ids.csv').write_text(prefix_csv, encoding='ascii')
    profile = directory / 'profile.txt'
    profile.write_bytes(seed_bytes)
    flags = {'{model}': str(model), '{per_run_profile}': str(profile)}
    budget = case['max_new_tokens']
    command = [
        sys.executable, '-B', str(ROOT / 'rdna2-rnd/scripts/run_guarded.py'), '--output', 'guard',
        '--warn-amdgpu-faults', '--',
        'timeout', '--foreground', '--kill-after=10s', f'{lock["benchmark"]["per_standalone_run_timeout_seconds"]}s',
        'sh', '-c', 'exec "$@" > output.ids.csv 2> stderr.log', 'sh', str(cli),
        *(flags.get(arg, arg) for arg in lock['standalone']['engine_argv_before_workload']),
        *lock['standalone']['sampling_argv'], '--eos-ids', str(stop), '--tokens', prefix_csv, '--max-new', str(budget),
    ]
    subprocess.run(command, cwd=directory, env={**fixed_env, 'TMPDIR': str(directory)}, check=True)
    if json.loads((directory / 'guard/stop-reasons.json').read_bytes()) != []:
        raise RuntimeError(f'{case["id"]}: guard did not clear')
    text = (directory / 'output.ids.csv').read_text(encoding='ascii')
    if not re.fullmatch(r'\d+(?:,\d+)*\n', text):
        raise ValueError(f'{case["id"]}: invalid raw CLI output')
    ids = list(map(int, text.strip().split(',')))
    if not 1 <= len(ids) <= budget or any(not 0 <= t < vocab for t in ids) or stop in ids[:-1]:
        raise ValueError(f'{case["id"]}: invalid CLI completion IDs')
    finish = 'stop' if ids[-1] == stop else 'length'
    stderr = (directory / 'stderr.log').read_text(encoding='utf-8')
    metric = rf'^prompt {len(prefix)} tokens .*?, generated {len(ids)} tokens .*? \({finish}\);'
    if (finish == 'length' and len(ids) != budget) or not re.search(metric, stderr, re.M):
        raise ValueError(f'{case["id"]}: unexpected CLI termination')
    raw, answer = answer_only(tok, ids[:-1] if finish == 'stop' else ids)
    (directory / 'raw.txt').write_text(raw, encoding='utf-8')
    (directory / 'answer.txt').write_text(answer, encoding='utf-8')
    passed, reason = validate(case, answer)
    record = {'id': case['id'], 'tokens': len(ids), 'finish': finish, 'has_final_answer': bool(answer.strip()),
              'pass': passed, 'reason': reason, 'smoke_candidate': case.get('smoke_candidate', False)}
    write_json(directory / 'result.json', record)
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='a new directory under the private experiment root')
    parser.add_argument('--suite', choices=('tasks', 'orca'), default='tasks')
    select = parser.add_mutually_exclusive_group()
    select.add_argument('--case', action='append', help='run only specified frozen case(s) instead of the full suite')
    select.add_argument('--smoke', action='store_true', help='run only the four H0-passing critical tasks')
    args = parser.parse_args()
    os.umask(0o077)
    lock = json.loads(LOCK.read_bytes())
    corpus = json.loads(checked_bytes(MANIFEST, lock['corpus_sha256']))
    for file, digest in corpus['tokenizer']['files_sha256'].items():
        checked_bytes(ROOT / file, digest)
    cases = corpus['tasks'] if args.suite == 'tasks' else [c for group in corpus['orca']['cases'].values() for c in group]
    if args.smoke:
        if args.suite != 'tasks':
            parser.error('--smoke applies only to the task suite')
        cases = [c for c in cases if c.get('smoke_candidate', False)]
    if args.case:
        cases = [c for c in cases if c['id'] in args.case]
        if {c['id'] for c in cases} != set(args.case):
            parser.error('unknown case for suite')
    effort = lock['standalone']['quality_chat_reasoning_effort'] if args.suite == 'tasks' else corpus['orca']['reasoning_effort']
    cli = Path(lock['release_binary']['path'])
    checked_bytes(cli, lock['release_binary']['sha256'])
    fixed_env = {'HOME': str(Path.home()), 'PATH': os.environ['PATH'], **lock['engine_env']}
    seed = checked_bytes(Path(lock['standalone']['profile_seed_path']), lock['standalone']['profile_seed_sha256'])
    template_path = Path(lock['production']['config']).parent / 'tokenizer/chat_template.jinja'
    checked_bytes(template_path, lock['production']['template_sha256'])
    template = ChatTemplate(template_path)
    tok = Tokenizer.from_gguf(lock['sources']['H0']['local_path'])
    stop = tok.encode(INKLING_STOP, parse_special=True)
    if tok.pre != 'inkling2' or len(stop) != 1 or stop != corpus['tokenizer']['cli_and_server_stop_ids']:
        raise ValueError('unexpected Inkling tokenizer')
    out = args.output.resolve()
    if not out.is_relative_to(WORK.resolve()):
        parser.error('--output must be under the experiment root')
    model = args.model.resolve(strict=True)
    if not model.is_file():
        parser.error('--model must be a file')
    active = subprocess.run(['systemctl', '--user', 'show', '-p', 'ActiveState', '--value', lock['production']['unit']],
                            check=True, capture_output=True, text=True).stdout.strip()
    if active != 'inactive':
        raise RuntimeError('production must be inactive for standalone runs')
    out.mkdir(mode=0o700)
    records = [run_case(case, model, out / case['id'], cli, fixed_env, seed, lock, tok, template, effort, stop[0])
               for case in cases]
    write_json(out / 'chat-result.json', {'suite': args.suite, 'effort': effort, 'model': str(model),
                                         'template_sha256': lock['production']['template_sha256'],
                                         'corpus_sha256': lock['corpus_sha256'], 'cases': records})


if __name__ == '__main__':
    main()
