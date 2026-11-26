#!/usr/bin/env python3
"""Compare native JSONL steps with independent CPU replay_matrix or capture_states NPZs."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(native_path, matrix_path, states_dir, domain, profile):
    rows = [json.loads(line) for line in native_path.read_text().splitlines()]
    report = {}
    with np.load(matrix_path) as cpu:
        positions = {int(p): i for i, p in enumerate(cpu['positions'])}
        for mode in sorted({r['mode'] for r in rows}):
            selected = [r for r in rows if r['mode'] == mode]
            indices = [positions[r['start']] for r in selected]
            refs = cpu['draft_tokens'][indices]
            actual = np.array([r['draft_tokens'] for r in selected])
            differences = np.argwhere(actual != refs)
            mismatch = None
            if len(differences):
                row, col = differences[0]
                mismatch = {'start': selected[row]['start'], 'offset': int(col) + 1,
                            'cpu': int(refs[row, col]), 'hip': int(actual[row, col])}
            expected_top = cpu['candidate_ids'][indices]
            observed_top = np.array([r['cand'] for r in selected])
            overlap = [len(set(a) & set(b)) for row_a, row_b in zip(expected_top, observed_top, strict=True)
                       for a, b in zip(row_a, row_b, strict=True)]
            report[mode] = {'steps': len(selected), 'matching_ids': int(np.count_nonzero(actual == refs)),
                            'offered_ids': int(actual.size), 'first_difference': mismatch,
                            'top16_overlap': float(np.mean(overlap)) / 16.0}
            if states_dir:
                samples = {}
                for r in selected:
                    path = states_dir / f'{domain}-{r["start"]}-{profile}.npz'
                    if not path.is_file() or 'h_draft' not in r:
                        continue
                    with np.load(path) as state:
                        expected = state['h_draft'].astype(np.float64)
                        got = np.asarray(r['h_draft'], dtype=np.float64).reshape(expected.shape)
                        raw_error, score_error = [], []
                        for j, (cand_ids, logits, scores) in enumerate(zip(r['cand'], r['unary'], r['sel_scores'],
                                                                          strict=True)):
                            oracle = dict(zip(state['cand'][j], zip(state['unary'][j], state['sel_scores'][j],
                                                                    strict=True), strict=True))
                            same_prev = j == 0 or r['draft_tokens'][j - 1] == state['draft_tokens'][j - 1]
                            for token, logit, score in zip(cand_ids, logits, scores, strict=True):
                                if token not in oracle:
                                    continue
                                cpu_logit, cpu_score = oracle[token]
                                raw_error.append(logit - float(cpu_logit))
                                if same_prev:
                                    score_error.append(score - float(cpu_score))
                        samples[r['start']] = {
                            'relative_hidden_rms': float(np.linalg.norm(got - expected) / np.linalg.norm(expected)),
                            'unary_matched_candidates_mae': float(np.mean(np.abs(raw_error))) if raw_error else None,
                            'score_matched_candidates_and_predecessor_mae':
                                float(np.mean(np.abs(score_error))) if score_error else None,
                        }
                report[mode]['states'] = samples
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--cpu-matrix', type=Path, required=True)
    parser.add_argument('--states', type=Path)
    parser.add_argument('--domain', default='code_python')
    parser.add_argument('--profile', default='BF16')
    args = parser.parse_args()
    print(json.dumps(compare(args.native, args.cpu_matrix, args.states, args.domain, args.profile), indent=2))


if __name__ == '__main__':
    main()
