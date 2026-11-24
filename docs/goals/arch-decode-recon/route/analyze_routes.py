#!/usr/bin/env python3
"""Offline P0/P1 screen for serial Inkling decode; see README.md for the data contract."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_VISITS, N_EXPERTS, K = 48, 128, 8
LAYERS = range(6, 18)
PAIRS = ((1, 2), (2, 3), (1, 3))
TIMINGS = (
    "visit_ns", "visit_host_enqueue_ns", "router_wait_ns", "gate_up_host_ns", "down_host_ns",
    "cpu_branch_ns", "gpu_expert_ns", "gpu_dispatch_gap_ns", "host_join_ns", "exposed_join_ns",
    "gpu_to_cpu_ready_marker_ns", "gpu_to_tail_gap_ns",
)


def physical(visit):
    return visit if visit < 6 else 6 + (visit - 6) % 12 if visit < 42 else visit - 24


def stage(visit):
    return 0 if visit < 6 else 1 + (visit - 6) // 12 if visit < 42 else 4


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass
class Trace:
    metadata: dict
    pos: np.ndarray
    ids: np.ndarray
    bits: np.ndarray
    cpu: np.ndarray
    resident: np.ndarray
    pending: np.ndarray
    ready_ns: np.ndarray
    timing: dict

    def __getitem__(self, key):
        return Trace(self.metadata, self.pos[key], self.ids[key], self.bits[key], self.cpu[key],
                     self.resident[key], self.pending[key], self.ready_ns[key], {k: v[key] for k, v in self.timing.items()})


def load(path):
    def integer(x, lo, hi):
        return type(x) is int and lo <= x < hi

    with path.open() as stream:
        meta = json.loads(next(stream, "{}"))
        check(meta.get("schema") == "inkling-route-v1", f"{path}: expected inkling-route-v1 header")
        check(meta.get("phase") == "serial_decode", f"{path}: only serial decode traces are accepted")
        rows = []
        for i, line in enumerate(stream):
            row = json.loads(line)
            at = f"{path}:{i+2}"
            v = i % N_VISITS
            check(row.get("visit") == v and row.get("layer") == physical(v) and row.get("pass") == stage(v),
                  f"{at}: bad visit/layer/pass or missing record")
            p = row.get("pos")
            check(integer(p, 0, 2**31), f"{at}: bad token position")
            if rows:
                check(p == rows[-1]["pos"] + (v == 0), f"{at}: token positions must be contiguous")
            ids, bits, cpu, resident = (row.get(f) for f in ("ids", "weight_bits", "cpu", "resident_ids"))
            check(isinstance(ids, list) and len(ids) == K and all(integer(e, 0, N_EXPERTS) for e in ids)
                  and len(set(ids)) == K, f"{at}: invalid ordered top-8 IDs")
            check(isinstance(bits, list) and len(bits) == K and all(integer(w, 0, 2**32) for w in bits),
                  f"{at}: invalid ordered FP32 weight bits")
            check(cpu is None or isinstance(cpu, list) and len(cpu) == K and all(type(x) is bool for x in cpu),
                  f"{at}: invalid ordered CPU mask")
            for f, val in (("resident_ids", resident), ("pending_ids", row.get("pending_ids"))):
                check(val is None or isinstance(val, list) and all(integer(e, 0, N_EXPERTS) for e in val)
                      and len(set(val)) == len(val), f"{at}: invalid {f}")
            if cpu is not None and resident is not None:
                check(cpu == [e not in set(resident) for e in ids], f"{at}: resident state disagrees with placement")
            for f in (*TIMINGS, "route_ready_ns"):
                value = row.get(f)
                check(value is None or integer(value, 0, 2**63), f"{at}: bad {f}")
            rows.append(row)
    check(rows and len(rows) % N_VISITS == 0, f"{path}: no complete trace or truncated last token")
    check(all(r.get("cpu") is not None for r in rows) or all(r.get("cpu") is None for r in rows),
          f"{path}: partial CPU labels")
    for f in ("resident_ids", "pending_ids"):
        check(all(r.get(f) is None for r in rows) or all(r.get(f) is not None for r in rows), f"{path}: partial {f}")
    sizes = meta.get("expert_bytes")
    check(sizes is None or isinstance(sizes, list) and len(sizes) == 24
          and all(integer(x, 1, 2**31) for x in sizes), f"{path}: bad per-layer expert payload sizes")
    timing = {key: np.array([r.get(key) if r.get(key) is not None else -1 for r in rows]).reshape(-1, N_VISITS)
              for key in TIMINGS}
    ready = np.array([r.get("route_ready_ns") if r.get("route_ready_ns") is not None else -1 for r in rows])
    def masks(f):
        m = np.full((len(rows), N_EXPERTS), -1, dtype=np.int8)
        if rows[0].get(f) is not None:
            m.fill(0)
            for i, r in enumerate(rows):
                m[i, r[f]] = 1
        return m.reshape(-1, N_VISITS, N_EXPERTS)
    return Trace(meta, np.array([r["pos"] for r in rows[::N_VISITS]]),
                 np.array([r["ids"] for r in rows], dtype=np.uint8).reshape(-1, N_VISITS, K),
                 np.array([r["weight_bits"] for r in rows], dtype="<u4").reshape(-1, N_VISITS, K),
                 np.array([r.get("cpu") if r.get("cpu") is not None else [-1] * K for r in rows],
                          dtype=np.int8).reshape(-1, N_VISITS, K),
                 masks("resident_ids"), masks("pending_ids"), ready.reshape(-1, N_VISITS), timing)


def ratio(a, b):
    return float(a / b) if b else None


def summary(x):
    if not len(x):
        return None
    return dict(n=len(x), mean=float(np.mean(x)), p10=float(np.percentile(x, 10)),
                p50=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)), max=float(np.max(x)))


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 and np.ptp(x) and np.ptp(y) else None


def miss_counts(counts):
    out = {}
    for a, b in PAIRS:
        by_layer = {}
        for l in LAYERS:
            x, y = counts[:, l + 12 * (a - 1)], counts[:, l + 12 * (b - 1)]
            by_layer[l] = dict(pearson=correlation(x, y), gt0_both=int(np.sum((x > 0) & (y > 0))),
                               prev_gt0=int(np.sum(x > 0)), target_gt0=int(np.sum(y > 0)),
                               adjacent_token_null_pearson=correlation(x[:-1], y[1:]),
                               count_table=np.bincount(9 * x + y, minlength=81).reshape(9, 9).tolist())
        out[f"{a}->{b}"] = by_layer
    return out


def critical_path(cpu, timing):
    """Observed timing surfaces; never sum overlapping spans or infer GPU timestamps from enqueue times."""
    valid = cpu is not None
    n = cpu.shape[0] if valid else next(iter(timing.values())).shape[0]
    out = {}
    passes = np.array([stage(v) for v in range(N_VISITS)])
    for key, times in timing.items():
        present = times >= 0
        if not np.any(present):
            continue
        x = times / 1e3
        # Only aggregate complete token/stage cells, not a possibly biased partial total.
        complete = present.all(axis=1)
        row = dict(visits_present=int(present.sum()), visits_expected=n * N_VISITS,
                   per_token_us=summary(x[complete].sum(axis=1)), by_stage={}, by_visit={})
        for part in range(5):
            sel = passes == part
            ok = present[:, sel].all(axis=1)
            row["by_stage"][part] = summary(x[ok][:, sel].sum(axis=1))
        for v in range(N_VISITS):
            row["by_visit"][v] = summary(x[present[:, v], v])
        if valid:
            row["visit_us_given_ncpu"] = {i: summary(x[(cpu == i) & present]) for i in range(K + 1)}
            row["visit_us_given_stage_ncpu"] = {
                p: {i: summary(x[(cpu == i) & present & (passes == p)]) for i in range(K + 1)} for p in range(5)
            }
        out[key] = row
    return out


def presence(ids):
    return np.eye(N_EXPERTS, dtype=bool)[ids].any(axis=-2)


def pair_stats(trace, va, vb):
    a, b = trace.ids[:, va], trace.ids[:, vb]
    m = a[:, :, None] == b[:, None, :]
    overlap = m.sum(axis=(1, 2))
    ra, rb = presence(a), presence(b)
    a_count, b_count = ra.sum(axis=0), rb.sum(axis=0)
    shared = (ra & rb).sum(axis=0)
    out = dict(tokens=len(a), intersection=summary(overlap), jaccard=summary(overlap / (2 * K - overlap)),
               independent_marginals_mean_intersection=float(a_count @ b_count / len(a)**2),
               prev_token_target_intersection=summary((rb[:-1] & rb[1:]).sum(axis=1)),
               cross_pass_lag1_intersection=summary((ra[:-1] & rb[1:]).sum(axis=1)),
               rank_to_rank_counts=m.sum(axis=0).tolist(),
               same_id_counts_by_expert=np.stack([a_count, b_count, shared], axis=1).tolist())
    ac, bc = trace.cpu[:, va], trace.cpu[:, vb]
    if ac[0, 0] >= 0:
        m_miss = m & (ac[:, :, None] == 1) & (bc[:, None, :] == 1)
        out["source_miss_target_miss_overlap"] = summary(m_miss.sum(axis=(1, 2)))
        out["target_cpu_misses_seen_source_route"] = ratio(np.sum(m & (bc[:, None, :] == 1)), bc.sum())
        out["target_cpu_misses_seen_source_misses"] = ratio(m_miss.sum(), bc.sum())
    z = trace.ready_ns[:, [va, vb]]
    present = (z >= 0).all(axis=1)
    if np.any(present):
        lead = np.diff(z[present], axis=1).ravel()
        check(np.all(lead >= 0), "route_ready_ns clocks/visit order disagree")
        out["source_to_target_route_ready_us"] = summary(lead / 1000)
    return out


def predictor_metrics(scores, truth, miss, cap, no_issue_mask=None, nbytes=None, join_times=None):
    order = np.argsort(-scores, axis=1, kind="stable")[:, :cap]
    pick = (np.take_along_axis(scores, order, axis=1) > -np.inf)
    routed = np.take_along_axis(truth, order, axis=1) & pick
    sent = int(pick.sum())
    hits = int(routed.sum())
    n = len(truth)
    out = dict(predicted=sent, route_hits=hits, route_targets=n * K)
    if miss is not None:
        useful = np.take_along_axis(miss, order, axis=1) & pick
        missed = int(miss.sum())
        got = int(useful.sum())
        out.update(miss_targets=missed, miss_hits=got,
                   predictions_for_target_gpu_experts=hits - got, predictions_not_routed=sent - hits)
        if no_issue_mask is not None:
            # An absent-at-issue prediction that becomes resident anyway is counted as not useful.
            actual_send = ~np.take_along_axis(no_issue_mask, order, axis=1) & pick
            issue_count = int(actual_send.sum())
            useful_count = int((actual_send & useful).sum())
            out.update(issued_nonresident=issue_count, useful_issued_nonresident=useful_count)
            if nbytes is not None:
                out.update(payload_bytes=issue_count * nbytes, false_positive_payload_bytes=(issue_count - useful_count) * nbytes)
            if join_times:
                for name, time in join_times.items():
                    measured = time >= 0
                    if not measured.any():
                        continue
                    # Join time on visits with at least one useful prediction, not estimated time saved.
                    out[f"observed_{name}"] = int(time[measured].sum())
                    out[f"{name}_on_visits_with_useful_pick"] = int(time[measured & (actual_send & useful).any(axis=1)].sum())
    return out


def rates(x):
    out = dict(x)
    out.update(route_precision=ratio(x["route_hits"], x["predicted"]), route_recall=ratio(x["route_hits"], x["route_targets"]))
    if "miss_hits" in x:
        out.update(miss_recall=ratio(x["miss_hits"], x["miss_targets"]),
                   miss_precision=ratio(x["miss_hits"], x["predicted"]))
    if "issued_nonresident" in x:
        out.update(useful_issue_rate=ratio(x["useful_issued_nonresident"], x["issued_nonresident"]),
                   issued_miss_recall=ratio(x["useful_issued_nonresident"], x["miss_targets"]))
    return out


def ranked(ids):
    rank = np.full((len(ids), N_EXPERTS), -np.inf)
    rank[np.arange(len(ids))[:, None], ids] = np.arange(K, 0, -1)
    return rank


def predict(eval_trace, training, cap_list):
    # Training and evaluation trajectories do not see one another's future routes or residency.
    train_ids = np.concatenate([t.ids for t in training], axis=0)
    train_by_visit = np.stack([np.bincount(train_ids[:, v].ravel(), minlength=N_EXPERTS) for v in range(N_VISITS)])
    global_freq = train_by_visit.sum(axis=0)
    ids = eval_trace.ids
    n = len(ids) - 1  # all arms scored on identical rows; no previous-token route on first eval token
    outer = {}
    for a, b in PAIRS:
        totals, per_layer = {}, {}
        for l in LAYERS:
            va, vb = l + 12 * (a - 1), l + 12 * (b - 1)
            ta, tb = presence(train_ids[:, va]).astype(np.float64), presence(train_ids[:, vb]).astype(np.float64)
            p = tb.mean(axis=0)
            cond = (ta.T @ tb + 16 * p) / (ta.sum(axis=0)[:, None] + 16)
            ma = ids[1:, va]
            current = ranked(ma)
            previous = ranked(ids[:-1, vb])
            base = np.broadcast_to(p, (n, N_EXPERTS))
            recency = np.zeros_like(base)
            target_all = presence(ids[:, vb])
            for lag in range(1, min(4, n) + 1):
                recency[lag - 1:] += target_all[:-lag] * (2.0 ** (1 - lag))
            row_scores = {
                "global_frequency": np.broadcast_to(global_freq, base.shape),
                "layer_frequency": np.broadcast_to(train_by_visit[[l, l + 12, l + 24]].sum(axis=0), base.shape),
                "layer_pass_frequency": base,
                "previous_token": previous,
                "previous_pass": current,
                "conditional_transition": cond[ma].mean(axis=1),
                "pass_recency": np.where(recency > 0, recency + 0.01 * base, -np.inf),
                "previous_token_plus_pass": np.where(np.isfinite(previous) | np.isfinite(current),
                    np.maximum(previous, 0) + np.maximum(current, 0) + 0.01 * base, -np.inf),
            }
            truth = target_all[1:]
            cpu = eval_trace.cpu[1:, vb]
            miss = None
            if cpu[0, 0] >= 0:
                miss = np.zeros_like(truth)
                miss[np.arange(n)[:, None], ids[1:, vb]] = cpu == 1
            resident = eval_trace.resident[1:, va]
            source_resident = (resident == 1) if np.all(resident != -1) else None
            pending = eval_trace.pending[1:, va]
            no_issue = source_resident if source_resident is None else (source_resident | (pending == 1))
            expert_bytes = eval_trace.metadata.get("expert_bytes")
            nbytes = expert_bytes[l] if expert_bytes else None
            joins = {k: eval_trace.timing[k][1:, vb] for k in ("exposed_join_ns", "gpu_to_cpu_ready_marker_ns")}
            rows = {}
            for name, scores in row_scores.items():
                arms = {name: scores}
                if no_issue is not None:
                    arms[name + "+source_mask"] = np.where(~no_issue, scores, -np.inf)
                for arm, s in arms.items():
                    for cap in cap_list:
                        key = f"{arm}@{cap}"
                        m = predictor_metrics(s, truth, miss, cap, no_issue, nbytes, joins)
                        rows[key] = rates(m)
                        acc = totals.setdefault(key, {})
                        for field, value in m.items():
                            acc[field] = acc.get(field, 0) + value
            per_layer[l] = rows
        outer[f"{a}->{b}"] = dict(micro={k: rates(v) for k, v in totals.items()}, by_layer=per_layer)
    return outer


def time_fit(test, train):
    """Held-out descriptive timing regressions, not cost estimates for a different placement."""
    if test.cpu[0, 0, 0] < 0 or any(t.cpu[0, 0, 0] < 0 for t in train):
        return {}
    miss = test.cpu.sum(axis=2)
    train_miss = np.concatenate([t.cpu.sum(axis=2) for t in train], axis=0)
    parts = np.array([stage(v) for v in range(N_VISITS)])
    out = {}
    for key in ("visit_ns", "exposed_join_ns", "gpu_to_cpu_ready_marker_ns"):
        actual = test.timing[key] / 1000
        data = np.concatenate([t.timing[key] for t in train], axis=0) / 1000
        train_present, test_present = data >= 0, actual >= 0
        if not train_present.any() or not test_present.any():
            continue
        valid_tokens = test_present.all(axis=1)
        pmiss = np.full(actual.shape, np.median(data[train_present]))
        pstage, seen_miss, seen_stage = pmiss.copy(), np.zeros(miss.shape, bool), np.zeros(miss.shape, bool)
        min_samples = 5
        for i in range(K + 1):
            x = data[(train_miss == i) & train_present]
            if len(x) >= min_samples:
                pmiss[miss == i] = np.median(x)
                pstage[miss == i] = np.median(x)
                seen_miss[miss == i] = True
            for p in range(5):
                x = data[(train_miss == i) & (parts == p) & train_present]
                if len(x) >= min_samples:
                    choose = (miss == i) & (parts == p)
                    pstage[choose] = np.median(x)
                    seen_stage[choose] = True
        def errors(prediction, supported):
            delta = prediction - actual
            return dict(mean_abs_visit_error_us=float(np.abs(delta[test_present]).mean()),
                        token_error_us=summary(delta[valid_tokens].sum(axis=1)),
                        mean_abs_token_error_us=ratio(np.abs(delta[valid_tokens].sum(axis=1)).sum(), int(valid_tokens.sum())),
                        measured_visits=int(test_present.sum()), fallback_visits=int((test_present & ~supported).sum()))
        out[key] = dict(min_train_cell_samples=min_samples, miss_count=errors(pmiss, seen_miss),
                        stage_miss_count=errors(pstage, seen_stage))
    return out


def analyze(trace, train, cap_list):
    counts = trace.cpu.sum(axis=2) if trace.cpu[0, 0, 0] >= 0 else None
    digest = hashlib.sha256()
    digest.update(trace.ids.tobytes())
    digest.update(trace.bits.tobytes())
    digest.update(trace.cpu.tobytes())
    statistics = {}
    for a, b in PAIRS:
        per_layer = {l: pair_stats(trace, l + 12 * (a - 1), l + 12 * (b - 1)) for l in LAYERS}
        statistics[f"{a}->{b}"] = dict(by_layer=per_layer,
            mean_intersection=float(np.mean([r["intersection"]["mean"] for r in per_layer.values()])),
            mean_jaccard=float(np.mean([r["jaccard"]["mean"] for r in per_layer.values()])),
            mean_independent_marginal_intersection=float(np.mean([r["independent_marginals_mean_intersection"] for r in per_layer.values()])))
    branch_gaps = {}
    for key in ("gpu_to_cpu_ready_marker_ns", "exposed_join_ns", "gpu_to_tail_gap_ns"):
        t = trace.timing[key]
        complete = (t >= 0).all(axis=1)
        if complete.any():
            branch_gaps[key] = summary(t[complete, 18:42].sum(axis=1) / 1000)
    return dict(held_out_tokens=len(trace.pos), held_out_positions=[int(trace.pos[0]), int(trace.pos[-1])],
                ordered_ids_weight_bits_and_placement_sha256=digest.hexdigest(),
                source_mask="resident_or_pending" if trace.pending[0, 0, 0] >= 0 else "resident_only_if_available",
                held_out_cross_pass=statistics, held_out_predictors=predict(trace, train, cap_list),
                observed_miss_count_correlation=miss_counts(counts) if counts is not None else None,
                observed_cpu_miss_fraction=float(np.mean(counts) / K) if counts is not None else None,
                timing=critical_path(counts, trace.timing), held_out_observed_timing_fit=time_fit(trace, train),
                held_out_core2_3_branch_gap_us_per_token=branch_gaps)


def save_transitions(path, trace):
    """C[a,b,layer,e,f] counts visits that select e in source and f in target (not exclusive events)."""
    counts, denom = [], []
    for a, b in PAIRS:
        for l in LAYERS:
            x = presence(trace.ids[:, l + 12 * (a - 1)]).astype(np.int32)
            y = presence(trace.ids[:, l + 12 * (b - 1)]).astype(np.int32)
            counts.append(x.T @ y)
            denom.append(x.sum(axis=0))
    with path.open("xb") as stream:
        np.savez_compressed(stream, pairs=PAIRS, layers=list(LAYERS),
                            joint=np.array(counts).reshape(3, 12, N_EXPERTS, N_EXPERTS),
                            source_count=np.array(denom).reshape(3, 12, N_EXPERTS))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path, help="held-out normalized trace, or full trace for a chronological split")
    p.add_argument("--train", type=Path, action="append", default=[], help="separate training trace (repeatable)")
    p.add_argument("--holdout-start", type=int, help="number of prefix tokens used for training (default floor(0.65*n))")
    p.add_argument("--eval-start", type=int, default=0, help="skip prefix rows of the external evaluation trace")
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--transition-output", type=Path, help="optional held-out 128×128 expert co-occurrence counts per pair and layer (NPZ)")
    args = p.parse_args()
    check(all(1 <= k <= N_EXPERTS for k in args.budgets), "bad prediction budget")
    check(not args.train or args.holdout_start is None, "choose either external training or a chronological split")
    trace = load(args.trace)
    check(0 <= args.eval_start <= len(trace.pos) - 2 and (args.train or args.eval_start == 0),
          "--eval-start requires external training and at least two remaining evaluation tokens")
    trace_hash = sha256(args.trace)
    if args.train:
        source_hashes = [sha256(path) for path in args.train]
        check(len(set(source_hashes)) == len(source_hashes) and trace_hash not in source_hashes, "duplicate/reused input")
        train = [load(path) for path in args.train]
        for t in train:
            pin = trace.metadata.get("model_sha256")
            check(pin and pin == t.metadata.get("model_sha256"), "external split needs matching model_sha256 pins")
            eid, tid = trace.metadata.get("trajectory_id"), t.metadata.get("trajectory_id")
            check(eid and tid and eid != tid, "external split needs distinct trajectory IDs (not A/A repeats)")
        test = trace[args.eval_start:]
        split = "external_train"
    else:
        source_hashes = [trace_hash]
        cut = args.holdout_start if args.holdout_start is not None else len(trace.pos) * 65 // 100
        check(1 <= cut <= len(trace.pos) - 2, "need at least one training token and two held-out tokens")
        train, test = [trace[:cut]], trace[cut:]
        split = "chronological_within_trajectory"
    check(len(test.pos) >= 2, "need two or more held-out tokens")
    report = dict(schema="inkling-route-analysis-v1", split=split, eval_path=str(args.trace), eval_sha256=trace_hash,
                  eval_metadata=trace.metadata, training_paths=list(map(str, args.train)) or [str(args.trace)],
                  train_sha256=source_hashes, train_tokens=[len(t.pos) for t in train],
                  training_metadata=[t.metadata for t in train], prediction_labels="observed CPU placement, not a replay of changed placement",
                  **analyze(test, train, args.budgets))
    if args.transition_output:
        save_transitions(args.transition_output, test)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
