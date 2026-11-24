#!/usr/bin/env python3
"""Frozen causal 32-token distribution forecast and paid same-class placement replay."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

import simulate as sim


CONFIG = Path(__file__).with_name("causal-32-frozen.json")
CONFIG_SHA = "5f410ced2e7d931ac775304cb4a199cc2e4eb14522eb381eebb5094b857670c6"
TRANSFER_SHA = "a293e0ca0d98c678e928eb4553d2dc27b156a7d931ae026adf915b68bb4699d1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Capture:
    routes: np.ndarray
    observed: np.ndarray
    times: np.ndarray
    clock: np.ndarray
    initial: np.ndarray
    outstanding: list
    classes: list
    size: np.ndarray
    budget_by_class: list
    queue: int
    first: int
    digest: str


def load(path, cfg):
    meta, visits, events = sim.read_trace(path)
    state, budgets, inflight, classes, size, queue = sim.replay(visits, events)
    n = len(state)
    start = cfg["score_from_token"]
    sim.require(meta["first"] == 65536 and n == cfg["required_total_tokens"],
                "expected frozen screen geometry (64K prefill, 256 serial tokens)")
    sim.require(n - start == cfg["score_length"] and np.all(np.asarray(budgets[start:]) == budgets[start]),
                "held-out range size/capacity changed")
    r = np.array([[v["layer"] * sim.E + e for e in v["ids"]] for v in visits]).reshape(n, sim.V, 8)
    miss = np.array([v["placement"].count("cpu") for v in visits]).reshape(n, sim.V)
    times = np.array([(v["gpu_dispatch_ms"] + v["gpu_experts_ms"] + v["gpu_join_ms"],
                       v["gpu_join_ms"], v["gpu_visit_ms"]) for v in visits]).reshape(n, sim.V, 3)
    ticks = np.array([v["host_begin_ns"] for v in visits[::sim.V]])
    epoch = ticks[0]
    clock = (ticks - epoch) / 1e6
    first_route = visits[start * sim.V]["route_ready_ns"]
    claims = {}
    for event in events:
        when = event["host_ns"]
        if event["kind"] == "upload_observed" and when > first_route:
            claims.setdefault(sim.key(event), (when - epoch) / 1e6)
    outstanding = [(claims.get(e, np.inf), e) for e in inflight[start]]
    return Capture(r, miss, times, clock, sim.as_mask(state[start]), outstanding, classes,
                   size, budgets[start], queue, meta["first"], sha(path))


def features(routes):
    """Binary same-logical-visit routes, plus same-token cross-pass expert intersections."""
    row = np.zeros((len(routes), sim.V, sim.E), dtype=bool)
    np.put_along_axis(row, routes % sim.E, True, axis=-1)
    core = row[:, 6:42].reshape(len(row), 3, 12, sim.E)
    repeat = np.concatenate([core[:, p] & core[:, q] for p, q in ((0, 1), (1, 2), (0, 2))], axis=1)
    return row, repeat


def similarity(anchors, query, repeat_weight):
    row, repeat = anchors
    qrow, qrepeat = query
    r = (row & qrow).sum(axis=(1, 2)) / (sim.V * 8)
    overlap = (repeat & qrepeat).sum(axis=(1, 2))
    denom = repeat.sum(axis=(1, 2)) + qrepeat.sum()
    repeat_score = np.divide(2 * overlap, denom, out=np.zeros(len(overlap)), where=denom > 0)
    return (1 - repeat_weight) * r + repeat_weight * repeat_score


class Forecast:
    """Estimate expected whole-visit bundles at 32 future horizons without drawing samples."""

    def __init__(self, source_prefix, cfg):
        self.source = source_prefix.copy()
        self.source.flags.writeable = False
        self.cfg = cfg
        self.source_features = features(self.source)

    def pool(self, routes, likeness, source=False):
        # Local window is a rolling feature, not a refit on held-out outcomes.
        n = len(routes)
        frequency = (np.ones(n) if source else
                     np.exp2((np.arange(n) - n + 1) / self.cfg["recent_half_life_tokens"]))
        frequency /= frequency.sum()
        rows = []
        for h in range(1, self.cfg["horizon_tokens"] + 1):
            # Query describes token t-1. Target for submission at t is t+h.
            lag = h + 1
            count = max(0, n - lag)
            mass = frequency.copy()
            if count:
                idx = np.arange(count)
                # Most recent anchor breaks exact ties in route/overlap similarity.
                nearest = np.lexsort((-idx, -likeness[:count]))[:self.cfg["anchor_neighbor_count"]]
                values = likeness[nearest]
                weights = np.exp((values - values.max()) / self.cfg["anchor_temperature"])
                f = self.cfg["conditional_fraction_at_one"] * np.exp2(
                    -(h - 1) / self.cfg["conditional_half_life_tokens"])
                mass *= 1 - f
                mass[nearest + lag] += f * weights / weights.sum()
            rows.append(mass)
        return np.asarray(rows)

    def predict(self, history):
        """Return donor routes and a horizon x donor probability table. No suffix argument."""
        local = history[-self.cfg["recent_donor_tokens"]:]
        sim.require(len(local) > 0, "forecast requires a completed previous token")
        local_features = features(local)
        query = tuple(x[-1] for x in local_features)
        a = similarity(local_features, query, self.cfg["repeat_feature_weight"])
        b = similarity(self.source_features, query, self.cfg["repeat_feature_weight"])
        prior = self.cfg["source_prior_fraction"]
        w = np.hstack(((1 - prior) * self.pool(local, a), prior * self.pool(self.source, b, source=True)))
        return np.concatenate((local, self.source)), w


def run(capture, forecast, cfg, cost, transfer):
    start = cfg["score_from_token"]
    threshold = sim.service(transfer, "pack", max(capture.size))
    plans = {}

    def choose(t, resident, budget):
        if (t - start) % cfg["cadence_tokens"]:
            return []
        donors, certainty = forecast.predict(capture.routes[:t])
        # Integrate the nonlinear visit objective over the empirical probability
        # at each horizon. Correlated misses within a route bundle stay together.
        weights = certainty.sum(axis=0)
        objective = sim.Objective(donors, cost[:, :, 0])
        objective.cost *= np.repeat(weights, sim.V)[:, None]
        pairs = objective.swaps(resident, capture.classes, limit=budget, min_gain=threshold,
                                shortlist=cfg["shortlist_per_trade_side"], once_per_id=True)
        plans[t] = (len(pairs), sum(v for _, _, v in pairs))
        return [(add, drop) for add, drop, _ in pairs]

    miss, debit, uploaded = sim.paid_replay(
        capture.routes, start, capture.clock, capture.initial, capture.outstanding, capture.size,
        capture.queue, transfer, "pack", choose, cfg["batch_limit"])
    return miss, debit, uploaded, plans


def score(capture, fitted_source, cfg, cost, boot_costs, transfer, forecast):
    start = cfg["score_from_token"]
    miss, debit, uploaded, plans = run(capture, forecast, cfg, cost, transfer)
    observed = capture.observed[start:]
    visits = np.arange(sim.V)
    choices = cost[visits, observed] - cost[visits, miss]
    gross = choices.sum(axis=1)
    net = gross - debit[:, None]
    # Calibration stays fixed to source for both fixtures. Only after replay do
    # we compare predictions to the routes/miss counts the engine observed.
    estimate_error = cost[visits, observed] - capture.times[start:]
    rng = np.random.default_rng(cfg["bootstrap_seed"] + 1)
    boot = []
    for model in boot_costs:
        h = sim.sample_blocks(rng, len(miss), cfg["bootstrap_block_tokens"])
        boot.append(((model[visits, observed[h]] - model[visits, miss[h]]).sum(axis=1)
                     - debit[h, None]).mean(axis=0))
    joint = sim.joint_counts(miss)
    baseline_joint = sim.joint_counts(observed)

    def joint_3_plus(hist):
        return {k: sum(sum(row[3:]) for row in matrix[3:]) for k, matrix in hist.items()}

    return dict(trace_sha256=capture.digest, source_fixture=fitted_source,
                first_position=capture.first + start, end_exclusive=capture.first + len(capture.routes),
                funded_slots_by_class=capture.budget_by_class,
                funded_bytes=sum(int(c) * int(capture.size[ix[0]])
                                 for c, ix in zip(capture.budget_by_class, capture.classes)),
                queue_entries=capture.queue,
                observed_hit_rate=1 - float(observed.mean()) / 8,
                causal_hit_rate=1 - float(miss.mean()) / 8,
                observed_miss_histogram=np.bincount(observed.ravel(), minlength=9).tolist(),
                causal_miss_histogram=np.bincount(miss.ravel(), minlength=9).tolist(),
                observed_same_layer_both_visits_3plus_misses=joint_3_plus(baseline_joint),
                causal_same_layer_both_visits_3plus_misses=joint_3_plus(joint),
                same_layer_joint_miss_histograms=joint,
                mean_branch_saved_ms_by_visit=choices[:, :, 0].mean(axis=0).tolist(),
                source_calibration_heldout_visit_MAE_ms=dict(zip(sim.SURFACES,
                                                                 np.abs(estimate_error).mean(axis=(0, 1)))),
                gross_ms_token=dict(zip(sim.SURFACES, gross.mean(axis=0))),
                net_ms_token=dict(zip(sim.SURFACES, net.mean(axis=0))),
                net_interval=dict(zip(sim.SURFACES, np.quantile(boot, [.025, .975], axis=0).T.tolist())),
                pack_charge_ms_token=float(debit.mean()), H2D_bytes=int(uploaded.sum()),
                scheduled_batches=len(plans), proposed_swaps=sum(n for n, _ in plans.values()),
                proposed_sum_of_forecast_32_token_savings_ms=sum(v for _, v in plans.values()),
                primary_gate_pass=bool(net[:, 0].mean() >= cfg["gate_net_ms_token_each_leg"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="frozen mixed 64K PSVST1 input (first 128 fit, suffix scored)")
    p.add_argument("transfer", type=Path, help="guard-transfer/command.log used in placement oracle")
    p.add_argument("--independent", type=Path, help="unseen 64K code trace (suffix scored, no fit)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sim.require(sha(CONFIG) == CONFIG_SHA, "frozen configuration bytes changed")
    cfg = json.loads(CONFIG.read_text())
    sim.require(sha(args.source) == cfg["training_trace_sha256"], "source trace does not match frozen config")
    sim.require(sha(args.transfer) == TRANSFER_SHA, "transfer data does not match placement oracle")
    source = load(args.source, cfg)
    train = cfg["training_tokens"]
    fit_miss, fit_times = source.observed[:train], source.times[:train]
    cost, _ = sim.surface(fit_miss, fit_times)
    forecast = Forecast(source.routes[:train], cfg)
    transfer = sim.transfer_csv(args.transfer)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    boots = []
    for _ in range(cfg["bootstrap_draws"]):
        t = sim.sample_blocks(rng, train, cfg["bootstrap_block_tokens"])
        boots.append(sim.surface(fit_miss[t], fit_times[t])[0] if len(np.unique(fit_miss[t])) == 9 else cost)
    result = dict(config_sha256=CONFIG_SHA, transfer_csv_sha256=TRANSFER_SHA,
                  mixed_64k=score(source, True, cfg, cost, boots, transfer, forecast))
    if args.independent:
        independent = load(args.independent, cfg)
        sim.require(independent.digest != source.digest, "independent leg duplicates training input")
        result["code_64k"] = score(independent, False, cfg, cost, boots, transfer, forecast)
        result["gate_pass"] = all(result[run]["primary_gate_pass"] for run in ("mixed_64k", "code_64k"))
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
