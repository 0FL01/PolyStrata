#!/usr/bin/env python3
"""Replay PSVST1 and screen placement policies on fixed routes (NumPy, CPU only)."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trace"))
import collect


N, V, E = 3072, 48, 128
GROUP = np.repeat(np.arange(5), [6, 12, 12, 12, 6])
GROUP_NAMES = ("prelude", "core1", "core2", "core3", "coda")
WEIGHT = np.repeat(np.r_[np.full(6, 2), np.ones(12), np.full(6, 2)], E)
SURFACES = ("branch_stream_ms", "exposed_join_ms", "visit_stream_ms")


def require(condition, problem):
    if not condition:
        raise ValueError(problem)


def key(row, victim=False):
    p = "victim_" if victim else ""
    return row[p + "layer"] * E + row[p + "expert"]


def read_trace(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as f:
        meta = collect.header(f)
        rows = list(collect.records(f, meta))
    visits = [r for r in rows if r["kind"] == "visit"]
    expected = np.arange(meta["first"], meta["end"])
    require(np.array_equal(expected, [r["pos"] for r in visits[::V]]),
            "expected precisely one complete token frame per requested position")
    require(all(not r["windowed"] and not r["frozen"] and "skipped" not in r["placement"]
                for r in visits), "expected live serial top-8 decode")
    events = sorted((r for r in rows if r["kind"] != "visit"), key=lambda r: r["host_ns"])
    return meta, visits, events


def replay(visits, events):
    """Reconstruct ownership from observed decisions, independently check every mask/offset."""
    resident = {key(r): r["slot_offset"] for r in events if r["kind"] == "initial_resident"}
    pending = {key(r): r["slot_offset"] for r in events if r["kind"] == "initial_pending"}
    slots = {r["layer"]: (r["trade_class"], r["slot_bytes"])
             for r in events if r["kind"] == "initial_resident"}
    require(len(slots) == 24 and len(set(resident.values()) | set(pending.values())) ==
            len(resident) + len(pending), "incomplete or conflicting initial map")
    trade, size = np.repeat(np.array([slots[i] for i in range(24)]), E, axis=0).T
    require(all(len(set(size[trade == c])) == 1 for c in np.unique(trade)), "mixed slot sizes in a trade class")
    class_ids = [np.flatnonzero(trade == c) for c in np.unique(trade)]
    max_initial = max(r["host_ns"] for r in events if r["kind"] in ("initial_resident", "initial_pending"))
    ev = iter(r for r in events if r["kind"] not in ("initial_resident", "initial_pending")
              and r["host_ns"] > max_initial)
    row = next(ev, None)
    snapshots, budgets, inflight, capacities = [], [], [], []
    masks = [0] * 24

    def attach(k, o):
        require(k not in resident, "double attach")
        resident[k] = o
        masks[k // E] |= 1 << (k % E)

    def evict(k):
        masks[k // E] &= ~(1 << (k % E))
        require(k in resident, "eviction from missing slot")
        return resident.pop(k)

    def apply(r):
        kind = r["kind"]
        if kind == "swap_asked":
            k, o = key(r), r["slot_offset"]
            require(k not in resident and k not in pending, "duplicate request")
            if r["victim_layer"] >= 0:
                victim = key(r, True)
                require(trade[victim] == trade[k] and evict(victim) == o, "swap victim/offset mismatch")
            pending[k] = o
        elif kind == "upload_observed":
            k, o = key(r), r["slot_offset"]
            require(pending.pop(k, None) == o, "completion without matching request")
            attach(k, o)
        elif kind == "shrink_slot":
            k = key(r)
            evict(k)
            if r["victim_layer"] >= 0:
                victim = key(r, True)
                o = evict(victim)
                require(o == r["new_slot_offset"] and trade[victim] == trade[k], "shrink offset mismatch")
                attach(k, o)
        elif kind == "follow":
            capacities.append(r["swap_capacity"])

    for k in resident:
        masks[k // E] |= 1 << (k % E)
    for v in visits:
        while row and row["host_ns"] <= v["route_ready_ns"]:
            apply(row)
            row = next(ev, None)
        layer = v["layer"]
        require(masks[layer] == int(v["resident_mask"], 16), f"resident mask mismatch at {v['pos']}:{v['visit']}")
        require(all(resident.get(layer * E + i, -1) == o for i, o in zip(v["ids"], v["slot_offset"])),
                f"selected offsets mismatch at {v['pos']}:{v['visit']}")
        if not v["visit"]:
            snapshots.append(list(resident))
            slots_in_use = Counter(trade[k] for k in resident.keys() | pending.keys())
            budgets.append([slots_in_use[c] for c in np.unique(trade)])
            inflight.append(list(pending))
    return snapshots, budgets, inflight, class_ids, size, min(capacities, default=0)


def surface(misses, y):
    """Median by visit×miss count, backed off to stage then global at sparse cells."""
    out = np.empty((V, 9, y.shape[-1]))
    samples = np.empty((V, 9), dtype=int)
    for count in range(9):
        use = misses == count
        require(use.any(), f"training set has no samples for miss count {count}")
        median = np.median(y[use], axis=0)
        for g in range(5):
            group = np.flatnonzero(GROUP == g)
            by_group = y[:, group, :][use[:, group]]
            n = len(by_group)
            prior = (n * np.median(by_group, axis=0) + 24 * median) / (n + 24) if n else median
            for v in group:
                observed = y[use[:, v], v]
                n = samples[v, count] = len(observed)
                out[v, count] = (n * np.median(observed, axis=0) + 16 * prior) / (n + 16) if n else prior
    return out, samples


def misses_for(routes, selected):
    return 8 - selected[routes].sum(axis=-1)


def scores(cost, routes, misses, delta):
    new = np.clip(misses + delta, 0, 8)
    row = np.arange(len(routes))
    change = cost[row, misses] - cost[row, new]
    return np.bincount(routes.ravel(), np.repeat(change, 8), minlength=N)


class Objective:
    """Joint miss-count objective over an entire visit window, using physical IDs."""

    def __init__(self, routes, cost):
        self.ids = routes.reshape(-1, 8)
        self.cost = np.tile(cost, (len(routes), 1))
        experts = self.ids.ravel()
        indices = np.argsort(experts, kind="stable")
        self.rows = indices // 8
        self.ends = np.r_[0, np.bincount(experts, minlength=N).cumsum()]

    def uses(self, k):
        return self.rows[self.ends[k]:self.ends[k + 1]]

    def greedy(self, caps, classes, frequency=False):
        selected = np.zeros(N, dtype=bool)
        miss = np.full(len(self.ids), 8)
        gain = np.diff(self.ends) if frequency else scores(self.cost, self.ids, miss, -1)
        cls = np.empty(N, int)
        for c, ids in enumerate(classes):
            cls[ids] = c
        remaining = caps.copy()
        available = np.ones(N, dtype=bool)
        for c, ids in enumerate(classes):
            if not remaining[c]:
                available[ids] = False
        for _ in range(sum(caps)):
            k = int(np.argmax(np.where(available, gain, -np.inf)))
            selected[k] = True
            available[k] = False
            remaining[cls[k]] -= 1
            if not remaining[cls[k]]:
                available[classes[cls[k]]] = False
            if not frequency:
                r = self.uses(k)
                old = miss[r].copy()
                miss[r] -= 1
                after = self.cost[r, miss[r]] - self.cost[r, np.maximum(0, miss[r] - 1)]
                before = self.cost[r, old] - self.cost[r, old - 1]
                np.add.at(gain, self.ids[r].ravel(), np.repeat(after - before, 8))
        return selected

    def swaps(self, selected, classes, limit, min_gain=0., shortlist=12, once_per_id=False):
        miss = misses_for(self.ids, selected)
        touched = np.zeros(N, dtype=bool)
        pairs = []
        for _ in range(limit):
            add = scores(self.cost, self.ids, miss, -1)
            drop = scores(self.cost, self.ids, miss, 1)
            best = (min_gain, None, None)
            for ids in classes:
                into, out = ids[~selected[ids] & ~touched[ids]], ids[selected[ids] & ~touched[ids]]
                into = into[np.argsort(-add[into], kind="stable")[:shortlist]]
                out = out[np.argsort(-drop[out], kind="stable")[:shortlist]]
                for i in into:
                    for j in out:
                        gain = add[i] + drop[j]
                        if i // E == j // E:
                            r = np.intersect1d(self.uses(i), self.uses(j), assume_unique=True)
                            m = miss[r]  # both present: applying the pair leaves this visit unchanged
                            gain -= np.sum(2 * self.cost[r, m] - self.cost[r, m - 1] - self.cost[r, m + 1])
                        if gain > best[0] + 1e-12:
                            best = (gain, i, j)
            gain, i, j = best
            if i is None:
                break
            miss[self.uses(i)] -= 1
            miss[self.uses(j)] += 1
            selected[i], selected[j] = True, False
            if once_per_id:
                touched[[i, j]] = True
            pairs.append((i, j, gain))
        return pairs


def transfer_csv(path):
    times = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            n, mode = int(row["bytes"]), row["mode"]
            ms = (float(row["pack_us"]) + float(row["complete_after_submit_start_us"])) / 1000
            times.setdefault((mode, n), []).append(ms)
    return {mode: dict(n=len(t), bytes=n, p10_ms=np.quantile(t, .1), p50_ms=np.quantile(t, .5),
                       p90_ms=np.quantile(t, .9)) for (mode, n), t in times.items()}


def service(transfer, mode, size):
    # Scale only the payload-dependent part, keeping the observed 4 B service as intercept.
    small, full = transfer["small"], transfer[mode]
    return small["p90_ms"] + (full["p90_ms"] - small["p90_ms"]) * size / full["bytes"]


def paid_replay(routes, split, clock, initial, outstanding, size, queue, transfer, mode, choose, max_swaps=8):
    """Serial copy service, no overlap credit; claim only at a subsequent row boundary."""
    selected = initial.copy()
    count = len(routes) - split
    out = np.empty((count, V), int)
    charge, uploaded = np.zeros(count), np.zeros(count, dtype=np.int64)
    pending = outstanding.copy()
    finish = clock[split]
    for t in range(split, len(routes)):
        still = []
        for ready, i in pending:
            if ready <= clock[t]:
                selected[i] = True
            else:
                still.append((ready, i))
        pending = still
        budget = min(queue - len(pending), max_swaps)
        if budget > 0 and t + 1 < len(routes):
            working = selected.copy()
            # Existing reservations enter the scheduler's objective but not the actual hit mask.
            working[[i for _, i in pending]] = True
            pairs = choose(t, working, budget)
            for i, j in pairs:
                if not selected[j]:
                    continue  # cannot replace an in-flight destination
                require(not selected[i] and all(i != p for _, p in pending), "double upload")
                selected[j] = False
                ms = service(transfer, mode, size[i])
                finish = max(finish, clock[t]) + ms
                pending.append((finish, i))
                charge[t - split] += ms
                uploaded[t - split] += size[i]
        out[t - split] = misses_for(routes[t], selected)
    return out, charge, uploaded


def stats(selected):
    return [[int(k // E), int(k % E)] for k in np.flatnonzero(selected)]


def as_mask(ids):
    out = np.zeros(N, dtype=bool)
    out[ids] = True
    return out


def sample_blocks(rng, n, block=16):
    anchors = rng.integers(0, n - block + 1, size=(n + block - 1) // block)
    return (anchors[:, None] + np.arange(block)).ravel()[:n]


def joint_counts(miss):
    core = miss[:, 6:42].reshape(len(miss), 3, 12)
    return {f"{p + 1}->{q + 1}": np.bincount((9 * core[:, p] + core[:, q]).ravel(),
                                             minlength=81).reshape(9, 9).tolist()
            for p, q in ((0, 1), (1, 2), (0, 2))}


def evaluate(args):
    meta, visits, events = read_trace(args.trace)
    snap, budgets, inflight, classes, size, queue = replay(visits, events)
    n = len(snap)
    split = args.train or n // 2
    require(32 <= split <= n - 32, "need at least 32 training and held-out tokens each")
    caps = np.min(budgets, axis=0)
    require(np.all(caps > 0), "empty trade class in steady budget")
    require(np.all(np.asarray(budgets)[split - 1:] == caps),
            "counterfactual requires stable funded slots during held-out; ledger still supports shrink")
    routes = np.array([[v["layer"] * E + i for i in v["ids"]] for v in visits]).reshape(n, V, 8)
    observed = np.array([v["placement"].count("cpu") for v in visits]).reshape(n, V)
    target = np.array([(v["gpu_experts_ms"] + v["gpu_dispatch_ms"] + v["gpu_join_ms"],
                        v["gpu_join_ms"], v["gpu_visit_ms"]) for v in visits]).reshape(n, V, 3)
    clock = np.array([v["host_begin_ns"] for v in visits[::V]])
    epoch = clock[0]
    clock = (clock - epoch) / 1e6
    model, support = surface(observed[:split], target[:split])
    primary = model[:, :, 0]
    train = Objective(routes[:split], primary)
    hold = Objective(routes[split:], primary)
    # Same actual pre-route boundary for every paid policy. Existing uploads
    # are sunk; assimilate these at their recorded claim (not as instant hits).
    frozen = as_mask(snap[split])
    initial = as_mask(snap[split] + inflight[split])
    first_route = visits[split * V]["route_ready_ns"]
    claims = {}
    for r in events:
        if r["kind"] == "upload_observed" and r["host_ns"] > first_route:
            claims.setdefault(key(r), (r["host_ns"] - epoch) / 1e6)
    outstanding = [(claims.get(k, np.inf), k) for k in inflight[split]]
    frequency = train.greedy(caps, classes, frequency=True)
    freq_oracle = hold.greedy(caps, classes, frequency=True)
    greedy = train.greedy(caps, classes)
    tuned = greedy.copy()
    tune_pairs = train.swaps(tuned, classes, limit=128)
    frequency_tuned = frequency.copy()
    freq_pairs = train.swaps(frequency_tuned, classes, limit=128)
    if train.cost[np.arange(len(train.ids)), misses_for(train.ids, frequency_tuned)].sum() < \
            train.cost[np.arange(len(train.ids)), misses_for(train.ids, tuned)].sum():
        tuned, tune_pairs = frequency_tuned, freq_pairs
        start = "frequency"
    else:
        start = "greedy"
    transfer = transfer_csv(args.transfer)
    test_r, test_m = routes[split:], observed[split:]
    predictions = {
        "current_observed_ledger": (test_m, 0, 0),
        "current_frozen_test_start": (misses_for(test_r, frozen), 0, 0),
        "frequency_static": (misses_for(test_r, frequency), 0, 0),
        "frequency_static_oracle": (misses_for(test_r, freq_oracle), 0, 0),
        "cost_static_greedy": (misses_for(test_r, greedy), 0, 0),
        "cost_static_local": (misses_for(test_r, tuned), 0, 0),
        "dynamic_free_all_hit": (np.zeros_like(test_m), 0, 0),
    }
    for c, ids in enumerate(classes):
        require(all(len(set(token.ravel()) & set(ids)) <= caps[c] for token in test_r),
                "free all-hit per-token oracle does not fit")
    zero_charge = np.zeros(len(test_r))
    for delta in (-128, 128):
        other = caps + np.rint(delta * caps / caps.sum()).astype(int)
        require(np.all(other > 0), "capacity sensitivity would empty a class")
        selected = train.greedy(other, classes)
        train.swaps(selected, classes, limit=64)
        predictions[f"cost_static_{other.sum()}_slots"] = (misses_for(test_r, selected), 0, 0)

    def install(t, current, budget):
        pairs = []
        add = scores(train.cost, train.ids, misses_for(train.ids, current), -1)
        drop = scores(train.cost, train.ids, misses_for(train.ids, current), 1)
        for ids in classes:
            into, out = ids[tuned[ids] & ~current[ids]], ids[~tuned[ids] & current[ids]]
            into = into[np.argsort(-add[into], kind="stable")]
            out = out[np.argsort(-drop[out], kind="stable")]
            pairs += [(add[i] + drop[j], i, j) for i, j in zip(into, out)]
        return [(i, j) for _, i, j in sorted(pairs, reverse=True)[:budget]]

    def future(t, current, budget, horizon=16):
        r = routes[t + 1:t + 1 + horizon]
        if len(r) < 4:
            return []
        trade = service(transfer, "pack", max(size))
        return [(i, j) for i, j, _ in Objective(r, primary).swaps(
            current, classes, limit=budget, min_gain=trade, once_per_id=True)]

    prior = np.diff(train.ends)
    if args.profile:
        prior = np.zeros(N, dtype=np.int64)
        for line in args.profile.read_text().splitlines():
            layer, expert, count = map(int, line.split())
            if 0 <= layer < N // E and 0 <= expert < E:
                prior[layer * E + expert] = count
    decay = np.exp2(-1 / 128)
    rec = np.zeros((n, N))
    for t in range(1, n):
        rec[t] = decay * (rec[t - 1] + np.bincount(routes[t - 1].ravel(), minlength=N) * WEIGHT)

    def recent(t, current, budget):
        pairs = []
        recent = rec[t]
        for ids in classes:
            into, out = ids[~current[ids]], ids[current[ids]]
            into = into[np.lexsort((into, -prior[into] * WEIGHT[into], -recent[into]))]
            out = out[np.lexsort((out, prior[out] * WEIGHT[out], recent[out]))]
            pairs += [(recent[i] - recent[j], i, j) for i, j in zip(into[:budget], out[:budget])
                      if recent[i] >= 2 * recent[j] + 2]
        return [(i, j) for _, i, j in sorted(pairs, reverse=True)[:budget]]

    def online(t, current, budget, horizon=16):
        r = routes[t - horizon:t]
        trade = service(transfer, "pack", max(size))
        return [(i, j) for i, j, _ in Objective(r, primary).swaps(
            current, classes, limit=budget, min_gain=trade, once_per_id=True)]

    for mode in ("pinned", "pack"):
        for label, chooser, cadence, max_swaps in (
                ("cost_static_install", install, 1, 8),
                ("future_16_every16", future, 16, 8),
                ("future_16_every4", future, 4, 16),
                ("future_32_every8", lambda t, s, b: future(t, s, b, horizon=32), 8, 32),
                ("weighted_recent_proxy", recent, 1, queue),
                ("cost_online_16_every16", online, 16, 8),
                ("cost_online_16_every4", online, 4, 16)):
            predictions[f"{label}_{mode}"] = paid_replay(
                routes, split, clock, frozen, outstanding, size, queue, transfer, mode,
                lambda t, s, b: chooser(t, s, b) if (t - split) % cadence == 0 else [], max_swaps)

    visit = np.arange(V)
    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(args.bootstrap):
        ti = sample_blocks(rng, split)
        hi = sample_blocks(rng, n - split)
        # Rare tail strata may drop out of a resample: keep the point estimate there.
        if len(np.unique(observed[ti])) != 9:
            surface_b = model
        else:
            surface_b, _ = surface(observed[ti], target[ti])
        boots.append((surface_b, hi))

    out = {}
    for name, (miss, charge, upload) in predictions.items():
        charge = np.broadcast_to(charge, zero_charge.shape)
        delta = model[visit, test_m] - model[visit, miss]
        saved = delta.sum(axis=1)
        boot = np.array([((b[visit, test_m[h]] - b[visit, miss[h]]).sum(axis=1)
                          - charge[h, None]).mean(axis=0) for b, h in boots])
        out[name] = dict(
            hit_rate=1 - miss.mean() / 8, miss_histogram=np.bincount(miss.ravel(), minlength=9).tolist(),
            by_stage_miss_histogram={g: np.bincount(miss[:, GROUP == gi].ravel(), minlength=9).tolist()
                                     for gi, g in enumerate(GROUP_NAMES)},
            same_layer_cross_pass_joint_miss_histograms=joint_counts(miss),
            by_visit_mean_misses=miss.mean(axis=0).tolist(),
            by_visit_branch_saved_ms=delta[:, :, 0].mean(axis=0).tolist(),
            exact_miss_count_agreement_with_observed=float((miss == test_m).mean()),
            saved_ms_token=dict(zip(SURFACES, saved.mean(axis=0))),
            net_ms_token=dict(zip(SURFACES, (saved - charge[:, None]).mean(axis=0))),
            net_interval=dict(zip(SURFACES, np.quantile(boot, [.025, .975], axis=0).T.tolist())),
            H2D_service_ms=float(charge.sum()), H2D_bytes=int(np.sum(upload)),
            nonzero_upload_tokens=int(np.count_nonzero(upload)))
        if args.baseline_ms and args.baseline_ms > out[name]["net_ms_token"][SURFACES[0]]:
            out[name]["predicted_tokens_s_if_additive"] = 1000 / (
                args.baseline_ms - out[name]["net_ms_token"][SURFACES[0]])

    hold_tuned = misses_for(test_r, tuned)
    train_tuned = misses_for(routes[:split], tuned)
    utility = []
    for k in range(N):
        # Report benefit of holding the slot; for a resident this is the loss on deletion.
        sign = -1 if tuned[k] else 1
        pass_train, pass_hold = [], []
        for objective, m, p, denom in ((train, train_tuned, pass_train, split),
                                       (hold, hold_tuned, pass_hold, len(test_r))):
            uses = objective.uses(k)
            before = m.ravel()[uses]
            after = before - sign
            change = sign * (objective.cost[uses, before] - objective.cost[uses, after])
            p.extend(np.bincount(GROUP[uses % V], change, minlength=5) / denom / (size[k] / 2**30))
        utility.append([k // E, k % E, int(size[k]), int(tuned[k]),
                        np.round(pass_train, 8).tolist(), np.round(pass_hold, 8).tolist()])
    err = model[visit, test_m] - target[split:]
    changes = tuned & ~initial
    install_cost = {mode: float(service(transfer, mode, size[changes]).sum()) for mode in ("pinned", "pack")}
    result = dict(
        trace=dict(sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                   first=int(meta["first"]), end_exclusive=int(meta["end"]),
                   evaluate_from=int(meta["first"] + split), visits_replayed=n * V,
                   observed_record_counts=dict(Counter(r["kind"] for r in events)),
                   pos_heldout_start_pending=len(outstanding),
                   pos_heldout_start_pending_with_observed_claim=sum(np.isfinite(when).item() for when, _ in outstanding)),
        capacity=[dict(slots=int(c), slot_bytes=int(size[ix[0]]), layers=np.unique(ix // E).tolist())
                  for c, ix in zip(caps, classes)],
        capacity_bytes=int(sum(c * size[ix[0]] for c, ix in zip(caps, classes))),
        min_observed_swap_queue_size=queue, capacity_model="same trade-class slots as visit ledger",
        transfer=transfer, transfer_csv_sha256=hashlib.sha256(args.transfer.read_bytes()).hexdigest(),
        profile_seed_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest() if args.profile else None,
        train_static_start=start, train_local_swaps=len(tune_pairs),
        bootstrap=dict(draws=args.bootstrap, seed=args.seed, contiguous_block_tokens=16),
        miss_surface_ms=model.tolist(), miss_surface_train_sample_counts=support.tolist(),
        heldout_calibration={label: dict(visit_MAE_ms=float(np.abs(err[:, :, i]).mean()),
                                          token_mean_bias_ms=float(err[:, :, i].sum(axis=1).mean()),
                                          token_SD_error_ms=float(err[:, :, i].sum(axis=1).std()))
                             for i, label in enumerate(SURFACES)},
        heldout_relaxed_any_miss_per_visit_bound_branch_ms=float(
            (model[visit, test_m, 0] - primary.min(axis=1)).sum(axis=1).mean()),
        train_cost_static_install=dict(uploads=int(changes.sum()), upload_bytes=int(size[changes].sum()),
                                       serial_p90_ms=install_cost),
        untraced_baseline_ms_token=args.baseline_ms, heldout=out,
        selected_cost_local=stats(tuned), selected_static_frequency=stats(frequency),
    )
    return result, dict(stages=GROUP_NAMES,
                        fields=["layer", "expert", "allocated_slot_bytes", "resident_in_cost_static",
                                "train_ms_per_token_per_GiB_by_stage", "heldout_ms_per_token_per_GiB_by_stage"],
                        rows=utility)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("transfer", type=Path)
    ap.add_argument("--train", type=int, help="prefix length; default is half the trace")
    ap.add_argument("--bootstrap", type=int, default=120)
    ap.add_argument("--seed", type=int, default=72431)
    ap.add_argument("--baseline-ms", type=float, help="matched untraced baseline ms/token; not used to fit")
    ap.add_argument("--profile", type=Path, help="original (not post-run) profile seed for recency tie-breaks")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--utility", type=Path, required=True)
    args = ap.parse_args()
    require(args.bootstrap > 1, "need at least two bootstrap draws")
    result, utility = evaluate(args)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    with args.utility.open("x") as f:
        # One row per physical expert, without archiving raw routes or timestamps.
        rows = utility.pop("rows")
        f.write(json.dumps(utility)[:-1] + ', "rows": [\n')
        f.write(",\n".join(json.dumps(row, separators=(",", ":")) for row in rows))
        f.write("\n]}\n")


if __name__ == "__main__":
    main()
