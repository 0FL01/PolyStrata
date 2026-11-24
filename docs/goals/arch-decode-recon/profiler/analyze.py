"""Analyze only serial Inkling decode from rocprofv3 CSVs (CPU only)."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re


FUSED = {"control_in_kernel": "control_out_kernel", "control_in3_kernel": "control_out3_kernel"}
HOIST_BUDGET = {"controller_input_fused", "controller_bank_mix", "controller_message_norm"}
CONTROLLER_STAGES = {
    "bank_load_rms_query_dot_softmax_mix": ["controller_input_fused", "controller_bank_mix"],
    "message_normalization": ["controller_input_fused", "controller_message_norm"],
    "current_residual_normalization": ["controller_input_fused", "controller_current_norm"],
    "rank64_projection": ["controller_rank"],
    "gain_forget_projection": ["controller_output_fused", "controller_gain_forget"],
    "controller_output": ["controller_output_fused", "controller_output_separate"],
}


@dataclass(slots=True, eq=False)
class Event:
    start: int
    end: int
    name: str
    symbol: str
    stream: str = ""
    agent: str = ""
    src: str = ""
    dst: str = ""
    correlation: str = ""
    grid: tuple = ()
    workgroup: tuple = ()
    vgprs: str = ""
    sgprs: str = ""
    lds: str = ""
    size: int | None = None
    category: str = ""


def symbol(name):
    return re.split(r"[<(]", name.replace("(anonymous namespace)::", ""), maxsplit=1)[0].split("::")[-1].split()[-1]


def read_events(path, domain):
    result = []
    if path is None:
        return result
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        names = {"kernel": ("Kernel_Name",), "copy": ("Direction", "Operation"),
                 "api": ("Function", "Operation", "Function_Name", "Name")}[domain]
        fields = set(reader.fieldnames or ())
        name_key = next((x for x in names if x in fields), None)
        if not name_key or not {"Start_Timestamp", "End_Timestamp"} <= fields:
            raise ValueError(f"unsupported {domain} CSV header in {path}: {reader.fieldnames}")
        for row in reader:
            name = row[name_key]
            start, end = (int(row[k]) for k in ("Start_Timestamp", "End_Timestamp"))
            if not name or end < start:
                raise ValueError(f"invalid {domain} row {reader.line_num} in {path}")
            size = next((int(row[k]) for k in ("Bytes", "Size_Bytes", "Copy_Bytes", "Transfer_Size", "Size")
                         if row.get(k)), None) if domain == "copy" else None
            result.append(Event(start, end, name, symbol(name) if domain == "kernel" else name,
                                row.get("Stream_Id", ""), row.get("Agent_Id", ""),
                                row.get("Source_Agent_Id", ""), row.get("Destination_Agent_Id", ""),
                                row.get("Correlation_Id", ""),
                                tuple(int(row.get(f"Grid_Size_{x}") or 0) for x in "XYZ"),
                                tuple(int(row.get(f"Workgroup_Size_{x}") or 0) for x in "XYZ"),
                                row.get("VGPR_Count", ""), row.get("SGPR_Count", ""),
                                row.get("LDS_Block_Size", ""), size))
    return sorted(result, key=lambda e: (e.start, e.end))


def copy_direction(name):
    norm = name.lower().replace("_", "").replace("-", "")
    return next((key for text, key in [("hosttodevice", "h2d"), ("devicetohost", "d2h"),
                                      ("devicetodevice", "d2d"), ("hosttohost", "h2h")]
                 if text in norm or key in norm), "unknown")


def projection(e):
    n = e.symbol
    return n.startswith(("gemv", "Cijk", "native_q6_k_warps", "native_q8_0_warps", "inkling_q4_0_segments")) or (
        "mmvq" in n and not any(x in n for x in ("slot", "indirect", "_ids"))
        and not any(x in e.name for x in ("MmvqSlots", "MmvqIds")))


def category(e):
    n = e.symbol
    if "inkling_q4_0_slots" in n:
        return "expert_q4_0"
    if "mmvq" in n and ("slot" in n or "indirect" in n or "_ids" in n
                       or "MmvqSlots" in e.name or "MmvqIds" in e.name):
        return "expert_other"
    if n in FUSED:
        return "controller_input_fused"
    if n in FUSED.values():
        return "controller_output_fused"
    if n == "control_kernel":
        return "controller_output_separate"
    if n.startswith(("gemv_rows_dual_kernel", "route_", "router_")):
        return "router"
    if projection(e) or "mmq" in n:
        return "projection_other"
    if any(x in n for x in ("flash", "attn", "attention")) or n in {
            "scores_kernel", "probs_kernel", "values_kernel", "decode_kernel", "decode_q8_kernel",
            "decode_combine_kernel"} or (n == "gate_kernel" and "::inkling::" in e.name):
        return "attention"
    if n.startswith(("heads_kv", "head_norm", "store_q8", "store2_q8", "rope", "kv_", "conv")):
        return "conv_residual_or_tail_moe" if n == "conv_add_kernel" else "attention_kv_conv"
    if n.startswith("__amd_rocclr"):
        return "runtime_blit_or_fill"
    if n.startswith(("mix", "axpy", "rms_norm", "native_rms_norm", "native_quantize", "native_add_rms_norm", "copy2")):
        return "norm_quant_mix_residual"
    if n.startswith(("add_scaled", "gate_", "situ_", "swiglu", "geglu", "moe_sum", "expert_weights")):
        return "moe_activation_or_sum"
    return "other"


def tag_stream(stream):
    matched, unmatched = Counter(), Counter()
    for i, e in enumerate(stream):
        n = e.symbol
        before = stream[max(0, i - 6):i]
        if n in FUSED:
            if i + 2 < len(stream) and stream[i + 2].symbol == FUSED[n] and projection(stream[i + 1]):
                stream[i + 1].category = "controller_rank"
                matched[n] += 1
            else:
                unmatched[n] += 1
        elif n == "control_kernel":
            # The unfused row updates are preceded by mix, current norm, message norm, down, gain, forget.
            if (len(before) == 6 and [x.symbol for x in before[:3]] == ["mix_kernel", "rms_norm_kernel",
                                                                        "rms_norm_kernel"]
                    and all(projection(x) for x in before[3:])):
                for x, tag in zip(before, ["controller_bank_mix", "controller_current_norm", "controller_message_norm",
                                           "controller_rank", "controller_gain_forget", "controller_gain_forget"]):
                    x.category = tag
                matched[n] += 1
            else:
                unmatched[n] += 1
        elif n == "softcap_kernel":
            e.category = "lm_head_softcap"
            # Engine::lin may scale the projection, and may quantize the head input.
            preceding = list(reversed(before))
            if preceding and preceding[0].symbol == "axpy_kernel":
                preceding.pop(0).category = "lm_head_projection"
            if preceding and projection(preceding[0]):
                preceding.pop(0).category = "lm_head_projection"
                if preceding and preceding[0].symbol == "native_quantize_q8_1_kernel":
                    preceding.pop(0).category = "lm_head_input"
                if preceding and preceding[0].symbol == "rms_norm_kernel":
                    preceding.pop(0).category = "lm_head_input"
                    if preceding and preceding[0].symbol == "mix_kernel":
                        preceding.pop(0).category = "lm_head_input"
                matched[n] += 1
            else:
                unmatched[n] += 1
    return matched, unmatched


def decode_windows(stream, copies, steps):
    heads = [(i, e) for i, e in enumerate(stream) if e.symbol == "softcap_kernel"]
    if len(heads) != steps + 1:
        raise ValueError(f"found {len(heads)} final-logits softcaps, expected {steps + 1} (prefill + {steps} decode)")
    # rocCLR can implement the logits D2H as a compute blit instead of a memory-copy record.
    ends = []
    for index, head in heads:
        following = stream[index + 1] if index + 1 < len(stream) else None
        blit = following is not None and following.symbol.startswith("__amd_rocclr_copyBuffer")
        after = stream[index + 2] if blit and index + 2 < len(stream) else following if not blit else None
        upper = after.start if after is not None else head.end + 10_000_000
        candidate = [c for c in copies if c.stream == head.stream and copy_direction(c.name) == "d2h"
                     and head.end <= c.start <= c.end <= upper]
        if len(candidate) == 1:
            c = candidate[0]
            c.category = "copy_logits_d2h"
            if blit:
                following.category = "lm_head_logits_blit"
            ends.append((max(c.end, following.end if blit else head.end), "d2h_record"))
        elif blit:
            following.category = "lm_head_logits_blit"
            ends.append((following.end, "rocclr_copybuffer"))
        else:
            ends.append((head.end, "softcap_only"))
    methods = Counter(method for _, method in ends)
    # Do not mix transfer-anchored boundaries and bare softcaps within one estimate.
    times = [head.end for _, head in heads] if methods["softcap_only"] else [t for t, _ in ends]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("logits boundaries are not strictly increasing")
    return list(zip(times, times[1:])), dict(methods), "softcap_end" if methods["softcap_only"] else "logits_copy_end"


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(b - a for a, b in intervals)


def clipped(events, start, end):
    return [(e, max(start, e.start), min(end, e.end)) for e in events if e.start < end and e.end > start]


def spans(rows):
    return [(a, b) for _, a, b in rows]


def group_summary(rows, key, steps):
    groups = defaultdict(list)
    for e, a, b in rows:
        groups[key(e)].append((a, b))
    return {k: dict(count=len(v), interval_sum_ms_per_token=duration(v) / 1e6 / steps,
                    union_ms_per_token=duration(union(v)) / 1e6 / steps)
            for k, v in sorted(groups.items(), key=lambda kv: -duration(kv[1]))}


def per_window_stats(kernels, copies, windows):
    result = []
    # Every event is sorted by start. Revisit crossing intervals without dropping activity from other streams.
    ki = ci = 0
    for start, end in windows:
        while ki < len(kernels) and kernels[ki].end <= start:
            ki += 1
        while ci < len(copies) and copies[ci].end <= start:
            ci += 1
        k_end, c_end = ki, ci
        while k_end < len(kernels) and kernels[k_end].start < end:
            k_end += 1
        while c_end < len(copies) and copies[c_end].start < end:
            c_end += 1
        k = spans(clipped(kernels[ki:k_end], start, end))
        c = spans(clipped(copies[ci:c_end], start, end))
        wall = end - start
        kernel_busy, copy_busy, device_busy = [duration(union(intervals)) for intervals in (k, c, k + c)]
        result.append(dict(wall_ns=wall, kernel_busy_ns=kernel_busy, kernel_idle_ns=wall - kernel_busy,
                           copy_busy_ns=copy_busy, kernel_copy_overlap_ns=kernel_busy + copy_busy - device_busy,
                           traced_device_busy_ns=device_busy, uncovered_by_kernel_or_copy_ns=wall - device_busy))
    return result


def api_type(e):
    if re.match(r"hip(?:Ext|Hcc)?(?:Module)?Launch|hipGraphLaunch", e.name):
        return "launch_api"
    if "Synchronize" in e.name:
        return "synchronize_api"
    if re.match(r"hip(?:Memcpy|Memset)", e.name):
        return "memory_api"
    return "other_hip_api"


def inspect(kernel_path, copy_path, hip_path, agent_path, steps):
    kernels = read_events(kernel_path, "kernel")
    if not kernels:
        raise ValueError("no kernel dispatch records")
    agents = {e.agent for e in kernels}
    if len(agents) != 1:
        raise ValueError(f"expected one kernel agent, got {sorted(agents)}")
    agent = next(iter(agents))
    gpu = None
    if agent_path:
        with agent_path.open(newline="") as f:
            gpu = next((r for r in csv.DictReader(f) if r.get("Agent_Type") == "GPU" and
                        (agent == f"Agent {r.get('Node_Id')}" or agent == r.get("Node_Id"))), None)
        if not gpu or gpu.get("Name") != "gfx1031":
            raise ValueError(f"expected traced agent {agent!r} to be gfx1031: {gpu}")
    copies = [e for e in read_events(copy_path, "copy") if agent in {e.src, e.dst}]
    for e in copies:
        e.category = "copy_" + copy_direction(e.name)
    streams = defaultdict(list)
    for e in kernels:
        e.category = category(e)
        streams[e.stream].append(e)
    main = [s for s in streams.values() if any(e.symbol == "softcap_kernel" for e in s)]
    if len(main) != 1:
        raise ValueError(f"expected one serial logits stream, found {len(main)}")
    stream = main[0]
    windows, boundary_methods, boundary = decode_windows(stream, copies, steps)
    start, end = windows[0][0], windows[-1][1]
    # Tag only the decode kernels; retain boundary markers from decode_windows.
    matched, unmatched = tag_stream([e for e in stream if e.start >= start and e.end <= end])
    rows = clipped(kernels, start, end)
    copy_rows = clipped(copies, start, end)
    bins = group_summary(rows, lambda e: e.category, steps)
    device = per_window_stats(kernels, copies, windows)
    totals = {k.removesuffix("_ns") + "_ms_per_token": sum(r[k] for r in device) / 1e6 / steps for k in device[0]}
    ctr = [(e, a, b) for e, a, b in rows if e.category.startswith("controller_")]
    budget = [(e, a, b) for e, a, b in ctr if e.category in HOIST_BUDGET]
    others = [(e, a, b) for e, a, b in rows if e.category not in HOIST_BUDGET] + copy_rows
    budget_ns = duration(union(spans(budget)))
    exclusive_ns = duration(union(spans(budget + others))) - duration(union(spans(others)))
    serial_fraction = budget_ns / (end - start)
    commands = clipped(read_events(hip_path, "api"), start, end)
    hip = None
    if hip_path:
        launch = [(e, a, b) for e, a, b in commands if api_type(e) == "launch_api"]
        corr = defaultdict(set)
        for e, _, _ in rows + copy_rows:
            if e.correlation and e.correlation != "0":
                corr[e.correlation].add(e.category.startswith("controller_"))
        ctr_launch = [(e, a, b) for e, a, b in launch if corr[e.correlation] == {True}]
        hip = dict(by_group=group_summary(commands, api_type, steps),
                   by_api=group_summary(commands, lambda e: e.name, steps),
                   api_union_ms_per_token=duration(union(spans(commands))) / 1e6 / steps,
                   controller_only_correlated_launch_calls=len(ctr_launch),
                   controller_only_correlated_launch_api_union_ms_per_token=
                   duration(union(spans(ctr_launch))) / 1e6 / steps)
    shape_resources = Counter((e.symbol, e.grid, e.workgroup, e.vgprs, e.sgprs, e.lds) for e, _, _ in ctr)
    controlled_streams = (bins.get("controller_input_fused", {}).get("count", 0)
                          + 2 * sum(e.symbol == "control_in3_kernel" for e, _, _ in ctr)
                          + bins.get("controller_output_separate", {}).get("count", 0))
    copy_bytes = defaultdict(lambda: [0, 0])
    for e, _, _ in copy_rows:
        if e.size is not None:
            item = copy_bytes[copy_direction(e.name)]
            item[0] += e.size
            item[1] += 1
    return dict(inputs=dict(kernel=str(kernel_path), copies=str(copy_path) if copy_path else None,
                            hip_api=str(hip_path) if hip_path else None, agent_info=str(agent_path) if agent_path else None),
                agent=agent, gpu=gpu, decode_steps=steps, boundary=boundary, boundary_matches=boundary_methods,
                logits_stream=stream[0].stream, window_ns=[start, end], per_token=device, mean=totals,
                kernels_by_category=bins, kernels_by_name=group_summary(rows, lambda e: e.name, steps),
                copies_by_category=group_summary(copy_rows, lambda e: e.category, steps) if copy_path else None,
                copy_bytes_where_reported={k: dict(bytes=v[0], records_with_bytes=v[1]) for k, v in copy_bytes.items()},
                hip_api=hip,
                controller=dict(stage_sources_not_additive=CONTROLLER_STAGES,
                                matched_decode_sequences=dict(matched), unmatched_decode_sequences=dict(unmatched),
                                observed_logical_stream_rows=controlled_streams, expected_logical_stream_rows=96 * steps,
                                attributable_dispatch_union_ms_per_token=duration(union(spans(ctr))) / 1e6 / steps,
                                hoist_input_dispatch_union_ms_per_token=budget_ns / 1e6 / steps,
                                hoist_input_exclusive_device_ms_per_token=exclusive_ns / 1e6 / steps,
                                optimistic_zero_cost_removal_traced_speedup_factor=1 / (1 - serial_fraction)
                                if serial_fraction < 1 else None,
                                hardware_occupancy=None,
                                launch_shapes=[dict(symbol=k[0], grid=k[1], workgroup=k[2], vgprs=k[3], sgprs=k[4],
                                                    lds_bytes=k[5], count=v) for k, v in sorted(shape_resources.items())]))


def read_json(path):
    return json.loads(path.read_text())


def compare(run, baseline):
    traces, plain = (read_json(p / "metrics.json") for p in (run, baseline))
    traced_cfg, plain_cfg = (read_json(p / "config.json") for p in (run, baseline))
    fields = ("binary_sha256", "config_sha256", "prompt_sha256", "seed_sha256", "length", "workload")
    mismatch = [k for k in fields if k not in traced_cfg or k not in plain_cfg or traced_cfg[k] != plain_cfg[k]]
    if mismatch:
        raise ValueError(f"untraced leg does not match campaign inputs: {', '.join(mismatch)}")
    n = min(traces["decode_steps"], plain["decode_steps"])
    if n < 1 or any(len(m["greedy_ids"]) < n + 1 or len(m["step_ms"]) < n for m in (traces, plain)):
        raise ValueError("missing matched-prefix IDs or step times")
    first = next((dict(zero_based_index=i, traced=x, untraced=y) for i, (x, y) in
                  enumerate(zip(traces["greedy_ids"][:n + 1], plain["greedy_ids"][:n + 1])) if x != y), None)
    return dict(untraced_run=str(baseline), matched_steps=n, compared_ids=n + 1, first_divergence=first,
                traced_mean_arrival_ms=sum(traces["step_ms"][:n]) / n,
                untraced_mean_arrival_ms=sum(plain["step_ms"][:n]) / n,
                untraced_full_run_decode_t_s=plain["decode_t_s"])


def find_csv(path, *patterns, required=False):
    matches = {p for pattern in patterns for p in path.rglob(pattern)}
    if len(matches) != 1 and (matches or required):
        raise ValueError(f"expected one CSV in {path} matching {patterns}, found {len(matches)}")
    return next(iter(matches), None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path, help="collector run directory with metrics.json and trace/")
    mode.add_argument("--kernel", type=Path, help="historical or externally captured kernel CSV")
    p.add_argument("--steps", type=int, help="required with --kernel: serial steps excluding the prefill-generated ID")
    p.add_argument("--copies", type=Path)
    p.add_argument("--hip-api", type=Path)
    p.add_argument("--agent-info", type=Path)
    p.add_argument("--untraced", type=Path, help="optional CP-01 run for matched inputs/greedy-prefix comparison")
    a = p.parse_args()
    try:
        if a.run:
            if any(v is not None for v in (a.steps, a.copies, a.hip_api, a.agent_info)):
                p.error("--run discovers CSVs and the decode count; other inputs are for --kernel")
            d = a.run / "trace"
            a.steps = int(read_json(a.run / "metrics.json")["decode_steps"])
            a.kernel = find_csv(d, "*_kernel_trace.csv", required=True)
            a.copies = find_csv(d, "*_memory_copy_trace.csv")
            a.hip_api = find_csv(d, "*_hip_api_trace.csv", "*_hip_runtime_api_trace.csv", required=True)
            a.agent_info = find_csv(d, "*_agent_info.csv", required=True)
        if not a.steps or a.steps < 1 or (a.untraced and not a.run):
            p.error("provide a positive --steps with --kernel; --untraced requires --run")
        output = inspect(a.kernel, a.copies, a.hip_api, a.agent_info, a.steps)
        if a.untraced:
            output["baseline_comparison"] = compare(a.run, a.untraced)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
