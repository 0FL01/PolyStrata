#!/usr/bin/env python3
"""Validate and stream CP-02 visit records; write a CPU/VRAM timing summary."""

import argparse
import gzip
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path


HEADER = struct.Struct("<8s3Q4I")
PREFIX = struct.Struct("<HHI")
VISIT = struct.Struct("<18Q8q8I4f2H16B2BbB")
EVENT = struct.Struct("<2Q5q")
MAGIC = b"PSVST1\r\n"
EVENTS = {
    2: ("cover", "end rows_before rows_after bytes_added growth"),
    3: ("shrink_cache", "bytes_requested cached_before cached_after mapped_before mapped_after"),
    4: ("follow", "written swap_capacity cached frozen occupied_bytes"),
    5: ("swap_asked", "layer expert victim_layer victim_expert slot_offset"),
    6: ("upload_observed", "layer expert adds_slot queue_index slot_offset"),
    7: ("initial_resident", "layer expert slot_offset slot_bytes trade_class"),
    8: ("shrink_slot", "layer expert victim_layer victim_expert new_slot_offset"),
    9: ("initial_pending", "layer expert queue_index status slot_offset"),
}
HOST = ("pos serial host_begin_ns host_end_ns router_wait_ns route_ready_ns cpu_begin_ns cpu_ready_ns "
        "gate_begin_ns gate_end_ns down_begin_ns down_end_ns").split()
GPU = "gpu_visit_ms gpu_experts_ms gpu_join_ms gpu_dispatch_ms".split()
PLACEMENT = ("cpu", "gpu", "skipped")


class TraceError(ValueError):
    pass


def header(source):
    raw = source.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise TraceError("truncated header")
    magic, first, end, epoch_ns, layers, visits, experts, top = HEADER.unpack(raw)
    if magic != MAGIC or (layers, visits, experts, top) != (24, 48, 128, 8) or not 0 <= first < end:
        raise TraceError("unsupported header, geometry or capture range")
    return dict(kind="header", version=1, first=first, end=end, epoch_ns=epoch_ns,
                layers=layers, visits=visits, experts=experts, top=top)


def on(mask, e):
    return bool(mask[e // 64] & (1 << (e % 64)))


def mask_json(mask):
    return f"{mask[1]:016x}{mask[0]:016x}"


def read_visit(raw):
    x = VISIT.unpack(raw)
    resident, asked, sent = x[12:14], x[14:16], x[16:18]
    slot, weight, gpu = x[18:26], x[26:34], x[34:38]
    total_asked, total_sent = x[38:40]
    ids, place = x[40:48], x[48:56]
    visit, layer, core_pass, flags = x[56:60]
    if not 0 <= visit < 48:
        raise TraceError(f"bad visit id {visit}")
    expected_layer = visit if visit < 6 else 6 + (visit - 6) % 12 if visit < 42 else visit - 24
    expected_pass = (visit - 6) // 12 if 6 <= visit < 42 else -1
    if layer != expected_layer or core_pass != expected_pass or flags & ~3:
        raise TraceError(f"bad mapping or flags at visit {visit}")
    if len(set(ids)) != 8 or any(e >= 128 for e in ids):
        raise TraceError(f"bad expert ids at visit {visit}")
    for e, s, p in zip(ids, slot, place):
        slot_ok = s >= 0 if p == 1 else s == -1
        if p >= len(PLACEMENT) or on(resident, e) != (p == 1) or not slot_ok:
            raise TraceError(f"inconsistent placement/slot for expert {e} at visit {visit}")
    if (sum(m.bit_count() for m in asked) > total_asked or sum(m.bit_count() for m in sent) > total_sent or
            any(a & b for a, b in zip(asked, sent))):
        raise TraceError(f"inconsistent pending masks at visit {visit}")
    if any(not math.isfinite(t) or t < 0 for t in gpu):
        raise TraceError(f"bad GPU duration at visit {visit}")
    begin, end, wait, route_ready, cpu_begin, cpu_ready, gate0, gate1, down0, down1 = x[2:12]
    if not begin <= route_ready <= end or wait > route_ready - begin:
        raise TraceError(f"bad router timestamps at visit {visit}")
    if 0 in place:
        if not route_ready <= cpu_begin <= gate0 <= gate1 <= down0 <= down1 <= cpu_ready <= end:
            raise TraceError(f"bad CPU timestamps at visit {visit}")
    elif any(x[6:12]) or gpu[2] != 0:
        raise TraceError(f"CPU timing without CPU experts at visit {visit}")
    if 1 not in place and (gpu[1] != 0 or gpu[3] != 0):
        raise TraceError(f"GPU expert timing without GPU experts at visit {visit}")
    return dict(zip(HOST, x[:12]), kind="visit", visit=visit, layer=layer, core_pass=core_pass,
                windowed=bool(flags & 1), frozen=bool(flags & 2), ids=ids, weight_bits=weight,
                placement=tuple(PLACEMENT[p] for p in place), slot_offset=slot,
                resident_mask=mask_json(resident), asked_mask=mask_json(asked), sent_mask=mask_json(sent),
                total_asked=total_asked, total_sent=total_sent, **dict(zip(GPU, gpu)))


def records(source, meta):
    serial, pos, visit = 0, None, 47
    while prefix := source.read(PREFIX.size):
        offset = source.tell() - len(prefix)
        try:
            if len(prefix) != PREFIX.size:
                raise TraceError("truncated record prefix")
            kind, size, reserved = PREFIX.unpack(prefix)
            if (reserved or (kind != 1 and kind not in EVENTS) or
                    size != PREFIX.size + (VISIT.size if kind == 1 else EVENT.size)):
                raise TraceError(f"invalid record prefix: kind={kind}, size={size}, reserved={reserved}")
            raw = source.read(size - PREFIX.size)
            if len(raw) != size - PREFIX.size:
                raise TraceError("truncated record payload")
            if kind == 1:
                row = read_visit(raw)
                if row["serial"] != serial:
                    if visit != 47 or row["visit"] != 0 or row["serial"] <= serial:
                        raise TraceError("incomplete or out-of-order token frame")
                    serial, pos, visit = row["serial"], row["pos"], -1
                if row["pos"] != pos or row["visit"] != visit + 1:
                    raise TraceError("out-of-order visit")
                visit = row["visit"]
            else:
                if visit != 47:
                    raise TraceError("event interleaved in a token frame")
                name, fields = EVENTS[kind]
                ts, p, *data = EVENT.unpack(raw)
                row = dict(kind=name, host_ns=ts, pos=p, **dict(zip(fields.split(), data)))
            if not meta["first"] <= row["pos"] < meta["end"]:
                raise TraceError("record outside capture range")
            yield row
        except TraceError as exc:
            raise TraceError(f"byte {offset}: {exc}") from exc
    if visit != 47:
        raise TraceError("incomplete final token frame")


class Summary:
    def __init__(self):
        self.count = Counter()
        self.by_visit = defaultdict(Counter)
        self.by_cpu_count = defaultdict(Counter)
        self.first = self.last = None

    def take(self, row):
        self.count[row["kind"]] += 1
        if row["kind"] != "visit":
            return
        self.first = row["pos"] if self.first is None else min(self.first, row["pos"])
        self.last = row["pos"] if self.last is None else max(self.last, row["pos"])
        n_cpu = row["placement"].count("cpu")
        values = dict(visits=1, cpu_experts=n_cpu, gpu_experts=row["placement"].count("gpu"),
                      skipped_experts=row["placement"].count("skipped"),
                      host_visit_ms=(row["host_end_ns"] - row["host_begin_ns"]) / 1e6,
                      router_wait_ms=row["router_wait_ns"] / 1e6,
                      cpu_branch_ms=(row["cpu_ready_ns"] - row["cpu_begin_ns"]) / 1e6,
                      cpu_gate_ms=(row["gate_end_ns"] - row["gate_begin_ns"]) / 1e6,
                      cpu_down_ms=(row["down_end_ns"] - row["down_begin_ns"]) / 1e6,
                      **{k: row[k] for k in GPU})
        self.by_visit[row["visit"]].update(values)
        self.by_cpu_count[n_cpu].update(values)

    def result(self, meta):
        def averages(groups):
            return {key: dict(visits=d["visits"], **{f"mean_{k}": v / d["visits"] for k, v in d.items() if k != "visits"})
                    for key, d in sorted(groups.items())}
        return dict(header=meta, record_counts=self.count, token_frames=self.count["visit"] // meta["visits"],
                    first_position=self.first, last_position=self.last,
                    by_visit=averages(self.by_visit), by_cpu_count=averages(self.by_cpu_count))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path, help="PSVST1 binary, optionally gzip-compressed")
    ap.add_argument("--jsonl", help="write ordered mixed records as JSON Lines to a new file, or - for stdout")
    ap.add_argument("--summary", help="write the summary to a new file, or - for stdout (default without --jsonl)")
    ap.add_argument("--allow-empty", action="store_true", help="accept an otherwise valid trace with no visits")
    args = ap.parse_args()
    if args.jsonl == args.summary == "-":
        ap.error("choose only one stdout output")
    try:
        with ExitStack() as stack:
            def output(path):
                return sys.stdout if path == "-" else stack.enter_context(open(path, "x", encoding="utf-8"))

            read = gzip.open if args.trace.suffix == ".gz" else open
            source = stack.enter_context(read(args.trace, "rb"))
            meta = header(source)
            stats = Summary()
            out = output(args.jsonl) if args.jsonl else None
            if out:
                print(json.dumps(meta, separators=(",", ":")), file=out)
            for row in records(source, meta):
                stats.take(row)
                if out:
                    print(json.dumps(row, separators=(",", ":")), file=out)
            if not args.allow_empty and not stats.count["visit"]:
                raise TraceError("no visits captured; check the capture range and use single-row decode")
            if args.summary or not out:
                print(json.dumps(stats.result(meta), indent=2, allow_nan=False), file=output(args.summary or "-"))
    except BrokenPipeError:
        return 1
    except (TraceError, EOFError, OSError) as exc:
        ap.exit(1, f"{ap.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
