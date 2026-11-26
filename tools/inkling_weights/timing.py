#!/usr/bin/env python3
"""Run a guarded 2K/16K Inkling weight timing pair. Stop production separately first."""

import argparse
from contextlib import suppress
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / "docs/goals/inkling2-weights/experiment-lock.json"
WORK = Path.home() / ".cache/opencode-tmp/opencode/inkling-weights"
GUARD = REPO / "rdna2-rnd/scripts/run_guarded.py"
CARD = Path("/sys/class/drm/card1/device")  # Same device as Guard.check().
LENGTHS = {"2k": 2048, "16k": 16384}
ORDER = "ABBABAAB"
SUFFIX_LEN = 512
Q4_FLAGS = ("STRATA_INKLING_Q4_0_SLOTS", "STRATA_INKLING_Q4_0_DENSE")
LOAD = (r"^strata-poly: loaded in (\d+\.\d) s; (\d+) of (\d+) routed experts in VRAM "
        r"\((\d+\.\d) GiB\), (\d+) CPU threads$")
TIMING = (r"^prompt (\d+) tokens (\d+\.\d) tok/s, generated (\d+) tokens (\d+\.\d\d) tok/s "
          r"\((\w+)\); experts from VRAM (\d+\.\d)%, CPU experts (\d+\.\d) ms/token, "
          r"router wait (\d+\.\d) ms/token, (\d+) experts swapped into VRAM, (\d+) left out, "
          r"(\d+) of (\d+) guessed tokens right$")
USAGE = ("peak_rss_kib", "major_page_faults", "minor_page_faults", "filesystem_inputs", "filesystem_outputs")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verified_bytes(path, expected):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, f"SHA-256 mismatch: {path}")
    return data


def verify_file(path, expected):
    with path.open("rb") as f:
        require(hashlib.file_digest(f, "sha256").hexdigest() == expected, f"SHA-256 mismatch: {path}")


def ids_csv(record, length, vocab):
    path = WORK / "corpus" / record["ids_path"]
    csv = verified_bytes(path, record["ids_sha256"]).decode("ascii")
    require(record["tokens"] == length and re.fullmatch(r"[0-9]+(?:,[0-9]+)*", csv), f"bad IDs: {path}")
    ids = csv.split(",")
    require(len(ids) == length and all(int(i) < vocab and int(i) != 200006 for i in ids), f"bad IDs: {path}")
    return csv


def one_line(pattern, text):
    lines = re.findall(pattern, text, re.MULTILINE | re.ASCII)
    require(len(lines) == 1, f"missing or duplicate CLI line: {pattern}")
    return lines[0]


def parse_cli(stdout, stderr, prompt_length, suffix, threads):
    # Pinned main.cpp once() writes exactly one CSV line to stdout and uses generated-1 for TG.
    require(stdout == suffix.encode("ascii") + b"\n", "CLI stdout is not the exact requested ID sequence")
    pp_n, pp, tg_n, tg, finish, hit, cpu, route, swaps, skipped, guessed, offered = one_line(TIMING, stderr)
    load, cached, total, gib, cpu_threads = one_line(LOAD, stderr)
    require((int(pp_n), int(tg_n), finish, int(guessed), int(offered)) == (prompt_length, SUFFIX_LEN, "length", 0, 0),
            "CLI reported incomplete or speculative work")
    require(int(cpu_threads) == threads and 0 <= int(cached) <= int(total), "unexpected CPU threads or expert capacity")
    dispatch = [line for line in stderr.splitlines()
                if any(line.startswith(f"strata-poly: {flag}=1:") and line.endswith(" on RDNA2") for flag in Q4_FLAGS)]
    require(all(any(f"{flag}=1:" in line for line in dispatch) for flag in Q4_FLAGS), "missing RDNA2 Q4_0 enable logs")
    pp, tg, hit = float(pp), float(tg), float(hit)
    require(math.isfinite(pp) and math.isfinite(tg) and pp > 0 and tg > 0 and 0 <= hit <= 100, "invalid CLI metrics")
    return dict(load_s=float(load), prefill_tokens=int(pp_n), prefill_tok_s=pp, prefill_ms_est=1000 * int(pp_n) / pp,
                emitted_tokens=int(tg_n), timed_decode_steps=int(tg_n) - 1, decode_tok_s=tg,
                decode_ms_est=1000 * (int(tg_n) - 1) / tg, expert_cache_experts_at_load=int(cached),
                total_experts=int(total), expert_cache_gib_at_load=float(gib), expert_vram_hit_pct_cli=hit,
                cpu_expert_ms_per_token=float(cpu), router_wait_ms_per_token=float(route), swapped_experts=int(swaps),
                skipped_experts=int(skipped), q4_0_enable_logs=dispatch)


def optional_int(path):
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def sample(sensors, phase):
    fields = {key: v for key, path in sensors.items() if (v := optional_int(path)) is not None}
    return dict(time=time.time(), phase=phase, **fields)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def wait_timed(proc, deadline, limit, record):
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            return proc.wait(timeout=min(remaining, 2))
        except subprocess.TimeoutExpired:
            if remaining <= 2:
                break
            record("running")
    raise subprocess.TimeoutExpired(proc.args[0], limit)


def run_one(guard_type, run, release, flags, sampling, model, seed, env, prefix, suffix, n, limit, total_mib, sensors):
    run.mkdir()
    guard_dir = run / "guard"
    guard_dir.mkdir()
    profile = run / "profile.txt"
    profile.write_bytes(seed)
    replacements = {"{model}": str(model), "{per_run_profile}": str(profile)}
    argv = [str(release), *(replacements.get(f, f) for f in flags), *sampling,
            "--tokens", prefix, "--follow", suffix, "--max-new", str(SUFFIX_LEN)]
    run_env = dict(HOME=str(run), TMPDIR=str(run), PATH="/usr/bin:/bin", LC_ALL="C", **env)
    timer = ["/usr/bin/time", "-o", "usage.txt", "-f", "%M %F %R %I %O"]
    write_json(run / "command.json", dict(argv=timer + argv, environment=run_env, cwd=str(run), timeout_s=limit))
    threads = int(flags[flags.index("--threads") + 1])
    with (run / "stdout.txt").open("wb") as out, (run / "stderr.log").open("wb") as err, \
            (run / "sensors.jsonl").open("w") as observed, guard_type(guard_dir) as guard:
        def record(phase):
            observed.write(json.dumps(sample(sensors, phase)) + "\n")
            observed.flush()

        record("before")
        require(not guard.unsafe.is_set(), "guard marked the device unsafe before launch")
        deadline = time.monotonic() + limit
        guard.proc = subprocess.Popen(timer + argv, cwd=run, env=run_env, stdin=subprocess.DEVNULL,
                                      stdout=out, stderr=err, start_new_session=True)
        code = None
        try:
            code = wait_timed(guard.proc, deadline, limit, record)
            record("after")
        finally:
            if code != 0 or guard.unsafe.is_set():
                # Guard monitors the whole process group; reap any timed child that outlives GNU time.
                with suppress(ProcessLookupError):
                    os.killpg(guard.proc.pid, signal.SIGKILL)
    require(code == 0, f"timed process exited {code}: {run}")
    metrics = parse_cli((run / "stdout.txt").read_bytes(), (run / "stderr.log").read_text(errors="replace"),
                        n, suffix, threads)
    usage = [int(field) for field in (run / "usage.txt").read_text().split()]
    require(len(usage) == len(USAGE) and usage[0] > 0 and min(usage) >= 0, "invalid process usage record")
    free = [json.loads(line)["free_mib"] for line in (guard_dir / "vram.jsonl").read_text().splitlines()]
    require(len(free) >= 2 and all(math.isfinite(x) and 1536 <= x <= total_mib for x in free), "invalid guard VRAM samples")
    result = dict(**metrics, **dict(zip(USAGE, usage)), guard_samples=len(free), min_free_vram_mib=min(free),
                  sampled_peak_card_vram_used_mib=total_mib - min(free))
    write_json(run / "metrics.json", result)
    return result


def distribution(values):
    med = statistics.median(values)
    return dict(values=values, median=med, min=min(values), max=max(values),
                mad=statistics.median(abs(x - med) for x in values))


def summarize(runs):
    summary = {}
    for key in ("load_s", "prefill_tok_s", "prefill_ms_est", "decode_tok_s", "decode_ms_est", "peak_rss_kib",
                "major_page_faults", "min_free_vram_mib", "sampled_peak_card_vram_used_mib",
                "expert_cache_experts_at_load", "expert_vram_hit_pct_cli"):
        a = [r[key] for arm, r in runs if arm == "A"]
        b = [r[key] for arm, r in runs if arm == "B"]
        summary[key] = dict(A=distribution(a), B=distribution(b), adjacent_B_minus_A=distribution(
            [y - x for x, y in zip(a, b)]))
        if key in ("prefill_tok_s", "decode_tok_s"):
            summary[key]["adjacent_B_vs_A_percent"] = distribution([100 * (y / x - 1) for x, y in zip(a, b)])
    return summary


def execute(args, out):
    lock_data = LOCK.read_bytes()
    lock = json.loads(lock_data)
    contract, standalone, hw = lock["benchmark"], lock["standalone"], lock["hardware"]
    require(lock["schema_version"] == 1 and contract["performance_order_each_length"] == ORDER and
            contract["prefix_lengths"] == list(LENGTHS.values()) and contract["forced_suffix_tokens"] == SUFFIX_LEN and
            contract["active_vocab"] == 200058, "lock does not describe this timing protocol")
    flags, sampling, env = standalone["engine_argv_before_workload"], standalone["sampling_argv"], lock["engine_env"]
    limit = contract["per_standalone_run_timeout_seconds"]
    require(flags.count("{model}") == flags.count("{per_run_profile}") == 1 and
            not set(flags) & {"--serve", "--disk-sessions", "--no-graphs", "--top-probs"} and
            flags[flags.index("--lookup") + 1] == "0" and "--no-draft" in flags and
            sampling == ["--temperature", "0", "--seed", "42"] and
            all(env[flag] == "1" for flag in Q4_FLAGS), "lock does not describe isolated, non-speculative gfx1031 runs")
    require(type(limit) is int and 0 < limit <= 300, "invalid run timeout")
    require(lock["gpu_guard"]["path"] == str(GUARD.relative_to(REPO)) and
            lock["gpu_guard"]["inkling_memory_fault_policy"] == "warn", "unexpected GPU guard contract")
    require((os.uname().nodename, os.uname().release) == (hw["host"], hw["kernel"]), "host/kernel does not match lock")
    pci = ":".join((CARD / name).read_text().strip().removeprefix("0x") for name in ("vendor", "device"))
    vram = int((CARD / "mem_info_vram_total").read_text())
    require(pci == hw["gpu_pci_id"] and vram == hw["gpu_vram_bytes"], "guard card does not match the locked GPU")
    active = subprocess.run(["systemctl", "--user", "show", "-p", "ActiveState", "--value", lock["production"]["unit"]],
                            check=True, capture_output=True, text=True).stdout.strip()
    require(active == "inactive", "production must be inactive for standalone timing")
    sensors = {f"{path.parent.name}_{path.name}": path for path in sorted((CARD / "hwmon").glob("hwmon*/*_input"))
               if path.name.startswith(("temp", "freq"))}

    corpus_file = LOCK.parent / lock["corpus"]
    corpus = json.loads(verified_bytes(corpus_file, lock["corpus_sha256"]))
    seed = verified_bytes(Path(standalone["profile_seed_path"]), standalone["profile_seed_sha256"])
    release = Path(lock["release_binary"]["path"])
    verify_file(release, lock["release_binary"]["sha256"])
    h0 = lock["sources"]["H0"]
    models, verified = {}, {}
    for arm in "ab":
        model, pin = getattr(args, f"model_{arm}"), getattr(args, f"sha256_{arm}")
        require(not pin or model is not None, f"--sha256-{arm} requires --model-{arm}")
        path = (model or Path(h0["local_path"])).resolve(strict=True)
        expected = pin or h0["verified_sha256"]
        require(re.fullmatch(r"[a-f0-9]{64}", expected) and path.is_file(), f"bad model path or SHA-256: {path}")
        if path not in verified:
            print(f"Verifying model SHA-256: {path}", flush=True)
            verify_file(path, expected)
            verified[path] = expected
        require(verified[path] == expected, f"conflicting SHA-256 for {path}")
        models[arm.upper()] = dict(path=str(path), sha256=expected, size_bytes=path.stat().st_size)

    workloads = {}
    for split, n in LENGTHS.items():
        perf = corpus["perf"][split]
        prefix, = (doc["prefix"] for doc in corpus["nll"] if doc["id"] == perf["nll_prefix_id"])
        suffix = perf["forced"]
        workloads[split] = (ids_csv(prefix, n, contract["active_vocab"]),
                            ids_csv(suffix, SUFFIX_LEN, contract["active_vocab"]))
        for kind, csv in zip(("prefix", "suffix"), workloads[split]):
            (out / f"{split}-{kind}.ids.csv").write_bytes(csv.encode("ascii"))

    orders = {split: ("AB" if split == "2k" else "BA") if args.quick else ORDER for split in workloads}
    write_json(out / "config.json", dict(lock_sha256=hashlib.sha256(lock_data).hexdigest(),
               guard_sha256=hashlib.sha256(GUARD.read_bytes()).hexdigest(), corpus_sha256=lock["corpus_sha256"],
               hardware=hw, boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
               release=lock["release_binary"], profile_seed_sha256=standalone["profile_seed_sha256"], models=models,
                warmup_order="AB", measured_order=orders if args.quick else ORDER,
                quick_screen=args.quick, amdgpu_fault_policy="warn", timeout_s=limit,
                sensor_paths={k: str(v) for k, v in sensors.items()}))
    sys.path.insert(0, str(GUARD.parent))
    sys.dont_write_bytecode = True
    from run_guarded import Guard
    guard_type = partial(Guard, warn_amdgpu_faults=True)

    # Guard uses journalctl by name. The child engine gets only the run's explicitly recorded environment.
    os.environ["PATH"] = "/usr/bin:/bin"
    result = {}
    for split, (prefix, suffix) in workloads.items():
        runs = []
        for i, arm in enumerate("AB" + orders[split]):
            run = out / (f"{split}-warmup-{i + 1}-{arm}" if i < 2 else f"{split}-{i - 1:02d}-{arm}")
            print(run.name, flush=True)
            metrics = run_one(guard_type, run, release, flags, sampling, Path(models[arm]["path"]), seed, env,
                              prefix, suffix, LENGTHS[split], limit, vram / 2**20, sensors)
            if i >= 2:
                runs.append((arm, metrics))
        result[split] = summarize(runs)
    write_json(out / "summary.json", result)
    for split, summary in result.items():
        tg = summary["decode_tok_s"]
        print(f"{split}: median TG A {tg['A']['median']:.2f}, B {tg['B']['median']:.2f} tok/s; "
              f"median adjacent B vs A {tg['adjacent_B_vs_A_percent']['median']:+.2f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="screen one measured A/B pair at each length, not a full timing matrix")
    for arm in "ab":
        p.add_argument(f"--model-{arm}", type=Path, help="GGUF path (default: locked H0)")
        p.add_argument(f"--sha256-{arm}", help="expected full GGUF SHA-256 (default: locked H0 SHA-256)")
    args = p.parse_args()
    os.umask(0o077)
    require(WORK.resolve() == WORK and WORK.is_dir() and WORK.stat().st_uid == os.getuid() and
            WORK.stat().st_mode & 0o077 == 0, f"experiment scratch root must be private: {WORK}")
    out = Path(tempfile.mkdtemp(prefix="timing-", dir=WORK))
    print(f"Private evidence: {out}", flush=True)
    try:
        execute(args, out)
    except (Exception, KeyboardInterrupt) as exc:
        message = f"{type(exc).__name__}: {exc}"
        write_json(out / "failure.json", dict(error=message))
        print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
