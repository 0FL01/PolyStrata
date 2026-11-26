#!/usr/bin/env python3
"""Greedy verifier replay and real target acceptance, with fail-closed per-arm checkpoints."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path.home() / ".cache/opencode-tmp/opencode"
DEFAULT_RUN = CACHE / "polystrata-runs/dflash2-recon"
CONTEXTS = {"2k": (2048, 4672), "16k": (16384, 4352), "64k": (65536, 3632),
            "128k": (131072, 2600), "199k": (None, 1536)}
WIDTHS = (1, 2, 4, 6, 8)
GEN = "temperature=0 top_p=1 top_k=0 min_p=0 seed=42"
RECEIPT_VERSION = 1  # bump when the interpretation of a request or a completed receipt changes


def digest(data):
    return hashlib.sha256(data).hexdigest()


def put(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        tmp = Path(f.name)
    tmp.replace(path)


def snapshot_prompt(path, suffix, steps):
    # STRSESS v1: fixed 64-byte header, 168-byte image metadata, then live.ids.
    # The engine additionally validates model/config/geometry and the payload checksum on RESTORE.
    with path.open("rb") as f:
        header = f.read(240)
        if len(header) != 240 or header[:8] != b"STRSESS\x01" or struct.unpack_from("<II", header, 8) != (1, 64):
            raise ValueError(f"unrecognized disk session header: {path}")
        tag, layout, max_context, vocab, width = struct.unpack_from("<5q", header, 64)
        n, = struct.unpack_from("<Q", header, 232)
        if (tag, layout, width) != (0x494e4b4c494e4732, 1, 2304) or n < 1:
            raise ValueError(f"not a native Inkling2 session: {path}")
        # GEN admits prompt + requested output IDs; its last emitted ID is never fed back.
        if n + 1 + steps + 1 > max_context:
            raise ValueError(f"snapshot has no room for {steps} decode rows plus the fresh input and last output ({path})")
        ids = struct.unpack(f"<{n}i", f.read(4 * n))
        st = os.fstat(f.fileno())
    if not 0 <= suffix < vocab or not all(0 <= t < vocab for t in ids):
        raise ValueError(f"invalid token ID in snapshot or suffix: {path}")
    identity = {"path": str(path.resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                "header_sha256": digest(header)}
    return (*ids, suffix), identity


def serve(req_path):
    req = json.loads(req_path.read_text())
    out = req_path.parent
    with (out / "stderr.log").open("x") as err, (out / "wire.log").open("x") as wire, \
         subprocess.Popen(req["argv"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=err, text=True, bufsize=1) as p:
        def read():
            s = p.stdout.readline()
            if not s:
                raise RuntimeError("unexpected EOF from engine")
            wire.write(s)
            wire.flush()
            if s.startswith(("ERR", "SERR", "FATAL")):
                raise RuntimeError(s.strip())
            return s.strip().split()

        def until(prefix):
            while (s := read())[0] != prefix:
                pass
            return s

        def send(line):
            p.stdin.write(line + "\n")
            p.stdin.flush()

        ready = until("READY")
        send("RESTORE " + req["snapshot"]["path"])
        restore = until("RESTORED")
        start = time.monotonic()
        send(f"GEN {req['gen']} {GEN} {req['prompt']}")
        resume, tokens, times = [], [], []
        while True:
            line = read()
            if line[0] == "DONE":
                done = line
                break
            if line[0] == "RESUME":
                resume.append(int(line[1]))
            if line[0] == "T":
                tokens.append(int(line[1]))
                times.append(time.monotonic())
        send("QUIT")
        p.stdin.close()
        if p.wait(timeout=30):
            raise RuntimeError(f"engine exited {p.returncode}")
    put(out / "wire.json", dict(ready=ready, restored=int(restore[1]), resume=resume, done=done,
                                tokens=tokens, ttft_s=times[0] - start, decode_s=times[-1] - times[0]))


def tagged(stderr, tag):
    return [line.split()[1:] for line in stderr.splitlines() if line.startswith(tag + " ")]


def first_difference(expected, observed):
    for i, (a, b) in enumerate(zip(expected, observed)):
        if a != b:
            return {"index": i, "expected": a, "observed": b}
    if len(expected) != len(observed):
        i = min(len(expected), len(observed))
        return {"index": i, "expected": expected[i] if i < len(expected) else None,
                "observed": observed[i] if i < len(observed) else None}
    return None


def request_digest(req):
    return digest(json.dumps(req, sort_keys=True).encode())


def checkpoint_key(req):
    # Source SHA stays in the receipt for provenance. Layout-compatible changes (e.g. using
    # reflink for the seed) do not invalidate an otherwise identical HIP run.
    logical = {k: v for k, v in req.items() if k != "runner_sha256"}
    logical.setdefault("receipt_version", 1)
    return request_digest(logical)


def clone_profile(src, dst):
    subprocess.run(["cp", "--reflink=always", str(src), str(dst)], check=True)


def validate(out, req, baseline):
    guard = out / "guard"
    if json.loads((guard / "stop-reasons.json").read_text()) != []:
        raise ValueError(f"guard stopped: {out}")
    result = json.loads((out / "wire.json").read_text())
    done = result["done"]
    ids = result["tokens"]
    n = req["prompt_count"]
    if (result["restored"], result["resume"], int(done[2]), int(done[8]), int(done[14])) != (n - 1, [n - 1], n, n - 1, 1):
        raise ValueError(f"wrong restore/resume position: {out}")
    if len(ids) != int(done[1]) or done[5] not in ("length", "stop") or len(ids) < 2:
        raise ValueError(f"incomplete generation: {out}")
    if done[5] == "length" and len(ids) != req["gen"]:
        raise ValueError(f"incorrect generation length: {out}")
    stderr = (out / "stderr.log").read_text()
    [load, end] = tagged(stderr, "RECON_PLACEMENT")
    if load[:1] != ["load"] or end[:1] != ["exit"] or load[1:] != end[1:] or len(load) < 2:
        raise ValueError(f"placement moved or was missing: {out}")
    placement = digest(" ".join(load[1:]).encode())
    if baseline is not None and placement != baseline["placement_sha256"]:
        raise ValueError(f"not the serial run's physical expert placement: {out}")
    warn = json.loads((guard / "warnings.json").read_text())
    vram = [json.loads(l)["free_mib"] for l in (guard / "vram.jsonl").read_text().splitlines()]
    result.update(placement_sha256=placement, gpu_slots=len(load)-1, warnings_fresh=sum(w["phase"] != "boot" for w in warn),
                  free_vram_min_mib=min(vram), mode=req["mode"], width=req["width"], fix=req["fix"],
                  prompt_count=n, draft_accepted=int(done[6]), draft_offered=int(done[7]),
                  gpu_expert_uses=int(done[9]), expert_uses=int(done[10]))
    result["wire_tps"] = (len(ids) - 1) / result["decode_s"]
    if baseline is None:
        if tagged(stderr, "RECON_RESULT"):
            raise ValueError("reference was not the native serial generation path")
        return result

    expected = baseline["tokens"]
    [[mode, width, position, rows, kept, elapsed, gpu, cpu]] = tagged(stderr, "RECON_RESULT")
    if (mode, int(width), int(position), int(kept)) != (req["mode"], req["width"], n, len(ids) - 1):
        raise ValueError(f"incorrect verifier metadata: {out}")
    checks = [(int(i), int(r), int(pick), float(margin)) for i, r, pick, margin in tagged(stderr, "RECON_CHECK")]
    if [c[0] for c in checks] != list(range(len(ids))):
        raise ValueError(f"incomplete prediction trace: {out}")
    picks = [c[2] for c in checks]
    windows = [(int(width), float(ms)) for width, ms in tagged(stderr, "RECON_WINDOW")]
    if sum(w for w, _ in windows) != int(rows) or any(not 1 <= w <= req["width"] for w, _ in windows):
        raise ValueError(f"incorrect verifier width trace: {out}")
    full = [ms for w, ms in windows if w == req["width"]]
    if mode == "forced":
        if ids != expected or not full or int(rows) != len(expected) - 1:
            raise ValueError(f"did not replay the requested complete reference under this width: {out}")
    elif ids != picks:
        raise ValueError(f"emitted sequence is not the verifier's accepted greedy sequence: {out}")
    mismatch = first_difference(expected, picks)
    if mismatch:
        if req["width"] == 1:
            raise ValueError(f"serial control was not repeatable: {mismatch} ({out})")
        i = mismatch["index"]
        mismatch["prediction_abs_pos"] = n + i
        if i < len(checks):
            mismatch.update(row=checks[i][1], winner_margin=checks[i][3])
    return result | dict(picks=picks, picks_matching=sum(a == b for a, b in zip(expected, picks)),
                         prediction_difference=mismatch, emitted_difference=first_difference(expected, ids),
                         window_sizes=[w for w, _ in windows], target_rows=int(rows), retained=int(kept),
                         probe_decode_ms=float(elapsed), probe_tps=int(kept) * 1000 / float(elapsed),
                         full_window_mean_ms=statistics.mean(full) if full else None,
                         full_window_p50_ms=statistics.median(full) if full else None,
                         gpu_expert_uses=int(gpu), expert_uses=int(gpu)+int(cpu))


def guarded_arm(arm, req, env, baseline, profile, gpu_lock, timeout):
    arm.mkdir(exist_ok=True)
    key = checkpoint_key(req)
    for prior in sorted(arm.glob("attempt-*/complete.json")):
        checkpoint = json.loads(prior.read_text())
        original = json.loads((prior.parent / "request.json").read_text())
        seed_path = str(prior.parent / "profile.seed")
        original["argv"] = ["<profile>" if arg == seed_path else arg for arg in original["argv"]]
        if checkpoint_key(original) != key:
            continue
        if checkpoint["key"] not in (key, request_digest(original)):
            raise ValueError(f"checkpoint does not match its original request: {prior}")
        result = validate(prior.parent, req, baseline)
        if result != checkpoint["result"]:
            raise ValueError(f"checkpoint does not match raw artifacts: {prior}")
        return result
    out = Path(tempfile.mkdtemp(prefix="attempt-", dir=arm))
    clone_profile(profile, out / "profile.seed")
    task = dict(req)
    task["argv"] = [str(out / "profile.seed") if x == "<profile>" else x for x in req["argv"]]
    put(out / "request.json", task)
    env = env | {"HOME": str(out), "XDG_CACHE_HOME": str(out / ".cache")}
    if baseline is not None:
        (out / "follow.csv").write_text(",".join(map(str, baseline["tokens"])))
        env = env | {"RECON_FOLLOW_FILE": str(out / "follow.csv")}
    put(out / "env.json", env)
    command = ["flock", "--exclusive", str(gpu_lock), sys.executable, str(ROOT / "rdna2-rnd/scripts/run_guarded.py"),
               "--warn-amdgpu-faults", "--output", str(out / "guard"), "--", "timeout", "--kill-after=15s", str(timeout),
               sys.executable, "-B", str(Path(__file__).resolve()), "--serve-request", str(out / "request.json")]
    # Cancellation of this coordinator cannot release flock while the guard's detached GPU process still runs.
    print(f"running {arm.parent.name}/{arm.name}: {out}", flush=True)
    rc = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True).wait()
    if rc:
        raise RuntimeError(f"guard/engine exited {rc}; stopped sweep, preserved attempt: {out}")
    result = validate(out, req, baseline)
    put(out / "complete.json", {"key": key, "result": result})
    return result


def sweep(a):
    run = a.run.resolve()
    output = run / "verifier" / a.series
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text())
    argv = [str(a.binary.resolve()), "--serve", *cfg["args"], "--no-draft", "--lookup", "0", "--eos-ids", "200006"]
    for flag, v in (("--expert-profile", "<profile>"), ("--expert-swaps", "0")):
        argv[argv.index(flag) + 1] = v
    if "--no-graphs" in argv or "--disk-sessions" not in argv:
        raise ValueError("need graphs and disk sessions enabled")
    env = cfg["env"] | dict(PATH="/usr/bin:/bin", LC_ALL="C.UTF-8", RECON_TARGET="1", STRATA_FREEZE_EXPERTS="1",
                            LD_LIBRARY_PATH=":".join(cfg["lib_dirs"]))
    env.pop("STRATA_INKLING_EXPERT_KV_MIB", None)
    identities = {"receipt_version": RECEIPT_VERSION, "binary_sha256": digest(a.binary.read_bytes()),
                  "profile_sha256": digest(a.profile.read_bytes()),
                  "runner_sha256": digest(Path(__file__).read_bytes()),
                  "guard_sha256": digest((ROOT / "rdna2-rnd/scripts/run_guarded.py").read_bytes())}
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for ctx in a.contexts:
            chunk, arena = CONTEXTS[ctx]
            snap = a.snapshot_root / f"mixed-{chunk}/session.bin" if chunk is not None else a.long_snapshot
            ids, snapshot = snapshot_prompt(snap, a.suffix_id, a.steps)
            req = identities | dict(argv=argv, snapshot=snapshot, prompt=",".join(map(str, ids)), prompt_count=len(ids))
            ctxpath = output / ctx
            ctxpath.mkdir(exist_ok=True)
            ctxenv = env | {"RECON_ARENA_MIB": str(a.arena_mib or arena)}

            def arm(mode, width, fix, base):
                name = f"{mode}-w{width}-f{fix}"
                gen = a.steps + 1 if base is None else len(base["tokens"])
                arm_env = ctxenv | {"RECON_ORACLE_W": str(width), "RECON_VERIFIER_FIX": str(fix)}
                if mode == "forced":
                    arm_env["RECON_FORCE"] = "1"
                ref = digest(json.dumps(base["tokens"]).encode()) if base else None
                areq = req | dict(gen=gen, mode=mode, width=width, fix=fix, reference_sha256=ref, env=arm_env)
                result = guarded_arm(ctxpath / name, areq, arm_env, base, a.profile, run / "gpu.lock", a.timeout)
                print(json.dumps({"ctx": ctx, "arm": name, "n": len(result["tokens"]),
                                  "slots": result["gpu_slots"], "tps": round(result["wire_tps"], 2),
                                  "first_diff": result.get("prediction_difference"),
                                  "accepted": result["draft_accepted"], "offered": result["draft_offered"]}), flush=True)
                return result

            base = arm("serial", 1, 0, None)
            for fix in a.fixes:
                for width in a.widths:
                    if len(base["tokens"]) <= width:
                        print(f"skipping {ctx} W{width}: serial EOS left no full window", flush=True)
                        continue
                    for mode in a.modes:
                        arm(mode, width, fix, base)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--serve-request":
        serve(Path(sys.argv[2]))
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--series", default="target-v2")
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--snapshot-root", type=Path, default=CACHE / "arch-campaign.XWM1yK")
    ap.add_argument("--long-snapshot", type=Path, default=CACHE / "swa-recon-20261125/seed200/session.bin")
    ap.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=list(CONTEXTS))
    ap.add_argument("--widths", nargs="+", type=int, choices=WIDTHS, default=WIDTHS)
    ap.add_argument("--fixes", nargs="+", type=int, choices=range(8), default=(0,))
    ap.add_argument("--modes", nargs="+", choices=("forced", "accepted"), default=("forced", "accepted"))
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--suffix-id", type=int, default=198, help="fixed fresh input ID after the saved tokens")
    ap.add_argument("--arena-mib", type=int, help="override the initial expert arena for every selected context")
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    if Path(a.series).name != a.series or a.series in ("", ".", "..") or a.steps < 1 or a.timeout < 1:
        ap.error("invalid series, steps or timeout")
    os.umask(0o077)
    sweep(a)


if __name__ == "__main__":
    main()
