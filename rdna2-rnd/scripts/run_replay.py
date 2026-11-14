#!/usr/bin/env python3
"""One pass per named arm; private fixed fixture/profile/config are supplied explicitly."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from run_guarded import Guard

ROOT = Path(__file__).resolve().parents[2]
FLAGS = ("STRATA_HIP_SPLIT_ATTEND", "STRATA_HIP_PREFILL_FP32", "STRATA_POLY_STAGED_TILES")
ARMS = {"baseline": {}, "split": {FLAGS[0]: "1"}, "prefill1": {FLAGS[1]: "1"},
        "prefill2": {FLAGS[1]: "2"}, "tiles": {FLAGS[2]: "8,16,32"}, "window4": {}, "window8": {},
        "mtp4": {}, "mtp8": {}}
SUMMARY = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in "
                     r"([\d.]+) ms \(([\d.]+) tok/s\), (\d+) generated in ([\d.]+) ms "
                     r"\(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")


def configure(cfg, arm, binary, profile, combo, combo_spec=None):
    c = copy.deepcopy(cfg)
    c["exe"], c["cwd"] = str(binary.resolve()), str(ROOT)
    for flag in FLAGS:
        c["env"].pop(flag, None)
    for selected in (combo if arm == "combo" else [arm]):
        c["env"].update(ARMS[selected])
    c["env"].pop("STRATA_RDNA2_IQ4_ROWS", None)  # Dormant on the user-selected mixed-quant model.
    c["env"].update(DEBUG_HIP_DYNAMIC_QUEUES="0", STRATA_HIP_VALUE_SPLIT="1")
    c["args"][c["args"].index("--expert-profile") + 1] = str(profile)
    for option, expected in (("--max-context", "200000"), ("--kv", "q8_0"),
                             ("--threads", "8"), ("--vram-reserve-mib", "3072")):
        if c["args"][c["args"].index(option) + 1] != expected:
            raise ValueError(f"expected {option} {expected}")
    if "--no-graphs" in c["args"]:
        raise ValueError("graphs must remain enabled")
    spec = combo_spec if arm == "combo" else arm
    guessing = spec is not None and spec.startswith(("window", "mtp"))
    c["args"][c["args"].index("--window") + 1] = spec.removeprefix("window").removeprefix("mtp") if guessing else "1"
    c["args"][c["args"].index("--lookup") + 1] = "3" if guessing else "0"
    if "--no-draft" in c["args"]:
        c["args"].remove("--no-draft")
    # UD-IQ4_NL actually has a draft layer (10496 vs 10240 experts). Keep the two
    # authorized speculation families distinct rather than attributing MTP to lookup.
    if spec is None or not spec.startswith("mtp"):
        c["args"].append("--no-draft")
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "binary", "python", "fixture", "profile", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="unique-id:arm, in execution order")
    ap.add_argument("--combo", nargs="+", choices=("split", "prefill1", "prefill2", "tiles"), default=[])
    ap.add_argument("--combo-spec", choices=("window4", "window8", "mtp4", "mtp8"))
    ap.add_argument("--needle", action="store_true", help="fixture is one structured retrieval request plus expected JSON")
    a = ap.parse_args()
    os.umask(0o077)
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    fixture = a.fixture.read_bytes()
    if not a.needle and hashlib.sha256(fixture).hexdigest() != "44b2a6a039fd407f0a135e0225a95854934971b6d3928de8954af75a7a4a1e1e":
        ap.error("unexpected replay fixture")
    if hashlib.sha256(a.profile.read_bytes()).hexdigest() != "b00c3e9d79c4fb8691050f1f72c4290caa77c44b578a162e468a20f19ed4590b":
        ap.error("unexpected initial profile")
    payload = json.loads(fixture)
    requests = [payload["request"]] if a.needle else payload
    for req in requests:
        req["model"] = cfg["model_name"]
    schedule = []
    for item in a.runs:
        name, arm = item.split(":")
        if not name or not all(x.isalnum() or x in "-_" for x in name) or arm not in (*ARMS, "combo"):
            ap.error("invalid run id or arm")
        if arm == "combo" and not a.combo:
            ap.error("combo requires --combo")
        schedule.append((name, arm))
    env = dict(os.environ, PYTHONUNBUFFERED="1", STRATA_REQUEST_LINES="1", DEBUG_HIP_DYNAMIC_QUEUES="0")
    for flag in (*FLAGS, "STRATA_FREEZE_EXPERTS", "STRATA_RDNA2_IQ4_ROWS"):
        env.pop(flag, None)
    manifest = {"binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(), "runs": schedule,
                "fixture_sha256": hashlib.sha256(fixture).hexdigest(), "combo": a.combo, "combo_spec": a.combo_spec}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with Guard(a.output) as guard:
        for name, arm in schedule:
            guard.check()
            run = a.output / name
            run.mkdir(mode=0o700)
            shutil.copy2(a.profile, run / "profile.txt")
            c = configure(cfg, arm, a.binary, run / "profile.txt", a.combo, a.combo_spec)
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            key = secrets.token_hex(32)
            c.update(log=str(run / "engine.log"), host="127.0.0.1", port=port, api_key=key, open_browser=False)
            (run / "config.json").write_text(json.dumps(c, indent=2))

            def http(path, req=None, timeout=5):
                q = urllib.request.Request(f"http://127.0.0.1:{port}" + path,
                    data=json.dumps(req).encode() if req is not None else None,
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                with urllib.request.urlopen(q, timeout=timeout) as response:
                    return json.load(response)

            with (run / "server.log").open("w") as log:
                guard.proc = subprocess.Popen([str(a.python), "serve/server.py", "--engine", "strata", "--config",
                    str(run / "config.json"), "--host", "127.0.0.1", "--port", str(port)],
                    cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                deadline = time.monotonic() + 300
                while True:
                    if guard.unsafe.is_set() or guard.proc.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"{name}: startup failed or timed out")
                    try:
                        if http("/health").get("loaded"):
                            break
                    except (OSError, urllib.error.URLError):
                        pass
                    time.sleep(1)
                children = Path(f"/proc/{guard.proc.pid}/task/{guard.proc.pid}/children").read_text().split()
                found = False
                for pid in children:
                    if Path(f"/proc/{pid}/exe").resolve() != a.binary.resolve():
                        continue
                    actual_env = dict(x.split(b"=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in x)
                    for flag, value in c["env"].items():
                        if actual_env.get(flag.encode()) != value.encode():
                            raise RuntimeError(f"{name}: wrong child environment: {flag}")
                    actual_args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    if b"--no-graphs" in actual_args:
                        raise RuntimeError("graphs disabled")
                    (run / "actual-engine.json").write_text(json.dumps({"argv": [x.decode() for x in actual_args], "env": c["env"]}, indent=2))
                    found = True
                if not found:
                    raise RuntimeError("engine child not identified")
                for turn, req in enumerate(requests):
                    guard.check()
                    before = time.monotonic()
                    answer = http("/v1/chat/completions", req, timeout=600)
                    (run / f"response-{turn}.json").write_text(json.dumps(answer))
                    if "error" in answer or (not a.needle and answer.get("usage", {}).get("completion_tokens") != 640):
                        raise RuntimeError(f"{name}/{turn}: incomplete response")
                    if a.needle:
                        actual = json.loads(answer["choices"][0]["message"]["content"])
                        passed = actual == payload["expected"] and answer["usage"]["prompt_tokens"] >= 44000
                        (run / "needle-result.json").write_text(json.dumps({"passed": passed,
                            "actual": actual, "expected": payload["expected"], "usage": answer["usage"]}, indent=2))
                        if not passed:
                            raise RuntimeError(f"{name}: long-context retrieval failed")
                    print(name, turn, "elapsed_s", round(time.monotonic() - before, 3), flush=True)
                guard.terminate()
                guard.proc = None
            guard.check()
            rows = []
            for match in SUMMARY.finditer((run / "engine.log").read_text()):
                p, cached, read, pp_ms, pp, out, tg_ms, tg, accepted, offered = match.groups()
                rows.append(dict(prompt=int(p), cached=int(cached), read=int(read), pp_ms=float(pp_ms),
                    pp=float(pp), output=int(out), tg_ms=float(tg_ms), tg=float(tg),
                    accepted=int(accepted or 0), offered=int(offered or 0)))
            if len(rows) != len(requests) or (not a.needle and any(row["output"] != 640 for row in rows)) or rows[0]["cached"] != 0:
                raise RuntimeError(f"{name}: missing or invalid native request metrics")
            if a.needle and rows[0]["read"] < 44000:
                raise RuntimeError(f"{name}: retrieval input shorter than 44k")
            (run / "metrics.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({"run": name, "arm": arm, "metrics": rows}), flush=True)


if __name__ == "__main__":
    main()
