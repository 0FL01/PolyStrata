#!/usr/bin/env python3
"""Strict 16K/512 continuation on the production-capacity contract, one guarded run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from run_guarded import Guard
from run_replay import ARMS, FLAGS, ROOT, configure

sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "binary", "profile", "source", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--arm", choices=(*ARMS, "combo"), default="baseline")
    ap.add_argument("--combo", nargs="+", choices=("split", "prefill1", "prefill2", "tiles"), default=[])
    a = ap.parse_args()
    if a.arm.startswith(("window", "mtp")):
        ap.error("speculation has a separate non-forced series")
    if a.arm == "combo" and not a.combo:
        ap.error("combo requires --combo")
    os.umask(0o077)
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(a.profile, a.output / "profile.txt")
    c = configure(json.loads(a.config.read_text()), a.arm, a.binary, a.output / "profile.txt", a.combo)
    tokens = Tokenizer.from_dir(Path(c["tokenizer"])).encode(a.source.read_text(), parse_special=False)
    prompt, follow = tokens[:16384], tokens[16384:16384 + 512]
    if len(prompt) != 16384 or len(follow) != 512:
        ap.error("fixed text must provide 16384+512 tokens")
    args = c["args"] + ["--tokens", ",".join(map(str, prompt)), "--follow", ",".join(map(str, follow)),
                        "--max-new", "512", "--temperature", "0"]
    env = dict(os.environ)
    for flag in (*FLAGS, "STRATA_FREEZE_EXPERTS", "STRATA_RDNA2_IQ4_ROWS"):
        env.pop(flag, None)
    env.update(c["env"])
    env["LD_LIBRARY_PATH"] = ":".join(c["lib_dirs"])
    result = {"arm": a.arm, "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(a.source.read_bytes()).hexdigest(),
              "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
              "follow_sha256": hashlib.sha256(json.dumps(follow).encode()).hexdigest(),
              "profile_seed_sha256": hashlib.sha256(a.profile.read_bytes()).hexdigest(),
              "capacity": 200000, "kv": "q8_0", "env": c["env"], "valid": False}
    try:
        with Guard(a.output) as guard, (a.output / "stdout.txt").open("w") as out, \
                (a.output / "stderr.txt").open("w") as err:
            guard.proc = subprocess.Popen([str(a.binary), *args], env=env, stdout=out, stderr=err,
                                          start_new_session=True)
            result["exit_code"] = guard.proc.wait(timeout=300)
        protocol = (a.output / "stdout.txt").read_text().strip()
        result["forced_ids_match"] = protocol == ",".join(map(str, follow))
        match = re.search(r"^prompt (\d+) tokens ([\d.]+) tok/s, generated (\d+) tokens ([\d.]+) tok/s \((\w+)\)",
                          (a.output / "stderr.txt").read_text(), re.MULTILINE)
        if match:
            result["metrics"] = dict(prompt=int(match[1]), pp=float(match[2]), output=int(match[3]), tg=float(match[4]))
        result["valid"] = (result["exit_code"] == 0 and result["forced_ids_match"] and
                           result.get("metrics", {}).get("prompt") == 16384 and
                           result.get("metrics", {}).get("output") == 512)
    finally:
        (a.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
