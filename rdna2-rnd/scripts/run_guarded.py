#!/usr/bin/env python3
"""Run one RDNA2 experiment with fail-closed VRAM and journal monitoring."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


class Guard:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.start = time.strftime("%Y-%m-%d %H:%M:%S")
        self.stop = threading.Event()
        self.unsafe = threading.Event()
        self.reasons = []
        self.proc = None
        self.watcher = None

    def terminate(self):
        proc = self.proc
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

    def check(self):
        card = Path("/sys/class/drm/card1/device")
        free = (int((card / "mem_info_vram_total").read_text()) -
                int((card / "mem_info_vram_used").read_text())) / 1048576
        with (self.directory / "vram.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "free_mib": free}) + "\n")
        if free < 1536:
            self.reasons.append(f"free VRAM below 1536 MiB: {free:.2f}")
            self.unsafe.set()
        for scope, options in (("kernel", ["-k"]), ("desktop", ["--user"])):
            p = subprocess.run(["journalctl", *options, "--since", self.start, "--no-pager", "-o", "json"],
                               capture_output=True, text=True, timeout=8, check=True)
            (self.directory / f"{scope}.jsonl").write_text(p.stdout)
            for line in p.stdout.splitlines():
                entry = json.loads(line)
                msg = entry.get("MESSAGE", "").lower()
                source = entry.get("_COMM", "") + entry.get("SYSLOG_IDENTIFIER", "")
                fault = ("amdgpu" in msg and any(x in msg for x in ("fault", "reset", "timeout")))
                fault |= any(x in msg for x in ("out of memory:", "oom-kill:", "killed process"))
                fault |= ("kwin" in source.lower() and any(x in msg for x in ("memory", "fault", "reset")))
                if fault:
                    self.reasons.append(f"fresh {scope} fault; see private journal")
                    self.unsafe.set()
                    break
        if self.unsafe.is_set():
            raise RuntimeError("; ".join(self.reasons))

    def monitor(self):
        while not self.stop.wait(2):
            try:
                self.check()
            except Exception as exc:
                self.reasons.append(f"monitor: {exc}")
                self.unsafe.set()
            if self.unsafe.is_set():
                self.terminate()
                return

    def __enter__(self):
        self.check()  # Refuse launch when the first safety sample is unavailable.
        self.watcher = threading.Thread(target=self.monitor)
        self.watcher.start()
        return self

    def __exit__(self, *_):
        try:
            self.terminate()
        finally:
            self.stop.set()
            self.watcher.join(timeout=40)
            if self.watcher.is_alive():
                self.reasons.append("monitor did not finish")
                self.unsafe.set()
            else:
                try:
                    self.check()  # Includes teardown and the last periodic sample's tail.
                except Exception as exc:
                    self.reasons.append(f"final check: {exc}")
                    self.unsafe.set()
            (self.directory / "stop-reasons.json").write_text(json.dumps(self.reasons, indent=2))
        if self.unsafe.is_set():
            raise RuntimeError("; ".join(self.reasons))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command or "--no-graphs" in command:
        ap.error("provide a command with graphs enabled")
    os.umask(0o077)
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, DEBUG_HIP_DYNAMIC_QUEUES="0")
    with Guard(a.output) as guard, (a.output / "command.log").open("w") as log:
        guard.proc = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
        code = guard.proc.wait()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
