#!/usr/bin/env python3
"""Run one RDNA2 experiment with fail-closed VRAM and journal monitoring."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


def journal_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(type(x) is int and 0 <= x <= 255 for x in value):
        return bytes(value).decode("utf-8", errors="replace")
    raise TypeError("unexpected journal text field type")


# Fault words as whole words, plurals included: "PERMISSION_FAULTS: 0x5" and "Pageflip timed out" are faults too,
# and "default" in a boot line is not.  The journal is read from a cursor, not from a start time: a boot whose clock
# was corrected after it started has its first lines dated ahead of the corrected time.
# An underscore is a separator here, so "PERMISSION_FAULTS" is read as two words.
FAULT_WORDS = re.compile(r"\b(faults?|resets?|timeouts?|timed?[- ]outs?|wedge[ds]?|hangs?|hanging|hung)\b")
# What the kernel writes when the GPU itself is in trouble.  On 2026-11-23 an RX 6700 XT hung five minutes after a
# page fault of a user process, and again four minutes into the next boot; both resets failed and the screen stayed
# black.  After such a line an experiment waits for a reboot.
GPU_TROUBLE = re.compile(r"(ring \S+ timeout|device wedged|gpu (reset|recovery failed|hang|fault)|page fault|"
                         r"permission[ _]faults|mapping[ _]error|vm.?fault)")


def is_fault(msg, amdgpu_line):
    if amdgpu_line and (FAULT_WORDS.search(msg.replace("_", " ")) or GPU_TROUBLE.search(msg)):
        return True
    return any(x in msg for x in ("out of memory:", "oom-kill:", "killed process"))


class Guard:
    def __init__(self, directory, accept_before=None):
        self.directory = Path(directory)
        # Epoch seconds: faults of this boot logged before it were reviewed by the user and do not refuse the launch.
        self.accept_before = accept_before
        self.start = time.strftime("%Y-%m-%d %H:%M:%S")
        c = subprocess.run(["journalctl", "-n", "1", "--show-cursor", "--no-pager"], capture_output=True,
                           text=True, timeout=8, check=True)
        m = re.search(r"^-- cursor: (\S+)$", c.stdout, re.M)
        if not m:
            raise RuntimeError("no journal cursor")
        self.cursor = m.group(1)
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
            p = subprocess.run(["journalctl", *options, "--cursor", self.cursor, "--no-pager", "-o", "json"],
                               capture_output=True, text=True, timeout=8, check=True)
            (self.directory / f"{scope}.jsonl").write_text(p.stdout)
            for line in p.stdout.splitlines():
                entry = json.loads(line)
                msg = journal_text(entry.get("MESSAGE", "")).lower()
                source = journal_text(entry.get("_COMM", "")) + journal_text(entry.get("SYSLOG_IDENTIFIER", ""))
                fault = is_fault(msg, "amdgpu" in msg)
                # re.search() yields None on no match (and a Match on one): both would poison the bool below
                fault |= "kwin" in source.lower() and FAULT_WORDS.search(msg) is not None
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

    def boot_clean(self):
        p = subprocess.run(["journalctl", "-k", "-b", "0", "--no-pager", "-o", "json"], capture_output=True,
                           text=True, timeout=8, check=True)
        for line in p.stdout.splitlines():
            entry = json.loads(line)
            msg = journal_text(entry.get("MESSAGE", "")).lower()
            if self.accept_before is not None and int(entry["__REALTIME_TIMESTAMP"]) < self.accept_before * 1e6:
                continue
            if "amdgpu" in msg and GPU_TROUBLE.search(msg):
                raise RuntimeError("the GPU took a fault or a reset earlier in this boot; reboot first: " +
                                   msg.strip()[:160])

    def __enter__(self):
        self.boot_clean()
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
    ap.add_argument("--accept-faults-before", metavar="'YYYY-MM-DD HH:MM:SS'",
                    help="ignore this boot's GPU faults logged before this local time (the user accepted them); "
                         "faults after it, and any fresh fault during the run, still stop the run")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command or "--no-graphs" in command:
        ap.error("provide a command with graphs enabled")
    os.umask(0o077)
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, DEBUG_HIP_DYNAMIC_QUEUES="0")
    accept = None
    if a.accept_faults_before:
        accept = time.mktime(time.strptime(a.accept_faults_before, "%Y-%m-%d %H:%M:%S"))
    with Guard(a.output, accept) as guard, (a.output / "command.log").open("w") as log:
        guard.proc = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
        code = guard.proc.wait()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
