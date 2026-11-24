"""Run a short CP-01 pipe fixture with rocprofv3 on the engine only; invoke under the RDNA2 guard."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline_run


class EngineTrace:
    # Replace only baseline_run's subprocess binding, not the global subprocess module.
    PIPE = subprocess.PIPE
    run = staticmethod(subprocess.run)

    def __init__(self, binary, prof, core_lib):
        self.binary = binary
        self.prof = prof
        self.core_lib = core_lib
        self.started = False

    def Popen(self, cmd, *, cwd, env, **kwargs):
        if self.started or Path(cmd[0]) != self.binary:
            raise RuntimeError(f"unexpected benchmark subprocess: {cmd!r}")
        self.started = True
        if any(k.startswith(("ROCPROF", "ROCTX")) for k in env) or "LD_PRELOAD" in env:
            raise RuntimeError("use the clean campaign environment, without inherited profiler settings")
        traced_env = dict(env, PYTHONDONTWRITEBYTECODE="1")
        # Campaign HIP/architecture libraries take precedence. The historical local recipe
        # appends the matching SDK core lib for rocprof dependencies.
        libs = traced_env["LD_LIBRARY_PATH"].split(":") if traced_env["LD_LIBRARY_PATH"] else []
        if str(self.core_lib) not in libs:
            libs.append(str(self.core_lib))
        traced_env["LD_LIBRARY_PATH"] = ":".join(libs)
        trace = Path(cwd).resolve() / "trace"
        trace.mkdir()
        version = subprocess.run([str(self.prof), "--version"], env=traced_env,
                                 capture_output=True, text=True, check=True).stdout.strip()
        # Request just these domains: this SDK's aggregate --runtime-trace also enables KFD tracing.
        traced_cmd = [str(self.prof), "--hip-runtime-trace", "--kernel-trace", "--memory-copy-trace",
                      "--output-format", "csv", "--output-directory", str(trace), "--output-file", "decode", "--", *cmd]
        (trace / "launch.json").write_text(json.dumps(dict(cmd=traced_cmd, version=version,
                                                            LD_LIBRARY_PATH=traced_env["LD_LIBRARY_PATH"]),
                                                       indent=2) + "\n")
        return subprocess.Popen(traced_cmd, cwd=cwd, env=traced_env, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("length", type=int)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--workload", choices=["mixed", "code", "prose"], default="mixed")
    p.add_argument("--prompt-csv", type=Path, help="reuse the exact CSV from the matched CP-01 run")
    for flag in ("binary", "config", "profile-seed"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    a = p.parse_args()
    if not 2 <= a.steps <= 64:
        p.error("request 2–64 serial decode steps")
    if "STRATA_VISIT_TRACE" in os.environ:
        p.error("unset STRATA_VISIT_TRACE before collecting the baseline")
    venv = baseline_run.ROOT / ".venv"
    prof = venv / "bin/rocprofv3"
    core_libs = sorted(venv.glob("lib/python*/site-packages/_rocm_sdk_core/lib"))
    if not os.access(prof, os.X_OK) or len(core_libs) != 1:
        p.error(f"expected rocprofv3 and one SDK core lib directory under {venv}")
    shim = EngineTrace(a.binary.resolve(), prof, core_libs[0])
    old_argv, old_subprocess = sys.argv, baseline_run.subprocess
    try:
        baseline_run.subprocess = shim
        sys.argv = [str(Path(baseline_run.__file__).resolve()), str(a.run), str(a.length), "--steps", str(a.steps),
                    "--workload", a.workload, "--binary", str(shim.binary), "--config", str(a.config),
                    "--profile-seed", str(a.profile_seed), "--profiled"]
        if a.prompt_csv:
            sys.argv += ["--prompt-csv", str(a.prompt_csv)]
        baseline_run.main()
    finally:
        sys.argv, baseline_run.subprocess = old_argv, old_subprocess
    if not list((a.run / "trace").rglob("*_kernel_trace.csv")):
        raise RuntimeError("rocprofv3 did not write a kernel trace; see engine.log")


if __name__ == "__main__":
    main()
