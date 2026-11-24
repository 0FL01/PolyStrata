# P3 — Real-file no-op copy (GPU parity pending)

Parent H0: `e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37`. See the [experiment lock](experiment-lock.json) for the pinned timing binary, profile and configuration.

```sh
WORK="$HOME/.cache/opencode-tmp/opencode/inkling-weights"
H0="$HOME/.lmstudio/models/unsloth/Inkling-2-Mini/Inkling-2-Mini-Q4_0.gguf"
mkdir -m 700 "$WORK/p3"
ionice -c3 nice -n 15 /usr/bin/time -v python -B -m tools.inkling_weights.transplant build \
  --variant H0-noop --base "$H0" --output "$WORK/p3/H0-noop-Inkling-2-Mini-Q4_0.gguf" \
  --manifest "$WORK/p3/noop-manifest.json"
```

The writer used the same descriptor serialization, alignment and payload-copy path as the H candidates; no plain filesystem copy. The independent reader reread the staged header and the writer reread every payload. All 704 payload SHA-256 values matched the pinned H0 scan; the typed metadata block matched and has SHA-256 `caa3e3718d3f917d4f213abac37a9b61c9fbb33fa0ef8c869c07ab8f0e669cc5`. The result is also container-identical: **15,064,977,088 bytes, full SHA-256 equal to H0**. The original is untouched. `/usr/bin/time -v` reported 125,480 KiB peak RSS for the offline Python writer, not runtime model memory.

The copy and full `noop-manifest.json` stay under `$WORK/p3`: the next step must load the copy alongside a separate original-H0 run and verify identical raw greedy IDs. Neither model need be modified for that check. Use the pinned binary, locked standalone and sampling arguments, `$WORK/corpus/nll/2k-en/prefix.ids.csv` as `--tokens`, `--max-new 32`, isolated fresh writable copies of the fixed profile and no snapshots. Compare raw CLI stdout and require both runs to load and emit the expected number of IDs. Start the GPU process only after a new `Guard.boot_clean()` passes, with the timeout inside the guard or via `wait(timeout=...)` under the guard's context manager. Restore unchanged production after the two standalone runs. The copy is the parent for H1 only after this gate is met.

**Verdict: container verified, runtime blocked.** The [boot gate](p0-report.md#preflight-blockers) refused this boot. No standalone HIP invocation, production restart or model promotion took place. The writer's staging files and the superseded initial no-op copy/sidecar were removed after the final copy passed; the remaining no-op GGUF, manifest and seed profile are required resumption artifacts, not deployed weights.
