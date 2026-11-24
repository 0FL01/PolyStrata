# P3 — Real-file no-op copy and GPU parity

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

The copy and the full `noop-manifest.json` stay under `$WORK/p3`. For runtime parity use the pinned binary, locked standalone and sampling arguments, `$WORK/corpus/nll/2k-en/prefix.ids.csv` as `--tokens`, `--max-new 32`, isolated fresh writable copies of the fixed profile and no snapshots. Run each model in its own process with the unmodified guard and a 300 s timeout.

**Verdict: passed.** After the user's reboot at 20:01:11 +03:00 on 2026-11-24, the unmodified guard's `/usr/bin/true` probe passed with 10,927.6 MiB sampled free before inference. Both GGUFs loaded on the RX 6700 XT at the same placement (1,343/3,072 expert slots, 8 CPU threads), produced **32 identical raw greedy IDs**, and passed the guard. Raw stdout SHA-256: `fbd02b16906fc153bf25bb84fbc9c1f76bf55435abb9c9853a23bc4593bc7b7c`. Minimum sampled free VRAM was 2,012 MiB (original) and 2,011 MiB (copy), rounded to the nearest MiB. Private logs: `$WORK/p3-runtime.4ZRD6A/{original,no-op}`. The rates in these 32-token smoke logs are not P5 timing evidence. Production was already inactive after the reboot; its files were not changed. The no-op is available as the H1 parent; P4 and P5 still gate candidate work.
