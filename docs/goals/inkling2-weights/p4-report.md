# P4 — Inkling-2-Mini NLL observer

**Status:** host arithmetic and gfx1031 compile/link passed. On-model alignment, finite outputs and repeatability are **blocked**, not verified: [`experiment-lock.json`](experiment-lock.json) records `Guard.boot_clean()` rejecting the current boot after the `[gfxhub]` page fault. No GPU command, model inference, service change, or guard override ran for this checkpoint.

- Source: local plan §§5.2 and 7. Parent H0 SHA-256: `e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37` (already verified for P0).
- Runtime pin: `c014d6ecc79d65b724cf98aae554cbdaab8242c5`, Release HIP `gfx1031`, ROCm `7.13.0a20260515`. The runtime source and build inputs match the pin; the pre-experiment checkout `c0e42140` differs from it only in a report. Lock SHA-256: `f260b75d7e61d560be7a7f6f30ec3d449d5e361f4442a992086d19e2908f705d`. Corpus [manifest](corpus-manifest.json) SHA-256 at this checkpoint: `7cd7d79bf597f5219b34b3f2402cd37f64fb40118225594e5f1ea99cfbc382ca`.
- Changes: `tools/inkling_weights/nll_observer.cpp`, the separate `EXCLUDE_FROM_ALL` target `inkling-nll-observer` in `CMakeLists.txt`, and this report. No tensor, timing CLI, engine source, or default configuration changed.

## What was checked

From the repository root, the same arithmetic as the runner was compiled and run without HIP headers, libraries or device setup:

```sh
scratch=$(mktemp -d "$HOME/.cache/opencode-tmp/opencode/inkling-weights/p4-host.XXXXXX")
c++ -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror -DINKLING_NLL_HOST_ONLY \
  tools/inkling_weights/nll_observer.cpp -o "$scratch/nll-host"
"$scratch/nll-host" --self-check
```

`Inkling NLL arithmetic self-check passed (host)`: uniform active logits with a large offset and dominant padded rows, separated logits with an analytic loss near zero, first and last active IDs, rejection of `-1` and `200058`, and rejection of NaN and Inf even when the offending active row is not the target. All active logits participate, using FP64 differences, exponentials, accumulation and `log1p`. The 966 padded rows are not read by the observer; `Engine::eval` only copies the active rows to the host.

For a normal build, configure a separate Release HIP `gfx1031` directory using [AMD_HIP.md](../../AMD_HIP.md#build) and the same compiler/ROCm, GGML and build settings as `build-poly-hip`, then run `cmake --build "$build" --target inkling-nll-observer -j2`. For this checkpoint, scratch `~/.cache/opencode-tmp/opencode/inkling-weights/p4-build.2WkpAJ` contains `configure-command`, `configure.log`, `compile.log`, `observer-link-command` and `link.log`. The new source was compiled by Ninja from the scratch target. Its generated HIP linker command used the existing unchanged engine objects and libraries read-only to avoid a full rebuild. Shared source compile commands matched the scratch `strata-poly` rules, and relinking the existing baseline inputs to a scratch path reproduced the pinned SHA-256 `981d969b87193eed0e7b48d329b4816b34142d247e1fcbbad4c95e025d6a642a`. Both the original local build and shipped binaries still have that SHA. The observer was not in the default Ninja `all` target. Observer ELF SHA-256: `f90b724f8c2d5bf063e506061d80467c875112adc9323cdeb6912483167217bc`. Retained the observer and its build evidence; removed the duplicate baseline relink and host-only test executable.

## Static alignment and next check

The runner uses the fresh CLI prefill flow from `src/poly/main.cpp`: text-only `Engine::read` in chunks of up to `max_read()`, with the default transient checkpoint marks and KV setup; then `end_read()`, `eval(prefix_last, 1, true)` and `end_read()`. It does not load, park or reuse a session. Per target, it checks `Engine::pos()` before scoring the current logits, writes one row, then forces that ID with `eval(..., 1, true)`. Like `--follow --lookup 0 --no-draft`, there is no sampling or speculative decode. Unlike the CLI, it also feeds the last target, after recording its score. None of these static checks establishes on-model alignment or repeatability.

After an externally arranged clean boot and successful [P3 runtime check](p3-report.md), run the following **only through the unchanged guard**, with production stopped for the standalone runs. The observer takes `MODEL.gguf PROFILE PREFIX.ids.csv TARGET.ids.csv`; the files are the manifest's exact nonempty ASCII decimal CSV (no trailing newline). For a corpus run give one frozen prefix/target pair per process, not a concatenation or a re-tokenized prompt. TSV stdout contains zero-based `step`, absolute `position`, `target_id` and full-precision `nll`. Use output only from runs that exit successfully and have the expected row count and ID order.

```bash
set -euo pipefail
shopt -s inherit_errexit
repo=$PWD
lock=docs/goals/inkling2-weights/experiment-lock.json
art="$HOME/.cache/opencode-tmp/opencode/inkling-weights"
observer="$art/p4-build.2WkpAJ/inkling-nll-observer"
release=$(jq -r '.release_binary.path' "$lock")
h0=$(jq -r '.sources.H0.local_path' "$lock")
seed=$(jq -r '.standalone.profile_seed_path' "$lock")
limit=$(jq -r '.benchmark.per_standalone_run_timeout_seconds' "$lock")
printf '%s  %s\n' "$(jq -r '.release_binary.sha256' "$lock")" "$release" | sha256sum -c -
printf '%s  %s\n' "$(jq -r '.standalone.profile_seed_sha256' "$lock")" "$seed" | sha256sum -c -
mapfile -t locked_env < <(jq -r '.engine_env | to_entries[] | "\(.key)=\(.value)"' "$lock")
mapfile -t cli_flags < <(jq -r '.standalone.engine_argv_before_workload[]' "$lock")
mapfile -t sampler < <(jq -r '.standalone.sampling_argv[]' "$lock")
prefix="$art/corpus/nll/2k-ru/prefix.ids.csv"

run_p4() (
  label=$1; mode=$2; shift 2
  dir=$(mktemp -d "$art/p4-$label.XXXXXX")
  cp -- "$seed" "$dir/profile.txt"
  if [[ $mode == cli ]]; then
    for i in "${!cli_flags[@]}"; do
      case "${cli_flags[$i]}" in
        '{model}') cli_flags[$i]=$h0 ;;
        '{per_run_profile}') cli_flags[$i]="$dir/profile.txt" ;;
      esac
    done
    args=("$release" "${cli_flags[@]}" "${sampler[@]}" "$@")
  else
    args=("$observer" "$h0" "$dir/profile.txt" "$@")
  fi
  cd "$dir"
  env -i "HOME=$HOME" "PATH=$PATH" "TMPDIR=$dir" "${locked_env[@]}" \
    python3 "$repo/rdna2-rnd/scripts/run_guarded.py" --output guard -- \
    timeout --foreground --kill-after=10s "${limit}s" \
    sh -c 'exec "$@" > stdout.txt 2> stderr.log' sh "${args[@]}"
  printf '%s\n' "$dir"
)

# The greedy run picks diagnostic targets expected in top-k, not new corpus targets.
greedy=$(run_p4 greedy cli --tokens "$(cat "$prefix")" --max-new 4)
ids=$(cat "$greedy/stdout.txt")
[[ $ids =~ ^[0-9]+(,[0-9]+){3}$ && ,$ids, != *,200006,* ]]
printf %s "$ids" > "$greedy/diagnostic.ids.csv"
forced=$(run_p4 follow cli --tokens "$(cat "$prefix")" --max-new 4 --follow "$ids" --top-probs 64)
[[ $(cat "$forced/stdout.txt") == "$ids" ]]
obs1=$(run_p4 obs1 observer "$prefix" "$greedy/diagnostic.ids.csv")
obs2=$(run_p4 obs2 observer "$prefix" "$greedy/diagnostic.ids.csv")
cmp "$obs1/stdout.txt" "$obs2/stdout.txt"
python3 - "$prefix" "$ids" "$forced/stderr.log" "$obs1/stdout.txt" <<'PY'
import csv, math, sys
from pathlib import Path

prefix, ids, cli, observer = sys.argv[1:]
n = len(Path(prefix).read_text().split(','))
targets = list(map(int, ids.split(',')))
tops = [dict(item.split(':') for item in line.split()[1:])
        for line in Path(cli).read_text().splitlines() if line.startswith('top ')]
with open(observer) as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
assert len(tops) == len(rows) == len(targets)
for i, (top, row, target) in enumerate(zip(tops, rows, targets)):
    assert [int(row[key]) for key in ('step', 'position', 'target_id')] == [i, n + i, target], row
    p = float(top[str(target)])  # Fail if this target is not in the CLI's printed top-k.
    nll = float(row['nll'])
    assert math.isfinite(nll) and math.isclose(nll, -math.log(p), abs_tol=1e-5, rel_tol=0), (i, target, p, nll)
print('P4 diagnostic alignment passed')
PY
```

The `1e-5` absolute NLL allowance is for the CLI's `%.6g` probabilities and its FP32 subtraction before `exp`. Check that the CLI and observer logs show the same loaded placement; if the guard refuses, a target is absent from top-k, or the repeat/alignment comparison fails, record it instead of passing P4. The next step is this guarded H0 check, not candidate evaluation.
