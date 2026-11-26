#!/usr/bin/env bash
# Call under run_guarded.py with the shared GPU lock held (see README).
set -euo pipefail
R=${1:?pass DFlash2 reconstruction directory}
M=${2:?pass Inkling model directory}
build=${3:?pass the stock/exact build directory}
out=${4:?pass the guard output directory}

# Measure attention only, with the same stock kernels and host selector in both arms.
export DF2_WAVE8=0 DF2_BOOKS=host DF2_BOOKS_TIMING=0

run() {
    local arm=$1 ctx=$2 profile=$3 tag=$4 n=$5; shift 5
    local label="$ctx-$profile-$arm" tap="$R/long-taps/$ctx/tap.dft1"
    local trace=()
    [[ $ctx != short ]] || tap="$R/taps/code_python.dft1"
    if [[ $tag == trace ]]; then
        trace=(rocprofv3 --kernel-trace --memory-copy-trace --hip-runtime-trace --stats -f csv -d "$out/$label" -o trace --)
    fi
    timeout --kill-after=10s 100 "${trace[@]}" "$build/$arm/dflash2-replay" \
        --target "$M/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf" \
        --model "$M/DFlash2/Inkling-2-Mini-DFlash2-$profile.gguf" --replay "$tap" \
        --mode all-positions --max-steps "$n" --dump-json "$out/$label-$tag.json" "$@"
}

timings=(short:Q4_0 2k:Q4_0 2k:Q4_K-mixed 64k:Q4_0 64k:Q4_K-mixed)
parities=("${timings[@]}" 2k:BF16 2k:Q8_0 2k:Q4_0-mixed)
for config in "${timings[@]}"; do
    ctx=${config%:*}; profile=${config#*:}
    for repeat in {1..5}; do
        if (( repeat % 2 )); then arms=(stock exact); else arms=(exact stock); fi
        for arm in "${arms[@]}"; do run "$arm" "$ctx" "$profile" "t$repeat" 25; done
    done
done

for config in "${parities[@]}"; do
    ctx=${config%:*}; profile=${config#*:}
    for arm in stock exact; do
        run "$arm" "$ctx" "$profile" parity 25 --dump-hidden --dump-steps "$out/$ctx-$profile-$arm.steps.jsonl"
    done
    cmp "$out/$ctx-$profile-stock.steps.jsonl" "$out/$ctx-$profile-exact.steps.jsonl"
done
sha256sum "$out"/*.steps.jsonl > "$out/parity.sha256"

# Trace the stages in a separate pass so profiling cannot affect the wall time A/B.
for config in 2k:Q4_0 2k:Q4_K-mixed; do
    ctx=${config%:*}; profile=${config#*:}
    for arm in stock exact; do run "$arm" "$ctx" "$profile" trace 16; done
done
