#!/usr/bin/env bash
# Run AFTER donor acquisition and quantization. No shared writable profile, no KV reuse.
set -euo pipefail
umask 077
: "${WORK:?new scratch directory}" "${HISTORY64:?original 64K coding history CSV}" "${HISTORY197:?original 197K coding history CSV}"
config=${CONFIG:-$HOME/.config/strata/inkling2-mini/strata-inkling-2-mini.json}
models=${MODELS:-$HOME/.lmstudio/models/unsloth/Inkling-2-Mini}
exe=${EXE:-build-poly-hip/strata-poly}
mkdir "$WORK"
run() {
    local arm=$1 length=$2 rep=$3 version=v2.1
    local extra=()
    [[ $arm == A ]] || version=v3
    [[ $rep == warm ]] || extra+=(--seed-profile "$WORK/$arm-2048-warm/profile")
    case $length in
        65536) extra+=(--history "$HISTORY64");;
        197000) extra+=(--history "$HISTORY197");;
    esac
    .venv/bin/python -m tools.inkling_weights.tideglass_timing --config "$config" --exe "$exe" \
        --model "$models/Tideglass-Inkling-2-Mini-Unbound-$version.gguf" --output "$WORK/$arm-$length-$rep" \
        --length "$length" "${extra[@]}"
}
run A 2048 warm; run B 2048 warm
for n in 2048 16384; do run A "$n" 1; run B "$n" 1; run B "$n" 2; run A "$n" 2; done
run A 65536 1; run B 65536 1
run B 197000 1; run A 197000 1
