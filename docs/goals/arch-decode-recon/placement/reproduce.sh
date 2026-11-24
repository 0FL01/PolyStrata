#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo 'usage: reproduce.sh CAMPAIGN_SCRATCH NEW_OUTPUT_DIR' >&2
    exit 2
fi

D=$1
O=$2
R=$(dirname "$0")
mkdir "$O"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

replay() {
    local run=$1
    shift
    nice -n 17 python -B "$R/simulate.py" "$D/$run/visits.psv" "$D/guard-transfer/command.log" \
        --profile "$D/profile.seed" --output "$O/placement-$run.json" --utility "$O/utility-$run.json" "$@"
}

# Whole-window means from the untraced CP-01 legs in ../baseline.json. Never used for calibration.
replay trace-on --baseline-ms 30.4330911015625
replay trace-code-2048
replay trace-mixed-65536 --baseline-ms 30.91726859375
nice -n 17 python -B "$R/summarize.py" "$O" "$O"
