#!/usr/bin/env bash
# Build only the ggml CPU libraries and the x86-64 AVX2 reference adapter.
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "Usage: $0 BUILD_DIRECTORY" >&2
    exit 2
fi
ref_dir=$(cd -- "$(dirname -- "$0")" && pwd)
out=$1
cmake -S "$ref_dir" -B "$out" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$out" --target ggml_qbridge --parallel "${BUILD_JOBS:-4}"
