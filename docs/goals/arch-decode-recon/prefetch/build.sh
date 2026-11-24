#!/usr/bin/env bash
# Single compiler process at a time; no GPU or engine executable.
set -eu
cd -- "$(dirname -- "$0")/../../../.."
D=docs/goals/arch-decode-recon/prefetch
B=${1:-build-poly-hip}
G=third_party/llama.cpp/ggml/include
git diff --exit-code c0dd0e52 -- src/poly/cpu_experts.cpp src/poly/cpu_experts.hpp \
    include/strata/artifact/gguf_reader.hpp third_party/llama.cpp CMakeLists.txt
mkdir -p "$D/out"
# Production Release host optimization/FP flags; the HIP compatibility include isn't used by this source.
c++ -O3 -DNDEBUG -std=c++20 -fPIE -Wall -Wextra -pthread -Iinclude -I"$G" \
    -c src/poly/cpu_experts.cpp -o "$D/out/cpu_experts.o"
c++ -O3 -DNDEBUG -std=c++20 -fPIE -Wall -Wextra -pthread -mclflushopt -Isrc/poly -Iinclude -I"$G" \
    "$D/bench.cpp" "$D/out/cpu_experts.o" "$B/ggml/src/libggml-cpu.a" "$B/ggml/src/libggml-base.a" \
    -ldl -o "$D/out/prefetch-bench"
sha256sum "$D/out/prefetch-bench" "$B/ggml/src/libggml-cpu.a" "$B/ggml/src/libggml-base.a"
