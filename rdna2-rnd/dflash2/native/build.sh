#!/usr/bin/env bash
# Reuse the archives and ROCm installation of an existing PolyStrata gfx1031 HIP build.
set -euo pipefail
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$src/../../.." && pwd)
out=${1:?usage: bash build.sh OUTPUT_DIR [POLY_HIP_BUILD_DIR]}
build=$(realpath -- "${2:-$repo/build-poly-hip}")
mkdir -p -- "$out"
out=$(realpath -- "$out")
tmp=$(mktemp -d -- "$out/.dflash2-build.XXXXXX")
trap 'rm -f -- "$tmp/libstrata-dflash2.so" "$tmp/dflash2-replay"; rmdir -- "$tmp"' EXIT

cache() { sed -n "s/^$1:[^=]*=//p" "$build/CMakeCache.txt"; }
if [[ $(cache STRATA_ENABLE_HIP) != ON || $(cache CMAKE_HIP_ARCHITECTURES) != gfx1031 ]]; then
    echo 'Expected a PolyStrata HIP build for gfx1031.' >&2
    exit 1
fi
hipc=$(cache CMAKE_HIP_COMPILER)
rocm=$(cache CMAKE_HIP_COMPILER_ROCM_ROOT)
ggml=$(cache STRATA_GGML_DIR)
ggml=${ggml:-$build/_deps/strata_llamacpp-src}
read -r -a hip_flags <<< "$(cache CMAKE_HIP_FLAGS)"
common=(-std=c++20 -O3 -DNDEBUG -I"$repo/include" -isystem "$rocm/include"
        -I"$ggml/ggml/include" -L"$rocm/lib" -L"$rocm/lib64"
        -Wl,-rpath,"$rocm/lib" -Wl,-rpath,"$rocm/lib64")

"$hipc" "${common[@]}" "${hip_flags[@]}" --offload-arch=gfx1031 -fPIC -shared --hip-link \
    -Wl,--exclude-libs,ALL -Wl,-z,defs \
    -Wl,-soname,libstrata-dflash2.so "$src/runtime.hip" "$build/libstrata_kernels.a" \
    "$build/libstrata_core.a" "$build/ggml/src/libggml-base.a" \
    -o "$tmp/libstrata-dflash2.so"

"$hipc" "${common[@]}" -D__HIP_PLATFORM_AMD__ "$src/replay.cpp" -L"$tmp" -lstrata-dflash2 -lamdhip64 \
    -Wl,-rpath,'$ORIGIN' -o "$tmp/dflash2-replay"
mv -f -- "$tmp/libstrata-dflash2.so" "$tmp/dflash2-replay" "$out/"
