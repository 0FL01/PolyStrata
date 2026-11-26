// include/strata/kernels/gfx_arch.hpp - which AMD architecture names the gfx11 matrix-core code (and the gfx12 native
// prompt-expert kernels) was built for.
//
// The WMMA kernels select their intrinsic with `#if defined(__gfx1100__) || ... || defined(__gfx1151__)`: a gfx11 part
// outside that list (gfx1152, ...) gets an empty or trapping body.  The runtime gates therefore match the SAME
// list, not the "gfx11" prefix, so such a part falls back to hipBLASLt / the portable kernels instead of launching code
// that is not there.  Pure string matching (no HIP headers), usable from host code of any build.
#pragma once

#include <cstddef>
#include <cstring>
#include <initializer_list>

namespace strata::kernels {

/// `gcn` is hipDeviceProp_t::gcnArchName ("gfx1151", or "gfx1151:sramecc-:xnack-"): true when it is exactly `name`.
inline bool gfx_arch_is(const char* gcn, const char* name) {
    if (gcn == nullptr) return false;
    const std::size_t n = std::strlen(name);
    return std::strncmp(gcn, name, n) == 0 && (gcn[n] == '\0' || gcn[n] == ':');
}

/// The gfx11 targets the matrix-core kernels (prompt GEMM / attention / scorer / fused experts) are compiled for.
inline bool gfx_arch_is_gfx11_wmma(const char* gcn) {
    return gfx_arch_is(gcn, "gfx1100") || gfx_arch_is(gcn, "gfx1101") || gfx_arch_is(gcn, "gfx1102") || gfx_arch_is(gcn, "gfx1103") ||
           gfx_arch_is(gcn, "gfx1150") || gfx_arch_is(gcn, "gfx1151");
}

/// The gfx12 (RDNA4) targets the gfx12 matrix-core kernels are compiled for (`#if defined(__gfx1200__) ||
/// defined(__gfx1201__)`): the R9700 / 9070 (XT) are gfx1201, the 9060 (XT) gfx1200.
inline bool gfx_arch_is_gfx12_wmma(const char* gcn) { return gfx_arch_is(gcn, "gfx1200") || gfx_arch_is(gcn, "gfx1201"); }

/// Strix Halo (Ryzen AI Max, RDNA3.5): the part the gfx1151 defaults (docs/STRIX_HALO.md) are measured on.
inline bool gfx_arch_is_gfx1151(const char* gcn) { return gfx_arch_is(gcn, "gfx1151"); }

/// RDNA4 (Radeon AI PRO R9700 = gfx1201, RX 9070 / 9070 XT = gfx1201, RX 9060 = gfx1200).
inline bool gfx_arch_is_gfx12(const char* gcn) { return gfx_arch_is(gcn, "gfx1200") || gfx_arch_is(gcn, "gfx1201"); }

/// RDNA2 (gfx1030 RX 6800 / 6900, gfx1031 RX 6700 XT, gfx1032 RX 6600, gfx1033 Van Gogh, gfx1034 RX 6500 / 6400,
/// gfx1035 / gfx1036 Ryzen 6000 / 7000 iGPUs): one wave32 ISA with 64 KiB LDS, v_dot4_i32_i8 and DPP8. Device code
/// names the same targets (hip_compat/intrinsics.hpp; ggml's RDNA2 macro).
inline bool gfx_arch_is_rdna2(const char* gcn) {
    for (const char* name : {"gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"})
        if (gfx_arch_is(gcn, name)) return true;
    return false;
}

}  // namespace strata::kernels
