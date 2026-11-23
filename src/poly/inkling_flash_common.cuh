// src/poly/inkling_flash_common.cuh - what the opt-in flash kernels (inkling_attend_lean.cu, inkling_attend_h2.cu)
// share: inkling_attend.cu's constants and fetch/unpack helpers, and the fused DPP8 forms. Inside an unnamed
// namespace: each unit has its own copies.
#pragma once

#include "inkling_attend.hpp"

#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace strata::poly::inkling {
namespace {

constexpr int kHeads = 24, kHead = 128, kKv = 8, kGroup = kHeads / kKv, kRelative = 16, kExtent = 1024;
constexpr int kTokens = 16, kKeys = 16, kSplit = 8, kFlashThreads = kSplit * kTokens;
constexpr int kVec = kHead / 4 / kSplit;
constexpr int kRelPart = kRelative / kSplit;
constexpr int kSlice = kTokens + kKeys - 1;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kLoads = kKeys * (kHead / 4) / kFlashThreads;
int div_up(int a, int b) { return (a + b - 1) / b; }

__device__ float sink2(float x) { return fminf(fmaxf(x * kLog2e, -FLT_MAX), FLT_MAX); }
__device__ int64_t oldest(int64_t end, int window) { return window > 0 && end > window ? end - window : 0; }

template <bool Q8> struct Raw;
template <> struct Raw<true> {
    char4 k[kLoads], v[kLoads];
    __half ks[kLoads], vs[kLoads];
};
template <> struct Raw<false> {
    uint2 k[kLoads], v[kLoads];
};

template <bool Q8> __device__ void fetch(Raw<Q8>& raw, KvView k, KvView v, int64_t kb, int64_t kmin, int64_t kmax,
                                         int ring, int g, int tid) {
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const int at_i = tid + i * kFlashThreads, c = at_i / (kHead / 4), w = at_i % (kHead / 4);
        const int64_t key = kb + c;
        const bool in = key >= kmin && key <= kmax;
        const size_t at = in ? ((size_t) (ring ? key % ring : key) * kKv + g) * kHead + 4 * w : 0;
        if constexpr (Q8) {
            raw.k[i] = in ? *reinterpret_cast<const char4*>(k.data + at) : make_char4(0, 0, 0, 0);
            raw.v[i] = in ? *reinterpret_cast<const char4*>(v.data + at) : make_char4(0, 0, 0, 0);
            raw.ks[i] = in ? reinterpret_cast<const __half*>(k.scales)[at / 32] : __float2half(0.f);
            raw.vs[i] = in ? reinterpret_cast<const __half*>(v.scales)[at / 32] : __float2half(0.f);
        } else {
            raw.k[i] = in ? *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(k.data) + at) : make_uint2(0, 0);
            raw.v[i] = in ? *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(v.data) + at) : make_uint2(0, 0);
        }
    }
}

template <bool Q8> __device__ float4 unpack(const Raw<Q8>& raw, int i, bool value) {
    if constexpr (Q8) {
        const char4 x = value ? raw.v[i] : raw.k[i];
        const float s = __half2float(value ? raw.vs[i] : raw.ks[i]);
        return make_float4((float) x.x * s, (float) x.y * s, (float) x.z * s, (float) x.w * s);
    } else {
        const uint2 x = value ? raw.v[i] : raw.k[i];
        const __half2 a = *reinterpret_cast<const __half2*>(&x.x), b = *reinterpret_cast<const __half2*>(&x.y);
        return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
    }
}

// flash_kernel<Q8, true> of inkling_attend.cu with fewer instructions and the same arithmetic.
// - Its DPP8 reductions and broadcasts fused: the compiler emits a copy, the DPP move and the operation where one
//   v_add_f32_dpp (x[lane ^ X] + x, the same sum), v_max_f32_dpp or v_mov_b32_dpp does it. gfx10 has no VALU->DPP
//   data hazard, so the asm needs no wait states; it is not volatile, so the scheduler still interleaves it.
// - A full layer's far key blocks (every key 1024 or more positions before the block's first token: each relative
//   term is 0 and no key is masked) in a loop of their own that stages no table slice and adds no relative terms:
//   fmaf(r, 0, s) is s. Branches inside a block's work cost the scheduler more than they save, so the versions are
//   separate loops. On RX 6700 XT: full layer at 32K 160 -> 130 ms, at 196K 984 -> 758 ms, a window layer -8%.
template <int X> __device__ __forceinline__ float add_xor8(float v) {
    float r;
    if constexpr (X == 1) asm("v_add_f32_dpp %0, %1, %1 dpp8:[1,0,3,2,5,4,7,6]" : "=v"(r) : "v"(v));
    else if constexpr (X == 2) asm("v_add_f32_dpp %0, %1, %1 dpp8:[2,3,0,1,6,7,4,5]" : "=v"(r) : "v"(v));
    else asm("v_add_f32_dpp %0, %1, %1 dpp8:[4,5,6,7,0,1,2,3]" : "=v"(r) : "v"(v));
    return r;
}
template <int X> __device__ __forceinline__ float max_xor8(float v) {
    float r;
    if constexpr (X == 1) asm("v_max_f32_dpp %0, %1, %1 dpp8:[1,0,3,2,5,4,7,6]" : "=v"(r) : "v"(v));
    else if constexpr (X == 2) asm("v_max_f32_dpp %0, %1, %1 dpp8:[2,3,0,1,6,7,4,5]" : "=v"(r) : "v"(v));
    else asm("v_max_f32_dpp %0, %1, %1 dpp8:[4,5,6,7,0,1,2,3]" : "=v"(r) : "v"(v));
    return r;
}
template <int O> __device__ __forceinline__ float lean_owner8(float v) {
    float r;
#define STRATA_LEAN_OWNER(N)                                                                                           \
    if constexpr (O == N)                                                                                              \
        asm("v_mov_b32_dpp %0, %1 dpp8:[" #N "," #N "," #N "," #N "," #N "," #N "," #N "," #N "]" : "=v"(r) : "v"(v));
    STRATA_LEAN_OWNER(0) STRATA_LEAN_OWNER(1) STRATA_LEAN_OWNER(2) STRATA_LEAN_OWNER(3)
    STRATA_LEAN_OWNER(4) STRATA_LEAN_OWNER(5) STRATA_LEAN_OWNER(6) STRATA_LEAN_OWNER(7)
#undef STRATA_LEAN_OWNER
    return r;
}
__device__ __forceinline__ float lean_broadcast8(float v, int owner) {
    switch (owner) {
        case 0: return lean_owner8<0>(v);
        case 1: return lean_owner8<1>(v);
        case 2: return lean_owner8<2>(v);
        case 3: return lean_owner8<3>(v);
        case 4: return lean_owner8<4>(v);
        case 5: return lean_owner8<5>(v);
        case 6: return lean_owner8<6>(v);
        default: return lean_owner8<7>(v);
    }
}

}  // namespace
}  // namespace strata::poly::inkling
