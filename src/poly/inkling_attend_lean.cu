// src/poly/inkling_attend_lean.cu - STRATA_INKLING_FLASH_LEAN=1, see inkling_attend_lean.hpp.
// Its own file: next to flash_kernel in one translation unit, a second kernel changes how the compiler builds
// flash_kernel, and the default path keeps its code. The constants and the fetch/unpack helpers mirror
// inkling_attend.cu's.
#include "inkling_attend_lean.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

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

template <bool Q8>
__global__ void __launch_bounds__(kFlashThreads)
flash_lean_kernel(const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ table,
                  const float* __restrict__ sink, const float* __restrict__ gate, KvView k, KvView v, int64_t pos,
                  int n, int window, int ring, float* __restrict__ out) {
    __shared__ float4 ks[kKeys][kHead / 4];
    __shared__ float4 vs[kKeys][kHead / 4];
    __shared__ float ts[kSlice * kRelative];
    const int tid = threadIdx.x, part = tid % kSplit, tl = tid / kSplit, g = blockIdx.y;
    const int t0 = blockIdx.x * kTokens, last = min(t0 + kTokens, n) - 1, t = min(t0 + tl, last);
    const bool live = t0 + tl < n;
    const int64_t p = pos + t, lo = oldest(p + 1, window);
    const int64_t kmin = oldest(pos + t0 + 1, window), kmax = pos + last;
    const size_t qh0 = (size_t) t * kHeads + g * kGroup;

    float qr[kGroup][4 * kVec], o[kGroup][4 * kVec], rr[kGroup][kRelPart], m[kGroup], l[kGroup], lc[kGroup];
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        const float4* qv = reinterpret_cast<const float4*>(q + (qh0 + r) * kHead);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            const float4 x = qv[part + kSplit * i];
            qr[r][4 * i] = x.x, qr[r][4 * i + 1] = x.y, qr[r][4 * i + 2] = x.z, qr[r][4 * i + 3] = x.w;
            o[r][4 * i] = o[r][4 * i + 1] = o[r][4 * i + 2] = o[r][4 * i + 3] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < kRelPart; ++i) rr[r][i] = rel[(qh0 + r) * kRelative + part * kRelPart + i] * kLog2e;
        m[r] = sink2(sink[qh0 + r]);
        l[r] = lc[r] = 0.f;
    }

    Raw<Q8> raw;
    fetch<Q8>(raw, k, v, kmin / kKeys * kKeys, kmin, kmax, ring, g, tid);
    // One block of keys, as flash_kernel<Q8, true> takes it; Near stages the table slice and adds the relative terms,
    // Mask masks the keys a token must not see. Both are true but for the far blocks.
    const auto tile = [&]<bool Near, bool Mask>(int64_t kb) __attribute__((always_inline)) {
        __syncthreads();   // the block before has been read
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int at_i = tid + i * kFlashThreads, c = at_i / (kHead / 4), w = at_i % (kHead / 4);
            ks[c][w] = unpack<Q8>(raw, i, false);
            vs[c][w] = unpack<Q8>(raw, i, true);
        }
        if (kb + kKeys <= kmax) fetch<Q8>(raw, k, v, kb + kKeys, kmin, kmax, ring, g, tid);   // under this block's work
        if constexpr (Near) {
            // ts[j]: the table row of distance pos + t0 - (kb + kKeys - 1) + j, zero past the bias's extent
            const int64_t dlo = pos + t0 - (kb + kKeys - 1);
            for (int i = tid; i < kSlice * kRelative; i += kFlashThreads) {
                const int64_t d = dlo + i / kRelative;
                ts[i] = d >= 0 && d < kExtent ? table[d * kRelative + i % kRelative] : 0.f;
            }
        }
        __syncthreads();

        float s[kGroup][kKeys / kSplit];
#pragma unroll
        for (int r = 0; r < kGroup; ++r) s[r][0] = s[r][1] = 0.f;
#pragma unroll
        for (int c = 0; c < kKeys; ++c) {
            float dot[kGroup] = {};
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const float4 x = ks[c][part + kSplit * i];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    dot[r] = fmaf(qr[r][4 * i], x.x, dot[r]);
                    dot[r] = fmaf(qr[r][4 * i + 1], x.y, dot[r]);
                    dot[r] = fmaf(qr[r][4 * i + 2], x.z, dot[r]);
                    dot[r] = fmaf(qr[r][4 * i + 3], x.w, dot[r]);
                }
            }
            const float* tr = ts + (tl + kKeys - 1 - c) * kRelative + part * kRelPart;
#pragma unroll
            for (int r = 0; r < kGroup; ++r) {
                float score = dot[r] * (0.08838834764831845f * kLog2e);
                if constexpr (Near) {
#pragma unroll
                    for (int i = 0; i < kRelPart; ++i) score = fmaf(rr[r][i], tr[i], score);
                }
                score = add_xor8<1>(score);
                score = add_xor8<2>(score);
                score = add_xor8<4>(score);
                const int64_t key = kb + c;
                if (part == c % kSplit) s[r][c / kSplit] = Mask && (key < lo || key > p) ? -INFINITY : score;
            }
        }
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            float top = fmaxf(s[r][0], s[r][1]);
            top = max_xor8<1>(top);
            top = max_xor8<2>(top);
            top = max_xor8<4>(top);
            const float mn = fmaxf(m[r], top), shrink = exp2f(m[r] - mn);
            m[r] = mn;
            l[r] *= shrink;
            lc[r] *= shrink;
#pragma unroll
            for (int i = 0; i < 4 * kVec; ++i) o[r][i] *= shrink;
            s[r][0] = exp2f(s[r][0] - mn);
            s[r][1] = exp2f(s[r][1] - mn);
        }
        float sum[kGroup] = {};
#pragma unroll
        for (int c = 0; c < kKeys; ++c) {
            float prob[kGroup];
#pragma unroll
            for (int r = 0; r < kGroup; ++r) {
                prob[r] = lean_broadcast8(s[r][c / kSplit], c % kSplit);
                sum[r] += prob[r];
            }
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const float4 x = vs[c][part + kSplit * i];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    o[r][4 * i] = fmaf(prob[r], x.x, o[r][4 * i]);
                    o[r][4 * i + 1] = fmaf(prob[r], x.y, o[r][4 * i + 1]);
                    o[r][4 * i + 2] = fmaf(prob[r], x.z, o[r][4 * i + 2]);
                    o[r][4 * i + 3] = fmaf(prob[r], x.w, o[r][4 * i + 3]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            const float y = sum[r] - lc[r], next = l[r] + y;
            lc[r] = (next - l[r]) - y;
            l[r] = next;
        }
    };
    int64_t kb = kmin / kKeys * kKeys;
    // Far blocks first: the block's least distance pos + t0 - (kb + kKeys - 1) is kExtent or more. Their skipped terms
    // are fmaf(r, 0, s) = s only for a finite r (an infinite one makes NaN): with any, the whole block takes the
    // general loop, as flash_kernel does.
    bool finite = true;
#pragma unroll
    for (int r = 0; r < kGroup; ++r)
#pragma unroll
        for (int i = 0; i < kRelPart; ++i) finite = finite && isfinite(rr[r][i]);
    const bool far_ok = __syncthreads_and(finite) != 0 && window == 0;
    const int64_t far_end = far_ok ? std::min<int64_t>(kmax + 1, pos + t0 - kExtent - kKeys + 2) : kb;
    for (; kb < far_end; kb += kKeys) tile.template operator()<false, false>(kb);
    for (; kb <= kmax; kb += kKeys) tile.template operator()<true, true>(kb);
    if (!live) return;
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        l[r] += exp2f(sink2(sink[qh0 + r]) - m[r]);
        const size_t at = (qh0 + r) * kHead;
#pragma unroll
        for (int i = 0; i < kVec; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const size_t d = at + 4 * (part + kSplit * i) + j;
                out[d] = o[r][4 * i + j] / l[r] / (1.f + expf(-gate[d]));
            }
    }
}

}  // namespace

void attend_flash_lean(const Attend& a, cudaStream_t st) {
    if (a.n <= 0) return;
    const dim3 grid(div_up(a.n, kTokens), kKv);
    if (a.k.scales)
        flash_lean_kernel<true><<<grid, kFlashThreads, 0, st>>>(a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n,
                                                            a.window, a.ring, a.out);
    else
        flash_lean_kernel<false><<<grid, kFlashThreads, 0, st>>>(a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos,
                                                             a.n, a.window, a.ring, a.out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("Inkling lean flash attention: ") + cudaGetErrorString(e));
}

}  // namespace strata::poly::inkling
