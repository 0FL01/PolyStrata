// src/poly/inkling_attend_h2.cu - STRATA_INKLING_FLASH_H2=1, see inkling_attend_lean.hpp.
// A precision change: the products' operands are rounded to halves (11 bits; the Q8_0 cache itself keeps 8), so the
// output differs from the FP32 kernels by a few 1e-4 of its magnitude. Its own file, as inkling_attend_lean.cu.
#include "inkling_attend_lean.hpp"
#include "inkling_flash_common.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strata::poly::inkling {
namespace {

// The products in packed halves: K in shared memory as halves of the dequantized values, V as halves of key pairs
// (c, c + 8) for each dimension, Q in registers as half pairs, P as the half pair a lane owns (keys part and part + 8,
// so one DPP8 move broadcasts a pair). v_dot2_f32_f16 multiplies two halves at a time into FP32 sums, twice the
// FP32 FMA rate on gfx1031; the scores' scale, the relative terms, the softmax and the denominators stay FP32, the
// denominator a tree sum over the 8 lanes. Otherwise flash_lean_kernel: its far-block loop, its masks.
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
__device__ __forceinline__ h2v as_h2(uint32_t u) { return __builtin_bit_cast(h2v, u); }
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const h2v x = {(_Float16) a, (_Float16) b};
    return __builtin_bit_cast(uint32_t, x);
}
__device__ __forceinline__ uint32_t bcast_u(uint32_t v, int owner) {
    return __float_as_uint(lean_broadcast8(__uint_as_float(v), owner));
}

// K is absolute Q8. Below the seal V is absolute symmetric V4, with its own refitted scales; above
// it V and its native Q8 scales live in a bounded ring, addressed by the absolute key modulo its size.
struct CompactRaw {
    uint32_t k[kLoads], v[kLoads];
    __half ks[kLoads], vs[kLoads];
    bool packed[kLoads];
};

__device__ void fetch_compact(CompactRaw& raw, KvView k, KvView v, int64_t kb, int64_t kmax,
                              int g, int tid, V4View v4) {
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const int at_i = tid + i * kFlashThreads, c = at_i / (kHead / 4), w = at_i % (kHead / 4);
        const int64_t key = kb + c;
        const bool packed = key < v4.sealed;
        raw.packed[i] = packed;
        if (key > kmax) {
            raw.k[i] = raw.v[i] = 0;
            raw.ks[i] = raw.vs[i] = __float2half(0.f);
            continue;
        }
        const size_t at = ((size_t) key * kKv + g) * kHead + 4 * w;
        raw.k[i] = *reinterpret_cast<const uint32_t*>(k.data + at);
        raw.ks[i] = reinterpret_cast<const __half*>(k.scales)[at / 32];
        if (packed) {
            raw.vs[i] = reinterpret_cast<const __half*>(v4.cold.scales)[at / 32];
            raw.v[i] = *reinterpret_cast<const uint16_t*>(v4.cold.data + at / 2);
        } else {
            const size_t off = ((size_t) (int(key) % v4.hot_ring) * kKv + g) * kHead + 4 * w;
            raw.vs[i] = reinterpret_cast<const __half*>(v.scales)[off / 32];
            raw.v[i] = *reinterpret_cast<const uint32_t*>(v.data + off);
        }
    }
}

template <bool Value>
__device__ float4 unpack_compact(const CompactRaw& raw, int i) {
    const uint32_t x = Value ? raw.v[i] : raw.k[i];
    const float s = __half2float(Value ? raw.vs[i] : raw.ks[i]);
    float4 q;
    if (Value && raw.packed[i])
        q = make_float4(int((x & 15) ^ 8) - 8, int(((x >> 4) & 15) ^ 8) - 8,
                        int(((x >> 8) & 15) ^ 8) - 8, int(((x >> 12) & 15) ^ 8) - 8);
    else
        q = make_float4(int8_t(x), int8_t(x >> 8), int8_t(x >> 16), int8_t(x >> 24));
    return make_float4(q.x * s, q.y * s, q.z * s, q.w * s);
}

template <bool Q8, bool Compact = false>
__global__ void __launch_bounds__(kFlashThreads)
flash_h2_kernel(const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ table,
                const float* __restrict__ sink, const float* __restrict__ gate, KvView k, KvView v, int64_t pos,
                int n, int window, int ring, float* __restrict__ out, V4View v4, int chunks, int chunk) {
    __shared__ uint2 ksh[kKeys][kHead / 4];              // 4 halves a float4 group
    alignas(16) __shared__ _Float16 vsh[kKeys / 2][kHead / 4][4][2];  // [pair c % 8][float4 group][dim][c / 8]
    __shared__ float ts[kSlice * kRelative];
    const int tid = threadIdx.x, part = tid % kSplit, tl = tid / kSplit, g = blockIdx.y;
    const int t0 = Compact ? 0 : blockIdx.x * kTokens, last = min(t0 + kTokens, n) - 1, t = min(t0 + tl, last);
    const bool live = t0 + tl < n;
    const int64_t p = pos + t, lo = Compact ? 0 : oldest(p + 1, window);
    const int64_t kmin = Compact ? int64_t(blockIdx.x) * chunk : oldest(pos + t0 + 1, window);
    const int64_t kmax = Compact ? min(kmin + chunk - 1, pos + last) : pos + last;
    const size_t qh0 = (size_t) t * kHeads + g * kGroup;

    uint32_t qh[kGroup][2 * kVec];
    float o[kGroup][4 * kVec], rr[kGroup][kRelPart], m[kGroup], l[kGroup], lc[kGroup];
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        const float4* qv = reinterpret_cast<const float4*>(q + (qh0 + r) * kHead);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            const float4 x = qv[part + kSplit * i];
            qh[r][2 * i] = pack_h2(x.x, x.y);
            qh[r][2 * i + 1] = pack_h2(x.z, x.w);
            o[r][4 * i] = o[r][4 * i + 1] = o[r][4 * i + 2] = o[r][4 * i + 3] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < kRelPart; ++i) rr[r][i] = rel[(qh0 + r) * kRelative + part * kRelPart + i] * kLog2e;
        // The running maximum over the keys only (flash_lean_kernel starts it at the sink): the half probabilities are
        // then relative to the largest key's, not to a sink far above every key, where they would be subnormal.
        m[r] = -INFINITY;
        l[r] = lc[r] = 0.f;
    }

    std::conditional_t<Compact, CompactRaw, Raw<Q8>> raw;
    const auto fetch_tile = [&](int64_t kb) __attribute__((always_inline)) {
        if constexpr (Compact) fetch_compact(raw, k, v, kb, kmax, g, tid, v4);
        else fetch<Q8>(raw, k, v, kb, kmin, kmax, ring, g, tid);
    };
    fetch_tile(kmin / kKeys * kKeys);
    const auto tile = [&]<bool Near, bool Mask>(int64_t kb) __attribute__((always_inline)) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int at_i = tid + i * kFlashThreads, c = at_i / (kHead / 4), w = at_i % (kHead / 4);
            float4 kx, vx;
            if constexpr (Compact) {
                kx = unpack_compact<false>(raw, i);
                vx = unpack_compact<true>(raw, i);
            } else {
                kx = unpack<Q8>(raw, i, false);
                vx = unpack<Q8>(raw, i, true);
            }
            ksh[c][w] = make_uint2(pack_h2(kx.x, kx.y), pack_h2(kx.z, kx.w));
            vsh[c % 8][w][0][c / 8] = (_Float16) vx.x;
            vsh[c % 8][w][1][c / 8] = (_Float16) vx.y;
            vsh[c % 8][w][2][c / 8] = (_Float16) vx.z;
            vsh[c % 8][w][3][c / 8] = (_Float16) vx.w;
        }
        if (kb + kKeys <= kmax) fetch_tile(kb + kKeys);
        if constexpr (Near) {
            const int64_t dlo = pos + t0 - (kb + kKeys - 1);
            for (int i = tid; i < kSlice * kRelative; i += kFlashThreads) {
                const int64_t d = dlo + i / kRelative;
                ts[i] = d >= 0 && d < kExtent ? table[d * kRelative + i % kRelative] : 0.f;
            }
        }
        __syncthreads();

        float s[kGroup][2];
#pragma unroll
        for (int r = 0; r < kGroup; ++r) s[r][0] = s[r][1] = 0.f;
#pragma unroll
        for (int c = 0; c < kKeys; ++c) {
            float dot[kGroup] = {};
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const uint2 kk = ksh[c][part + kSplit * i];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    dot[r] = __builtin_amdgcn_fdot2(as_h2(qh[r][2 * i]), as_h2(kk.x), dot[r], false);
                    dot[r] = __builtin_amdgcn_fdot2(as_h2(qh[r][2 * i + 1]), as_h2(kk.y), dot[r], false);
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
        uint32_t ph[kGroup];
        float sum[kGroup];
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            float top = fmaxf(s[r][0], s[r][1]);
            top = max_xor8<1>(top);
            top = max_xor8<2>(top);
            top = max_xor8<4>(top);
            // no key seen yet (all masked so far): base 0 gives exp2(-inf) = 0 for the scale and the probabilities
            const float mn = fmaxf(m[r], top), base = mn == -INFINITY ? 0.f : mn, shrink = exp2f(m[r] - base);
            m[r] = mn;
            l[r] *= shrink;
            lc[r] *= shrink;
#pragma unroll
            for (int i = 0; i < 4 * kVec; ++i) o[r][i] *= shrink;
            s[r][0] = exp2f(s[r][0] - base);
            s[r][1] = exp2f(s[r][1] - base);
            ph[r] = pack_h2(s[r][0], s[r][1]);
            float x = s[r][0] + s[r][1];
            x = add_xor8<1>(x);
            x = add_xor8<2>(x);
            x = add_xor8<4>(x);
            sum[r] = x;
        }
#pragma unroll
        for (int pq = 0; pq < kKeys / 2; ++pq) {
            uint32_t pp[kGroup];
#pragma unroll
            for (int r = 0; r < kGroup; ++r) pp[r] = bcast_u(ph[r], pq);
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const uint4 vv = *reinterpret_cast<const uint4*>(&vsh[pq][part + kSplit * i][0][0]);
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    o[r][4 * i] = __builtin_amdgcn_fdot2(as_h2(pp[r]), as_h2(vv.x), o[r][4 * i], false);
                    o[r][4 * i + 1] = __builtin_amdgcn_fdot2(as_h2(pp[r]), as_h2(vv.y), o[r][4 * i + 1], false);
                    o[r][4 * i + 2] = __builtin_amdgcn_fdot2(as_h2(pp[r]), as_h2(vv.z), o[r][4 * i + 2], false);
                    o[r][4 * i + 3] = __builtin_amdgcn_fdot2(as_h2(pp[r]), as_h2(vv.w), o[r][4 * i + 3], false);
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
    // far blocks first, as flash_lean_kernel takes them, with its guard for relative terms that are not finite
    bool finite = true;
#pragma unroll
    for (int r = 0; r < kGroup; ++r)
#pragma unroll
        for (int i = 0; i < kRelPart; ++i) finite = finite && isfinite(rr[r][i]);
    const bool far_ok = __syncthreads_and(finite) != 0 && (Compact || window == 0);
    const int64_t far_end = far_ok ? std::min<int64_t>(kmax + 1, pos + t0 - kExtent - kKeys + 2) : kb;
    for (; kb < far_end; kb += kKeys) tile.template operator()<false, false>(kb);
    for (; kb <= kmax; kb += kKeys) tile.template operator()<true, true>(kb);
    if (!live) return;
    if constexpr (Compact) {
        // The existing decode combiner adds the sink once, normalizes and gates across all splits.
        float* dst = out + ((size_t(t) * kKv + g) * chunks + blockIdx.x) * (kGroup * (kHead + 2));
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            float* head = dst + r * (kHead + 2);
            if (part == 0) {
                head[0] = m[r] * 0.6931471805599453f;   // from the base-2 recurrence to natural-score units
                head[1] = l[r];
            }
#pragma unroll
            for (int i = 0; i < kVec; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) head[2 + 4 * (part + kSplit * i) + j] = o[r][4 * i + j];
        }
    } else {
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            // the sink joins in the frame of the larger of it and the keys' maximum, so neither exponential overflows
            const float sk = sink2(sink[qh0 + r]), top = fmaxf(m[r], sk), keep = exp2f(m[r] - top);
            l[r] = l[r] * keep + exp2f(sk - top);
#pragma unroll
            for (int i = 0; i < 4 * kVec; ++i) o[r][i] *= keep;
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
}

}  // namespace

void attend_flash_h2(const Attend& a, cudaStream_t st) {
    if (a.n <= 0) return;
    if (!a.k.scales) throw std::runtime_error("Inkling half flash attention: needs the Q8_0 cache");
    const dim3 grid((a.n + kTokens - 1) / kTokens, kKv);
    flash_h2_kernel<true><<<grid, kFlashThreads, 0, st>>>(a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n,
                                                       a.window, a.ring, a.out, V4View{}, 0, 0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("Inkling half flash attention: ") + cudaGetErrorString(e));
}

void launch_compact_suffix_h2(const Attend& tile, float* scratch, cudaStream_t st, V4View v4,
                              int chunks, int chunk) {
    flash_h2_kernel<true, true><<<dim3(chunks, kKv), kFlashThreads, 0, st>>>(
        tile.q, tile.rel, tile.table, nullptr, nullptr, tile.k, tile.v, tile.pos, tile.n, 0, 0, scratch,
        v4, chunks, chunk);
}

}  // namespace strata::poly::inkling
