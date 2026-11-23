// src/poly/inkling_attend_lean.cu - STRATA_INKLING_FLASH_LEAN=1, see inkling_attend_lean.hpp.
// Its own file: next to flash_kernel in one translation unit, a second kernel changes how the compiler builds
// flash_kernel, and the default path keeps its code. The shared helpers are in inkling_flash_common.cuh.
#include "inkling_attend_lean.hpp"
#include "inkling_flash_common.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strata::poly::inkling {
namespace {

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
