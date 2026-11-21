// src/poly/inkling_attend.cu - see the header.
#include "inkling_attend.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::poly::inkling {
namespace {

constexpr int kThreads = 128, kValueTile = 128;
constexpr int kHeads = 24, kHead = 128, kKv = 8, kGroup = kHeads / kKv, kRelative = 16, kExtent = 1024;
constexpr int kQuery = kHeads * kHead;
int div_up(int a, int b) { return (a + b - 1) / b; }

template <bool Maximum> __device__ float wave_reduce(float x) {
    for (int d = 16; d > 0; d /= 2) {
        const float y = __shfl_down(x, d, 32);
        if constexpr (Maximum) x = fmaxf(x, y);
        else x += y;
    }
    return x;
}

template <bool Maximum = false> __device__ float block_reduce(float x, float* waves) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
    x = wave_reduce<Maximum>(x);
    if (lane == 0) waves[wave] = x;
    __syncthreads();
    if (wave == 0) {
        x = lane < kThreads / 32 ? waves[lane] : Maximum ? -INFINITY : 0.f;
        x = wave_reduce<Maximum>(x);
        if (lane == 0) waves[0] = x;
    }
    __syncthreads();
    const float result = waves[0];
    __syncthreads();
    return result;
}

template <bool Q8> __device__ float kv_at(KvView cache, size_t i) {
    if constexpr (Q8)
        return (float) reinterpret_cast<const int8_t*>(cache.data)[i] *
               __half2float(reinterpret_cast<const __half*>(cache.scales)[i / 32]);
    return __half2float(reinterpret_cast<const __half*>(cache.data)[i]);
}

__device__ int64_t oldest(int64_t end, int window) { return window > 0 && end > window ? end - window : 0; }

// A wave computes one score, with 4 values from a 128-wide head per lane. The FP32 query is not quantized.
template <bool Q8> __global__ void scores_kernel(const float* q, const float* rel, const float* table, KvView k,
                                                 int64_t pos, int span, int window, int ring, float* scores) {
    const int lane = threadIdx.x & 31, col = blockIdx.x * (kThreads / 32) + threadIdx.x / 32;
    const int h = blockIdx.y, row = blockIdx.z;
    const int64_t end = pos + row + 1, start = oldest(end, window), key = start + col;
    if (key >= end) return;
    const size_t qh = (size_t) row * 24 + h;
    const size_t key_at = ((size_t) (ring ? key % ring : key) * 8 + h / 3) * 128;
    float dot = 0.f;
    for (int i = lane; i < 128; i += 32) dot += q[qh * 128 + i] * kv_at<Q8>(k, key_at + i);
    const int64_t delta = end - 1 - key;
    float score = dot * 0.08838834764831845f;
    if (lane < 16 && delta < 1024) score += rel[qh * 16 + lane] * table[delta * 16 + lane];
    score = wave_reduce<false>(score);
    if (lane == 0) scores[qh * span + col] = score;
}

// The projected sink joins the softmax once per head and query and has no value vector.
__global__ void probs_kernel(float* scores, const float* sink, int64_t pos, int span, int window) {
    __shared__ float waves[kThreads / 32];
    const int64_t end = pos + blockIdx.y + 1;
    const int count = (int) (end - oldest(end, window));
    const size_t h = (size_t) blockIdx.y * 24 + blockIdx.x;
    float* dst = scores + h * span;
    float top = sink[h];
    for (int i = threadIdx.x; i < count; i += kThreads) top = fmaxf(top, dst[i]);
    top = block_reduce<true>(top, waves);
    float s = threadIdx.x == 0 ? expf(sink[h] - top) : 0.f;
    for (int i = threadIdx.x; i < count; i += kThreads) s += (dst[i] = expf(dst[i] - top));
    s = block_reduce(s, waves);
    for (int i = threadIdx.x; i < count; i += kThreads) dst[i] /= s;
}

template <bool Q8> __global__ void values_kernel(const float* probs, KvView v, int64_t pos, int span, int window,
                                                 int ring, int parts, float* out) {
    const int h = blockIdx.x, row = blockIdx.y, tile = blockIdx.z, d = threadIdx.x;
    const int64_t end = pos + row + 1, start = oldest(end, window);
    const size_t qh = (size_t) row * 24 + h;
    const float* p = probs + qh * span;
    const int count = (int) (end - start), first = tile * kValueTile;
    const int last = min(first + kValueTile, count);
    float sum = 0.f;
    for (int i = first; i < last; ++i) {
        const int64_t key = start + i;
        const size_t at = ((size_t) (ring ? key % ring : key) * 8 + h / 3) * 128 + d;
        sum += p[i] * kv_at<Q8>(v, at);
    }
    out[(qh * parts + tile) * 128 + d] = sum;
}

__global__ void gate_kernel(const float* parts, int count, const float* gate, float* out) {
    const size_t h = (size_t) blockIdx.y * 24 + blockIdx.x;
    const int d = threadIdx.x;
    float sum = 0.f;
    for (int i = 0; i < count; ++i) sum += parts[(h * count + i) * 128 + d];
    out[h * 128 + d] = sum / (1.f + expf(-gate[h * 128 + d]));
}

// attend_flash(): a block takes kTokens tokens of one K/V group. Eight threads own a token: its 3 query heads (which
// share the keys, the mask and the distances) over an eighth of the head's values each, float4 w = lane + 8 i, so
// that 8 threads read 128 consecutive bytes of a key. Keys come kKeys at a time, loaded a block ahead into registers
// and dequantized once into shared memory as the plain kernels dequantize them, with the slice of the relative table
// their distances need; the 8 parts of a score meet by DPP8 moves. A block of keys starts at a multiple of kKeys, so
// a token's keys fall into the same blocks whatever tokens share its launch; a block it attends none of leaves its
// sums as they were. On RX 6700 XT, 16 tokens and 16 keys beat 32 and 8 tokens and 32 keys (which spilled).
constexpr int kTokens = 16, kKeys = 16, kSplit = 8, kFlashThreads = kSplit * kTokens;
constexpr int kVec = kHead / 4 / kSplit;               // float4s of a head a thread owns
constexpr int kRelPart = kRelative / kSplit;           // relative terms a thread owns
constexpr int kSlice = kTokens + kKeys - 1;
constexpr float kLog2e = 1.4426950408889634f;   // the softmax runs in powers of 2: scores and the sink times log2(e)
// the sink in those units, finite: a running maximum of +-inf would make exp2f(inf - inf)
__device__ float sink2(float x) { return fminf(fmaxf(x * kLog2e, -FLT_MAX), FLT_MAX); }

// A thread's share of a block of keys as the cache holds it, loaded a block ahead (flash_kernel): kLoads float4s of
// K and of V, as codes and their scales (Q8_0) or as halves.
constexpr int kLoads = kKeys * (kHead / 4) / kFlashThreads;
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
        const bool in = key >= kmin && key <= kmax;   // no stale or unwritten rows: their weight is 0
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

// lane ^ X within each group of 8 lanes, as one VALU move (DPP8, gfx10 and later) rather than through LDS
template <int X> __device__ float xor8(float v) {
    constexpr int sel = (0 ^ X) | (1 ^ X) << 3 | (2 ^ X) << 6 | (3 ^ X) << 9 | (4 ^ X) << 12 | (5 ^ X) << 15 |
                        (6 ^ X) << 18 | (7 ^ X) << 21;
    return __int_as_float(__builtin_amdgcn_mov_dpp8(__float_as_int(v), sel));
}

template <bool Q8>
__global__ void __launch_bounds__(kFlashThreads)
flash_kernel(const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ table,
             const float* __restrict__ sink, const float* __restrict__ gate, KvView k, KvView v, int64_t pos, int n,
             int window, int ring, float* __restrict__ out) {
    __shared__ float4 ks[kKeys][kHead / 4];
    __shared__ float4 vs[kKeys][kHead / 4];
    __shared__ float ts[kSlice * kRelative];
    const int tid = threadIdx.x, part = tid % kSplit, tl = tid / kSplit, g = blockIdx.y;
    const int t0 = blockIdx.x * kTokens, last = min(t0 + kTokens, n) - 1, t = min(t0 + tl, last);
    const bool live = t0 + tl < n;
    const int64_t p = pos + t, lo = oldest(p + 1, window);
    const int64_t kmin = oldest(pos + t0 + 1, window), kmax = pos + last;
    const size_t qh0 = (size_t) t * kHeads + g * kGroup;   // the token's first head of the group

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
        // the sink's term, exp(sink - m), joins l at the end; m stays finite, so masked keys give exp(-inf) = 0
        m[r] = sink2(sink[qh0 + r]);
        l[r] = lc[r] = 0.f;
    }

    Raw<Q8> raw;
    fetch<Q8>(raw, k, v, kmin / kKeys * kKeys, kmin, kmax, ring, g, tid);
    for (int64_t kb = kmin / kKeys * kKeys; kb <= kmax; kb += kKeys) {
        __syncthreads();   // the block before has been read
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int at_i = tid + i * kFlashThreads, c = at_i / (kHead / 4), w = at_i % (kHead / 4);
            ks[c][w] = unpack<Q8>(raw, i, false);
            vs[c][w] = unpack<Q8>(raw, i, true);
        }
        if (kb + kKeys <= kmax) fetch<Q8>(raw, k, v, kb + kKeys, kmin, kmax, ring, g, tid);   // under this block's work
        // ts[j]: the table row of distance pos + t0 - (kb + kKeys - 1) + j, zero past the bias's extent
        const int64_t dlo = pos + t0 - (kb + kKeys - 1);
        for (int i = tid; i < kSlice * kRelative; i += kFlashThreads) {
            const int64_t d = dlo + i / kRelative;
            ts[i] = d >= 0 && d < kExtent ? table[d * kRelative + i % kRelative] : 0.f;
        }
        __syncthreads();

        float s[kGroup][kKeys];
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
#pragma unroll
                for (int i = 0; i < kRelPart; ++i) score = fmaf(rr[r][i], tr[i], score);
                s[r][c] = score;
            }
        }
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            float top = -INFINITY;
#pragma unroll
            for (int c = 0; c < kKeys; ++c) {
                float x = s[r][c];
                x += xor8<1>(x);
                x += xor8<2>(x);
                x += xor8<4>(x);
                const int64_t key = kb + c;
                s[r][c] = key < lo || key > p ? -INFINITY : x;
                top = fmaxf(top, s[r][c]);
            }
            // a block the row attends none of scales by exp(0) and adds exp(-inf): it changes nothing
            const float mn = fmaxf(m[r], top), shrink = exp2f(m[r] - mn);
            m[r] = mn;
            l[r] *= shrink;
            lc[r] *= shrink;
#pragma unroll
            for (int i = 0; i < 4 * kVec; ++i) o[r][i] *= shrink;
            // a block's share can be below the running sum's rounding (one key far above the rest of a long row):
            // the block's own sum first, then a compensated sum over the blocks
            float sum = 0.f;
#pragma unroll
            for (int c = 0; c < kKeys; ++c) {
                s[r][c] = exp2f(s[r][c] - mn);
                sum += s[r][c];
            }
            const float y = sum - lc[r], next = l[r] + y;
            lc[r] = (next - l[r]) - y;
            l[r] = next;
        }
#pragma unroll
        for (int c = 0; c < kKeys; ++c) {
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const float4 x = vs[c][part + kSplit * i];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    o[r][4 * i] = fmaf(s[r][c], x.x, o[r][4 * i]);
                    o[r][4 * i + 1] = fmaf(s[r][c], x.y, o[r][4 * i + 1]);
                    o[r][4 * i + 2] = fmaf(s[r][c], x.z, o[r][4 * i + 2]);
                    o[r][4 * i + 3] = fmaf(s[r][c], x.w, o[r][4 * i + 3]);
                }
            }
        }
    }
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

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

size_t plain_score_floats(int tiles) { return (size_t) tiles * kHeads * kValueTile; }
size_t plain_partial_floats(int tiles) { return (size_t) tiles * kQuery; }

void attend_plain(const Attend& a, float* scores, float* partial, int tiles, cudaStream_t st) {
    const int window = a.window, ring = a.ring;
    const int max_span = (int) (window ? std::min<int64_t>(window, a.pos + a.n) : a.pos + a.n);
    const int attend_rows = tiles / div_up(max_span, kValueTile);
    for (int at = 0; at < a.n; at += attend_rows) {
        const int m = std::min(a.n - at, attend_rows);
        const int64_t pos = a.pos + at;
        const int span = (int) (window ? std::min<int64_t>(window, pos + m) : pos + m);
        const int parts = div_up(span, kValueTile);
        const dim3 sg(div_up(span, kThreads / 32), kHeads, m), vg(kHeads, m, parts), hp(kHeads, m);
        const size_t off = (size_t) at * kQuery, rh = (size_t) at * kHeads;
        const auto launch = [&]<bool Q8>() {
            scores_kernel<Q8><<<sg, kThreads, 0, st>>>(a.q + off, a.rel + rh * kRelative, a.table, a.k, pos, span,
                                                       window, ring, scores);
            probs_kernel<<<hp, kThreads, 0, st>>>(scores, a.sink + rh, pos, span, window);
            values_kernel<Q8><<<vg, kHead, 0, st>>>(scores, a.v, pos, span, window, ring, parts, partial);
        };
        if (a.k.scales) launch.template operator()<true>();
        else launch.template operator()<false>();
        gate_kernel<<<hp, kHead, 0, st>>>(partial, parts, a.gate + off, a.out + off);
    }
}

void attend_flash(const Attend& a, cudaStream_t st) {
    if (a.n <= 0) return;
    const dim3 grid(div_up(a.n, kTokens), kKv);
    if (a.k.scales)
        flash_kernel<true><<<grid, kFlashThreads, 0, st>>>(a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n,
                                                           a.window, a.ring, a.out);
    else
        flash_kernel<false><<<grid, kFlashThreads, 0, st>>>(a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos,
                                                            a.n, a.window, a.ring, a.out);
    check("Inkling flash attention");
}

}  // namespace strata::poly::inkling
