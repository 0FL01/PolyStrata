// src/poly/inkling_attend.cu - see the header.
#include "inkling_attend.hpp"

#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/gfx_arch.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// Broadcast a key's FP32 probability from its owning lane within each group of 8.
template <int Owner> __device__ float owner8(float v) {
    constexpr int sel = Owner | Owner << 3 | Owner << 6 | Owner << 9 | Owner << 12 | Owner << 15 |
                        Owner << 18 | Owner << 21;
    return __int_as_float(__builtin_amdgcn_mov_dpp8(__float_as_int(v), sel));
}

// The fully unrolled caller makes owner constant and removes this switch before ISA generation.
__device__ __forceinline__ float broadcast8(float v, int owner) {
    switch (owner) {
        case 0: return owner8<0>(v);
        case 1: return owner8<1>(v);
        case 2: return owner8<2>(v);
        case 3: return owner8<3>(v);
        case 4: return owner8<4>(v);
        case 5: return owner8<5>(v);
        case 6: return owner8<6>(v);
        default: return owner8<7>(v);
    }
}

template <bool Q8, bool LaneProb = false>
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

        float s[kGroup][LaneProb ? kKeys / kSplit : kKeys];
        if constexpr (LaneProb) {
#pragma unroll
            for (int r = 0; r < kGroup; ++r) s[r][0] = s[r][1] = 0.f;
        }
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
                if constexpr (LaneProb) {
                    score += xor8<1>(score);
                    score += xor8<2>(score);
                    score += xor8<4>(score);
                    const int64_t key = kb + c;
                    // This is a cndmask, not divergent execution around the DPP8 reductions.
                    if (part == c % kSplit) s[r][c / kSplit] = key < lo || key > p ? -INFINITY : score;
                } else {
                    s[r][c] = score;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            if constexpr (LaneProb) {
                float top = fmaxf(s[r][0], s[r][1]);
                top = fmaxf(top, xor8<1>(top));
                top = fmaxf(top, xor8<2>(top));
                top = fmaxf(top, xor8<4>(top));
                // Every lane gets the same maximum.
                const float mn = fmaxf(m[r], top), shrink = exp2f(m[r] - mn);
                m[r] = mn;
                l[r] *= shrink;
                lc[r] *= shrink;
#pragma unroll
                for (int i = 0; i < 4 * kVec; ++i) o[r][i] *= shrink;
                s[r][0] = exp2f(s[r][0] - mn);
                s[r][1] = exp2f(s[r][1] - mn);
            } else {
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
        }
        if constexpr (LaneProb) {
            float sum[kGroup] = {};
#pragma unroll
            for (int c = 0; c < kKeys; ++c) {
                float prob[kGroup];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    prob[r] = broadcast8(s[r][c / kSplit], c % kSplit);
                    // Match the original sequential 16-key denominator sum; no new tree-sum rounding.
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
                // The compensated sum across key blocks is unchanged; l/lc are not used by PV.
                const float y = sum[r] - lc[r], next = l[r] + y;
                lc[r] = (next - l[r]) - y;
                l[r] = next;
            }
        } else {
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

// attend_decode(): a token's keys in chunks of whole tiles of kThreads keys, one block a chunk and K/V group. A thread
// takes a key of a tile and its 3 query heads' scores; then four groups of 32 lanes add the tile's values, 4 dims a lane
// and every fourth key, and fold their sums together at the end. A chunk leaves its maximum, its sum of exponentials and
// its unnormalized values per head; decode_combine_kernel() adds the chunks and the sink, divides and gates.
constexpr int kDecodeBlocks = 4096;   // chunks times K/V groups times rows at most: the scratch's size
constexpr int kChunkFloats = kGroup * (kHead + 2);
constexpr int kAhead = 8;   // decode_kernel(): a wave's values loaded at a time

template <bool Q8> struct KeyRow;
template <> struct KeyRow<true> {
    static __device__ void dot(KvView k, size_t at, const float (*q)[kHead], float* d) {
        const int4* codes = reinterpret_cast<const int4*>(reinterpret_cast<const int8_t*>(k.data) + at);
        const __half* sc = reinterpret_cast<const __half*>(k.scales) + at / 32;
#pragma unroll
        for (int b = 0; b < kHead / 32; ++b) {
            float part[kGroup] = {};
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int4 w = codes[2 * b + c];
                const int words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y < 4; ++y) {
                        const float val = (float) (int8_t) (words[x] >> (8 * y));
                        const int i = 32 * b + 16 * c + 4 * x + y;
#pragma unroll
                        for (int r = 0; r < kGroup; ++r) part[r] += q[r][i] * val;
                    }
            }
            const float scale = __half2float(sc[b]);
#pragma unroll
            for (int r = 0; r < kGroup; ++r) d[r] += part[r] * scale;
        }
    }
    static __device__ void value(KvView v, size_t at, float* x) {
        const char4 c = *reinterpret_cast<const char4*>(reinterpret_cast<const int8_t*>(v.data) + at);
        const float scale = __half2float(reinterpret_cast<const __half*>(v.scales)[at / 32]);
        x[0] = c.x * scale;
        x[1] = c.y * scale;
        x[2] = c.z * scale;
        x[3] = c.w * scale;
    }
};
template <> struct KeyRow<false> {
    static __device__ void dot(KvView k, size_t at, const float (*q)[kHead], float* d) {
        const uint4* halves = reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(k.data) + at);
#pragma unroll
        for (int b = 0; b < kHead / 8; ++b) {
            const uint4 w = halves[b];
            const unsigned words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const __half2 h2 = *reinterpret_cast<const __half2*>(&words[x]);
                const float2 f = __half22float2(h2);
                const int i = 8 * b + 2 * x;
#pragma unroll
                for (int r = 0; r < kGroup; ++r) d[r] += q[r][i] * f.x + q[r][i + 1] * f.y;
            }
        }
    }
    static __device__ void value(KvView v, size_t at, float* x) {
        const uint2 w = *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(v.data) + at);
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&w.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&w.y));
        x[0] = a.x;
        x[1] = a.y;
        x[2] = b.x;
        x[3] = b.y;
    }
};

// the kGroup values of a block's threads reduced together, every thread gets them
template <bool Maximum> __device__ void group_reduce(float* x, float (*waves)[kThreads / 32]) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        x[r] = wave_reduce<Maximum>(x[r]);
        if (lane == 0) waves[r][wave] = x[r];
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        float y = waves[r][0];
#pragma unroll
        for (int w = 1; w < kThreads / 32; ++w) y = Maximum ? fmaxf(y, waves[r][w]) : y + waves[r][w];
        x[r] = y;
    }
    __syncthreads();
}

template <bool Q8> __global__ void decode_kernel(const float* q, const float* rel, const float* table, KvView k,
                                                 KvView v, int64_t pos, int window, int ring, int chunk, float* part) {
    const int c = blockIdx.x, g = blockIdx.y, row = blockIdx.z, t = threadIdx.x;
    const int64_t end = pos + row + 1, start = oldest(end, window);
    const int count = (int) (end - start), first = c * chunk, last = min(first + chunk, count);
    __shared__ float qs[kGroup][kHead], rs[kGroup][kRelative], ps[kThreads][kGroup], red[4][kGroup][kHead];
    __shared__ float waves[kGroup][kThreads / 32];
    const size_t qh0 = (size_t) row * kHeads + g * kGroup;
    for (int i = t; i < kGroup * kHead; i += kThreads) qs[i / kHead][i % kHead] = q[qh0 * kHead + i];
    if (t < kGroup * kRelative) rs[t / kRelative][t % kRelative] = rel[qh0 * kRelative + t];
    __syncthreads();
    float m[kGroup], l[kGroup], acc[kGroup][4] = {};
#pragma unroll
    for (int r = 0; r < kGroup; ++r) m[r] = -INFINITY, l[r] = 0.f;
    const int lane = t & 31, slot = t / 32;
    for (int tile = first; tile < last; tile += kThreads) {
        const int i = tile + t;
        float s[kGroup];
        if (i < last) {
            const int64_t key = start + i, delta = end - 1 - key;
            const size_t at = ((size_t) (ring ? key % ring : key) * kKv + g) * kHead;
            float d[kGroup] = {};
            KeyRow<Q8>::dot(k, at, qs, d);
#pragma unroll
            for (int r = 0; r < kGroup; ++r) s[r] = d[r] * 0.08838834764831845f;
            if (delta < kExtent) {
                const float* tb = table + delta * kRelative;
#pragma unroll
                for (int j = 0; j < kRelative; ++j) {
                    const float b = tb[j];
#pragma unroll
                    for (int r = 0; r < kGroup; ++r) s[r] += rs[r][j] * b;
                }
            }
        } else {
#pragma unroll
            for (int r = 0; r < kGroup; ++r) s[r] = -INFINITY;
        }
        float top[kGroup];
#pragma unroll
        for (int r = 0; r < kGroup; ++r) top[r] = s[r];
        group_reduce<true>(top, waves);
        float e[kGroup];
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            const float next = fmaxf(m[r], top[r]);
            // every score so far -inf (overflowed terms): the exponentials are 0, as attend_plain() gives them
            const float base = next == -INFINITY ? 0.f : next;
            const float keep = expf(m[r] - base);
            m[r] = next;
            e[r] = i < last ? expf(s[r] - base) : 0.f;
            ps[t][r] = e[r];
            l[r] *= keep;
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[r][j] *= keep;
        }
        group_reduce<false>(e, waves);   // also makes ps visible
#pragma unroll
        for (int r = 0; r < kGroup; ++r) l[r] += e[r];
        const int keys = min(kThreads, last - tile);
        int j = slot;
        // kAhead keys' values loaded before any is added: one wait for memory where each key waited for its own.
        // The sums go in the same order as below.
        for (; j + 4 * (kAhead - 1) < keys; j += 4 * kAhead) {
            float x[kAhead][4];
#pragma unroll
            for (int u = 0; u < kAhead; ++u) {
                const int64_t key = start + tile + j + 4 * u;
                KeyRow<Q8>::value(v, ((size_t) (ring ? key % ring : key) * kKv + g) * kHead + 4 * lane, x[u]);
            }
#pragma unroll
            for (int u = 0; u < kAhead; ++u)
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    const float p = ps[j + 4 * u][r];
#pragma unroll
                    for (int y = 0; y < 4; ++y) acc[r][y] += p * x[u][y];
                }
        }
        for (; j < keys; j += 4) {
            const int64_t key = start + tile + j;
            float x[4];
            KeyRow<Q8>::value(v, ((size_t) (ring ? key % ring : key) * kKv + g) * kHead + 4 * lane, x);
#pragma unroll
            for (int r = 0; r < kGroup; ++r) {
                const float p = ps[j][r];
#pragma unroll
                for (int y = 0; y < 4; ++y) acc[r][y] += p * x[y];
            }
        }
        __syncthreads();   // ps is written again by the next tile
    }
#pragma unroll
    for (int r = 0; r < kGroup; ++r)
#pragma unroll
        for (int y = 0; y < 4; ++y) red[slot][r][4 * lane + y] = acc[r][y];
    __syncthreads();
    float* dst = part + (((size_t) row * kKv + g) * gridDim.x + c) * kChunkFloats;
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        dst[r * (kHead + 2) + 2 + t] = red[0][r][t] + red[1][r][t] + red[2][r][t] + red[3][r][t];
        if (t == 0) {
            dst[r * (kHead + 2)] = m[r];
            dst[r * (kHead + 2) + 1] = l[r];
        }
    }
}

// attend_decode() with Q8_0 K/V: a block a chunk of keys and every K/V group, a wave a group, so the block reads each
// key's whole row of the cache (8 groups of 128 bytes) at once. In a wave eight lanes take a key, 16 dimensions a
// lane (one 16-byte load of K and one of V, inside one Q8_0 block), so a wave's load is four keys' 128-byte rows
// whole; decode_kernel read a key a thread, 1,024 bytes apart across the threads. The eight lanes add their parts of
// the 3 heads' scores and relative terms by shuffles; each group of eight lanes keeps its own running
// maximum, sum and values, and the four groups of a wave fold together at the end. Same output as decode_kernel
// (decode_combine_kernel reads it), other sums.
constexpr int kAttWaves = kKv, kAttThreads = 32 * kAttWaves, kAttUnroll = 2;
__global__ void __launch_bounds__(kAttThreads, 2)
decode_q8_kernel(const float* q, const float* rel, const float* table, KvView k, KvView v, int64_t pos, int window,
                 int ring, int chunk, float* part) {
    const int c = blockIdx.x, row = blockIdx.y, g = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int sub = lane & 7, quad = lane >> 3;   // the 16 dimensions of the lane, and which of four keys
    const int64_t end = pos + row + 1, start = oldest(end, window);
    const int count = (int) (end - start), first = c * chunk, last = min(first + chunk, count);
    const size_t qh0 = (size_t) row * kHeads + g * kGroup;
    float qr[kGroup][16], rr[kGroup][2];
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        const float4* qv = reinterpret_cast<const float4*>(q + (qh0 + r) * kHead + 16 * sub);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 x = qv[j];
            qr[r][4 * j] = x.x;
            qr[r][4 * j + 1] = x.y;
            qr[r][4 * j + 2] = x.z;
            qr[r][4 * j + 3] = x.w;
        }
        rr[r][0] = rel[(qh0 + r) * kRelative + 2 * sub];
        rr[r][1] = rel[(qh0 + r) * kRelative + 2 * sub + 1];
    }
    float m[kGroup], l[kGroup], acc[kGroup][16];
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) acc[r][j] = 0.f;
    }
    const int8_t* kd = reinterpret_cast<const int8_t*>(k.data);
    const int8_t* vd = reinterpret_cast<const int8_t*>(v.data);
    const __half* ksc = reinterpret_cast<const __half*>(k.scales);
    const __half* vsc = reinterpret_cast<const __half*>(v.scales);
    for (int i0 = first; i0 < last; i0 += 4 * kAttUnroll) {
        int4 kw[kAttUnroll], vw[kAttUnroll];
        float ks[kAttUnroll], vs[kAttUnroll];
        // every key's loads before any is used
#pragma unroll
        for (int u = 0; u < kAttUnroll; ++u) {
            const int i = i0 + 4 * u + quad;
            if (i < last) {
                const int64_t key = start + i;
                const size_t at = ((size_t) (ring ? key % ring : key) * kKv + g) * kHead + 16 * sub;
                kw[u] = *reinterpret_cast<const int4*>(kd + at);
                vw[u] = *reinterpret_cast<const int4*>(vd + at);
                ks[u] = __half2float(ksc[at / 32]);
                vs[u] = __half2float(vsc[at / 32]);
            }
        }
#pragma unroll
        for (int u = 0; u < kAttUnroll; ++u) {
            const int i = i0 + 4 * u + quad;
            if (i >= last) continue;   // a whole group of eight lanes: the shuffles below stay inside it
            const int64_t key = start + i, delta = end - 1 - key;
            const int kwords[4] = {kw[u].x, kw[u].y, kw[u].z, kw[u].w};
            const int vwords[4] = {vw[u].x, vw[u].y, vw[u].z, vw[u].w};
            float sc[kGroup];
#pragma unroll
            for (int r = 0; r < kGroup; ++r) {
                float d = 0.f;
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y < 4; ++y) d += qr[r][4 * x + y] * (float) (int8_t) (kwords[x] >> (8 * y));
                sc[r] = d * ks[u] * 0.08838834764831845f;
            }
            if (delta < kExtent) {
                const float* tb = table + delta * kRelative + 2 * sub;
                const float b0 = tb[0], b1 = tb[1];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) sc[r] += rr[r][0] * b0 + rr[r][1] * b1;
            }
#pragma unroll
            for (int r = 0; r < kGroup; ++r)
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) sc[r] += __shfl_xor(sc[r], o, 32);
            float val[16];
#pragma unroll
            for (int x = 0; x < 4; ++x)
#pragma unroll
                for (int y = 0; y < 4; ++y) val[4 * x + y] = (float) (int8_t) (vwords[x] >> (8 * y)) * vs[u];
#pragma unroll
            for (int r = 0; r < kGroup; ++r) {
                const float next = fmaxf(m[r], sc[r]);
                // every score so far -inf (overflowed terms): the exponentials are 0, as attend_plain() gives them
                const float base = next == -INFINITY ? 0.f : next;
                const float keep = expf(m[r] - base), p = expf(sc[r] - base);
                m[r] = next;
                l[r] = l[r] * keep + p;
#pragma unroll
                for (int j = 0; j < 16; ++j) acc[r][j] = acc[r][j] * keep + p * val[j];
            }
        }
    }
    // the wave's four groups of keys into one (lanes 8 and 16 apart hold the same dimensions)
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        float top = fmaxf(m[r], __shfl_xor(m[r], 8, 32));
        top = fmaxf(top, __shfl_xor(top, 16, 32));
        const float base = top == -INFINITY ? 0.f : top;
        const float f = expf(m[r] - base);
        float sum = l[r] * f;
        sum += __shfl_xor(sum, 8, 32);
        sum += __shfl_xor(sum, 16, 32);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float a = acc[r][j] * f;
            a += __shfl_xor(a, 8, 32);
            a += __shfl_xor(a, 16, 32);
            acc[r][j] = a;
        }
        m[r] = top;
        l[r] = sum;
    }
    if (quad != 0) return;
    float* dst = part + (((size_t) row * kKv + g) * gridDim.x + c) * kChunkFloats;
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[r * (kHead + 2) + 2 + 16 * sub + j] = acc[r][j];
        if (sub == 0) {
            dst[r * (kHead + 2)] = m[r];
            dst[r * (kHead + 2) + 1] = l[r];
        }
    }
}

// A chunk with no keys left -inf, 0 and zeros. The sink as probs_kernel takes it, -inf (no sink) included.
struct Q81 {
    __half2 ds;
    int8_t qs[32];
};

__global__ void decode_combine_kernel(const float* part, int chunks, const float* sink, const float* gate,
                                      float* out, Q81* q8) {
    const int h = blockIdx.x, row = blockIdx.y, d = threadIdx.x, g = h / kGroup, r = h % kGroup;
    const size_t qh = (size_t) row * kHeads + h;
    const float* base = part + ((size_t) row * kKv + g) * chunks * kChunkFloats + r * (kHead + 2);
    const float s = sink[qh];
    float top = s;
    // unrolled for the loads, the max and the sums in chunk order as one chunk at a time
#pragma unroll 8
    for (int c = 0; c < chunks; ++c) top = fmaxf(top, base[(size_t) c * kChunkFloats]);
    const float at = top == -INFINITY ? 0.f : top;
    float sum = expf(s - at), o = 0.f;
#pragma unroll 8
    for (int c = 0; c < chunks; ++c) {
        const float* p = base + (size_t) c * kChunkFloats;
        const float w = expf(p[0] - at);
        sum += p[1] * w;
        o += p[2 + d] * w;
    }
    const float y = o / sum / (1.f + expf(-gate[qh * kHead + d]));
    out[qh * kHead + d] = y;
    if (q8) {   // a warp holds a block's 32 values in lane order, as native_quantize_q8_1_kernel's do
        float amax = fabsf(y), bsum = y;
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, o2, 32));
            bsum += __shfl_xor(bsum, o2, 32);
        }
        const float dq = kernels::q8_1_finite(amax / 127.0f);
        Q81& b = q8[(qh * kHead + d) / 32];
        b.qs[d % 32] = kernels::q8_1_quant(y, dq, amax);
        if (d % 32 == 0) b.ds = kernels::q8_1_ds(dq, bsum);
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

size_t decode_floats() { return (size_t) kDecodeBlocks * kChunkFloats; }

void attend_decode(const Attend& a, float* scratch, cudaStream_t st, void* q8) {
    if (a.n <= 0) return;
    if (kKv * a.n > kDecodeBlocks) throw std::runtime_error("Inkling decode attention: too many rows");
    const int64_t end = a.pos + a.n;
    const int span = (int) (a.window ? std::min<int64_t>(a.window, end) : end);
    const int most = std::max(1, kDecodeBlocks / (kKv * a.n));
    int chunks = 0;
    if (a.k.scales) {
        // up to kAttChunks chunks of at least kAttMin keys (whole steps of a wave's keys): enough blocks to keep the
        // card's memory busy at long spans, and few steps a block at short ones
        constexpr int kAttChunks = 128, kAttMin = 32, kStep = 4 * kAttUnroll;
        const int chunk = std::max(kAttMin, div_up(div_up(span, std::min(most, kAttChunks)), kStep) * kStep);
        chunks = div_up(span, chunk);
        decode_q8_kernel<<<dim3(chunks, a.n), kAttThreads, 0, st>>>(a.q, a.rel, a.table, a.k, a.v, a.pos, a.window,
                                                                   a.ring, chunk, scratch);
    } else {
        const int chunk = div_up(div_up(span, most), kThreads) * kThreads;
        const dim3 grid(div_up(span, chunk), kKv, a.n);
        chunks = (int) grid.x;
        decode_kernel<false><<<grid, kThreads, 0, st>>>(a.q, a.rel, a.table, a.k, a.v, a.pos, a.window, a.ring, chunk,
                                                        scratch);
    }
    decode_combine_kernel<<<dim3(kHeads, a.n), kHead, 0, st>>>(scratch, chunks, a.sink, a.gate, a.out,
                                                               static_cast<Q81*>(q8));
    check("Inkling decode attention");
}

void attend_flash(const Attend& a, cudaStream_t st) {
    if (a.n <= 0) return;
    const dim3 grid(div_up(a.n, kTokens), kKv);
    static const bool lane_prob = [] {
        const char* value = std::getenv("STRATA_INKLING_LANE_PROB");
        if (!value || value[0] != '1' || value[1] != '\0') return false;
        int device = 0;
        cudaDeviceProp prop{};
        cudaError_t error = cudaGetDevice(&device);
        if (error == cudaSuccess) error = cudaGetDeviceProperties(&prop, device);
        if (error != cudaSuccess)
            throw std::runtime_error(std::string("Inkling lane probabilities: ") + cudaGetErrorString(error));
        if (!kernels::gfx_arch_is(prop.gcnArchName, "gfx1031")) return false;
        std::fputs("strata-poly: STRATA_INKLING_LANE_PROB=1: lane-owned FP32 flash probabilities on gfx1031\n", stderr);
        return true;
    }();
    if (lane_prob) {
        if (a.k.scales)
            flash_kernel<true, true><<<grid, kFlashThreads, 0, st>>>(a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n,
                                                               a.window, a.ring, a.out);
        else
            flash_kernel<false, true><<<grid, kFlashThreads, 0, st>>>(a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos,
                                                                a.n, a.window, a.ring, a.out);
    } else {
        if (a.k.scales)
            flash_kernel<true><<<grid, kFlashThreads, 0, st>>>(a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n,
                                                               a.window, a.ring, a.out);
        else
            flash_kernel<false><<<grid, kFlashThreads, 0, st>>>(a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos,
                                                                a.n, a.window, a.ring, a.out);
    }
    check("Inkling flash attention");
}

}  // namespace strata::poly::inkling
