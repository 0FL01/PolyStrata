// Inkling-2-Mini's loop visits, short convolutions and relative attention. The GGUF has 24 sets of weights;
// the three core traversals reuse those weights and expert slots, not the sequence state.
#include "inkling2.hpp"

#include "inkling_attend.hpp"
#include "inkling_dspark.hpp"
#include "runtime.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int kThreads = 128;
constexpr int kRowWidth = 2304;   // Inkling2::kWidth, for the kernels that keep a whole row in registers
constexpr int kKvWidth = 1024;    // Inkling2::kKey
constexpr int kValueTile = 128;
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

struct MixInputs { const float* x[5]; };

// A depth query attends all finished blocks and the current partial block. RMS-normalized states are values too.
// Each input's two sums take the same per-thread order and the same reduction tree as one block_reduce() each, but
// a thread loads its 18 values of every input at once and all ten sums go up the tree together: a single block (one
// token) waits for memory once, not 36 times. The products are fused as the compiler fuses them in the per-element
// loops (explicitly: with five inputs in registers it would not).
__device__ __forceinline__ void mix_row(const MixInputs& in, int count, const float* query, float* out, size_t row) {
    constexpr int kMost = 5, kWaves = kThreads / 32, kPer = kRowWidth / kThreads;
    static_assert(kRowWidth % kThreads == 0, "a row is whole loads of the block");
    __shared__ float waves[2 * kMost][kWaves];
    float x[kMost][kPer], q[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) q[k] = query[threadIdx.x + k * kThreads];
#pragma unroll
    for (int j = 0; j < kMost; ++j)
        if (j < count)
#pragma unroll
            for (int k = 0; k < kPer; ++k) x[j][k] = in.x[j][row + threadIdx.x + k * kThreads];
    const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        if (j >= count) break;
        float ss = 0.f, dot = 0.f;
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            ss = __fmaf_rn(x[j][k], x[j][k], ss);
            dot = __fmaf_rn(x[j][k], q[k], dot);
        }
        ss = wave_reduce<false>(ss);
        dot = wave_reduce<false>(dot);
        if (lane == 0) {
            waves[2 * j][wave] = ss;
            waves[2 * j + 1][wave] = dot;
        }
    }
    __syncthreads();
    if (wave == 0)
#pragma unroll
        for (int v = 0; v < 2 * kMost; ++v) {
            if (v >= 2 * count) break;
            const float y = wave_reduce<false>(lane < kWaves ? waves[v][lane] : 0.f);
            if (lane == 0) waves[v][0] = y;
        }
    __syncthreads();
    float scale[kMost], w[kMost], top = -INFINITY;
#pragma unroll
    for (int j = 0; j < kMost; ++j)
        if (j < count) {
            scale[j] = rsqrtf(waves[2 * j][0] / kRowWidth + 1.e-6f);
            w[j] = waves[2 * j + 1][0] * scale[j] * rsqrtf((float) kRowWidth);
            top = fmaxf(top, w[j]);
        }
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kMost; ++j)
        if (j < count) sum += (w[j] = expf(w[j] - top));
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        float val = 0.f;
#pragma unroll
        for (int j = 0; j < kMost; ++j)
            if (j < count) val = __fmaf_rn(w[j] * scale[j], x[j][k], val);
        out[row + threadIdx.x + k * kThreads] = val / sum;
    }
}

__global__ void mix_kernel(MixInputs in, int count, const float* query, float* out) {
    mix_row(in, count, query, out, (size_t) blockIdx.x * kRowWidth);
}

// A visit's streams (Inkling2::heads()): stream blockIdx.y's query and output, each row as mix_kernel's.
struct Streams3 {
    const float* query[3];
    float* out[3];
};
__global__ void mix3_kernel(MixInputs in, int count, Streams3 s) {
    mix_row(in, count, s.query[blockIdx.y], s.out[blockIdx.y], (size_t) blockIdx.x * kRowWidth);
}

__global__ void control_kernel(float* cur, const float* msg, const float* gain, const float* forget, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) cur[i] = (1.f - tanhf(forget[i])) * cur[i] + tanhf(gain[i]) * msg[i];
}

// One thread owns one channel and keeps its raw input history across the chunk. In GGUF the static taps
// run oldest to newest; the input-dependent taps of each KV head start with the current token.
// With `snap`, a window's (Engine::window()): the state after each of its tokens but the last as well, token t's
// `stride` floats after token t - 1's, at the place the state has in state_ (Engine::keep() puts one back).
__device__ __forceinline__ void conv_channel(float* x, float* state, const float* w, const float* dynamic, int width,
                                             int n, int i, float* snap = nullptr, size_t stride = 0) {
    float* h = state + 3 * i;
    float a = h[0], b = h[1], c = h[2];
    w += 4 * i;
    const float scale = dynamic ? 1.f / 3.f : 0.4472135954999579f;
    for (int t = 0; t < n; ++t) {
        const size_t at = (size_t) t * width + i;
        const float d = x[at];
        float s = a * w[0] + b * w[1] + c * w[2] + d * w[3];
        if (dynamic) {
            const float* p = dynamic + 4 * (t * (width / 128) + i / 128);
            s += d * p[0] + c * p[1] + b * p[2] + a * p[3];
        }
        x[at] = (d + s) * scale;
        a = b;
        b = c;
        c = d;
        if (snap && t + 1 < n) {
            float* k = snap + (size_t) t * stride + 3 * i;
            k[0] = a;
            k[1] = b;
            k[2] = c;
        }
    }
    h[0] = a;
    h[1] = b;
    h[2] = c;
}

__global__ void conv_kernel(float* x, float* state, const float* w, const float* dynamic, int width, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) conv_channel(x, state, w, dynamic, width, n, i);
}

// The decode switch's launches that each do what two to four of the ones above do, with the same sums.
struct Conv {
    float* x;
    float* state;
    const float* w;
    const float* dynamic;
    float* snap = nullptr;   // conv_channel()'s
};
// K's and V's convolutions (kKey channels each) in one launch.
__global__ void conv2_kernel(Conv k, Conv v, int width, int n, size_t stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool second = i >= width;
    if (second) i -= width;
    if (i >= width) return;
    const Conv& c = second ? v : k;
    conv_channel(c.x, c.state, c.w, c.dynamic, width, n, i, c.snap, stride);
}

// A static convolution of x (plus `add`, the CPU experts' sum, when there is one) added to the residual r, as
// axpy(add, 1, x), conv() and axpy(x, 1, r) do. With `depth`, a block of layers starts: depth takes r and r takes x,
// as axpy(r, 1, depth) and axpy(x, 1, r) without adding do. With `moe`, the routed experts' rows in the router's
// order (STRATA_INKLING_TAIL_MOE, one token's: n == 1) are the x itself, as the add_scaled() chain wrote it into fo_:
// the same sums, one launch less.
__global__ void conv_add_kernel(float* x, const float* add, float* state, const float* w, float* r, float* depth,
                                int width, int n, bool clean, TailSum moe, float* snap = nullptr, size_t stride = 0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    if (moe.n)
        for (int t = 0; t < n; ++t) {
            const size_t at = (size_t) t * width + i;
            // fo_ was 0 where add_scaled() started (moe_shared()), so this chain is its sums bit for bit
            float v = 0.f;
            for (int k = 0; k < moe.n; ++k) v = v + moe.x[k][at] * moe.w[k];
            x[at] = v;
        }
    if (add)
        for (int t = 0; t < n; ++t) x[(size_t) t * width + i] = x[(size_t) t * width + i] + add[(size_t) t * width + i];
    conv_channel(x, state, w, nullptr, width, n, i, snap, stride);
    for (int t = 0; t < n; ++t) {
        const size_t at = (size_t) t * width + i;
        if (depth) {
            depth[at] = r[at];
            r[at] = x[at];
        } else r[at] = r[at] + x[at];
        if (clean) x[at] = 0.f;   // fo_: the next layer's experts add into it (moe_shared())
    }
}

// A prompt batch's convolution in spans of kConvSpan tokens, a thread a channel and span: conv_halo_kernel keeps the
// three raw inputs before each span (the state's before the first) and conv_span_kernel computes each span as
// conv_kernel does, with the same sums, so the results are conv_kernel's bit for bit.
constexpr int kConvSpan = 64;
__global__ void conv_halo_kernel(const float* x, const float* state, float* halo, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, span = blockIdx.y;
    if (i >= width) return;
    float* to = halo + ((size_t) span * width + i) * 3;
    for (int j = 0; j < 3; ++j) {
        const int t = span * kConvSpan - 3 + j;
        to[j] = t >= 0 ? x[(size_t) t * width + i] : state[3 * i + 3 + t];
    }
}

__global__ void conv_span_kernel(float* x, float* state, const float* halo, const float* w, const float* dynamic,
                                 int width, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, span = blockIdx.y;
    if (i >= width) return;
    const float* from = halo + ((size_t) span * width + i) * 3;
    float a = from[0], b = from[1], c = from[2];
    w += 4 * i;
    const float scale = dynamic ? 1.f / 3.f : 0.4472135954999579f;
    const int first = span * kConvSpan, end = min(n, first + kConvSpan);
    for (int t = first; t < end; ++t) {
        const size_t at = (size_t) t * width + i;
        const float d = x[at];
        float s = a * w[0] + b * w[1] + c * w[2] + d * w[3];
        if (dynamic) {
            const float* p = dynamic + 4 * (t * (width / 128) + i / 128);
            s += d * p[0] + c * p[1] + b * p[2] + a * p[3];
        }
        x[at] = (d + s) * scale;
        a = b;
        b = c;
        c = d;
    }
    if (end == n) {
        float* h = state + 3 * i;
        h[0] = a;
        h[1] = b;
        h[2] = c;
    }
}

__device__ __forceinline__ void head_norm(float* x, const float* w, int width, int heads, float clip, float eps,
                                          int64_t pos, int floor, float alpha, int head, int row, float* waves) {
    const int i = threadIdx.x;
    float* dest = x + ((size_t) row * heads + head) * width;
    const float v = i < width ? dest[i] * (w ? w[head * width + i] : 1.f) : 0.f;
    const float ss = block_reduce(v * v, waves);
    const float s = clip > 0.f ? fminf(1.f, clip / (sqrtf(ss) + 1.e-6f)) : rsqrtf(ss / width + eps);
    const float tau = floor > 0 ? 1.f + alpha * logf(fmaxf(1.f, float(pos + row + 1) / floor)) : 1.f;
    if (i < width) dest[i] = v * s * tau;
}

__global__ void head_norm_kernel(float* x, const float* w, int width, int heads, float clip, float eps,
                                 int64_t pos, int floor, float alpha) {
    __shared__ float waves[kThreads / 32];
    head_norm(x, w, width, heads, clip, eps, pos, floor, alpha, blockIdx.x, blockIdx.y, waves);
}

// Q's, K's, V's and the relative terms' norms in one launch, a block a head as head_norm_kernel.
struct HeadNorm {
    float* x;
    const float* w;
    int width, heads;
    float clip;
    int floor;
};
struct HeadNorms { HeadNorm s[4]; };
__global__ void head_norms_kernel(HeadNorms h, float eps, int64_t pos, float alpha) {
    __shared__ float waves[kThreads / 32];
    int head = blockIdx.x, j = 0;
    while (j < 3 && head >= h.s[j].heads) head -= h.s[j++].heads;
    const HeadNorm& s = h.s[j];
    head_norm(s.x, s.w, s.width, s.heads, s.clip, eps, pos, s.floor, alpha, head, blockIdx.y, waves);
}

// One decode row's conv2_kernel, head_norms_kernel and store2_q8_kernel in one launch, a block a head as
// head_norms_kernel: a K or V head's block first convolves its channels (a thread a channel, as conv2_kernel), then
// norms the head, then each wave writes its 32 values' Q8_0 block as store2_q8_kernel does (the block's maximum by
// shuffles: the same maximum, so the same scale and codes).  The normed K and V stay in k and v as well.
struct KvStore {
    int8_t *kc, *vc;
    __half *ks, *vs;
    int64_t at;   // the row's place in the cache
};
__global__ void heads_kv_kernel(Conv k, Conv v, HeadNorms h, KvStore c, float eps, int64_t pos, float alpha,
                                size_t stride) {
    __shared__ float waves[kThreads / 32];
    int head = blockIdx.x, j = 0;
    while (j < 3 && head >= h.s[j].heads) head -= h.s[j++].heads;
    static_assert(kThreads == 128, "a thread a value of a 128-wide K or V head");
    const int i = head * kThreads + threadIdx.x;
    if (j == 1 || j == 2) {
        const Conv& cv = j == 2 ? v : k;
        conv_channel(cv.x, cv.state, cv.w, cv.dynamic, kKvWidth, 1, i, cv.snap, stride);
    }
    const HeadNorm& s = h.s[j];
    head_norm(s.x, s.w, s.width, s.heads, s.clip, eps, pos, s.floor, alpha, head, 0, waves);
    if (j != 1 && j != 2) return;
    const float x = s.x[i];
    float amax = fmaxf(0.f, fabsf(x));   // store2_q8_kernel's maximum starts at 0: a NaN does not count
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    const float d = amax / 127.0f;
    const float id = d > 0.f ? 1.0f / d : 0.0f;
    constexpr int nb = kKvWidth / 32;
    if (threadIdx.x % 32 == 0) (j == 2 ? c.vs : c.ks)[(size_t) c.at * nb + i / 32] = __float2half(d);
    float q = rintf(x * id);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    (j == 2 ? c.vc : c.kc)[(size_t) c.at * kKvWidth + i] = (int8_t) q;
}

// K and V into their Q8_0 caches in one launch, as store_q8_kernel writes them, a row at (pos + row) % ring (or
// pos + row in a full layer's cache).
__global__ void store2_q8_kernel(const float* k, const float* v, int8_t* kc, __half* ks, int8_t* vc, __half* vs,
                                 int64_t pos, int ring, int rows) {
    constexpr int nb = kKvWidth / 32;
    int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= 2 * nb * rows) return;
    const bool second = gb >= nb * rows;
    if (second) gb -= nb * rows;
    const int row = gb / nb, blk = gb % nb;
    const int64_t at = ring ? (pos + row) % ring : pos + row;
    const float* xr = (second ? v : k) + (size_t) row * kKvWidth + blk * 32;
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xr[j]));
    const float d = amax / 127.0f;
    const float id = d > 0.f ? 1.0f / d : 0.0f;
    (second ? vs : ks)[(size_t) at * nb + blk] = __float2half(d);
    int8_t* cr = (second ? vc : kc) + (size_t) at * kKvWidth + blk * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float q = rintf(xr[j] * id);
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        cr[j] = (int8_t) q;
    }
}

// A loop stream's control (Inkling2::control) in three launches instead of seven, the first: the message as
// mix_kernel mixes the banks, and the RMS norms of the stream and the message side by side in pair.
__device__ __forceinline__ void control_in_row(const float* bank0, const float* bank1, const float* query,
                                               const float* in, float* msg, float* pair, size_t row) {
    constexpr int kPer = kRowWidth / kThreads, kWaves = kThreads / 32;
    __shared__ float waves[6][kWaves];
    float b0[kPer], b1[kPer], x[kPer], q[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = threadIdx.x + k * kThreads;
        q[k] = query[i];
        b0[k] = bank0[row + i];
        b1[k] = bank1[row + i];
        x[k] = in[row + i];
    }
    float sums[6] = {};   // the banks' squares and products with the query, the stream's squares, the message's
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        sums[0] = __fmaf_rn(b0[k], b0[k], sums[0]);
        sums[1] = __fmaf_rn(b0[k], q[k], sums[1]);
        sums[2] = __fmaf_rn(b1[k], b1[k], sums[2]);
        sums[3] = __fmaf_rn(b1[k], q[k], sums[3]);
        sums[4] += x[k] * x[k];
    }
    const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
    const auto reduce = [&](int first, int count) {
        for (int j = first; j < first + count; ++j) {
            const float y = wave_reduce<false>(sums[j]);
            if (lane == 0) waves[j][wave] = y;
        }
        __syncthreads();
        if (wave == 0)
            for (int j = first; j < first + count; ++j) {
                const float y = wave_reduce<false>(lane < kWaves ? waves[j][lane] : 0.f);
                if (lane == 0) waves[j][0] = y;
            }
        __syncthreads();
        for (int j = first; j < first + count; ++j) sums[j] = waves[j][0];
    };
    reduce(0, 5);
    const float s0 = rsqrtf(sums[0] / kRowWidth + 1.e-6f), s1 = rsqrtf(sums[2] / kRowWidth + 1.e-6f);
    float w0 = sums[1] * s0 * rsqrtf((float) kRowWidth), w1 = sums[3] * s1 * rsqrtf((float) kRowWidth);
    const float top = fmaxf(w0, w1);
    w0 = expf(w0 - top);
    w1 = expf(w1 - top);
    const float sum = w0 + w1;
    float m[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        float val = __fmaf_rn(w0 * s0, b0[k], 0.f);
        val = __fmaf_rn(w1 * s1, b1[k], val);
        m[k] = val / sum;
        msg[row + threadIdx.x + k * kThreads] = m[k];
        sums[5] += m[k] * m[k];
    }
    __syncthreads();
    reduce(5, 1);
    const float sx = 1.0f / sqrtf(sums[4] / float(kRowWidth) + 1.e-6f), sm = 1.0f / sqrtf(sums[5] / float(kRowWidth) + 1.e-6f);
    float* out = pair + 2 * row;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = threadIdx.x + k * kThreads;
        out[i] = x[k] * sx;
        out[kRowWidth + i] = m[k] * sm;
    }
}

__global__ void control_in_kernel(const float* bank0, const float* bank1, const float* query, const float* in,
                                  float* msg, float* pair) {
    control_in_row(bank0, bank1, query, in, msg, pair, (size_t) blockIdx.x * kRowWidth);
}

// The streams' control (Inkling2::heads()): stream blockIdx.y's, each row as the one-stream kernels'.
struct Control3 {
    const float* query[3];
    float* cur[3];
    float* msg[3];
    float* pair[3];
    const float* gain[3];
    const float* forget[3];
    const float* rank[3];
};
__global__ void control_in3_kernel(const float* bank0, const float* bank1, Control3 c) {
    const int s = blockIdx.y;
    control_in_row(bank0, bank1, c.query[s], c.cur[s], c.msg[s], c.pair[s], (size_t) blockIdx.x * kRowWidth);
}

// The third: the gain and the forget gate of each value from the rank, eight lanes an output, applied to the stream.
__device__ __forceinline__ void control_out_body(const float* gain, const float* forget, const float* rank,
                                                 const float* msg, float* cur, int ranks) {
    const int lane = threadIdx.x % 8, o = blockIdx.x * 32 + threadIdx.x / 8, t = blockIdx.y;
    const float4* g = reinterpret_cast<const float4*>(gain + (size_t) o * ranks);
    const float4* f = reinterpret_cast<const float4*>(forget + (size_t) o * ranks);
    const float4* r = reinterpret_cast<const float4*>(rank + (size_t) t * ranks);
    float ga = 0.f, fa = 0.f;
    for (int i = lane; i < ranks / 4; i += 8) {
        const float4 a = g[i], b = f[i], x = r[i];
        ga += a.x * x.x + a.y * x.y + a.z * x.z + a.w * x.w;
        fa += b.x * x.x + b.y * x.y + b.z * x.z + b.w * x.w;
    }
#pragma unroll
    for (int d = 4; d > 0; d /= 2) {
        ga += __shfl_xor(ga, d, 32);
        fa += __shfl_xor(fa, d, 32);
    }
    if (lane == 0) {
        const size_t at = (size_t) t * kRowWidth + o;
        cur[at] = (1.f - tanhf(fa)) * cur[at] + tanhf(ga) * msg[at];
    }
}

__global__ void __launch_bounds__(256) control_out_kernel(const float* gain, const float* forget, const float* rank,
                                                          const float* msg, float* cur, int ranks) {
    control_out_body(gain, forget, rank, msg, cur, ranks);
}

__global__ void __launch_bounds__(256) control_out3_kernel(Control3 c, int ranks) {
    const int s = blockIdx.z;
    control_out_body(c.gain[s], c.forget[s], c.rank[s], c.msg[s], c.cur[s], ranks);
}

}  // namespace

void Inkling2::load_geometry() {
    int device = 0;
    cudaDeviceProp p{};
    ck(cudaGetDevice(&device), "Inkling device");
    ck(cudaGetDeviceProperties(&p, device), "Inkling device properties");
    bool rdna2 = false;
    for (const char* gcn : {"gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"})
        rdna2 = rdna2 || kernels::gfx_arch_is(p.gcnArchName, gcn);
    if (!rdna2) throw std::runtime_error("Inkling-2-Mini requires a HIP RDNA2 card");
    if (opt_.batch || !opt_.layer_split.empty() || opt_.tail_skip || opt_.skip_weight > 0.f)
        throw std::runtime_error("Inkling-2-Mini requires one sequence and one card, without expert skipping");
    // --draft-model is the DSpark drafter, a model of its own and not a file with a block load_shape() would look
    // for.  Without it the model checks no guesses: a window of one token.
    std::unique_ptr<Shards> drafter = std::move(dg_);
    opt_.draft = false;
    const char* flash = std::getenv("STRATA_INKLING_FLASH");
    flash_ = flash && flash[0] == '1' && flash[1] == '\0';
    if (flash_) std::fputs("strata-poly: STRATA_INKLING_FLASH=1: fused attention for prompt batches\n", stderr);
    const char* prefetch = std::getenv("STRATA_INKLING_PREFETCH");
    prefetch_early_ = prefetch && prefetch[0] == '1' && prefetch[1] == '\0';
    if (prefetch_early_)
        std::fputs("strata-poly: STRATA_INKLING_PREFETCH=1: prompt experts queued before router readback\n", stderr);
    const char* kcopy = std::getenv("STRATA_INKLING_KCOPY");
    kcopy_ = kcopy && kcopy[0] == '1' && kcopy[1] == '\0';
    if (kcopy_)
        std::fputs("strata-poly: STRATA_INKLING_KCOPY=1: prompt router scores and expert tables copied by a kernel\n",
                   stderr);
    const char* decode = std::getenv("STRATA_INKLING_DECODE");
    decode_ = decode && decode[0] == '1' && decode[1] == '\0';
    wave_gemv_ = head_warps_ = situ_q8_ = decode_;
    // A miss in a once-visited layer (0-5, 18-23) stalls the card: the CPU's experts there are 2-3 a visit and the
    // card waited 200-340 us a visit for them, against 5-70 us in the core visits, whose card experts hide one or two.
    // Counted per visit, the core's uses (three a token) held ~1,200 of the ~1,330 slots.  Weighed twice, forced decode
    // 39.07 -> 40.15 tok/s and 16K 30.87 -> 32.65 (RX 6700 XT; 1.5 to 3 measured the same).
    if (decode_) {
        use_w_.assign(kPhysical, 1.f);
        for (int il = 0; il < kPhysical; ++il)
            if (il < kStart || il >= kStart + kCore) use_w_[(size_t) il] = 2.f;
    }
    if (decode_) std::fputs("strata-poly: STRATA_INKLING_DECODE=1: decode kernels for a few tokens at a time\n", stderr);
    const char* q4_slots = std::getenv("STRATA_INKLING_Q4_0_SLOTS");
    inkling_q4_0_slots_ = decode_ && kernels::gfx_arch_is(p.gcnArchName, "gfx1031") &&
                         q4_slots && q4_slots[0] == '1' && q4_slots[1] == '\0';
    if (inkling_q4_0_slots_)
        std::fputs("strata-poly: STRATA_INKLING_Q4_0_SLOTS=1: four-row expert MMVQ on gfx1031\n", stderr);
    const char* norm_q8 = std::getenv("STRATA_INKLING_NORM_Q8");
    norm_q8_ = decode_ && norm_q8 && norm_q8[0] == '1' && norm_q8[1] == '\0';
    if (norm_q8_)
        std::fputs("strata-poly: STRATA_INKLING_NORM_Q8=1: a visit's stream norms and their Q8_1 in one launch\n",
                   stderr);
    const char* tail_moe = std::getenv("STRATA_INKLING_TAIL_MOE");
    moe_tail_ = decode_ && tail_moe && tail_moe[0] == '1' && tail_moe[1] == '\0';
    if (moe_tail_)
        std::fputs("strata-poly: STRATA_INKLING_TAIL_MOE=1: a fused visit's routed experts summed in its tail launch\n",
                   stderr);
    const char* route_dual = std::getenv("STRATA_INKLING_ROUTE_DUAL");
    route_dual_ = decode_ && route_dual && route_dual[0] == '1' && route_dual[1] == '\0';
    if (route_dual_)
        std::fputs("strata-poly: STRATA_INKLING_ROUTE_DUAL=1: the router's copies to the host ride its producers\n",
                   stderr);
    if (!q8() && opt_.kv_format != KvFmt::kFp16)
        throw std::runtime_error("Inkling-2-Mini supports Q8_0 or FP16 KV");
    const int window = opt_.window;
    opt_.window = 1;
    load_shape();

    const auto need = [](bool ok, const char* key) {
        if (!ok) throw std::runtime_error(std::string("not an Inkling-2-Mini layout: ") + key);
    };
    const auto integer = [&](const char* key, int expected) { need(shape_i(key) == expected, key); };
    loop_layout_ = gg_->get("general.architecture")->s == "inkling2_loop";
    const int declared_layers = loop_layout_ ? kPhysical : kVisits;
    need(n_layer_ == declared_layers && n_embd_ == kWidth && n_expert_ == 128 && n_used_ == 8 && n_ff_exp_ == 1152,
         "layers, width or experts");
    integer(loop_layout_ ? "loop.count" : "num_loops", 3);
    integer(loop_layout_ ? "loop.start" : "loop.core_start", kStart);
    integer(loop_layout_ ? "loop.end" : "loop.core_end", kStart + kCore);
    if (loop_layout_) {
        const auto* varying = meta_in(*gg_, "loop.varying");
        need(!varying || (varying->is_num() && varying->u == 0), "variable loop");
    }
    integer("loop.gate_rank", kRank);
    integer("attn_res.block_size", kBlock);
    integer("attention.sliding_window", kWindow);
    integer("attention.relative_length", kRelative);
    integer("attention.relative_extent", kWindow);
    integer("shortconv.l_cache", kTaps);
    integer("rope.dimension_count", 0);
    for (const char* key : {"attention.key_length", "attention.value_length", "attention.key_length_swa",
                            "attention.value_length_swa"}) integer(key, kHead);
    const auto array = [&](const char* key, auto expected) {
        const auto* value = meta_in(*gg_, key);
        need(value && value->count == (uint64_t) declared_layers && value->items.size() == (size_t) declared_layers, key);
        for (int i = 0; i < declared_layers; ++i) need(value->items[i].u == (uint64_t) expected(i), key);
    };
    array("attention.head_count", [](int) { return kHeads; });
    array("attention.head_count_kv", [](int) { return kKv; });
    array("attention.sliding_window_pattern", [](int i) { return local(i); });
    padded_vocab_ = n_vocab_;
    n_vocab_ = shape_i("vocab_size_unpadded");
    const auto* tokens = gg_->get("tokenizer.ggml.tokens");
    need(n_vocab_ == 200058 && padded_vocab_ == 201024 && tokens && tokens->count == (uint64_t) n_vocab_, "vocabulary");
    need(gg_->find("output.weight") && !draft_, "output or draft");
    qk_clip_ = shape_f("attention.qk_clip");
    r_clip_ = shape_f("attention.relative_clip");
    floor_ = shape_i("attention.log_scale_floor");
    alpha_ = shape_f("attention.log_scale_alpha");
    expert_scale_ = shape_f("expert_weights_scale");
    cap_ = shape_f("final_logit_softcapping");
    situ_ = {shape_f("activation.situ_beta"), shape_f("activation.situ_linear_beta")};
    for (float value : {rms_eps_, qk_clip_, r_clip_, expert_scale_, cap_, situ_.beta, situ_.linear_beta})
        need(std::isfinite(value) && value > 0.f, "numerical parameters");
    need(std::isfinite(alpha_) && alpha_ >= 0.f && floor_ > 0, "log scale");
    n_layer_ = kPhysical;
    looped_ = true;
    clamp_exp_.assign(kPhysical + 1, INFINITY);
    if (drafter) {
        // a window's guesses need the short convolutions' state after each of its tokens, which the decode switch's
        // kernels leave (layer_head(), layer_tail())
        if (!decode_) throw std::runtime_error("the DSpark drafter of Inkling-2-Mini needs STRATA_INKLING_DECODE=1");
        // its block of 8 rows goes through the scratch that holds a call's rows
        if (opt_.prompt_batch < kDsBlock)
            throw std::runtime_error("the DSpark drafter of Inkling-2-Mini needs --prompt-batch 8 or more");
        load_drafter_shape(*drafter);
        dg_ = std::move(drafter);
        block_ = dg_.get();   // the file the drafter's tensors are read from (home())
        draft_ = true;
        block_layer_ = false;
        cols_kernels_ = true;
        opt_.draft = true;
        opt_.window = window > 0 ? std::min(window, kDsBlock) : kDsBlock;
        ds_rows_ = std::min(kDsBlock - 1, opt_.window - 1);
        std::fprintf(stderr, "strata-poly: DSpark drafter %s: up to %d guesses a window\n", opt_.draft_model.c_str(),
                     ds_rows_);
    }
}

// The drafter's shape must be the one graph_inkling2 runs for this model (llama.cpp's dspark.cpp): three layers of
// this width and heads, the 8-token block, taps at visits 5, 23, 41 and the output, a recurrent head of rank 256.
void Inkling2::load_drafter_shape(const Shards& g) {
    const std::string file = opt_.draft_model;
    const auto need = [&](bool ok, const std::string& what) {
        if (!ok) throw std::runtime_error(file + " is not the DSpark drafter of Inkling-2-Mini: " + what);
    };
    const MetaValue* arch = g.get("general.architecture");
    need(arch && arch->s == "dflash", "its architecture is not 'dflash'");
    need(g.find("dspark_hidden_out.weight") && g.find("markov_w1.weight") && g.find("joint_proj.weight") &&
             g.find("dspark_fuse.weight"),
         "no Inkling2 DSpark head (dspark_hidden_out, markov_w1, joint_proj, dspark_fuse)");
    const auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = meta_in(g, key);
        need(v != nullptr, std::string("no dflash.") + key);
        return *v;
    };
    const auto is = [&](const char* key, uint64_t value) { need(must(key).u == value, std::string("dflash.") + key); };
    is("block_count", kDsLayers);
    is("embedding_length", kWidth);
    is("feed_forward_length", kDsFfn);
    is("attention.head_count", kHeads);
    is("attention.head_count_kv", kKv);
    is("attention.key_length", kHead);
    is("attention.value_length", kHead);
    is("rope.dimension_count", kHead);
    is("block_size", kDsBlock);
    is("attention.sliding_window", kWindow);
    is("vocab_size_unpadded", (uint64_t) n_vocab_);
    need(must("sample_from_anchor").u == 0 && must("attention.causal").u == 0,
         "it samples from the anchor or is causal");
    const MetaValue& taps = must("target_layers");
    need(taps.items.size() == 4 && taps.items[0].u == (uint64_t) kTap[0] && taps.items[1].u == (uint64_t) kTap[1] &&
             taps.items[2].u == (uint64_t) kTap[2] && taps.items[3].u == (uint64_t) kVisits,
         "its target_layers are not [5, 23, 41, 48]");
    const MetaValue& pattern = must("attention.sliding_window_pattern");
    need(pattern.items.size() == kDsLayers, "dflash.attention.sliding_window_pattern");
    for (int l = 0; l < kDsLayers; ++l) need((pattern.items[(size_t) l].u != 0) == ds_swa(l), "its window pattern");
    ds_eps_ = (float) must("attention.layer_norm_rms_epsilon").f;
    ds_cap_ = (float) must("final_logit_softcapping").f;
    ds_theta_ = std::pow((float) must("rope.freq_base").f, -2.f / kHead);
    need(std::isfinite(ds_eps_) && ds_eps_ > 0.f && std::isfinite(ds_cap_) && ds_cap_ > 0.f && ds_theta_ > 0.f,
         "its epsilon, logit cap or rope base");
    const TensorInfo* w1 = g.find("markov_w1.weight");
    need(w1->type == GGML_TYPE_F16 && w1->shape.size() == 2 && w1->shape[0] == (uint64_t) kDsRank &&
             w1->shape[1] >= (uint64_t) n_vocab_,
         "markov_w1 is not F16 [256, vocabulary]");
    const TensorInfo* fuse = g.find("dspark_fuse.weight");
    need(fuse->type == kTypeF32 && fuse->elements() == 4 * (kDsLayers + 1), "dspark_fuse is not F32 [4, 4]");
    const TensorInfo* hidden = g.find("dspark_hidden_out.weight");
    const TensorInfo* hidden_b = g.find("dspark_hidden_out.bias");
    need(hidden->type == GGML_TYPE_F16 && hidden->elements() == (uint64_t) kDsRank * kWidth && hidden_b &&
             hidden_b->type == kTypeF32 && hidden_b->elements() == (uint64_t) kWidth,
         "dspark_hidden_out is not F16 [256, 2304] with an F32 bias");
}

void Inkling2::begin_pass() {
    const auto part = [&](const char* name, int rows) { return upload(gg_->find(name), name, kWidth, rows, kF32); };
    token_norm_ = part("token_embd_norm.weight", 1);
    output_query_ = part("output_res_query.weight", 1);
    if (!loop_layout_) {
        loop_query_ = part("loop_res_query.weight", 2 * kCore);
        loop_kv_query_ = part("loop_kv_res_query.weight", 2 * kCore);
    }
}

void Inkling2::load_layer(int il) {
    Block& b = blocks_[il];
    Layer& L = layers_[il];
    const auto part = [&](const char* name, int cols, int rows = 1, Want want = kF32) {
        return upload(tensor(il, name), blk(il, name), cols, rows, want);
    };
    b.q = part("attn_q.weight", kWidth, kQuery, kAnyMat);
    b.k = part("attn_k.weight", kWidth, kKey, kAnyMat);
    b.v = part("attn_v.weight", kWidth, kKey, kAnyMat);
    b.o = part("attn_output.weight", kQuery, kWidth, kAnyMat);
    b.gate = part("attn_gate.weight", kWidth, kQuery, kAnyMat);
    b.rel = part("attn_rel.weight", kWidth, kRelative * kHeads, kAnyMat);
    b.sink = part("attn_sink_proj.weight", kWidth, kHeads);
    b.qclip = part("attn_q_clip.weight", kHead, kHeads);
    b.rclip = part("attn_rel_clip.weight", kRelative, kHeads);
    b.table = part("attn_rel_bias.weight", kRelative, kWindow);
    b.kconv = part("attn_k_conv.weight", kTaps, kKey);
    b.vconv = part("attn_v_conv.weight", kTaps, kKey);
    b.aconv = part("attn_post_conv.weight", kTaps, kWidth);
    b.fconv = part("ffn_post_conv.weight", kTaps, kWidth);
    if (local(il)) L.attn_norm = part("attn_norm.weight", kWidth);
    else {
        L.attn_norm = part("attn_q_inp_norm.weight", kWidth);
        b.knorm = part("attn_k_inp_norm.weight", kWidth);
        b.vnorm = part("attn_v_inp_norm.weight", kWidth);
        b.kproj = part("attn_k_conv_proj.weight", kWidth, kTaps * kKv);
        b.vproj = part("attn_v_conv_proj.weight", kWidth, kTaps * kKv);
    }
    if (il > 0) b.aquery = part("attn_res_query.weight", kWidth, local(il) ? 1 : 3);
    b.fquery = part("ffn_res_query.weight", kWidth);
    L.ffn_norm = part("ffn_norm.weight", kWidth);
    if (!tensor(il, "ffn_gate_up_exps.weight"))
        throw std::runtime_error(blk(il, "ffn_gate_up_exps.weight is required for Inkling-2-Mini"));
    load_moe(il, false);
    if ((L.gu_type == GGML_TYPE_NVFP4 && L.h_s_gate.empty()) ||
        (L.d_type == GGML_TYPE_NVFP4 && L.h_s_down.empty()))
        throw std::runtime_error(blk(il, "NVFP4 experts require their scales"));
    const auto* bias = tensor(il, "exp_probs_b.bias");
    if (!bias || bias->type != kTypeF32 || bias->elements() != (uint64_t) n_expert_)
        throw std::runtime_error(blk(il, "exp_probs_b.bias must be an F32 expert vector"));
    b.bias.resize(n_expert_);
    home(*bias).payload(*bias);
    std::memcpy(b.bias.data(), home(*bias).data(*bias), b.bias.size() * sizeof(float));

    if (il >= kStart && il < kStart + kCore) {
        const char* streams[4] = {"attn", "k", "v", "mlp"};
        const char* loop_streams[4] = {"q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"};
        if (loop_layout_) b.message_query = part("loop_msg_query.weight", kWidth, 4);
        for (int i = 0; i < 4; ++i) {
            const std::string pre = std::string("loop_") + (loop_layout_ ? loop_streams[i] : streams[i]);
            b.control[i] = {part((pre + "_down.weight").c_str(), 2 * kWidth, kRank),
                            part((pre + "_gain.weight").c_str(), kRank, kWidth),
                            part((pre + "_forget.weight").c_str(), kRank, kWidth)};
        }
    }
}

// The drafter's weights (both passes of load(): the first adds up the arena).  The recurrent head's token table and
// hidden_out stay on the host: a step reads one row of the table, and hidden_out goes into W' (prepare_head()).
void Inkling2::load_drafter() {
    const Shards& g = *dg_;
    const auto up = [&](const std::string& name, int64_t n_in, int64_t n_out, Want want) {
        const TensorInfo* t = g.find(name);
        if (!t) throw std::runtime_error(opt_.draft_model + " has no " + name);
        return upload(t, name, n_in, n_out, want);
    };
    for (int l = 0; l < kDsLayers; ++l) {
        DsLayer& L = ds_[(size_t) l];
        const auto name = [&](const char* n) { return "blk." + std::to_string(l) + "." + n; };
        L.attn_norm = up(name("attn_norm.weight"), kWidth, 1, kF32);
        L.ffn_norm = up(name("ffn_norm.weight"), kWidth, 1, kF32);
        L.ctx_norm = up(name("dflare_norm.weight"), kWidth, 1, kF32);
        L.q = up(name("attn_q.weight"), kWidth, kQuery, kAnyMat);
        L.k = up(name("attn_k.weight"), kWidth, kKey, kAnyMat);
        L.v = up(name("attn_v.weight"), kWidth, kKey, kAnyMat);
        L.o = up(name("attn_output.weight"), kQuery, kWidth, kAnyMat);
        L.k_ctx = up(name("attn_k_ctx.weight"), kWidth, kKey, kAnyMat);
        L.v_ctx = up(name("attn_v_ctx.weight"), kWidth, kKey, kAnyMat);
        L.q_norm = up(name("attn_q_norm.weight"), kHead, 1, kF32);
        L.k_norm = up(name("attn_k_norm.weight"), kHead, 1, kF32);
        L.k_ctx_norm = up(name("attn_k_ctx_norm.weight"), kHead, 1, kF32);
        L.gate = up(name("ffn_gate.weight"), kWidth, kDsFfn, kAnyMat);
        L.up = up(name("ffn_up.weight"), kWidth, kDsFfn, kAnyMat);
        L.down = up(name("ffn_down.weight"), kDsFfn, kWidth, kAnyMat);
    }
    ds_out_norm_ = up("output_norm.weight", kWidth, 1, kF32);
    ds_mask_ = up("mask_embd.weight", kWidth, 1, kF32);
    ds_rnn_norm_ = up("dspark_norm.weight", kWidth, 1, kF32);
    ds_joint_ = up("joint_proj.weight", 2 * kDsRank + 2 * kWidth, 3 * kDsRank, kF32);
    ds_joint_b_ = up("joint_proj.bias", 3 * kDsRank, 1, kF32);

    const auto host = [&](const char* name) {
        const TensorInfo* t = g.find(name);
        g.payload(*t);
        return g.data(*t);
    };
    const float* fuse = (const float*) host("dspark_fuse.weight");
    for (int r = 0; r <= kDsLayers; ++r) {   // a row's weights over the four taps: softmax(dspark_fuse[r])
        float most = -INFINITY, sum = 0.f;
        for (int j = 0; j < 4; ++j) most = std::max(most, fuse[r * 4 + j]);
        for (int j = 0; j < 4; ++j) sum += ds_mix_[r][j] = std::exp(fuse[r * 4 + j] - most);
        for (int j = 0; j < 4; ++j) ds_mix_[r][j] /= sum;
    }
    ds_w1_ = host("markov_w1.weight");
    const uint16_t* hidden = (const uint16_t*) host("dspark_hidden_out.weight");
    ds_hidden_.resize((size_t) kWidth * kDsRank);
    for (size_t i = 0; i < ds_hidden_.size(); ++i) ds_hidden_[i] = ggml_fp16_to_fp32(hidden[i]);
    const float* bias = (const float*) host("dspark_hidden_out.bias");
    ds_hidden_b_.assign(bias, bias + kWidth);
}

size_t Inkling2::recurrent_floats() const { return (size_t) kVisits * kConvFloats; }
size_t Inkling2::kv_bytes() const {
    return 2 * (q8() ? kKey + (kKey / 32) * sizeof(uint16_t) : kKey * sizeof(uint16_t));
}
size_t Inkling2::card_row_bytes() const { return kVisits / kBlock * kv_bytes() + (draft_ ? ds_row_bytes() : 0); }

void Inkling2::alloc_caches() {
    ring_ = std::min(opt_.max_context, kWindow + rows_ - 1);
    auto* rings = (uint8_t*) seq_alloc((size_t) (kVisits - kVisits / kBlock) * ring_ * kv_bytes(), "Inkling SWA KV");
    const size_t wide = kKey * (q8() ? 1 : sizeof(uint16_t));
    const size_t scale = q8() ? kKey / 32 * sizeof(uint16_t) : 0;
    for (int i = 0; i < kVisits; ++i) {
        for (int j = 0; j < 4; ++j) {
            const size_t stride = j % 2 ? scale : wide;
            auto& p = caches_[i].planes[j];
            p = {nullptr, stride};
            if (!stride) continue;
            if (local(i)) {
                p.data = rings;
                rings += (size_t) ring_ * stride;
            } else p.data = (uint8_t*) kv_alloc(stride, 1);
        }
    }
    if (!draft_) return;
    // the drafter's K/V, Q8_0 whatever --kv is: its window layers keep the last kWindow positions, the other every one
    for (int l = 0; l < kDsLayers; ++l)
        for (int j = 0; j < 4; ++j) {
            const size_t stride = j % 2 ? kKey / 32 * sizeof(uint16_t) : kKey;
            auto& p = ds_cache_[(size_t) l].planes[j];
            p = {nullptr, stride};
            p.data = ds_swa(l) ? (uint8_t*) seq_alloc((size_t) kWindow * stride, "DSpark window KV")
                               : (uint8_t*) kv_alloc(stride, 1);
        }
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context) throw std::logic_error("save_rows: outside context");
    if (!n) return;
    ck(cudaStreamSynchronize(cs(st_)), "save Inkling rows");
    for (int i = 0; i < kVisits; ++i) if (!local(i)) {
        for (const auto& p : caches_[i].planes) if (p.stride) {
            const size_t bytes = (size_t) n * p.stride;
            ck(cudaMemcpy(dst, p.data + first * p.stride, bytes, cudaMemcpyDeviceToHost), "save Inkling rows");
            dst += bytes;
        }
    }
    if (draft_)
        for (const auto& p : ds_cache_[1].planes) {
            const size_t bytes = (size_t) n * p.stride;
            ck(cudaMemcpy(dst, p.data + first * p.stride, bytes, cudaMemcpyDeviceToHost), "save DSpark rows");
            dst += bytes;
        }
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: outside saved rows");
    if (!n) return;
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load Inkling rows");
    for (int i = 0; i < kVisits; ++i) if (!local(i)) {
        for (const auto& p : caches_[i].planes) if (p.stride) {
            ck(cudaMemcpy(p.data + first * p.stride, src + (first - from) * p.stride, (size_t) n * p.stride,
                          cudaMemcpyHostToDevice), "load Inkling rows");
            src += (size_t) count * p.stride;
        }
    }
    if (draft_)
        for (const auto& p : ds_cache_[1].planes) {
            ck(cudaMemcpy(p.data + first * p.stride, src + (first - from) * p.stride, (size_t) n * p.stride,
                          cudaMemcpyHostToDevice), "load DSpark rows");
            src += (size_t) count * p.stride;
        }
}

size_t Inkling2::extra_bytes() const {
    return (kVisits - kVisits / kBlock) * (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes() +
           (draft_ ? 2 * (size_t) std::min(opt_.max_context, kWindow) * ds_row_bytes() : 0);
}

void Inkling2::ring_state(int visit, int64_t pos, const uint8_t* src, uint8_t* dst) {
    const int64_t keep = std::min(opt_.max_context, kWindow - 1);
    const int64_t first = std::max<int64_t>(0, pos - keep);
    for (const auto& p : caches_[visit].planes) if (p.stride) {
        if (dst) std::memset(dst, 0, (size_t) (keep - (pos - first)) * p.stride);
        for (int64_t i = first; i < pos;) {
            const int64_t at = i % ring_, n = std::min(pos - i, ring_ - at);
            const size_t off = (size_t) (i - (pos - keep)) * p.stride, bytes = (size_t) n * p.stride;
            if (dst) ck(cudaMemcpy(dst + off, p.data + at * p.stride, bytes, cudaMemcpyDeviceToHost), "save Inkling SWA");
            else ck(cudaMemcpy(p.data + at * p.stride, src + off, bytes, cudaMemcpyHostToDevice), "load Inkling SWA");
            i += n;
        }
        if (dst) dst += keep * p.stride;
        else src += keep * p.stride;
    }
}

// A drafter window layer's last kWindow rows before `pos`, from the ring (dst) or into it (src): a plane after another,
// the oldest row first, zeros before the sequence's first row.
void Inkling2::ds_ring(int l, int64_t pos, const uint8_t* src, uint8_t* dst) {
    const int64_t keep = std::min(opt_.max_context, kWindow);
    const int64_t first = std::max<int64_t>(0, pos - keep);
    for (const auto& p : ds_cache_[(size_t) l].planes) {
        if (dst) std::memset(dst, 0, (size_t) (keep - (pos - first)) * p.stride);
        for (int64_t i = first; i < pos;) {
            const int64_t at = i % kWindow, n = std::min(pos - i, kWindow - at);
            const size_t off = (size_t) (i - (pos - keep)) * p.stride, bytes = (size_t) n * p.stride;
            if (dst) ck(cudaMemcpy(dst + off, p.data + at * p.stride, bytes, cudaMemcpyDeviceToHost), "save DSpark");
            else ck(cudaMemcpy(p.data + at * p.stride, src + off, bytes, cudaMemcpyHostToDevice), "load DSpark");
            i += n;
        }
        if (dst) dst += keep * p.stride;
        else src += keep * p.stride;
    }
}

void Inkling2::save_card_state(uint8_t* dst) {
    static const bool pack = [] {
        const char* p = std::getenv("STRATA_INKLING_STATE_PACK");
        return p && p[0] == '1' && p[1] == '\0';
    }();
    if (pack && looped_ && try_pack_state(dst)) {
        static const bool logged = [] {
            std::fputs("strata-poly: STRATA_INKLING_STATE_PACK=1: packed checkpoints in expert stage\n", stderr);
            return true;
        }();
        (void) logged;
        return;
    }
    Engine::save_card_state(dst);
}

bool Inkling2::try_pack_state(uint8_t* dst) {
    // save_state() joined st_. Use only the complete one-card, non-DSpark layout and the expert stage's
    // currently mapped capacity; never allocate or widen it here.
    if (!dst || draft_ || !direct_stage_ || !stage_ || split_ || next_ || recording_ ||
        lo_ != 0 || hi_ != n_layer_ || n_layer_ != kPhysical || layers_.size() != kPhysical ||
        state_n_ != rec_n_ || rec_n_ != (size_t) kVisits * kConvFloats || !state_ ||
        ring_ <= 0 || pos_ < 0 || pos_ > opt_.max_context)
        return false;
    const size_t state_bytes = state_n_ * sizeof(float), bytes = card_state_bytes();
    if (stage_bytes_ < bytes) return false;
    const int64_t keep = std::min(opt_.max_context, kWindow - 1);
    const int64_t first = std::max<int64_t>(0, pos_ - keep);
    const size_t one = (size_t) keep * kv_bytes();
    if (bytes != state_bytes + (kVisits - kVisits / kBlock) * one ||
        ring_ != std::min(opt_.max_context, kWindow + rows_ - 1))
        return false;
    for (int i = 0; i < kVisits; ++i) if (local(i))
        for (int j = 0; j < 4; ++j) {
            const auto& p = caches_[i].planes[j];
            const size_t stride = j % 2 ? (q8() ? kKey / 32 * sizeof(uint16_t) : 0)
                                       : kKey * (q8() ? 1 : sizeof(uint16_t));
            if (p.stride != stride || (stride && !p.data)) return false;
        }

    // Fault this checkpoint in parallel before the bulk D2H.
    if (workers_ && workers_->threads() > 1) {
        constexpr size_t kTouchJob = size_t(2) << 20;
        const int jobs = (int) (bytes / kTouchJob + (bytes % kTouchJob != 0));
        const auto touch_pages = [dst, bytes](int job) {
            constexpr size_t kJob = size_t(2) << 20, kStep = 4096;
            const size_t first = (size_t) job * kJob, end = std::min(bytes, first + kJob);
            volatile uint8_t* const out = dst;
            for (size_t at = first; at < end; at += kStep) out[at] = 0;
            if (first < end) out[end - 1] = 0;
        };
        workers_->run(jobs, std::ref(touch_pages));
    }

    // stage_ no longer holds weights, including if an enqueue below throws. clear() retains capacity.
    staged_.clear();
    staged_il_ = -1;
    ck(cudaMemsetAsync(stage_, 0, bytes, cs(st_)), "pack Inkling state padding");
    ck(cudaMemcpyAsync(stage_, state_, state_bytes, cudaMemcpyDeviceToDevice, cs(st_)),
       "pack Inkling convolution state");
    for (int i = 0; i < kVisits; ++i) if (local(i)) {
        uint8_t* plane = stage_ + state_bytes + (i - i / kBlock) * one;
        for (const auto& p : caches_[i].planes) if (p.stride) {
            // Same source row, destination padding and oldest-first order as ring_state().
            for (int64_t row = first; row < pos_;) {
                const int64_t at = row % ring_, count = std::min(pos_ - row, ring_ - at);
                const size_t off = (size_t) (row - (pos_ - keep)) * p.stride;
                ck(cudaMemcpyAsync(plane + off, p.data + at * p.stride, (size_t) count * p.stride,
                                   cudaMemcpyDeviceToDevice, cs(st_)), "pack Inkling SWA state");
                row += count;
            }
            plane += (size_t) keep * p.stride;
        }
    }
    ck(cudaMemcpyAsync(dst, stage_, bytes, cudaMemcpyDeviceToHost, cs(st_)), "save packed Inkling state");
    ck(cudaStreamSynchronize(cs(st_)), "save packed Inkling state");
    // No caller can start its next prefetch or narrow/unmap stage_ until all copies above have completed.
    return true;
}

void Inkling2::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (draft_ && il == kPhysical - 1) {   // the drafter's rings, behind the model's
        uint8_t* at = dst + state_n_ * sizeof(float) +
                      (kVisits - kVisits / kBlock) * (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
        for (int l = 0; l < kDsLayers; ++l)
            if (ds_swa(l)) {
                ds_ring(l, pos, nullptr, at);
                at += (size_t) std::min(opt_.max_context, kWindow) * ds_row_bytes();
            }
    }
    for (int i = 0; i < kVisits; ++i) if (physical(i) == il) {
        const size_t at = (size_t) i * kConvFloats;
        ck(cudaMemcpy(dst + at * sizeof(float), state_ + at, kConvFloats * sizeof(float), cudaMemcpyDeviceToHost),
           "save Inkling convolution");
        if (local(i)) {
            const size_t offset = (i - i / kBlock) * (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
            ring_state(i, pos, nullptr, dst + state_n_ * sizeof(float) + offset);
        }
    }
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
    const size_t one = (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
    for (int i = 0; i < kVisits; ++i) if (local(i)) {
        ring_state(i, pos, src, nullptr);
        src += one;
    }
    if (draft_)
        for (int l = 0; l < kDsLayers; ++l)
            if (ds_swa(l)) {
                ds_ring(l, pos, src, nullptr);
                src += (size_t) std::min(opt_.max_context, kWindow) * ds_row_bytes();
            }
}

void Inkling2::assign_scratch() {
    const size_t n = rows_, width = n * kWidth;
    q_ = fbuf(n * kQuery);
    k_ = fbuf(n * kKey);
    v_ = fbuf(n * kKey);
    att_ = fbuf(n * kQuery);
    gate_ = fbuf(n * kQuery);
    rel_ = fbuf(n * kHeads * kRelative);
    sink_ = fbuf(n * kHeads);
    taps_ = fbuf(n * kKv * kTaps);
    halo_ = fbuf((size_t) div_up(rows_, kConvSpan) * 3 * kWidth);
    // A tile takes kValueTile scores and one kHead-wide partial value vector per query head. Keep room
    // for a single query at the configured limit, but share that room across queries on shorter prefixes.
    constexpr int target_tiles = (1 << 20) / (kHeads * kValueTile);
    attend_tiles_ = std::max(div_up(opt_.max_context, kValueTile),
                             std::min(target_tiles, rows_ * div_up(opt_.max_context, kValueTile)));
    scores_ = fbuf(inkling::plain_score_floats(attend_tiles_));
    partial_ = fbuf(inkling::plain_partial_floats(attend_tiles_));
    if (decode_) {
        chunks_ = fbuf(inkling::decode_floats());
        taps2_ = fbuf(n * kKv * kTaps);
        attq_ = fbuf((size_t) kStageRows * (kQuery / 32) * 36 / sizeof(float));   // Q8_1: 36 bytes a block of 32
        kq8_ = fbuf((size_t) (kWidth / 32) * 36 / sizeof(float));
        vq8_ = fbuf((size_t) (kWidth / 32) * 36 / sizeof(float));
        // heads(): the three streams of up to kStageRows - 1 decode rows
        const size_t w3 = 3 * (size_t) kStageRows * kWidth;
        s3cur_ = fbuf(w3);
        s3msg_ = fbuf(w3);
        s3pair_ = fbuf(2 * w3);
        s3rank_ = fbuf(3 * (size_t) kStageRows * kRank);
        s3in_ = fbuf(w3);
        s3q8_ = fbuf(3 * (size_t) kStageRows * kernels::native_q8_1_bytes(kWidth) / sizeof(float));
    }
    for (auto& x : input_) x = fbuf(width);
    for (auto& x : depth_) x = fbuf(width);
    prelude_ = fbuf(width);
    bank0_ = fbuf(width);
    bank1_ = fbuf(width);
    msg_ = fbuf(width);
    pair_ = fbuf(2 * width);
    rank_ = fbuf(n * kRank);
    gain_ = fbuf(width);
    forget_ = fbuf(width);
    if (!draft_) return;
    ds_taps_ = fbuf(3 * width);
    const size_t b = kDsBlock;
    ds_x_ = fbuf(b * kWidth);
    ds_xa_ = fbuf(b * kWidth);
    ds_q_ = fbuf(b * kQuery);
    ds_qr_ = fbuf(b * kQuery);
    ds_k_ = fbuf(b * kKey);
    ds_kr_ = fbuf(b * kKey);
    ds_v_ = fbuf(b * kKey);
    ds_ao_ = fbuf(b * kQuery);
    ds_out_ = fbuf(b * kWidth);
    ds_g_ = fbuf(b * kDsFfn);
    ds_u_ = fbuf(b * kDsFfn);
    ds_h_ = fbuf(b * kDsFfn);
    ds_cur_ = fbuf(b * kWidth);
    ds_base_ = fbuf((b - 1) * (size_t) n_vocab_);
    ds_state_ = fbuf(kDsRank);
    ds_w1row_ = fbuf(kDsRank);
    ds_jx_ = fbuf(2 * kDsRank + 2 * kWidth);
    ds_jo_ = fbuf(3 * kDsRank);
    ds_lat_ = fbuf(kDsRank);
    ds_att_ = fbuf(dspark::attend_floats());
    ds_b2_ = fbuf((size_t) n_vocab_);
    ds_w2_ = (uint16_t*) fbuf(((size_t) kDsRank * n_vocab_ + 1) / 2);
    if (!ds_host_) ds_host_ = (float*) hmalloc(b * kWidth * sizeof(float), "pinned memory");
    ds_head_ready_ = false;
}

void Inkling2::conv(float* x, float* state, const float* w, const float* dynamic, int width, int n) {
    if (n <= kConvSpan) {
        conv_kernel<<<div_up(width, kThreads), kThreads, 0, cs(st_)>>>(x, state, w, dynamic, width, n);
        return;
    }
    const dim3 grid(div_up(width, kThreads), div_up(n, kConvSpan));
    conv_halo_kernel<<<grid, kThreads, 0, cs(st_)>>>(x, state, halo_, width);
    conv_span_kernel<<<grid, kThreads, 0, cs(st_)>>>(x, state, halo_, w, dynamic, width, n);
}

void Inkling2::store(const float* x, Cache& c, int side, int n) {
    auto& values = c.planes[2 * side];
    auto& scales = c.planes[2 * side + 1];
    const bool swa = local(visit_);
    int64_t pos = swa ? pos_ % ring_ : pos_;
    for (int at = 0; at < n;) {
        const int m = swa ? std::min(n - at, ring_ - (int) pos) : n;
        if (q8()) store_q8(x, (int8_t*) values.data, (uint16_t*) scales.data, kKey, pos, m, st_);
        else store_f16(x, (uint16_t*) values.data, kKey, pos, m, st_);
        at += m;
        x += (size_t) m * kKey;
        pos = 0;
    }
}

void Inkling2::attention(const Block& b, int n) {
    Cache& c = caches_[visit_];
    const bool swa = local(visit_);
    if (fused(n) && q8()) {
        const int blocks = 2 * (kKey / 32) * n;
        if (!kv_stored_)
            store2_q8_kernel<<<div_up(blocks, 256), 256, 0, cs(st_)>>>(
                k_, v_, (int8_t*) c.planes[0].data, (__half*) c.planes[1].data, (int8_t*) c.planes[2].data,
                (__half*) c.planes[3].data, pos_, swa ? ring_ : 0, n);
        kv_stored_ = false;
    } else {
        store(k_, c, 0, n);
        store(v_, c, 1, n);
    }
    inkling::Attend a;
    a.q = q_;
    a.rel = rel_;
    a.table = fp(b.table);
    a.sink = sink_;
    a.gate = gate_;
    a.k = {c.planes[0].data, (const uint16_t*) c.planes[1].data};
    a.v = {c.planes[2].data, (const uint16_t*) c.planes[3].data};
    a.pos = pos_;
    a.n = n;
    a.window = swa ? kWindow : 0;
    a.ring = swa ? ring_ : 0;
    a.out = att_;
    // Prompt batches only: a token's answer and short calls keep the plain kernels and their sums.
    if (flash_ && n >= kStageRows) inkling::attend_flash(a, cs(st_));
    else if (decode_ && n < kStageRows) inkling::attend_decode(a, chunks_, cs(st_), attq_);
    else inkling::attend_plain(a, scores_, partial_, attend_tiles_, cs(st_));
}

void Inkling2::depth_mix(const float* query, float* dst, int first, int n) {
    static_assert(kWidth == kRowWidth);
    const size_t at = (size_t) first * kWidth;
    MixInputs m{};
    for (int i = 0; i < depth_n_; ++i) m.x[i] = depth_[i] + at;
    m.x[depth_n_] = r_ + at;
    mix_kernel<<<n, kThreads, 0, cs(st_)>>>(m, depth_n_ + 1, query, dst);
}

// Both formats use the same previous-pass bank; they store its Q, K, V and FFN query rows differently.
const float* Inkling2::message_query(const Block& b, int il, int stream) const {
    if (loop_layout_) return fp(b.message_query) + stream * kWidth;
    const size_t row = 2 * (size_t) (il - kStart) + (stream >= 2 ? 1 : 0);
    return fp(stream == 0 || stream == 3 ? loop_query_ : loop_kv_query_) + row * kWidth;
}

void Inkling2::control(const Block& b, int il, int stream, float* in, int n) {
    const float* query = message_query(b, il, stream);
    if (fused(n)) control_in_kernel<<<n, kThreads, 0, cs(st_)>>>(bank0_, bank1_, query, in, msg_, pair_);
    else {
        mix_kernel<<<n, kThreads, 0, cs(st_)>>>(MixInputs{{bank0_, bank1_}}, 2, query, msg_);
        rms_norm(in, nullptr, pair_, kWidth, n, 1.e-6f, st_, 2 * kWidth);
        rms_norm(msg_, nullptr, pair_ + kWidth, kWidth, n, 1.e-6f, st_, 2 * kWidth);
    }
    const Control& c = b.control[stream];
    if (fused(n) && c.gain.type == kTypeF32 && c.forget.type == kTypeF32 && c.gain.scale == 1.f &&
        c.forget.scale == 1.f) {
        lin(c.down, pair_, rank_, n);
        control_out_kernel<<<dim3(kWidth / 32, n), 256, 0, cs(st_)>>>(fp(c.gain), fp(c.forget), rank_, msg_, in,
                                                                      kRank);
        return;
    }
    lin(c.down, pair_, rank_, n);
    lin(c.gain, rank_, gain_, n);
    lin(c.forget, rank_, forget_, n);
    control_kernel<<<div_up(n * kWidth, kThreads), kThreads, 0, cs(st_)>>>(in, msg_, gain_, forget_, n * kWidth);
}

// Decode rows of a visit with three streams (q, k, v): each stream's mix, control and norm as control() and
// layer_head()'s loop do them, all three in each launch, into s3in_ (the streams one after the other).  Five launches
// in place of fifteen with the control, two in place of six without; every row bit for bit the per-stream launches'.
// False, and nothing launched, where the per-stream path has to run (a prompt batch, the first layer, a control that
// is not the F32 products control() fuses).
bool Inkling2::heads(const Block& b, int il, bool swa, bool loop, int n, const Dev* const* norm) {
    if (!fused(n) || il == 0 || !s3in_ || depth_n_ + 1 > 5) return false;
    if (loop)
        for (const Control& c : {b.control[0], b.control[1], b.control[2]})
            if (c.down.type != kTypeF32 || c.down.scale != 1.f || c.gain.type != kTypeF32 || c.forget.type != kTypeF32 ||
                c.gain.scale != 1.f || c.forget.scale != 1.f || c.down.n_in != 2 * kWidth || c.down.n_out != kRank)
                return false;
    const size_t w = (size_t) n * kWidth;
    MixInputs m{};
    for (int i = 0; i < depth_n_; ++i) m.x[i] = depth_[i];
    m.x[depth_n_] = r_;
    Streams3 ms{};
    for (int i = 0; i < 3; ++i) {
        ms.query[i] = fp(b.aquery) + (swa ? 0 : i * kWidth);
        ms.out[i] = s3cur_ + i * w;
    }
    mix3_kernel<<<dim3(n, 3), kThreads, 0, cs(st_)>>>(m, depth_n_ + 1, ms);
    if (loop) {
        Control3 c{};
        F32Segments down;
        for (int i = 0; i < 3; ++i) {
            c.query[i] = message_query(b, il, i);
            c.cur[i] = s3cur_ + i * w;
            c.msg[i] = s3msg_ + i * w;
            c.pair[i] = s3pair_ + 2 * i * w;
            c.gain[i] = fp(b.control[i].gain);
            c.forget[i] = fp(b.control[i].forget);
            c.rank[i] = s3rank_ + (size_t) i * n * kRank;
            down.w[i] = fp(b.control[i].down);
            down.x[i] = c.pair[i];
            down.y[i] = s3rank_ + (size_t) i * n * kRank;
        }
        control_in3_kernel<<<dim3(n, 3), kThreads, 0, cs(st_)>>>(bank0_, bank1_, c);
        // lin(c.down, pair_, rank_, n) of each stream: gemv_f32_waves() with a row of outputs to its input
        gemv_f32_waves_segments(down, 3, 2 * kWidth, kRank, kRank, n, st_);
        control_out3_kernel<<<dim3(kWidth / 32, n, 3), 256, 0, cs(st_)>>>(c, kRank);
    }
    if (norm_q8_) {   // the three stream norms and their Q8_1 in one launch (s3in_ then already Q8_1 in s3q8_)
        kernels::NormQ8Row rs[3];
        // a stream's Q8_1 rows are n of them, back to back as the flat quantize laid them out: the stream's
        // slot in s3q8_ starts n rows in (layer_head's `wq` below reads at the same stride)
        const size_t wq = (size_t) n * kernels::native_q8_1_bytes(kWidth);
        for (int i = 0; i < 3; ++i)
            rs[i] = {s3cur_ + i * w, fp(*norm[i]), s3in_ + i * w, (uint8_t*) s3q8_ + i * wq};
        kernels::native_rms_norm_q8_1_rows(rs, 3, kWidth, n, rms_eps_, sum_threads(kWidth), st_);
        return true;
    }
    NormSegments ns;
    for (int i = 0; i < 3; ++i) {
        ns.x[i] = s3cur_ + i * w;
        ns.w[i] = fp(*norm[i]);
        ns.y[i] = s3in_ + i * w;
    }
    rms_norm_segments(ns, 3, kWidth, n, rms_eps_, st_);
    return true;
}

void Inkling2::run_visits(int n) {
    // Engine::input already put the projected patches in r_. Only token embeddings take the learned norm.
    const auto norm_text = [&](int first, int count) {
        const size_t at = (size_t) first * kWidth;
        rms_norm(r_ + at, fp(token_norm_), cur_ + at, kWidth, count, rms_eps_, st_);
        axpy(cur_ + at, 1.f, r_ + at, count * kWidth, false, st_);
    };
    int first = 0;
    for (const Image& im : images_) {   // sorted, non-overlapping; a prompt chunk can cut through a picture
        if (im.at >= pos_ + n) break;
        if (im.at + im.n <= pos_) continue;
        if (im.at > pos_ + first) norm_text(first, (int) (im.at - pos_) - first);
        first = (int) std::min<int64_t>(n, im.at + im.n - pos_);
    }
    if (first < n) norm_text(first, n - first);
    fo_clean_ = false;   // fo_ is what the last call left
    // half a warp an NVFP4 expert row for this call's few tokens only (iq_set_slot_lanes() is the process's)
    kernels::iq_set_slot_lanes(fused(n));
    struct Lanes {
        ~Lanes() { kernels::iq_set_slot_lanes(false); }
    } lanes;
    depth_n_ = 0;
    for (visit_ = 0; visit_ < kVisits; ++visit_) {
        if (visit_ == kStart) axpy(r_, 1.f, prelude_, n * kWidth, false, st_);
        if (visit_ == kStart + kCore || visit_ == kStart + 2 * kCore) {
            // The old partial and the one just before it are the previous pass's two core blocks.
            axpy(depth_[2], 1.f, bank0_, n * kWidth, false, st_);
            axpy(r_, 1.f, bank1_, n * kWidth, false, st_);
            axpy(prelude_, 1.f, r_, n * kWidth, false, st_);
            depth_n_ = 1;
        }
        if (draft_)   // the drafter's taps: the residual entering visits 5, 23 and 41 (drafter_kept())
            for (int j = 0; j < 3; ++j)
                if (visit_ == kTap[j])
                    ck(cudaMemcpyAsync(ds_taps_ + (size_t) j * rows_ * kWidth, r_, (size_t) n * kWidth * sizeof(float),
                                       cudaMemcpyDeviceToDevice, cs(st_)),
                       "DSpark taps");
        layer(physical(visit_), n);
    }
}

void Inkling2::layer_head(Layer& L, int il, int n) {
    const Block& b = blocks_[il];
    // STRATA_INKLING_TAIL_MOE: this visit's cached experts wait in tail_sum_ for the tail launch, or fo_ holds
    // their add_scaled() sums as before.  One token only: the record carries no rows.  Every visit starts fresh.
    tail_sum_.n = 0;
    tail_record_ = moe_tail_ && fused(n) && L.moe && n == 1;
    const bool swa = local(visit_), loop = messages();
    const Dev* norm[3] = {&L.attn_norm, swa ? &L.attn_norm : &b.knorm, swa ? &L.attn_norm : &b.vnorm};
    const int streams = swa && !loop ? 1 : 3;
    const float *xq = input_[0], *xk = xq, *xv = xq;
    const void *kq = nullptr, *vq = nullptr;   // K's and V's own inputs as Q8_1, when heads() quantized them
    const bool mmv = n < kStageRows;
    const bool three = streams == 3 && heads(b, il, swa, loop, n, norm);
    bool q8in = three;   // the stream inputs are Q8_1 already (heads(), or the loop below when it fuses)
    if (three) {
        const size_t w = (size_t) n * kWidth, wq = (size_t) n * kernels::native_q8_1_bytes(kWidth);
        xq = s3in_;
        xk = s3in_ + w;
        xv = s3in_ + 2 * w;
        // the three inputs lie one after the other, and a Q8_1 block never straddles two: one launch for all
        // (heads() wrote their Q8_1 too when norm_q8_)
        if (!norm_q8_) kernels::native_quantize_q8_1(s3in_, s3q8_, kWidth * n * 3, 1, st_);
        kq = (const uint8_t*) s3q8_ + wq;
        vq = (const uint8_t*) s3q8_ + 2 * wq;
    } else {
        for (int i = 0; i < streams; ++i) {
            if (il == 0) axpy(r_, 1.f, cur_, n * kWidth, false, st_);
            else depth_mix(fp(b.aquery) + (swa ? 0 : i * kWidth), cur_, 0, n);
            if (loop) control(b, il, i, cur_, n);
            // this stream's norm and its Q8_1 in one launch; K and V write their own single-row buffers
            if (norm_q8_ && mmv && (streams == 1 || n == 1)) {
                void* q = streams == 1 || i == 0 ? xq_ : i == 1 ? (void*) kq8_ : (void*) vq8_;
                const kernels::NormQ8Row r{cur_, fp(*norm[i]), input_[i], q};
                kernels::native_rms_norm_q8_1_rows(&r, 1, kWidth, n, rms_eps_, sum_threads(kWidth), st_);
                q8in = true;
            } else rms_norm(cur_, fp(*norm[i]), input_[i], kWidth, n, rms_eps_, st_);
        }
        if (streams == 3) {
            xk = input_[1];
            xv = input_[2];
            if (q8in && n == 1) kq = kq8_, vq = vq8_;
        }
    }
    // MMVQ uses the same Q8_1 input for Q, relative bias and the attention gate (also K and V when
    // they share the input). The matrix-matrix prompt path uses its own layout instead.
    if (mmv && !q8in) kernels::native_quantize_q8_1(xq, xq_, kWidth * n, 1, st_);
    const void* shared = three ? s3q8_ : mmv ? xq_ : nullptr;
    if (fused(n) && n == 1) project(b, xq, xk, xv, streams == 1, shared, kq, vq);
    else {
        lin(b.q, xq, q_, n, shared);
        lin(b.k, xk, k_, n, streams == 1 ? shared : kq);
        lin(b.v, xv, v_, n, streams == 1 ? shared : vq);
        lin(b.rel, xq, rel_, n, shared);
        lin(b.gate, xq, gate_, n, shared);
    }
    lin(b.sink, xq, sink_, n);

    float* hist = state_ + (size_t) visit_ * kConvFloats;
    const auto kv_conv = [&](float* x, float* state, const Dev& w, const Dev& dyn, const float* input) {
        if (!swa) lin(dyn, input, taps_, n);
        conv(x, state, fp(w), swa ? nullptr : taps_, kKey, n);
    };
    const int floor = swa ? 0 : floor_;
    if (fused(n)) {
        if (!swa) {
            lin(b.kproj, xk, taps_, n);
            lin(b.vproj, xv, taps2_, n);
        }
        const Conv ck{k_, hist, fp(b.kconv), swa ? nullptr : taps_, snap_of(hist)};
        const Conv cv{v_, hist + 3 * kKey, fp(b.vconv), swa ? nullptr : taps2_, snap_of(hist + 3 * kKey)};
        HeadNorms h{{{q_, fp(b.qclip), kHead, kHeads, qk_clip_, floor}, {k_, nullptr, kHead, kKv, qk_clip_, 0},
                     {v_, nullptr, kHead, kKv, 0.f, 0}, {rel_, fp(b.rclip), kRelative, kHeads, r_clip_, floor}}};
        if (n == 1 && q8()) {   // attention() finds K and V stored
            Cache& c = caches_[visit_];
            const KvStore to{(int8_t*) c.planes[0].data, (int8_t*) c.planes[2].data, (__half*) c.planes[1].data,
                             (__half*) c.planes[3].data, swa ? pos_ % ring_ : pos_};
            heads_kv_kernel<<<2 * kHeads + 2 * kKv, kThreads, 0, cs(st_)>>>(ck, cv, h, to, rms_eps_, pos_, alpha_,
                                                                           rec_n_);
            kv_stored_ = true;
        } else {
            conv2_kernel<<<div_up(2 * kKey, kThreads), kThreads, 0, cs(st_)>>>(ck, cv, kKey, n, rec_n_);
            head_norms_kernel<<<dim3(2 * kHeads + 2 * kKv, n), kThreads, 0, cs(st_)>>>(h, rms_eps_, pos_, alpha_);
        }
    } else {
        kv_conv(k_, hist, b.kconv, b.kproj, xk);
        kv_conv(v_, hist + 3 * kKey, b.vconv, b.vproj, xv);
    }
    const auto heads = [&](float* x, const float* w, int width, int count, float clip, int log_floor) {
        head_norm_kernel<<<dim3(count, n), kThreads, 0, cs(st_)>>>(x, w, width, count, clip, rms_eps_, pos_, log_floor,
                                                                  alpha_);
    };
    if (!fused(n)) {
        heads(q_, fp(b.qclip), kHead, kHeads, qk_clip_, floor);
        heads(k_, nullptr, kHead, kKv, qk_clip_, 0);
        heads(v_, nullptr, kHead, kKv, 0.f, 0);
        heads(rel_, fp(b.rclip), kRelative, kHeads, r_clip_, floor);
    }
    dbg("Qcur", visit_, q_, kQuery);
    dbg("Kcur", visit_, k_, kKey);
    attention(b, n);
    lin(b.o, att_, ao_, n, fused(n) ? attq_ : nullptr);   // attend_decode() quantized it
    if (fused(n))
        conv_add_kernel<<<div_up(kWidth, kThreads), kThreads, 0, cs(st_)>>>(
            ao_, nullptr, hist + 6 * kKey, fp(b.aconv), r_, il % kBlock == 0 ? depth_[depth_n_++] : nullptr, kWidth, n,
            false, TailSum{}, snap_of(hist + 6 * kKey), rec_n_);
    else conv(ao_, hist + 6 * kKey, fp(b.aconv), nullptr, kWidth, n);
    if (fused(n)) {   // conv_add_kernel() did the rest
    } else if (il % kBlock == 0) {
        axpy(r_, 1.f, depth_[depth_n_++], n * kWidth, false, st_);
        axpy(ao_, 1.f, r_, n * kWidth, false, st_);
    } else axpy(ao_, 1.f, r_, n * kWidth, true, st_);

    depth_mix(fp(b.fquery), cur_, 0, n);
    if (loop) control(b, il, 3, cur_, n);
    kernels::native_add_rms_norm_q8_1(cur_, nullptr, fp(L.ffn_norm), xn_, xq_, kWidth, n, rms_eps_, 128, st_);
    // with STRATA_INKLING_ROUTE_DUAL and one token, the router's gemv below also writes the host h_rl_ and h_x_,
    // and moe_router() launches no copy2 for them
    if (!batched(L, n)) moe_router(L, n);
}

bool Inkling2::fused(int n) const { return decode_ && n < kStageRows; }

// A window's states after its tokens (Engine::snap_at_, a token's rec_n_ floats after the one before), at the place
// `state` has in state_; null outside a window.  The window takes every token's (snap_tokens(): a place for each).
float* Inkling2::snap_of(float* state) const {
    if (!windowed_ || !snap_bits_) return nullptr;
    return snap_at_ + (state - state_);
}

// A token's q, gate and relative terms of input_[0] (xq_, quantized), and its K and V of the same or of their own
// inputs: one launch a weight format (kernels::native_mmvq_segments(), the rows bit for bit lin()'s), lin() for what
// that does not take.
void Inkling2::project(const Block& b, const float* xq, const float* xk, const float* xv, bool shared_kv,
                       const void* q8, const void* k8, const void* v8) {
    if (!shared_kv && !k8) {
        kernels::native_quantize_q8_1(xk, kq8_, kWidth, 1, st_);
        kernels::native_quantize_q8_1(xv, vq8_, kWidth, 1, st_);
        k8 = kq8_;
        v8 = vq8_;
    }
    struct Part {
        const Dev* w;
        const float* x;
        const void* xq;
        float* y;
        bool done;
    } parts[5] = {{&b.q, xq, q8, q_, false},
                  {&b.gate, xq, q8, gate_, false},
                  {&b.rel, xq, q8, rel_, false},
                  {&b.k, xk, shared_kv ? q8 : k8, k_, false},
                  {&b.v, xv, shared_kv ? q8 : v8, v_, false}};
    const auto fits = [](const Dev& w) { return w.n_in == kWidth && w.scale == 1.f; };
    for (int i = 0; i < 5; ++i) {
        if (parts[i].done || !fits(*parts[i].w)) continue;
        kernels::MmvqSegments s;
        const uint32_t type = parts[i].w->type;
        for (int j = i; j < 5; ++j)
            if (!parts[j].done && parts[j].w->type == type && fits(*parts[j].w)) {
                s.w[s.count] = parts[j].w->p;
                s.x[s.count] = parts[j].xq;
                s.y[s.count] = parts[j].y;
                s.rows[s.count++] = parts[j].w->n_out;
            }
        if (!kernels::native_mmvq_segments((int) type, s, kWidth, st_)) continue;
        for (int j = i; j < 5; ++j)
            if (parts[j].w->type == type && fits(*parts[j].w)) parts[j].done = true;
    }
    for (const Part& p : parts)
        if (!p.done) lin(*p.w, p.x, p.y, 1, p.xq);
}

void Inkling2::moe_add(int n) {
    if (fused(n)) cpu_sum_ = true;   // layer_tail() adds it
    else Engine::moe_add(n);
}

void Inkling2::layer_tail(Layer&, int il, int n) {
    float* hist = state_ + (size_t) visit_ * kConvFloats + 6 * kKey + 3 * kWidth;
    if (fused(n)) {
        // with the record's experts summed here (STRATA_INKLING_TAIL_MOE), moe() launched nothing for them
        conv_add_kernel<<<div_up(kWidth, kThreads), kThreads, 0, cs(st_)>>>(
            fo_, cpu_sum_ ? h_cpu_ : nullptr, hist, fp(blocks_[il].fconv), r_, nullptr, kWidth, n, true,
            tail_record_ ? tail_sum_ : TailSum{}, snap_of(hist), rec_n_);
        cpu_sum_ = false;
        fo_clean_ = true;
        tail_record_ = false;   // one record's life: layer_head() fills the next visit's
        dbg("l_out", visit_, r_, kWidth);
        return;
    }
    conv(fo_, hist, fp(blocks_[il].fconv), nullptr, kWidth, n);
    axpy(fo_, 1.f, r_, n * kWidth, true, st_);
    dbg("l_out", visit_, r_, kWidth);
}

void Inkling2::route(int il, const float* logits, int32_t, int* ids, float* w, float* tmp) const {
    float* pick = tmp + n_expert_;
    for (int i = 0; i < n_expert_; ++i) {
        tmp[i] = 1.f / (1.f + std::exp(-logits[i]));
        pick[i] = tmp[i] + blocks_[il].bias[i];
    }
    float sum = 0.f;
    for (int i = 0; i < n_used_; ++i) {
        const int id = (int) (std::max_element(pick, pick + n_expert_) - pick);
        pick[id] = -INFINITY;
        ids[i] = id;
        sum += (w[i] = tmp[id]);
    }
    for (int i = 0; i < n_used_; ++i) w[i] *= expert_scale_ / (sum + 1.e-20f);
}

void Inkling2::moe_shared(const Layer&, int, int n) {
    // the previous layer's fused tail left fo_ 0 for the same rows
    if (!(fused(n) && fo_clean_)) zero_vram(fo_, (size_t) n * kWidth * sizeof(float));
    fo_clean_ = false;
}

// The kept tokens' rows into the drafter's K/V, at pos() on: each layer's mix of the four taps, normed, through its
// context projections (keys normed and turned, as graph_inkling2's injection does), as Q8_0; then the recurrent
// head's mix of the last token's taps into hn_'s first row.  The call's scratch (input_, q_, k_, v_) is free by now.
void Inkling2::drafter_kept(int n) {
    if (n <= 0) return;
    const size_t plane = (size_t) rows_ * kWidth;
    dspark::Taps t{{ds_taps_, ds_taps_ + plane, ds_taps_ + 2 * plane, hn_ + n_embd_}, {}};
    float* x = input_[0];
    for (int l = 0; l < kDsLayers; ++l) {
        const DsLayer& L = ds_[(size_t) l];
        std::copy(ds_mix_[l], ds_mix_[l] + 4, t.w);
        dspark::mix_taps(t, fp(L.ctx_norm), x, 0, n, ds_eps_, cs(st_));
        lin(L.k_ctx, x, k_, n);
        lin(L.v_ctx, x, v_, n);
        head_norm_rope(k_, kHead, fp(L.k_ctx_norm), q_, kHead, kKv, kHead, ds_theta_, pos_, n, ds_eps_, st_);
        Cache& c = ds_cache_[(size_t) l];
        const bool ring = ds_swa(l);
        int64_t at = ring ? pos_ % kWindow : pos_;
        for (int done = 0; done < n;) {
            const int m = ring ? (int) std::min<int64_t>(n - done, kWindow - at) : n - done;
            store_q8(q_ + (size_t) done * kKey, (int8_t*) c.planes[0].data, (uint16_t*) c.planes[1].data, kKey, at, m,
                     st_);
            store_q8(v_ + (size_t) done * kKey, (int8_t*) c.planes[2].data, (uint16_t*) c.planes[3].data, kKey, at, m,
                     st_);
            done += m;
            at = ring ? 0 : at + m;
        }
    }
    std::copy(ds_mix_[kDsLayers], ds_mix_[kDsLayers] + 4, t.w);
    dspark::mix_taps(t, fp(ds_rnn_norm_), hn_, n - 1, 1, ds_eps_, cs(st_));
}

// W' = W_head W_hidden_out and b' = W_head b_hidden_out, so that a step scores the vocabulary from its 256 latents
// and not from a 2304-wide correction (ninfer-3090 does the same).  Once, at the first guess: the head's rows a few
// columns of hidden_out at a time, through wlogits_.
void Inkling2::prepare_head() {
    const int rows = std::min(kDsBlock, opt_.window);
    for (int j0 = 0; j0 < kDsRank; j0 += rows) {
        const int m = std::min(rows, kDsRank - j0);
        for (int r = 0; r < m; ++r)
            for (int i = 0; i < kWidth; ++i)
                ds_host_[(size_t) r * kWidth + i] = ds_hidden_[(size_t) i * kDsRank + (size_t) (j0 + r)];
        ck(cudaMemcpyAsync(ds_xa_, ds_host_, (size_t) m * kWidth * sizeof(float), cudaMemcpyHostToDevice, cs(st_)),
           "DSpark head");
        lin(output_, ds_xa_, wlogits_, m);
        dspark::to_half(wlogits_, ds_w2_ + (size_t) j0 * n_vocab_, n_vocab_, m, n_vocab_, cs(st_));
        ck(cudaStreamSynchronize(cs(st_)), "DSpark head");
    }
    std::copy(ds_hidden_b_.begin(), ds_hidden_b_.end(), ds_host_);
    ck(cudaMemcpyAsync(ds_xa_, ds_host_, kWidth * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "DSpark head");
    lin(output_, ds_xa_, ds_b2_, 1);
    ck(cudaStreamSynchronize(cs(st_)), "DSpark head");
    ds_head_ready_ = true;
}

// The block at the anchor `token`, the token at pos() that is picked and not fed: its embedding as the model takes it
// and seven masks, through the three layers (each query sees the drafter's rows before pos() and the block), the
// output norm, and the head's logits of the rows that guess.  The recurrent state starts at zero.
void Inkling2::ds_block(int32_t token) {
    const int64_t a = pos_;
    input(token, a, ds_host_);
    ck(cudaMemcpyAsync(ds_out_, ds_host_, kWidth * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "DSpark");
    rms_norm(ds_out_, fp(token_norm_), ds_x_, kWidth, 1, rms_eps_, st_);
    dspark::fill_rows(fp(ds_mask_), ds_x_ + kWidth, kWidth, kDsBlock - 1, cs(st_));
    for (int l = 0; l < kDsLayers; ++l) {
        const DsLayer& L = ds_[(size_t) l];
        rms_norm(ds_x_, fp(L.attn_norm), ds_xa_, kWidth, kDsBlock, ds_eps_, st_);
        lin(L.q, ds_xa_, ds_q_, kDsBlock);
        lin(L.k, ds_xa_, ds_k_, kDsBlock);
        lin(L.v, ds_xa_, ds_v_, kDsBlock);
        head_norm_rope(ds_q_, kHead, fp(L.q_norm), ds_qr_, kHead, kHeads, kHead, ds_theta_, a, kDsBlock, ds_eps_, st_);
        head_norm_rope(ds_k_, kHead, fp(L.k_norm), ds_kr_, kHead, kKv, kHead, ds_theta_, a, kDsBlock, ds_eps_, st_);
        const Cache& c = ds_cache_[(size_t) l];
        const bool ring = ds_swa(l);
        dspark::attend(ds_qr_, ds_kr_, ds_v_, {(int8_t*) c.planes[0].data, (uint16_t*) c.planes[1].data},
                       {(int8_t*) c.planes[2].data, (uint16_t*) c.planes[3].data},
                       ring ? std::max<int64_t>(0, a - kWindow) : 0, a, ring ? kWindow : 0, ds_att_, ds_ao_, cs(st_));
        lin(L.o, ds_ao_, ds_out_, kDsBlock);
        axpy(ds_out_, 1.f, ds_x_, kDsBlock * kWidth, true, st_);
        rms_norm(ds_x_, fp(L.ffn_norm), ds_xa_, kWidth, kDsBlock, ds_eps_, st_);
        lin(L.gate, ds_xa_, ds_g_, kDsBlock);
        lin(L.up, ds_xa_, ds_u_, kDsBlock);
        swiglu_clamp(ds_g_, ds_u_, ds_h_, kDsBlock * kDsFfn, INFINITY, st_);
        lin(L.down, ds_h_, ds_out_, kDsBlock);
        axpy(ds_out_, 1.f, ds_x_, kDsBlock * kWidth, true, st_);
    }
    rms_norm(ds_x_, fp(ds_out_norm_), ds_cur_, kWidth, kDsBlock, ds_eps_, st_);
    ck(cudaMemsetAsync(ds_state_, 0, kDsRank * sizeof(float), cs(st_)), "DSpark");
    ds_based_ = 0;
}

// The head's logits of the block's rows 1 .. to (before the recurrent bias): the first three at once (a round
// mostly asks that many, --draft-max), the rest together when a round asks for a fourth.
void Inkling2::ds_base(int to) {
    if (to <= ds_based_) return;
    const int first = ds_based_, n = (first ? ds_rows_ : std::max(to, std::min(ds_rows_, 3))) - first;
    lin(output_, ds_cur_ + (size_t) (first + 1) * kWidth, ds_base_ + (size_t) first * n_vocab_, n);
    ds_based_ = first + n;
}

// Guess i + 1 of the block (Engine::draft()): the recurrent head's step from the token before it, the anchor for the
// first guess and guess i after it; its logits into logits_.
void Inkling2::draft_step(int32_t token, int i) {
    if (!ds_head_ready_) prepare_head();
    if (i < 0 || i >= ds_rows_) throw std::logic_error("draft: the DSpark block has no more guesses");
    if (i == 0) ds_block(token);
    float* row = ds_host_ + kWidth;   // past the anchor's embedding, which may still be on its way
    const uint16_t* w1 = (const uint16_t*) ds_w1_ + (size_t) token * kDsRank;
    for (int j = 0; j < kDsRank; ++j) row[j] = ggml_fp16_to_fp32(w1[j]);
    ck(cudaMemcpyAsync(ds_w1row_, row, kDsRank * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "DSpark");
    dspark::joint(ds_state_, ds_w1row_, ds_cur_ + (size_t) (i + 1) * kWidth, hn_, ds_jx_, cs(st_));
    lin(ds_joint_, ds_jx_, ds_jo_, 1);
    dspark::recur(ds_jo_, fp(ds_joint_b_), ds_state_, ds_lat_, cs(st_));
    ds_base(i + 1);
    dspark::head(ds_base_ + (size_t) i * n_vocab_, ds_b2_, ds_w2_, ds_lat_, logits_, n_vocab_, ds_cap_, cs(st_));
}

void Inkling2::output_rows(int first, int rows, float* h) {
    depth_mix(fp(output_query_), cur_, first, rows);
    rms_norm(cur_, fp(out_norm_), h, kWidth, rows, rms_eps_, st_);
}

void Inkling2::finish_logits(float* logits, int rows) { softcap(logits, (size_t) rows * n_vocab_, cap_, st_); }

}  // namespace strata::poly
