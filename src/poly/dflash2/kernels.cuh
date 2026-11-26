#pragma once

#include "runtime.hpp"

namespace dflash2 {

inline constexpr int layers = 5, head_dim = 128, q_heads = 32, kv_heads = 8;
inline constexpr int q_width = q_heads * head_dim, kv_width = kv_heads * head_dim;
inline constexpr int ffn = 6912, group_size = 16, groups = hidden / group_size, conv_width = 4 * groups;
inline constexpr int tile = 1024, tiles = (vocab + tile - 1) / tile;
inline constexpr float eps = 1.e-6f;

__device__ inline float wave_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down(x, d, 32);
    return x;
}

template<bool Max = false>
__device__ float block_reduce(float x) {
    __shared__ float waves[8]; // All callers have 128 or 256 threads.
    __shared__ float result;
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    for (int d = 16; d; d >>= 1) {
        const float other = __shfl_down(x, d, 32);
        x = Max ? fmaxf(x, other) : x + other;
    }
    if (lane == 0) waves[wave] = x;
    __syncthreads();
    if (wave == 0) {
        x = lane < blockDim.x / 32 ? waves[lane] : (Max ? -INFINITY : 0.f);
        for (int d = 16; d; d >>= 1) {
            const float other = __shfl_down(x, d, 32);
            x = Max ? fmaxf(x, other) : x + other;
        }
        if (lane == 0) result = x;
    }
    __syncthreads();
    x = result;
    __syncthreads();
    return x;
}

__global__ void gather_taps(const float* t5, const float* t23, const float* t41, const float* t48,
                            float* output, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 4 * hidden) return;
    const int tap = (i / hidden) % 4;
    const float* src = tap == 0 ? t5 : tap == 1 ? t23 : tap == 2 ? t41 : t48;
    output[i] = src[i / (4 * hidden) * hidden + i % hidden];
}

__global__ void rms_norm(const float* x, const float* w, float* y, int width) {
    x += blockIdx.x * width;
    y += blockIdx.x * width;
    float sum = 0;
    for (int c = threadIdx.x; c < width; c += blockDim.x) sum = fmaf(x[c], x[c], sum);
    const float inv = rsqrtf(block_reduce(sum) / width + eps);
    for (int c = threadIdx.x; c < width; c += blockDim.x) y[c] = (x[c] * inv) * w[c];
}

// One block per head, one thread per head coordinate. Ring writes are only used for committed K.
__global__ void norm_rope(const float* x, const float* w, const double* frequencies, float* y,
                          int heads, int start, bool ring) {
    __shared__ float normed[head_dim];
    const int c = threadIdx.x, half = head_dim / 2, row = blockIdx.x / heads;
    const int position = start + row;
    const float value = x[blockIdx.x * head_dim + c];
    const float inv = rsqrtf(block_reduce(value * value) / head_dim + eps);
    normed[c] = (value * inv) * w[c];
    __syncthreads();
    if (c < half) {
        double s, cs;
        sincos(position * frequencies[c], &s, &cs);
        const float first = normed[c], second = normed[c + half];
        const int dst_row = ring ? position % window : row;
        y += (dst_row * heads + blockIdx.x % heads) * head_dim;
        y[c] = first * float(cs) - second * float(s);
        y[c + half] = second * float(cs) + first * float(s);
    }
}

__global__ void copy_v_ring(const float* src, float* ring, int rows, int start) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * kv_width) ring[(start + i / kv_width) % window * kv_width + i % kv_width] = src[i];
}

// Side 0 prepares the normalized input. Side 1 uses the SAME delta, before adding the original residual.
__global__ void conv(const float* u, const float* delta, const float* base, float* y, const float* residual,
                     int side) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= block * hidden) return;
    const int row = i / hidden, c = i % hidden;
    const int di = row * conv_width + side * 2 * groups + c / group_size;
    const int bi = side * 2 * hidden + c;
    float v = (base[bi] + delta[di]) * u[i];
    if (row != 0) v += (base[bi + hidden] + delta[di + groups]) * u[i - hidden];
    if (residual) v += residual[i];
    y[i] = v;
}

__global__ void swiglu(float* gate, const float* up) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < block * ffn) gate[i] = (gate[i] / (1.f + expf(-gate[i]))) * up[i];
}

__device__ inline const float* kv_at(const float* ring, const float* noise, int pos, int start, int head) {
    return pos < start ? ring + ((pos % window) * kv_heads + head) * head_dim
                       : noise + ((pos - start) * kv_heads + head) * head_dim;
}

// Four waves split the visible keys of each query/head. Keep scores in shared memory, then do a stable softmax
// and weighted V sum. Each query uses its own lower SWA bound and all eight ephemeral block keys.
template<bool clamp_history>
__global__ void attend(const float* q, const float* ctx_k, const float* ctx_v, const float* noise_k,
                       const float* noise_v, float* output, int start, int first_valid) {
    constexpr int waves = 4;
    __shared__ float scores[window - 1 + block];
    __shared__ float partial[waves][head_dim];
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int query = blockIdx.x / q_heads, q_head = blockIdx.x % q_heads;
    const int kv_head = q_head / (q_heads / kv_heads);
    const int lo = max(clamp_history ? first_valid : 0, start + query - (window - 1));
    const int keys = start + block - lo;
    q += blockIdx.x * head_dim;
    float qv[head_dim / 32];
    for (int c = 0; c < head_dim / 32; ++c) qv[c] = q[c * 32 + lane];
    for (int key = wave; key < keys; key += waves) {
        const float* k = kv_at(ctx_k, noise_k, lo + key, start, kv_head);
        float val = 0;
        for (int c = 0; c < head_dim / 32; ++c) val = fmaf(qv[c], k[c * 32 + lane], val);
        val = wave_sum(val) * 0.0883883476483f; // 1/sqrt(128)
        if (lane == 0) scores[key] = val;
    }
    __syncthreads();
    float largest = -INFINITY;
    for (int key = threadIdx.x; key < keys; key += blockDim.x) largest = fmaxf(largest, scores[key]);
    largest = block_reduce<true>(largest);
    float denom = 0;
    for (int key = threadIdx.x; key < keys; key += blockDim.x) {
        const float weight = expf(scores[key] - largest);
        scores[key] = weight;
        denom += weight;
    }
    const float inv_denom = 1.f / block_reduce(denom);
    float acc[head_dim / 32]{};
    for (int key = wave; key < keys; key += waves) {
        const float* v = kv_at(ctx_v, noise_v, lo + key, start, kv_head);
        const float weight = scores[key];
        for (int c = 0; c < head_dim / 32; ++c) acc[c] = fmaf(weight, v[c * 32 + lane], acc[c]);
    }
    for (int c = 0; c < head_dim / 32; ++c) partial[wave][c * 32 + lane] = acc[c];
    __syncthreads();
    float sum = 0;
    for (int w = 0; w < waves; ++w) sum += partial[w][threadIdx.x];
    output[blockIdx.x * head_dim + threadIdx.x] = sum * inv_denom;
}

__device__ inline bool better(Candidate a, Candidate b) {
    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
}

__device__ Candidate top_one(Candidate x) {
    __shared__ Candidate waves[8], result;
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    for (int d = 16; d; d >>= 1) {
        const Candidate next{__shfl_down(x.id, d, 32), __shfl_down(x.logit, d, 32)};
        if (better(next, x)) x = next;
    }
    if (lane == 0) waves[wave] = x;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < 8; ++i) if (better(waves[i], x)) x = waves[i];
        result = x;
    }
    __syncthreads();
    x = result;
    __syncthreads();
    return x;
}

__global__ void topk_tiles(const float* logits, Candidate* candidates) {
    constexpr int per_thread = tile / 256;
    const int offset = blockIdx.x * tile + threadIdx.x;
    logits += blockIdx.y * vocab;
    candidates += (blockIdx.y * tiles + blockIdx.x) * top_k;
    Candidate values[per_thread];
    for (int i = 0; i < per_thread; ++i) {
        const int id = offset + i * 256;
        const float v = id < vocab ? logits[id] : -INFINITY;
        values[i] = isfinite(v) ? Candidate{id, v} : Candidate{vocab, -INFINITY};
    }
    for (int k = 0; k < top_k; ++k) {
        Candidate best{vocab, -INFINITY};
        for (int i = 0; i < per_thread; ++i) if (better(values[i], best)) best = values[i];
        best = top_one(best);
        if (threadIdx.x == 0) candidates[k] = best;
        for (int i = 0; i < per_thread; ++i)
            if (values[i].id == best.id) values[i] = {vocab, -INFINITY};
    }
}

__global__ void topk_merge(const Candidate* tiles_in, Candidate* out) {
    tiles_in += blockIdx.x * tiles * top_k;
    out += blockIdx.x * top_k;
    Candidate ceiling{-1, INFINITY};
    for (int k = 0; k < top_k; ++k) {
        Candidate best{vocab, -INFINITY};
        for (int i = threadIdx.x; i < tiles * top_k; i += blockDim.x) {
            const Candidate c = tiles_in[i];
            if (better(ceiling, c) && better(c, best)) best = c;
        }
        ceiling = top_one(best);
        if (threadIdx.x == 0) out[k] = ceiling;
    }
}

} // namespace dflash2
