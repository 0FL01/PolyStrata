#pragma once

#include "kernels.cuh"

namespace dflash2 {

constexpr int exact_gqa = q_heads / kv_heads, exact_tile = 128;
constexpr int exact_score_stride = ((window - 1 + block + exact_tile - 1) / exact_tile) * exact_tile;
static_assert(exact_gqa == 4 && head_dim == 128);

// Share each K load across the four Q heads and distribute contiguous key tiles across workgroups.
// Each dot uses the stock lane components, FMA sequence, and wave reduction.
__global__ __launch_bounds__(128) void attend_exact_scores(const float* q, const float* ctx_k,
                                                          const float* noise_k, float* scores, int start) {
    constexpr int waves = 4;
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int first_q = blockIdx.x * exact_gqa;
    const int query = blockIdx.x / kv_heads, kv_head = blockIdx.x % kv_heads;
    const int tile = blockIdx.y * exact_tile;
    const int lo = max(0, start + query - (window - 1));
    const int keys = max(0, min(exact_tile, start + block - lo - tile));
    if (keys == 0) return;
    q += first_q * head_dim;
    scores += first_q * exact_score_stride + tile;

    float qv[exact_gqa][head_dim / 32];
    #pragma unroll
    for (int h = 0; h < exact_gqa; ++h)
        for (int c = 0; c < head_dim / 32; ++c) qv[h][c] = q[h * head_dim + c * 32 + lane];
    for (int key = wave; key < keys; key += waves) {
        const float* k = kv_at(ctx_k, noise_k, lo + tile + key, start, kv_head);
        float val[exact_gqa]{};
        #pragma unroll
        for (int c = 0; c < head_dim / 32; ++c) {
            const float kc = k[c * 32 + lane];
            #pragma unroll
            for (int h = 0; h < exact_gqa; ++h) val[h] = fmaf(qv[h][c], kc, val[h]);
        }
        #pragma unroll
        for (int h = 0; h < exact_gqa; ++h) {
            const float score = wave_sum(val[h]) * 0.0883883476483f;
            if (lane == 0) scores[h * exact_score_stride + key] = score;
        }
    }
}

// Reproduce the stock 128-thread softmax. Changing the order of even small FP32 sums can
// change downstream Q8_1 blocks and greedy IDs. Only reassign V coordinates within each
// wave: four consecutive elements per lane permit aligned 16-byte loads.
__global__ __launch_bounds__(128) void attend_exact_values(const float* scores_global, const float* ctx_v,
                                                          const float* noise_v, float* out, int start) {
    constexpr int waves = 4, width = head_dim / 32;
    __shared__ float scores[window - 1 + block];
    __shared__ float partial[waves][head_dim], inv_denom;
    __shared__ float reduce[waves], common;
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const int query = blockIdx.x / q_heads, kv_head = blockIdx.x % q_heads / exact_gqa;
    const int lo = max(0, start + query - (window - 1));
    const int keys = start + block - lo;
    scores_global += blockIdx.x * exact_score_stride;

    float largest = -INFINITY;
    for (int key = tid; key < keys; key += 128) {
        const float score = scores_global[key];
        scores[key] = score;
        largest = fmaxf(largest, score);
    }
    for (int d = 16; d; d >>= 1) largest = fmaxf(largest, __shfl_down(largest, d, 32));
    if (lane == 0) reduce[wave] = largest;
    __syncthreads();
    if (wave == 0) {
        largest = lane < waves ? reduce[lane] : -INFINITY;
        for (int d = 16; d; d >>= 1) largest = fmaxf(largest, __shfl_down(largest, d, 32));
        if (lane == 0) common = largest;
    }
    __syncthreads();

    float denom = 0;
    for (int key = tid; key < keys; key += 128) {
        const float weight = expf(scores[key] - common);
        scores[key] = weight;
        denom += weight;
    }
    denom = wave_sum(denom);
    if (lane == 0) reduce[wave] = denom;
    __syncthreads();
    if (wave == 0) {
        denom = wave_sum(lane < waves ? reduce[lane] : 0.f);
        if (lane == 0) inv_denom = 1.f / denom;
    }

    float acc[width]{};
    for (int key = wave; key < keys; key += waves) {
        const float* v = kv_at(ctx_v, noise_v, lo + key, start, kv_head);
        const float4 v4 = reinterpret_cast<const float4*>(v)[lane];
        const float weight = scores[key];
        acc[0] = fmaf(weight, v4.x, acc[0]);
        acc[1] = fmaf(weight, v4.y, acc[1]);
        acc[2] = fmaf(weight, v4.z, acc[2]);
        acc[3] = fmaf(weight, v4.w, acc[3]);
    }
    for (int c = 0; c < width; ++c) partial[wave][lane * width + c] = acc[c];
    __syncthreads();
    float sum = 0;
    for (int w = 0; w < waves; ++w) sum += partial[w][tid];
    out[blockIdx.x * head_dim + tid] = sum * inv_denom;
}

inline void attend_exact(const float* q, const float* ctx_k, const float* ctx_v, const float* noise_k,
                         const float* noise_v, float* scores, float* output, int start, hipStream_t stream) {
    // Retain the original path for short windows, which do not amortize the second launch.
    if (start < 512) {
        attend<false><<<block * q_heads, head_dim, 0, stream>>>(q, ctx_k, ctx_v, noise_k, noise_v, output, start, 0);
        return;
    }
    const int keys = std::min(start, window - 1) + block;
    const int parts = (keys + exact_tile - 1) / exact_tile;
    attend_exact_scores<<<dim3(block * kv_heads, parts), head_dim, 0, stream>>>(q, ctx_k, noise_k, scores, start);
    attend_exact_values<<<block * q_heads, head_dim, 0, stream>>>(scores, ctx_v, noise_v, output, start);
}

} // namespace dflash2
