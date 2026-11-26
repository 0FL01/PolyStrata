// SPDX-License-Identifier: MIT
// Grouped NVFP4 down MMQ, adapted from llama.cpp's ggml-cuda/mmq.cuh.
// Copyright (c) 2023-2026 The ggml authors. MIT; see third_party/llama.cpp/LICENSE.
// Keep its K256 loader and FP32 dot order; omit the final zero-scale Q8_1 half.
#include "mmq.cuh"

namespace strata::poly {
namespace {

constexpr ggml_type kType = GGML_TYPE_NVFP4;
constexpr int kI = 128, kJ = 64, kThreads = 256, kCols = 1152, kRows = 2304;
constexpr bool kFallback = false;
constexpr int kYWords = sizeof(block_q8_1_mmq) / sizeof(int);
constexpr int kPaddedYBlocks = ((kCols + 511) / 512 * 512) / QK8_1_MMQ;
constexpr auto kConfig = ggml_cuda_mmq_get_config_rdna2(kType, kJ, kFallback);
constexpr auto kTileX = mmq_get_dp4a_tile_x_sizes(kType, kI);
constexpr int kShared = kJ * sizeof(int) + (kTileX.qs + kTileX.dm + kTileX.sc) * sizeof(int) +
                        GGML_PAD(kJ * sizeof(block_q8_1_mmq), kThreads * sizeof(int));

static_assert(kConfig.type == kType && kConfig.I == kI && kConfig.J == kJ && kConfig.nthreads == kThreads);
static_assert(kConfig.K_vram == 256 && !kConfig.stream_k && !kConfig.fallback);
static_assert(kConfig.sram_layout == GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4);
static_assert(QK_NVFP4 == 64 && QK8_1_MMQ == 128 && MMQ_ITER_K == 256 && MMQ_TILE_NE_K == 32);
static_assert(sizeof(block_q8_1_mmq) == MMQ_TILE_Y_K * sizeof(int));
static_assert(kJ * MMQ_TILE_Y_K % kThreads == 0 && kRows % kI == 0 && kCols % QK8_1_MMQ == 0);
static_assert(kTileX.qs == 8320 && kTileX.dm == 2112 && kTileX.sc == 0 && kShared == 51200);

__launch_bounds__(kThreads, kConfig.occupancy)
static __global__ void nvfp4_tail_kernel(
        const char* __restrict__ x, const int* __restrict__ y, const int32_t* __restrict__ ids_dst,
        const int32_t* __restrict__ bounds, float* __restrict__ dst, const int nblocks_x,
        const int stride_row_x, const int ncols_y, const int stride_col_dst, const int stride_channel_x) {
#if defined(RDNA2)
    static_assert(ggml_cuda_get_physical_warp_size() == 32);
    static_assert(ggml_cuda_mmq_get_I(kType, kJ, kFallback) == kI);
    static_assert(ggml_cuda_mmq_get_nthreads(kType, kJ, kFallback) == kThreads);
    static_assert(ggml_cuda_mmq_get_K_vram(kType, kJ, kFallback) == 256);
    static_assert(!ggml_cuda_mmq_get_config(kType, kJ, kFallback).use_mma_data_layout());

    constexpr int warp_size = 32, nwarps = kThreads / warp_size;
    constexpr int qk = QK_NVFP4, ne_block = QK8_1_MMQ;
    constexpr int blocks_per_iter = kConfig.K_vram / qk;
    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
    constexpr auto load_tiles = ggml_cuda_mmq_get_load_tiles<kType, kJ, kFallback>();
    constexpr auto vec_dot = ggml_cuda_mmq_get_vec_dot<kType, kJ, kFallback>();
    constexpr auto write_back = ggml_cuda_mmq_get_write_back<kType, kJ, kFallback>();

    const int expert = blockIdx.z, jt = blockIdx.y, it = blockIdx.x;
    const int col_low = bounds[expert], col_diff = bounds[expert + 1] - col_low;
    if (jt * kJ >= col_diff) return;

    extern __shared__ int data_nvfp4_tail[];
    int* ids = data_nvfp4_tail;
    int* tile_y = ids + kJ;
    int* tile_x = tile_y + GGML_PAD(kJ * MMQ_TILE_Y_K, kThreads);
    const int lane = threadIdx.y * warp_size + threadIdx.x;
    if (lane < kJ) ids[lane] = ids_dst[col_low + jt * kJ + lane];
    __syncthreads();

    const int offset_x = expert * stride_channel_x + it * kI * stride_row_x;
    y += (col_low + jt * kJ) * sz;
    float sum[kJ * kI / (nwarps * warp_size)] = {0.0f};

    for (int kb0 = 0; kb0 < nblocks_x; kb0 += blocks_per_iter) {
        // The weight tile still reads K256; retain the original readable finite tail.
        load_tiles(x, tile_x, offset_x + kb0, kI - 1, stride_row_x);
        {
            const int* by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < kJ * MMQ_TILE_Y_K; l0 += kThreads) {
                const int l = l0 + lane;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot(tile_x, tile_y, sum, 0);
        __syncthreads();

        // Uniform: the last down half is wholly padded and has zero D4 scales.
        if (kb0 * qk + ne_block < nblocks_x * qk) {
            const int* by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < kJ * MMQ_TILE_Y_K; l0 += kThreads) {
                const int l = l0 + lane;
                tile_y[l] = by0[l];
            }
            __syncthreads();
            vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    write_back(sum, ids, dst + it * kI, nullptr, stride_col_dst, kI - 1, col_diff - jt * kJ - 1);
#endif
}

bool selects_j64(const mmq_args& args, int cc, size_t smpbo) {
    // Same increasing-J loop and strict tie policy as ggml's mul_mat_q_switch_J.
    int best_j = 0, best_tiles = INT_MAX;
    for (int j = 8; j <= 128 && best_tiles > 1; j += 8) {
        const auto c = ggml_cuda_mmq_get_config(kType, j, kFallback, cc);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
        const int tiles = (args.ncols_opt + c.J - 1) / c.J;
        if (tiles < best_tiles) {
            best_j = j;
            best_tiles = tiles;
        }
    }
    return best_j == kJ;
}

}  // namespace

bool nvfp4_tail_try(const mmq_args& a, cudaStream_t stream) {
    if (a.type_x != kType || a.nrows_x != kRows || a.ncols_x != kCols || a.y_scale ||
        !a.x || !a.y || !a.dst || !a.ids_dst || !a.expert_bounds || a.ncols_max < 128 ||
        a.ncols_max > a.ncols_y || a.ncols_opt < 1 || a.ncols_opt > INT_MAX - 128 ||
        a.nchannels_x != a.nchannels_y || a.nchannels_y < 1 || a.nchannels_y > INT_MAX ||
        a.nsamples_x != 1 || a.nsamples_y != 1 || a.stride_row_x < kCols / QK_NVFP4 ||
        a.stride_row_x > INT_MAX || a.stride_channel_x < 0 || a.stride_channel_x > INT_MAX ||
        a.nrows_dst < kRows || a.nrows_dst > INT_MAX ||
        a.ncols_y > INT_MAX / ((kPaddedYBlocks + 1) * kYWords)) return false;

    const int64_t last_block = (a.nchannels_y - 1) * a.stride_channel_x +
                               (kRows - 1) * a.stride_row_x + ((kCols + 255) / 256) * 256 / QK_NVFP4;
    if (last_block > INT_MAX) return false;
    const auto& dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (!GGML_CUDA_CC_IS_RDNA2(dev.cc) || dev.warp_size != 32 ||
        mmq_get_nbytes_shared(kConfig, dev.cc) != kShared || kShared > dev.smpbo ||
        !selects_j64(a, dev.cc, dev.smpbo)) return false;

    const dim3 block(32, kThreads / 32, 1);
    const dim3 grid(kRows / kI, (a.ncols_max + kJ - 1) / kJ, a.nchannels_y);
    CUDA_SET_SHARED_MEMORY_LIMIT(nvfp4_tail_kernel, kShared);
    nvfp4_tail_kernel<<<grid, block, kShared, stream>>>(
        a.x, a.y, a.ids_dst, a.expert_bounds, a.dst, a.ncols_x / QK_NVFP4,
        a.stride_row_x, a.ncols_y, a.nrows_dst, a.stride_channel_x);
    return true;
}

}  // namespace strata::poly
