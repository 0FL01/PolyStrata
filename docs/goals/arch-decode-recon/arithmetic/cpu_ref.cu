// Bounded RDNA2 replay of the pinned Ryzen AVX2 Q8_0 and Q4_0 leaf arithmetic.
#include "cpu_ref.hpp"
#include <hip/hip_fp16.h>

namespace cp04 {
namespace {
constexpr int threads = 128;

__global__ void quantize_avx2_q8_0(const float* __restrict__ x, Q80* __restrict__ out, int nb) {
    const int b = (blockIdx.x * threads + threadIdx.x) / 32;
    if (b >= nb) return;  // Full waves only.
    const int lane = threadIdx.x % 32;
    const float v = x[b * 32 + lane];
    float amax = fabsf(v);
    for (int step = 16; step > 0; step /= 2) amax = fmaxf(amax, __shfl_xor(amax, step, 32));
    const float inv = amax != 0.f ? __fdiv_rn(127.f, amax) : 0.f;
    const int q = __float2int_rn(__fmul_rn(v, inv));
    out[b].q[lane] = static_cast<int8_t>(min(127, max(-128, q)));
    if (lane == 0) out[b].d = __half_as_ushort(__float2half_rn(__fdiv_rn(amax, 127.f)));
}

template<class T> __device__ __forceinline__ T load(const void* p) {
    T v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

__global__ void project_avx2_q4_0_q8_0(const uint8_t* __restrict__ w, const Q80* __restrict__ x,
                                      float* __restrict__ y, int nb, int rows) {
    constexpr int lanes = 8;
    const int row = (blockIdx.x * threads + threadIdx.x) / lanes;
    if (row >= rows) return;
    const int lane = threadIdx.x % lanes;
    const int qoff = 4 * lane;
    const uint8_t* wr = w + row * nb * 18;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = wr + b * 18;
        // Each lane represents one AVX2 group of four consecutive weights.
        // The high nibbles form groups 4..7, not alternating with the lows.
        const uint32_t p = load<uint32_t>(wb + 2 + qoff % 16);
        const uint32_t u = ((lane < 4 ? p : p >> 4) & 0x0f0f0f0fu);
        const uint32_t s = ((u | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
        const int qx = load<int>(x[b].q + qoff);
        const int dot = __builtin_amdgcn_sdot4(static_cast<int>(s), qx, 0, false);
        const float dw = __half2float(__ushort_as_half(load<uint16_t>(wb)));
        const float dx = __half2float(__ushort_as_half(x[b].d));
        acc = __fmaf_rn(__fmul_rn(dw, dx), static_cast<float>(dot), acc);
    }
    // hsum_float_8: high+low halves, then 2+2, then 1+1.
    acc = __fadd_rn(__shfl_xor(acc, 4, lanes), acc);
    acc = __fadd_rn(acc, __shfl_xor(acc, 2, lanes));
    acc = __fadd_rn(acc, __shfl_xor(acc, 1, lanes));
    if (lane == 0) y[row] = acc;
}
}

void quantize(const float* x, Q80* q, int nb, hipStream_t st) {
    quantize_avx2_q8_0<<<(nb * 32 + threads - 1) / threads, threads, 0, st>>>(x, q, nb);
}

void project(const uint8_t* w, const Q80* x, float* y, int nb, int rows, hipStream_t st) {
    project_avx2_q4_0_q8_0<<<(rows * 8 + threads - 1) / threads, threads, 0, st>>>(w, x, y, nb, rows);
}
}
