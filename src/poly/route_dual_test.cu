// src/poly/route_dual_test.cu - the router's copies riding its gemv against the plain launches.
//
// STRATA_INKLING_ROUTE_DUAL (the engine's router_scores()/moe_router()): one token's router writes the host its
// copies itself - gemv_f32_waves_dual() the scores into h_rl_ on top of rl_ and the experts' input into h_x_ -
// and the visit launches no copy2 for them.  The gate here: the dual kernel's products are the plain
// gemv_f32_waves()'s bits (rl_ unchanged - route() reads the same scores), and the host copies are those
// products' and the input's bits (what copy2() would have carried).  Every "FAIL" line fails the run.
//
// The norm kernel that feeds the gemv is NOT touched by that path and is not tested here: it is frozen
// (its four shipped divergences, documented in poly_norm_q8_test), and the first attempt at writing h_x_ inside
// it broke one of them - the extra store materialized the normed product and killed the contract's contraction
// into the block sum - so the copy rides the gemv instead, which reads the input after the norm wrote it.
//
// usage: poly_route_dual_test   (exit 0: every comparison the same bits)
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata;

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> floats(const float* d, size_t n, bool* ok) {
    std::vector<float> v(n);
    if (const cudaError_t e = cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost); e != cudaSuccess) {
        std::printf("D2H %p +%zu: %s\n", (const void*) d, n, cudaGetErrorString(e));
        if (ok) *ok = false;
    }
    return v;
}

int same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// gemv_f32_waves' product against gemv_f32_waves_dual's, the dual's h_rl_ against it, and the dual's h_x_
// against the input the copy would have carried.  The host copies go to pinned memory, as the engine's do.  A
// shape the dual refuses must leave every destination exactly as it was (sentinels).
int check_gemv(int n_in, int n_out, bool fits, std::mt19937& rng) {
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    std::vector<float> w((size_t) n_in * n_out), x((size_t) n_in);
    for (float& v : w) v = ud(rng);
    for (float& v : x) v = ud(rng);
    const float* dw = upload(w);
    const float* dx = upload(x);
    float* y1 = nullptr;
    cudaMalloc(&y1, (size_t) n_out * sizeof(float));
    // the product stays on the card (rl_); the host's copies go to pinned memory, as the engine's h_rl_/h_x_ do -
    // and they are seeded, so a refusal must not touch them
    float* y2 = nullptr;
    cudaMalloc(&y2, (size_t) n_out * sizeof(float));
    std::vector<float> srl((size_t) n_out, -8.f), sx((size_t) n_in, -9.f);
    float *hrl = nullptr, *hx = nullptr;
    cudaHostAlloc(&hrl, (size_t) n_out * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&hx, (size_t) n_in * sizeof(float), cudaHostAllocDefault);
    std::memcpy(hrl, srl.data(), srl.size() * sizeof(float));
    std::memcpy(hx, sx.data(), sx.size() * sizeof(float));
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    poly::gemv_f32_waves(dw, dx, y1, n_in, n_out, n_out, 1, s);
    const bool took = poly::gemv_f32_waves_dual(dw, dx, y2, hrl, hx, n_in, n_out, n_out, 1, s);
    cudaStreamSynchronize(s);
    bool ok = true;
    const std::vector<float> fy1 = floats(y1, (size_t) n_out, &ok), fy2 = floats(y2, (size_t) n_out, &ok);
    std::vector<float> frl((size_t) n_out), fx((size_t) n_in);
    std::memcpy(frl.data(), hrl, frl.size() * sizeof(float));
    std::memcpy(fx.data(), hx, fx.size() * sizeof(float));
    int bad = !ok;
    if (took != fits) ++bad;
    if (took) {
        if (!same(fy1, fy2)) ++bad;
        if (!same(fy1, frl)) ++bad;
        if (!same(x, fx)) ++bad;
    } else {
        if (!same(srl, frl)) ++bad;   // a refusal must have written nothing
        if (!same(sx, fx)) ++bad;
    }
    std::printf("%s gemv %4d x %3d %s: %s, y %s, h_rl %s, h_x %s\n", bad ? "FAIL" : "ok  ", n_in, n_out,
                fits ? "dual" : "back", took == fits ? "right" : "WRONG", same(fy1, fy2) ? "same" : "DIFF",
                took ? (same(fy1, frl) ? "same" : "DIFF") : (same(srl, frl) ? "kept" : "WROTE"),
                took ? (same(x, fx) ? "same" : "DIFF") : (same(sx, fx) ? "kept" : "WROTE"));
    cudaFree(y1), cudaFree(y2), cudaFree((void*) dw), cudaFree((void*) dx);
    cudaFreeHost(hrl), cudaFreeHost(hx);
    cudaStreamDestroy(s);
    return bad;
}

}  // namespace

int main() {
    std::mt19937 rng(31);
    int bad = 0;
    // the router's shape (2304 in, 128 out) takes the dual; shorter rows, a taller matrix and a copy the launch
    // cannot carry whole fall back, writing nothing
    bad += check_gemv(2304, 128, true, rng);
    bad += check_gemv(1024, 64, true, rng);
    bad += check_gemv(512, 128, false, rng);   // gemv_f32_waves would take the waves kernel, not a block a row
    bad += check_gemv(2304, 640, false, rng);  // (640 + 7) / 8 = 80 blocks: the plain path's waves kernel
    bad += check_gemv(2304, 8, false, rng);    // 8 blocks of 256 threads cannot copy 2304 floats whole
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
