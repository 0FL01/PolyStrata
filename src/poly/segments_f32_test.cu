// src/poly/segments_f32_test.cu - gemv_f32_waves_segments() and rms_norm_segments() against one gemv_f32_waves() or
// rms_norm() call a segment.  Every output must be the same bytes (Inkling's decode streams, Inkling2::heads()).
//
// usage: poly_segments_f32_test   (exit 0: every byte the same)
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

float* upload(const std::vector<float>& v) {
    float* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(float));
    cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> down(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](float x, float y) {
               return std::memcmp(&x, &y, sizeof(float)) == 0;
           });
}

std::vector<float> random(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

int gemv_case(int count, int n_in, int n_out, int rows, std::mt19937& rng) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    F32Segments seg;
    std::vector<float*> w(count), x(count), y1(count), y2(count);
    for (int g = 0; g < count; ++g) {
        w[g] = upload(random((size_t) n_in * n_out, rng));
        x[g] = upload(random((size_t) n_in * rows, rng));
        y1[g] = upload(std::vector<float>((size_t) n_out * rows, -1.f));
        y2[g] = upload(std::vector<float>((size_t) n_out * rows, -2.f));
        gemv_f32_waves(w[g], x[g], y1[g], n_in, n_out, n_out, rows, s);
        seg.w[g] = w[g];
        seg.x[g] = x[g];
        seg.y[g] = y2[g];
    }
    gemv_f32_waves_segments(seg, count, n_in, n_out, n_out, rows, s);
    cudaStreamSynchronize(s);
    bool ok = true;
    for (int g = 0; g < count; ++g) {
        ok = ok && same(down(y1[g], (size_t) n_out * rows), down(y2[g], (size_t) n_out * rows));
        cudaFree(w[g]), cudaFree(x[g]), cudaFree(y1[g]), cudaFree(y2[g]);
    }
    cudaStreamDestroy(s);
    std::printf("%s gemv  segments %d n_in %5d n_out %5d rows %2d\n", ok ? "ok  " : "FAIL", count, n_in, n_out, rows);
    return ok ? 0 : 1;
}

int norm_case(int count, int n, int rows, std::mt19937& rng) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    NormSegments seg;
    std::vector<float*> x(count), w(count), y1(count), y2(count);
    for (int g = 0; g < count; ++g) {
        x[g] = upload(random((size_t) n * rows, rng));
        std::vector<float> wv = random(n, rng);
        for (float& v : wv) v = 1.f + 0.1f * v;
        w[g] = upload(wv);
        y1[g] = upload(std::vector<float>((size_t) n * rows, -1.f));
        y2[g] = upload(std::vector<float>((size_t) n * rows, -2.f));
        rms_norm(x[g], w[g], y1[g], n, rows, 1.e-6f, s);
        seg.x[g] = x[g];
        seg.w[g] = w[g];
        seg.y[g] = y2[g];
    }
    rms_norm_segments(seg, count, n, rows, 1.e-6f, s);
    cudaStreamSynchronize(s);
    bool ok = true;
    for (int g = 0; g < count; ++g) {
        ok = ok && same(down(y1[g], (size_t) n * rows), down(y2[g], (size_t) n * rows));
        cudaFree(x[g]), cudaFree(w[g]), cudaFree(y1[g]), cudaFree(y2[g]);
    }
    cudaStreamDestroy(s);
    std::printf("%s norm  segments %d n %5d rows %2d\n", ok ? "ok  " : "FAIL", count, n, rows);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(17);
    int bad = 0;
    // Inkling's control down products (4608 -> 64) take the block-a-row kernel for few rows, the wave one for more
    for (int rows : {1, 2, 7, 9, 31})
        for (int count : {1, 3, 4}) bad += gemv_case(count, 4608, 64, rows, rng);
    for (int count : {2, 3}) bad += gemv_case(count, 256, 300, 3, rng);
    for (int count : {2, 3}) bad += gemv_case(count, 100, 77, 2, rng);   // n_in % 4: gemv() per segment
    for (int rows : {1, 3, 31})
        for (int count : {1, 3}) bad += norm_case(count, 2304, rows, rng);
    bad += norm_case(4, 4096, 2, rng);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
