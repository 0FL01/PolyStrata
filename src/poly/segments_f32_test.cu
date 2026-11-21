// src/poly/segments_f32_test.cu - gemv_f32_waves_segments() and rms_norm_segments() against one gemv_f32_waves() or
// rms_norm() call a segment, and situ_rows_q8_1() against situ_rows() then native_quantize_q8_1().  Every output must
// be the same bytes (Inkling's decode path).
//
// usage: poly_segments_f32_test   (exit 0: every byte the same)
#include "strata/kernels/native_mmvq.hpp"
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

// gate and up of `rows` rows (n each), the SiTU of the routed experts, scaled by expert e's scales
int situ_case(int n, int rows, float beta, float lbeta, const std::vector<float>& gu, bool scaled, const char* what) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    float* g = upload(gu);
    float* h = upload(std::vector<float>((size_t) n * rows, 0.f));
    std::vector<float> sc(8, 1.f);
    for (int i = 0; i < 8; ++i) sc[i] = 0.75f + 0.125f * i;
    float* dsc = upload(sc);
    ExpertScales es{};
    if (scaled) {
        es.gate = dsc;
        es.up = dsc + 1;
        for (int r = 0; r < rows; ++r) es.e[r] = r % 6;
    }
    const size_t qb = strata::kernels::native_q8_1_bytes(n) * rows;
    void *q1 = nullptr, *q2 = nullptr;
    cudaMalloc(&q1, qb);
    cudaMalloc(&q2, qb);
    cudaMemset(q1, 0x5a, qb);
    cudaMemset(q2, 0xa5, qb);
    situ_rows(g, h, n, rows, beta, lbeta, s, es);
    strata::kernels::native_quantize_q8_1(h, q1, n, rows, s);
    situ_rows_q8_1(g, q2, n, rows, beta, lbeta, s, es);
    cudaStreamSynchronize(s);
    std::vector<unsigned char> a(qb), b(qb);
    cudaMemcpy(a.data(), q1, qb, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), q2, qb, cudaMemcpyDeviceToHost);
    const bool ok = a == b;
    for (size_t i = 0, shown = 0; i < qb && shown < 6; ++i)
        if (a[i] != b[i]) {
            std::printf("     byte %zu (block %zu, offset %zu): %02x vs %02x\n", i, i / 36, i % 36, a[i], b[i]);
            ++shown;
        }
    cudaFree(g), cudaFree(h), cudaFree(dsc), cudaFree(q1), cudaFree(q2);
    cudaStreamDestroy(s);
    std::printf("%s situ  q8_1 n %5d rows %2d %s\n", ok ? "ok  " : "FAIL", n, rows, what);
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
    for (int rows : {1, 3, 8})
        for (bool scaled : {false, true}) {
            std::vector<float> gu = random((size_t) 2 * 1152 * rows, rng);
            for (float& v : gu) v *= 3.f;
            bad += situ_case(1152, rows, 10.f, 10.f, gu, scaled, scaled ? "random, scaled" : "random");
        }
    {   // the review's block: lanes 0 and 16 large and opposite, the rest 0, betas 83.2
        std::vector<float> gu((size_t) 2 * 1152, 0.f);
        gu[0] = 2048.f, gu[16] = 2048.f, gu[1152 + 0] = 2048.f, gu[1152 + 16] = -2048.f;
        bad += situ_case(1152, 1, 83.2f, 83.2f, gu, false, "opposite pair");
    }
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
