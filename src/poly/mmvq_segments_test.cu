// src/poly/mmvq_segments_test.cu - native_mmvq_segments() against one native_mmvq() call a matrix: the shapes of an
// Inkling-2-Mini decode step's q, gate, rel, k and v (2,304 inputs; k and v with inputs of their own), in Q8_0,
// Q5_K and Q6_K, every output bit for bit. Also times a step's five Q8_0 projections both ways, over copies of the
// weights that keep each call out of the cache the one before it read. And native_q6_k_mmvq_warps() against
// native_mmvq() on the output head's shape (200,058 x 2,304 Q6_K): within 1e-5 of the largest output (it sums in
// another order), and timed.
//
// The few-row kernels a guesser uses (cols()).
// usage: poly_mmvq_segments_test   (exit 0: all equal, the head within tolerance; 1: not; 77: no card)
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using strata::kernels::MmvqSegments;
constexpr int kIn = 2304, kSegs = 5;
constexpr int kRows[kSegs] = {3072, 3072, 384, 1024, 1024};   // q, gate, rel, k, v

// random blocks of the format, with half scales of a sensible size where the format keeps them
std::vector<uint8_t> blocks(int type, size_t bytes, std::mt19937& rng) {
    std::vector<uint8_t> h(bytes);
    for (auto& b : h) b = (uint8_t) rng();
    const auto half_at = [&](size_t at) {   // 2^-8 with a random mantissa
        const uint16_t v = (uint16_t) (0x1c00 | (rng() & 0x3ff));
        std::memcpy(&h[at], &v, 2);
    };
    if (type == 8)
        for (size_t at = 0; at + 34 <= bytes; at += 34) half_at(at);
    else if (type == 14)
        for (size_t at = 0; at + 210 <= bytes; at += 210) half_at(at + 208);
    else   // Q5_K: d and dmin first
        for (size_t at = 0; at + 176 <= bytes; at += 176) half_at(at), half_at(at + 2);
    return h;
}

bool check(int type, const char* name, std::mt19937& rng, cudaStream_t st) {
    std::vector<void*> w(kSegs), x(kSegs);
    std::vector<float*> a(kSegs), b(kSegs);
    std::vector<float> xf(kIn);
    for (int j = 0; j < kSegs; ++j) {
        const size_t bytes = strata::kernels::native_mmvq_weight_bytes(type, kIn, kRows[j]);
        const auto h = blocks(type, bytes, rng);
        cudaMalloc(&w[j], bytes);
        cudaMemcpy(w[j], h.data(), bytes, cudaMemcpyHostToDevice);
        cudaMalloc(&a[j], kRows[j] * sizeof(float));
        cudaMalloc(&b[j], kRows[j] * sizeof(float));
        // q, gate, rel share an input; k and v have their own
        if (j < 3 && j > 0) {
            x[j] = x[0];
            continue;
        }
        std::normal_distribution<float> d(0.f, 1.f);
        for (auto& v : xf) v = d(rng);
        float* xd = nullptr;
        cudaMalloc(&xd, kIn * sizeof(float));
        cudaMemcpy(xd, xf.data(), kIn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&x[j], kIn / 32 * 36);
        strata::kernels::native_quantize_q8_1(xd, x[j], kIn, 1, st);
        cudaFree(xd);
    }
    MmvqSegments s;
    s.count = kSegs;
    for (int j = 0; j < kSegs; ++j) {
        strata::kernels::native_mmvq(type, w[j], x[j], a[j], kIn, kRows[j], 1, st);
        s.w[j] = w[j];
        s.x[j] = x[j];
        s.y[j] = b[j];
        s.rows[j] = kRows[j];
    }
    const bool ran = strata::kernels::native_mmvq_segments(type, s, kIn, st);
    cudaStreamSynchronize(st);
    int differ = 0, total = 0;
    for (int j = 0; j < kSegs && ran; ++j) {
        std::vector<float> ha(kRows[j]), hb(kRows[j]);
        cudaMemcpy(ha.data(), a[j], kRows[j] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), b[j], kRows[j] * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < kRows[j]; ++i) differ += std::memcmp(&ha[i], &hb[i], 4) != 0;
        total += kRows[j];
    }
    std::printf("%-5s: %s, %d of %d outputs differ\n", name, ran ? "segments ran" : "NOT RUN", differ, total);
    for (int j = 0; j < kSegs; ++j) {
        cudaFree(w[j]);
        cudaFree(a[j]);
        cudaFree(b[j]);
        if (j == 0 || j >= 3) cudaFree(x[j]);
    }
    return ran && differ == 0;
}

void timed(std::mt19937& rng, cudaStream_t st) {
    size_t step = 0;
    for (int j = 0; j < kSegs; ++j) step += strata::kernels::native_mmvq_weight_bytes(8, kIn, kRows[j]);
    const int copies = 200;   // 4 GB of steps: no step finds the one before it in the 96 MB cache
    uint8_t* w = nullptr;
    if (cudaMalloc(&w, step * copies) != cudaSuccess) {
        cudaGetLastError();
        std::puts("timing skipped: no room for the copies");
        return;
    }
    const auto h = blocks(8, step, rng);
    for (int c = 0; c < copies; ++c) cudaMemcpy(w + c * step, h.data(), step, cudaMemcpyHostToDevice);
    void* x = nullptr;
    float *xd = nullptr, *y = nullptr;
    cudaMalloc(&x, kIn / 32 * 36);
    cudaMalloc(&xd, kIn * sizeof(float));
    cudaMemset(xd, 0, kIn * sizeof(float));
    cudaMalloc(&y, 16384 * sizeof(float));
    strata::kernels::native_quantize_q8_1(xd, x, kIn, 1, st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int rep = 0; rep < 3; ++rep) {   // the first warms the clocks
        float apart = 0, together = 0;
        cudaEventRecord(e0, st);
        for (int c = 0; c < copies; ++c) {
            size_t at = c * step;
            for (int j = 0; j < kSegs; ++j) {
                strata::kernels::native_mmvq(8, w + at, x, y, kIn, kRows[j], 1, st);
                at += strata::kernels::native_mmvq_weight_bytes(8, kIn, kRows[j]);
            }
        }
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&apart, e0, e1);
        cudaEventRecord(e0, st);
        for (int c = 0; c < copies; ++c) {
            MmvqSegments s;
            s.count = kSegs;
            size_t at = c * step;
            int out = 0;
            for (int j = 0; j < kSegs; ++j) {
                s.w[j] = w + at;
                s.x[j] = x;
                s.y[j] = y + out;
                s.rows[j] = kRows[j];
                at += strata::kernels::native_mmvq_weight_bytes(8, kIn, kRows[j]);
                out += kRows[j];
            }
            strata::kernels::native_mmvq_segments(8, s, kIn, st);
        }
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&together, e0, e1);
        if (rep) std::printf("Q8_0 step of five projections: %.1f us as five launches, %.1f us as one\n",
                             apart * 1e3f / copies, together * 1e3f / copies);
    }
    cudaFree(w);
    cudaFree(x);
    cudaFree(xd);
    cudaFree(y);
}

// native_q6_k_mmvq_warps_cols() and native_q8_0_mmvq_warps_cols() on a few rows (a guesser's window and block): a
// Q6_K column bit for bit native_q6_k_mmvq_warps()'s, a Q8_0 column within 1e-5 of native_mmvq()'s largest output
// (another order of the sums); and the times against native_mmvq() for the same rows.
bool cols(std::mt19937& rng, cudaStream_t st, int vocab) {
    bool ok = true;
    for (const int type : {14, 8}) {
        const int n_out = type == 14 ? vocab : 9216;
        const size_t bytes = strata::kernels::native_mmvq_weight_bytes(type, kIn, n_out);
        void* w = nullptr;
        if (cudaMalloc(&w, bytes) != cudaSuccess) {
            cudaGetLastError();
            std::puts("cols: no room for the matrix  FAIL");
            return false;
        }
        const auto h = blocks(type, bytes, rng);
        cudaMemcpy(w, h.data(), bytes, cudaMemcpyHostToDevice);
        constexpr int kRows = 8;
        std::vector<float> xf((size_t) kRows * kIn);
        std::normal_distribution<float> d(0.f, 1.f);
        for (auto& v : xf) v = d(rng);
        float *xd = nullptr, *a = nullptr, *b = nullptr;
        void* x = nullptr;
        cudaMalloc(&xd, xf.size() * sizeof(float));
        cudaMemcpy(xd, xf.data(), xf.size() * sizeof(float), cudaMemcpyHostToDevice);
        const size_t row = (size_t) kIn / 32 * 36;
        cudaMalloc(&x, kRows * row);
        cudaMalloc(&a, (size_t) kRows * n_out * sizeof(float));
        cudaMalloc(&b, (size_t) kRows * n_out * sizeof(float));
        strata::kernels::native_quantize_q8_1(xd, x, kIn * kRows, 1, st);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (const int nc : {2, 4, 8}) {
            float ms_native = 0, ms_cols = 0;
            for (int rep = 0; rep < 4; ++rep) {
                cudaEventRecord(e0, st);
                if (type == 14)   // the reference: a column at a time, a warp a row
                    for (int c = 0; c < nc; ++c)
                        strata::kernels::native_q6_k_mmvq_warps(w, (const uint8_t*) x + c * row, a + (size_t) c * n_out,
                                                                 kIn, n_out, st);
                else strata::kernels::native_mmvq(type, w, x, a, kIn, n_out, nc, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float t = 0;
                cudaEventElapsedTime(&t, e0, e1);
                if (rep) ms_native += t;
                cudaEventRecord(e0, st);
                const bool took = type == 14 ? strata::kernels::native_q6_k_mmvq_warps_cols(w, x, b, kIn, n_out, nc, st)
                                             : strata::kernels::native_q8_0_mmvq_warps_cols(w, x, b, kIn, n_out, nc, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&t, e0, e1);
                if (rep) ms_cols += t;
                ok = ok && took;
            }
            std::vector<float> ha((size_t) nc * n_out), hb((size_t) nc * n_out);
            cudaMemcpy(ha.data(), a, ha.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), b, hb.size() * sizeof(float), cudaMemcpyDeviceToHost);
            double top = 0, diff = 0;
            for (size_t i = 0; i < ha.size(); ++i) {
                top = std::max(top, (double) std::fabs(ha[i]));
                diff = std::max(diff, (double) std::fabs(ha[i] - hb[i]));
            }
            const bool same = type == 14 ? diff == 0 : diff <= 1e-5 * top;
            ok = ok && same;
            std::printf("%s %d x %d, %d rows: reference %.1f us, warps_cols %.1f us; difference %.3g of %.3g%s\n",
                        type == 14 ? "Q6_K" : "Q8_0", n_out, kIn, nc, ms_native * 1e3f / 3, ms_cols * 1e3f / 3, diff, top,
                        same ? "" : "  FAIL");
        }
        cudaFree(w);
        cudaFree(x);
        cudaFree(xd);
        cudaFree(a);
        cudaFree(b);
    }
    return ok;
}

bool head(std::mt19937& rng, cudaStream_t st) {
    constexpr int kVocab = 200058;
    const size_t bytes = strata::kernels::native_mmvq_weight_bytes(14, kIn, kVocab);
    void* w = nullptr;
    void* x = nullptr;
    float *xd = nullptr, *a = nullptr, *b = nullptr;
    if (cudaMalloc(&w, bytes) != cudaSuccess) {
        cudaGetLastError();
        std::puts("head: no room for the matrix  FAIL");
        return false;
    }
    const auto h = blocks(14, bytes, rng);
    cudaMemcpy(w, h.data(), bytes, cudaMemcpyHostToDevice);
    std::vector<float> xf(kIn);
    std::normal_distribution<float> d(0.f, 1.f);
    for (auto& v : xf) v = d(rng);
    cudaMalloc(&xd, kIn * sizeof(float));
    cudaMemcpy(xd, xf.data(), kIn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&x, kIn / 32 * 36);
    cudaMalloc(&a, kVocab * sizeof(float));
    cudaMalloc(&b, kVocab * sizeof(float));
    strata::kernels::native_quantize_q8_1(xd, x, kIn, 1, st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float ms_native = 0, ms_warps = 0;
    for (int rep = 0; rep < 4; ++rep) {   // the last three timed
        cudaEventRecord(e0, st);
        strata::kernels::native_mmvq(14, w, x, a, kIn, kVocab, 1, st);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float t = 0;
        cudaEventElapsedTime(&t, e0, e1);
        if (rep) ms_native += t;
        cudaEventRecord(e0, st);
        strata::kernels::native_q6_k_mmvq_warps(w, x, b, kIn, kVocab, st);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t, e0, e1);
        if (rep) ms_warps += t;
    }
    std::vector<float> ha(kVocab), hb(kVocab);
    cudaMemcpy(ha.data(), a, kVocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, kVocab * sizeof(float), cudaMemcpyDeviceToHost);
    double top = 0, diff = 0;
    bool finite = true;
    for (int i = 0; i < kVocab; ++i) {
        finite = finite && std::isfinite(hb[i]);
        top = std::max(top, (double) std::fabs(ha[i]));
        diff = std::max(diff, (double) std::fabs(ha[i] - hb[i]));
    }
    const bool ok = finite && diff <= 1e-5 * top;
    std::printf("head Q6_K %d x %d: native %.1f us, a warp a row %.1f us; difference %.3g of %.3g%s\n", kVocab, kIn,
                ms_native * 1e3f / 3, ms_warps * 1e3f / 3, diff, top, ok ? "" : "  FAIL");
    cudaFree(w);
    cudaFree(x);
    cudaFree(xd);
    cudaFree(a);
    cudaFree(b);
    return ok && cols(rng, st, kVocab);
}

}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::puts("no card");
        return 77;
    }
    cudaStream_t st;
    cudaStreamCreate(&st);
    std::mt19937 rng(7);
    bool ok = check(8, "Q8_0", rng, st);
    ok = check(13, "Q5_K", rng, st) && ok;
    ok = check(14, "Q6_K", rng, st) && ok;
    timed(rng, st);
    ok = head(rng, st) && ok;
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
