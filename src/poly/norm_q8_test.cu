// src/poly/norm_q8_test.cu - the fused norm+quantize kernels against the launches they stand for.
//
// native_rms_norm_q8_1_rows() (the engine's STRATA_INKLING_NORM_Q8 path) against rms_norm() followed by
// native_quantize_q8_1() of each entry: the normed rows and the Q8_1 blocks must be the same bytes.  This is the
// gate: every "FAIL" line fails the run.
//
// native_add_rms_norm_q8_1() (the decode tail since ce9e9b28) is printed as "diff (expected)" and does not fail the
// run: its rows are NOT the two launches' bits and never were.  On this toolchain (HIP 7.13, plain -O3) the
// differences are, measured on gfx1031:
//   - the sum of squares: poly's rms_norm accumulates an fma (the contract-flagged multiply and add fuse at SASS),
//     and the tail kernel matches that, but its 128-thread stride groups the values differently than rms_norm's
//     sum_threads(n) block - a data-dependent 1-ulp scale;
//   - the scale: the tail kernel's __fsqrt_rn/__fdiv_rn is the raw v_sqrt/v_div sequence, while rms_norm's
//     1.0f/sqrtf() takes the libdevice sqrt, which refines v_sqrt to the correctly rounded value - off by 1 ulp
//     on boundary values;
//   - the Q8_1 block sum: the contract-flagged normed product fuses into the block sum's first butterfly add
//     (v_fmac), so the ds high half rounds differently in ~1/2500 blocks (qs and d stay equal).
// Production answers were validated with these divergences in place; changing them would change the model's bits,
// so they are frozen, not fixed.  native_rms_norm_q8_1_rows() above is the kernel that finally is the two
// launches' bits (fma squares on the same stride, libdevice scale, no contraction in the quantize loop).
//
// usage: poly_norm_q8_test   (exit 0: every rows comparison the same bytes)
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

std::vector<unsigned char> bytes(const void* d, size_t n) {
    std::vector<unsigned char> v(n);
    if (const cudaError_t e = cudaMemcpy(v.data(), d, n, cudaMemcpyDeviceToHost); e != cudaSuccess)
        std::printf("D2H %p +%zu: %s\n", d, n, cudaGetErrorString(e));
    return v;
}

// native_add_rms_norm_q8_1() against axpy() + rms_norm() + native_quantize_q8_1(), with and without the sum.
// Printed only (see the header): the residual tells the add fused right, the normed/q8_1 columns tell which of the
// three known divergences the data hit.
void check(int n, int rows, bool with_add, float spread, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, spread);
    std::vector<float> r((size_t) n * rows), add(r.size()), w(n);
    for (float& x : r) x = nd(rng);
    for (float& x : add) x = nd(rng);
    for (float& x : w) x = 1.f + 0.1f * nd(rng);
    const size_t qb = kernels::native_q8_1_bytes(n) * rows, fb = r.size() * sizeof(float);
    float *r1 = upload(r), *r2 = upload(r), *a = upload(add), *dw = upload(w), *y1 = nullptr, *y2 = nullptr;
    void *q1 = nullptr, *q2 = nullptr;
    cudaMalloc(&y1, fb), cudaMalloc(&y2, fb), cudaMalloc(&q1, qb), cudaMalloc(&q2, qb);
    cudaStream_t s;
    cudaStreamCreate(&s);
    const float eps = 1e-5f;
    if (with_add) poly::axpy(a, 1.f, r1, n * rows, true, s);
    poly::rms_norm(r1, dw, y1, n, rows, eps, s);
    kernels::native_quantize_q8_1(y1, q1, n * rows, 1, s);
    kernels::native_add_rms_norm_q8_1(r2, with_add ? a : nullptr, dw, y2, q2, n, rows, eps, poly::sum_threads(n), s);
    cudaStreamSynchronize(s);
    const bool same_r = bytes(r1, fb) == bytes(r2, fb), same_y = bytes(y1, fb) == bytes(y2, fb),
               same_q = bytes(q1, qb) == bytes(q2, qb);
    std::printf("%s n %5d rows %4d %s spread %g: residual %s, normed %s, q8_1 %s\n", same_y && same_q ? "ok  " : "diff",
                n, rows, with_add ? "add " : "norm", spread, same_r ? "same" : "DIFF", same_y ? "same" : "DIFF",
                same_q ? "same" : "DIFF");
    cudaFree(r1), cudaFree(r2), cudaFree(a), cudaFree(dw), cudaFree(y1), cudaFree(y2), cudaFree(q1), cudaFree(q2);
    cudaStreamDestroy(s);
}

// native_rms_norm_q8_1_rows() against rms_norm() + native_quantize_q8_1() of each entry: the normed rows and the
// Q8_1 blocks must be the same bytes.  `adjacent` lays the three entries out as heads() keeps them - one buffer a
// kind, an entry's Q8_1 slot n rows into it - with the reference quantizing the whole y flat, as the old path did.
int check_rows(int n, int entries, int tokens, bool adjacent, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t fb = (size_t) n * tokens;
    std::vector<std::vector<float>> xs(entries), ws(entries);
    std::vector<float*> x1(entries), x2(entries), y1(entries), y2(entries), dws(entries);
    std::vector<void*> q1(entries), q2(entries);
    const size_t qb = kernels::native_q8_1_bytes(n) * tokens;
    float* flat_y1 = nullptr, * flat_y2 = nullptr, * flat_x2 = nullptr;
    void* flat_q1 = nullptr, * flat_q2 = nullptr;
    const size_t wq1 = kernels::native_q8_1_bytes(n);
    if (adjacent) {
        flat_x2 = upload(std::vector<float>(fb * entries));
        cudaMalloc(&flat_y1, fb * entries * sizeof(float));
        cudaMalloc(&flat_y2, fb * entries * sizeof(float));
        cudaMalloc(&flat_q1, qb * entries);
        cudaMalloc(&flat_q2, qb * entries);
        std::vector<float> all(fb * entries);
        for (int e = 0; e < entries; ++e) {
            xs[e].resize(fb);
            for (float& x : xs[e]) x = nd(rng);
            ws[e].resize(n);
            for (float& x : ws[e]) x = 1.f + 0.1f * nd(rng);
            x1[e] = upload(xs[e]);
            std::copy(xs[e].begin(), xs[e].end(), all.begin() + (size_t) e * fb);
            y1[e] = flat_y1 + (size_t) e * fb;
            y2[e] = flat_y2 + (size_t) e * fb;
            q1[e] = (uint8_t*) flat_q1 + (size_t) e * tokens * wq1;   // heads(): the slot starts n rows in
            q2[e] = (uint8_t*) flat_q2 + (size_t) e * tokens * wq1;
            dws[e] = upload(ws[e]);
        }
        // the fused path reads the same adjacent inputs
        cudaMemcpy(flat_x2, all.data(), all.size() * sizeof(float), cudaMemcpyHostToDevice);
        for (int e = 0; e < entries; ++e) x2[e] = flat_x2 + (size_t) e * fb;
    } else {
        for (int e = 0; e < entries; ++e) {
            xs[e].resize(fb);
            for (float& x : xs[e]) x = nd(rng);
            ws[e].resize(n);
            for (float& x : ws[e]) x = 1.f + 0.1f * nd(rng);
            x1[e] = upload(xs[e]);
            x2[e] = upload(xs[e]);
            dws[e] = upload(ws[e]);
            cudaMalloc(&y1[e], fb * sizeof(float));
            cudaMalloc(&y2[e], fb * sizeof(float));
            cudaMalloc(&q1[e], qb);
            cudaMalloc(&q2[e], qb);
        }
    }
    cudaStream_t s;
    cudaStreamCreate(&s);
    const float eps = 1e-5f;
    for (int e = 0; e < entries; ++e) {
        poly::rms_norm(x1[e], dws[e], y1[e], n, tokens, eps, s);
        if (!adjacent) kernels::native_quantize_q8_1(y1[e], q1[e], n * tokens, 1, s);
    }
    if (adjacent) kernels::native_quantize_q8_1(flat_y1, flat_q1, n * tokens * entries, 1, s);
    std::vector<kernels::NormQ8Row> rs(entries);
    for (int e = 0; e < entries; ++e) rs[e] = {x2[e], dws[e], y2[e], q2[e]};
    kernels::native_rms_norm_q8_1_rows(rs.data(), entries, n, tokens, eps, poly::sum_threads(n), s);
    cudaStreamSynchronize(s);
    int bad = 0;
    for (int e = 0; e < entries; ++e) {
        const bool same_y = bytes(y1[e], fb * sizeof(float)) == bytes(y2[e], fb * sizeof(float)),
                   same_q = bytes(q1[e], qb) == bytes(q2[e], qb);
        if (!same_y || !same_q) ++bad;
        std::printf("%s n %5d entries %d tokens %4d %s entry %d: normed %s, q8_1 %s\n",
                    same_y && same_q ? "ok  " : "FAIL", n, entries, tokens, adjacent ? "adj" : "sep", e,
                    same_y ? "same" : "DIFF", same_q ? "same" : "DIFF");
    }
    for (int e = 0; e < entries; ++e) cudaFree(x1[e]), cudaFree(dws[e]);   // the per-entry uploads
    if (adjacent) {
        cudaFree(flat_x2), cudaFree(flat_y1), cudaFree(flat_y2), cudaFree(flat_q1), cudaFree(flat_q2);
    } else {
        for (int e = 0; e < entries; ++e) {
            cudaFree(x2[e]), cudaFree(y1[e]), cudaFree(y2[e]), cudaFree(q1[e]), cudaFree(q2[e]);
        }
    }
    cudaStreamDestroy(s);
    return bad;
}

}  // namespace

int main() {
    std::mt19937 rng(11);
    for (int n : {512, 1024, 2048, 2816, 4096, 5120})
        for (int rows : {1, 3, 8, 64})
            for (bool add : {false, true}) check(n, rows, add, 1.f, rng);
    check(2048, 1024, true, 1.f, rng);   // a prompt batch
    check(2048, 8, true, 300.f, rng);    // large activations
    int bad = 0;
    // 2304: Inkling-2-Mini's kWidth; the adjacent layout is heads()' one buffer a kind, a stream's Q8_1 slot
    // starting n rows in - the flat quantize's layout the consumer still reads
    for (int n : {1024, 2304, 2816})
        for (int entries : {1, 3})
            for (int tokens : {1, 8})
                for (bool adjacent : {false, true}) bad += check_rows(n, entries, tokens, adjacent, rng);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
