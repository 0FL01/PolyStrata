// src/poly/mmq_tiles_test.cu - the size-class launches of STRATA_POLY_STAGED_TILES (engine.cpp, moe_rows) give the same
// results as one launch: to the bit, except on a card whose MMQ uses stream-k (Mmq::split_exact()), where they are
// the same products added in another order and must agree within rounding.  llama.cpp's MMQ takes a launch's tile
// width from the largest expert in it, so the engine launches the experts of up to N rows apart from the larger
// ones, each class with bounds in which the other classes' experts have no rows, over rows laid out class after
// class.  That layout is built here the way the engine builds it, for row counts with many small experts, a few
// large, empty ones and equal ones.
//
// usage: poly_mmq_tiles_test   (exit 0: every case agrees; 77: no card)
#include "mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using strata::poly::Mmq;

namespace {

struct Quant { const char* name; int type; size_t block; int qk = 256; };   // ggml_type, block bytes and values
constexpr Quant kQuants[] = {{"Q4_K", 12, 144}, {"Q5_K", 13, 176}, {"Q6_K", 14, 210}};
constexpr int kRows = 256, kCols = 512;   // a matrix of the shapes the experts have, 256 rows a multiple of 128

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("%s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

int tile_of(const std::vector<int>& t, int rows) {
    int k = 0;
    while (k < (int) t.size() && rows > t[(size_t) k]) ++k;
    return k;
}

// one case: the rows of each expert, the classes' limits; returns the number of rows that differ from one launch,
// to the bit where `exact`, else by more than 8 float epsilons of the case's largest value
int run_case(const Quant& q, const std::vector<int>& rows, const std::vector<int>& limits, bool exact,
             std::mt19937& rng, int out_rows = kRows, int cols = kCols) {
    const int n = (int) rows.size();
    const size_t per = (size_t) out_rows * (cols / q.qk) * q.block;
    std::vector<uint8_t> w(per * n + 4096, 0);   // zero bytes behind the matrices (kTail)
    for (size_t i = 0; i < per * n; ++i) w[i] = (uint8_t) rng();
    for (size_t b = 0; b < per * n / q.block; ++b) {   // finite block scales: a half of 2^-3 where each type keeps its d
        uint8_t* p = &w[b * q.block];
        if (q.type == 11 || q.type == 14) { p[q.block - 2] = 0; p[q.block - 1] = 0x30; }
        else {
            p[0] = 0; p[1] = 0x30;
            if (q.type == 12 || q.type == 13) { p[2] = 0; p[3] = 0x30; }
        }
    }
    const int total = std::accumulate(rows.begin(), rows.end(), 0);
    std::vector<float> x((size_t) total * cols);
    for (auto& v : x) v = (float) ((rng() & 0xffff) / 32768.0 - 1.0);
    std::vector<int32_t> start(n + 1, 0);
    for (int e = 0; e < n; ++e) start[(size_t) e + 1] = start[(size_t) e] + rows[(size_t) e];

    // the reference: rows by expert, one launch
    std::vector<int32_t> iota(total), gather_ref(total);
    std::iota(iota.begin(), iota.end(), 0);
    std::iota(gather_ref.begin(), gather_ref.end(), 0);
    // the classes: rows class after class (an expert's rows in order), a bounds array for each class
    const int nclass = (int) limits.size() + 1;
    std::vector<int32_t> gather_cls, bounds((size_t) nclass * (n + 1)), most(nclass, 0);
    for (int k = 0, at = 0; k < nclass; ++k) {
        for (int e = 0; e < n; ++e) {
            bounds[(size_t) k * (n + 1) + e] = at;
            if (rows[(size_t) e] > 0 && tile_of(limits, rows[(size_t) e]) == k) {
                for (int r = 0; r < rows[(size_t) e]; ++r) gather_cls.push_back(start[(size_t) e] + r);
                at += rows[(size_t) e];
                most[(size_t) k] = std::max(most[(size_t) k], rows[(size_t) e]);
            }
        }
        bounds[(size_t) k * (n + 1) + n] = at;
    }
    std::vector<int32_t> ref_bounds(start.begin(), start.end());

    Mmq mmq;
    uint8_t* dw = (uint8_t*) upload(w);
    float* dx = upload(x);
    int32_t *d_iota = upload(iota), *d_gref = upload(gather_ref), *d_gcls = upload(gather_cls), *d_rb = upload(ref_bounds),
            *d_cb = upload(bounds);
    void* xq = nullptr;
    float *dref = nullptr, *dcls = nullptr;
    bool good = dw && dx && d_iota && d_gref && d_gcls && d_rb && d_cb;
    good = good && ok(cudaMalloc(&xq, Mmq::q8_bytes(total, cols)), "xq") &&
           ok(cudaMalloc((void**) &dref, (size_t) total * out_rows * 4), "ref") &&
           ok(cudaMalloc((void**) &dcls, (size_t) total * out_rows * 4), "cls");
    int bad = -1;
    if (good) {
        cudaMemset(dref, 0, (size_t) total * out_rows * 4);
        cudaMemset(dcls, 0, (size_t) total * out_rows * 4);
        const int maxrows = *std::max_element(rows.begin(), rows.end());
        mmq.quantize(dx, d_gref, xq, q.type, cols, cols, total, nullptr);
        mmq.mul(dw, q.type, out_rows, cols, per, n, xq, d_rb, d_iota, total, maxrows, dref, out_rows, nullptr);
        mmq.quantize(dx, d_gcls, xq, q.type, cols, cols, total, nullptr);
        for (int k = 0; k < nclass; ++k)
            if (most[(size_t) k] > 0)
                mmq.mul(dw, q.type, out_rows, cols, per, n, xq, d_cb + (size_t) k * (n + 1), d_iota, total, most[(size_t) k],
                        dcls, out_rows, nullptr);
        if (ok(cudaDeviceSynchronize(), "run")) {
            std::vector<float> a((size_t) total * out_rows), b((size_t) total * out_rows);
            cudaMemcpy(a.data(), dref, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), dcls, b.size() * 4, cudaMemcpyDeviceToHost);
            float big = 0;
            for (float v : a) big = std::max(big, std::fabs(v));
            const float tol = 8 * FLT_EPSILON * big;
            const auto near = [tol](float u, float v) { return std::fabs(u - v) <= tol; };
            bad = 0;   // row i of the class layout is row gather_cls[i] of the reference
            for (int i = 0; i < total; ++i) {
                const float* c = &b[(size_t) i * out_rows];
                const float* r = &a[(size_t) gather_cls[(size_t) i] * out_rows];
                const auto finite = [](float v) { return std::isfinite(v); };
                if (!std::all_of(c, c + out_rows, finite) || !std::all_of(r, r + out_rows, finite) ||
                    (exact ? std::memcmp(c, r, out_rows * 4) != 0 : !std::equal(c, c + out_rows, r, near))) ++bad;
            }
        }
    }
    for (void* p : {(void*) dw, (void*) dx, (void*) d_iota, (void*) d_gref, (void*) d_gcls, (void*) d_rb, (void*) d_cb, xq,
                    (void*) dref, (void*) dcls})
        if (p) cudaFree(p);
    return bad;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no card: skipped\n");
        return 77;
    }
    std::mt19937 rng(7);
    struct Dist { const char* name; std::vector<int> rows; };
    std::vector<Dist> dists;
    {   // many small experts and a few large, as a prompt batch's staged experts are
        std::vector<int> r;
        for (int e = 0; e < 96; ++e) r.push_back(1 + (int) (rng() % 16) * (rng() % 4 == 0 ? 6 : 1));
        for (int e : {5, 40, 77}) r[(size_t) e] = 200 + (int) (rng() % 300);
        dists.push_back({"skewed", r});
    }
    {   // empty experts between the others, one with a single row
        std::vector<int> r(40, 0);
        for (int e = 0; e < 40; e += 3) r[(size_t) e] = 1 + (int) (rng() % 90);
        r[1] = 1;
        dists.push_back({"with empty", r});
    }
    dists.push_back({"all small", std::vector<int>(33, 7)});
    dists.push_back({"one class edge", std::vector<int>{8, 9, 16, 17, 32, 33, 64, 65, 8, 16, 32, 64}});
    const std::vector<std::vector<int>> limits = {{8, 16, 32}, {16}, {4, 8, 12, 16, 24, 32, 48}, {64}};
    int failures = 0, cases = 0;
    for (const Quant& q : kQuants) {
        const bool exact = Mmq::split_exact(q.type, kRows);
        if (!exact) std::printf("%s: this card's MMQ uses stream-k, compared within rounding\n", q.name);
        for (const Dist& d : dists)
            for (const auto& l : limits) {
                const int bad = run_case(q, d.rows, l, exact, rng);
                ++cases;
                if (bad != 0) {
                    ++failures;
                    std::printf("FAIL %s %s, %zu classes: %d rows differ%s\n", q.name, d.name, l.size() + 1, bad,
                                exact ? "" : " by more than rounding");
                }
            }
    }
    // The selected Qwen3.6 UD GGUF: separate/combined gate-up and down shapes, including its
    // last layer and Q6_K down experts. One fixed row distribution crosses all 8/16/32 classes.
    struct Actual { Quant q; int rows, cols; };
    const Actual actual[] = {
        {{"IQ3_S", 21, 110}, 512, 2048}, {{"IQ3_S", 21, 110}, 1024, 2048},
        {{"IQ4_NL", 20, 18, 32}, 512, 2048}, {{"IQ4_NL", 20, 18, 32}, 1024, 2048},
        {{"IQ4_NL", 20, 18, 32}, 2048, 512},
        {{"Q3_K", 11, 110}, 512, 2048}, {{"Q3_K", 11, 110}, 1024, 2048},
        {{"Q4_K", 12, 144}, 2048, 512}, {{"Q6_K", 14, 210}, 2048, 512},
    };
    for (const auto& c : actual) {
        const bool fits = Mmq::fits(c.q.type, c.rows, c.cols);
        const bool exact = Mmq::split_exact(c.q.type, c.rows);
        const int bad = fits ? run_case(c.q, {0, 1, 7, 8, 9, 16, 17, 32, 33, 199, 8, 0},
                                       {8, 16, 32}, exact, rng, c.rows, c.cols) : -1;
        ++cases;
        failures += bad != 0;
        std::printf("%s actual %s %d x %d fits=%d split_exact=%d differing_rows=%d\n",
                    bad == 0 ? "PASS" : "FAIL", c.q.name, c.rows, c.cols, fits, exact, bad);
    }
    if (failures) {
        std::printf("FAIL: %d of %d cases\n", failures, cases);
        return 1;
    }
    std::printf("PASS cases=%d\n", cases);
    return 0;
}
