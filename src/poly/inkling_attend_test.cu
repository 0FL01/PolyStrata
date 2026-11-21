// src/poly/inkling_attend_test.cu - attend_flash() and attend_decode() against attend_plain() for Inkling-2-Mini's
// attention: a window
// layer's ring before and after it wraps, a full layer, a batch's last partial block of tokens and FP16 caches,
// with the plain path's scratch as the engine sizes it at 200,000 tokens of context. The two read the same
// dequantized values and differ only in the order of their FP32 sums. Also times them on a 512-token batch and one
// token's decode.
//
// usage: poly_inkling_attend_test   (exit 0: every case within 1e-4 of the plain path's largest output, no NaN;
//                                    77: no card)
#include "inkling_attend.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly::inkling;

constexpr int kHeads = 24, kHead = 128, kKv = 8, kRelative = 16, kWindow = 1024, kContext = 200000;

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> normal(size_t n, float sd, std::mt19937& rng) {
    std::normal_distribution<float> d(0.f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

struct Case {
    const char* name;
    int64_t pos;
    int n;
    bool swa, q8;
    float sink_add = 0.f;   // a sink far above the scores: each key's share is below the rounding of 1
    bool spike = false;     // key 0 about 20 above every other key: the others' shares are below its rounding
    bool dec = false;       // attend_decode() in place of attend_flash()
};

// rows of cache the case reads: a window layer's ring is the engine's (window + batch - 1), a full layer's
// every position up to the batch's end
int64_t cache_rows(const Case& c) { return c.swa ? kWindow + 512 - 1 : c.pos + c.n; }

float* chunks = nullptr;   // attend_decode()'s scratch

bool run(const Case& c, float* scores, float* partial, int tiles, bool timed) {
    std::mt19937 rng(1234 + (unsigned) c.pos + 7u * (unsigned) c.n);
    const int64_t rows = cache_rows(c);
    const size_t width = (size_t) kKv * kHead;
    std::vector<uint8_t> kd, vd;
    std::vector<uint16_t> ksc, vsc;
    auto fill = [&](std::vector<uint8_t>& data, std::vector<uint16_t>& scales) {
        if (c.q8) {
            std::uniform_int_distribution<int> code(-127, 127);
            std::uniform_real_distribution<float> sc(0.002f, 0.03f);
            data.resize(rows * width);
            for (auto& x : data) x = (uint8_t) (int8_t) code(rng);
            scales.resize(rows * width / 32);
            for (auto& s : scales) {
                const __half h = __float2half(sc(rng));
                s = *reinterpret_cast<const uint16_t*>(&h);
            }
        } else {
            const auto f = normal(rows * width, 0.6f, rng);
            data.resize(rows * width * 2);
            for (size_t i = 0; i < f.size(); ++i) {
                const __half h = __float2half(f[i]);
                std::memcpy(data.data() + 2 * i, &h, 2);
            }
        }
    };
    fill(kd, ksc);
    fill(vd, vsc);
    if (c.spike) {   // positive queries and key 0 at the largest code with a large scale, in every K/V head
        for (size_t i = 0; i < width; ++i) kd[i] = 127;
        for (size_t i = 0; i < width / 32; ++i) {
            const __half h = __float2half(0.05f);
            ksc[i] = *reinterpret_cast<const uint16_t*>(&h);
        }
    }
    // queries a little larger than a clipped head; relative terms and the table in the range the model's are
    auto q = normal((size_t) c.n * kHeads * kHead, 0.35f, rng);
    if (c.spike)
        for (auto& x : q) x = std::fabs(x);
    const auto rel = normal((size_t) c.n * kHeads * kRelative, 0.5f, rng);
    const auto table = normal((size_t) kWindow * kRelative, 0.5f, rng);
    auto sink = normal((size_t) c.n * kHeads, 1.5f, rng);
    for (auto& x : sink) x += c.sink_add;
    const auto gate = normal((size_t) c.n * kHeads * kHead, 1.f, rng);

    Attend a;
    a.q = upload(q);
    a.rel = upload(rel);
    a.table = upload(table);
    a.sink = upload(sink);
    a.gate = upload(gate);
    a.k = {upload(kd), c.q8 ? upload(ksc) : nullptr};
    a.v = {upload(vd), c.q8 ? upload(vsc) : nullptr};
    a.pos = c.pos;
    a.n = c.n;
    a.window = c.swa ? kWindow : 0;
    a.ring = c.swa ? (int) rows : 0;
    const size_t out_n = (size_t) c.n * kHeads * kHead;
    float *plain = nullptr, *flash = nullptr;
    cudaMalloc(&plain, out_n * sizeof(float));
    cudaMalloc(&flash, out_n * sizeof(float));
    cudaMemset(flash, 0xff, out_n * sizeof(float));   // NaN wherever a row is left unwritten

    a.out = plain;
    attend_plain(a, scores, partial, tiles, nullptr);
    a.out = flash;
    const auto other = [&] {
        if (c.dec) attend_decode(a, chunks, nullptr);
        else attend_flash(a, nullptr);
    };
    other();
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("%s: %s\n", c.name, cudaGetErrorString(e));
        return false;
    }
    std::vector<float> hp(out_n), hf(out_n);
    cudaMemcpy(hp.data(), plain, out_n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hf.data(), flash, out_n * sizeof(float), cudaMemcpyDeviceToHost);
    double top = 0, diff = 0;
    bool finite = true;
    for (size_t i = 0; i < out_n; ++i) {
        finite = finite && std::isfinite(hp[i]) && std::isfinite(hf[i]);
        top = std::max(top, (double) std::fabs(hp[i]));
        diff = std::max(diff, (double) std::fabs(hp[i] - hf[i]));
    }
    const double relative = diff / std::max(top, 1e-30);
    const bool ok = finite && relative <= 1e-4;
    std::printf("%-26s pos %6lld n %4d: largest %.4g, difference %.3g (%.2e of largest)%s\n", c.name,
                (long long) c.pos, c.n, top, diff, relative, ok ? "" : finite ? "  FAIL" : "  NaN FAIL");

    if (timed) {
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        constexpr int kRuns = 10;
        float ms_plain = 0, ms_flash = 0;
        a.out = plain;
        cudaEventRecord(t0);
        for (int i = 0; i < kRuns; ++i) attend_plain(a, scores, partial, tiles, nullptr);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_plain, t0, t1);
        a.out = flash;
        cudaEventRecord(t0);
        for (int i = 0; i < kRuns; ++i) other();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_flash, t0, t1);
        std::printf("%-26s plain %.3f ms, %s %.3f ms a call\n", "", ms_plain / kRuns, c.dec ? "decode" : "flash",
                    ms_flash / kRuns);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    for (const void* p : {(const void*) a.q, (const void*) a.rel, (const void*) a.table, (const void*) a.sink,
                          (const void*) a.gate, (const void*) a.k.data, (const void*) a.k.scales,
                          (const void*) a.v.data, (const void*) a.v.scales, (const void*) plain, (const void*) flash})
        if (p) cudaFree(const_cast<void*>(p));
    return ok;
}

}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::puts("no card");
        return 77;
    }
    // the engine's scratch at 200,000 tokens of context and batches of 512 (Inkling2::assign_scratch)
    const int value_tiles = (kContext + 127) / 128, target = (1 << 20) / (kHeads * 128);
    const int tiles = std::max(value_tiles, std::min(target, 512 * value_tiles));
    float *scores = nullptr, *partial = nullptr;
    cudaMalloc(&scores, plain_score_floats(tiles) * sizeof(float));
    cudaMalloc(&partial, plain_partial_floats(tiles) * sizeof(float));
    cudaMalloc(&chunks, decode_floats() * sizeof(float));
    const Case cases[] = {
        {"window, first batch", 0, 512, true, true},
        {"window, inside the ring", 512, 512, true, true},
        {"window, ring wrapped", 3001, 512, true, true},
        {"window, partial block", 1500, 77, true, true},
        {"window, 16 tokens", 4090, 16, true, true},
        {"full, first batch", 0, 512, false, true},
        {"full, past the bias", 2000, 512, false, true},
        {"full, partial block", 5003, 45, false, true},
        {"window, fp16", 3001, 512, true, false},
        {"full, fp16", 1200, 300, false, false},
        {"full, sink above 60K keys", 60000, 64, false, true, 16.f},
        {"window, sink above", 3001, 64, true, true, 16.f},
        {"window, no sink, masked block", 1023, 32, true, true, -INFINITY},
        {"full, one key above 190K", 190000, 32, false, true, 0.f, true},
        {"decode: window, first", 0, 1, true, true, 0.f, false, true},
        {"decode: window, 150", 150, 1, true, true, 0.f, false, true},
        {"decode: window, wrapped", 3001, 1, true, true, 0.f, false, true},
        {"decode: window, 7 rows", 4090, 7, true, true, 0.f, false, true},
        {"decode: window, 31 rows", 1000, 31, true, true, 0.f, false, true},
        {"decode: full, first", 0, 1, false, true, 0.f, false, true},
        {"decode: full, 777", 777, 1, false, true, 0.f, false, true},
        {"decode: full, 16K", 16384, 1, false, true, 0.f, false, true},
        {"decode: full, 31 rows", 5003, 31, false, true, 0.f, false, true},
        {"decode: window, fp16", 3001, 3, true, false, 0.f, false, true},
        {"decode: full, fp16", 1200, 1, false, false, 0.f, false, true},
        {"decode: full, sink above", 60000, 1, false, true, 16.f, false, true},
        {"decode: window, no sink", 1023, 2, true, true, -INFINITY, false, true},
        {"decode: one key above 190K", 190000, 1, false, true, 0.f, true, true},
    };
    bool ok = true;
    for (const Case& c : cases) ok = run(c, scores, partial, tiles, false) && ok;
    ok = run({"timed: window", 8192, 512, true, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: full at 4K", 4096, 512, false, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: full at 32K", 32768, 512, false, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: decode window", 8192, 1, true, true, 0.f, false, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: decode full at 16K", 16384, 1, false, true, 0.f, false, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: decode full at 190K", 190000, 1, false, true, 0.f, false, true}, scores, partial, tiles, true) &&
         ok;
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
