// src/poly/inkling_attend_test.cu - attend_flash() and attend_decode() against attend_plain() for Inkling-2-Mini's
// attention: a window
// layer's ring before and after it wraps, a full layer, a batch's last partial block of tokens and FP16 caches,
// with the plain path's scratch as the engine sizes it at 200,000 tokens of context. The two read the same
// dequantized values and differ only in the order of their FP32 sums. Also times them on a 512-token batch and one
// token's decode.
//
// usage: poly_inkling_attend_test   (exit 0: every case within 1e-4 of the plain path's largest output, no NaN;
//                                    77: no card)
// With STRATA_INKLING_FLASH_H2=1 (and LANE_PROB=1) the flash cases on a Q8_0 cache take the half-product kernel:
// its operands are rounded to 11 bits, so those cases are held to 1e-3 instead.
#include "inkling_attend.hpp"
#include "inkling_kv_policy.hpp"

#include <cuda_fp16.h>
#include <cstdlib>

#include <string>
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
    static const bool half = [] {   // as attend_flash() decides it: both switches, on gfx1031
        const char* v = std::getenv("STRATA_INKLING_FLASH_H2");
        const char* l = std::getenv("STRATA_INKLING_LANE_PROB");
        cudaDeviceProp prop{};
        int device = 0;
        const bool gfx1031 = cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
                             std::string(prop.gcnArchName).rfind("gfx1031", 0) == 0;
        return v && std::string(v) == "1" && l && std::string(l) == "1" && gfx1031;
    }();
    const double limit = half && !c.dec && a.k.scales ? 1e-3 : 1e-4;
    const bool ok = finite && relative <= limit;
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
        if (c.dec) {   // with the K/V out of the caches first, as a decode visit meets them (the card's 96 MB L3)
            void* flush = nullptr;
            constexpr size_t kFlush = size_t(256) << 20;
            cudaMalloc(&flush, kFlush);
            float cold = 0;
            for (int i = 0; i < kRuns; ++i) {
                cudaMemsetAsync(flush, i, kFlush);
                cudaEventRecord(t0);
                other();
                cudaEventRecord(t1);
                cudaEventSynchronize(t1);
                float ms = 0;
                cudaEventElapsedTime(&ms, t0, t1);
                cold += ms;
            }
            std::printf("%-26s decode, cold caches %.3f ms a call\n", "", cold / kRuns);
            cudaFree(flush);
        }
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    for (const void* p : {(const void*) a.q, (const void*) a.rel, (const void*) a.table, (const void*) a.sink,
                          (const void*) a.gate, (const void*) a.k.data, (const void*) a.k.scales,
                          (const void*) a.v.data, (const void*) a.v.scales, (const void*) plain, (const void*) flash})
        if (p) cudaFree(const_cast<void*>(p));
    return ok;
}

// Independent K/V marker values and a non-page-multiple hot ring. Compare a mixed call to an expanded Q8 V
// view with the same scales/codes, not to the different native V values before refit. Using 33 query rows
// tests causal groups and the last partial group on a realistic tool-return-sized suffix.
bool run_v4(float* scores, float* partial, int tiles) {
    constexpr int pos = 4285, n = 33, span = pos + n, sealed = v4_seal(pos), ring = kKvRecent + 2 * kKvPage - 1 + n;
    const int64_t source_first = sealed - kKvPage;
    static_assert(span - (sealed - kKvPage) <= ring && sealed < ring && ring < span);
    std::mt19937 rng(1338);
    auto q = normal(n * kHeads * kHead, .35f, rng);
    auto rel = normal(n * kHeads * kRelative, .12f, rng), table = normal(1024 * kRelative, .17f, rng);
    auto sink = normal(n * kHeads, 1.f, rng), gate = normal(n * kHeads * kHead, 1.f, rng);
    std::vector<uint8_t> key(span * kKvColumns), full(span * kKvColumns), hot(ring * kKvColumns);
    std::vector<uint8_t> cold(sealed * kV4Columns);
    std::vector<uint16_t> key_sc(span * kKvScales, 0x2000), full_sc(span * kKvScales, 0x2000),
                          hot_sc(ring * kKvScales, 0x2000), cold_sc(sealed * kKvScales, 0x2000);
    for (size_t i = 0; i < key.size(); ++i) key[i] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
    for (int row = 0; row < span; ++row) {
        if (row >= source_first && row < sealed) {
            // Use both the exact case and a real scale from the first 128 natural seal rows: for 0x2658 and
            // amax=127, FP32-then-FP16 product gives 0x3732, versus 0x3731 with a direct-to-FP16 product.
            const int hot_row = row % ring;
            for (int b = 0; b < kKvScales; ++b) {
                const int amax = b % 2 ? 127 : 112;
                hot_sc[hot_row * kKvScales + b] = b % 2 ? 0x2658 : 0x2000;
                full_sc[row * kKvScales + b] = b % 2 ? 0x3732 : 0x3000;
                for (int d = 0; d < 32; ++d) {
                    const int v = d == 0 ? -amax : (int) (rng() % (2 * amax + 1)) - amax;
                    hot[hot_row * kKvColumns + 32 * b + d] = (uint8_t) (int8_t) v;
                    const int code = (int) std::nearbyint(float(v * 7) / float(amax));
                    full[row * kKvColumns + 32 * b + d] = (uint8_t) (int8_t) code;
                }
            }
        } else if (row < sealed) {
            for (int d = 0; d < kKvColumns; d += 2) {
                const int x = (int) (rng() % 15) - 7, y = (int) (rng() % 15) - 7;
                cold[row * kV4Columns + d / 2] = (uint8_t) ((x & 15) | (y & 15) << 4);
                full[row * kKvColumns + d] = (uint8_t) (int8_t) x;
                full[row * kKvColumns + d + 1] = (uint8_t) (int8_t) y;
            }
        } else {
            const int hot_row = row % ring;
            for (int d = 0; d < kKvColumns; ++d) {
                const uint8_t code = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
                hot[hot_row * kKvColumns + d] = full[row * kKvColumns + d] = code;
            }
        }
    }
    auto* packed = upload(cold);
    auto* packed_sc = upload(cold_sc);
    const KvView hot_gpu{upload(hot), upload(hot_sc)};
    const KvView full_gpu{upload(full), upload(full_sc)};
    const V4View v4{{packed, packed_sc}, sealed, ring};
    seal_v4(hot_gpu, ring, packed, packed_sc, source_first, sealed, nullptr);
    float *expected = nullptr, *actual = nullptr;
    cudaMalloc(&expected, q.size() * sizeof(float));
    cudaMalloc(&actual, q.size() * sizeof(float));
    cudaMemset(actual, 0xff, q.size() * sizeof(float));
    Attend a{upload(q), upload(rel), upload(table), upload(sink), upload(gate), {upload(key), upload(key_sc)},
             full_gpu, pos, n, 0, 0, expected};
    attend_plain(a, scores, partial, tiles, nullptr);
    a.v = hot_gpu;
    a.out = actual;
    attend_v4(a, v4, chunks, nullptr);
    auto error = cudaDeviceSynchronize();
    std::vector<float> p(q.size()), v(q.size());
    if (error == cudaSuccess) error = cudaMemcpy(p.data(), expected, p.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) error = cudaMemcpy(v.data(), actual, v.size() * sizeof(float), cudaMemcpyDeviceToHost);
    const int first_scale = source_first * kKvScales, last_scale = sealed * kKvScales;
    std::vector<uint16_t> sealed_sc(last_scale - first_scale);
    if (error == cudaSuccess)
        error = cudaMemcpy(sealed_sc.data(), packed_sc + first_scale, sealed_sc.size() * sizeof(uint16_t),
                           cudaMemcpyDeviceToHost);
    float top = 0.f, delta = 0.f;
    const bool exact_scales = std::equal(sealed_sc.begin(), sealed_sc.end(), full_sc.begin() + first_scale);
    std::printf("V4 seal canonical FP16 scales: %s\n", exact_scales ? "exact" : "FAIL");
    bool ok = error == cudaSuccess && exact_scales;
    for (size_t i = 0; i < p.size(); ++i) {
        ok = ok && std::isfinite(p[i]) && std::isfinite(v[i]);
        top = std::max(top, std::fabs(p[i]));
        delta = std::max(delta, std::fabs(p[i] - v[i]));
    }
    ok = ok && top > 0.f && delta / top <= 1e-4f;
    std::printf("V4 seal and wrapped hot suffix: delta %.4g, largest %.4g%s\n", delta, top, ok ? "" : "  FAIL");
    for (const void* p : {(const void*) a.q, (const void*) a.rel, (const void*) a.table, (const void*) a.sink,
                          (const void*) a.gate, (const void*) a.k.data, (const void*) a.k.scales, (const void*) expected,
                          (const void*) actual, (const void*) hot_gpu.data, (const void*) hot_gpu.scales,
                          (const void*) full_gpu.data, (const void*) full_gpu.scales, (const void*) packed,
                          (const void*) packed_sc})
        cudaFree(const_cast<void*>(p));
    return ok;
}

// One selected W1 row, at the first span where one old page can be dropped. Two old pages
// tie for last place in each group: keep the lower ID. Seal by the input position, so the last
// eligible old page is still hot, and the subsequent native Q8 tail wraps the ring.
bool run_k256() {
#if defined(STRATA_USE_HIP)
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    if (std::strncmp(prop.gcnArchName, "gfx1031", 7) != 0) return true;
    constexpr int page = 128, top = 256, old = 258, span = old * page + 2048, pos = span - 1;
    constexpr int sealed = (old - 1) * page, ring = 4223, width = 1024, heads = 24, head = 128;
    static_assert(sealed / ring < pos / ring);
    const auto omit = [](int g) { return 60 + 5 * g; };
    const auto key = [&](int r, int g) {
        const int p = r / page;
        return (p == 0 ? 23 : p == 12 + 3 * g || p == omit(g) ? -13 : (p * 7 + g) % 31) +
               (r % page < 64 ? -2 : 1);
    };
    const auto val = [&](int r, int g, int d) {
        return r < sealed ? (r / 7 + r / page + 3 * g + 2 * d) % 15 - 7
                          : (r / 3 + r / page + 11 * g + 3 * d) % 97 - 48;
    };
    std::vector<float> q(heads * head), rel(heads * 16), table(1024 * 16), sink(heads), gate(heads * head), out(q.size());
    std::vector<uint8_t> kd(span * width), cold(sealed * width / 2), hot(ring * width, 126);
    std::vector<uint16_t> ks(span * width / 32, 0x3000), cs(sealed * width / 32, 0x3000), hs(ring * width / 32, 0x3400);
    for (int h = 0; h < heads; ++h) {
        q[h * head + 17 * (h / 3) % head] = .5f + .5f * (h % 3);
        rel[h * 16] = .125f * (h % 5 - 2);
        sink[h] = 8.f + .0625f * h;
        for (int d = 0; d < head; ++d) gate[h * head + d] = .0625f * ((h + 3 * d) % 13 - 6);
    }
    for (int r = 0; r < 1024; ++r) table[r * 16] = .0625f * (r % 23 - 11);
    for (int r = 0; r < span; ++r)
        for (int g = 0; g < 8; ++g) {
            kd[r * width + g * head + 17 * g % head] = (uint8_t) key(r, g);
            const int v = (r < sealed ? r : r % ring) * width + g * head;
            if (r < sealed) for (int d = 0; d < head; d += 2)
                cold[(v + d) / 2] = (uint8_t) ((val(r, g, d) & 15) | ((val(r, g, d + 1) & 15) << 4));
            else for (int d = 0; d < head; ++d) hot[v + d] = (uint8_t) val(r, g, d);
        }
    Attend a{upload(q), upload(rel), upload(table), upload(sink), upload(gate), {upload(kd), upload(ks)},
             {upload(hot), upload(hs)}, pos - 1, 1, 0, 0, nullptr};
    const V4View v4{{upload(cold), upload(cs)}, sealed, ring};
    float* scratch = nullptr;
    uint32_t* index = nullptr;
    const auto scratch_n = decode_floats() + k256_scratch_floats();
    if (cudaMalloc(&scratch, scratch_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&index, (span / page) * width * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&a.out, out.size() * sizeof(float)) != cudaSuccess) return false;
    k256_build(a.k, index, 0, span / page, nullptr);
    bool ok = !attend_k256(a, v4, index, scratch, nullptr, nullptr);
    a.pos = pos;
    ok = attend_k256(a, v4, index, scratch, nullptr, nullptr) && ok;
    std::vector<int> picked(8 * top);
    ok = cudaMemcpy(picked.data(), scratch + scratch_n - picked.size(), picked.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess && ok;
    ok = cudaMemcpy(out.data(), a.out, out.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess && ok;
    double max_diff = 0;
    for (int g = 0; g < 8; ++g) {
        int j = 0;
        for (int p = 1; p < old; ++p) if (p != omit(g)) ok = (picked[g * top + j++] == p) && ok;
        for (int h = 3 * g; h < 3 * g + 3; ++h) {
            double num[head]{}, den = std::exp((double) sink[h]);
            for (int r = 0; r < span; ++r) {
                if (r / page == omit(g)) continue;
                double score = key(r, g) * .125 * q[h * head + 17 * g % head] / std::sqrt(128.);
                const int age = pos - r;
                if (age < 1024) score += (double) rel[h * 16] * table[age * 16];
                const double w = std::exp(score);
                den += w;
                for (int d = 0; d < head; ++d) num[d] += w * val(r, g, d) * (r < sealed ? .125 : .25);
            }
            for (int d = 0; d < head; ++d) {
                const int i = h * head + d;
                const double y = num[d] / den / (1 + std::exp(-(double) gate[i]));
                ok = std::isfinite(out[i]) && ok;
                max_diff = std::max(max_diff, std::abs(out[i] - y));
            }
        }
    }
    for (const void* p : {(const void*) a.q, (const void*) a.rel, (const void*) a.table, (const void*) a.sink,
                          (const void*) a.gate, (const void*) a.k.data, (const void*) a.k.scales, (const void*) a.v.data,
                          (const void*) a.v.scales, (const void*) v4.cold.data, (const void*) v4.cold.scales,
                          (const void*) a.out, (const void*) scratch, (const void*) index}) (void) cudaFree(const_cast<void*>(p));
    ok = max_diff < 3e-5 && ok;
    std::printf("K256 boundary, tie and wrapped mixed output: max abs %.5g%s\n", max_diff, ok ? "" : "  FAIL");
    return ok;
#else
    return true;
#endif
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
        {"decode: full, 31 rows 190K", 190000, 31, false, true, 0.f, false, true},
        {"decode: 31 rows, spike 190K", 190000, 31, false, true, 0.f, true, true},
    };
    bool ok = run_k256();
    ok = run_v4(scores, partial, tiles) && ok;
    for (const Case& c : cases) ok = run(c, scores, partial, tiles, false) && ok;
    ok = run({"timed: window", 8192, 512, true, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: full at 4K", 4096, 512, false, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: full at 32K", 32768, 512, false, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: decode at 200", 200, 1, true, true, 0.f, false, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: decode window", 8192, 1, true, true, 0.f, false, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: decode full at 16K", 16384, 1, false, true, 0.f, false, true}, scores, partial, tiles, true) && ok;
    ok = run({"timed: decode full at 190K", 190000, 1, false, true, 0.f, false, true}, scores, partial, tiles, true) &&
         ok;
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
