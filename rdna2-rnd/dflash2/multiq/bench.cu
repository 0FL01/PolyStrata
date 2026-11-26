// Compile the real W1 kernel and combine in this TU, with production HIP flags.
#include "../../../src/poly/inkling_attend.cu"

#include <array>
#include <cstring>
#include <random>
#include <vector>

namespace strata::poly::inkling {
namespace {
#include "shared.cuh"

void hip_check(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}

template <class T> struct Device {
    T* p{};
    size_t n;
    explicit Device(size_t count) : n(count) { hip_check(hipMalloc(&p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& host) : Device(host.size()) {
        hip_check(hipMemcpy(p, host.data(), n * sizeof(T), hipMemcpyHostToDevice));
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { hipFree(p); }
    void fill(int x, hipStream_t st) { hip_check(hipMemsetAsync(p, x, n * sizeof(T), st)); }
    std::vector<T> get() const {
        std::vector<T> host(n);
        hip_check(hipMemcpy(host.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
        return host;
    }
};

// Same distributions as inkling_attend_test.cu, sized for four independent rows.
std::vector<float> normal(size_t n, float sd, std::mt19937& rng) {
    std::vector<float> x(n);
    std::normal_distribution<float> d(0.f, sd);
    for (float& y : x) y = d(rng);
    return x;
}
std::vector<uint8_t> codes(size_t n, std::mt19937& rng) {
    std::vector<uint8_t> x(n);
    std::uniform_int_distribution<int> d(-127, 127);
    for (uint8_t& y : x) y = (uint8_t) (int8_t) d(rng);
    return x;
}
std::vector<uint16_t> scales(size_t n, std::mt19937& rng) {
    std::vector<uint16_t> x(n);
    std::uniform_real_distribution<float> d(0.002f, 0.03f);
    for (uint16_t& y : x) {
        const __half h = __float2half(d(rng));
        std::memcpy(&y, &h, sizeof(y));
    }
    return x;
}

enum Arm { serial, batch8, group4, lds2g32, lds1g64, arm_count };
constexpr std::array names{"serial", "batch8", "group4", "lds2g32", "lds1g64"};
constexpr int kMaxRows = 4, kMaxChunks = kDecodeBlocks / kKv;
constexpr size_t kMaxScratch = (size_t) kMaxRows * kMaxChunks * kKv * kChunkFloats;
constexpr size_t kQuants = kMaxRows * kQuery / 32;

struct Data {
    std::mt19937 rng{1234};
    Device<float> q{normal(kMaxRows * kQuery, 0.35f, rng)}, rel{normal(kMaxRows * kHeads * kRelative, 0.5f, rng)},
        table{normal(kExtent * kRelative, 0.5f, rng)}, sink{normal(kMaxRows * kHeads, 1.5f, rng)},
        gate{normal(kMaxRows * kQuery, 1.f, rng)};
    Device<uint8_t> kd, vd;
    Device<uint16_t> ks, vs;
    explicit Data(int slots)
        : kd(codes((size_t) slots * kKv * kHead, rng)), vd(codes(kd.n, rng)), ks(scales(kd.n / 32, rng)),
          vs(scales(kd.n / 32, rng)) {}
    Attend args(int pos, int start, int rows, float* out) const {
        return {q.p + start * kQuery, rel.p + start * kHeads * kRelative, table.p, sink.p + start * kHeads,
                gate.p + start * kQuery, {kd.p, ks.p}, {vd.p, vs.p}, pos + start, rows, 0, 0, out + start * kQuery};
    }
};

struct Result {
    Device<float> part{kMaxScratch}, out{kMaxRows * kQuery};
    Device<Q81> q8{kQuants};
};

struct Geometry {
    int keys, chunks;
    bool operator==(const Geometry&) const = default;
    size_t scratch() const { return (size_t) chunks * kKv * kChunkFloats; }
};
Geometry row_geometry(int span) {
    const int chunk = q8_chunk(span, kMaxChunks);  // ask the stock W1 sizing rule, not Wn's scratch cap
    return {chunk, div_up(span, chunk)};
}

template <int Groups, int Steps>
void launch_shared(const Attend& a, Geometry geom, float* tmp, hipStream_t st) {
    const dim3 grid(kKv / Groups, geom.chunks);
    const auto launch = [&]<int Rows>() {
        shared_q8_kernel<Rows, Groups, Steps><<<grid, 32 * Groups * Rows, 0, st>>>(
            a.q, a.rel, a.table, a.k, a.v, (int) a.pos, geom.keys, tmp);
    };
    if (a.n == 2) launch.template operator()<2>();
    else launch.template operator()<4>();
}

// Leaf includes its direct scatter into stock layout. Full adds the exact stock combine, sink,
// gate and packed Q8_1; no copy or transpose kernel. Serial reuses scratch when timing.
size_t run(Arm arm, bool full, bool keep_part, const Data& in, Result& out, int pos, int rows, hipStream_t st) {
    size_t at = 0;
    for (int r = 0; r < rows;) {
        const Geometry geom = row_geometry(pos + r + 1);
        int n = 1;
        if (arm != serial)
            while (r + n < rows && geom == row_geometry(pos + r + n + 1)) ++n;
        if (n == 3) n = 2;  // experimental shaders only instantiate W2 and W4
        const Attend a = in.args(pos, r, n, out.out.p);
        float* tmp = out.part.p + (keep_part ? at : 0);
        auto* q8 = out.q8.p + r * kQuery / 32;
        if (arm == serial && full) {
            attend_decode(a, tmp, st, q8);  // actual public W1 entry point, unchanged
        } else {
            if (n > 1 && arm == group4)
                decode_q8_kernel<4><<<dim3(2, n, geom.chunks), 128, 0, st>>>(
                    a.q, a.rel, a.table, a.k, a.v, a.pos, 0, 0, geom.keys, tmp);
            else if (n > 1 && arm == lds2g32) launch_shared<2, 8>(a, geom, tmp, st);
            else if (n > 1 && arm == lds1g64) launch_shared<1, 16>(a, geom, tmp, st);
            else
                decode_q8_kernel<><<<dim3(geom.chunks, n), kAttThreads, 0, st>>>(
                    a.q, a.rel, a.table, a.k, a.v, a.pos, 0, 0, geom.keys, tmp);
            if (full)
                decode_combine_kernel<<<dim3(kHeads, n), kHead, 0, st>>>(tmp, geom.chunks, a.sink, a.gate, a.out, q8);
            hip_check(hipGetLastError());
        }
        at += (size_t) n * geom.scratch();
        r += n;
    }
    return at;
}

template <class T>
bool equal(const char* field, const std::vector<T>& ref, const std::vector<T>& got, size_t n) {
    size_t diff = 0, first = n;
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&ref[i], &got[i], sizeof(T))) {
            first = std::min(first, i);
            ++diff;
        }
    }
    std::printf(" %s=%zu/%zu", field, diff, n);
    if (diff) {
        std::printf(" first_%s=%zu:", field, first);
        const auto bits = [](const T& x) {
            for (size_t i = 0; i < sizeof(T); ++i) std::printf("%02x", reinterpret_cast<const uint8_t*>(&x)[i]);
        };
        bits(ref[first]);
        std::putchar('/');
        bits(got[first]);
    }
    return diff == 0;
}

void validate(const Data& in, Result& ref, Result& got, int pos, int rows, hipStream_t st) {
    ref.part.fill(0xff, st); ref.out.fill(0xff, st); ref.q8.fill(0xff, st);
    const size_t n = run(serial, true, true, in, ref, pos, rows, st);
    hip_check(hipStreamSynchronize(st));
    const auto rp = ref.part.get(), rf = ref.out.get();
    const auto rq = ref.q8.get();
    const auto bytes = [](const std::vector<Q81>& b) {
        auto* p = reinterpret_cast<const uint8_t*>(b.data());
        return std::vector<uint8_t>(p, p + b.size() * sizeof(Q81));
    };
    bool all = true;
    for (int mode = batch8; mode < arm_count; ++mode) {
        got.part.fill(0xba, st); got.out.fill(0xba, st); got.q8.fill(0xba, st);
        if (run((Arm) mode, true, true, in, got, pos, rows, st) != n) throw std::logic_error("scratch size");
        hip_check(hipStreamSynchronize(st));
        std::printf("check pos=%d width=%d arm=%s", pos, rows, names[mode]);
        all &= equal("part_f32", rp, got.part.get(), n);
        all &= equal("out_f32", rf, got.out.get(), rows * kQuery);
        all &= equal("q8_bytes", bytes(rq), bytes(got.q8.get()), rows * kQuery / 32 * sizeof(Q81));
        std::putchar('\n');
    }
    if (!all) throw std::runtime_error("not bitwise equal to W1");
}

// A read, not only a fill, of a disjoint region > exposed Infinity Cache. Outside the timed interval.
__global__ void evict(const uint4* src, uint32_t* dst, size_t n) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t s = 0;
    for (size_t i = tid; i < n; i += gridDim.x * blockDim.x) {
        const uint4 v = src[i];
        s += v.x + v.y + v.z + v.w;
    }
    dst[tid] = s;
}

void timings(const Data& in, Result& out, int pos, int rows, int samples, hipStream_t st,
             Device<uint4>& flush, Device<uint32_t>& trash) {
    constexpr int arms = 2 * arm_count;
    std::array<hipGraphExec_t, arms> exec{};
    for (int i = 0; i < arms; ++i) {
        hipGraph_t graph;
        hip_check(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
        run((Arm) (i % arm_count), i >= arm_count, false, in, out, pos, rows, st);
        hip_check(hipStreamEndCapture(st, &graph));
        hip_check(hipGraphInstantiateWithFlags(&exec[i], graph, 0));
        hip_check(hipGraphDestroy(graph));
        for (int j = 0; j < 3; ++j) hip_check(hipGraphLaunch(exec[i], st));
    }
    hip_check(hipStreamSynchronize(st));
    hipEvent_t start, end;
    hip_check(hipEventCreate(&start));
    hip_check(hipEventCreate(&end));
    for (bool cold : {false, true}) {
        std::array<std::vector<float>, arms> times;
        for (int s = 0; s < samples; ++s) {
            for (int j = 0; j < arms; ++j) {
                const int i = (s + (s % 2 ? arms - 1 - j : j)) % arms;
                if (cold) evict<<<256, 256, 0, st>>>(flush.p, trash.p, flush.n);
                else for (int j2 = 0; j2 < 2; ++j2) hip_check(hipGraphLaunch(exec[i], st));
                hip_check(hipEventRecord(start, st));
                hip_check(hipGraphLaunch(exec[i], st));
                hip_check(hipEventRecord(end, st));
                hip_check(hipEventSynchronize(end));
                float ms = 0;
                hip_check(hipEventElapsedTime(&ms, start, end));
                times[i].push_back(ms);
            }
        }
        for (int i = 0; i < arms; ++i) {
            auto& t = times[i];
            std::sort(t.begin(), t.end());
            std::printf("time pos=%d width=%d cache=%s path=%s arm=%s p25=%.5f median=%.5f p75=%.5f ms raw=",
                        pos, rows, cold ? "read-flushed" : "warm", i >= arm_count ? "full" : "leaf",
                        names[i % arm_count], t[t.size() / 4], t[t.size() / 2], t[3 * t.size() / 4]);
            for (float x : t) std::printf(" %.5f", x);
            std::putchar('\n');
        }
    }
    hip_check(hipEventDestroy(start));
    hip_check(hipEventDestroy(end));
    for (auto e : exec) hip_check(hipGraphExecDestroy(e));
}

int bench(int samples) {
    hipDeviceProp_t gpu{};
    hip_check(hipGetDeviceProperties(&gpu, 0));
    if (std::string(gpu.gcnArchName).rfind("gfx1031", 0) != 0) throw std::runtime_error("gfx1031 required");
    std::printf("device=%s %s wave=%d samples=%d\n", gpu.name, gpu.gcnArchName, gpu.warpSize, samples);
    int runtime = 0;
    hip_check(hipRuntimeGetVersion(&runtime));
    std::printf("hip_runtime=%d\n", runtime);
    hipStream_t st;
    hip_check(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    constexpr int kLong = 199069;
    Data in(kLong + kMaxRows);
    Result ref, got;
    Device<uint4> flush{((size_t) 256 << 20) / sizeof(uint4)};
    Device<uint32_t> trash{256 * 256};
    flush.fill(0x62, st);
    // Exact tails, a change in chunk count, and a change in chunk size (with the program's 160-chunk setting).
    for (int pos : {0, 30, 65726, 66558, 65535, kLong}) {
        for (int rows : {2, 4}) {
            std::printf("geom pos=%d width=%d", pos, rows);
            for (int r = 0; r < rows; ++r) {
                const auto [keys, n] = row_geometry(pos + r + 1);
                std::printf(" %d:%d", keys, n);
            }
            std::putchar('\n');
            validate(in, ref, got, pos, rows, st);
        }
    }
    if (samples)
        for (int pos : {65535, kLong})
            for (int rows : {2, 4}) timings(in, got, pos, rows, samples, st, flush, trash);
    hip_check(hipStreamDestroy(st));
    return 0;
}
}  // namespace
}  // namespace strata::poly::inkling

int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const int samples = argc > 1 ? std::stoi(argv[1]) : 21;
    if (argc > 2 || samples < 0) throw std::runtime_error("usage: bench [non-negative samples]");
    return strata::poly::inkling::bench(samples);
} catch (const std::exception& e) {
    std::fprintf(stderr, "multiq: %s\n", e.what());
    return 1;
}
