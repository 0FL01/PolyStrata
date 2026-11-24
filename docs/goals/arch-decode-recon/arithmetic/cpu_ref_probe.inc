// Included by parity.cpp to reuse the exact fixture loader and ggml checks.
namespace {
constexpr int xb = width / qk, hb = ff / qk;
constexpr size_t gate_up_bytes = 2 * ff * xb * 18, down_bytes = width * hb * 18;

std::vector<Q80> model_cpu_ref_quant(const std::vector<float>& x) {
    std::vector<Q80> out(x.size() / qk);
    for (size_t b = 0; b < out.size(); ++b) {
        float amax = 0.f;
        for (int i = 0; i < qk; ++i) amax = std::max(amax, std::abs(x[b * qk + i]));
        const float inv = amax != 0.f ? 127.f / amax : 0.f;
        out[b].d = hbits(amax / 127.f);
        for (int i = 0; i < qk; ++i)
            out[b].q[i] = static_cast<int8_t>(std::clamp(std::nearbyint(x[b * qk + i] * inv), -128.f, 127.f));
    }
    return out;
}

std::vector<float> model_cpu_ref_dot(const uint8_t* w, const std::vector<Q80>& q, int rows) {
    std::vector<float> out(rows);
    for (int row = 0; row < rows; ++row) {
        float v[8]{};
        for (size_t b = 0; b < q.size(); ++b) {
            const uint8_t* block = w + (row * q.size() + b) * 18;
            const float d = fhalf(half_at(block)) * fhalf(q[b].d);
            for (int j = 0; j < 8; ++j) {
                int sum = 0;
                for (int k = 4 * j; k < 4 * j + 4; ++k) {
                    const int nib = (block[2 + k % 16] >> (k >= 16 ? 4 : 0)) & 15;
                    sum += (nib - 8) * q[b].q[k];
                }
                v[j] = std::fma(d, static_cast<float>(sum), v[j]);
            }
        }
        const float a = v[4] + v[0], b = v[5] + v[1], c = v[6] + v[2], d = v[7] + v[3];
        out[row] = (a + c) + (b + d);
    }
    return out;
}

size_t compare_gate_up(const char* stage, const Stages<Q80>& a, const std::vector<float>& gu) {
    std::printf("%s\n", stage);
    size_t issues = compare_fp("gate FP32", a.g, {gu.begin(), gu.begin() + ff});
    issues += compare_fp("up FP32", a.u, {gu.begin() + ff, gu.end()});
    return issues;
}

void cpu_ref_model_check(const Model& m, const Stages<Q80>& a) {
    std::puts("production AVX2 vs scalar CPU model of candidate (not a GPU result)");
    auto q = model_cpu_ref_quant(m.x);
    size_t issues = compare_q("activation Q8_0", a.xq, q);
    issues += compare_gate_up("scalar candidate gate/up", a, model_cpu_ref_dot(m.gate, q, 2 * ff));
    q = model_cpu_ref_quant(a.h);
    issues += compare_q("CPU SiTU -> Q8_0", a.hq, q);
    issues += compare_fp("CPU SiTU -> down FP32", a.d, model_cpu_ref_dot(m.down, q, width));
    require(!issues, "scalar model disagrees with production AVX2");
}

void label_cpu_ref_fixture(const Model& m) {
    std::printf("fixture=%s layer=%d expert=%d route_weight_bits=0x%08x (not applied)\n",
                m.captured_q.empty() ? "model-derived proxy" : "captured serial decode", m.layer, m.expert, bits(m.weight));
    std::printf("activation=%s\n", m.source.c_str());
}

#if defined(CP_HIP)
bool cpu_ref_hip_check(const Model& m, const Stages<Q80>& a) {
    Stream stream;
    const auto st = stream.s;
    Device<float> input(width), gu(2 * ff), h(ff), d(width);
    Device<Q80> xq(xb), hq(hb);
    Device<uint8_t> wg(gate_up_bytes), wd(down_bytes);
    wg.put(m.gate, gate_up_bytes);
    wd.put(m.down, down_bytes);
    input.put(m.x.data(), width);
    std::vector<Q80> gq(xb), mid(hb);
    std::vector<float> host_gu(2 * ff), host_h(ff), host_d(width);
    auto sync = [&] { ck(hipGetLastError()); ck(hipStreamSynchronize(st)); };

    std::puts("production AVX2 vs actual CPU-reference HIP (no production engine integration)");
    cp04::quantize(input.p, xq.p, xb, st);
    cp04::project(wg.p, xq.p, gu.p, xb, 2 * ff, st);
    sync();
    xq.get(gq.data(), gq.size());
    gu.get(host_gu.data(), host_gu.size());
    size_t issues = compare_q("activation Q8_0", a.xq, gq);
    issues += compare_gate_up("candidate gate/up from candidate Q8_0", a, host_gu);
    xq.put(a.xq.data(), a.xq.size());
    cp04::project(wg.p, xq.p, gu.p, xb, 2 * ff, st);
    sync();
    gu.get(host_gu.data(), host_gu.size());
    issues += compare_gate_up("isolated candidate gate/up from exact AVX2 Q8_0", a, host_gu);

    h.put(a.h.data(), ff);
    auto down = [&] {
        cp04::quantize(h.p, hq.p, hb, st);
        cp04::project(wd.p, hq.p, d.p, hb, width, st);
        sync();
        hq.get(mid.data(), mid.size());
        d.get(host_d.data(), width);
    };
    down();
    issues += compare_q("CPU SiTU -> Q8_0", a.hq, mid);
    issues += compare_fp("CPU SiTU -> down FP32", a.d, host_d);
    if (issues) {
        std::puts("candidate HIP leaf parity failed on this fixture; skipping SiTU-chain diagnostic");
        return false;
    }

    // gu is verified bit-identical to the candidate-quant pass. Use the
    // existing HIP SiTU, not a claimed port of host libm.
    poly::situ_rows(gu.p, h.p, ff, 1, m.act.beta, m.act.linear_beta, st, poly::ExpertScales{});
    sync();
    h.get(host_h.data(), ff);
    std::puts("diagnostic only: exact gate/up -> existing HIP SiTU -> candidate Q8_0/down; no weighted aggregation");
    compare_fp("SiTU FP32", a.h, host_h);
    down();
    compare_q("HIP SiTU -> Q8_0", a.hq, mid);
    compare_fp("HIP SiTU -> down FP32", a.d, host_d);
    return true;
}

struct RefEvent {
    hipEvent_t e{};
    RefEvent() { ck(hipEventCreate(&e)); }
    ~RefEvent() { (void)hipEventDestroy(e); }
};

void cpu_ref_time(const Model& m) {
    Stream stream;
    const auto st = stream.s;
    Device<float> input(width), y(2 * ff);
    Device<Q80> avx(xb);
    Device<Q81> native(xb);
    Device<uint8_t> w(gate_up_bytes);
    input.put(m.x.data(), width);
    w.put(m.gate, gate_up_bytes);
    auto launch = [&](int arm) {
        if (arm == 2) cp04::quantize(input.p, avx.p, xb, st);
        if (arm == 3) nk::native_quantize_q8_1(input.p, native.p, width, 1, st);
        if (arm % 2 == 0) cp04::project(w.p, avx.p, y.p, xb, 2 * ff, st);
        else nk::native_mmvq(GGML_TYPE_Q4_0, w.p, native.p, y.p, width, 2 * ff, 1, st);
    };
    constexpr int warmup = 24, batches = 7, repeats = 48, arms = 4;
    for (int i = 0; i < warmup; ++i) { launch(2); launch(3); }
    ck(hipGetLastError());
    ck(hipStreamSynchronize(st));
    RefEvent start, stop;
    std::array<std::array<double, batches>, arms> times{};
    for (int b = 0; b < batches; ++b) {
        for (int j = 0; j < arms; ++j) {
            const int arm = (j + b) % arms;
            ck(hipEventRecord(start.e, st));
            for (int i = 0; i < repeats; ++i) launch(arm);
            ck(hipEventRecord(stop.e, st));
            ck(hipGetLastError());
            ck(hipEventSynchronize(stop.e));
            float ms = 0.f;
            ck(hipEventElapsedTime(&ms, start.e, stop.e));
            require(ms > 0.f, "invalid HIP event interval");
            times[arm][b] = ms * 1000.0 / repeats;
        }
    }
    std::printf("leaf latency: us/expert, 1 expert/launch, resident fixed input/weights, HIP events, "
                "%d batches x %d reps, no SiTU/down/aggregation/transfers, not an engine timing\n", batches, repeats);
    constexpr const char* names[] = {"candidate prepacked dot", "native prepacked dot",
                                     "candidate Q8_0 + dot", "native standalone Q8_1 + dot"};
    for (int arm = 0; arm < arms; ++arm) {
        auto& s = times[arm];
        std::printf("  %-28s samples:", names[arm]);
        for (double v : s) std::printf(" %.3f", v);
        std::sort(s.begin(), s.end());
        std::printf("; p50=%.3f min=%.3f max=%.3f\n", s[batches / 2], s.front(), s.back());
    }
}
#endif

void run_cpu_ref(const Model& m, bool suite) {
    require(!suite || !m.captured_q.empty(), "CPU-reference suite needs the captured decode input");
    auto check = [](const Model& fixture) {
        label_cpu_ref_fixture(fixture);
        const auto a = cpu_stages(fixture);
        cpu_ref_model_check(fixture, a);
#if defined(CP_HIP)
        return cpu_ref_hip_check(fixture, a);
#else
        return true;
#endif
    };
#if defined(CP_HIP)
    hipDeviceProp_t props{};
    ck(hipGetDeviceProperties(&props, 0));
    require(std::string(props.gcnArchName).substr(0, 7) == "gfx1031" && props.warpSize == 32,
            "candidate requires wave32 gfx1031");
    std::printf("device: %s (%s)\n", props.name, props.gcnArchName);
#endif
    std::printf("model=%s bytes=%llu\n", m.file.path().c_str(), static_cast<unsigned long long>(m.file.file_size()));
    if (suite) {
        const Model proxy(m.file.path(), 3, 0, 64, 0.125f, nullptr);
        const bool proxy_ok = check(proxy);
        const bool capture_ok = check(m);
        require(proxy_ok && capture_ok, "candidate HIP leaf parity failed; timings skipped");
#if defined(CP_HIP)
        std::puts("Both fixture leaf checks passed. Timing only the leaf variants.");
        label_cpu_ref_fixture(proxy);
        cpu_ref_time(proxy);
        label_cpu_ref_fixture(m);
        cpu_ref_time(m);
#endif
    } else require(check(m), "candidate HIP leaf parity failed");
}
}
