// CP-P0-04: CPU vs CPU emulation, or CPU vs the actual production HIP kernels.
#include "cpu_experts.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#if defined(CP_HIP)
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace poly = strata::poly;
#if defined(CP_HIP)
namespace nk = strata::kernels;
#endif
using strata::GgufFile;
constexpr int width = 2304, ff = 1152, qk = 32;
struct Q80 { uint16_t d; int8_t q[qk]; };
struct Q81 { uint16_t d, s; int8_t q[qk]; };
static_assert(sizeof(Q80) == 34 && sizeof(Q81) == 36);
static_assert(std::endian::native == std::endian::little);

void require(bool b, const std::string& why) { if (!b) throw std::runtime_error(why); }
uint32_t bits(float f) { return std::bit_cast<uint32_t>(f); }
uint16_t hbits(float f) { return std::bit_cast<uint16_t>(static_cast<_Float16>(f)); }
float fhalf(uint16_t b) { return std::bit_cast<_Float16>(b); }
uint16_t half_at(const uint8_t* p) { uint16_t b; std::memcpy(&b, p, 2); return b; }

float xor_sum(std::array<float, qk> a, int first = 16) {
    for (int step = first; step; step /= 2) {
        const auto prev = a;
        for (int i = 0; i < qk; ++i) a[i] = prev[i] + prev[i ^ step];
    }
    return a[0];
}

std::vector<Q81> gpu_quant_model(const std::vector<float>& x, const std::vector<float>* prescale = nullptr,
                                const std::vector<float>* norm = nullptr) {
    std::vector<Q81> q(x.size() / qk);
    for (size_t b = 0; b < q.size(); ++b) {
        std::array<float, qk> a;
        std::copy_n(x.data() + b * qk, qk, a.begin());
        float amax = 0;
        for (float v : a) amax = std::max(amax, std::abs(v));
        const float d = std::min(amax / 127.f, 65504.f);
        q[b].d = hbits(d);
        float s;
        if (prescale) {
            // gfx1031 codegen for the fused FFN norm contracts this lane's
            // final multiply into the first XOR add. The peer is pre-rounded.
            std::array<float, qk> first;
            for (int i = 0; i < qk; ++i)
                first[i] = std::fma((*prescale)[b * qk + i], (*norm)[b * qk + i], a[i ^ 16]);
            s = xor_sum(first, 8);
        } else s = xor_sum(a);
        q[b].s = hbits(std::clamp(s, -65504.f, 65504.f));
        for (int i = 0; i < qk; ++i)
            q[b].q[i] = amax == 0 ? 0 : static_cast<int8_t>(std::clamp(std::round(a[i] / d), -127.f, 127.f));
    }
    return q;
}

std::vector<Q80> cpu_quant(const std::vector<float>& x) {
    std::vector<Q80> q(x.size() / qk);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), q.data(), x.size());
    return q;
}

// Inkling Q4_0 works on 2 lanes per block, 64 blocks per round. Four waves
// are folded in order, with the 36-block down projection's last wave zero.
// centered=true is a counterfactual: same waves but center the integer nibbles
// exactly before scaling, so the half-rounded raw sum is not used.
float gpu_dot_model(const uint8_t* w, const Q81* x, int n, bool centered = false) {
    const int nb = n / qk;
    float lane[128]{};
    for (int t = 0; t < 128; ++t) {
        for (int b = t / 2; b < nb; b += 64) {
            const uint8_t* wb = w + 18 * b;
            const float dw = fhalf(half_at(wb)), d = fhalf(x[b].d), s = fhalf(x[b].s);
            int sum = 0;
            const int mid = 8 * (t % 2), offset = centered ? 8 : 0;
            for (int j = mid; j < mid + 8; ++j) {
                const int v = wb[2 + j];
                sum += ((v & 15) - offset) * x[b].q[j] + ((v >> 4) - offset) * x[b].q[j + 16];
            }
            const float term = centered ? sum * d : std::fma(static_cast<float>(sum), d, -4.f * s);
            lane[t] = std::fma(dw, term, lane[t]);
        }
    }
    std::array<float, qk> a;
    for (int i = 0; i < qk; ++i) a[i] = ((lane[i] + lane[i+32]) + lane[i+64]) + lane[i+96];
    return xor_sum(a);
}

float situ(float g, float u, poly::SiTU s) {
    return s.beta * std::tanh(g / s.beta) / (1.f + std::exp(-g)) * (s.linear_beta * std::tanh(u / s.linear_beta));
}

struct Model {
    GgufFile file;
    const uint8_t *gate, *up, *down;
    int layer, expert, token;
    float eps, weight;
    poly::SiTU act;
    std::vector<float> x, residual, norm, prescale;
    std::vector<Q81> captured_q;
    std::string source;
    Model(const std::string& path, int l, int e, int t, float w, const char* fixture)
        : file(path), layer(l), expert(e), token(t), weight(w), x(width), residual(width), norm(width) {
        const auto* arch = file.get("general.architecture");
        require(arch && arch->s == "inkling2_loop", "expected inkling2_loop");
        auto param = [&](const char* suffix) {
            const auto* value = file.get(arch->s + "." + suffix);
            require(value && std::isfinite(value->f) && value->f > 0, suffix);
            return static_cast<float>(value->f);
        };
        eps = param("attention.layer_norm_rms_epsilon");
        act = {param("activation.situ_beta"), param("activation.situ_linear_beta")};
        const std::string p = "blk." + std::to_string(layer) + ".";
        const auto* gu = file.find(p + "ffn_gate_up_exps.weight");
        const auto* d = file.find(p + "ffn_down_exps.weight");
        require(gu && gu->type == GGML_TYPE_Q4_0 && gu->shape == std::vector<uint64_t>{width, 2*ff, 128},
                "need packed Q4_0 gate then up");
        require(d && d->type == GGML_TYPE_Q4_0 && d->shape == std::vector<uint64_t>{ff, width, 128},
                "need Q4_0 down (Tideglass layers 3-23)");
        require(expert >= 0 && expert < 128 && std::isfinite(weight), "invalid expert or weight");
        gate = file.tensor_data(*gu) + expert * 2 * ff * (width / qk * 18);
        up = gate + ff * (width / qk * 18);
        down = file.tensor_data(*d) + expert * width * (ff / qk * 18);
        const auto* n = file.find(p + "ffn_norm.weight");
        require(n && n->type == GGML_TYPE_F32 && n->shape == std::vector<uint64_t>{width}, "need F32 ffn_norm");
        std::memcpy(norm.data(), file.tensor_data(*n), width * sizeof(float));
        if (fixture) {
            source = "captured input (raw xn followed by xq; provenance supplied by caller): " + std::string(fixture);
            std::ifstream f(fixture, std::ios::binary);
            captured_q.resize(width / qk);
            f.read(reinterpret_cast<char*>(x.data()), width * sizeof(float));
            f.read(reinterpret_cast<char*>(captured_q.data()), captured_q.size() * sizeof(Q81));
            require(f && f.peek() == EOF, "expected exactly 11808 bytes: xn FP32 then xq Q8_1");
        } else {
            source = "GGUF Q8_0 embedding through selected layer's FFN RMSNorm (proxy, not a decoded FFN row)";
            const auto* emb = file.find("token_embd.weight");
            require(emb && emb->type == GGML_TYPE_Q8_0 && emb->shape.size() == 2 && emb->shape[0] == width &&
                    token >= 0 && static_cast<uint64_t>(token) < emb->shape[1], "invalid Q8_0 embedding or token");
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(
                file.tensor_data(*emb) + token * (width / qk * sizeof(Q80)), residual.data(), width);
            float parts[4]{};  // 128 sum threads, 4 waves, same order as native_add_rms_norm_q8_1.
            for (int wave = 0; wave < 4; ++wave) {
                std::array<float, qk> lanes{};
                for (int i = 0; i < qk; ++i)
                    for (int j = wave * qk + i; j < width; j += 128)
                        lanes[i] = std::fma(residual[j], residual[j], lanes[i]);
                parts[wave] = xor_sum(lanes);
            }
            const float scale = 1.f / std::sqrt(((parts[0] + parts[1]) + parts[2] + parts[3]) / width + eps);
            prescale.resize(width);
            for (int i = 0; i < width; ++i) {
                prescale[i] = residual[i] * scale;
                x[i] = prescale[i] * norm[i];
            }
        }
        require(std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); }), "nonfinite input");
    }
};

template<class Q> struct Stages {
    std::vector<Q> xq, hq;
    std::vector<float> g = std::vector<float>(ff), u = std::vector<float>(ff), h = std::vector<float>(ff),
                       d = std::vector<float>(width), sum = std::vector<float>(width);
};

Stages<Q80> cpu_stages(const Model& m) {
    Stages<Q80> a;
    a.xq = cpu_quant(m.x);
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0)->vec_dot;
    for (int r = 0; r < ff; ++r) {
        dot(width, &a.g[r], 0, m.gate + r * (width / qk * 18), 0, a.xq.data(), 0, 1);
        dot(width, &a.u[r], 0, m.up + r * (width / qk * 18), 0, a.xq.data(), 0, 1);
        a.h[r] = situ(a.g[r], a.u[r], m.act);
    }
    a.hq = cpu_quant(a.h);
    for (int r = 0; r < width; ++r) {
        dot(ff, &a.d[r], 0, m.down + r * (ff / qk * 18), 0, a.hq.data(), 0, 1);
        a.sum[r] = 0.f + m.weight * a.d[r];
    }
    // Ensure the split CPU replay is the actual production chain at its output.
    poly::Workers workers(8);
    const poly::CpuExpert e{m.gate, m.up, m.down, m.weight, 0, 1.f, 1.f};
    std::vector<float> check(width);
    poly::cpu_experts(workers, &e, 1, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,
                      width, ff, m.x.data(), INFINITY, false, check.data(), nullptr, nullptr, nullptr, m.act);
    require(std::memcmp(check.data(), a.sum.data(), width * sizeof(float)) == 0, "CPU stage replay vs cpu_experts mismatch");
    return a;
}

Stages<Q81> gpu_model_stages(const Model& m) {
    Stages<Q81> b;
    b.xq = m.captured_q.empty() ? gpu_quant_model(m.x, &m.prescale, &m.norm) : m.captured_q;
    for (int r = 0; r < ff; ++r) {
        b.g[r] = gpu_dot_model(m.gate + r * (width / qk * 18), b.xq.data(), width);
        b.u[r] = gpu_dot_model(m.up + r * (width / qk * 18), b.xq.data(), width);
        // Host libm only; do not assert OCML tanhf/expf/division bits.
        b.h[r] = situ(b.g[r], b.u[r], m.act);
    }
    b.hq = gpu_quant_model(b.h);
    for (int r = 0; r < width; ++r) {
        b.d[r] = gpu_dot_model(m.down + r * (ff / qk * 18), b.hq.data(), ff);
        b.sum[r] = std::fma(m.weight, b.d[r], 0.f);
    }
    return b;
}

size_t compare_fp(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "float size mismatch");
    size_t count = 0, first = 0;
    double maxdiff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), std::string(what) + ": nonfinite result");
        if (bits(a[i]) != bits(b[i])) { if (count == 0) first = i; ++count; }
        maxdiff = std::max(maxdiff, std::abs(static_cast<double>(a[i]) - b[i]));
    }
    std::printf("%-31s %zu/%zu differ", what, count, a.size());
    if (count) std::printf(", first [%zu] 0x%08x (%+.9g) vs 0x%08x (%+.9g), max_abs=%.9g", first,
                           bits(a[first]), a[first], bits(b[first]), b[first], maxdiff);
    std::puts("");
    return count;
}

template<class A, class B> size_t compare_q(const char* what, const std::vector<A>& a, const std::vector<B>& b) {
    require(a.size() == b.size(), "quant size mismatch");
    size_t sd = 0, qd = 0, ss = 0;
    std::string s_first, q_first;
    for (size_t j = 0; j < a.size(); ++j) {
        if (a[j].d != b[j].d && sd++ == 0) {
            char line[80];
            std::snprintf(line, sizeof(line), ", first block %zu 0x%04x vs 0x%04x", j, a[j].d, b[j].d);
            s_first = line;
        }
        for (int i = 0; i < qk; ++i)
            if (a[j].q[i] != b[j].q[i] && qd++ == 0) {
                char line[80];
                std::snprintf(line, sizeof(line), ", first block %zu lane %d 0x%02x (%d) vs 0x%02x (%d)", j, i,
                              static_cast<uint8_t>(a[j].q[i]), a[j].q[i], static_cast<uint8_t>(b[j].q[i]), b[j].q[i]);
                q_first = line;
            }
        if constexpr (sizeof(A) == 36 && sizeof(B) == 36) {
            if (a[j].s != b[j].s && ss++ == 0)
                std::printf("%-31s first sum block %zu 0x%04x vs 0x%04x\n", what, j, a[j].s, b[j].s);
        }
    }
    std::printf("%-31s scales %zu/%zu differ%s; codes %zu/%zu differ%s", what, sd, a.size(), s_first.c_str(),
                qd, a.size() * qk, q_first.c_str());
    if constexpr (sizeof(A) == 36 && sizeof(B) == 36) std::printf("; sums %zu/%zu differ", ss, a.size());
    std::puts("");
    return sd + qd + ss;
}

void first_code_detail(const std::vector<float>& x, const std::vector<Q80>& a, const std::vector<Q81>& b) {
    for (size_t j = 0; j < a.size(); ++j)
        for (int i = 0; i < qk; ++i) {
            if (a[j].q[i] == b[j].q[i]) continue;
            float amax = 0;
            for (int k = 0; k < qk; ++k) amax = std::max(amax, std::abs(x[j*qk+k]));
            const float xi = x[j*qk+i], cd = xi * (127.f / amax), gd = xi / (amax / 127.f);
            std::printf("first code detail: x[%zu]=%.9g (0x%08x) amax=%.9g (0x%08x) scale_half=0x%04x; "
                        "CPU before rounding %.9g (0x%08x), emulated HIP %.9g (0x%08x)\n",
                        j*qk+i, xi, bits(xi), amax, bits(amax), a[j].d, cd, bits(cd), gd, bits(gd));
            return;
        }
}

#if defined(CP_HIP)
void ck(hipError_t err) { if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err)); }
struct Stream {
    hipStream_t s{};
    Stream() { ck(hipStreamCreate(&s)); }
    ~Stream() { (void)hipStreamDestroy(s); }
};
template<class T> struct Device {
    T* p{};
    explicit Device(size_t n) { ck(hipMalloc(&p, n * sizeof(T))); }
    ~Device() { (void)hipFree(p); }
    void put(const T* x, size_t n) { ck(hipMemcpy(p, x, n * sizeof(T), hipMemcpyHostToDevice)); }
    void get(T* x, size_t n) const { ck(hipMemcpy(x, p, n * sizeof(T), hipMemcpyDeviceToHost)); }
};

std::array<Stages<Q81>, 2> hip_stages(Model& m) {
    hipDeviceProp_t props{};
    ck(hipGetDeviceProperties(&props, 0));
    require(std::string(props.gcnArchName).substr(0, 7) == "gfx1031", "only gfx1031 has this production slot path");
    std::printf("device: %s (%s)\n", props.name, props.gcnArchName);
    Stream stream;
    const auto st = stream.s;
    Device<float> input(width), norm(width), xn(width), gu(4 * ff), h(2 * ff), d(2 * width), sum(width);
    Device<Q81> xq(width / qk), direct_xq(width / qk), hq(2 * ff / qk), separate_hq(2 * ff / qk);
    constexpr size_t gu_bytes = 2 * ff * (width / qk * 18), d_bytes = width * (ff / qk * 18);
    Device<uint8_t> wg(gu_bytes), wd(d_bytes);
    wg.put(m.gate, gu_bytes);
    wd.put(m.down, d_bytes);
    if (m.captured_q.empty()) {
        const auto emulated_x = m.x;
        input.put(m.residual.data(), width);
        norm.put(m.norm.data(), width);
        nk::native_add_rms_norm_q8_1(input.p, nullptr, norm.p, xn.p, xq.p, width, 1, m.eps, 128, st);
        ck(hipGetLastError());
        ck(hipStreamSynchronize(st));
        xn.get(m.x.data(), width);
        if (compare_fp("proxy norm: CPU vs HIP", emulated_x, m.x))
            std::puts("proxy RMSNorm differed: the emulated fused sum uses host prescale and is not a bit oracle");
    } else {
        xn.put(m.x.data(), width);
        xq.put(m.captured_q.data(), m.captured_q.size());
    }
    nk::native_quantize_q8_1(xn.p, direct_xq.p, width, 1, st);
    ck(hipGetLastError());
    ck(hipStreamSynchronize(st));
    std::array<Stages<Q81>, 2> stages;
    std::vector<Q81> direct(width / qk);
    direct_xq.get(direct.data(), direct.size());
    for (int si = 0; si < 2; ++si) {
        auto& b = stages[si];
        b.xq.resize(width / qk);
        b.hq.resize(ff / qk);
        const int rows = si + 1;
        if (rows == 1) nk::native_mmvq(GGML_TYPE_Q4_0, wg.p, xq.p, gu.p, width, 2 * ff, 1, st);
        else {
            nk::MmvqSlots slots{};
            for (int j = 0; j < rows; ++j) {
                slots.w[j] = wg.p; slots.x[j] = xq.p; slots.y[j] = gu.p + j * 2 * ff;
            }
            require(nk::native_inkling_q4_0_slots(slots, rows, width, 2 * ff, st), "no specialized gate/up");
        }
        const poly::ExpertScales scales{};
        poly::situ_rows(gu.p, h.p, ff, rows, m.act.beta, m.act.linear_beta, st, scales);
        poly::situ_rows_q8_1(gu.p, hq.p, ff, rows, m.act.beta, m.act.linear_beta, st, scales);
        nk::native_quantize_q8_1(h.p, separate_hq.p, ff, rows, st);
        if (rows == 1) nk::native_mmvq(GGML_TYPE_Q4_0, wd.p, hq.p, d.p, ff, width, 1, st);
        else {
            nk::MmvqSlots slots{};
            for (int j = 0; j < rows; ++j) {
                slots.w[j] = wd.p; slots.x[j] = hq.p + j * (ff / qk); slots.y[j] = d.p + j * width;
            }
            require(nk::native_inkling_q4_0_slots(slots, rows, ff, width, st), "no specialized down");
        }
        ck(hipMemsetAsync(sum.p, 0, width * sizeof(float), st));
        poly::ScaledRows acc{};
        acc.x[0] = d.p; acc.acc[0] = sum.p; acc.w[0] = m.weight;
        poly::add_scaled(acc, 1, width, st);
        ck(hipGetLastError());
        ck(hipStreamSynchronize(st));
        std::vector<float> both_gu(rows * 2 * ff), both_d(rows * width);
        std::vector<Q81> both_hq(rows * (ff / qk)), separated(both_hq.size());
        xq.get(b.xq.data(), b.xq.size());
        gu.get(both_gu.data(), both_gu.size());
        d.get(both_d.data(), both_d.size());
        h.get(b.h.data(), ff);
        hq.get(both_hq.data(), both_hq.size());
        separate_hq.get(separated.data(), separated.size());
        sum.get(b.sum.data(), width);
        std::copy_n(both_gu.begin(), ff, b.g.begin());
        std::copy_n(both_gu.begin() + ff, ff, b.u.begin());
        std::copy_n(both_d.begin(), width, b.d.begin());
        std::copy_n(both_hq.begin(), ff / qk, b.hq.begin());
        std::printf("HIP arm: %s\n", rows == 1 ? "one resident / native_mmvq" : "two resident slots / cloned expert");
        compare_q("HIP fused vs separate hq", both_hq, separated);
        if (rows == 2) {
            require(std::memcmp(both_gu.data(), both_gu.data() + 2 * ff, 2 * ff * sizeof(float)) == 0 &&
                    std::memcmp(both_hq.data(), both_hq.data() + ff / qk, ff / qk * sizeof(Q81)) == 0 &&
                    std::memcmp(both_d.data(), both_d.data() + width, width * sizeof(float)) == 0,
                    "same expert in two slots gave different bytes");
        }
    }
    compare_q("HIP fused/capture vs direct xq", stages[0].xq, direct);
    return stages;
}
#endif

template<class A, class B> void compare(const Stages<A>& a, const Stages<B>& b) {
    compare_q("activation xq", a.xq, b.xq);
    compare_fp("gate FP32", a.g, b.g);
    compare_fp("up FP32", a.u, b.u);
    compare_fp("SiTU FP32", a.h, b.h);
    compare_q("intermediate hq", a.hq, b.hq);
    compare_fp("down FP32", a.d, b.d);
    compare_fp("one weighted expert sum", a.sum, b.sum);
}
} // namespace

int main(int argc, char** argv) try {
    require(argc == 2 || argc == 6 || argc == 7,
            "usage: parity MODEL [LAYER EXPERT PROXY_TOKEN WEIGHT [XN_XQ_CAPTURE]]");
    ggml_cpu_init();
    Model m(argv[1], argc >= 6 ? std::stoi(argv[2]) : 3, argc >= 6 ? std::stoi(argv[3]) : 0,
            argc >= 6 ? std::stoi(argv[4]) : 64, argc >= 6 ? std::stof(argv[5]) : 0.125f, argc == 7 ? argv[6] : nullptr);
    std::printf("model=%s bytes=%llu layer=%d expert=%d token_for_proxy=%d weight=%.9g (0x%08x) beta=%.9g linear=%.9g\n",
                m.file.path().c_str(), static_cast<unsigned long long>(m.file.file_size()), m.layer, m.expert, m.token,
                m.weight, bits(m.weight), m.act.beta, m.act.linear_beta);
    std::printf("activation=%s\n", m.source.c_str());
#if defined(CP_HIP)
    // After hip_stages the CPU and emulation both consume HIP's exact fused norm FP32 output.
    const auto actual = hip_stages(m);
#endif
    const auto a = cpu_stages(m);
    const auto b = gpu_model_stages(m);
    std::puts("CPU AVX2 vs CPU EMULATION of HIP: CPU replay matches production cpu_experts weighted output");
    compare(a, b);
    first_code_detail(m.x, a.xq, b.xq);
    if (m.captured_q.empty()) compare_q("emulated fused vs direct xq", b.xq, gpu_quant_model(m.x));
    std::vector<float> centered(ff), same_codes(ff);
    std::vector<Q81> matched = b.xq;
    for (size_t j = 0; j < matched.size(); ++j) {
        matched[j].d = a.xq[j].d;
        std::copy_n(a.xq[j].q, qk, matched[j].q);
    }
    for (int r = 0; r < ff; ++r) {
        same_codes[r] = gpu_dot_model(m.gate + r * (width / qk * 18), matched.data(), width);
        centered[r] = gpu_dot_model(m.gate + r * (width / qk * 18), matched.data(), width, true);
    }
    compare_fp("gate: CPU vs matched d/codes", a.g, same_codes);
    compare_fp("gate: HIP model codes vs CPU", b.g, same_codes);
    compare_fp("gate: CPU vs centered control", a.g, centered);
    int sumq = 0;
    for (int i : a.xq[0].q) sumq += i;
    std::printf("gate row0: weight scale b0=0x%04x raw-sum b0=0x%04x (%+.9g), b0 d*sum(codes)=%+.9g; "
                "centered control=0x%08x\n", half_at(m.gate), b.xq[0].s, fhalf(b.xq[0].s),
                fhalf(a.xq[0].d) * sumq, bits(centered[0]));
#if defined(CP_HIP)
    std::puts("CPU AVX2 vs ACTUAL HIP two-slot");
    compare(a, actual[1]);
    std::puts("CPU EMULATION vs ACTUAL HIP two-slot");
    compare(b, actual[1]);
    std::puts("ACTUAL HIP generic vs ACTUAL HIP two-slot");
    compare(actual[0], actual[1]);
#endif
    return 0;  // Differing arithmetic is evidence, not an execution failure.
} catch (const std::exception& e) {
    std::fprintf(stderr, "parity: %s\n", e.what());
    return 1;
}
