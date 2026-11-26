// Replays only the ordered CPU branch of previously captured W1 rows. No GPU or model forward pass.
#include "cpu_experts.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"

#include <array>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <random>
#include <sched.h>
#include <sys/resource.h>

namespace strata::poly {
decltype(cpu_experts) cpu_experts_reuse;
}

namespace {
using namespace strata;
using namespace strata::poly;
using Clock = std::chrono::steady_clock;
constexpr int width = 2304, ff = 1152, experts = 128, visits = 48, top = 8;
constexpr int q40 = GGML_TYPE_Q4_0, q41 = GGML_TYPE_Q4_1;
using Vec = std::array<float, width>;

void need(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }

int physical(int visit) { return visit < 6 ? visit : visit < 42 ? 6 + (visit - 6) % 12 : visit - 24; }

struct Layer {
    std::array<CpuExpert, experts> expert;
    size_t gu, down;
    int dt;
};

std::array<Layer, 24> load(const GgufFile& gg) {
    std::array<Layer, 24> result;
    const auto* arch = gg.get("general.architecture");
    need(arch && arch->s == "inkling2_loop", "expected Inkling loop GGUF");
    for (int l = 0; l < 24; ++l) {
        const auto prefix = "blk." + std::to_string(l) + '.';
        const auto* gu = gg.find(prefix + "ffn_gate_up_exps.weight");
        const auto* down = gg.find(prefix + "ffn_down_exps.weight");
        need(gu && gu->type == q40 && gu->shape == std::vector<uint64_t>{width, 2 * ff, experts}, "bad gate/up");
        need(down && (down->type == q40 || down->type == q41) &&
             down->shape == std::vector<uint64_t>{ff, width, experts}, "bad down");
        auto& L = result[l];
        L.dt = (int) down->type;
        L.gu = ggml_row_size(GGML_TYPE_Q4_0, width) * ff;
        L.down = ggml_row_size((ggml_type) L.dt, ff) * width;
        need(tensor_payload_bytes(*gu) == 2 * L.gu * experts && tensor_payload_bytes(*down) == L.down * experts,
             "bad expert bytes");
        const auto* g = gg.tensor_data(*gu);
        const auto* d = gg.tensor_data(*down);
        for (int e = 0; e < experts; ++e) L.expert[e] = {g + 2 * e * L.gu, g + (2 * e + 1) * L.gu, d + e * L.down, 0.f};
    }
    return result;
}

struct Frame { int64_t pos; int32_t visit, layer, n, n_gpu, down, reserved, id[top]; float w[top]; Vec x, sum; };
static_assert(sizeof(Frame) == 18528);

struct Capture {
    int first, count;
    bool have_sum;
    std::vector<Frame> frames;
    std::vector<const Frame*> by;

    explicit Capture(const char* path) {
        std::ifstream f(path, std::ios::binary);
        char magic[8];
        uint32_t h[4];
        need(bool(f.read(magic, 8).read((char*) h, sizeof(h))), "missing capture header");
        have_sum = !memcmp(magic, "PSCPU01\n", 8);
        need((have_sum || !memcmp(magic, "PSCPUNS\n", 8)) && h[1] == width &&
             h[3] && h[3] <= 4096 && h[0] && h[0] <= h[3] * visits, "bad capture header");
        first = h[2];
        count = h[3];
        frames.resize(h[0]);
        by.resize(count * visits);
        need(bool(f.read((char*) frames.data(), frames.size() * sizeof(Frame))) && f.peek() == EOF, "bad capture length");
        for (const Frame& a : frames) {
            need(a.pos >= first && a.pos < int64_t(first) + count && a.visit >= 0 && a.visit < visits &&
                 a.layer == physical(a.visit) && a.n >= (have_sum ? 1 : 0) && a.n <= top && a.n + a.n_gpu == top &&
                 (a.down == q40 || a.down == q41), "bad capture row");
            std::array<bool, experts> seen{};
            for (int i = 0; i < a.n; ++i) {
                need(a.id[i] >= 0 && a.id[i] < experts && !seen[a.id[i]] && std::isfinite(a.w[i]), "bad route");
                seen[a.id[i]] = true;
            }
            auto& ptr = by[(a.pos - first) * visits + a.visit];
            need(!ptr, "duplicate capture row");
            ptr = &a;
        }
        if (!have_sum) for (const Frame* p : by) need(p, "no-sum input must contain every row");
    }
};

struct Case {
    int rows, pos, visit;
    const Layer& layer;
    std::vector<CpuExpert> uses;
    std::vector<float> x, archived;
    std::array<int, 9> counts{};
    size_t unique_bytes = 0;

    Case(int rows, int at, int v, const Capture& cap, const Layer& L)
        : rows(rows), pos(cap.first + at), visit(v), layer(L), x(rows * width), archived(rows * width) {
        std::array<int, experts> freq{};
        for (int row = 0; row < rows; ++row) {
            const Frame* a = cap.by[(at + row) * visits + v];
            if (!a) continue;
            need(a->down == L.dt, "down format differs from capture");
            std::copy(a->x.begin(), a->x.end(), x.begin() + row * width);
            std::copy(a->sum.begin(), a->sum.end(), archived.begin() + row * width);
            for (int i = 0; i < a->n; ++i) {
                auto e = L.expert[a->id[i]];
                e.row = row;
                e.weight = a->w[i];
                uses.push_back(e);
                ++freq[a->id[i]];
            }
        }
        for (int n : freq) if (n) ++counts[n];
        unique_bytes = std::accumulate(counts.begin(), counts.end(), 0) * (2 * L.gu + L.down);
    }

    void prepare(bool cold) const {
        std::vector<const uint8_t*> seen;
        for (const auto& e : uses) {
            if (std::find(seen.begin(), seen.end(), e.gate) != seen.end()) continue;
            seen.push_back(e.gate);
            for (auto [start, bytes] : {std::pair{e.gate, 2 * layer.gu}, {e.down, layer.down}}) {
                const size_t step = cold ? 64 : 4096;
                const size_t skew = reinterpret_cast<uintptr_t>(start) % step;
                for (size_t i = 0; i < bytes + skew; i += step) {
                    const uint8_t* p = start + (i == 0 ? 0 : i - skew);
                    if (cold) _mm_clflushopt(const_cast<uint8_t*>(p));
                    else (void)*static_cast<const volatile uint8_t*>(p);
                }
            }
        }
        if (cold) _mm_sfence();
    }
};

enum Arm { serial, grouped, multi };
constexpr const char* arm_name[] = {"serial", "grouped", "multi"};

void run(Workers& workers, const Case& c, SiTU situ, Arm arm, std::vector<float>& out) {
    const int n = (int) c.uses.size();
    if (!n) {
        std::fill(out.begin(), out.end(), 0.f);
        return;
    }
    if (arm != serial) {
        auto fn = arm == grouped ? cpu_experts : cpu_experts_reuse;
        fn(workers, c.uses.data(), n, c.rows, q40, q40, c.layer.dt, width, ff,
           c.x.data(), 0.f, false, out.data(), nullptr, nullptr, nullptr, situ);
        return;
    }
    std::array<CpuExpert, top> one{};
    int j = 0;
    for (int r = 0; r < c.rows; ++r) {
        int m = 0;
        for (; j < n && c.uses[j].row == r; ++j) {
            one[m] = c.uses[j];
            one[m++].row = 0;
        }
        if (m) cpu_experts(workers, one.data(), m, 1, q40, q40, c.layer.dt, width, ff,
                           c.x.data() + r * width, 0.f, false, out.data() + r * width, nullptr, nullptr, nullptr, situ);
        else std::fill_n(out.data() + r * width, width, 0.f);
    }
}

void equal(const Case& c, const std::string& name, const std::vector<float>& out, const std::vector<float>& ref) {
    for (int r = 0; r < c.rows; ++r) for (int i = 0; i < width; ++i) {
        const int k = r * width + i;
        if (!memcmp(&out[k], &ref[k], sizeof(float))) continue;
        std::cerr << name << " first difference: pos=" << c.pos + r << " visit=" << c.visit << " row=" << r
                  << " component=" << i << " actual=" << std::hexfloat << out[k] << " expected=" << ref[k] << '\n';
        throw std::runtime_error("CPU output mismatch");
    }
}

void shape(std::ostream& f, const Case& c) {
    f << c.rows << ',' << c.pos << ',' << c.visit << ',' << physical(c.visit) << ',' << c.layer.dt << ','
      << c.uses.size() << ',' << c.unique_bytes;
    for (int k = 1; k <= 8; ++k) f << ',' << c.counts[k];
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 7) throw std::runtime_error("usage: replay MODEL CAPTURE OUT THREADS ROUNDS check|cold|warm");
    const Capture cap(argv[2]);
    need(cap.count >= 2 && cap.count % 2 == 0, "expected an even count of capture rows");
    const int threads = std::stoi(argv[4]), rounds = std::stoi(argv[5]);
    const std::string mode = argv[6];
    const bool check = mode == "check", cold = mode == "cold";
    cpu_set_t cpus;
    need((threads == 1 || threads == 2) && (!sched_getaffinity(0, sizeof(cpus), &cpus)) && CPU_COUNT(&cpus) == threads,
         "set taskset to exactly the requested one or two CPUs");
    need((check || cold || mode == "warm") && rounds >= 1 && rounds <= 12, "bad mode or rounds");
    const GgufFile gg(argv[1]);
    const auto layers = load(gg);
    auto meta = [&](const char* field) {
        const auto* v = gg.get(std::string("inkling2_loop.activation.") + field);
        need(v && v->is_num() && std::isfinite(v->num()) && v->num() > 0, "invalid SiTU metadata");
        return (float) v->num();
    };
    const SiTU situ{meta("situ_beta"), meta("situ_linear_beta")};
    std::string err;
    if (!cpu_expert_types_ok(q40, q40, width, ff, err) || !cpu_expert_types_ok(q40, q41, width, ff, err))
        throw std::runtime_error(err);
    Workers workers(threads);
    std::ofstream result(argv[3]);
    need(bool(result), "cannot create output");
    result << "width,pos,visit,layer,down,uses,unique_bytes,n1,n2,n3,n4,n5,n6,n7,n8";
    if (!check) result << ",round,part,arm,wall_ns,thread_ns,major,minor,invol";
    result << '\n';
    long long compared = 0, reference = 0;
    std::mt19937 rng(20261126);
    const int check_width = cap.count <= 8 ? cap.count : 8;
    for (int w : {2, 4, 6, 8}) for (int v = 0; v < visits; ++v) {
        if (w > cap.count || (cap.count <= 8 && w != cap.count)) continue;
        // A reproducible position for every visit/width, regardless of multiplicity. Also time the first visit at first row.
        const int sampled = (int) (rng() % (cap.count / w));
        for (int block = 0; block < cap.count / w; ++block) {
            if (!check && block != sampled && !(v == 0 && block == 0)) continue;
            Case c(w, block * w, v, cap, layers[physical(v)]);
            std::vector<float> ref(w * width), out(w * width);
            if (check) {
                shape(result, c);
                result << '\n';
                // W8 covers all captured positions for these datasets. Other widths are checked per timed case.
                if (w != check_width) continue;
            }
            c.prepare(false);
            run(workers, c, situ, serial, ref);
            if (cap.have_sum) {
                equal(c, "stock W1 vs capture", ref, c.archived);
                reference += (long long) w * width;
            }
            for (Arm arm : {grouped, multi}) {
                run(workers, c, situ, arm, out);
                equal(c, arm_name[arm], out, ref);
                compared += (long long) w * width;
            }
            if (check || c.uses.empty()) continue;
            for (int round = 0; round < rounds; ++round) {
                for (int part = 0; part < 6; ++part) {
                    constexpr int order[] = {0, 1, 2, 2, 1, 0};
                    const auto arm = (Arm) ((order[part] + (round + v + block) % 3) % 3);
                    if (cold) c.prepare(true);
                    else { run(workers, c, situ, arm, out); equal(c, arm_name[arm], out, ref); }
                    rusage before{}, after{};
                    timespec tc0{}, tc1{};
                    getrusage(RUSAGE_SELF, &before);
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tc0);
                    const auto start = Clock::now();
                    run(workers, c, situ, arm, out);
                    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tc1);
                    getrusage(RUSAGE_SELF, &after);
                    equal(c, arm_name[arm], out, ref);
                    compared += (long long) w * width;
                    shape(result, c);
                    result << ',' << round << ',' << part << ',' << arm_name[arm] << ',' << ns << ','
                           << 1000000000LL * (tc1.tv_sec - tc0.tv_sec) + tc1.tv_nsec - tc0.tv_nsec << ','
                           << after.ru_majflt - before.ru_majflt << ',' << after.ru_minflt - before.ru_minflt << ','
                           << after.ru_nivcsw - before.ru_nivcsw << '\n';
                }
            }
        }
    }
    need(bool(result.flush()), "output failed");
    std::cerr << "mode=" << mode << " threads=" << threads << " capture_first=" << cap.first << " rows=" << cap.count
              << " capture_records=" << cap.frames.size() << " capture_sum=" << cap.have_sum << " compared_words="
              << compared << " archive_words=" << reference << '\n';
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
