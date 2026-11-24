// CPU-only experiment. No model state, GPU runtime, predictor output used as routing, or changes to cpu_experts.
#include "cpu_experts.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"

#include <array>
#include <bit>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <sched.h>
#include <sys/resource.h>

namespace {
using namespace strata;
using namespace strata::poly;
using Clock = std::chrono::steady_clock;
constexpr int width = 2304, ff = 1152, q4 = GGML_TYPE_Q4_0, experts = 128;
constexpr size_t matrix = 1'492'992, line = 64;
using Vec = std::array<float, width>;
using Bank = std::array<std::array<CpuExpert, experts>, 12>;
enum Mode { cold, oracle, inline_read, helper_read, helper_t2, helper_early };
constexpr const char* names[] = {"cold", "oracle", "inline_read", "helper_read", "helper_t2", "helper_early"};

void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
int64_t ns(Clock::duration t) { return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count(); }
int layer(int visit) { return 6 + (visit - 6) % 12; }

struct Span {
    const uint8_t* p;
    size_t skew() const { return reinterpret_cast<uintptr_t>(p) % line; }
    size_t count() const { return (matrix + skew() + line - 1) / line; }
    const uint8_t* at(size_t i) const { return p + (i == 0 ? 0 : i * line - skew()); }
};

std::array<Span, 3> payload(const CpuExpert& e) { return {{{e.gate}, {e.up}, {e.down}}}; }

void flush(const CpuExpert& e) {
    for (const Span s : payload(e))
        for (size_t i = 0; i < s.count(); ++i) _mm_clflushopt(const_cast<uint8_t*>(s.at(i)));
}

void fault_in(const CpuExpert& e) {
    for (const Span s : payload(e)) {
        for (size_t i = 0; i < matrix; i += 4096) (void)*static_cast<const volatile uint8_t*>(s.p + i);
        (void)*static_cast<const volatile uint8_t*>(s.p + matrix - 1);
    }
}

size_t sweep(Span s, size_t start, size_t end, bool hints) {
    if (hints) {
        for (size_t i = start; i < end; ++i) _mm_prefetch(s.at(i), _MM_HINT_T2);
    } else {
        for (size_t i = start; i < end; ++i) (void)*static_cast<const volatile uint8_t*>(s.at(i));
    }
    return end - start;
}

size_t single(const CpuExpert& e, bool hints = false, const std::atomic<bool>* stop = nullptr) {
    size_t done = 0;
    for (const Span s : payload(e)) for (size_t i = 0; i < s.count(); i += 512) {
        if (stop && stop->load(std::memory_order_relaxed)) return done;
        done += sweep(s, i, std::min(i + 512, s.count()), hints);
    }
    return done;
}

void parallel_read(Workers& workers, const CpuExpert& e) {
    auto spans = payload(e);
    workers.run(96, [&](int task) {
        Span s = spans[task / 32];
        sweep(s, s.count() * (task % 32) / 32, s.count() * (task % 32 + 1) / 32, false);
    });
}

class Helper {
    std::atomic<unsigned> posted{0}, done{0};
    std::atomic<int> ready{0};
    std::atomic<bool> stop{false};
    const CpuExpert* job = nullptr;
    bool hints = false;

public:
    struct Result { Clock::time_point begin{}, end{}; size_t lines = 0; };

private:
    Result result;
    std::thread thread;

public:
    explicit Helper(int cpu) : thread([this, cpu] {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        ready.store(rc ? -rc : 1, std::memory_order_release);
        ready.notify_one();
        if (rc) return;
        unsigned seen = 0;
        for (;;) {
            posted.wait(seen);
            seen = posted.load(std::memory_order_acquire);
            if (!job) return;
            result.begin = Clock::now();
            result.lines = single(*job, hints, &stop);
            result.end = Clock::now();
            done.store(seen, std::memory_order_release);
            done.notify_one();
        }
    }) {
        ready.wait(0);
        if (ready.load(std::memory_order_acquire) < 0) {
            thread.join();
            throw std::runtime_error("cannot pin the helper to the requested CPU");
        }
    }

    ~Helper() {
        cancel();
        collect();
        post(nullptr, false);
        thread.join();
    }

    void post(const CpuExpert* e, bool use_hints) {
        job = e;
        hints = use_hints;
        stop.store(false, std::memory_order_relaxed);
        posted.fetch_add(1, std::memory_order_release);
        posted.notify_one();
    }
    void cancel() { stop.store(true, std::memory_order_relaxed); }
    Result collect() {
        const unsigned n = posted.load(std::memory_order_relaxed);
        for (unsigned m; (m = done.load(std::memory_order_acquire)) != n;) done.wait(m);
        return result;
    }
};

struct Branch {
    std::array<CpuExpert, 8> list{};
    int n = 0;
    void run(Workers& workers, const Vec& x, Vec& out, SiTU situ) const {
        const static std::function<void()> noop = [] {};
        if (n) cpu_experts(workers, list.data(), n, 1, q4, q4, q4, width, ff, x.data(), INFINITY, false,
                           out.data(), n < 8 ? &noop : nullptr, nullptr, nullptr, situ);
        else out.fill(0.f);
    }
};

struct Case {
    int pos, visit, expert, pop, sampled;
    int64_t near, join;
    bool hit;
    Branch prev{}, target{};
    CpuExpert pick{};
    Vec x{}, xp{}, ref{}, refp{};
};

void read_branch(std::istream& in, Branch& b, const std::array<CpuExpert, experts>& bank, int* match = nullptr) {
    std::array<bool, experts> used{};
    for (int i = 0; i < b.n; ++i) {
        int e;
        uint32_t bits;
        require(bool(in >> e >> bits) && e >= 0 && e < experts && !used[e], "bad expert IDs/weights in case");
        used[e] = true;
        b.list[i] = bank[e];
        b.list[i].weight = std::bit_cast<float>(bits);
        require(std::isfinite(b.list[i].weight), "non-finite trace weight");
        if (match && e == *match) *match = -1;
    }
}

std::vector<Case> load_cases(const char* file, const Bank& bank, int& tokens) {
    std::ifstream in(file);
    std::string version;
    int count;
    require(bool(in >> version >> tokens >> count) && version == "PREFETCH1" && tokens > 0 && count > 0 && count <= 54,
            "bad case file header");
    std::vector<Case> cases(count);
    for (auto& c : cases) {
        require(bool(in >> c.pos >> c.visit >> c.expert >> c.near >> c.join >> c.pop >> c.sampled >> c.prev.n >> c.target.n) &&
                c.pos >= 0 && c.visit >= 18 && c.visit < 42 && c.expert >= 0 && c.expert < experts &&
                c.near > 0 && c.near < 10'000'000 && c.join >= 0 && c.sampled > 0 && c.pop >= c.sampled &&
                0 <= c.prev.n && c.prev.n <= 8 && 0 <= c.target.n && c.target.n <= 8, "bad case header");
        c.pick = bank[layer(c.visit) - 6][c.expert];
        int match = c.expert;
        read_branch(in, c.prev, bank[layer(c.visit - 1) - 6]);
        read_branch(in, c.target, bank[layer(c.visit) - 6], &match);
        c.hit = match == -1;
        // Reproducible residual-like inputs, independent of intervention. Real GGUF, real routes and weights,
        // but the CP-02 trace does not contain activations.
        std::mt19937 rng(uint32_t(c.pos) * 48U + c.visit);
        for (int i = 0; i < width; ++i) {
            float scale = i % 97 == 0 ? 1.f / 32768.f : 1.f / 524288.f;
            c.x[i] = (int(rng() & 65535) - 32768) * scale;
            c.xp[i] = (int(rng() & 65535) - 32768) * scale;
        }
    }
    require(!(in >> version), "extra case data");
    return cases;
}

void parity(const Case& c, const Vec& got, const Vec& expected, const char* branch) {
    if (std::memcmp(got.data(), expected.data(), sizeof(Vec)) == 0) return;
    for (int i = 0; i < width; ++i) {
        auto a = std::bit_cast<uint32_t>(got[i]), b = std::bit_cast<uint32_t>(expected[i]);
        if (a == b) continue;
        std::cerr << "FIRST_DIVERGENCE " << c.pos << '/' << c.visit << ' ' << branch << " output " << i
                  << " got " << a << " expected " << b << '\n';
        throw std::runtime_error("CPU branch output mismatch");
    }
}

void trial(Mode m, const Case& c, Workers& workers, SiTU situ, Helper* helper, int round, const char* pair, int part) {
    const bool scheduled = helper != nullptr;
    const bool async = m == helper_read || m == helper_t2 || m == helper_early;
    Vec out, prev_out;
    if (scheduled) for (int i = 0; i < c.prev.n; ++i) flush(c.prev.list[i]);
    for (int i = 0; i < c.target.n; ++i) flush(c.target.list[i]);
    if (!c.hit) flush(c.pick);
    _mm_sfence();
    if (m == oracle) parallel_read(workers, c.pick); // deliberately untimed, not a scheduler

    size_t lines = 0;
    const auto begin = Clock::now();
    if (m == helper_early) helper->post(&c.pick, false);
    const auto p0 = Clock::now();
    if (scheduled) c.prev.run(workers, c.xp, prev_out, situ);
    const auto near_begin = Clock::now();
    int64_t prep_start = -1, prep_end = -1;
    if (m == helper_read || m == helper_t2) helper->post(&c.pick, m == helper_t2);
    if (m == inline_read) {
        prep_start = ns(Clock::now() - begin);
        lines = single(c.pick);
        prep_end = ns(Clock::now() - begin);
    }
    const auto due = scheduled ? near_begin + std::chrono::nanoseconds(c.near) : near_begin;
    if (scheduled) {
        while (Clock::now() < due) _mm_pause();
        if (async) helper->cancel();   // never await a prediction on the target compute path
    }
    const auto cpu = Clock::now();
    c.target.run(workers, c.x, out, situ);
    const auto end = Clock::now();
    if (async) {
        const auto r = helper->collect(); // drain outside the synthetic critical path before another trial
        prep_start = ns(r.begin - begin);
        prep_end = ns(r.end - begin);
        lines = r.lines;
    }
    parity(c, out, c.ref, "target");
    if (scheduled) parity(c, prev_out, c.refp, "predecessor");
    size_t expected_lines = 0;
    for (auto s : payload(c.pick)) expected_lines += s.count();
    std::cout << c.pos << ',' << c.visit << ',' << c.expert << ',' << c.hit << ',' << c.prev.n << ',' << c.target.n << ','
              << c.pop << ',' << c.sampled << ',' << c.join << ',' << c.near << ',' << round << ',' << pair << ',' << part
              << ',' << names[m] << ',' << ns(end - begin) << ',' << ns(near_begin - p0) << ',' << ns(cpu - begin) << ','
              << ns(end - cpu) << ',' << prep_start << ',' << prep_end << ',' << lines << ',' << expected_lines << ','
              << ns(due - begin) << '\n';
}

Bank load_bank(const GgufFile& gg) {
    const auto* arch = gg.get("general.architecture");
    require(arch && arch->s == "inkling2_loop", "expected Inkling loop GGUF");
    Bank bank{};
    for (int l = 6; l <= 17; ++l) {
        auto get = [&](const char* name, int cols, int rows, size_t stride) {
            const auto* t = gg.find("blk." + std::to_string(l) + '.' + name + ".weight");
            require(t && t->type == q4 && t->shape == std::vector<uint64_t>{uint64_t(cols), uint64_t(rows), experts} &&
                    tensor_payload_bytes(*t) == stride * experts, "unexpected core Q4_0 tensor geometry");
            return gg.tensor_data(*t);
        };
        const auto* gu = get("ffn_gate_up_exps", width, 2 * ff, 2 * matrix);
        const auto* d = get("ffn_down_exps", ff, width, matrix);
        for (size_t e = 0; e < experts; ++e) {
            const auto* gate = gu + e * 2 * matrix;
            bank[l - 6][e] = {gate, gate + matrix, d + e * matrix, 0.f};
        }
    }
    return bank;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 6 || (std::string(argv[1]) != "ceiling" && std::string(argv[1]) != "schedule")) {
        std::cerr << "usage: bench ceiling|schedule MODEL.gguf CASES.tsv ROUNDS(1-3) HELPER_CPU\n";
        return 2;
    }
    const bool scheduled = std::string(argv[1]) == "schedule";
    const int rounds = std::stoi(argv[4]), cpu = std::stoi(argv[5]);
    cpu_set_t mask;
    require(rounds >= 1 && rounds <= 3 && cpu >= 0 && cpu < CPU_SETSIZE &&
            sched_getaffinity(0, sizeof(mask), &mask) == 0 && CPU_COUNT(&mask) == 8 && !CPU_ISSET(cpu, &mask),
            "need 1-3 rounds, a taskset of eight worker CPUs and a distinct helper CPU");
    const GgufFile gg(argv[2]);
    const auto bank = load_bank(gg);
    auto get_meta = [&](const char* key) {
        const auto* v = gg.get(std::string("inkling2_loop.activation.") + key);
        require(v && v->is_num() && std::isfinite(v->num()) && v->num() > 0, "missing Tideglass SiTU metadata");
        return (float)v->num();
    };
    SiTU situ{get_meta("situ_beta"), get_meta("situ_linear_beta")};
    int tokens;
    auto cases = load_cases(argv[3], bank, tokens);
    std::string err;
    if (!cpu_expert_types_ok(q4, q4, width, ff, err)) throw std::runtime_error(err);
    Workers workers(8);
    std::unique_ptr<Helper> helper = scheduled ? std::make_unique<Helper>(cpu) : nullptr;
    std::cerr << "tokens=" << tokens << " cases=" << cases.size() << " rounds=" << rounds << " SiTU=" << situ.beta
              << '/' << situ.linear_beta << " model_bytes=" << gg.file_size() << '\n';
    // Resolve pages and initialize thread-local scratch/reference outputs outside all measured regions.
    for (auto& c : cases) {
        if (!scheduled && !c.hit) continue;
        fault_in(c.pick);
        for (int i = 0; i < c.target.n; ++i) fault_in(c.target.list[i]);
        c.target.run(workers, c.x, c.ref, situ);
        if (scheduled) {
            for (int i = 0; i < c.prev.n; ++i) fault_in(c.prev.list[i]);
            c.prev.run(workers, c.xp, c.refp, situ);
        }
    }
    rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);
    int calibration = 0;
    for (const auto& c : cases) if (c.hit && calibration++ < 9) {
        flush(c.pick);
        _mm_sfence();
        const auto a = Clock::now();
        parallel_read(workers, c.pick);
        const auto b = Clock::now();
        parallel_read(workers, c.pick);
        std::cerr << "read_cal_ns cold=" << ns(b - a) << " warm=" << ns(Clock::now() - b) << '\n';
    }
    std::cout << "pos,visit,pick,hit,prev_n,n,pop,sampled,trace_join_ns,window_ns,round,pair,part,arm,whole_ns,prev_ns,"
                 "compute_start_ns,compute_ns,prep_start_ns,prep_end_ns,prep_lines,expected_lines,deadline_ns\n";
    for (int r = 0; r < rounds; ++r) for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[r % 2 == 0 ? i : cases.size() - 1 - i];
        if (!scheduled && !c.hit) continue;
        std::vector<Mode> arms = scheduled ? std::vector<Mode>{cold, inline_read, helper_read, helper_t2, helper_early}
                                          : std::vector<Mode>{cold, oracle};
        if (i % 2) std::reverse(arms.begin(), arms.end());
        for (auto arm : arms) {
            std::array seq{cold, arm, arm, cold};
            if ((r + i) % 2) {
                std::swap(seq[0], seq[1]);
                std::swap(seq[2], seq[3]);
            }
            for (int j = 0; j < 4; ++j) trial(seq[j], c, workers, situ, helper.get(), r, names[arm], j);
        }
    }
    getrusage(RUSAGE_SELF, &after);
    std::cerr << "timed_major_faults=" << after.ru_majflt - before.ru_majflt
              << " timed_minor_faults=" << after.ru_minflt - before.ru_minflt << " parity=bitwise\n";
    return before.ru_majflt != after.ru_majflt ? 1 : 0;
} catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
}
