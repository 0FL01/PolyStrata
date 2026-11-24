// Teacher-forced NLL for the locked Inkling-2-Mini experiment. Not part of strata-poly.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef INKLING_NLL_HOST_ONLY
#include "poly/conversations.hpp"
#include "poly/engine.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iterator>
#endif

namespace {

constexpr int kActiveVocab = 200058;

void check_id(int32_t id) {
    if (id < 0 || id >= kActiveVocab) throw std::out_of_range("token ID outside active vocabulary: " + std::to_string(id));
}

double target_nll(const float* logits, int32_t target) {
    check_id(target);
    if (!logits) throw std::runtime_error("missing logits");
    int peak = 0;
    for (int i = 0; i < kActiveVocab; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logit for ID " + std::to_string(i));
        if (logits[i] > logits[peak]) peak = i;
    }
    const double top = logits[peak];
    double rest = 0.0;
    for (int i = 0; i < kActiveVocab; ++i)
        if (i != peak) rest += std::exp(double(logits[i]) - top);
    // Keep the common offset out of the final addition and retain small losses near probability 1.
    const double nll = (top - double(logits[target])) + std::log1p(rest);
    if (!std::isfinite(nll)) throw std::runtime_error("nonfinite NLL");
    return nll;
}

void self_check() {
    auto near = [](double got, double want, double tolerance) {
        if (!(std::abs(got - want) <= tolerance)) throw std::runtime_error("arithmetic self-check failed");
    };
    std::vector<float> logits(201024, 1e30f);
    // Uniform active vocabulary with a large offset. The padded rows must contribute nothing.
    for (int i = kActiveVocab; i < 201024; ++i) logits[i] = 2e30f;
    near(target_nll(logits.data(), kActiveVocab - 1), std::log(double(kActiveVocab)), 1e-14);
    // Two distinct high logits, all other active logits far below both.
    for (int i = 0; i < kActiveVocab; ++i) logits[i] = -1000.0f;
    logits[0] = 1000.0f;
    logits[kActiveVocab - 1] = 960.0f;
    near(target_nll(logits.data(), 0), std::log1p(std::exp(-40.0)), 1e-30);
    near(target_nll(logits.data(), kActiveVocab - 1), 40.0, 1e-14);
    for (int32_t id : {-1, kActiveVocab}) {
        try {
            target_nll(logits.data(), id);
        } catch (const std::out_of_range&) {
            continue;
        }
        throw std::runtime_error("out-of-range ID self-check failed");
    }
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        logits[kActiveVocab - 1] = bad;
        try {
            target_nll(logits.data(), 0);
        } catch (const std::runtime_error&) {
            continue;
        }
        throw std::runtime_error("nonfinite logit self-check failed");
    }
    std::puts("Inkling NLL arithmetic self-check passed (host)");
}

#ifndef INKLING_NLL_HOST_ONLY
using strata::poly::Engine;

// Exact manifest format: nonempty ASCII decimal CSV, no spaces or newline.
std::vector<int32_t> read_ids(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open token file: ") + path);
    const std::string csv{std::istreambuf_iterator<char>(in), {}};
    std::vector<int32_t> ids;
    const char* at = csv.data();
    const char* end = at + csv.size();
    for (;;) {
        int32_t id;
        const auto [next, ec] = std::from_chars(at, end, id);
        if (at == end || *at < '0' || *at > '9' || ec != std::errc{} || (next != end && *next != ','))
            throw std::runtime_error(std::string("invalid token CSV: ") + path);
        check_id(id);
        ids.push_back(id);
        if (next == end) return ids;
        at = next + 1;
    }
}

void observe(const char* model, const char* profile, const char* prefix_file, const char* suffix_file) {
    const auto prefix = read_ids(prefix_file);
    const auto suffix = read_ids(suffix_file);
    // experiment-lock.json: standalone, no requests other than this document, no draft or lookup.
    strata::poly::Options o;
    o.model = model;
    o.profile = profile;
    o.kv_format = strata::poly::KvFmt::kQ8_0;
    o.max_context = 200000;
    o.prompt_batch = 2048;
    o.vram_reserve_mib = 2048;
    o.threads = 8;
    o.expert_swaps = 64;
    o.draft = false;
    if (prefix.size() + suffix.size() > size_t(o.max_context))
        throw std::runtime_error("prefix and suffix exceed locked context limit");

    auto engine = Engine::open(o);
    if (engine->n_vocab() != kActiveVocab) throw std::runtime_error("unexpected engine vocabulary size");
    std::fprintf(stderr, "inkling-nll-observer: %d of %d routed experts in VRAM, %d CPU threads\n",
                 engine->cached_experts(), engine->total_experts(), engine->threads());

    // The CLI's fresh, text-only, uninterrupted prefill, including its default in-memory checkpoint marks.
    // Marks affect the call boundaries even if there are no disk sessions or parked conversations.
    strata::poly::Conversations<Engine> kept(*engine, {});
    engine->images(nullptr, 0);
    if (kept.resume(prefix) != 0) throw std::logic_error("expected a fresh engine sequence");
    if (engine->trim_kv(prefix.size())) engine->mind_vram();
    const int last = int(prefix.size()) - 1;
    std::vector<std::pair<int64_t, uint8_t*>> due;
    std::vector<Engine::Mark> marks;
    for (int at = 0; at < last;) {
        const int n = std::min(last - at, engine->max_read());
        kept.before(at, false);
        kept.inside(at, n, due);
        marks.clear();
        for (const auto& [pos, state] : due) marks.push_back({int(pos), state});
        if (!engine->read(prefix.data() + at, n, marks.data(), int(marks.size())))
            throw std::runtime_error("prefix read did not finish");
        kept.read(prefix.data() + at, n);
        at += n;
    }
    kept.before(last, true);
    engine->end_read();
    engine->eval(&prefix.back(), 1, true);
    engine->end_read();

    std::puts("step\tposition\ttarget_id\tnll");
    for (size_t i = 0; i < suffix.size(); ++i) {
        const size_t pos = prefix.size() + i;
        if (engine->pos() != int64_t(pos)) throw std::logic_error("unexpected position before target");
        const int32_t id = suffix[i];
        const double nll = target_nll(engine->logits(), id);
        std::printf("%zu\t%zu\t%d\t%.17g\n", i, pos, int(id), nll);
        engine->eval(&suffix[i], 1, true);
    }
    if (std::fflush(stdout) != 0 || std::ferror(stdout)) throw std::runtime_error("could not write NLL output");
    // Intentionally do not save the expert profile; the caller gives each process a fresh working copy.
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--self-check") {
            self_check();
            return 0;
        }
#ifndef INKLING_NLL_HOST_ONLY
        if (argc == 5) {
            observe(argv[1], argv[2], argv[3], argv[4]);
            return 0;
        }
        std::fputs("usage: inkling-nll-observer MODEL.gguf EXPERT_PROFILE PREFIX.ids.csv TARGET.ids.csv\n", stderr);
#endif
        std::fputs("usage: inkling-nll-observer --self-check\n", stderr);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "inkling-nll-observer: %s\n", e.what());
        return 1;
    }
}
