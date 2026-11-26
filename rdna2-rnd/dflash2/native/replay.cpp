#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>
#include <vector>

#include "device.hpp"
#include "runtime.hpp"

namespace {
using namespace dflash2;
using Clock = std::chrono::steady_clock;
constexpr size_t tap_values = 4 * hidden;
constexpr size_t header_bytes = 16, record_bytes = header_bytes + tap_values * sizeof(float);
static_assert(std::endian::native == std::endian::little && sizeof(float) == 4 &&
              std::numeric_limits<float>::is_iec559);

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

class Trace {
public:
    explicit Trace(const char* path) : file_(path, std::ios::binary | std::ios::ate) {
        const auto bytes = file_.tellg();
        if (bytes < 4 || (bytes - std::streamoff(4)) % record_bytes != 0 ||
            (bytes - std::streamoff(4)) / record_bytes > max_positions)
            throw std::runtime_error("DFT1 file has invalid length or could not be opened");
        file_.seekg(0);
        char magic[4];
        if (!file_.read(magic, sizeof(magic)) || std::string_view(magic, sizeof(magic)) != "DFT1")
            throw std::runtime_error("missing DFT1 magic");
        const int n = (bytes - std::streamoff(4)) / record_bytes;
        tokens_.resize(n);
        for (int i = 0; i < n; ++i) {
            int64_t pos;
            int32_t generated;
            if (!file_.read(reinterpret_cast<char*>(&pos), sizeof(pos)) ||
                !file_.read(reinterpret_cast<char*>(&tokens_[i]), sizeof(tokens_[i])) ||
                !file_.read(reinterpret_cast<char*>(&generated), sizeof(generated)))
                throw std::runtime_error("could not read DFT1 record");
            if (i == 0) first_pos_ = pos;
            if (first_pos_ < 0 || first_pos_ > max_positions - n || pos != first_pos_ + i ||
                tokens_[i] < 0 || tokens_[i] >= vocab || (generated != 0 && generated != 1) ||
                (generated == 0 && prompt_ >= 0))
                throw std::runtime_error("DFT1 requires contiguous valid tokens, prompt then generation");
            if (generated && prompt_ < 0) prompt_ = i;
            file_.seekg(tap_values * sizeof(float), std::ios::cur);
        }
        if (prompt_ < 0) throw std::runtime_error("DFT1 trace has no generated tokens");
        if (first_pos_ != 0 && prompt_ < window)
            throw std::runtime_error("nonzero initial DFT1 position needs at least 2048 warmup rows");
    }

    int prompt() const { return prompt_; }
    int size() const { return int(tokens_.size()); }
    int32_t token(int i) const { return tokens_[i]; }
    int64_t position(int i) const { return first_pos_ + i; }

    void taps(int pos, int n, std::array<std::vector<float>, 4>& dst) {
        file_.seekg(4 + size_t(pos) * record_bytes + header_bytes);
        for (int row = 0; row < n; ++row) {
            for (auto& tap : dst)
                if (!file_.read(reinterpret_cast<char*>(tap.data() + row * hidden), hidden * sizeof(float)))
                    throw std::runtime_error("could not read DFT1 taps");
            file_.seekg(header_bytes, std::ios::cur);
        }
    }

private:
    std::ifstream file_;
    std::vector<int32_t> tokens_;
    int64_t first_pos_ = 0;
    int prompt_ = -1;
};

struct Result {
    double elapsed = 0, prefill = 0, propose = 0, append = 0;
    int steps = 0, changed = 0, fixed = 0, broke = 0, both = 0, neither = 0;
    std::array<int, block> hist{};
    std::array<int, proposals> survival{}, matches{}, unary_matches{}, recall{};
};

template<class Range, class Render>
std::ostream& array(std::ostream& out, const Range& range, Render render) {
    out << '[';
    bool first = true;
    for (const auto& v : range) {
        if (!first) out << ',';
        first = false;
        render(out, v);
    }
    return out << ']';
}

template<class Range> std::ostream& array(std::ostream& out, const Range& range) {
    return array(out, range, [](auto& s, const auto& v) { s << v; });
}

void dump_step(std::ostream& s, bool closed, int64_t start, int anchor, const Proposal& p, int accepted,
               bool hidden_dump) {
    s << "{\"mode\":\"" << (closed ? "closed-loop" : "all-positions") << "\",\"start\":" << start
      << ",\"anchor\":" << anchor << ",\"accepted\":" << accepted << ",\"draft_tokens\":";
    array(s, p.ids) << ",\"cand\":";
    array(s, p.candidates, [](auto& s, const auto& row) {
        array(s, row, [](auto& s, Candidate c) { s << c.id; });
    }) << ",\"unary\":";
    array(s, p.candidates, [](auto& s, const auto& row) {
        array(s, row, [](auto& s, Candidate c) { s << c.logit; });
    }) << ",\"sel_scores\":";
    array(s, p.scores, [](auto& s, const auto& row) { array(s, row); }) << ",\"sel_probs\":";
    array(s, p.probs, [](auto& s, const auto& row) { array(s, row); });
    if (hidden_dump) {
        s << ",\"h_draft\":";
        array(s, p.hidden_states); // Flattened [7, 2304].
    }
    s << "}\n";
}

class Replay {
public:
    Replay(Trace& trace, Runtime& model) : trace_(trace), model_(model) {
        for (int i = 0; i < 4; ++i) {
            host_[i].resize(window * hidden);
            dev_[i].allocate(window * hidden);
        }
    }

    Result pass(bool closed, int max_steps, std::ostream* details, bool hidden_dump) {
        model_.reset();
        Result result;
        auto start = trace_.prompt();
        const auto prefill = Clock::now();
        const int warmup = std::min(start, window);
        if (warmup) append(start - warmup, warmup);
        result.prefill = ms_since(prefill);
        const auto begin = Clock::now();
        std::ostringstream step_json;
        step_json << std::setprecision(9);
        while (start + block <= trace_.size() && result.steps < max_steps) {
            Proposal p;
            const int anchor = trace_.token(start);
            auto now = Clock::now();
            model_.propose(trace_.position(start), anchor, proposals, p, hidden_dump);
            result.propose += ms_since(now);

            int acc = 0;
            while (acc < proposals && p.ids[acc] == trace_.token(start + acc + 1)) ++acc;
            ++result.hist[acc];
            for (int i = 0; i < proposals; ++i) {
                const int truth = trace_.token(start + i + 1);
                const bool draft = p.ids[i] == truth, unary = p.candidates[i][0].id == truth;
                result.survival[i] += i < acc;
                result.matches[i] += draft;
                result.unary_matches[i] += unary;
                result.recall[i] += std::any_of(p.candidates[i].begin(), p.candidates[i].end(),
                                                [&](Candidate c) { return c.id == truth; });
                result.changed += p.ids[i] != p.candidates[i][0].id;
                result.fixed += draft && !unary;
                result.broke += !draft && unary;
                result.both += draft && unary;
                result.neither += !draft && !unary;
            }
            if (details) {
                step_json.str("");
                dump_step(step_json, closed, trace_.position(start), anchor, p, acc, hidden_dump);
                *details << step_json.str();
            }
            // Consume only committed ground-truth taps. In all-positions mode take one step even after a match.
            const int advance = closed ? acc + 1 : 1;
            now = Clock::now();
            append(start, advance);
            result.append += ms_since(now);
            start += advance;
            ++result.steps;
        }
        result.elapsed = ms_since(begin);
        return result;
    }

private:
    void append(int pos, int rows) {
        trace_.taps(pos, rows, host_);
        std::array<const float*, 4> taps;
        for (int i = 0; i < 4; ++i) {
            hip_check(hipMemcpy(dev_[i].get(), host_[i].data(), size_t(rows) * hidden * sizeof(float),
                                hipMemcpyHostToDevice), "upload DFT1 taps");
            taps[i] = dev_[i].get();
        }
        model_.append(trace_.position(pos), rows, taps);
    }

    Trace& trace_;
    Runtime& model_;
    std::array<std::vector<float>, 4> host_;
    std::array<DeviceBuffer<float>, 4> dev_;
};

void summary(std::ostream& out, const Result& r) {
    const int count = std::max(r.steps, 1);
    int draft = 0, top16 = 0;
    for (int i = 0; i < proposals; ++i) {
        draft += r.survival[i];
        top16 += r.recall[i];
    }
    const auto rate = [&](std::ostream& s, int n) { s << double(n) / count; };
    out << "{\n    \"steps\":" << r.steps << ",\n    \"elapsed_s\":" << r.elapsed / 1000.
        << ",\n    \"ms_per_step\":" << r.elapsed / count << ",\n    \"proposal_ms_per_step\":" << r.propose / count
        << ",\n    \"append_ms_per_step\":" << r.append / count << ",\n    \"prompt_append_ms\":" << r.prefill
        << ",\n    \"mean_acceptance_len\":" << 1. + double(draft) / count
        << ",\n    \"mean_draft_accepted\":" << double(draft) / count
        << ",\n    \"survival_rates\":";
    array(out, r.survival, rate) << ",\n    \"pos_match_rates\":";
    array(out, r.matches, rate) << ",\n    \"unary_pos_match_rates\":";
    array(out, r.unary_matches, rate) << ",\n    \"top16_recall_by_pos\":";
    array(out, r.recall, rate) << ",\n    \"top16_recall_overall\":" << double(top16) / (count * proposals)
        << ",\n    \"acc_histogram_0_to_7\":";
    array(out, r.hist) << ",\n    \"selector_stats\":{\"changed_from_unary\":" << r.changed
        << ",\"fixed_unary_error\":" << r.fixed << ",\"broke_unary_correct\":" << r.broke
        << ",\"net_gain\":" << r.fixed - r.broke << ",\"win_rate_when_disagree\":"
        << double(r.fixed) / std::max(1, r.fixed + r.broke) << ",\"both_right\":" << r.both
        << ",\"both_wrong\":" << r.neither << "}\n  }";
}

void json_string(std::ostream& out, std::string_view s) {
    out << '"';
    for (const unsigned char c : s) {
        if (c == '\\' || c == '"') out << '\\' << char(c);
        else if (c < 32) {
            char escaped[7];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", unsigned(c));
            out << escaped;
        } else out << char(c);
    }
    out << '"';
}

constexpr std::string_view usage =
    "Usage: dflash2-replay --target TARGET.gguf --model DRAFT.gguf --replay TAPS.dft1\n"
    "    [--mode both|closed-loop|all-positions] [--max-steps N] [--dump-json SUMMARY.json]\n"
    "    [--dump-steps TRACE.jsonl [--dump-hidden]]\n"
    "Greedy only. Accepts captured DFT1 rows; does not run or time a live target verifier.\n";

int run(int argc, char** argv) {
    const char *target = nullptr, *model = nullptr, *trace_path = nullptr, *json = nullptr, *steps = nullptr;
    std::string_view mode = "both";
    bool hidden_dump = false;
    int max_steps = std::numeric_limits<int>::max();
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") { std::cout << usage; return 0; }
        if (arg == "--dump-hidden") { hidden_dump = true; continue; }
        if (++i == argc) throw std::invalid_argument("missing value for " + std::string(arg));
        const char* v = argv[i];
        if (arg == "--target") target = v;
        else if (arg == "--model") model = v;
        else if (arg == "--replay") trace_path = v;
        else if (arg == "--dump-json") json = v;
        else if (arg == "--dump-steps") steps = v;
        else if (arg == "--mode") mode = v;
        else if (arg == "--temperature" && std::string_view(v) == "0") continue;
        else if (arg == "--max-steps") {
            const auto end = v + std::strlen(v);
            auto [p, err] = std::from_chars(v, end, max_steps);
            if (err != std::errc() || p != end || max_steps < 1)
                throw std::invalid_argument("--max-steps must be a positive integer");
        } else throw std::invalid_argument("unknown argument or unsupported value: " + std::string(arg));
    }
    if (!target || !model || !trace_path || (hidden_dump && !steps) ||
        (mode != "both" && mode != "closed-loop" && mode != "all-positions"))
        throw std::invalid_argument("set target, model, trace and a supported mode; hidden dump needs --dump-steps");
    Trace trace(trace_path);
    Runtime runtime(model, target);
    Replay replay(trace, runtime);
    std::ofstream details;
    if (steps) {
        details.open(steps);
        if (!details) throw std::runtime_error("cannot create step dump");
    }
    std::ostringstream report;
    report << std::setprecision(9) << "[{\n  \"n_total\":" << trace.size() << ",\n  \"n_prompt\":" << trace.prompt()
           << ",\n  \"n_gen\":" << trace.size() - trace.prompt()
           << ",\n  \"first_pos\":" << trace.position(0)
           << ",\n  \"kv_quant\":\"fp32\",\n  \"temperature\":0,\n  \"model\":";
    json_string(report, model);
    report << ",\n  \"replay_only\":true";
    if (mode != "all-positions") {
        report << ",\n  \"closed_loop\":";
        summary(report, replay.pass(true, max_steps, steps ? &details : nullptr, hidden_dump));
    }
    if (mode != "closed-loop") {
        report << ",\n  \"all_positions\":";
        summary(report, replay.pass(false, max_steps, steps ? &details : nullptr, hidden_dump));
    }
    report << "\n}]\n";
    if (steps) {
        details.close();
        if (!details) throw std::runtime_error("cannot write step dump");
    }
    if (json) {
        std::ofstream out(json);
        out << report.str();
        out.close();
        if (!out) throw std::runtime_error("cannot write summary");
    }
    std::cout << report.str();
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) {
        std::cerr << "dflash2-replay: " << e.what() << '\n';
        return 1;
    }
}
