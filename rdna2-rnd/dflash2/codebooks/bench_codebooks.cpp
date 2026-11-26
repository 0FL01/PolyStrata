// CPU-only DFlash2 selector microbenchmark. The LM head, top-k and gate projection are outside the timer.
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

constexpr int vocab = 200058, rank = 256, steps = 7, topk = 16;
constexpr double mib = 1048576.0;
constexpr auto clock_now = std::chrono::steady_clock::now;
constexpr const char * names[] = {"selector_predecessor.weight", "selector_successor.weight"};
constexpr ggml_type types[] = {
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,
    GGML_TYPE_Q5_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K, GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
};

static void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

struct Mapping {
    size_t length;
    uint8_t * data;
    explicit Mapping(const char * path) {
        int fd = open(path, O_RDONLY | O_CLOEXEC);
        require(fd >= 0, std::string("open: ") + path);
        struct stat st{};
        if (fstat(fd, &st) || st.st_size <= 0) {
            close(fd);
            throw std::runtime_error(std::string("bad file size: ") + path);
        }
        length = st.st_size;
        data = static_cast<uint8_t *>(mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0));
        close(fd);
        require(data != MAP_FAILED, std::string("mmap: ") + path);
    }
    ~Mapping() { munmap(data, length); }
    Mapping(const Mapping &) = delete;
    Mapping & operator=(const Mapping &) = delete;
};

struct Table {
    ggml_type type;
    ggml_to_float_t decode;
    size_t row_bytes;
    const uint8_t * data;

    const uint8_t * row(int id) const { return data + size_t(id) * row_bytes; }
    size_t bytes() const { return row_bytes * vocab; }

    void get(int id, float * dst, bool fast = false) const {
        const auto * src = row(id);
        if (type == GGML_TYPE_F32) {
            std::memcpy(dst, src, rank * sizeof(float));
        } else if (fast && (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) {
            for (int i = 0; i < rank; i += 8) {
                const auto h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 2*i));
                const auto f = type == GGML_TYPE_F16 ? _mm256_cvtph_ps(h) :
                    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
                _mm256_store_ps(dst + i, f);
            }
        } else {
            decode(src, dst, rank);
        }
    }

    std::pair<uint8_t *, size_t> pages() const {
        const size_t ps = sysconf(_SC_PAGESIZE);
        const auto start = reinterpret_cast<uintptr_t>(data) & ~(ps - 1);
        const auto end = (reinterpret_cast<uintptr_t>(data) + bytes() + ps - 1) & ~(ps - 1);
        return {reinterpret_cast<uint8_t *>(start), end - start};
    }
};

struct Books {
    Mapping map;
    std::array<Table, 2> tables;
    size_t backbone_bytes = 0;

    explicit Books(const char * path) : map(path) {
        std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(
            gguf_init_from_buffer(map.data, map.length, {true, nullptr}), gguf_free);
        require(bool(gguf), std::string("invalid GGUF: ") + path);
        const auto data_offset = gguf_get_data_offset(gguf.get());
        for (size_t i = 0; i < tables.size(); ++i) {
            const auto id = gguf_find_tensor(gguf.get(), names[i]);
            require(id >= 0, names[i]);
            const auto * ne = gguf_get_tensor_ne(gguf.get(), id);
            require(ne[0] == rank && ne[1] == vocab && ne[2] == 1 && ne[3] == 1, "unexpected codebook shape");
            const auto type = gguf_get_tensor_type(gguf.get(), id);
            const auto decode = ggml_get_type_traits(type)->to_float;
            require(decode || type == GGML_TYPE_F32, "no ggml decoder for codebook");
            const size_t off = gguf_get_tensor_offset(gguf.get(), id);
            const size_t len = gguf_get_tensor_size(gguf.get(), id);
            const size_t row_bytes = ggml_row_size(type, rank);
            require(data_offset <= map.length && off <= map.length - data_offset &&
                    len <= map.length - data_offset - off && len == row_bytes * vocab, "truncated codebook");
            tables[i] = {type, decode, row_bytes, map.data + data_offset + off};
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf.get()); ++i) {
            const std::string name = gguf_get_tensor_name(gguf.get(), i);
            if (name != names[0] && name != names[1]) backbone_bytes += gguf_get_tensor_size(gguf.get(), i);
        }
    }
};

struct Query {
    int32_t anchor, ids[steps][topk];
    float unary[steps][topk], gate[steps][rank];
};

struct Result {
    int32_t picked[steps];
    float scores[steps][topk], probs[steps][topk];
};

static_assert(sizeof(Query) == 8068 && sizeof(Result) == 924, "DFQ1 record layout");

struct TraceQuery {
    Query query;
    Result oracle;
};

static std::vector<TraceQuery> read_trace(const char * path) {
    std::ifstream input(path, std::ios::binary);
    char magic[4];
    uint32_t n = 0;
    require(bool(input.read(magic, 4)) && std::memcmp(magic, "DFQ1", 4) == 0 &&
            bool(input.read(reinterpret_cast<char *>(&n), 4)) && n > 0 &&
            std::filesystem::file_size(path) == 8 + size_t(n) * (sizeof(Query) + sizeof(Result)), "bad DFQ1 file");
    std::vector<TraceQuery> cases(n);
    const auto valid_id = [](int id) { return id >= 0 && id < vocab; };
    for (auto & [q, out] : cases) {
        require(bool(input.read(reinterpret_cast<char *>(&q), sizeof(q))) &&
                bool(input.read(reinterpret_cast<char *>(&out), sizeof(out))), "truncated DFQ1 record");
        require(valid_id(q.anchor), "bad DFQ1 anchor");
        for (int t = 0; t < steps; ++t) {
            require(valid_id(out.picked[t]) && std::all_of(q.ids[t], q.ids[t] + topk, valid_id), "bad DFQ1 token id");
        }
    }
    return cases;
}

static std::vector<Query> synthetic(int mode, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<double> zipf(vocab);
    std::vector<int> perm(vocab);
    double mass = 0;
    for (int id = 0; id < vocab; ++id) {
        mass += std::pow(id + 1.0, -1.07);
        zipf[id] = mass;
        perm[id] = id;
    }
    if (mode == 1) std::shuffle(perm.begin(), perm.end(), rng);
    std::uniform_real_distribution<double> uniform(0, mass);
    auto token = [&]() {
        if (mode == 0) return int(rng() % 512) * 389 % vocab;
        if (mode == 1) return perm[std::lower_bound(zipf.begin(), zipf.end(), uniform(rng)) - zipf.begin()];
        return int(rng() % vocab);
    };
    std::vector<Query> out(2000);
    for (auto & q : out) {
        q.anchor = token();
        for (int t = 0; t < steps; ++t) {
            for (int k = 0; k < topk; ++k) {
                q.ids[t][k] = token();
                q.unary[t][k] = 12.0f - 0.5f*k + 0.25f*normal(rng);
            }
            for (float & g : q.gate[t]) g = 0.1f*normal(rng);
        }
    }
    return out;
}

static float dot(const float * x, const float * y) {
    __m256 acc[4]{};
    for (int i = 0; i < rank; i += 32) {
        for (int j = 0; j < 4; ++j) {
            acc[j] = _mm256_fmadd_ps(_mm256_load_ps(x + i + j*8), _mm256_load_ps(y + i + j*8), acc[j]);
        }
    }
    const auto v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    auto h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_hadd_ps(h, h);
    return _mm_cvtss_f32(_mm_hadd_ps(h, h));
}

template<bool scalar>
static Result walk(const std::array<Table, 2> & books, const Query & q, bool softmax,
                   const Result * teacher = nullptr, float (*corr_out)[topk] = nullptr) {
    Result out{};
    int prev = q.anchor;
    for (int t = 0; t < steps; ++t) {
        if constexpr (!scalar) {
            for (int id : q.ids[t]) {
                const uint8_t * row = books[1].row(id);
                for (size_t j = 0; j < books[1].row_bytes; j += 64) _mm_prefetch(row + j, _MM_HINT_T0);
            }
        }

        alignas(32) float a[rank], b[rank];
        books[0].get(prev, a, !scalar);
        for (int r = 0; r < rank; ++r) a[r] *= q.gate[t][r];
        int best = 0;
        for (int k = 0; k < topk; ++k) {
            books[1].get(q.ids[t][k], b, !scalar);
            float corr;
            if constexpr (scalar) {
                double sum = 0;
                for (int r = 0; r < rank; ++r) sum += double(a[r]) * double(b[r]);
                corr = float(sum);
            } else {
                corr = dot(a, b);
            }
            if (corr_out) corr_out[t][k] = corr;
            out.scores[t][k] = 30.0f * std::tanh((q.unary[t][k] + corr) / 30.0f);
            if (out.scores[t][k] > out.scores[t][best]) best = k;
        }
        if (softmax) {
            float sum = 0;
            for (int k = 0; k < topk; ++k) {
                out.probs[t][k] = std::exp(out.scores[t][k] - out.scores[t][best]);
                sum += out.probs[t][k];
            }
            for (float & p : out.probs[t]) p /= sum;
        }
        out.picked[t] = q.ids[t][best];
        prev = teacher ? teacher->picked[t] : out.picked[t];
    }
    return out;
}

static long proc_value(const char * key) {
    std::ifstream file("/proc/self/status");
    for (std::string line; std::getline(file, line);) {
        if (line.starts_with(key)) return std::stol(line.substr(std::strlen(key)));
    }
    throw std::runtime_error(std::string("missing /proc field: ") + key);
}

static size_t resident(const std::array<Table, 2> & books) {
    const size_t ps = sysconf(_SC_PAGESIZE);
    size_t bytes = 0;
    for (const auto & book : books) {
        const auto [base, len] = book.pages();
        std::vector<unsigned char> pages(len / ps);
        require(mincore(base, len, pages.data()) == 0, "mincore failed");
        for (auto state : pages) bytes += (state & 1) * ps;
    }
    return bytes;
}

static volatile uint64_t sink;

static void time_walks(const std::array<Table, 2> & books, const std::vector<Query> & queries,
                       const char * label, bool softmax, size_t count, size_t warmup) {
    auto run = [&](size_t i) {
        const auto out = walk<false>(books, queries[i % queries.size()], softmax);
        uint64_t hash = 0;
        for (int t = 0; t < steps; ++t) {
            hash = hash * 1315423911ULL + out.picked[t];
            if (softmax) for (float p : out.probs[t]) hash += uint64_t(p * 1000000.0f);
        }
        return hash;
    };
    for (size_t i = 0; i < warmup; ++i) sink = sink ^ run(i);
    std::vector<double> times(count);
    rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);
    for (size_t i = 0; i < count; ++i) {
        const auto start = clock_now();
        const auto hash = run(i);
        const auto stop = clock_now();
        sink = sink ^ hash;
        times[i] = std::chrono::duration<double, std::micro>(stop - start).count();
    }
    getrusage(RUSAGE_SELF, &after);
    std::sort(times.begin(), times.end());
    const auto p = [&](size_t pct) { return times[(count - 1)*pct/100]; };
    printf("TIME %s %s n=%zu p50=%.2f p90=%.2f p99=%.2f us minflt=%ld majflt=%ld\n",
           label, softmax ? "T1" : "greedy", count, p(50), p(90), p(99),
           after.ru_minflt - before.ru_minflt, after.ru_majflt - before.ru_majflt);
}

static void bench(const std::array<Table, 2> & books, const std::vector<TraceQuery> & trace) {
    const auto hot = synthetic(0, 101), zipf = synthetic(1, 202), random = synthetic(2, 303);
    const long rss0 = proc_value("RssFile:");
    for (const auto & book : books) {
        const auto [base, len] = book.pages();
        require(madvise(base, len, MADV_DONTNEED) == 0, "MADV_DONTNEED failed");
    }
    printf("MMAP before=%.3f MiB page-cache resident (DONTNEED discarded mapping PTEs, not page cache)\n",
           resident(books) / mib);
    time_walks(books, random, "first-random", true, 200, 0);
    printf("MMAP first-random RssFile_delta=%.3f MiB\n", (proc_value("RssFile:") - rss0) / 1024.0);

    const size_t ps = sysconf(_SC_PAGESIZE);
    const auto start = clock_now();
    for (const auto & book : books) {
        const auto [base, len] = book.pages();
        require(madvise(base, len, MADV_WILLNEED) == 0, "MADV_WILLNEED failed");
        const volatile uint8_t * p = base;
        uint64_t sum = 0;
        for (size_t i = 0; i < len; i += ps) sum += p[i];
        sink = sink ^ sum;
        require(madvise(base, len, MADV_RANDOM) == 0, "MADV_RANDOM failed");
    }
    printf("MMAP prefault=%.2f ms RssFile_delta=%.3f MiB page-cache resident=%.3f MiB\n",
           std::chrono::duration<double, std::milli>(clock_now() - start).count(),
           (proc_value("RssFile:") - rss0) / 1024.0, resident(books) / mib);
    std::vector<std::pair<uint8_t *, size_t>> locked;
    int lock_error = 0;
    for (const auto & book : books) {
        const auto [base, len] = book.pages();
        if (mlock(base, len) != 0) { lock_error = errno; break; }
        locked.emplace_back(base, len);
    }
    rlimit lim{};
    getrlimit(RLIMIT_MEMLOCK, &lim);
    printf("MLOCK result=%s VmLck=%ld KiB limit=%.3f MiB\n",
           lock_error ? std::strerror(lock_error) : "success", proc_value("VmLck:"), double(lim.rlim_cur) / mib);
    for (const auto & [base, len] : locked) munlock(base, len);

    for (const auto & [label, data] : {std::pair{"hot512", &hot}, {"zipf", &zipf}, {"random", &random}}) {
        time_walks(books, *data, label, false, data->size(), 200);
        time_walks(books, *data, label, true, data->size(), 200);
    }
    if (!trace.empty()) {
        std::vector<Query> data;
        for (const auto & entry : trace) data.push_back(entry.query);
        time_walks(books, data, "DFT1-repeated", false, 2000, 200);
        time_walks(books, data, "DFT1-repeated", true, 2000, 200);
    }
}

struct Error {
    double worst_score = 0, score_sum = 0, tv_sum = 0;
    int equal = 0, n = 0;
    int first_q = -1, first_step = -1, actual = -1, wanted = -1;
    void add(const Result & got, const Result & want) {
        for (int t = 0; t < steps; ++t) {
            if (got.picked[t] == want.picked[t]) {
                ++equal;
            } else if (first_q < 0) {
                first_q = n / steps; first_step = t;
                actual = got.picked[t]; wanted = want.picked[t];
            }
            ++n;
            for (int k = 0; k < topk; ++k) {
                const double e = std::abs(double(got.scores[t][k]) - want.scores[t][k]);
                worst_score = std::max(worst_score, e);
                score_sum += e;
                tv_sum += 0.5 * std::abs(double(got.probs[t][k]) - want.probs[t][k]);
            }
        }
    }
    void report(const char * label) const {
        printf("CHECK %s picks=%d/%d score_MAE=%.7g score_max=%.7g mean_TVD=%.7g",
               label, equal, n, score_sum / (n*topk), worst_score, tv_sum / n);
        if (first_q >= 0) printf(" first_diff=%d:%d got=%d want=%d", first_q, first_step, actual, wanted);
        printf("\n");
    }
};

static void check(const std::array<Table, 2> & books, const std::array<Table, 2> & ref,
                  const std::vector<TraceQuery> & trace) {
    Error kernel, oracle, precision;
    double corr_sq_ref = 0, corr_sq_diff = 0, kernel_corr_max = 0;
    for (const auto & [q, want] : trace) {
        float fast_corr[steps][topk], slow_corr[steps][topk], ref_corr[steps][topk];
        const auto got = walk<false>(books, q, true);
        const auto slow = walk<true>(books, q, true);
        kernel.add(got, slow);
        oracle.add(got, want);
        const auto teacher = walk<true>(ref, q, true, &want, ref_corr);
        const auto forced = walk<false>(books, q, true, &want, fast_corr);
        walk<true>(books, q, true, &want, slow_corr);
        precision.add(forced, teacher);
        for (int t = 0; t < steps; ++t) for (int k = 0; k < topk; ++k) {
            const double r = ref_corr[t][k];
            const double diff = fast_corr[t][k] - r;
            corr_sq_ref += r*r; corr_sq_diff += diff*diff;
            kernel_corr_max = std::max(kernel_corr_max, std::abs(double(fast_corr[t][k]) - slow_corr[t][k]));
        }
    }
    kernel.report("same-quant-scalar");
    oracle.report("free-walk-vs-oracle");
    precision.report("oracle-predecessor-vs-BF16");
    printf("CHECK max_kernel_corr_error=%.8g corr_rel_RMSE_vs_BF16=%.8g\n",
           kernel_corr_max, std::sqrt(corr_sq_diff / corr_sq_ref));
    require(kernel.equal == kernel.n && kernel.worst_score <= 1e-4 && kernel_corr_max <= 1e-4,
            "sparse kernel differs from same-quant scalar oracle");
}

static void pack(const Books & source, const std::filesystem::path & dir) {
    require(std::filesystem::is_directory(dir), "pack output directory must exist");
    constexpr int batch = 1024;
    std::vector<float> rows(batch * rank);
    for (auto type : types) {
        if (type == GGML_TYPE_BF16) continue;
        const auto path = dir / (std::string(ggml_type_name(type)) + ".gguf");
        require(!std::filesystem::exists(path), "pack refuses to overwrite " + path.string());
        const size_t row_bytes = ggml_row_size(type, rank);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ggml(
            ggml_init({2*ggml_tensor_overhead(), nullptr, true}), ggml_free);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_empty(), gguf_free);
        std::array<std::vector<uint8_t>, 2> data;
        for (size_t s = 0; s < source.tables.size(); ++s) {
            const auto & table = source.tables[s];
            require(table.type == GGML_TYPE_BF16, "pack expects BF16 weights");
            data[s].resize(row_bytes * vocab);
            for (int i = 0; i < vocab; i += batch) {
                const int count = std::min(batch, vocab - i);
                for (int j = 0; j < count; ++j) table.get(i + j, rows.data() + j*rank);
                const size_t bytes = ggml_quantize_chunk(type, rows.data(), data[s].data() + i*row_bytes,
                                                         0, count, rank, nullptr);
                require(bytes == count*row_bytes, "bad ggml quantizer write size");
            }
            ggml_tensor * tensor = ggml_new_tensor_2d(ggml.get(), type, rank, vocab);
            ggml_set_name(tensor, names[s]);
            gguf_add_tensor(gguf.get(), tensor);
            gguf_set_tensor_data(gguf.get(), names[s], data[s].data());
        }
        require(gguf_write_to_file(gguf.get(), path.c_str(), false), "failed to write " + path.string());
        printf("PACK %s %.3f MiB\n", path.c_str(), 2.0*row_bytes*vocab/mib);
    }
}

int main(int argc, char ** argv) try {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc == 4 && std::strcmp(argv[1], "pack") == 0) {
        pack(Books(argv[2]), argv[3]);
        return 0;
    }
    if (argc < 5 || std::strcmp(argv[1], "bench") != 0) {
        fprintf(stderr, "usage: %s pack BF16.gguf EMPTY_DIR\n"
                        "       %s bench BF16.gguf {trace.dfq1|-} GGUF [GGUF|PRED.gguf:SUCC.gguf ...]\n", argv[0], argv[0]);
        return 2;
    }
    const Books ref(argv[2]);
    require(ref.tables[0].type == GGML_TYPE_BF16 && ref.tables[1].type == GGML_TYPE_BF16, "bench expects BF16 reference");
    const auto trace = std::strcmp(argv[3], "-") ? read_trace(argv[3]) : std::vector<TraceQuery>{};
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto sep = arg.find(':');
        const Books base(arg.substr(0, sep).c_str());
        auto pair = base.tables;
        std::unique_ptr<Books> succ;
        if (sep != std::string::npos) {
            succ = std::make_unique<Books>(arg.substr(sep + 1).c_str());
            pair[1] = succ->tables[1];
        }
        const size_t bytes = pair[0].bytes() + pair[1].bytes();
        printf("\nPROFILE %s predecessor=%s/%zuB successor=%s/%zuB codebooks=%zuB/%.3fMiB backbone=%.3fMiB\n",
               argv[i], ggml_type_name(pair[0].type), pair[0].row_bytes, ggml_type_name(pair[1].type), pair[1].row_bytes,
               bytes, bytes / mib, base.backbone_bytes / mib);
        bench(pair, trace);
        if (!trace.empty()) check(pair, ref.tables, trace);
    }
    printf("sink=%llu\n", static_cast<unsigned long long>(sink));
    return 0;
} catch (const std::exception & e) {
    fprintf(stderr, "bench_codebooks: %s\n", e.what());
    return 1;
}
