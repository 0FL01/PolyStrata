#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "device.hpp"
#include "kernels.cuh"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace dflash2::books {

inline constexpr int rank = 256;
inline constexpr float softcap = 30.f;

enum class Placement { host, gpu, host_lru };

inline Placement placement() {
    const char* value = std::getenv("DF2_BOOKS");
    if (!value || std::strcmp(value, "host") == 0) return Placement::host;
    if (std::strcmp(value, "gpu") == 0) return Placement::gpu;
    if (std::strcmp(value, "host-lru") == 0) return Placement::host_lru;
    throw std::invalid_argument("DF2_BOOKS must be host, gpu or host-lru");
}

inline const char* name(Placement p) {
    if (p == Placement::gpu) return "gpu";
    if (p == Placement::host_lru) return "host-lru";
    return "host";
}

struct DeviceTable {
    const uint8_t* base;
    size_t row_bytes;
    uint32_t type;

    __device__ float at(int id, int col) const {
        const auto* row = base + size_t(id) * row_bytes;
        if (type == GGML_TYPE_F16)
            return __half2float(__ushort_as_half(reinterpret_cast<const uint16_t*>(row)[col]));
        // Q4_0 packs columns [0,16) in low nibbles and [16,32) in high nibbles.
        row += col / 32 * 18;
        const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(row)));
        const unsigned q = row[2 + (col & 15)];
        return d * float(int((q >> ((col & 16) ? 4 : 0)) & 15) - 8);
    }
};

// One block walks all positions in order. Each wave scores one candidate at a time.
__global__ void walk(DeviceTable pred, DeviceTable succ, int anchor, int n,
                     const Candidate* candidates, const float* gate, int32_t* ids,
                     float* all_scores, float* all_probs) {
    __shared__ float ag[rank], scores[top_k];
    __shared__ int prev;
    const int t = threadIdx.x, lane = t % 32, wave = t / 32;
    if (t == 0) prev = anchor;
    __syncthreads();
    for (int row = 0; row < n; ++row) {
        ag[t] = pred.at(prev, t) * gate[row * rank + t];
        __syncthreads();
        for (int k = wave; k < top_k; k += rank / 32) {
            const Candidate c = candidates[row * top_k + k];
            float s = 0.f;
            if (c.id >= 0 && c.id < vocab) {
                for (int col = lane; col < rank; col += 32)
                    s = fmaf(ag[col], succ.at(c.id, col), s);
                s = wave_sum(s);
                if (lane == 0) scores[k] = softcap * tanhf((s + c.logit) / softcap);
            } else if (lane == 0) {
                scores[k] = -INFINITY;
            }
        }
        __syncthreads();
        if (t == 0) {
            int best = 0;
            bool valid = true;
            for (int k = 0; k < top_k; ++k) {
                valid &= isfinite(scores[k]);
                if (scores[k] > scores[best]) best = k;
            }
            ids[row] = valid ? candidates[row * top_k + best].id : -1;
            prev = valid ? ids[row] : 0;
            if (all_scores) {
                float sum = 0.f;
                for (int k = 0; k < top_k; ++k) {
                    all_scores[row * top_k + k] = scores[k];
                    const float p = expf(scores[k] - scores[best]);
                    all_probs[row * top_k + k] = p;
                    sum += p;
                }
                for (int k = 0; k < top_k; ++k) all_probs[row * top_k + k] /= sum;
            }
        }
        __syncthreads();
    }
}

// Four-way local LRU: 256 decoded rows per table; no vocabulary-sized index.
class RowCache {
    static constexpr int sets = 64, ways = 4;
    struct Tags {
        int id[ways] = {-1, -1, -1, -1};
        uint64_t age[ways]{};
    };
    Tags tags[sets];
    alignas(32) float rows[sets][ways][rank];
    uint64_t clock = 0, hits_ = 0;

public:
    uint64_t hits() const { return hits_; }

    template<class Decode>
    const float* get(int id, Decode decode) {
        const int set = id % sets;
        auto& tag = tags[set];
        int way = 0;
        for (int i = 0; i < ways; ++i) {
            if (tag.id[i] == id) {
                tag.age[i] = ++clock;
                ++hits_;
                return rows[set][i];
            }
            if (tag.age[i] < tag.age[way]) way = i;
        }
        decode(rows[set][way]);
        tag.id[way] = id;
        tag.age[way] = ++clock;
        return rows[set][way];
    }
};

class Event {
    hipEvent_t e_{};
public:
    Event() { hip_check(hipEventCreate(&e_), "selector timing event"); }
    ~Event() { (void)hipEventDestroy(e_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void record(hipStream_t stream) { hip_check(hipEventRecord(e_, stream), "record selector timing event"); }
    double us_after(const Event& before) const {
        float ms = 0.f;
        hip_check(hipEventElapsedTime(&ms, before.e_, e_), "selector elapsed time");
        return double(ms) * 1000;
    }
};

class Timing {
    struct Sample {
        int64_t pos;
        int anchor, n, bytes;
        double d2h_us, device_us, cpu_us;
        uint64_t pred_hits, succ_hits;
        std::array<int32_t, proposals> ids;
    };
    std::vector<Sample> samples;

public:
    Event ready, corrected, downloaded;
    using Clock = std::chrono::steady_clock;

    void add(int64_t pos, int anchor, int n, int bytes, bool gpu, double cpu_us,
             uint64_t pred_hits, uint64_t succ_hits, const Proposal& out) {
        samples.push_back({pos, anchor, n, bytes, downloaded.us_after(gpu ? corrected : ready),
                           gpu ? corrected.us_after(ready) : 0., cpu_us, pred_hits, succ_hits, out.ids});
    }

    ~Timing() {
        // No I/O in the proposal timer. One row per real selector walk, including its offered IDs.
        for (const auto& s : samples) {
            std::fprintf(stderr, "DF2_BOOK_WALK %lld %d %d %d %.3f %.3f %.3f %llu %llu",
                         static_cast<long long>(s.pos), s.anchor, s.n, s.bytes, s.d2h_us, s.device_us,
                         s.cpu_us, static_cast<unsigned long long>(s.pred_hits),
                         static_cast<unsigned long long>(s.succ_hits));
            for (int i = 0; i < s.n; ++i) std::fprintf(stderr, " %d", s.ids[i]);
            std::fputc('\n', stderr);
        }
    }
};

class PinnedReadback {
public:
    struct Data {
        std::array<Candidate, proposals * top_k> candidates;
        std::array<float, proposals * rank> gates;
        std::array<int32_t, proposals> ids;
    };

    PinnedReadback() = default;
    PinnedReadback(const PinnedReadback&) = delete;
    PinnedReadback& operator=(const PinnedReadback&) = delete;
    ~PinnedReadback() {
        if (data_) (void)hipHostFree(data_);
    }
    void allocate() {
        void* storage = nullptr;
        hip_check(hipHostMalloc(&storage, sizeof(Data), hipHostMallocDefault), "pinned selector readback");
        data_ = std::construct_at(static_cast<Data*>(storage));
    }
    Data* get() const { return data_; }

private:
    Data* data_ = nullptr;
};

class Selector {
    Placement mode_ = placement();
    std::unique_ptr<RowCache> pred_cache_, succ_cache_;
    std::unique_ptr<Timing> timing_;
    DeviceBuffer<uint8_t> pred_gpu_, succ_gpu_;
    DeviceBuffer<int32_t> ids_;
    DeviceBuffer<float> scores_, probs_;
    PinnedReadback pinned_;

    template<class Tensor>
    static void upload(DeviceBuffer<uint8_t>& buffer, Tensor table, hipStream_t stream) {
        buffer.allocate(table.bytes);
        hip_check(hipMemcpyAsync(buffer.get(), table.data, table.bytes, hipMemcpyHostToDevice, stream),
                  "upload resident selector codebook");
    }

public:
    template<class Tensor>
    void load(Tensor pred, Tensor succ, hipStream_t stream) {
        if (mode_ == Placement::gpu) {
            for (const auto type : {pred.type, succ.type})
                if (type != GGML_TYPE_F16 && type != GGML_TYPE_Q4_0)
                    throw std::runtime_error("DF2_BOOKS=gpu requires F16 or Q4_0 selector books");
            upload(pred_gpu_, pred, stream);
            upload(succ_gpu_, succ, stream);
            ids_.allocate(proposals);
            scores_.allocate(proposals * top_k);
            probs_.allocate(proposals * top_k);
        } else if (mode_ == Placement::host_lru) {
            pred_cache_ = std::make_unique<RowCache>();
            succ_cache_ = std::make_unique<RowCache>();
        }
        const char* diag = std::getenv("DF2_BOOKS_TIMING");
        if (diag && std::strcmp(diag, "1") == 0) timing_ = std::make_unique<Timing>();
        const char* pin = std::getenv("DF2_BOOKS_PINNED");
        if (pin && std::strcmp(pin, "1") == 0) pinned_.allocate();
        if (mode_ != Placement::host || timing_ || pinned_.get())
            std::fprintf(stderr, "DF2_BOOKS placement=%s table_bytes=%zu device_table_bytes=%zu cache_bytes=%zu pinned_bytes=%zu\n",
                         name(mode_), pred.bytes + succ.bytes, pred_gpu_.get() ? pred.bytes + succ.bytes : 0,
                         pred_cache_ ? 2 * sizeof(RowCache) : 0, pinned_.get() ? sizeof(PinnedReadback::Data) : 0);
    }

    template<class Tensor, class Decode>
    void run(Tensor pred, Tensor succ, const Stream& stream, const Candidate* top, const float* gate,
             const float* hidden_rows, int64_t pos, int anchor, int n, Proposal& out,
             bool read_hidden, bool details, Decode decode) {
        if (timing_) timing_->ready.record(stream);
        const auto download = [&](void* dst, const void* src, size_t size, const char* message) {
            hip_check(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, stream), message);
        };
        int bytes;
        std::array<float, proposals * rank> gate_rows;
        auto* const pinned = pinned_.get();
        constexpr size_t rowsize = top_k * sizeof(Candidate);
        if (mode_ == Placement::gpu) {
            walk<<<1, rank, 0, stream>>>(
                {pred_gpu_.get(), pred.row_bytes, pred.type}, {succ_gpu_.get(), succ.row_bytes, succ.type},
                anchor, n, top, gate, ids_.get(), details ? scores_.get() : nullptr, details ? probs_.get() : nullptr);
            if (timing_) timing_->corrected.record(stream);
            bytes = n * sizeof(int32_t);
            download(pinned ? pinned->ids.data() : out.ids.data(), ids_.get(), bytes, "DFlash2 proposal download");
            if (details) {
                const size_t cand_bytes = n * rowsize, score_bytes = n * top_k * sizeof(float);
                download(out.candidates.data(), top, cand_bytes, "DFlash2 candidates download");
                download(out.scores.data(), scores_.get(), score_bytes, "DFlash2 scores download");
                download(out.probs.data(), probs_.get(), score_bytes, "DFlash2 probability download");
                bytes += cand_bytes + 2 * score_bytes;
            }
        } else {
            download(pinned ? static_cast<void*>(pinned->candidates.data()) : out.candidates.data(),
                     top, n * rowsize, "DFlash2 candidates download");
            const size_t gate_bytes = n * rank * sizeof(float);
            download(pinned ? pinned->gates.data() : gate_rows.data(), gate, gate_bytes, "DFlash2 selector download");
            bytes = n * rowsize + gate_bytes;
        }
        if (timing_) timing_->downloaded.record(stream);
        if (read_hidden)
            download(out.hidden_states.data(), hidden_rows, n * hidden * sizeof(float), "DFlash2 hidden download");
        hip_check(hipGetLastError(), "DFlash2 kernel launch");
        stream.wait();

        const auto start = timing_ ? Timing::Clock::now() : Timing::Clock::time_point{};
        uint64_t pred_hits = 0, succ_hits = 0;
        if (mode_ == Placement::gpu) {
            if (pinned) std::copy_n(pinned->ids.data(), n, out.ids.data());
            for (int i = 0; i < n; ++i)
                if (out.ids[i] < 0 || out.ids[i] >= vocab) throw std::runtime_error("nonfinite GPU selector score");
        } else {
            if (pinned) std::memcpy(out.candidates.data(), pinned->candidates.data(), n * rowsize);
            const float* gates = pinned ? pinned->gates.data() : gate_rows.data();
            if (pred_cache_) {
                pred_hits = pred_cache_->hits();
                succ_hits = succ_cache_->hits();
                host_walk<true>(pred, succ, anchor, n, gates, out, decode);
                pred_hits = pred_cache_->hits() - pred_hits;
                succ_hits = succ_cache_->hits() - succ_hits;
            } else {
                host_walk<false>(pred, succ, anchor, n, gates, out, decode);
            }
        }
        if (timing_) {
            const double cpu_us = std::chrono::duration<double, std::micro>(Timing::Clock::now() - start).count();
            timing_->add(pos, anchor, n, bytes, mode_ == Placement::gpu, cpu_us, pred_hits, succ_hits, out);
        }
    }

private:
    template<bool Cache, class Tensor, class Decode>
    void host_walk(Tensor pred, Tensor succ, int anchor, int n, const float* gate_rows, Proposal& out, Decode decode) {
        int prev = anchor;
        std::array<float, rank> ag, b;
        for (int row = 0; row < n; ++row) {
            if constexpr (Cache) {
                const auto* r = pred_cache_->get(prev, [&](float* dst) { decode(pred, prev, dst); });
                std::copy_n(r, rank, ag.data());
            } else {
                decode(pred, prev, ag.data());
            }
            for (int r = 0; r < rank; ++r) ag[r] *= gate_rows[row * rank + r];
            auto& scores = out.scores[row];
            for (int i = 0; i < top_k; ++i) {
                const Candidate c = out.candidates[row][i];
                if (c.id < 0 || c.id >= vocab) throw std::runtime_error("nonfinite proposal logits");
                const float* r;
                if constexpr (Cache) r = succ_cache_->get(c.id, [&](float* dst) { decode(succ, c.id, dst); });
                else { decode(succ, c.id, b.data()); r = b.data(); }
                float pair = 0.f;
                for (int col = 0; col < rank; ++col) pair += ag[col] * r[col];
                scores[i] = softcap * std::tanh((pair + c.logit) / softcap);
                if (!std::isfinite(scores[i])) throw std::runtime_error("nonfinite selector score");
            }
            const auto best = std::max_element(scores.begin(), scores.end());
            prev = out.ids[row] = out.candidates[row][best - scores.begin()].id;
            auto& probs = out.probs[row];
            float sum = 0.f;
            for (int i = 0; i < top_k; ++i) sum += probs[i] = std::exp(scores[i] - *best);
            for (float& p : probs) p /= sum;
        }
    }
};

} // namespace dflash2::books
