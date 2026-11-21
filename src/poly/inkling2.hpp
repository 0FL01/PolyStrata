// The looped Inkling-2-Mini GGUF on one HIP RDNA2 card.
#pragma once

#include "engine.hpp"

#include <array>

namespace strata::poly {

class Inkling2 final : public Engine {
public:
    bool sees() const override { return true; }
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void begin_pass() override;
    void load_layer(int il) override;
    void run_visits(int n) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return 2 * kWidth; }
    int streams() const override { return 1; }
    size_t recurrent_floats() const override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void finish_logits(float* logits, int rows) override;
    void route(int il, const float* logits, int32_t, int* ids, float* w, float* tmp) const override;
    void moe_shared(const Layer&, int, int n) override;
    void moe_add(int n) override;

private:
    static constexpr int kPhysical = 24, kVisits = 48;
    static constexpr int kWidth = 2304, kHead = 128, kHeads = 24, kKv = 8;
    static constexpr int kQuery = kHeads * kHead, kKey = kKv * kHead, kRelative = 16;
    static constexpr int kTaps = 4, kWindow = 1024, kBlock = 6, kRank = 64;
    static constexpr int kCore = 12, kStart = 6, kCoda = 42;
    static constexpr int kConvFloats = (kTaps - 1) * (2 * kKey + 2 * kWidth);
    static constexpr bool local(int visit) { return visit % kBlock != kBlock - 1; }
    static constexpr int physical(int visit) {
        return visit < kStart ? visit : visit < kCoda ? kStart + (visit - kStart) % kCore : visit - 2 * kCore;
    }

    struct Control { Dev down, gain, forget; };
    struct Block {
        Dev q, k, v, o, gate, sink, rel, table, qclip, rclip;
        Dev knorm, vnorm, kproj, vproj, kconv, vconv, aconv, fconv;
        Dev aquery, fquery;
        std::vector<float> bias;
        std::array<Control, 4> control;
    };
    /// Split Q8_0 planes (or half values only). Each plane's stride is in bytes.
    struct Plane { uint8_t* data = nullptr; size_t stride = 0; };
    struct Cache { std::array<Plane, 4> planes; };
    std::array<Block, kPhysical> blocks_;
    std::array<Cache, kVisits> caches_;
    Dev token_norm_, output_query_, loop_query_, loop_kv_query_;
    float qk_clip_ = 0, r_clip_ = 0, alpha_ = 0, cap_ = 0, expert_scale_ = 0;
    int floor_ = 0, ring_ = 0, visit_ = 0, depth_n_ = 0, attend_tiles_ = 0;
    bool flash_ = false;    // STRATA_INKLING_FLASH=1 (load_geometry())
    bool decode_ = false;   // STRATA_INKLING_DECODE=1 (load_geometry())
    bool cpu_sum_ = false;  // moe_add() left the CPU experts' sum for layer_tail()
    bool fo_clean_ = false; // layer_tail() left fo_ 0 for the next layer's experts (moe_shared())
    void* attq_ = nullptr;  // the attention's output as Q8_1 (attend_decode()), for the output projection
    bool fused(int n) const;   // decode_ and fewer rows than a prompt batch

    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *gate_ = nullptr, *rel_ = nullptr,
          *sink_ = nullptr, *taps_ = nullptr, *scores_ = nullptr, *partial_ = nullptr, *halo_ = nullptr,
          *chunks_ = nullptr, *taps2_ = nullptr;
    std::array<float*, 3> input_{};
    std::array<float*, 4> depth_{};
    float *prelude_ = nullptr, *bank0_ = nullptr, *bank1_ = nullptr, *msg_ = nullptr, *pair_ = nullptr,
          *rank_ = nullptr, *gain_ = nullptr, *forget_ = nullptr;

    bool q8() const { return opt_.kv_format == KvFmt::kQ8_0; }
    bool messages() const { return visit_ >= kStart + kCore && visit_ < kCoda; }
    size_t kv_bytes() const;
    void conv(float* x, float* state, const float* w, const float* dynamic, int width, int n);
    void store(const float* x, Cache& c, int side, int n);
    void attention(const Block& b, int n);
    void depth_mix(const float* query, float* dst, int first, int n);
    void control(const Block& b, int il, int stream, float* in, int n);
    void ring_state(int visit, int64_t pos, const uint8_t* src, uint8_t* dst);
};

}  // namespace strata::poly
