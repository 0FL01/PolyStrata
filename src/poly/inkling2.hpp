// The looped Inkling-2-Mini GGUF on one HIP RDNA2 card.
#pragma once

#include "engine.hpp"

#include <array>
#include <vector>

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
    // the drafter's FFN down projection takes rows of 9216 (kDsFfn), its block 8 of them
    int widest() const override { return draft_ ? std::max(2 * kWidth, kDsFfn) : 2 * kWidth; }
    int streams() const override { return 1; }
    size_t recurrent_floats() const override;
    size_t extra_bytes() const override;
    void save_card_state(uint8_t* dst) override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void finish_logits(float* logits, int rows) override;
    void route(int il, const float* logits, int32_t, int* ids, float* w, float* tmp) const override;
    void moe_shared(const Layer&, int, int n) override;
    void moe_add(int n) override;
    void load_drafter() override;
    void draft_step(int32_t token, int i) override;
    void drafter_kept(int n) override;

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
        Dev aquery, fquery, message_query;
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
    bool loop_layout_ = false;  // inkling2_loop stores physical-layer geometry and per-layer control queries
    bool flash_ = false;    // STRATA_INKLING_FLASH=1 (load_geometry())
    bool decode_ = false;   // STRATA_INKLING_DECODE=1 (load_geometry())
    bool ds_attend_group_ = false;   // STRATA_INKLING_DSPARK_TARGET_GROUP=1: long verifier attention on gfx1031
    bool dense_q4_0_ = false;  // STRATA_INKLING_Q4_0_DENSE=1: group one-token Q4_0 projections on gfx1031
    bool norm_q8_ = false;  // STRATA_INKLING_NORM_Q8=1 (load_geometry()): a visit's stream norms and Q8_1 together
    bool moe_tail_ = false; // STRATA_INKLING_TAIL_MOE=1 (load_geometry()): a fused visit's expert sum in the tail
    bool cpu_sum_ = false;  // moe_add() left the CPU experts' sum for layer_tail()
    bool fo_clean_ = false; // layer_tail() left fo_ 0 for the next layer's experts (moe_shared())
    bool kv_stored_ = false;   // layer_head() stored the decode row's K and V (heads_kv_kernel) for attention()
    void* attq_ = nullptr;  // the attention's output as Q8_1 (attend_decode()), for the output projection
    void *kq8_ = nullptr, *vq8_ = nullptr;   // K's and V's own inputs as Q8_1 (project())
    // heads(): the three streams' current rows, messages, norm pairs, ranks, normed inputs (one after the other)
    // and those as Q8_1, for decode rows
    float *s3cur_ = nullptr, *s3msg_ = nullptr, *s3pair_ = nullptr, *s3rank_ = nullptr, *s3in_ = nullptr;
    void* s3q8_ = nullptr;
    void project(const Block& b, const float* xq, const float* xk, const float* xv, bool shared_kv, const void* q8,
                 const void* k8 = nullptr, const void* v8 = nullptr);
    bool heads(const Block& b, int il, bool swa, bool loop, int n, const Dev* const* norm);
    bool fused(int n) const;   // decode_ and fewer rows than a prompt batch
    float* snap_of(float* state) const;

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
    const float* message_query(const Block& b, int il, int stream) const;
    void control(const Block& b, int il, int stream, float* in, int n);
    void ring_state(int visit, int64_t pos, const uint8_t* src, uint8_t* dst);
    bool try_pack_state(uint8_t* dst);

    // The DSpark drafter (--draft-model, a `dflash` GGUF with dspark_hidden_out): Engine::draft_ without a block
    // layer.  It reads the model's rows entering visits kTap (ds_taps_, every call) and its normed output (hn_ + n_embd_);
    // the kept tokens' rows go into its own K/V (ds_cache_: layers 0 and 2 rings of kWindow rows, layer 1 every row),
    // and the last one's mix for the recurrent head into hn_'s first row.
    static constexpr int kDsLayers = 3, kDsFfn = 9216, kDsRank = 256, kDsBlock = 8;
    static constexpr int kTap[3] = {5, 23, 41};
    static constexpr bool ds_swa(int l) { return l != 1; }
    struct DsLayer {
        Dev attn_norm, ffn_norm, ctx_norm, q, k, v, o, k_ctx, v_ctx, q_norm, k_norm, k_ctx_norm, gate, up, down;
    };
    std::array<DsLayer, kDsLayers> ds_;
    std::array<Cache, kDsLayers> ds_cache_;
    Dev ds_out_norm_, ds_mask_, ds_rnn_norm_, ds_joint_, ds_joint_b_;
    float ds_mix_[kDsLayers + 1][4] = {};   // softmax of dspark_fuse's rows: the layers' and the recurrent head's
    const uint8_t* ds_w1_ = nullptr;        // markov_w1: F16 rows of kDsRank, in the mapped file
    std::vector<float> ds_hidden_, ds_hidden_b_;   // dspark_hidden_out [kWidth][kDsRank] and its bias, for prepare_head()
    float ds_eps_ = 0, ds_cap_ = 0, ds_theta_ = 0;
    bool ds_head_ready_ = false, ds_reuse_v_ = false;
    float* ds_taps_ = nullptr;              // the rows entering visits kTap of the last call, rows_ apart
    float *ds_x_ = nullptr, *ds_xa_ = nullptr, *ds_q_ = nullptr, *ds_qr_ = nullptr, *ds_k_ = nullptr,
          *ds_kr_ = nullptr, *ds_v_ = nullptr, *ds_ao_ = nullptr, *ds_out_ = nullptr, *ds_g_ = nullptr,
          *ds_u_ = nullptr, *ds_h_ = nullptr, *ds_cur_ = nullptr, *ds_base_ = nullptr, *ds_state_ = nullptr,
          *ds_w1row_ = nullptr, *ds_jx_ = nullptr, *ds_jo_ = nullptr, *ds_lat_ = nullptr, *ds_att_ = nullptr,
          *ds_b2_ = nullptr;
    float* ds_host_ = nullptr;              // pinned: a markov_w1 row, or prepare_head()'s rows, on their way
    uint16_t* ds_w2_ = nullptr;             // W_head W_hidden_out, half [kDsRank][n_vocab_]
    int ds_rows_ = 0;                       // the guesses a block has (base rows): min(7, window - 1)
    int ds_based_ = 0;                      // the block's rows whose head logits ds_base_ holds
    void load_drafter_shape(const Shards& g);
    void prepare_head();
    void ds_block(int32_t token);
    void ds_base(int to);
    size_t ds_row_bytes() const { return 2 * (kKey + kKey / 32 * sizeof(uint16_t)); }
    void ds_ring(int l, int64_t pos, const uint8_t* src, uint8_t* dst);
};

}  // namespace strata::poly
