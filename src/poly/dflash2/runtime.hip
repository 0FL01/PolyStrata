// Port of rdna2-rnd/dflash2/native/runtime.hip at fa000e5c. The R&D build and
// its variants remain independent; this copy applies the exact-attention overlay.
#include "runtime.hpp"
#include "device.hpp"
#include "kernels.cuh"
#include "attention_exact.cuh"
#include "q40.cuh"
#include "native_books.cuh"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace dflash2 {
namespace {
namespace sk = strata::kernels;
constexpr int rank = 256, mask = 200056, threads = 256, padded_vocab = 201024;
constexpr float softcap = 30.f;
using strata::GgufFile;

int grid(int elements) { return (elements + threads - 1) / threads; }

const char* need_path(const char* s) {
    if (!s || !*s) throw std::invalid_argument("both GGUF paths are required");
    return s;
}

void select_device() {
    hip_check(hipSetDevice(0), "hipSetDevice(0)");
}

void check_arch() {
    hipDeviceProp_t props{};
    hip_check(hipGetDeviceProperties(&props, 0), "hipGetDeviceProperties(0)");
    if (std::string(props.gcnArchName).substr(0, 7) != "gfx1031" || props.warpSize != 32)
        throw std::runtime_error(std::string("gfx1031 wave32 required, got ") + props.gcnArchName);
}

void check_model(const GgufFile& draft, const GgufFile& target) {
    const auto arch_matches = [](const GgufFile& f, const char* name) {
        const auto* arch = f.get("general.architecture");
        return arch && arch->s == name;
    };
    if (!arch_matches(draft, "dflash") || !arch_matches(target, "inkling2_loop"))
        throw std::runtime_error("expected dflash and inkling2_loop GGUFs");
    for (const auto& [key, expected] : std::initializer_list<std::pair<const char*, double>>{
             {"block_count", layers}, {"embedding_length", hidden}, {"feed_forward_length", ffn},
             {"attention.head_count", q_heads}, {"attention.head_count_kv", kv_heads},
             {"attention.key_length", head_dim}, {"attention.value_length", head_dim},
             {"attention.layer_norm_rms_epsilon", eps}, {"attention.causal", 0},
             {"rope.freq_base", 10000}, {"attention.sliding_window", window},
             {"block_size", block}, {"conv_kernel_size", 2}, {"conv_group_size", group_size},
             {"selector_rank", rank}, {"selector_top_k", top_k}, {"final_logit_softcapping", softcap},
             {"selector_softcap_after_correction", 1}}) {
        const std::string name = std::string("dflash.") + key;
        const auto* m = draft.get(name);
        if (!m || !m->is_num() || m->num() != expected) throw std::runtime_error("unsupported " + name);
    }
    const auto check_array = [&](const char* name, std::initializer_list<uint64_t> expected) {
        const auto* m = draft.get(name);
        if (!m || m->items.size() != expected.size() ||
            !std::equal(expected.begin(), expected.end(), m->items.begin(),
                        [](uint64_t a, const strata::MetaValue& b) { return b.is_num() && b.num() == a; }))
            throw std::runtime_error(std::string("unsupported ") + name);
    };
    check_array("dflash.target_visits", {5, 23, 41, 48});
    check_array("dflash.attention.sliding_window_pattern", {1, 1, 1, 1, 1});
}

struct Tensor {
    const uint8_t* data = nullptr;
    size_t row_bytes = 0;
    size_t bytes = 0;
    uint32_t type = 0;
    int in = 0;
};

Tensor tensor(const GgufFile& file, const std::string& name, std::initializer_list<uint64_t> dims) {
    const auto* t = file.find(name);
    if (!t || t->shape != std::vector<uint64_t>(dims))
        throw std::runtime_error(file.path() + ": wrong or missing shape for " + name);
    const uint64_t total = strata::tensor_payload_bytes(*t);
    const auto size = file.file_size() - file.data_start();
    if (!total || t->offset > size || total > size - t->offset)
        throw std::runtime_error(file.path() + ": invalid payload for " + name);
    const uint64_t width = t->shape.front();
    return {file.tensor_data(*t), total / (t->elements() / width), total, t->type, int(width)};
}

void read_row(Tensor t, int row, float* out) {
    const auto* src = t.data + row * t.row_bytes;
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(out, src, t.row_bytes);
    } else {
        if (t.type >= GGML_TYPE_COUNT) throw std::runtime_error("unsupported row type");
        const auto* traits = ggml_get_type_traits(static_cast<ggml_type>(t.type));
        if (!traits->to_float) throw std::runtime_error("cannot decode GGUF row type");
        traits->to_float(src, out, t.in);
    }
}

template<class T>
void upload(DeviceBuffer<T>& dest, const void* src, size_t bytes, hipStream_t stream) {
    dest.allocate(bytes / sizeof(T));
    hip_check(hipMemcpyAsync(dest.get(), src, bytes, hipMemcpyHostToDevice, stream), "upload DFlash2 tensor");
}

struct Matrix {
    DeviceBuffer<uint8_t> owned;
    const void* data = nullptr;
    int type = 0, in = 0, out = 0;

    void load(Tensor t, int rows, hipStream_t stream, const void* borrowed = nullptr) {
        type = int(t.type);
        in = t.in;
        out = rows;
        if (!sk::native_mmvq_supported(type) && type != GGML_TYPE_BF16)
            throw std::runtime_error("unsupported projection type: " + std::to_string(type));
        if (!borrowed) upload(owned, t.data, t.row_bytes * rows, stream);
        data = borrowed ? borrowed : owned.get();
    }
    bool quantized() const { return type != GGML_TYPE_BF16; }
};

struct Layer {
    Matrix q, k, v, o, gate, up, down, d_attn, d_ffn;
    DeviceBuffer<float> attn_norm, ffn_norm, q_norm, k_norm, attn_base, ffn_base, k_ring, v_ring;
};

void load_f32(DeviceBuffer<float>& to, Tensor from, hipStream_t stream) {
    if (from.type != GGML_TYPE_F32) throw std::runtime_error("expected F32 norm or convolution base");
    upload(to, from.data, from.bytes, stream);
}

} // namespace

struct Runtime::Impl {
    GgufFile draft, target;
    Stream stream;
    Tensor embedding, pred, succ;
    books::Selector books;
    std::array<Layer, layers> layer;
    Matrix fusion, selector, head;
    DeviceBuffer<float> fusion_norm, out_norm, embd_norm;
    DeviceBuffer<double> freqs;
    DeviceBuffer<uint8_t> quant;
    DeviceBuffer<float> exact_scores;
    DeviceBuffer<float> concat, fused, x, h, conv_in, delta, q, q_rot, k, k_rot, v, attended, gate, up,
                        logits, gates;
    DeviceBuffer<Candidate> tile_top, top;
    std::array<float, hidden> mask_embedding{};
    int64_t next = 0;
    int warmup_rows = window;
    bool poisoned = false;

    Impl(const char* draft_path, const char* target_path, const void* shared_head)
        : draft(need_path(draft_path)), target(need_path(target_path)) {
        check_model(draft, target);
        fusion.load(tensor(draft, "fc.weight", {4 * hidden, hidden}), hidden, stream);
        selector.load(tensor(draft, "selector_hidden.weight", {hidden, rank}), rank, stream);
        load_f32(fusion_norm, tensor(draft, "enc_output_norm.weight", {hidden}), stream);
        load_f32(out_norm, tensor(draft, "output_norm.weight", {hidden}), stream);
        pred = tensor(draft, "selector_predecessor.weight", {rank, vocab});
        succ = tensor(draft, "selector_successor.weight", {rank, vocab});
        books.load(pred, succ, stream);

        auto head_tensor = tensor(target, "output.weight", {hidden, padded_vocab});
        if (shared_head && head_tensor.type != GGML_TYPE_Q6_K)
            throw std::runtime_error("borrowed output.weight must be raw Q6_K");
        head.load(head_tensor, vocab, stream, shared_head);
        embedding = tensor(target, "token_embd.weight", {hidden, padded_vocab});
        load_f32(embd_norm, tensor(target, "token_embd_norm.weight", {hidden}), stream);
        read_row(embedding, mask, mask_embedding.data());

        for (int i = 0; i < layers; ++i) {
            Layer& l = layer[i];
            const auto prefix = "blk." + std::to_string(i) + ".";
            const auto matrix = [&](Matrix& m, const char* name, int in, int out) {
                m.load(tensor(draft, prefix + name, {uint64_t(in), uint64_t(out)}), out, stream);
            };
            const auto f32 = [&](DeviceBuffer<float>& d, const char* name, std::initializer_list<uint64_t> dims) {
                load_f32(d, tensor(draft, prefix + name, dims), stream);
            };
            matrix(l.q, "attn_q.weight", hidden, q_width);
            matrix(l.k, "attn_k.weight", hidden, kv_width);
            matrix(l.v, "attn_v.weight", hidden, kv_width);
            matrix(l.o, "attn_output.weight", q_width, hidden);
            matrix(l.gate, "ffn_gate.weight", hidden, ffn);
            matrix(l.up, "ffn_up.weight", hidden, ffn);
            matrix(l.down, "ffn_down.weight", ffn, hidden);
            matrix(l.d_attn, "attn_conv_proj.weight", hidden, conv_width);
            matrix(l.d_ffn, "ffn_conv_proj.weight", hidden, conv_width);
            f32(l.attn_norm, "attn_norm.weight", {hidden});
            f32(l.ffn_norm, "ffn_norm.weight", {hidden});
            f32(l.q_norm, "attn_q_norm.weight", {head_dim});
            f32(l.k_norm, "attn_k_norm.weight", {head_dim});
            f32(l.attn_base, "attn_conv_base", {hidden, 2, 2});
            f32(l.ffn_base, "ffn_conv_base", {hidden, 2, 2});
            l.k_ring.allocate(window * kv_width);
            l.v_ring.allocate(window * kv_width);
        }

        double frequencies[head_dim / 2];
        for (int i = 0; i < head_dim / 2; ++i) frequencies[i] = std::pow(10000., -2. * i / head_dim);
        upload(freqs, frequencies, sizeof(frequencies), stream);
        stream.wait();

        quant.allocate(sk::native_q8_1_bytes(4 * hidden, block));
        concat.allocate(block * 4 * hidden);
        for (auto* b : {&fused, &x, &h, &conv_in}) b->allocate(block * hidden);
        for (auto* b : {&q, &q_rot, &attended}) b->allocate(block * q_width);
        for (auto* b : {&k, &k_rot, &v}) b->allocate(block * kv_width);
        for (auto* b : {&gate, &up}) b->allocate(block * ffn);
        delta.allocate(block * conv_width);
        logits.allocate(proposals * vocab);
        gates.allocate(proposals * rank);
        exact_scores.allocate(block * q_heads * exact_score_stride);
        tile_top.allocate(proposals * tiles * top_k);
        top.allocate(proposals * top_k);
    }

    void prepare(const float* src, int rows, const Matrix& m, bool extra = false) {
        if (m.quantized() || extra) sk::native_quantize_q8_1(src, quant.get(), m.in, rows, stream);
    }

    void project(const Matrix& m, const float* src, float* dst, int rows) {
        static const bool wave8=[] { const char* v=std::getenv("DF2_WAVE8"); return v && std::string_view(v)=="1"; }();
        if (wave8 && m.type==GGML_TYPE_Q4_0 && m.in==2304 && rows==8) {
            df_proj::wave8<false,true><<<(m.out+3)/4,128,0,stream>>>(
                static_cast<const df_proj::Q40*>(m.data), reinterpret_cast<const df_proj::Q81*>(quant.get()), dst,m.out);
            return;
        }
        if (m.type == GGML_TYPE_BF16) {
            sk::bf16_gemv_fp32_mmvf_multi(src, m.in, static_cast<const uint16_t*>(m.data), dst, m.out,
                                          m.in, m.out, rows, stream);
        } else if (m.type == GGML_TYPE_Q6_K &&
                   sk::native_q6_k_mmvq_warps_cols(m.data, quant.get(), dst, m.in, m.out, rows, stream)) {
            return;
        } else if (m.type == GGML_TYPE_Q8_0 && m.out >= 8192 &&
                   sk::native_q8_0_mmvq_warps_cols(m.data, quant.get(), dst, m.in, m.out, rows, stream)) {
            return;
        } else {
            sk::native_mmvq(m.type, m.data, quant.get(), dst, m.in, m.out, rows, stream);
        }
    }

    void linear(const Matrix& m, const float* src, float* dst, int rows) {
        prepare(src, rows, m);
        project(m, src, dst, rows);
    }

    void barrier() {
        hip_check(hipGetLastError(), "DFlash2 kernel launch");
        stream.wait();
    }

    void append(int64_t pos, int rows, const std::array<const float*, 4>& taps) {
        if (poisoned) throw std::runtime_error("draft handle is unusable after a device failure");
        const bool window_warmup = next == 0 && pos > 0 && rows >= window;
        if ((!window_warmup && pos != next) || pos < 0 || pos > max_positions ||
            rows < 0 || rows > max_positions - pos ||
            (rows && std::any_of(taps.begin(), taps.end(), [](const void* p) { return !p; })))
            throw std::invalid_argument("append requires contiguous device rows (nonzero initial pos needs 2048 rows)");
        if (!rows) return;

        // The oldest rows of a long append cannot be visible to any later proposal.
        try {
            for (int j = std::max(0, rows - window); j < rows; j += block) {
                const int n = std::min(block, rows - j), start = int(pos + j);
                const size_t off = size_t(j) * hidden;
                gather_taps<<<grid(n * 4 * hidden), threads, 0, stream>>>(
                    taps[0] + off, taps[1] + off, taps[2] + off, taps[3] + off, concat.get(), n);
                linear(fusion, concat.get(), fused.get(), n);
                rms_norm<<<n, threads, 0, stream>>>(fused.get(), fusion_norm.get(), h.get(), hidden);
                bool quantized = false;
                for (const auto& l : layer) quantized |= l.k.quantized() || l.v.quantized();
                prepare(h.get(), n, layer[0].k, quantized);
                for (auto& l : layer) {
                    project(l.k, h.get(), k.get(), n);
                    project(l.v, h.get(), v.get(), n);
                    norm_rope<<<n * kv_heads, head_dim, 0, stream>>>(k.get(), l.k_norm.get(), freqs.get(),
                                                                   l.k_ring.get(), kv_heads, start, true);
                    copy_v_ring<<<grid(n * kv_width), threads, 0, stream>>>(v.get(), l.v_ring.get(), n, start);
                }
            }
            barrier(); // The producer can reuse its tap buffers only after this completes.
        } catch (...) {
            poisoned = true;
            try { stream.wait(); } catch (...) {}
            throw;
        }
        if (window_warmup) warmup_rows = 0;
        warmup_rows = std::min(window, warmup_rows + std::min(rows, window));
        next = pos + rows;
    }

    void propose(int64_t pos, int anchor, int n, Proposal& out, bool read_hidden, bool details) {
        if (poisoned) throw std::runtime_error("draft handle is unusable after a device failure");
        if (warmup_rows < window)
            throw std::invalid_argument("draft requires 2048 committed warmup rows after nonzero reset");
        if (pos != next || pos + block > max_positions || anchor < 0 || anchor >= vocab || n < 1 || n > proposals)
            throw std::invalid_argument("invalid proposal position, anchor or count");

        std::array<float, block * hidden> embedding_rows;
        read_row(embedding, anchor, embedding_rows.data());
        for (int b = 1; b < block; ++b)
            std::copy(mask_embedding.begin(), mask_embedding.end(), embedding_rows.data() + b * hidden);

        try {
            hip_check(hipMemcpyAsync(h.get(), embedding_rows.data(), sizeof(embedding_rows), hipMemcpyHostToDevice,
                                     stream), "DFlash2 embedding upload");
            rms_norm<<<block, threads, 0, stream>>>(h.get(), embd_norm.get(), x.get(), hidden);
            for (auto& l : layer) {
                rms_norm<<<block, threads, 0, stream>>>(x.get(), l.attn_norm.get(), h.get(), hidden);
                linear(l.d_attn, h.get(), delta.get(), block);
                conv<<<grid(block * hidden), threads, 0, stream>>>(h.get(), delta.get(), l.attn_base.get(),
                                                                  conv_in.get(), nullptr, 0);
                prepare(conv_in.get(), block, l.q, l.k.quantized() || l.v.quantized());
                project(l.q, conv_in.get(), q.get(), block);
                project(l.k, conv_in.get(), k.get(), block);
                project(l.v, conv_in.get(), v.get(), block);
                norm_rope<<<block * q_heads, head_dim, 0, stream>>>(q.get(), l.q_norm.get(), freqs.get(),
                                                                   q_rot.get(), q_heads, int(pos), false);
                norm_rope<<<block * kv_heads, head_dim, 0, stream>>>(k.get(), l.k_norm.get(), freqs.get(),
                                                                    k_rot.get(), kv_heads, int(pos), false);
                attend_exact(q_rot.get(), l.k_ring.get(), l.v_ring.get(), k_rot.get(), v.get(),
                             exact_scores.get(), attended.get(), int(pos), stream);
                linear(l.o, attended.get(), h.get(), block);
                conv<<<grid(block * hidden), threads, 0, stream>>>(h.get(), delta.get(), l.attn_base.get(),
                                                                  x.get(), x.get(), 1);

                rms_norm<<<block, threads, 0, stream>>>(x.get(), l.ffn_norm.get(), h.get(), hidden);
                linear(l.d_ffn, h.get(), delta.get(), block);
                conv<<<grid(block * hidden), threads, 0, stream>>>(h.get(), delta.get(), l.ffn_base.get(),
                                                                  conv_in.get(), nullptr, 0);
                prepare(conv_in.get(), block, l.gate, l.up.quantized());
                project(l.gate, conv_in.get(), gate.get(), block);
                project(l.up, conv_in.get(), up.get(), block);
                swiglu<<<grid(block * ffn), threads, 0, stream>>>(gate.get(), up.get());
                linear(l.down, gate.get(), h.get(), block);
                conv<<<grid(block * hidden), threads, 0, stream>>>(h.get(), delta.get(), l.ffn_base.get(),
                                                                  x.get(), x.get(), 1);
            }
            rms_norm<<<n, threads, 0, stream>>>(x.get() + hidden, out_norm.get(), h.get(), hidden);
            prepare(h.get(), n, head, selector.quantized());
            project(head, h.get(), logits.get(), n);
            project(selector, h.get(), gates.get(), n);
            topk_tiles<<<dim3(tiles, n), threads, 0, stream>>>(logits.get(), tile_top.get());
            topk_merge<<<n, threads, 0, stream>>>(tile_top.get(), top.get());
            books.run(pred, succ, stream, top.get(), gates.get(), h.get(), pos, anchor, n, out,
                      read_hidden, details, read_row);
        } catch (...) {
            poisoned = true;
            try { stream.wait(); } catch (...) {}
            throw;
        }
    }
};

Runtime::Runtime(const char* draft, const char* target, const void* head) {
    select_device();
    check_arch();
    impl_ = std::make_unique<Impl>(draft, target, head);
}
Runtime::~Runtime() { (void)hipSetDevice(0); }
bool Runtime::usable() const { return !impl_->poisoned; }

void Runtime::reset(int64_t next_pos) {
    if (next_pos < 0 || next_pos > max_positions)
        throw std::invalid_argument("invalid draft reset position");
    impl_->next = next_pos;
    // Old ring slots remain invisible until genuinely overwritten.
    impl_->warmup_rows = next_pos == 0 ? window : 0;
}
void Runtime::append(int64_t pos, int rows, const std::array<const float*, 4>& taps) {
    select_device();
    impl_->append(pos, rows, taps);
}
void Runtime::propose(int64_t pos, int32_t anchor, int count, Proposal& out, bool read_hidden, bool details) {
    select_device();
    impl_->propose(pos, anchor, count, out, read_hidden, details);
}

} // namespace dflash2
