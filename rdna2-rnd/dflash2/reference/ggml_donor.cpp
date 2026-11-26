// CPU harness for the literal graph builders extracted by extract_donor.py.
// External tensor views borrow caller-owned CPU storage; the graph owns its outputs.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace {
constexpr int H = 2304, W = 2048, D = 128, Q = 32, KV = 8, B = 8, FF = 6912;
constexpr int DYN = 576, K = 16, R = 256, V = 200058;
constexpr float eps = 1e-6f;

// The original functions require these graph-building fields. The harness is
// one sequence, no LoRA/ALiBi/MLA, and selects the non-flash FP32 CPU path.
enum { LLM_ARCH_GROK, LLM_FUSED_OP_FLASH_ATTN };
enum llama_swa_type { LLAMA_SWA_TYPE_NONE, LLAMA_SWA_TYPE_STANDARD, LLAMA_SWA_TYPE_CHUNKED, LLAMA_SWA_TYPE_SYMMETRIC };
using llama_pos = int32_t;
struct result {
    ggml_tensor *t_logits = nullptr, *t_embd = nullptr, *t_h_nextn = nullptr;
    struct fused_node { int type; ggml_tensor *tensor; int layer; };
    void add_fused_node(fused_node) {}
};
struct llama_model {
    ggml_tensor *dflash_selector_hidden, *dflash_selector_prev, *dflash_selector_next;
};
struct llm_graph_context {
    ggml_context *ctx0;
    ggml_cgraph *gf;
    result local_res;
    result *res = &local_res;
    struct { int n_seqs_unq = 1; } ubatch;
    int n_tokens = B, n_embd = H;
    struct {
        int dflash_conv_kernel_size = 2, dflash_conv_group_size = 16;
        int dflash_selector_top_k = K, dflash_selector_rank = R, dflash_block_size = B;
        bool dflash_selector_softcap_post = true, attn_soft_cap = false;
        float f_final_logit_softcapping = 30.0f, f_max_alibi_bias = 0.0f;
        float f_attn_out_scale = 1.0f, f_attn_logit_softcapping = 0.0f;
    } hparams;
    struct { bool flash_attn = false, offload_kqv = true; } cparams;
    int arch = -1;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_tensor *build_lora_mm(ggml_tensor *w, ggml_tensor *x) { return ggml_mul_mat(ctx0, w, x); }
    void cb(ggml_tensor *, const char *, int) const {}
    ggml_tensor *build_attn_mha(ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
                               ggml_tensor *, ggml_tensor *, int64_t, float, int) const;
};

#include "literal_donor.inc"

struct arena {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{
        ggml_init({128u * 1024 * 1024, nullptr, false}), ggml_free};
    llm_graph_context g{ctx.get(), ggml_new_graph(ctx.get())};
    std::vector<std::pair<ggml_tensor *, float *>> outputs;

    ggml_tensor *input(const void *data, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1,
                       ggml_type type = GGML_TYPE_F32) {
        ggml_set_no_alloc(ctx.get(), true);
        auto *t = ggml_new_tensor_3d(ctx.get(), type, ne0, ne1, ne2);
        ggml_set_no_alloc(ctx.get(), false);
        t->data = const_cast<void *>(data);
        return t;
    }
    ggml_tensor *linear(const float *w, int out, ggml_tensor *x) {
        return ggml_mul_mat(ctx.get(), input(w, x->ne[0], out), x);
    }
    ggml_tensor *norm(ggml_tensor *x, const float *w) {
        return ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), x, eps), input(w, x->ne[0]));
    }
    ggml_tensor *rope(ggml_tensor *x, ggml_tensor *pos) {
        return ggml_rope_ext(ctx.get(), x, pos, nullptr, D, GGML_ROPE_TYPE_NEOX, 1048576,
                             10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    void save(ggml_tensor *t, float *out) {
        ggml_build_forward_expand(g.gf, t);
        outputs.emplace_back(t, out);
    }
    int run(int threads) {
        const auto status = ggml_graph_compute_with_ctx(ctx.get(), g.gf, threads);
        if (status != GGML_STATUS_SUCCESS) {
            return int(status);
        }
        for (auto [t, out] : outputs) {
            GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
            std::memcpy(out, t->data, ggml_nbytes(t));
        }
        return 0;
    }
};

struct layer_weights {
    const float *attn_norm, *attn_conv_proj, *attn_conv_base, *q, *k, *v, *o, *q_norm, *k_norm;
    const float *ffn_norm, *ffn_conv_proj, *ffn_conv_base, *gate, *up, *down;
};
struct layer_outputs {
    float *ctx_k, *ctx_v, *attn_delta, *attn_preconv, *block_q, *block_k, *block_v;
    float *attn_out, *attn_residual, *ffn_delta, *ffn_preconv, *out;
};
} // namespace

extern "C" const char *dflash_donor_hashes() { return literal_donor_hashes; }

extern "C" int dflash_donor_project(const float *w, const float *x, int in, int out,
                                    int rows, int threads, float *res) {
    arena a;
    a.save(a.linear(w, out, a.input(x, in, rows)), res);
    return a.run(threads);
}

extern "C" int dflash_donor_norm(const float *x, const float *w, int rows, int threads, float *res) {
    arena a;
    a.save(a.norm(a.input(x, H, rows), w), res);
    return a.run(threads);
}

extern "C" int dflash_donor_fuse(const float *taps, const float *w, const float *norm,
                                 int rows, int threads, float *res) {
    arena a;
    a.save(a.norm(a.linear(w, H, a.input(taps, 4 * H, rows)), norm), res);
    return a.run(threads);
}

extern "C" int dflash_donor_conv(const float *x, const float *dynamic, const float *base,
                                 int side, int threads, float *out) {
    arena a;
    a.save(build_dflash2_conv(a.g, a.input(x, H, B), a.input(dynamic, DYN, B), a.input(base, H, 2, 2), side), out);
    return a.run(threads);
}

extern "C" int dflash_donor_layer(const float *x, const float *fused, const int32_t *pos, int nctx,
                                  const layer_weights *w, const layer_outputs *o, int threads) {
    arena a;
    auto *c = a.ctx.get();
    auto *ctx_pos = a.input(pos, nctx, 1, 1, GGML_TYPE_I32);
    auto *block_pos = a.input(pos + nctx, B, 1, 1, GGML_TYPE_I32);
    auto *cx = a.input(fused, H, nctx);
    auto *ck = a.rope(a.norm(ggml_reshape_3d(c, a.linear(w->k, KV * D, cx), D, KV, nctx), w->k_norm), ctx_pos);
    auto *cv = ggml_reshape_3d(c, a.linear(w->v, KV * D, cx), D, KV, nctx);

    auto *res = a.input(x, H, B);
    auto *h = a.norm(res, w->attn_norm);
    auto *da = a.linear(w->attn_conv_proj, DYN, h);
    auto *ba = a.input(w->attn_conv_base, H, 2, 2);
    h = build_dflash2_conv(a.g, h, da, ba, 0);
    a.save(h, o->attn_preconv);
    auto *q = a.rope(a.norm(ggml_reshape_3d(c, a.linear(w->q, Q * D, h), D, Q, B), w->q_norm), block_pos);
    auto *k = a.rope(a.norm(ggml_reshape_3d(c, a.linear(w->k, KV * D, h), D, KV, B), w->k_norm), block_pos);
    auto *v = ggml_reshape_3d(c, a.linear(w->v, KV * D, h), D, KV, B);

    std::vector<float> mask(size_t(nctx + B) * B);
    for (int j = 0; j < B; ++j) {
        for (int i = 0; i < nctx + B; ++i) {
            mask[size_t(j) * (nctx + B) + i] = is_masked_swa(W, LLAMA_SWA_TYPE_STANDARD, pos[i], pos[nctx + j])
                ? -INFINITY : 0.0f;
        }
    }
    auto *aout = a.g.build_attn_mha(q, ggml_concat(c, ck, k, 2), ggml_concat(c, cv, v, 2), nullptr,
                                   a.input(mask.data(), nctx + B, B), nullptr, nullptr, 0, 1.0f/std::sqrt(float(D)), 0);
    auto *after = ggml_add(c, res, build_dflash2_conv(a.g, a.linear(w->o, H, aout), da, ba, 1));
    h = a.norm(after, w->ffn_norm);
    auto *dm = a.linear(w->ffn_conv_proj, DYN, h);
    auto *bm = a.input(w->ffn_conv_base, H, 2, 2);
    h = build_dflash2_conv(a.g, h, dm, bm, 0);
    a.save(h, o->ffn_preconv);
    auto *mlp = ggml_mul(c, ggml_silu(c, a.linear(w->gate, FF, h)), a.linear(w->up, FF, h));
    a.save(ggml_add(c, after, build_dflash2_conv(a.g, a.linear(w->down, H, mlp), dm, bm, 1)), o->out);

    a.save(ck, o->ctx_k); a.save(cv, o->ctx_v);
    a.save(q, o->block_q); a.save(k, o->block_k); a.save(v, o->block_v);
    a.save(da, o->attn_delta); a.save(dm, o->ffn_delta);
    a.save(aout, o->attn_out); a.save(after, o->attn_residual);
    return a.run(threads);
}

extern "C" int dflash_donor_lattice(const float *h, const float *logits, const int32_t *tokens,
                                    const float *gate, const void *pred, const void *succ,
                                    int postcap, int threads, float *out) {
    arena a;
    a.g.hparams.dflash_selector_softcap_post = postcap != 0;
    a.g.res->t_embd = a.input(h, H, B);
    a.g.res->t_logits = a.input(logits, V, B);
    const llama_model model{a.input(gate, H, R), a.input(pred, R, V, 1, GGML_TYPE_BF16),
                            a.input(succ, R, V, 1, GGML_TYPE_BF16)};
    build_dflash2_selector(a.g, model, a.input(tokens, B, 1, 1, GGML_TYPE_I32));
    a.save(a.g.res->t_h_nextn, out);
    return a.run(threads);
}
