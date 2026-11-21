// Inkling-2-Mini's loop visits, short convolutions and relative attention. The GGUF has 24 sets of weights;
// the three core traversals reuse those weights and expert slots, not the sequence state.
#include "inkling2.hpp"

#include "inkling_attend.hpp"
#include "runtime.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int kThreads = 128;
constexpr int kValueTile = 128;
int div_up(int a, int b) { return (a + b - 1) / b; }

template <bool Maximum> __device__ float wave_reduce(float x) {
    for (int d = 16; d > 0; d /= 2) {
        const float y = __shfl_down(x, d, 32);
        if constexpr (Maximum) x = fmaxf(x, y);
        else x += y;
    }
    return x;
}

template <bool Maximum = false> __device__ float block_reduce(float x, float* waves) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
    x = wave_reduce<Maximum>(x);
    if (lane == 0) waves[wave] = x;
    __syncthreads();
    if (wave == 0) {
        x = lane < kThreads / 32 ? waves[lane] : Maximum ? -INFINITY : 0.f;
        x = wave_reduce<Maximum>(x);
        if (lane == 0) waves[0] = x;
    }
    __syncthreads();
    const float result = waves[0];
    __syncthreads();
    return result;
}

struct MixInputs { const float* x[5]; };

// A depth query attends all finished blocks and the current partial block. RMS-normalized states are values too.
__global__ void mix_kernel(MixInputs in, int count, const float* query, float* out, int width) {
    const size_t row = (size_t) blockIdx.x * width;
    __shared__ float waves[kThreads / 32];
    float scale[5], w[5], top = -INFINITY;
    for (int j = 0; j < count; ++j) {
        const float* x = in.x[j] + row;
        float ss = 0.f, dot = 0.f;
        for (int i = threadIdx.x; i < width; i += kThreads) {
            ss += x[i] * x[i];
            dot += x[i] * query[i];
        }
        scale[j] = rsqrtf(block_reduce(ss, waves) / width + 1.e-6f);
        w[j] = block_reduce(dot, waves) * scale[j] * rsqrtf((float) width);
        top = fmaxf(top, w[j]);
    }
    float sum = 0.f;
    for (int j = 0; j < count; ++j) sum += (w[j] = expf(w[j] - top));
    for (int i = threadIdx.x; i < width; i += kThreads) {
        float val = 0.f;
        for (int j = 0; j < count; ++j) val += w[j] * scale[j] * in.x[j][row + i];
        out[row + i] = val / sum;
    }
}

__global__ void control_kernel(float* cur, const float* msg, const float* gain, const float* forget, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) cur[i] = (1.f - tanhf(forget[i])) * cur[i] + tanhf(gain[i]) * msg[i];
}

// One thread owns one channel and keeps its raw input history across the chunk. In GGUF the static taps
// run oldest to newest; the input-dependent taps of each KV head start with the current token.
__global__ void conv_kernel(float* x, float* state, const float* w, const float* dynamic, int width, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    float* h = state + 3 * i;
    float a = h[0], b = h[1], c = h[2];
    w += 4 * i;
    const float scale = dynamic ? 1.f / 3.f : 0.4472135954999579f;
    for (int t = 0; t < n; ++t) {
        const size_t at = (size_t) t * width + i;
        const float d = x[at];
        float s = a * w[0] + b * w[1] + c * w[2] + d * w[3];
        if (dynamic) {
            const float* p = dynamic + 4 * (t * (width / 128) + i / 128);
            s += d * p[0] + c * p[1] + b * p[2] + a * p[3];
        }
        x[at] = (d + s) * scale;
        a = b;
        b = c;
        c = d;
    }
    h[0] = a;
    h[1] = b;
    h[2] = c;
}

// A prompt batch's convolution in spans of kConvSpan tokens, a thread a channel and span: conv_halo_kernel keeps the
// three raw inputs before each span (the state's before the first) and conv_span_kernel computes each span as
// conv_kernel does, with the same sums, so the results are conv_kernel's bit for bit.
constexpr int kConvSpan = 64;
__global__ void conv_halo_kernel(const float* x, const float* state, float* halo, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, span = blockIdx.y;
    if (i >= width) return;
    float* to = halo + ((size_t) span * width + i) * 3;
    for (int j = 0; j < 3; ++j) {
        const int t = span * kConvSpan - 3 + j;
        to[j] = t >= 0 ? x[(size_t) t * width + i] : state[3 * i + 3 + t];
    }
}

__global__ void conv_span_kernel(float* x, float* state, const float* halo, const float* w, const float* dynamic,
                                 int width, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, span = blockIdx.y;
    if (i >= width) return;
    const float* from = halo + ((size_t) span * width + i) * 3;
    float a = from[0], b = from[1], c = from[2];
    w += 4 * i;
    const float scale = dynamic ? 1.f / 3.f : 0.4472135954999579f;
    const int first = span * kConvSpan, end = min(n, first + kConvSpan);
    for (int t = first; t < end; ++t) {
        const size_t at = (size_t) t * width + i;
        const float d = x[at];
        float s = a * w[0] + b * w[1] + c * w[2] + d * w[3];
        if (dynamic) {
            const float* p = dynamic + 4 * (t * (width / 128) + i / 128);
            s += d * p[0] + c * p[1] + b * p[2] + a * p[3];
        }
        x[at] = (d + s) * scale;
        a = b;
        b = c;
        c = d;
    }
    if (end == n) {
        float* h = state + 3 * i;
        h[0] = a;
        h[1] = b;
        h[2] = c;
    }
}

__global__ void head_norm_kernel(float* x, const float* w, int width, int heads, float clip, float eps,
                                 int64_t pos, int floor, float alpha) {
    __shared__ float waves[kThreads / 32];
    const int i = threadIdx.x, head = blockIdx.x, row = blockIdx.y;
    float* dest = x + ((size_t) row * heads + head) * width;
    const float v = i < width ? dest[i] * (w ? w[head * width + i] : 1.f) : 0.f;
    const float ss = block_reduce(v * v, waves);
    const float s = clip > 0.f ? fminf(1.f, clip / (sqrtf(ss) + 1.e-6f)) : rsqrtf(ss / width + eps);
    const float tau = floor > 0 ? 1.f + alpha * logf(fmaxf(1.f, float(pos + row + 1) / floor)) : 1.f;
    if (i < width) dest[i] = v * s * tau;
}

}  // namespace

void Inkling2::load_geometry() {
    int device = 0;
    cudaDeviceProp p{};
    ck(cudaGetDevice(&device), "Inkling device");
    ck(cudaGetDeviceProperties(&p, device), "Inkling device properties");
    bool rdna2 = false;
    for (const char* gcn : {"gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"})
        rdna2 = rdna2 || kernels::gfx_arch_is(p.gcnArchName, gcn);
    if (!rdna2) throw std::runtime_error("Inkling-2-Mini requires a HIP RDNA2 card");
    if (opt_.batch || !opt_.layer_split.empty() || !opt_.draft_model.empty() || opt_.tail_skip || opt_.skip_weight > 0.f)
        throw std::runtime_error("Inkling-2-Mini requires one sequence and one card, without draft or expert skipping");
    opt_.window = 1;
    opt_.draft = false;
    const char* flash = std::getenv("STRATA_INKLING_FLASH");
    flash_ = flash && flash[0] == '1' && flash[1] == '\0';
    if (flash_) std::fputs("strata-poly: STRATA_INKLING_FLASH=1: fused attention for prompt batches\n", stderr);
    if (!q8() && opt_.kv_format != KvFmt::kFp16)
        throw std::runtime_error("Inkling-2-Mini supports Q8_0 or FP16 KV");
    load_shape();

    const auto need = [](bool ok, const char* key) {
        if (!ok) throw std::runtime_error(std::string("not an Inkling-2-Mini layout: ") + key);
    };
    const auto integer = [&](const char* key, int expected) { need(shape_i(key) == expected, key); };
    need(n_layer_ == kVisits && n_embd_ == kWidth && n_expert_ == 128 && n_used_ == 8 && n_ff_exp_ == 1152,
         "layers, width or experts");
    integer("num_loops", 3);
    integer("loop.core_start", kStart);
    integer("loop.core_end", kStart + kCore);
    integer("loop.gate_rank", kRank);
    integer("attn_res.block_size", kBlock);
    integer("attention.sliding_window", kWindow);
    integer("attention.relative_length", kRelative);
    integer("attention.relative_extent", kWindow);
    integer("shortconv.l_cache", kTaps);
    integer("rope.dimension_count", 0);
    for (const char* key : {"attention.key_length", "attention.value_length", "attention.key_length_swa",
                            "attention.value_length_swa"}) integer(key, kHead);
    const auto array = [&](const char* key, auto expected) {
        const auto* value = meta_in(*gg_, key);
        need(value && value->count == kVisits && value->items.size() == kVisits, key);
        for (int i = 0; i < kVisits; ++i) need(value->items[i].u == (uint64_t) expected(i), key);
    };
    array("attention.head_count", [](int) { return kHeads; });
    array("attention.head_count_kv", [](int) { return kKv; });
    array("attention.sliding_window_pattern", [](int i) { return local(i); });
    padded_vocab_ = n_vocab_;
    n_vocab_ = shape_i("vocab_size_unpadded");
    const auto* tokens = gg_->get("tokenizer.ggml.tokens");
    need(n_vocab_ == 200058 && padded_vocab_ == 201024 && tokens && tokens->count == (uint64_t) n_vocab_, "vocabulary");
    need(gg_->find("output.weight") && !draft_, "output or draft");
    qk_clip_ = shape_f("attention.qk_clip");
    r_clip_ = shape_f("attention.relative_clip");
    floor_ = shape_i("attention.log_scale_floor");
    alpha_ = shape_f("attention.log_scale_alpha");
    expert_scale_ = shape_f("expert_weights_scale");
    cap_ = shape_f("final_logit_softcapping");
    situ_ = {shape_f("activation.situ_beta"), shape_f("activation.situ_linear_beta")};
    for (float value : {rms_eps_, qk_clip_, r_clip_, expert_scale_, cap_, situ_.beta, situ_.linear_beta})
        need(std::isfinite(value) && value > 0.f, "numerical parameters");
    need(std::isfinite(alpha_) && alpha_ >= 0.f && floor_ > 0, "log scale");
    n_layer_ = kPhysical;
    looped_ = true;
    clamp_exp_.assign(kPhysical + 1, INFINITY);
}

void Inkling2::begin_pass() {
    const auto part = [&](const char* name, int rows) { return upload(gg_->find(name), name, kWidth, rows, kF32); };
    token_norm_ = part("token_embd_norm.weight", 1);
    output_query_ = part("output_res_query.weight", 1);
    loop_query_ = part("loop_res_query.weight", 2 * kCore);
    loop_kv_query_ = part("loop_kv_res_query.weight", 2 * kCore);
}

void Inkling2::load_layer(int il) {
    Block& b = blocks_[il];
    Layer& L = layers_[il];
    const auto part = [&](const char* name, int cols, int rows = 1, Want want = kF32) {
        return upload(tensor(il, name), blk(il, name), cols, rows, want);
    };
    b.q = part("attn_q.weight", kWidth, kQuery, kAnyMat);
    b.k = part("attn_k.weight", kWidth, kKey, kAnyMat);
    b.v = part("attn_v.weight", kWidth, kKey, kAnyMat);
    b.o = part("attn_output.weight", kQuery, kWidth, kAnyMat);
    b.gate = part("attn_gate.weight", kWidth, kQuery, kAnyMat);
    b.rel = part("attn_rel.weight", kWidth, kRelative * kHeads, kAnyMat);
    b.sink = part("attn_sink_proj.weight", kWidth, kHeads);
    b.qclip = part("attn_q_clip.weight", kHead, kHeads);
    b.rclip = part("attn_rel_clip.weight", kRelative, kHeads);
    b.table = part("attn_rel_bias.weight", kRelative, kWindow);
    b.kconv = part("attn_k_conv.weight", kTaps, kKey);
    b.vconv = part("attn_v_conv.weight", kTaps, kKey);
    b.aconv = part("attn_post_conv.weight", kTaps, kWidth);
    b.fconv = part("ffn_post_conv.weight", kTaps, kWidth);
    if (local(il)) L.attn_norm = part("attn_norm.weight", kWidth);
    else {
        L.attn_norm = part("attn_q_inp_norm.weight", kWidth);
        b.knorm = part("attn_k_inp_norm.weight", kWidth);
        b.vnorm = part("attn_v_inp_norm.weight", kWidth);
        b.kproj = part("attn_k_conv_proj.weight", kWidth, kTaps * kKv);
        b.vproj = part("attn_v_conv_proj.weight", kWidth, kTaps * kKv);
    }
    if (il > 0) b.aquery = part("attn_res_query.weight", kWidth, local(il) ? 1 : 3);
    b.fquery = part("ffn_res_query.weight", kWidth);
    L.ffn_norm = part("ffn_norm.weight", kWidth);
    if (!tensor(il, "ffn_gate_up_exps.weight"))
        throw std::runtime_error(blk(il, "ffn_gate_up_exps.weight is required for Inkling-2-Mini"));
    load_moe(il, false);
    if ((L.gu_type == GGML_TYPE_NVFP4 && L.h_s_gate.empty()) ||
        (L.d_type == GGML_TYPE_NVFP4 && L.h_s_down.empty()))
        throw std::runtime_error(blk(il, "NVFP4 experts require their scales"));
    const auto* bias = tensor(il, "exp_probs_b.bias");
    if (!bias || bias->type != kTypeF32 || bias->elements() != (uint64_t) n_expert_)
        throw std::runtime_error(blk(il, "exp_probs_b.bias must be an F32 expert vector"));
    b.bias.resize(n_expert_);
    home(*bias).payload(*bias);
    std::memcpy(b.bias.data(), home(*bias).data(*bias), b.bias.size() * sizeof(float));

    if (il >= kStart && il < kStart + kCore) {
        const char* stream[4] = {"attn", "k", "v", "mlp"};
        for (int i = 0; i < 4; ++i) {
            const std::string pre = std::string("loop_") + stream[i];
            b.control[i] = {part((pre + "_down.weight").c_str(), 2 * kWidth, kRank),
                            part((pre + "_gain.weight").c_str(), kRank, kWidth),
                            part((pre + "_forget.weight").c_str(), kRank, kWidth)};
        }
    }
}

size_t Inkling2::recurrent_floats() const { return (size_t) kVisits * kConvFloats; }
size_t Inkling2::kv_bytes() const {
    return 2 * (q8() ? kKey + (kKey / 32) * sizeof(uint16_t) : kKey * sizeof(uint16_t));
}
size_t Inkling2::card_row_bytes() const { return kVisits / kBlock * kv_bytes(); }

void Inkling2::alloc_caches() {
    ring_ = std::min(opt_.max_context, kWindow + rows_ - 1);
    auto* rings = (uint8_t*) seq_alloc((size_t) (kVisits - kVisits / kBlock) * ring_ * kv_bytes(), "Inkling SWA KV");
    const size_t wide = kKey * (q8() ? 1 : sizeof(uint16_t));
    const size_t scale = q8() ? kKey / 32 * sizeof(uint16_t) : 0;
    for (int i = 0; i < kVisits; ++i) {
        for (int j = 0; j < 4; ++j) {
            const size_t stride = j % 2 ? scale : wide;
            auto& p = caches_[i].planes[j];
            p = {nullptr, stride};
            if (!stride) continue;
            if (local(i)) {
                p.data = rings;
                rings += (size_t) ring_ * stride;
            } else p.data = (uint8_t*) kv_alloc(stride, 1);
        }
    }
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context) throw std::logic_error("save_rows: outside context");
    if (!n) return;
    ck(cudaStreamSynchronize(cs(st_)), "save Inkling rows");
    for (int i = 0; i < kVisits; ++i) if (!local(i)) {
        for (const auto& p : caches_[i].planes) if (p.stride) {
            const size_t bytes = (size_t) n * p.stride;
            ck(cudaMemcpy(dst, p.data + first * p.stride, bytes, cudaMemcpyDeviceToHost), "save Inkling rows");
            dst += bytes;
        }
    }
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: outside saved rows");
    if (!n) return;
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load Inkling rows");
    for (int i = 0; i < kVisits; ++i) if (!local(i)) {
        for (const auto& p : caches_[i].planes) if (p.stride) {
            ck(cudaMemcpy(p.data + first * p.stride, src + (first - from) * p.stride, (size_t) n * p.stride,
                          cudaMemcpyHostToDevice), "load Inkling rows");
            src += (size_t) count * p.stride;
        }
    }
}

size_t Inkling2::extra_bytes() const {
    return (kVisits - kVisits / kBlock) * (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
}

void Inkling2::ring_state(int visit, int64_t pos, const uint8_t* src, uint8_t* dst) {
    const int64_t keep = std::min(opt_.max_context, kWindow - 1);
    const int64_t first = std::max<int64_t>(0, pos - keep);
    for (const auto& p : caches_[visit].planes) if (p.stride) {
        if (dst) std::memset(dst, 0, (size_t) (keep - (pos - first)) * p.stride);
        for (int64_t i = first; i < pos;) {
            const int64_t at = i % ring_, n = std::min(pos - i, ring_ - at);
            const size_t off = (size_t) (i - (pos - keep)) * p.stride, bytes = (size_t) n * p.stride;
            if (dst) ck(cudaMemcpy(dst + off, p.data + at * p.stride, bytes, cudaMemcpyDeviceToHost), "save Inkling SWA");
            else ck(cudaMemcpy(p.data + at * p.stride, src + off, bytes, cudaMemcpyHostToDevice), "load Inkling SWA");
            i += n;
        }
        if (dst) dst += keep * p.stride;
        else src += keep * p.stride;
    }
}

void Inkling2::save_layer(int il, int64_t pos, uint8_t* dst) {
    for (int i = 0; i < kVisits; ++i) if (physical(i) == il) {
        const size_t at = (size_t) i * kConvFloats;
        ck(cudaMemcpy(dst + at * sizeof(float), state_ + at, kConvFloats * sizeof(float), cudaMemcpyDeviceToHost),
           "save Inkling convolution");
        if (local(i)) {
            const size_t offset = (i - i / kBlock) * (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
            ring_state(i, pos, nullptr, dst + state_n_ * sizeof(float) + offset);
        }
    }
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
    const size_t one = (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
    for (int i = 0; i < kVisits; ++i) if (local(i)) {
        ring_state(i, pos, src, nullptr);
        src += one;
    }
}

void Inkling2::assign_scratch() {
    const size_t n = rows_, width = n * kWidth;
    q_ = fbuf(n * kQuery);
    k_ = fbuf(n * kKey);
    v_ = fbuf(n * kKey);
    att_ = fbuf(n * kQuery);
    gate_ = fbuf(n * kQuery);
    rel_ = fbuf(n * kHeads * kRelative);
    sink_ = fbuf(n * kHeads);
    taps_ = fbuf(n * kKv * kTaps);
    halo_ = fbuf((size_t) div_up(rows_, kConvSpan) * 3 * kWidth);
    // A tile takes kValueTile scores and one kHead-wide partial value vector per query head. Keep room
    // for a single query at the configured limit, but share that room across queries on shorter prefixes.
    constexpr int target_tiles = (1 << 20) / (kHeads * kValueTile);
    attend_tiles_ = std::max(div_up(opt_.max_context, kValueTile),
                             std::min(target_tiles, rows_ * div_up(opt_.max_context, kValueTile)));
    scores_ = fbuf(inkling::plain_score_floats(attend_tiles_));
    partial_ = fbuf(inkling::plain_partial_floats(attend_tiles_));
    for (auto& x : input_) x = fbuf(width);
    for (auto& x : depth_) x = fbuf(width);
    prelude_ = fbuf(width);
    bank0_ = fbuf(width);
    bank1_ = fbuf(width);
    msg_ = fbuf(width);
    pair_ = fbuf(2 * width);
    rank_ = fbuf(n * kRank);
    gain_ = fbuf(width);
    forget_ = fbuf(width);
}

void Inkling2::conv(float* x, float* state, const float* w, const float* dynamic, int width, int n) {
    if (n <= kConvSpan) {
        conv_kernel<<<div_up(width, kThreads), kThreads, 0, cs(st_)>>>(x, state, w, dynamic, width, n);
        return;
    }
    const dim3 grid(div_up(width, kThreads), div_up(n, kConvSpan));
    conv_halo_kernel<<<grid, kThreads, 0, cs(st_)>>>(x, state, halo_, width);
    conv_span_kernel<<<grid, kThreads, 0, cs(st_)>>>(x, state, halo_, w, dynamic, width, n);
}

void Inkling2::store(const float* x, Cache& c, int side, int n) {
    auto& values = c.planes[2 * side];
    auto& scales = c.planes[2 * side + 1];
    const bool swa = local(visit_);
    int64_t pos = swa ? pos_ % ring_ : pos_;
    for (int at = 0; at < n;) {
        const int m = swa ? std::min(n - at, ring_ - (int) pos) : n;
        if (q8()) store_q8(x, (int8_t*) values.data, (uint16_t*) scales.data, kKey, pos, m, st_);
        else store_f16(x, (uint16_t*) values.data, kKey, pos, m, st_);
        at += m;
        x += (size_t) m * kKey;
        pos = 0;
    }
}

void Inkling2::attention(const Block& b, int n) {
    Cache& c = caches_[visit_];
    store(k_, c, 0, n);
    store(v_, c, 1, n);
    const bool swa = local(visit_);
    inkling::Attend a;
    a.q = q_;
    a.rel = rel_;
    a.table = fp(b.table);
    a.sink = sink_;
    a.gate = gate_;
    a.k = {c.planes[0].data, (const uint16_t*) c.planes[1].data};
    a.v = {c.planes[2].data, (const uint16_t*) c.planes[3].data};
    a.pos = pos_;
    a.n = n;
    a.window = swa ? kWindow : 0;
    a.ring = swa ? ring_ : 0;
    a.out = att_;
    // Prompt batches only: a token's answer and short calls keep the plain kernels and their sums.
    if (flash_ && n >= kStageRows) inkling::attend_flash(a, cs(st_));
    else inkling::attend_plain(a, scores_, partial_, attend_tiles_, cs(st_));
}

void Inkling2::depth_mix(const float* query, float* dst, int first, int n) {
    const size_t at = (size_t) first * kWidth;
    MixInputs m{};
    for (int i = 0; i < depth_n_; ++i) m.x[i] = depth_[i] + at;
    m.x[depth_n_] = r_ + at;
    mix_kernel<<<n, kThreads, 0, cs(st_)>>>(m, depth_n_ + 1, query, dst, kWidth);
}

void Inkling2::control(const Block& b, int il, int stream, float* in, int n) {
    const size_t row = 2 * (size_t) (il - kStart) + (stream >= 2 ? 1 : 0);
    const float* query = fp(stream == 0 || stream == 3 ? loop_query_ : loop_kv_query_) + row * kWidth;
    mix_kernel<<<n, kThreads, 0, cs(st_)>>>(MixInputs{{bank0_, bank1_}}, 2, query, msg_, kWidth);
    rms_norm(in, nullptr, pair_, kWidth, n, 1.e-6f, st_, 2 * kWidth);
    rms_norm(msg_, nullptr, pair_ + kWidth, kWidth, n, 1.e-6f, st_, 2 * kWidth);
    const Control& c = b.control[stream];
    lin(c.down, pair_, rank_, n);
    lin(c.gain, rank_, gain_, n);
    lin(c.forget, rank_, forget_, n);
    control_kernel<<<div_up(n * kWidth, kThreads), kThreads, 0, cs(st_)>>>(in, msg_, gain_, forget_, n * kWidth);
}

void Inkling2::run_visits(int n) {
    rms_norm(r_, fp(token_norm_), cur_, kWidth, n, rms_eps_, st_);
    axpy(cur_, 1.f, r_, n * kWidth, false, st_);
    depth_n_ = 0;
    for (visit_ = 0; visit_ < kVisits; ++visit_) {
        if (visit_ == kStart) axpy(r_, 1.f, prelude_, n * kWidth, false, st_);
        if (visit_ == kStart + kCore || visit_ == kStart + 2 * kCore) {
            // The old partial and the one just before it are the previous pass's two core blocks.
            axpy(depth_[2], 1.f, bank0_, n * kWidth, false, st_);
            axpy(r_, 1.f, bank1_, n * kWidth, false, st_);
            axpy(prelude_, 1.f, r_, n * kWidth, false, st_);
            depth_n_ = 1;
        }
        layer(physical(visit_), n);
    }
}

void Inkling2::layer_head(Layer& L, int il, int n) {
    const Block& b = blocks_[il];
    const bool swa = local(visit_), loop = messages();
    const Dev* norm[3] = {&L.attn_norm, swa ? &L.attn_norm : &b.knorm, swa ? &L.attn_norm : &b.vnorm};
    const int streams = swa && !loop ? 1 : 3;
    for (int i = 0; i < streams; ++i) {
        if (il == 0) axpy(r_, 1.f, cur_, n * kWidth, false, st_);
        else depth_mix(fp(b.aquery) + (swa ? 0 : i * kWidth), cur_, 0, n);
        if (loop) control(b, il, i, cur_, n);
        rms_norm(cur_, fp(*norm[i]), input_[i], kWidth, n, rms_eps_, st_);
    }
    const float* xq = input_[0];
    const float* xk = streams == 1 ? xq : input_[1];
    const float* xv = streams == 1 ? xq : input_[2];
    // MMVQ uses the same Q8_1 input for Q, relative bias and the attention gate (also K and V when
    // they share the input). The matrix-matrix prompt path uses its own layout instead.
    const bool mmv = n < kStageRows;
    if (mmv) kernels::native_quantize_q8_1(xq, xq_, kWidth * n, 1, st_);
    const void* shared = mmv ? xq_ : nullptr;
    lin(b.q, xq, q_, n, shared);
    lin(b.k, xk, k_, n, streams == 1 ? shared : nullptr);
    lin(b.v, xv, v_, n, streams == 1 ? shared : nullptr);
    lin(b.rel, xq, rel_, n, shared);
    lin(b.sink, xq, sink_, n);
    lin(b.gate, xq, gate_, n, shared);

    float* hist = state_ + (size_t) visit_ * kConvFloats;
    const auto kv_conv = [&](float* x, float* state, const Dev& w, const Dev& dyn, const float* input) {
        if (!swa) lin(dyn, input, taps_, n);
        conv(x, state, fp(w), swa ? nullptr : taps_, kKey, n);
    };
    kv_conv(k_, hist, b.kconv, b.kproj, xk);
    kv_conv(v_, hist + 3 * kKey, b.vconv, b.vproj, xv);
    const int floor = swa ? 0 : floor_;
    const auto heads = [&](float* x, const float* w, int width, int count, float clip, int log_floor) {
        head_norm_kernel<<<dim3(count, n), kThreads, 0, cs(st_)>>>(x, w, width, count, clip, rms_eps_, pos_, log_floor,
                                                                  alpha_);
    };
    heads(q_, fp(b.qclip), kHead, kHeads, qk_clip_, floor);
    heads(k_, nullptr, kHead, kKv, qk_clip_, 0);
    heads(v_, nullptr, kHead, kKv, 0.f, 0);
    heads(rel_, fp(b.rclip), kRelative, kHeads, r_clip_, floor);
    dbg("Qcur", visit_, q_, kQuery);
    dbg("Kcur", visit_, k_, kKey);
    attention(b, n);
    lin(b.o, att_, ao_, n);
    conv(ao_, hist + 6 * kKey, fp(b.aconv), nullptr, kWidth, n);
    if (il % kBlock == 0) {
        axpy(r_, 1.f, depth_[depth_n_++], n * kWidth, false, st_);
        axpy(ao_, 1.f, r_, n * kWidth, false, st_);
    } else axpy(ao_, 1.f, r_, n * kWidth, true, st_);

    depth_mix(fp(b.fquery), cur_, 0, n);
    if (loop) control(b, il, 3, cur_, n);
    kernels::native_add_rms_norm_q8_1(cur_, nullptr, fp(L.ffn_norm), xn_, xq_, kWidth, n, rms_eps_, 128, st_);
    if (!batched(L, n)) moe_router(L, n);
}

void Inkling2::layer_tail(Layer&, int il, int n) {
    float* hist = state_ + (size_t) visit_ * kConvFloats + 6 * kKey + 3 * kWidth;
    conv(fo_, hist, fp(blocks_[il].fconv), nullptr, kWidth, n);
    axpy(fo_, 1.f, r_, n * kWidth, true, st_);
    dbg("l_out", visit_, r_, kWidth);
}

void Inkling2::route(int il, const float* logits, int32_t, int* ids, float* w, float* tmp) const {
    float* pick = tmp + n_expert_;
    for (int i = 0; i < n_expert_; ++i) {
        tmp[i] = 1.f / (1.f + std::exp(-logits[i]));
        pick[i] = tmp[i] + blocks_[il].bias[i];
    }
    float sum = 0.f;
    for (int i = 0; i < n_used_; ++i) {
        const int id = (int) (std::max_element(pick, pick + n_expert_) - pick);
        pick[id] = -INFINITY;
        ids[i] = id;
        sum += (w[i] = tmp[id]);
    }
    for (int i = 0; i < n_used_; ++i) w[i] *= expert_scale_ / (sum + 1.e-20f);
}

void Inkling2::moe_shared(const Layer&, int, int n) { zero_vram(fo_, (size_t) n * kWidth * sizeof(float)); }

void Inkling2::output_rows(int first, int rows, float* h) {
    depth_mix(fp(output_query_), cur_, first, rows);
    rms_norm(cur_, fp(out_norm_), h, kWidth, rows, rms_eps_, st_);
}

void Inkling2::finish_logits(float* logits, int rows) { softcap(logits, (size_t) rows * n_vocab_, cap_, st_); }

}  // namespace strata::poly
