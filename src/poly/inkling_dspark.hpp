// The DSpark drafter of Inkling-2-Mini (a `dflash` GGUF with dspark_hidden_out): three dense layers that guess the
// seven tokens after the one the model picked, from the model's rows at four of its visits, and a recurrent head
// that turns each guess into the next one's input.  These are its launches that the engine has no kernel for; the
// matrix products are the engine's (lin()).  The graph is llama.cpp's graph_inkling2 (PR 32218, src/models/dspark.cpp).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace strata::poly::dspark {

constexpr int kWidth = 2304, kHeads = 24, kKv = 8, kHead = 128, kKey = kKv * kHead, kRank = 256;
constexpr int kTaps = 4, kBlock = 8;

/// out[r] = rmsnorm(sum_j w[j] * tap_j[r]) * norm for rows first .. first + n - 1 of the taps (kWidth floats a row);
/// out row r - first.
struct Taps {
    const float* tap[kTaps];
    float w[kTaps];
};
void mix_taps(const Taps& t, const float* norm, float* out, int first, int n, float eps, cudaStream_t st);

/// One side of a draft cache: ggml Q8_0 rows of kKey codes and kKey / 32 half scales.
struct Q8Rows {
    int8_t* codes = nullptr;
    uint16_t* scales = nullptr;
};

/// The non-causal attention of a block of kBlock queries at positions pos .. pos + 7: each query sees the cached rows
/// of positions lo .. pos - 1 (row p at p % ring, ring 0: row p) and the block's own keys and values.  q: [8][24][128]
/// (normed, rotated), kb and vb: [8][8][128]; out: [8][24][128].  scratch: attend_floats() floats.
size_t attend_floats();
void attend(const float* q, const float* kb, const float* vb, Q8Rows k, Q8Rows v, int64_t lo, int64_t pos, int ring,
            float* scratch, float* out, cudaStream_t st, bool reuse_v);

/// x = [state, w1, hidden, context] (256 + 256 + 2304 + 2304 floats).
void joint(const float* state, const float* w1, const float* hidden, const float* context, float* x,
           cudaStream_t st);

/// From the joint projection j (768, its bias not yet added): state = c + sigmoid(g) (state - c) with c = tanh, and the
/// latent tanh of the third part.
void recur(const float* j, const float* bias, float* state, float* latent, cudaStream_t st);

/// logits[v] = cap tanh((base[v] + bias2[v] + sum_i latent[i] w2[i][v]) / cap), w2 half [kRank][n].
void head(const float* base, const float* bias2, const uint16_t* w2, const float* latent, float* logits, int n,
          float cap, cudaStream_t st);

/// dst[r] = src for rows 0 .. rows - 1 of width floats.
void fill_rows(const float* src, float* dst, int width, int rows, cudaStream_t st);

/// dst[r * ld + i] = half(src[r * n + i]) for rows rows of n.
void to_half(const float* src, uint16_t* dst, int n, int rows, int64_t ld, cudaStream_t st);

}  // namespace strata::poly::dspark
