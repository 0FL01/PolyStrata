// Inkling-2-Mini's attention over its K/V caches: 24 query heads of 128 in groups of 3 per K/V head, the scaled
// product plus a learned relative bias over the last 1024 positions, a projected sink in the softmax's
// denominator, and the output gated by sigmoid(gate). A window layer reads a ring of rows.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace strata::poly::inkling {

/// One side of a cache: Q8_0 codes with a half scale per 32 (scales set) or halves (scales null).
struct KvView {
    const uint8_t* data = nullptr;
    const uint16_t* scales = nullptr;
};

/// Tokens pos .. pos + n - 1, row-major as the engine keeps them: q, gate and out [n][24][128], rel [n][24][16],
/// sink [n][24]; table [1024][16]. window 0 is a full layer (ring 0); else each token attends its last `window`
/// positions, kept at row key % ring.
struct Attend {
    const float* q = nullptr;
    const float* rel = nullptr;
    const float* table = nullptr;
    const float* sink = nullptr;
    const float* gate = nullptr;
    KvView k, v;
    int64_t pos = 0;
    int n = 0;
    int window = 0;
    int ring = 0;
    float* out = nullptr;
};

/// The scratch attend_plain() needs: `tiles` value tiles of 128 scores per head.
size_t plain_score_floats(int tiles);
size_t plain_partial_floats(int tiles);

/// The scores, softmax, values and gate as four launches, with as many tokens a launch as `tiles` allows.
void attend_plain(const Attend& a, float* scores, float* partial, int tiles, cudaStream_t st);

/// The same in one launch (STRATA_INKLING_FLASH=1, prompt batches): 32 tokens of one K/V group a block, keys 32 at
/// a time with an online softmax, FP32 products of the same dequantized values; only the order of the sums differs.
void attend_flash(const Attend& a, cudaStream_t st);

}  // namespace strata::poly::inkling
