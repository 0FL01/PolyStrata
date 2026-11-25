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

/// Mixed full attention: keys remain absolute Q8. V rows below sealed are packed and absolute; the others
/// are native Q8 in Attend::v, addressed by absolute row modulo hot_ring. Attend::ring is still zero.
struct V4View {
    KvView cold;
    int64_t sealed = 0;
    int hot_ring = 0;
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

/// The scratch attend_decode() needs, whatever the rows and context.
size_t decode_floats();

/// The same for a few tokens (STRATA_INKLING_DECODE=1, fewer rows than a prompt batch): each token's keys in chunks
/// across the card, a chunk's maximum, sum and unnormalized values per head, then a launch that adds the chunks and the
/// sink. FP32 products of the same dequantized values; the order of the sums differs. With q8, the output's Q8_1
/// blocks too, as native_quantize_q8_1() makes them of the rows back to back. On gfx1031, group_rows changes only
/// the launch order of long full-context Q8_0 windows of two to four rows, not their chunks or arithmetic.
void attend_decode(const Attend& a, float* scratch, cudaStream_t st, void* q8 = nullptr, bool group_rows = false);

/// Dense causal mixed reader, including every prompt row. The caller may substitute a tiled flash reader.
void attend_v4(const Attend& a, V4View v4, float* scratch, cudaStream_t st, void* q8 = nullptr);
/// Approximate opt-in for wide global V4-refit calls only. False does not launch or touch the output.
bool try_attend_compact_suffix_h2(const Attend& a, V4View v4, float* scratch, cudaStream_t st,
                                  void* q8 = nullptr);
/// Build a contiguous range of P128 centroids from committed/stored native Q8 keys; `stop` is exclusive.
void k256_build(KvView k, uint32_t* index, int first, int stop, cudaStream_t st);
size_t k256_scratch_floats();
/// A single decode query: page 0, 256 ranked old pages per group, and the entire recent tail. False when dense.
bool attend_k256(const Attend& a, V4View v4, const uint32_t* index, float* scratch, cudaStream_t st, void* q8);
/// Refit rows in a native Q8 hot ring directly into disjoint absolute V4 backing. Only committed rows are sealed.
void seal_v4(KvView hot, int ring, uint8_t* packed, uint16_t* scales, int64_t first, int64_t stop, cudaStream_t st);

/// The same in one launch (STRATA_INKLING_FLASH=1, prompt batches): 16 tokens of one K/V group a block, keys 16 at
/// a time with an online softmax in base 2, FP32 products of the same dequantized values; the order of the sums and
/// the exponential's rounding differ.
void attend_flash(const Attend& a, cudaStream_t st);

}  // namespace strata::poly::inkling
