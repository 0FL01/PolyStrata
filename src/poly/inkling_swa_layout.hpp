#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::poly::inkling {

// The base/Lane, Lean and H2 Flash readers all partition absolute keys into these tiles.
inline constexpr int kSwaKeyTile = 16;
inline constexpr size_t kSwaCopyAlign = 16;
inline constexpr size_t kSwaScratchAlign = 256;   // Engine::fbuf() rounds each slice to 64 floats

struct SwaPlan {
    int ring_rows = 0;
    int stage_rows = 0;
    bool borrow_input = false;
    size_t scratch_bytes = 0;
    size_t saved_bytes = 0;
};

constexpr SwaPlan plan_swa(int context, int window, int rows, int verify, int visits, size_t row_bytes,
                           size_t input_bytes, bool enabled) {
    const int stock = std::min(context, window - 1 + rows);
    if (!enabled) return {stock};
    // A verifier appends all its rows before keep() chooses a prefix. Leave room for the
    // entire provisional append without overwriting the old tail needed by its first query.
    const int ring = std::min(context, window - 1 + verify);
    if (ring >= stock) return {stock};

    const int stage_rows = stock + kSwaKeyTile - 1;
    const size_t stage_bytes = (size_t) stage_rows * row_bytes;
    const bool borrow = stage_bytes + kSwaCopyAlign - 1 <= input_bytes;
    const size_t scratch = borrow ? 0 : (stage_bytes + kSwaScratchAlign - 1) / kSwaScratchAlign * kSwaScratchAlign;
    const size_t ring_savings = (size_t) visits * (stock - ring) * row_bytes;
    if (ring_savings <= scratch) return {stock};
    return {ring, stage_rows, borrow, scratch, ring_savings - scratch};
}

struct SwaStageLayout {
    int64_t first = 0;        // absolute oldest row the first query can read
    int history = 0, head = 0; // number of old rows and their first row in the stage planes
    constexpr int append() const { return head + history; }
    constexpr bool fits_ring(int n, int ring) const { return history + n <= ring; }
};

constexpr SwaStageLayout swa_stage_layout(int64_t pos, int window) {
    const int history = (int) std::min<int64_t>(pos, window - 1);
    const int64_t first = pos - history;
    // Rebase by a whole Flash tile, not by `first`, to preserve the order of the sums.
    return {first, history, (int) (first % kSwaKeyTile)};
}

struct SwaRingSpan { int row, count; };

constexpr SwaRingSpan swa_ring_span(int64_t first, int64_t end, int ring) {
    const int row = (int) (first % ring);
    return {row, (int) std::min<int64_t>(end - first, ring - row)};
}

}  // namespace strata::poly::inkling
