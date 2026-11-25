// STRATA_INKLING_FLASH_LEAN=1 (with STRATA_INKLING_LANE_PROB=1 on gfx1031): attend_flash() through a kernel of the
// same arithmetic and fewer instructions - fused DPP8 reductions, and a full layer's far key blocks without the
// relative table. Bitwise the same output as the lane-owned flash kernel.
#pragma once

#include "inkling_attend.hpp"

namespace strata::poly::inkling {

void attend_flash_lean(const Attend& a, cudaStream_t st);

/// STRATA_INKLING_FLASH_H2=1 (with STRATA_INKLING_LANE_PROB=1 on gfx1031, Q8_0 cache): the lean kernel with its
/// products in packed halves (v_dot2_f32_f16, FP32 sums) - a precision change, opt-in.
void attend_flash_h2(const Attend& a, cudaStream_t st);

// A single 1..16-row tile of split-K H2 attention over disjoint global K/V (gfx1031).
// The caller gives 16-key-aligned splits and combines their scratch; this does not write a.out.
void launch_compact_suffix_h2(const Attend& tile, float* scratch, cudaStream_t st, V4View v4,
                              int chunks, int chunk);

}  // namespace strata::poly::inkling
