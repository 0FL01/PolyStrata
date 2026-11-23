// STRATA_INKLING_FLASH_LEAN=1 (with STRATA_INKLING_LANE_PROB=1 on gfx1031): attend_flash() through a kernel of the
// same arithmetic and fewer instructions - fused DPP8 reductions, and a full layer's far key blocks without the
// relative table. Bitwise the same output as the lane-owned flash kernel.
#pragma once

#include "inkling_attend.hpp"

namespace strata::poly::inkling {

void attend_flash_lean(const Attend& a, cudaStream_t st);

}  // namespace strata::poly::inkling
