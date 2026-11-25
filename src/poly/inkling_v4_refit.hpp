#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly::codec {

// Convert Inkling V rows from Q8_0 to symmetric V4, with one refitted FP16 scale per 32 values. Each
// row holds 1024 values: 1024 input code bytes, 512 output code bytes, and 64 bytes in either scale plane.
// The even value in a pair goes in the low nibble. Input and output planes must not overlap.
void pack_refit(const uint8_t* code, const uint8_t* scales, size_t rows, uint8_t* packed, uint8_t* refitted);

// Same bytes, without CPU dispatch; useful as the reference and for an explicit scalar control.
void pack_refit_scalar(const uint8_t* code, const uint8_t* scales, size_t rows, uint8_t* packed, uint8_t* refitted);

}  // namespace strata::poly::codec
