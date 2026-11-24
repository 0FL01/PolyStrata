#pragma once

#include <cstdint>

#if defined(CP_HIP)
#include <hip/hip_runtime.h>
#endif

namespace cp04 {
struct Q80 { uint16_t d; int8_t q[32]; };
static_assert(sizeof(Q80) == 34);

#if defined(CP_HIP)
void quantize(const float* x, Q80* q, int nb, hipStream_t st);
void project(const uint8_t* w, const Q80* x, float* y, int nb, int rows, hipStream_t st);
#endif
}
