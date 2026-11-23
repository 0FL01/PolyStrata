#pragma once

// All finite normal UE4M3 values are exact in FP32. Construct their exponent and fraction after ggml's
// factor of 1/2, and keep the same subnormals and the original treatment of byte 0x7f (not 0xff).
static __device__ __forceinline__ float strata_nvfp4_scale(uint8_t byte) {
    const unsigned exp = (byte >> 3) & 15, man = byte & 7;
    const float normal = __uint_as_float(((exp + 119) << 23) | (man << 20));
    return byte == 0x7f ? 0.f : exp ? normal : (float) man * 0x1p-10f;
}
#define ggml_cuda_ue4m3_to_fp32 strata_nvfp4_scale
