// src/poly/mmq_nvfp4_lean.cu - STRATA_HIP_NVFP4_LEAN=1: mmq_nvfp4.cu's W4A8 tiles again, with every DP4A through
// Strata's wrapper below. Its own unit with its own names, so the default unit's code stays as it was.
// Every DP4A of this unit through Strata's wrapper: ggml's ggml_cuda_dp4a is renamed while common.cuh defines it,
// and a sum that starts at 0 gets one VOP3 v_dot4_i32_i8 with the inline constant 0 (the compiler otherwise moves 0
// into the sum's register for the tied v_dot4c form). The same integers, so the same products to the bit.
#define ggml_cuda_dp4a ggml_cuda_dp4a_ggml
#include "common.cuh"
#undef ggml_cuda_dp4a
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP) && defined(RDNA2)
    if (__builtin_constant_p(c) && c == 0) {
        int r;
        asm("v_dot4_i32_i8 %0, %1, %2, 0" : "=v"(r) : "v"(a), "v"(b));
        return r;
    }
#endif
    return ggml_cuda_dp4a_ggml(a, b, c);
}
#define STRATA_W4A8_NAME(x) strata_poly_w4a8_lean_##x
#define nvfp4_w4a8 nvfp4_w4a8_lean
#define nvfp4_w4a8_fits nvfp4_w4a8_lean_fits
#define nvfp4_w4a8_stream_k nvfp4_w4a8_lean_stream_k
#include "mmq_nvfp4.cu"
