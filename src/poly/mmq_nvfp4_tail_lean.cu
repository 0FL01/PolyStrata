// src/poly/mmq_nvfp4_tail_lean.cu - STRATA_HIP_NVFP4_LEAN=1 for the down tail: mmq_nvfp4_tail.cu again, with every
// DP4A through Strata's wrapper below (see mmq_nvfp4_lean.cu).
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
#define nvfp4_tail_try nvfp4_tail_lean_try
#include "mmq_nvfp4_tail.cu"
