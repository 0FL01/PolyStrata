// src/poly/mmq_nvfp4.cu - llama.cpp's NVFP4 tiles with 8-bit activations (W4A8), on CUDA and RDNA2.
//
// On sm_120 mmq.cuh multiplies NVFP4 as FP4 x FP4 (W4A4): the activations are rounded to E2M1 too.  Every other
// card gets its int8 path for NVFP4: weights decoded to int8, activations Q8_1, the precision decode runs at.  This
// unit compiles that path for sm_120 as well by hiding Blackwell's FP4 MMA from mmq.cuh on both sides, the device
// code and the host's tile config, which must agree on the tiles' shared memory (Strata's
// src/prefill/mmq_nvfp4_w4a8.cu).  The two non-static templates are renamed so the linker cannot fold them with
// the W4A4 instance built for 12xa (CMakeLists.txt).
// For HIP RDNA2, ggml's RDNA2 config already uses wave32 DP4A and the same Q8_1 layout.
#include "common.cuh"

#if !defined(GGML_USE_HIP)
#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#endif
#ifndef STRATA_W4A8_NAME   // mmq_nvfp4_lean.cu builds this unit again under other names
#define STRATA_W4A8_NAME(x) strata_poly_w4a8_##x
#endif
#define mul_mat_q_switch_J STRATA_W4A8_NAME(mul_mat_q_switch_J)
#define mul_mat_q_case STRATA_W4A8_NAME(mul_mat_q_case)
#include "mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_NVFP4);

namespace strata::poly {

void nvfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s);
}

// Mmq::fits() and Mmq::split_exact() for these tiles: the host's config as this unit sees the card
bool nvfp4_w4a8_fits(int64_t w_rows, int cc, size_t smpbo) {
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, w_rows % 128 != 0, cc);
        if (c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo) return true;
    }
    return false;
}

bool nvfp4_w4a8_stream_k(int64_t w_rows, int cc) {
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, w_rows % 128 != 0, cc);
        if (c.type != GGML_TYPE_COUNT && c.stream_k) return true;
    }
    return false;
}

}  // namespace strata::poly
