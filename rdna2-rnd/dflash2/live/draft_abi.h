#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Research ABI, one sequence on one gfx1031 device. The Q6_K head belongs to the
// target and lives until close. A null head requests a private replay copy.
void* df2_open(const char* gguf, const char* target_gguf, const void* shared_head_device);
void df2_close(void* model);
// Contiguous COMMITTED rows, starting at 0 (or a full >=2048-row warmup at
// a nonzero absolute position when replaying a suffix). All four arrays are device FP32
// [n,2304]. The producer has synchronized; consume them before returning.
int df2_append(void* model, int64_t pos, int n, const float* tap5, const float* tap23,
               const float* tap41, const float* norm48);
// Anchor is at pos, not fed; out_ids is host memory for k (1..7) guesses.
int df2_propose(void* model, int64_t pos, int32_t anchor, int k, int32_t* out_ids);

#ifdef __cplusplus
}
#endif
