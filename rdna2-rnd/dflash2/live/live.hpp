#pragma once
#include <stddef.h>
#include <cstdint>

// For the one-shot, single-target research executable only. No persistent session ABI.
namespace df_live {
void open(const char* target, const void* head);
void close();
void pending(int64_t pos, int n, const float* base, size_t plane);
void commit(int n);
void propose(int32_t anchor, int k, int32_t* out);
}
