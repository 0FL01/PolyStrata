// The global KV geometry is independent of the SWA rings and of the expert arena.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::poly::inkling {

constexpr int kKvPage = 128, kKvRecent = 2048, kKvTop = 256, kKvMaxPages = 2048;
constexpr int kKvColumns = 8 * 128, kV4Columns = kKvColumns / 2, kKvScales = kKvColumns / 32;
constexpr size_t kKvIndexPageBytes = kKvColumns * sizeof(uint32_t);
constexpr int kV4DiskLayout = 3, kV4DiskFormat = 0x49325634;   // I2V4, never native Q8_0

// A call seals only complete pages from before its first input, then retains its entire input as native Q8.
// A checkpoint seals to its committed position instead; its exact Q8 tail captures everything above that seal.
constexpr int64_t v4_seal(int64_t pos) {
    return std::max<int64_t>(0, pos - kKvRecent) / kKvPage * kKvPage;
}
constexpr int v4_checkpoint_rows(int max_context) { return std::min(max_context, kKvRecent + kKvPage - 1); }
constexpr int v4_hot_rows(int max_context, int max_rows) {
    return (int) std::min<int64_t>(max_context, (int64_t) v4_checkpoint_rows(max_context) + max_rows);
}
constexpr int k256_old_pages(int span) { return (span - kKvRecent) / kKvPage; }
constexpr bool k256_worthwhile(int span) { return k256_old_pages(span) - 1 > kKvTop; }

#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
inline uint8_t v4_nibble(int x, int amax) {
    // Symmetric refit of the original signed Q8 codes. Integer round-to-nearest-even, including -128.
    const int d = amax ? amax : 1, z = (x < 0 ? -x : x) * 7;
    int q = z / d;
    const int twice = 2 * (z % d);
    q += twice > d || (twice == d && (q & 1));
    return (x < 0 ? -q : q) & 15;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
inline int k256_page(int chunk, int old, const int* picked) {
    if (!chunk) return 0;
    return chunk <= kKvTop ? picked[chunk - 1] : old + chunk - kKvTop - 1;
}

}  // namespace strata::poly::inkling
