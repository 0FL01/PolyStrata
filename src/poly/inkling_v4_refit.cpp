#include "inkling_v4_refit.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#define STRATA_POLY_V4_REFIT_AVX2 1
#include <immintrin.h>
#endif

namespace strata::poly::codec {
namespace {

constexpr size_t kBlock = 32, kBlocksPerRow = 1024 / kBlock, kPackedBlock = kBlock / 2;
constexpr size_t kScaleBytes = 2;
static_assert(sizeof(_Float16) == kScaleBytes);

// Round to the nearest integer, with ties even. Index by the original byte to avoid a branch per value.
using QuantLut = std::array<std::array<uint8_t, 256>, 129>;
const QuantLut& refit_lut() {
    static const QuantLut lut = [] {
        QuantLut t{};
        for (int a = 0; a <= 128; ++a) {
            const int div = std::max(a, 1);
            for (int x = -128; x <= 127; ++x) {
                const int z = std::abs(x) * 7, r2 = 2 * (z % div);
                int q = z / div;
                q += r2 > div || (r2 == div && (q & 1));
                t[a][static_cast<uint8_t>(x)] = std::clamp(x < 0 ? -q : q, -8, 7) & 15;
            }
        }
        return t;
    }();
    return lut;
}

inline void refit_scale(const uint8_t* src, int amax, uint8_t* dst) {
    _Float16 scale;
    std::memcpy(&scale, src, kScaleBytes);
    scale = static_cast<_Float16>(static_cast<float>(scale) * (static_cast<float>(amax) / 7.f));
    std::memcpy(dst, &scale, kScaleBytes);
}

int pack_scalar_block(const uint8_t* p, uint8_t* dst) {
    int amax = 0;
    for (size_t j = 0; j < kBlock; ++j) amax = std::max(amax, std::abs(static_cast<int8_t>(p[j])));
    const auto& lut = refit_lut()[amax];
    for (size_t j = 0; j < kBlock; j += 2) dst[j / 2] = lut[p[j]] | (lut[p[j + 1]] << 4);
    return amax;
}

#if defined(STRATA_POLY_V4_REFIT_AVX2)
// Full-range Q8 blocks have amax == 127. For those, |x| * 7 / 127 has no half-integer ties; the seven
// strict thresholds below give exactly the scalar nibble. V(P)ABSB leaves -128 negative, so the sign
// check also excludes blocks containing -128, even if they contain a separate +/-127 value.
__attribute__((target("avx2,f16c")))
void pack_refit_avx2(const uint8_t* code, const uint8_t* scales, size_t rows, uint8_t* packed, uint8_t* refitted) {
    for (size_t b = 0; b < rows * kBlocksPerRow; ++b) {
        const uint8_t* src = code + b * kBlock;
        uint8_t* dst = packed + b * kPackedBlock;
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        const __m256i a = _mm256_abs_epi8(x);
        if (_mm256_movemask_epi8(a) || !_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, _mm256_set1_epi8(127)))) {
            const int amax = pack_scalar_block(src, dst);
            refit_scale(scales + b * kScaleBytes, amax, refitted + b * kScaleBytes);
            continue;
        }
        const __m256i c0 = _mm256_cmpgt_epi8(a, _mm256_set1_epi8(9));
        const __m256i c1 = _mm256_cmpgt_epi8(a, _mm256_set1_epi8(27));
        const __m256i c2 = _mm256_cmpgt_epi8(a, _mm256_set1_epi8(45));
        const __m256i c3 = _mm256_cmpgt_epi8(a, _mm256_set1_epi8(63));
        const __m256i c4 = _mm256_cmpgt_epi8(a, _mm256_set1_epi8(81));
        const __m256i c5 = _mm256_cmpgt_epi8(a, _mm256_set1_epi8(99));
        const __m256i c6 = _mm256_cmpgt_epi8(a, _mm256_set1_epi8(117));
        const __m256i lo = _mm256_add_epi8(_mm256_add_epi8(c0, c1), _mm256_add_epi8(c2, c3));
        const __m256i hi = _mm256_add_epi8(_mm256_add_epi8(c4, c5), c6);
        const __m256i mag = _mm256_sub_epi8(_mm256_setzero_si256(), _mm256_add_epi8(lo, hi));
        const __m256i nibble = _mm256_and_si256(_mm256_sign_epi8(mag, x), _mm256_set1_epi8(15));
        // Each byte pair [even, odd] becomes one u16 with the low byte even + 16*odd (at most 255).
        const __m256i pairs = _mm256_maddubs_epi16(nibble, _mm256_set1_epi16(0x1001));
        const __m128i bytes = _mm_packus_epi16(_mm256_castsi256_si128(pairs), _mm256_extracti128_si256(pairs, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), bytes);
        refit_scale(scales + b * kScaleBytes, 127, refitted + b * kScaleBytes);
    }
}
#endif

}  // namespace

void pack_refit_scalar(const uint8_t* code, const uint8_t* scales, size_t rows, uint8_t* packed, uint8_t* refitted) {
    for (size_t b = 0; b < rows * kBlocksPerRow; ++b) {
        const int amax = pack_scalar_block(code + b * kBlock, packed + b * kPackedBlock);
        refit_scale(scales + b * kScaleBytes, amax, refitted + b * kScaleBytes);
    }
}

void pack_refit(const uint8_t* code, const uint8_t* scales, size_t rows, uint8_t* packed, uint8_t* refitted) {
#if defined(STRATA_POLY_V4_REFIT_AVX2)
    static const bool fast = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
    if (fast) return pack_refit_avx2(code, scales, rows, packed, refitted);
#endif
    pack_refit_scalar(code, scales, rows, packed, refitted);
}

}  // namespace strata::poly::codec
