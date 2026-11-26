// CPU verifier experiment: same ggml AVX2 dot, with one Q4 decode per block for 2-4 inputs.
#pragma once

#include <algorithm>
#include <cstddef>
#include <immintrin.h>
#include <cstdint>
#include <type_traits>
#include <utility>

#pragma GCC push_options
#pragma GCC target("avx2,f16c,fma")

namespace reuse {

inline float sum8(__m256 x) {
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}

inline float half(const uint8_t* p) {
    uint16_t h;
    __builtin_memcpy(&h, p, sizeof(h));
    return _cvtsh_ss(h);
}

template <bool Affine, size_t... J>
void dot(int n, const uint8_t* w, const uint8_t* const* x, float* out, std::index_sequence<J...>) {
    __m256 accum[sizeof...(J)]{};
    __m128 corr[sizeof...(J)]{};
    const uint8_t* ys[] = {x[J]...};
    constexpr int header = Affine ? 4 : 2;
    for (int b = 0; b < n / 32; ++b, w += 16 + header) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + header));
        auto bytes = _mm256_inserti128_si256(_mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1);
        bytes = _mm256_and_si256(bytes, _mm256_set1_epi8(15));
        if constexpr (!Affine) bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(8));
        const __m256i abs = Affine ? bytes : _mm256_sign_epi8(bytes, bytes);
        const float d = half(w);
        const float m = Affine ? half(w + 2) : 0.f;
        auto step = [&](auto j) {
            const uint8_t* y = ys[j];
            auto q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + header));
            if constexpr (!Affine) q = _mm256_sign_epi8(q, bytes);
            const __m256i pairs = _mm256_maddubs_epi16(abs, q);
            const __m256 v = _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
            const __m256 scale = _mm256_set1_ps(d * half(y));
            accum[j] = _mm256_fmadd_ps(scale, v, accum[j]);
            if constexpr (Affine) corr[j] = _mm_fmadd_ss(_mm_set_ss(m), _mm_set_ss(half(y + 2)), corr[j]);
            ys[j] += 32 + header;
        };
        (step(std::integral_constant<size_t, J>{}), ...);
    }
    // The linked ggml Q4_1 code uses a separate scalar FMA chain for the affine term.
    ((out[J] = Affine ? sum8(accum[J]) + _mm_cvtss_f32(corr[J]) : sum8(accum[J])), ...);
}

template <bool Affine>
[[gnu::noinline]] void dot(int n, const uint8_t* w, const uint8_t* const* x, int k, float* out) {
    switch (k) {
        case 2: return dot<Affine>(n, w, x, out, std::make_index_sequence<2>{});
        case 3: return dot<Affine>(n, w, x, out, std::make_index_sequence<3>{});
        case 4: return dot<Affine>(n, w, x, out, std::make_index_sequence<4>{});
    }
    __builtin_unreachable();
}

}  // namespace reuse

#pragma GCC pop_options

// Never leave a singleton tail. The caller sends actual singleton groups to ggml.
namespace reuse {
inline int chunk(int left) { return left == 5 ? 3 : std::min(left, 4); }
}
