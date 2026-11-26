// libggml_qbridge.so — C/AVX2 quantization, dequantization, and fast Q6_K matvec bridge
// Linked against the vendored ggml's CPU-only libggml-base.a and libggml-cpu.a.
// Exposes the exact ABI expected by tools/convert_inkling2_dflash2_to_gguf.py plus
// fast AVX2+FMA+F16C matvec and sparse row-gather helpers for dflash2_ref.py.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <pthread.h>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"

extern "C" {
void quantize_row_q8_K(const float * __restrict__ x, void * __restrict__ y, int64_t k);
void ggml_vec_dot_q6_K_q8_K(int n, float * __restrict__ s, size_t bs,
                            const void * __restrict__ vx, size_t bx,
                            const void * __restrict__ vy, size_t by, int nrc);
}

#define QK_K 256

struct q6_k_block {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(q6_k_block) == 210, "block_q6_K must be 210 bytes");

struct q8_k_block {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(q8_k_block) == 292, "block_q8_K must be 292 bytes");

static inline float fp16_bits_to_f32(uint16_t h) {
    __m128i v = _mm_cvtsi32_si128((int)h);
    return _mm_cvtss_f32(_mm_cvtph_ps(v));
}

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// Fast AVX2+F16C dequantize of a single block_q6_K (256 floats) into y[0..255]
static inline void dequant_q6_k_block_avx2(const q6_k_block * __restrict__ blk, float * __restrict__ y) {
    const float d = fp16_bits_to_f32(blk->d);
    const uint8_t * ql = blk->ql;
    const uint8_t * qh = blk->qh;
    const int8_t  * sc = blk->scales;

    const __m128i m_lo4 = _mm_set1_epi8(0x0F);
    const __m128i m_hi2 = _mm_set1_epi8(0x03);
    const __m128i v_32  = _mm_set1_epi8(32);

    for (int half = 0; half < 2; ++half) {
        for (int is = 0; is < 2; ++is) {
            const int off = is * 16;
            const __m128i ql0 = _mm_loadu_si128((const __m128i *)(ql + off));
            const __m128i ql1 = _mm_loadu_si128((const __m128i *)(ql + off + 32));
            const __m128i qhv = _mm_loadu_si128((const __m128i *)(qh + off));

            const __m128i q1_i8 = _mm_sub_epi8(
                _mm_or_si128(_mm_and_si128(ql0, m_lo4),
                             _mm_slli_epi16(_mm_and_si128(qhv, m_hi2), 4)),
                v_32);
            const __m128i q2_i8 = _mm_sub_epi8(
                _mm_or_si128(_mm_and_si128(ql1, m_lo4),
                             _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 2), m_hi2), 4)),
                v_32);
            const __m128i q3_i8 = _mm_sub_epi8(
                _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), m_lo4),
                             _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 4), m_hi2), 4)),
                v_32);
            const __m128i q4_i8 = _mm_sub_epi8(
                _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), m_lo4),
                             _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 6), m_hi2), 4)),
                v_32);

            auto store16 = [](float * dst, __m128i q_i8, float scale) {
                const __m256 vs = _mm256_set1_ps(scale);
                const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_i8));
                const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_i8, 8)));
                _mm256_storeu_ps(dst + 0, _mm256_mul_ps(f0, vs));
                _mm256_storeu_ps(dst + 8, _mm256_mul_ps(f1, vs));
            };

            store16(y + off +  0, q1_i8, d * (float)sc[is + 0]);
            store16(y + off + 32, q2_i8, d * (float)sc[is + 2]);
            store16(y + off + 64, q3_i8, d * (float)sc[is + 4]);
            store16(y + off + 96, q4_i8, d * (float)sc[is + 6]);
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static inline void dot_rows_avx2(const float * __restrict__ w,
                                 const float * __restrict__ x,
                                 float * __restrict__ out,
                                 int64_t ncols, int64_t nrows, int64_t row_idx,
                                 int m_batch) {
    // Accumulate dot(w, x[m]) for m in 0..m_batch-1 in chunks of up to 8
    int m = 0;
    for (; m + 7 <= m_batch; m += 7) {
        const float * x0 = x + (int64_t)(m + 0) * ncols;
        const float * x1 = x + (int64_t)(m + 1) * ncols;
        const float * x2 = x + (int64_t)(m + 2) * ncols;
        const float * x3 = x + (int64_t)(m + 3) * ncols;
        const float * x4 = x + (int64_t)(m + 4) * ncols;
        const float * x5 = x + (int64_t)(m + 5) * ncols;
        const float * x6 = x + (int64_t)(m + 6) * ncols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps(), a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 8 <= ncols; j += 8) {
            __m256 wv = _mm256_loadu_ps(w + j);
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + j), a0);
            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + j), a1);
            a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2 + j), a2);
            a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3 + j), a3);
            a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x4 + j), a4);
            a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x5 + j), a5);
            a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x6 + j), a6);
        }
        float s0 = hsum256_ps(a0), s1 = hsum256_ps(a1), s2 = hsum256_ps(a2);
        float s3 = hsum256_ps(a3), s4 = hsum256_ps(a4), s5 = hsum256_ps(a5), s6 = hsum256_ps(a6);
        for (; j < ncols; ++j) {
            float wf = w[j];
            s0 += wf * x0[j]; s1 += wf * x1[j]; s2 += wf * x2[j];
            s3 += wf * x3[j]; s4 += wf * x4[j]; s5 += wf * x5[j]; s6 += wf * x6[j];
        }
        out[(int64_t)(m + 0) * nrows + row_idx] = s0;
        out[(int64_t)(m + 1) * nrows + row_idx] = s1;
        out[(int64_t)(m + 2) * nrows + row_idx] = s2;
        out[(int64_t)(m + 3) * nrows + row_idx] = s3;
        out[(int64_t)(m + 4) * nrows + row_idx] = s4;
        out[(int64_t)(m + 5) * nrows + row_idx] = s5;
        out[(int64_t)(m + 6) * nrows + row_idx] = s6;
    }
    for (; m < m_batch; ++m) {
        const float * xm = x + (int64_t)m * ncols;
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 16 <= ncols; j += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j),     _mm256_loadu_ps(xm + j),     a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j + 8), _mm256_loadu_ps(xm + j + 8), a1);
        }
        float s = hsum256_ps(_mm256_add_ps(a0, a1));
        for (; j < ncols; ++j) s += w[j] * xm[j];
        out[(int64_t)m * nrows + row_idx] = s;
    }
}

// Generic pthread parallel-for helper over [0, total)
template <typename Fn>
static void parallel_for_rows(int64_t total, int nthreads, const Fn & fn) {
    if (total <= 0) return;
    if (nthreads < 1) nthreads = 1;
    if ((int64_t)nthreads > total) nthreads = (int)total;
    if (nthreads == 1) {
        fn(0, total);
        return;
    }
    struct Task {
        int64_t r0, r1;
        const Fn * fn;
    };
    std::vector<pthread_t> tids(nthreads);
    std::vector<Task> tasks(nthreads);
    auto tramp = [](void * arg) -> void * {
        Task * t = (Task *)arg;
        (*(t->fn))(t->r0, t->r1);
        return nullptr;
    };
    const int64_t chunk = (total + nthreads - 1) / nthreads;
    int launched = 0;
    for (int t = 0; t < nthreads; ++t) {
        int64_t r0 = t * chunk;
        if (r0 >= total) break;
        int64_t r1 = r0 + chunk;
        if (r1 > total) r1 = total;
        tasks[t] = {r0, r1, &fn};
        if (t == 0) {
            launched = 1;
        } else {
            if (pthread_create(&tids[t], nullptr, tramp, &tasks[t]) != 0) {
                std::fprintf(stderr, "qbridge: unable to start worker\n");
                std::abort();
            }
            ++launched;
        }
    }
    // Run slice 0 on calling thread
    fn(tasks[0].r0, tasks[0].r1);
    for (int t = 1; t < launched; ++t) {
        pthread_join(tids[t], nullptr);
    }
}

extern "C" {

size_t qbridge_row_size(int qtype, int64_t ncols) {
    return ggml_row_size((enum ggml_type)qtype, ncols);
}

size_t qbridge_quantize(int qtype, const float * src, void * dst,
                        int64_t nrows, int64_t ncols, int nthreads) {
    const enum ggml_type t = (enum ggml_type)qtype;
    const size_t row_bytes = ggml_row_size(t, ncols);
    if (t == GGML_TYPE_F32) {
        std::memcpy(dst, src, (size_t)nrows * row_bytes);
        return (size_t)nrows * row_bytes;
    }
    // Initialize quantization lookup tables once before worker threads spawn
    ggml_quantize_init(t);

    const auto * traits = ggml_get_type_traits(t);

    parallel_for_rows(nrows, nthreads, [&](int64_t r0, int64_t r1) {
        if (t == GGML_TYPE_F16 || t == GGML_TYPE_BF16) {
            // Use traits->from_float_ref row by row
            for (int64_t r = r0; r < r1; ++r) {
                traits->from_float_ref(src + r * ncols,
                                       (uint8_t *)dst + (size_t)r * row_bytes,
                                       ncols);
            }
        } else {
            ggml_quantize_chunk(t, src, dst, r0 * ncols, r1 - r0, ncols, nullptr);
        }
    });
    return (size_t)nrows * row_bytes;
}

void qbridge_dequantize(int qtype, const void * src, float * dst,
                        int64_t nrows, int64_t ncols, int nthreads) {
    const enum ggml_type t = (enum ggml_type)qtype;
    const size_t row_bytes = ggml_row_size(t, ncols);
    if (t == GGML_TYPE_F32) {
        std::memcpy(dst, src, (size_t)nrows * row_bytes);
        return;
    }
    const auto * traits = ggml_get_type_traits(t);
    if (!traits || !traits->to_float) {
        std::fprintf(stderr, "qbridge_dequantize: unsupported qtype=%d\n", qtype);
        std::abort();
    }
    parallel_for_rows(nrows, nthreads, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            traits->to_float((const uint8_t *)src + (size_t)r * row_bytes,
                             dst + r * ncols, ncols);
        }
    });
}

void qbridge_dequantize_rows(int qtype, const void * src, const int32_t * row_ids,
                             int64_t n_ids, float * dst, int64_t ncols) {
    const enum ggml_type t = (enum ggml_type)qtype;
    const size_t row_bytes = ggml_row_size(t, ncols);
    if (t == GGML_TYPE_F32) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int64_t r = (int64_t)row_ids[i];
            std::memcpy(dst + i * ncols, (const uint8_t *)src + (size_t)r * row_bytes, row_bytes);
        }
        return;
    }
    const auto * traits = ggml_get_type_traits(t);
    if (!traits || !traits->to_float) {
        std::fprintf(stderr, "qbridge_dequantize_rows: unsupported qtype=%d\n", qtype);
        std::abort();
    }
    for (int64_t i = 0; i < n_ids; ++i) {
        const int64_t r = (int64_t)row_ids[i];
        traits->to_float((const uint8_t *)src + (size_t)r * row_bytes,
                         dst + i * ncols, ncols);
    }
}

void qbridge_eval_error(const float * ref, const float * test, int64_t n, double * out_stats) {
    double ref_sq = 0.0, test_sq = 0.0, diff_sq = 0.0, dot = 0.0;
    double max_abs = 0.0, sum_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double a = (double)ref[i];
        const double b = (double)test[i];
        const double d = std::fabs(a - b);
        ref_sq  += a * a;
        test_sq += b * b;
        diff_sq += d * d;
        dot     += a * b;
        sum_abs += d;
        if (d > max_abs) max_abs = d;
    }
    const double rel_rmse = (ref_sq > 0.0) ? std::sqrt(diff_sq / ref_sq) : 0.0;
    const double sqnr_db  = (diff_sq > 0.0 && ref_sq > 0.0) ? 10.0 * std::log10(ref_sq / diff_sq) : 999.0;
    const double mean_abs = (n > 0) ? (sum_abs / (double)n) : 0.0;
    const double cos_sim  = (ref_sq > 0.0 && test_sq > 0.0) ? (dot / (std::sqrt(ref_sq) * std::sqrt(test_sq))) : 1.0;
    out_stats[0] = rel_rmse;
    out_stats[1] = sqnr_db;
    out_stats[2] = max_abs;
    out_stats[3] = mean_abs;
    out_stats[4] = cos_sim;
}

// Fast multi-threaded AVX2+FMA+F16C exact-FP32-activation Q6_K matvec:
// out_f32[m * nrows + r] = dot(dequant(w_q6k[r]), x_f32[m])
void qbridge_q6k_matvec_f32(const void * w_q6k, const float * x_f32, float * out_f32,
                            int64_t nrows, int64_t ncols, int m_batch, int nthreads) {
    const int64_t nb = ncols / QK_K;
    const size_t row_bytes = (size_t)nb * sizeof(q6_k_block);
    const uint8_t * w_base = (const uint8_t *)w_q6k;

    parallel_for_rows(nrows, nthreads, [&](int64_t r0, int64_t r1) {
        alignas(32) float wbuf[256];
        if (m_batch == 7) {
            const float * x0 = x_f32 + 0 * ncols;
            const float * x1 = x_f32 + 1 * ncols;
            const float * x2 = x_f32 + 2 * ncols;
            const float * x3 = x_f32 + 3 * ncols;
            const float * x4 = x_f32 + 4 * ncols;
            const float * x5 = x_f32 + 5 * ncols;
            const float * x6 = x_f32 + 6 * ncols;
            for (int64_t r = r0; r < r1; ++r) {
                const q6_k_block * row = (const q6_k_block *)(w_base + (size_t)r * row_bytes);
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps(), a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
                __m256 a6 = _mm256_setzero_ps();
                for (int64_t b = 0; b < nb; ++b) {
                    dequant_q6_k_block_avx2(row + b, wbuf);
                    const int64_t base = b * 256;
                    for (int k = 0; k < 256; k += 8) {
                        __m256 wv = _mm256_load_ps(wbuf + k);
                        a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + base + k), a0);
                        a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + base + k), a1);
                        a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2 + base + k), a2);
                        a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3 + base + k), a3);
                        a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x4 + base + k), a4);
                        a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x5 + base + k), a5);
                        a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x6 + base + k), a6);
                    }
                }
                out_f32[0 * nrows + r] = hsum256_ps(a0);
                out_f32[1 * nrows + r] = hsum256_ps(a1);
                out_f32[2 * nrows + r] = hsum256_ps(a2);
                out_f32[3 * nrows + r] = hsum256_ps(a3);
                out_f32[4 * nrows + r] = hsum256_ps(a4);
                out_f32[5 * nrows + r] = hsum256_ps(a5);
                out_f32[6 * nrows + r] = hsum256_ps(a6);
            }
        } else {
            std::vector<float> row_f32((size_t)ncols);
            for (int64_t r = r0; r < r1; ++r) {
                const q6_k_block * row = (const q6_k_block *)(w_base + (size_t)r * row_bytes);
                for (int64_t b = 0; b < nb; ++b) {
                    dequant_q6_k_block_avx2(row + b, row_f32.data() + b * 256);
                }
                dot_rows_avx2(row_f32.data(), x_f32, out_f32, ncols, nrows, r, m_batch);
            }
        }
    });
}

// Fast multi-threaded ggml-cpu Q6_K x Q8_K matvec (matches llama.cpp CPU quantized dot)
void qbridge_q6k_matvec_q8k(const void * w_q6k, const float * x_f32, float * out_f32,
                            int64_t nrows, int64_t ncols, int m_batch, int nthreads) {
    static bool cpu_inited = false;
    if (!cpu_inited) {
        ggml_cpu_init();
        cpu_inited = true;
    }
    const int64_t nb = ncols / QK_K;
    const size_t w_row_bytes = (size_t)nb * sizeof(q6_k_block);
    const size_t x_row_bytes = (size_t)nb * sizeof(q8_k_block);
    std::vector<uint8_t> x_q8k((size_t)m_batch * x_row_bytes);
    for (int m = 0; m < m_batch; ++m) {
        quantize_row_q8_K(x_f32 + (int64_t)m * ncols,
                          x_q8k.data() + (size_t)m * x_row_bytes, ncols);
    }
    const uint8_t * w_base = (const uint8_t *)w_q6k;
    parallel_for_rows(nrows, nthreads, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const void * wr = w_base + (size_t)r * w_row_bytes;
            for (int m = 0; m < m_batch; ++m) {
                float s = 0.0f;
                ggml_vec_dot_q6_K_q8_K((int)ncols, &s, 0, wr, 0,
                                       x_q8k.data() + (size_t)m * x_row_bytes, 0, 1);
                out_f32[(int64_t)m * nrows + r] = s;
            }
        }
    });
}

// Generic exact-FP32-activation matvec for any ggml_type
void qbridge_matvec_f32(int qtype, const void * w_raw, const float * x_f32, float * out_f32,
                        int64_t nrows, int64_t ncols, int m_batch, int nthreads) {
    const enum ggml_type t = (enum ggml_type)qtype;
    if (t == GGML_TYPE_Q6_K) {
        qbridge_q6k_matvec_f32(w_raw, x_f32, out_f32, nrows, ncols, m_batch, nthreads);
        return;
    }
    const size_t row_bytes = ggml_row_size(t, ncols);
    const auto * traits = ggml_get_type_traits(t);
    const uint8_t * w_base = (const uint8_t *)w_raw;

    parallel_for_rows(nrows, nthreads, [&](int64_t r0, int64_t r1) {
        std::vector<float> row_f32((size_t)ncols);
        for (int64_t r = r0; r < r1; ++r) {
            const float * wf = nullptr;
            if (t == GGML_TYPE_F32) {
                wf = (const float *)(w_base + (size_t)r * row_bytes);
            } else {
                traits->to_float(w_base + (size_t)r * row_bytes, row_f32.data(), ncols);
                wf = row_f32.data();
            }
            dot_rows_avx2(wf, x_f32, out_f32, ncols, nrows, r, m_batch);
        }
    });
}

// Fast F32 -> QTYPE -> F32 roundtrip (for Q8_0 KV simulation or per-tensor ablation)
void qbridge_roundtrip(int qtype, const float * src, float * dst,
                       int64_t nrows, int64_t ncols, int nthreads) {
    const enum ggml_type t = (enum ggml_type)qtype;
    if (t == GGML_TYPE_F32) {
        if (src != dst) std::memcpy(dst, src, (size_t)nrows * (size_t)ncols * sizeof(float));
        return;
    }
    const size_t row_bytes = ggml_row_size(t, ncols);
    std::vector<uint8_t> tmp((size_t)nrows * row_bytes);
    qbridge_quantize(qtype, src, tmp.data(), nrows, ncols, nthreads);
    qbridge_dequantize(qtype, tmp.data(), dst, nrows, ncols, nthreads);
}

} // extern "C"
