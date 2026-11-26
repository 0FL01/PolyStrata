// KVQ V3: full-history W1 decode attention with Q8 K and packed rotated V, gfx1031 wave32.
// Same geometry as PolyStrata decode_q8_kernel: one block per chunk, 8 waves = 8 KV groups, lane = (sub: 16 dims,
// quad: one of 4 keys), 2 key steps unrolled. V formats (rows in the WHT128-rotated basis, known norm, no scale):
//   Q8  : native int8 + fp16 scale per 32 (production reference, native basis)
//   U4/5/6 : uniform codes, value = (code - (2^b-1)/2) * step (offset and step folded into the combine)
//   L5/6   : Lloyd-Max codes looked up in an LDS float table
//   P4     : Lloyd-Max 4-bit int8 table via v_perm_b32 (Octave-style), value = int8 / 46
// Packing per (row, KV head): nibble plane 64 B (lane sub: 8 B, byte j = dim j | dim j+4 << 4 within each 8 dims),
// high planes 16 B (5th bit) and 16 B (6th bit), lane sub: 2 B each, bit j = dim j of its 16.
// Build: S=.venv/lib/python3.13/site-packages/_rocm_sdk_devel; $S/llvm/bin/clang++ -x hip --offload-arch=gfx1031 -O3
//   -std=c++17 --rocm-path=$S --rocm-device-lib-path=$S/lib/llvm/amdgcn/bitcode -I$S/include -L$S/lib -lamdhip64
//   -o vread vread.hip.  Run under rdna2-rnd/scripts/run_guarded.py: vread <samples> <span>...
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

constexpr int kHeads = 24, kHead = 128, kKv = 8, kGroup = 3, kRelative = 16, kExtent = 1024;
constexpr int kChunkFloats = kGroup * (kHead + 2);
constexpr int kUnroll = 2;

enum Fmt { Q8 = 0, U4, U5, U6, L5, L6, P4, NFMT };
const char* kName[NFMT] = {"Q8", "U4", "U5", "U6", "L5lds", "L6lds", "P4perm"};
__host__ __device__ constexpr int bits_of(int f) { return f == U4 || f == P4 ? 4 : f == U5 || f == L5 ? 5 : f == Q8 ? 8 : 6; }

struct VPlanes {
    const int8_t* q8;          // [rows][8][128]
    const __half* q8s;         // [rows][8][4]
    const uint8_t* nib;        // [rows][8][64]
    const uint16_t* h5;        // [rows][8][8]  (per lane sub: one u16)
    const uint16_t* h6;        // [rows][8][8]
};

__constant__ float c_lut[64];
__constant__ uint32_t c_p4[4];   // int8 LM4 table words: entries 0-3, 4-7, 8-11, 12-15

__device__ __forceinline__ uint32_t spread4(uint32_t bits) {   // 4 bits -> bit 0 of 4 bytes
    return ((bits & 0xF) * 0x00204081u) & 0x01010101u;
}

template <int F>
__device__ __forceinline__ void unpack_v(const VPlanes& v, size_t rowhead, int sub, const int4& q8w, float q8s, float* val,
                                         const float* lds) {
    if constexpr (F == Q8) {
        const int w[4] = {q8w.x, q8w.y, q8w.z, q8w.w};
#pragma unroll
        for (int x = 0; x < 4; ++x)
#pragma unroll
            for (int y = 0; y < 4; ++y) val[4 * x + y] = (float) (int8_t) (w[x] >> (8 * y)) * q8s;
    } else {
        const uint2 nw = *reinterpret_cast<const uint2*>(v.nib + rowhead * 64 + 8 * sub);
        const uint32_t nwords[2] = {nw.x, nw.y};
        uint32_t hb5 = 0, hb6 = 0;
        if constexpr (bits_of(F) >= 5) hb5 = v.h5[rowhead * 8 + sub];
        if constexpr (bits_of(F) >= 6) hb6 = v.h6[rowhead * 8 + sub];
#pragma unroll
        for (int h = 0; h < 2; ++h) {          // 8 dims per word: bytes = dims 0..3 (lo) and 4..7 (hi)
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                uint32_t c = (nwords[h] >> (4 * part)) & 0x0F0F0F0Fu;
                const int dim0 = 8 * h + 4 * part;   // within the lane's 16
                if constexpr (bits_of(F) >= 5) c |= spread4(hb5 >> dim0) << 4;
                if constexpr (bits_of(F) >= 6) c |= spread4(hb6 >> dim0) << 5;
                if constexpr (F == U4 || F == U5 || F == U6) {
                    val[dim0 + 0] = (float) ((c >> 0) & 0xFFu);
                    val[dim0 + 1] = (float) ((c >> 8) & 0xFFu);
                    val[dim0 + 2] = (float) ((c >> 16) & 0xFFu);
                    val[dim0 + 3] = (float) (c >> 24);
                } else if constexpr (F == L5 || F == L6) {
                    val[dim0 + 0] = lds[c & 0xFF];
                    val[dim0 + 1] = lds[(c >> 8) & 0xFF];
                    val[dim0 + 2] = lds[(c >> 16) & 0xFF];
                    val[dim0 + 3] = lds[c >> 24];
                } else {   // P4: two v_perm_b32 over 8-entry banks, blend by bit 3
                    const uint32_t sel = c & 0x07070707u;
                    const uint32_t lo = __builtin_amdgcn_perm(c_p4[1], c_p4[0], sel);
                    const uint32_t hi = __builtin_amdgcn_perm(c_p4[3], c_p4[2], sel);
                    const uint32_t m = ((c >> 3) & 0x01010101u) * 0xFFu;
                    const uint32_t b = (hi & m) | (lo & ~m);
                    val[dim0 + 0] = (float) (int8_t) (b);
                    val[dim0 + 1] = (float) (int8_t) (b >> 8);
                    val[dim0 + 2] = (float) (int8_t) (b >> 16);
                    val[dim0 + 3] = (float) (int8_t) (b >> 24);
                }
            }
        }
    }
}

template <int F>
__global__ void __launch_bounds__(256, 2)
decode_kernel(const float* q, const float* rel, const float* table, const int8_t* kd, const __half* ksc, VPlanes v,
              int64_t pos, int chunk, float* part) {
    const int c = blockIdx.x, row = blockIdx.y;
    const int g = threadIdx.x / 32, lane = threadIdx.x & 31, sub = lane & 7, quad = lane >> 3;
    const int64_t end = pos + row + 1;
    const int count = (int) end, first = c * chunk, last = min(first + chunk, count);
    __shared__ float lds[64];
    if constexpr (F == L5 || F == L6) {
        if (threadIdx.x < 64) lds[threadIdx.x] = c_lut[threadIdx.x];
        __syncthreads();
    }
    const size_t qh0 = (size_t) row * kHeads + g * kGroup;
    float qr[kGroup][16], rr[kGroup][2];
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        const float4* qv = reinterpret_cast<const float4*>(q + (qh0 + r) * kHead + 16 * sub);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 x = qv[j];
            qr[r][4 * j] = x.x; qr[r][4 * j + 1] = x.y; qr[r][4 * j + 2] = x.z; qr[r][4 * j + 3] = x.w;
        }
        rr[r][0] = rel[(qh0 + r) * kRelative + 2 * sub];
        rr[r][1] = rel[(qh0 + r) * kRelative + 2 * sub + 1];
    }
    float m[kGroup], l[kGroup], acc[kGroup][16];
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) acc[r][j] = 0.f;
    }
    for (int i0 = first; i0 < last; i0 += 4 * kUnroll) {
        int4 kw[kUnroll], vw[kUnroll];
        float ks[kUnroll], vs[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int i = i0 + 4 * u + quad;
            if (i < last) {
                const size_t at = ((size_t) i * kKv + g) * kHead + 16 * sub;
                kw[u] = *reinterpret_cast<const int4*>(kd + at);
                ks[u] = __half2float(ksc[at / 32]);
                if constexpr (F == Q8) {
                    vw[u] = *reinterpret_cast<const int4*>(v.q8 + at);
                    vs[u] = __half2float(v.q8s[at / 32]);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int i = i0 + 4 * u + quad;
            if (i >= last) continue;
            const int64_t delta = end - 1 - i;
            const int kwords[4] = {kw[u].x, kw[u].y, kw[u].z, kw[u].w};
            float sc[kGroup];
#pragma unroll
            for (int r = 0; r < kGroup; ++r) {
                float d = 0.f;
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y < 4; ++y) d += qr[r][4 * x + y] * (float) (int8_t) (kwords[x] >> (8 * y));
                sc[r] = d * ks[u] * 0.08838834764831845f;
            }
            if (delta < kExtent) {
                const float* tb = table + delta * kRelative + 2 * sub;
                const float b0 = tb[0], b1 = tb[1];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) sc[r] += rr[r][0] * b0 + rr[r][1] * b1;
            }
#pragma unroll
            for (int r = 0; r < kGroup; ++r)
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) sc[r] += __shfl_xor(sc[r], o, 32);
            float val[16];
            unpack_v<F>(v, (size_t) i * kKv + g, sub, vw[u], vs[u], val, lds);
#pragma unroll
            for (int r = 0; r < kGroup; ++r) {
                const float next = fmaxf(m[r], sc[r]);
                const float base = next == -INFINITY ? 0.f : next;
                const float keep = expf(m[r] - base), p = expf(sc[r] - base);
                m[r] = next;
                l[r] = l[r] * keep + p;
#pragma unroll
                for (int j = 0; j < 16; ++j) acc[r][j] = acc[r][j] * keep + p * val[j];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        float top = fmaxf(m[r], __shfl_xor(m[r], 8, 32));
        top = fmaxf(top, __shfl_xor(top, 16, 32));
        const float base = top == -INFINITY ? 0.f : top;
        const float f = expf(m[r] - base);
        float sum = l[r] * f;
        sum += __shfl_xor(sum, 8, 32);
        sum += __shfl_xor(sum, 16, 32);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float a = acc[r][j] * f;
            a += __shfl_xor(a, 8, 32);
            a += __shfl_xor(a, 16, 32);
            acc[r][j] = a;
        }
        m[r] = top;
        l[r] = sum;
    }
    if (quad != 0) return;
    const int chunks = gridDim.x;
    float* dst = part + (((size_t) row * kKv + g) * chunks + c) * kChunkFloats;
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[r * (kHead + 2) + 2 + 16 * sub + j] = acc[r][j];
        if (sub == 0) {
            dst[r * (kHead + 2)] = m[r];
            dst[r * (kHead + 2) + 1] = l[r];
        }
    }
}

// rotated: o_rot = (A - offset * L) * step / sum, then x = s1 * H128(s2 * o_rot) (orthonormal), then gate.
__global__ void combine_kernel(const float* part, int chunks, const float* sink, const float* gate, float* out,
                               int rotated, float offset, float step, const float* s1, const float* s2) {
    const int h = blockIdx.x, row = blockIdx.y, d = threadIdx.x, g = h / kGroup, r = h % kGroup;
    const size_t qh = (size_t) row * kHeads + h;
    const float* base = part + ((size_t) row * kKv + g) * chunks * kChunkFloats + r * (kHead + 2);
    const float s = sink[qh];
    float top = s;
#pragma unroll 8
    for (int c = 0; c < chunks; ++c) top = fmaxf(top, base[(size_t) c * kChunkFloats]);
    const float at = top == -INFINITY ? 0.f : top;
    float sum = expf(s - at), o = 0.f, L = 0.f;
#pragma unroll 8
    for (int c = 0; c < chunks; ++c) {
        const float* p = base + (size_t) c * kChunkFloats;
        const float w = expf(p[0] - at);
        sum += p[1] * w;
        L += p[1] * w;
        o += p[2 + d] * w;
    }
    float y = o / sum;
    if (rotated) {
        __shared__ float sh[kHead];
        sh[d] = (o - offset * L) * step / sum * s2[d];
        __syncthreads();
        for (int len = 1; len < kHead; len <<= 1) {
            const int i = d ^ len;
            const float a = sh[d], b = sh[i];
            __syncthreads();
            sh[d] = (d & len) ? b - a : a + b;
            __syncthreads();
        }
        y = sh[d] * 0.08838834764831845f * s1[d];
    }
    out[qh * kHead + d] = y / (1.f + expf(-gate[qh * kHead + d]));
}

__global__ void flush_kernel(const uint4* src, uint32_t* dst, size_t n) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) dst[0] = acc;
}

// ------------------------------------------------------------------------------------------ host side
static float lm_gauss[7][64];
static void lloyd(int b, float* out) {   // Lloyd-Max N(0,1) by iteration on a dense grid
    const int n = 1 << b;
    std::vector<double> c(n);
    for (int i = 0; i < n; ++i) c[i] = -3.5 + 7.0 * (i + 0.5) / n;
    const int G = 200001;
    for (int it = 0; it < 400; ++it) {
        std::vector<double> s(n, 0), w(n, 0);
        for (int k = 0; k < G; ++k) {
            const double x = -8.0 + 16.0 * k / (G - 1), p = std::exp(-0.5 * x * x);
            int j = (int) (std::upper_bound(c.begin(), c.end(), x) - c.begin());
            if (j > 0 && (j == n || x - c[j - 1] < c[j] - x)) --j;
            s[j] += p * x;
            w[j] += p;
        }
        for (int i = 0; i < n; ++i) if (w[i] > 0) c[i] = s[i] / w[i];
    }
    for (int i = 0; i < n; ++i) out[i] = (float) c[i];
}
static const float kStep[7] = {0, 0, 0, 0, 0.3286f, 0.1834f, 0.1008f};
static const int8_t kP4[16] = {-126, -95, -74, -58, -43, -30, -18, -6, 6, 18, 30, 43, 58, 74, 95, 126};

static uint32_t sgn_words1[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
static uint32_t sgn_words2[4] = {0x16ACEE90u, 0x3F628FDCu, 0xB7A5357Au, 0xBEA56562u};
static float S1[128], S2[128];

static void wht(float* y) {
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = y[j], b = y[j + h];
                y[j] = a + b;
                y[j + h] = a - b;
            }
    for (int i = 0; i < 128; ++i) y[i] *= 0.08838834764831845f;
}

struct Host {
    int rows;
    std::vector<int8_t> kq, vq;
    std::vector<__half> ks, vs;
    std::vector<uint8_t> nib[NFMT];
    std::vector<uint16_t> h5[NFMT], h6[NFMT];
    std::vector<float> vdec[NFMT];     // decoded rotated values (in value units, offset/step applied) for the oracle
    std::vector<float> vnative;        // dequantized Q8 values (native)
    std::vector<float> q, rel, table, sink, gate;
};

static void q8_row(const float* x, int8_t* c, __half* s, float* deq) {
    for (int b = 0; b < 4; ++b) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32 * b + j]));
        const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
        s[b] = __float2half(d);
        const float dh = __half2float(s[b]);
        for (int j = 0; j < 32; ++j) {
            const float qv = std::min(127.f, std::max(-127.f, std::rint(x[32 * b + j] * id)));
            c[32 * b + j] = (int8_t) qv;
            if (deq) deq[32 * b + j] = qv * dh;
        }
    }
}

static int nearest(const float* book, int n, float x) {
    int best = 0;
    float e = 1e30f;
    for (int i = 0; i < n; ++i) {
        const float d = std::fabs(book[i] - x);
        if (d < e) e = d, best = i;
    }
    return best;
}

static void build(Host& H, int rows, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.f, 1.f);
    H.rows = rows;
    const size_t vals = (size_t) rows * kKv * kHead;
    H.kq.resize(vals); H.vq.resize(vals); H.ks.resize(vals / 32); H.vs.resize(vals / 32); H.vnative.resize(vals);
    for (int f = 1; f < NFMT; ++f) {
        H.nib[f].assign((size_t) rows * kKv * 64, 0);
        H.h5[f].assign((size_t) rows * kKv * 8, 0);
        H.h6[f].assign((size_t) rows * kKv * 8, 0);
        H.vdec[f].resize(vals);
    }
    std::vector<float> x(128), y(128);
    for (size_t rh = 0; rh < (size_t) rows * kKv; ++rh) {
        // K: random direction at the clip norm 22.627 with a per-head mean; V: RMS-normalized (norm sqrt(128))
        double nn = 0;
        for (int i = 0; i < 128; ++i) { x[i] = N(rng) + 0.5f * ((rh % kKv) == (size_t) (i % 8)); nn += x[i] * x[i]; }
        for (int i = 0; i < 128; ++i) x[i] = (float) (x[i] / std::sqrt(nn) * 22.627417);
        q8_row(x.data(), &H.kq[rh * 128], &H.ks[rh * 4], nullptr);
        nn = 0;
        for (int i = 0; i < 128; ++i) { x[i] = N(rng) * (i < 8 ? 3.f : 1.f); nn += x[i] * x[i]; }
        for (int i = 0; i < 128; ++i) x[i] = (float) (x[i] / std::sqrt(nn / 128.0));
        q8_row(x.data(), &H.vq[rh * 128], &H.vs[rh * 4], &H.vnative[rh * 128]);
        for (int i = 0; i < 128; ++i) y[i] = x[i] * S1[i];
        wht(y.data());
        for (int i = 0; i < 128; ++i) y[i] *= S2[i];
        for (int f = 1; f < NFMT; ++f) {
            const int b = bits_of(f), n = 1 << b;
            for (int dim = 0; dim < 128; ++dim) {
                int code;
                float dec;
                if (f == U4 || f == U5 || f == U6) {
                    const float off = (n - 1) * 0.5f;
                    code = (int) std::floor(y[dim] / kStep[b] + n * 0.5f);
                    code = std::min(n - 1, std::max(0, code));
                    dec = (code - off) * kStep[b];
                } else if (f == P4) {
                    float book[16];
                    for (int i = 0; i < 16; ++i) book[i] = kP4[i] / 46.f;
                    code = nearest(book, 16, y[dim]);
                    dec = book[code];
                } else {
                    code = nearest(lm_gauss[b], n, y[dim]);
                    dec = lm_gauss[b][code];
                }
                H.vdec[f][rh * 128 + dim] = dec;
                const int sub = dim / 16, k = dim % 16, h = k / 8, j = k % 8;
                // byte (j % 4) of word h: low nibble dim j (j<4), high nibble dim j (j>=4)
                uint8_t& byte = H.nib[f][rh * 64 + 8 * sub + 4 * h + (j % 4)];
                byte |= (uint8_t) ((code & 15) << (j < 4 ? 0 : 4));
                // high bits: the unpack takes dims 4*part + 8*h .. +3 from bit dim0 = 8h + 4part
                const int bit = 8 * h + (j < 4 ? j : 4 + (j - 4));
                if (b >= 5) H.h5[f][rh * 8 + sub] |= (uint16_t) (((code >> 4) & 1) << bit);
                if (b >= 6) H.h6[f][rh * 8 + sub] |= (uint16_t) (((code >> 5) & 1) << bit);
            }
        }
    }
    H.table.resize(1024 * 16);
    for (auto& t : H.table) t = 0.05f * N(rng);
}

static void oracle(const Host& H, int f, int64_t pos, std::vector<double>& out) {   // one row, all heads
    out.assign(kHeads * kHead, 0);
    const int n = (int) pos + 1;
    std::vector<double> sc(n);
    for (int h = 0; h < kHeads; ++h) {
        const int g = h / kGroup;
        const float* q = &H.q[h * kHead];
        double mx = H.sink[h];
        for (int i = 0; i < n; ++i) {
            double d = 0;
            const size_t rh = (size_t) i * kKv + g;
            for (int j = 0; j < 128; ++j) d += (double) q[j] * H.kq[rh * 128 + j] * __half2float(H.ks[rh * 4 + j / 32]);
            d *= 0.08838834764831845;
            const int64_t delta = pos - i;
            if (delta < kExtent) for (int j = 0; j < 16; ++j) d += (double) H.rel[h * 16 + j] * H.table[delta * 16 + j];
            sc[i] = d;
            mx = std::max(mx, d);
        }
        double sum = std::exp(H.sink[h] - mx);
        std::vector<double> o(128, 0);
        for (int i = 0; i < n; ++i) {
            const double p = std::exp(sc[i] - mx);
            sum += p;
            const size_t rh = (size_t) i * kKv + g;
            for (int j = 0; j < 128; ++j) o[j] += p * (f == Q8 ? H.vnative[rh * 128 + j] : H.vdec[f][rh * 128 + j]);
        }
        std::vector<float> y(128);
        for (int j = 0; j < 128; ++j) y[j] = (float) (o[j] / sum);
        if (f != Q8) {
            for (int j = 0; j < 128; ++j) y[j] *= S2[j];
            wht(y.data());
            for (int j = 0; j < 128; ++j) y[j] *= S1[j];
        }
        for (int j = 0; j < 128; ++j) out[h * kHead + j] = y[j] / (1.0 + std::exp(-(double) H.gate[h * kHead + j]));
    }
}

template <class T> T* up(const std::vector<T>& v) {
    T* p = nullptr;
    CK(hipMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T)));
    CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

int main(int argc, char** argv) {
    const int samples = argc > 1 ? std::atoi(argv[1]) : 15;
    std::vector<int> spans;
    for (int i = 2; i < argc; ++i) spans.push_back(std::atoi(argv[i]));
    if (spans.empty()) spans = {4096};
    const int chunks_pref = std::getenv("CHUNKS") ? std::atoi(std::getenv("CHUNKS")) : 160;
    for (int i = 0; i < 128; ++i) {
        S1[i] = (sgn_words1[i >> 5] >> (i & 31)) & 1 ? -1.f : 1.f;
        S2[i] = (sgn_words2[i >> 5] >> (i & 31)) & 1 ? -1.f : 1.f;
    }
    for (int b = 4; b <= 6; ++b) lloyd(b, lm_gauss[b]);
    const int rows = *std::max_element(spans.begin(), spans.end());
    Host H;
    build(H, rows, 7);
    std::mt19937 rng(11);
    std::normal_distribution<float> N(0.f, 1.f);
    H.q.resize(kHeads * kHead); H.rel.resize(kHeads * 16); H.sink.resize(kHeads); H.gate.resize(kHeads * kHead);
    for (auto& v : H.q) v = N(rng) * 2.0f;
    for (auto& v : H.rel) v = N(rng);
    for (auto& v : H.sink) v = N(rng) * 2.f;
    for (auto& v : H.gate) v = N(rng);
    int8_t* kq = up(H.kq);
    __half* ks = up(H.ks);
    VPlanes vp[NFMT];
    vp[Q8].q8 = up(H.vq);
    vp[Q8].q8s = up(H.vs);
    for (int f = 1; f < NFMT; ++f) {
        vp[f] = vp[Q8];
        vp[f].nib = up(H.nib[f]);
        vp[f].h5 = up(H.h5[f]);
        vp[f].h6 = up(H.h6[f]);
    }
    float* dq = up(H.q); float* drel = up(H.rel); float* dtab = up(H.table); float* dsink = up(H.sink);
    float* dgate = up(H.gate);
    std::vector<float> s1v(S1, S1 + 128), s2v(S2, S2 + 128);
    float* ds1 = up(s1v); float* ds2 = up(s2v);
    float* part = nullptr; float* out = nullptr;
    CK(hipMalloc(&part, (size_t) 4096 * kChunkFloats * sizeof(float)));
    CK(hipMalloc(&out, kHeads * kHead * sizeof(float)));
    const size_t flush_n = (size_t(256) << 20) / sizeof(uint4);
    uint4* flush = nullptr; uint32_t* trash = nullptr;
    CK(hipMalloc(&flush, flush_n * sizeof(uint4)));
    CK(hipMemset(flush, 0x5a, flush_n * sizeof(uint4)));
    CK(hipMalloc(&trash, 64));
    hipStream_t st;
    CK(hipStreamCreate(&st));
    hipEvent_t e0, e1, e2;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1)); CK(hipEventCreate(&e2));
    std::vector<float> lut(64, 0.f);
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s samples=%d chunks<=%d\n", prop.name, prop.gcnArchName, samples, chunks_pref);
    uint32_t p4w[4];
    std::memcpy(p4w, kP4, 16);
    CK(hipMemcpyToSymbol(HIP_SYMBOL(c_p4), p4w, sizeof p4w));
    auto launch = [&](int f, int64_t pos, int chunk, int chunks) {
        const dim3 grid(chunks, 1);
        switch (f) {
            case Q8: decode_kernel<Q8><<<grid, 256, 0, st>>>(dq, drel, dtab, kq, ks, vp[f], pos, chunk, part); break;
            case U4: decode_kernel<U4><<<grid, 256, 0, st>>>(dq, drel, dtab, kq, ks, vp[f], pos, chunk, part); break;
            case U5: decode_kernel<U5><<<grid, 256, 0, st>>>(dq, drel, dtab, kq, ks, vp[f], pos, chunk, part); break;
            case U6: decode_kernel<U6><<<grid, 256, 0, st>>>(dq, drel, dtab, kq, ks, vp[f], pos, chunk, part); break;
            case L5: decode_kernel<L5><<<grid, 256, 0, st>>>(dq, drel, dtab, kq, ks, vp[f], pos, chunk, part); break;
            case L6: decode_kernel<L6><<<grid, 256, 0, st>>>(dq, drel, dtab, kq, ks, vp[f], pos, chunk, part); break;
            case P4: decode_kernel<P4><<<grid, 256, 0, st>>>(dq, drel, dtab, kq, ks, vp[f], pos, chunk, part); break;
        }
        const int b = bits_of(f);
        const bool uni = f == U4 || f == U5 || f == U6;
        const float off = uni ? ((1 << b) - 1) * 0.5f : 0.f, step = uni ? kStep[b] : f == P4 ? 1.f / 46.f : 1.f;
        CK(hipEventRecord(e1, st));
        combine_kernel<<<dim3(kHeads, 1), kHead, 0, st>>>(part, chunks, dsink, dgate, out, f != Q8, off, step, ds1, ds2);
    };
    for (int span : spans) {
        const int64_t pos = span - 1;
        const int step = 4 * kUnroll;
        const int chunk = std::max(32, ((span + chunks_pref - 1) / chunks_pref + step - 1) / step * step);
        const int chunks = (span + chunk - 1) / chunk;
        std::vector<double> ref;
        std::vector<float> got(kHeads * kHead);
        for (int f = 0; f < NFMT; ++f) {
            if (f == L5 || f == L6) {
                for (int i = 0; i < 64; ++i) lut[i] = i < (1 << bits_of(f)) ? lm_gauss[bits_of(f)][i] : 0.f;
                CK(hipMemcpyToSymbol(HIP_SYMBOL(c_lut), lut.data(), 64 * sizeof(float)));
            }
            launch(f, pos, chunk, chunks);
            CK(hipStreamSynchronize(st));
            CK(hipMemcpy(got.data(), out, got.size() * sizeof(float), hipMemcpyDeviceToHost));
            double err = 0, nrm = 0;
            if (span <= 8192) {
                oracle(H, f, pos, ref);
                for (size_t i = 0; i < got.size(); ++i) { err += (got[i] - ref[i]) * (got[i] - ref[i]); nrm += ref[i] * ref[i]; }
            }
            std::vector<float> leaf, full;
            for (int s = 0; s < samples + 2; ++s) {
                flush_kernel<<<256, 256, 0, st>>>(flush, trash, flush_n);
                CK(hipEventRecord(e0, st));
                launch(f, pos, chunk, chunks);
                CK(hipEventRecord(e2, st));
                CK(hipEventSynchronize(e2));
                float a, b;
                CK(hipEventElapsedTime(&a, e0, e1));
                CK(hipEventElapsedTime(&b, e0, e2));
                if (s >= 2) { leaf.push_back(a); full.push_back(b); }
            }
            std::sort(leaf.begin(), leaf.end());
            std::sort(full.begin(), full.end());
            const double vbytes = f == Q8 ? 136 : bits_of(f) * 16;
            const double mb = (double) span * kKv * (136 + vbytes) / 1e6;
            std::printf("span=%d chunks=%d %-7s Vbytes=%3.0f leaf_ms=%.4f full_ms=%.4f GBps=%.0f oracle_rel=%s\n", span,
                        chunks, kName[f], vbytes, leaf.empty() ? 0 : leaf[leaf.size() / 2],
                        full.empty() ? 0 : full[full.size() / 2], leaf.empty() ? 0 : mb / leaf[leaf.size() / 2],
                        span <= 8192 ? std::to_string(std::sqrt(err / nrm)).c_str() : "-");
            std::fflush(stdout);
        }
    }
    return 0;
}
