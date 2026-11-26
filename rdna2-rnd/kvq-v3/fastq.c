// KVQ V3: fast CPU reference for rotated scalar codecs over 128-wide rows (OpenMP). Mirrors kvcodec.scalar().
#include <math.h>
#include <stdint.h>
#include <string.h>

static void wht(float* y, int width) {
    for (int b = 0; b < 128; b += width) {
        for (int h = 1; h < width; h *= 2)
            for (int i = b; i < b + width; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    float a = y[j], c = y[j + h];
                    y[j] = a + c;
                    y[j + h] = a - c;
                }
        const float s = 1.0f / sqrtf((float) width);
        for (int i = b; i < b + width; ++i) y[i] *= s;
    }
}

static inline float to_f16(float x) { return (float) (_Float16) x; }

static inline int nearest(float v, const float* edges, int ne) {   // edges sorted, ne = nbook-1
    int lo = 0, hi = ne;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (v > edges[mid]) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// rot: 0 none, 1 h32, 2 h64, 3 h128 plain, 4 h128 with pre signs s1 and post signs s2, 5 h128 with pre signs s1
// norm: 0 absmax, 1 rms, 2 fixed(1). fit: 0 none, 1 ls, 2 nc, 3 oct, 4 nc_free. scale_bytes: 2 fp16, 1 log8, 0.
void scalar_codec(const float* x, float* out, int64_t n, int rot, int group, const float* book, int nbook, int norm,
                  int fit, int scale_bytes, int iters, const float* s1, const float* s2, uint8_t* codes_out) {
    float edges[256];
    for (int i = 0; i + 1 < nbook; ++i) edges[i] = 0.5f * (book[i] + book[i + 1]);
    float bmax = 0;
    for (int i = 0; i < nbook; ++i) bmax = fmaxf(bmax, fabsf(book[i]));
    const int rw = rot == 1 ? 32 : rot == 2 ? 64 : 128;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < n; ++r) {
        float y[128], z[128];
        int idx[128];
        memcpy(y, x + r * 128, sizeof y);
        if (rot >= 4) for (int i = 0; i < 128; ++i) y[i] *= s1[i];
        if (rot) wht(y, rw);
        if (rot == 4) for (int i = 0; i < 128; ++i) y[i] *= s2[i];
        for (int g0 = 0; g0 < 128; g0 += group) {
            double s0;
            if (norm == 0) {
                float m = 0;
                for (int i = g0; i < g0 + group; ++i) m = fmaxf(m, fabsf(y[i]));
                s0 = m / bmax;
            } else if (norm == 1) {
                double e = 0;
                for (int i = g0; i < g0 + group; ++i) e += (double) y[i] * y[i];
                s0 = sqrt(e / group);
            } else s0 = 1.0;
            if (s0 <= 0) s0 = 1e-12;
            if (scale_bytes == 2 && fit == 0 && norm != 2) s0 = to_f16((float) s0);
            double s = s0;
            for (int it = 0; it <= iters; ++it) {
                const float inv = (float) (1.0 / (it == 0 ? s0 : (s > 1e-12 ? s : 1e-12)));
                if (it == 0 || fit != 0) for (int i = g0; i < g0 + group; ++i) idx[i] = nearest(y[i] * inv, edges, nbook - 1);
                double yz = 0, zz = 0, yy = 0;
                for (int i = g0; i < g0 + group; ++i) {
                    z[i] = book[idx[i]];
                    yz += (double) y[i] * z[i];
                    zz += (double) z[i] * z[i];
                    yy += (double) y[i] * y[i];
                }
                if (fit == 0) s = s0;
                else if (fit == 1) s = zz > 0 ? yz / zz : 0;
                else if (fit == 2) s = zz > 0 ? sqrt(yy / zz) : 0;
                else if (fit == 3) s = yz > 0 ? yy / yz : 0;
                else s = zz > 0 ? sqrt((double) group / zz) : 0;
                if (fit == 0) break;
            }
            const int stored = !(fit == 4 || (norm == 2 && fit == 0));
            if (stored) {
                if (scale_bytes == 2) s = to_f16((float) s);
                else if (scale_bytes == 1) {
                    double q = rint(log2(s > 1e-30 ? s : 1e-30) * 32) + 128;
                    if (q < 0) q = 0;
                    if (q > 255) q = 255;
                    s = exp2((q - 128) / 32);
                }
            }
            for (int i = g0; i < g0 + group; ++i) y[i] = (float) (z[i] * s);
            if (codes_out) for (int i = g0; i < g0 + group; ++i) codes_out[r * 128 + i] = (uint8_t) idx[i];
        }
        if (rot == 4) for (int i = 0; i < 128; ++i) y[i] *= s2[i];
        if (rot) wht(y, rw);
        if (rot >= 4) for (int i = 0; i < 128; ++i) y[i] *= s1[i];
        memcpy(out + r * 128, y, sizeof y);
    }
}
