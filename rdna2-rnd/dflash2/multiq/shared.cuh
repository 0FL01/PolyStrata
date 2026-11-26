// Standalone test path, included alongside the stock reader by bench.cu.
// One wave per (target query, GQA group). Only the raw K/V loads are shared.
template <int Rows, int Groups, int Steps>
__global__ void __launch_bounds__(32 * Rows * Groups, 2)
shared_q8_kernel(const float* q, const float* rel, const float* table, KvView k, KvView v,
                 int pos, int chunk, float* part) {
    static_assert(Steps % kAttUnroll == 0);
    constexpr int tile = Steps * 4;  // four keys per wave step, exactly as in the W1 reader
    struct Stage {
        int4 k[Steps * 32], v[Steps * 32];
        float2 scales[Steps * 32];
    };
    __shared__ Stage stage[Groups];

    const int wave = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int group = wave / Rows, row = wave % Rows, g = blockIdx.x * Groups + group;
    const int sub = lane & 7, quad = lane >> 3, c = blockIdx.y;
    const int first = c * chunk, last = min(first + chunk, pos + row + 1);
    const int last_any = min(first + chunk, pos + Rows);
    const size_t qh0 = (size_t) row * kHeads + g * kGroup;
    float qr[kGroup][16], rr[kGroup][2];
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        const float4* qv = reinterpret_cast<const float4*>(q + (qh0 + r) * kHead + 16 * sub);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 x = qv[j];
            qr[r][4 * j] = x.x;
            qr[r][4 * j + 1] = x.y;
            qr[r][4 * j + 2] = x.z;
            qr[r][4 * j + 3] = x.w;
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
    const auto* ksc = reinterpret_cast<const __half*>(k.scales);
    const auto* vsc = reinterpret_cast<const __half*>(v.scales);
    for (int block = first; block < last_any; block += tile) {
        // Distribute the work across all query waves, but load each lane's segment only once.
        // Scale loads coalesce over two adjacent lanes, like the stock kernel.
        for (int step = row; step < Steps; step += Rows) {
            const int i = block + 4 * step + quad;
            if (i < last_any) {
                const size_t at = ((size_t) i * kKv + g) * kHead + 16 * sub;
                const int ix = step * 32 + lane;
                stage[group].k[ix] = *reinterpret_cast<const int4*>(k.data + at);
                stage[group].v[ix] = *reinterpret_cast<const int4*>(v.data + at);
                stage[group].scales[ix] = make_float2(__half2float(ksc[at / 32]), __half2float(vsc[at / 32]));
            }
        }
        __syncthreads();
#pragma unroll 1
        for (int step = 0; step < Steps; step += kAttUnroll) {
            int4 kw[kAttUnroll], vw[kAttUnroll];
            float2 scales[kAttUnroll];
#pragma unroll
            for (int u = 0; u < kAttUnroll; ++u) {
                const int i = block + 4 * (step + u) + quad;
                if (i < last) {
                    const int ix = (step + u) * 32 + lane;
                    kw[u] = stage[group].k[ix];
                    vw[u] = stage[group].v[ix];
                    scales[u] = stage[group].scales[ix];
                }
            }
#pragma unroll
            for (int u = 0; u < kAttUnroll; ++u) {
                const int i = block + 4 * (step + u) + quad;
                if (i >= last) continue;
                const int delta = pos + row - i;
                const int kwords[4] = {kw[u].x, kw[u].y, kw[u].z, kw[u].w};
                const int vwords[4] = {vw[u].x, vw[u].y, vw[u].z, vw[u].w};
                float sc[kGroup];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    float d = 0.f;
#pragma unroll
                    for (int x = 0; x < 4; ++x)
#pragma unroll
                        for (int y = 0; y < 4; ++y) d += qr[r][4 * x + y] * (float) (int8_t) (kwords[x] >> (8 * y));
                    sc[r] = d * scales[u].x * 0.08838834764831845f;
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
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y < 4; ++y)
                        val[4 * x + y] = (float) (int8_t) (vwords[x] >> (8 * y)) * scales[u].y;
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
        __syncthreads();
    }
    // Exactly the stock quad fold, followed by direct scatter to the stock combine layout.
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
    float* dst = part + (((size_t) row * kKv + g) * gridDim.y + c) * kChunkFloats;
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
