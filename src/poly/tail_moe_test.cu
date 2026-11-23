// src/poly/tail_moe_test.cu - conv_add_kernel's folded expert sum against add_scaled() + the kernel as it was.
//
// The engine's STRATA_INKLING_TAIL_MOE path (inkling2.cu: conv_add_kernel with a TailSum record - the code as it
// is there, copied below) against the launches it stands for: kernels::add_scaled() of the record's rows into the
// zeroed fo_ (moe_shared() left it 0), then conv_add_kernel as it was without the record (git HEAD's, copied below
// too), the tail launch's other work unchanged around the sums.  One token's experts, as the record is at n == 1:
// the same bytes to both paths, and x (fo_ kept or cleaned), the residual r and the conv state must come out the
// same bits.  This is the gate: every "FAIL" line fails the run.
//
// The kernels are inkling2.cu's own code, so what runs here is what the engine runs up to the compilation unit;
// the engine's kernel itself is checked besides this at SASS level (its conv segment's arithmetic did not move).
//
// usage: poly_tail_moe_test   (exit 0: every comparison the same bits)
#include "runtime.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata;
using namespace strata::poly;

// ---- the copies of inkling2.cu's kernel code (marked where the two versions differ) ----

// inkling2.cu's conv_channel(): unchanged by the record, the same text in both versions.
__device__ __forceinline__ void conv_channel(float* x, float* state, const float* w, const float* dynamic, int width,
                                             int n, int i, float* snap = nullptr, size_t stride = 0) {
    float* h = state + 3 * i;
    float a = h[0], b = h[1], c = h[2];
    w += 4 * i;
    const float scale = dynamic ? 1.f / 3.f : 0.4472135954999579f;
    for (int t = 0; t < n; ++t) {
        const size_t at = (size_t) t * width + i;
        const float d = x[at];
        float s = a * w[0] + b * w[1] + c * w[2] + d * w[3];
        if (dynamic) {
            const float* p = dynamic + 4 * (t * (width / 128) + i / 128);
            s += d * p[0] + c * p[1] + b * p[2] + a * p[3];
        }
        x[at] = (d + s) * scale;
        a = b;
        b = c;
        c = d;
        if (snap && t + 1 < n) {
            float* k = snap + (size_t) t * stride + 3 * i;
            k[0] = a;
            k[1] = b;
            k[2] = c;
        }
    }
    h[0] = a;
    h[1] = b;
    h[2] = c;
}

// inkling2.cu's conv_add_kernel() as it was before the record (git HEAD), the tail launch moe() used to feed.
__global__ void conv_add_kernel_ref(float* x, const float* add, float* state, const float* w, float* r, float* depth,
                                    int width, int n, bool clean, float* snap = nullptr, size_t stride = 0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    if (add)
        for (int t = 0; t < n; ++t) x[(size_t) t * width + i] = x[(size_t) t * width + i] + add[(size_t) t * width + i];
    conv_channel(x, state, w, nullptr, width, n, i, snap, stride);
    for (int t = 0; t < n; ++t) {
        const size_t at = (size_t) t * width + i;
        if (depth) {
            depth[at] = r[at];
            r[at] = x[at];
        } else r[at] = r[at] + x[at];
        if (clean) x[at] = 0.f;   // fo_: the next layer's experts add into it (moe_shared())
    }
}

// inkling2.cu's conv_add_kernel() with the record (the working tree's), the sums folded in ahead of `add`.
__global__ void conv_add_kernel_new(float* x, const float* add, float* state, const float* w, float* r, float* depth,
                                    int width, int n, bool clean, TailSum moe, float* snap = nullptr,
                                    size_t stride = 0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    if (moe.n)
        for (int t = 0; t < n; ++t) {
            const size_t at = (size_t) t * width + i;
            // fo_ was 0 where add_scaled() started (moe_shared()), so this chain is its sums bit for bit
            float v = 0.f;
            for (int k = 0; k < moe.n; ++k) v = v + moe.x[k][at] * moe.w[k];
            x[at] = v;
        }
    if (add)
        for (int t = 0; t < n; ++t) x[(size_t) t * width + i] = x[(size_t) t * width + i] + add[(size_t) t * width + i];
    conv_channel(x, state, w, nullptr, width, n, i, snap, stride);
    for (int t = 0; t < n; ++t) {
        const size_t at = (size_t) t * width + i;
        if (depth) {
            depth[at] = r[at];
            r[at] = x[at];
        } else r[at] = r[at] + x[at];
        if (clean) x[at] = 0.f;   // fo_: the next layer's experts add into it (moe_shared())
    }
}

// ---- the test ----

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> floats(const float* d, size_t n, bool* ok) {
    std::vector<float> v(n);
    if (const cudaError_t e = cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost); e != cudaSuccess) {
        std::printf("D2H %p +%zu: %s\n", (const void*) d, n, cudaGetErrorString(e));
        if (ok) *ok = false;   // the comparison below must not call zeros the same bits
    }
    return v;
}

constexpr int kThreads = 128;   // inkling2.cu's, the tail launch's block

// One combo: n_exp experts (0: the record empty, add_scaled() never launched), with or without the CPU experts'
// `add`, fo_ kept or cleaned by the tail as the engine has it.  Returns the number of FAILs.
int check(int width, int n_exp, bool with_add, bool clean, std::mt19937& rng) {
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    // the record moe() leaves: the down rows (the scratch ring's) and the router's weights, in the router's order
    std::vector<std::vector<float>> rows((size_t) n_exp, std::vector<float>((size_t) width));
    for (auto& row : rows)
        for (float& x : row) x = ud(rng);
    std::vector<float> wts((size_t) n_exp);
    for (float& x : wts) x = 0.05f + 0.25f * ud(rng);   // the router's: positive and small
    // the visit's other inputs: the conv taps, the state, the residual, the CPU experts' sum
    std::vector<float> cw((size_t) 4 * width), state((size_t) 3 * width), r0((size_t) width), add((size_t) width);
    for (float& x : cw) x = 0.5f * ud(rng);
    for (float& x : state) x = ud(rng);
    for (float& x : r0) x = 2.f * ud(rng);
    for (float& x : add) x = 0.5f * ud(rng);

    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    std::vector<const float*> d_rows;
    for (const auto& row : rows) d_rows.push_back(upload(row));
    float *d_cw = upload(cw), *d_state0 = upload(state), *d_r00 = upload(r0), *d_add = upload(add);
    const dim3 grid((unsigned) ((width + kThreads - 1) / kThreads));

    // the reference: add_scaled() into the zeroed fo_, then the old kernel
    float *x1 = nullptr, *state1 = nullptr, *r1 = nullptr;
    cudaMalloc(&x1, (size_t) width * sizeof(float));
    cudaMemset(x1, 0, (size_t) width * sizeof(float));   // moe_shared() left fo_ 0
    cudaMalloc(&state1, state.size() * sizeof(float));
    cudaMemcpy(state1, d_state0, state.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMalloc(&r1, r0.size() * sizeof(float));
    cudaMemcpy(r1, d_r00, r0.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    if (n_exp > 0) {
        ScaledRows sum{};
        for (int k = 0; k < n_exp; ++k) {
            sum.x[k] = d_rows[(size_t) k];
            sum.acc[k] = x1;
            sum.w[k] = wts[(size_t) k];
        }
        add_scaled(sum, n_exp, width, s);
    }
    conv_add_kernel_ref<<<grid, kThreads, 0, s>>>(x1, with_add ? d_add : nullptr, state1, d_cw, r1, nullptr, width, 1,
                                                 clean, nullptr, 0);

    // the record: the one launch, the experts' sums folded in
    float *x2 = nullptr, *state2 = nullptr, *r2 = nullptr;
    cudaMalloc(&x2, (size_t) width * sizeof(float));
    cudaMemset(x2, 0, (size_t) width * sizeof(float));
    cudaMalloc(&state2, state.size() * sizeof(float));
    cudaMemcpy(state2, d_state0, state.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMalloc(&r2, r0.size() * sizeof(float));
    cudaMemcpy(r2, d_r00, r0.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    TailSum moe{};
    for (int k = 0; k < n_exp; ++k) {
        moe.x[k] = d_rows[(size_t) k];
        moe.w[k] = wts[(size_t) k];
    }
    moe.n = n_exp;
    conv_add_kernel_new<<<grid, kThreads, 0, s>>>(x2, with_add ? d_add : nullptr, state2, d_cw, r2, nullptr, width, 1,
                                                 clean, moe, nullptr, 0);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("launch: %s\n", cudaGetErrorString(e));
        return 1;
    }
    if (const cudaError_t e = cudaStreamSynchronize(s); e != cudaSuccess) {
        std::printf("sync: %s\n", cudaGetErrorString(e));
        return 1;
    }

    bool read_ok = true;
    const std::vector<float> fx1 = floats(x1, (size_t) width, &read_ok), fx2 = floats(x2, (size_t) width, &read_ok);
    const std::vector<float> fs1 = floats(state1, state.size(), &read_ok), fs2 = floats(state2, state.size(), &read_ok);
    const std::vector<float> fr1 = floats(r1, r0.size(), &read_ok), fr2 = floats(r2, r0.size(), &read_ok);
    if (!read_ok) {
        std::printf("FAIL width %4d experts %d add %s clean %d: readback failed\n", width, n_exp,
                    with_add ? "yes" : "no ", clean);
        cudaFree(x1), cudaFree(state1), cudaFree(r1), cudaFree(x2), cudaFree(state2), cudaFree(r2);
        for (const float* d : d_rows) cudaFree((void*) d);
        cudaFree(d_cw), cudaFree(d_state0), cudaFree(d_r00), cudaFree(d_add);
        cudaStreamDestroy(s);
        return 1;
    }
    int bad = 0;
    if (std::memcmp(fx1.data(), fx2.data(), fx1.size() * sizeof(float)) != 0) ++bad;
    if (std::memcmp(fs1.data(), fs2.data(), fs1.size() * sizeof(float)) != 0) ++bad;
    if (std::memcmp(fr1.data(), fr2.data(), fr1.size() * sizeof(float)) != 0) ++bad;
    std::printf("%s width %4d experts %d add %s clean %d: fo_ %s, state %s, residual %s\n",
                bad ? "FAIL" : "ok  ", width, n_exp, with_add ? "yes" : "no ", clean,
                std::memcmp(fx1.data(), fx2.data(), fx1.size() * sizeof(float)) == 0 ? "same" : "DIFF",
                std::memcmp(fs1.data(), fs2.data(), fs1.size() * sizeof(float)) == 0 ? "same" : "DIFF",
                std::memcmp(fr1.data(), fr2.data(), fr1.size() * sizeof(float)) == 0 ? "same" : "DIFF");

    cudaFree(x1), cudaFree(state1), cudaFree(r1), cudaFree(x2), cudaFree(state2), cudaFree(r2);
    for (const float* d : d_rows) cudaFree((void*) d);
    cudaFree(d_cw), cudaFree(d_state0), cudaFree(d_r00), cudaFree(d_add);
    cudaStreamDestroy(s);
    return bad;
}

}  // namespace

int main() {
    std::mt19937 rng(23);
    int bad = 0;
    // kWidth = 2304 is Inkling-2-Mini's; the record empty (0 experts) is the fallback shape every model keeps
    for (int width : {256, 1024, 2304})
        for (int n_exp : {0, 1, 2, 5, 8})
            for (bool with_add : {false, true}) {
                bad += check(width, n_exp, with_add, false, rng);   // fo_ kept: its bits compared too
                bad += check(width, n_exp, with_add, true, rng);   // the engine's tail: fo_ cleaned
            }
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
