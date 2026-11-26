#pragma once

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "cuda_runtime.h"

namespace strata::poly::dft1 {

inline constexpr int width = 2304;
static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);

inline const char* enabled() {
    static const char* path = std::getenv("DFT1_CAPTURE");
    return path;
}

inline float* norm = nullptr;  // output_rows scratch, only needed without DSpark

class Tape {
    FILE* file_ = nullptr;
    std::vector<float> host_;
    int64_t rows_ = 0, generated_ = 0;
    uint64_t layout_ = 0;

    template<class Layers>
    static uint64_t layout(const Layers& layers) {
        uint64_t hash = 14695981039346656037ull;
        for (const auto& layer : layers) {
            uintptr_t first = UINTPTR_MAX;
            for (const auto* p : layer.slot)
                if (p && reinterpret_cast<uintptr_t>(p) < first) first = reinterpret_cast<uintptr_t>(p);
            for (const auto* p : layer.slot) {
                uint64_t value = p ? reinterpret_cast<uintptr_t>(p) - first : UINT64_MAX;
                for (int i = 0; i < 8; ++i, value >>= 8) hash = (hash ^ (value & 255)) * 1099511628211ull;
            }
        }
        return hash;
    }

    void put(const void* data, size_t size) {
        if (std::fwrite(data, 1, size, file_) != size) throw std::runtime_error("DFT1 write failed");
    }

public:
    ~Tape() { if (file_) std::fclose(file_); }
    bool active() const { return file_ != nullptr; }

    template<class Layers>
    void write(int64_t pos, int n, const int32_t* tokens, bool decode, const float* taps, const float* end,
               int stride, const Layers& layers, void* stream) {
        if (pos != rows_ || n <= 0 || !tokens || (decode ? n != 1 : generated_ != 0))
            throw std::logic_error("DFT1 is one contiguous prompt plus W1 decode");
        if (!file_) {
            file_ = std::fopen(enabled(), "wbx");
            if (!file_) throw std::runtime_error("cannot create DFT1 tape");
            put("DFT1", 4);
            layout_ = layout(layers);
        }
        const size_t size = (size_t) n * width;
        host_.resize(4 * size);
        for (int j = 0; j < 4; ++j) {
            const float* plane = j == 3 ? end : taps + (size_t) j * stride * width;
            if (cudaMemcpyAsync(host_.data() + j * size, plane, size * sizeof(float), cudaMemcpyDeviceToHost,
                                static_cast<cudaStream_t>(stream)) != cudaSuccess)
                throw std::runtime_error("DFT1 tap download failed");
        }
        if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) != cudaSuccess)
            throw std::runtime_error("DFT1 tap synchronization failed");
        const int32_t flag = decode;
        for (int i = 0; i < n; ++i, ++rows_) {
            put(&rows_, sizeof(rows_));
            put(tokens + i, sizeof(*tokens));
            put(&flag, sizeof(flag));
            for (int j = 0; j < 4; ++j) put(host_.data() + j * size + i * width, width * sizeof(float));
        }
        generated_ += decode ? n : 0;
    }

    template<class Layers>
    void finish(const Layers& layers, int64_t swaps) noexcept {
        const auto end = layout(layers);
        const int rc = std::fclose(file_);
        file_ = nullptr;
        if (!rc && end == layout_ && !swaps)
            std::fprintf(stderr, "DFT1_COMPLETE rows=%lld generated=%lld layout=%016llx swaps=0\n",
                         (long long) rows_, (long long) generated_, (unsigned long long) layout_);
    }
};

inline Tape tape;

}  // namespace strata::poly::dft1
