#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace strata::poly::divtrace {

inline FILE* tape = nullptr;
inline int visit = -1;
inline int64_t base = -1;

inline bool enabled(const char* name) {
    const char* s = std::getenv(name);
    return s && s[0] == '1' && s[1] == '\0';
}

inline void write(const void* p, size_t n) {
    if (std::fwrite(p, 1, n, tape) != n) throw std::runtime_error("divergence tape write failed");
}

struct Scope {
    explicit Scope(const char* path, int64_t pos) {
        tape = std::fopen(path, "wb");
        if (!tape) throw std::runtime_error("cannot open divergence tape");
        base = pos;
        write("DVT1", 4);
    }
    ~Scope() {
        std::fclose(tape);
        tape = nullptr;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

// Keys are absolute token position, visit, stage. Header fields are little-endian on the RDNA2 host.
// kind=0: FP32; kind=1: opaque bytes (Q8 rows etc.); kind=2: signed int32.
inline void host(const char* name, int64_t pos, int stage_visit, const void* data, int row_bytes, int rows,
                 int kind = 0) {
    char key[32] = {};
    if (std::snprintf(key, sizeof(key), "%s", name) >= (int) sizeof(key))
        throw std::runtime_error("divergence stage name too long");
    for (int r = 0; r < rows; ++r) {
        const int64_t at = pos + r;
        const int32_t fields[]{stage_visit, kind, row_bytes};
        write(&at, sizeof(at));
        write(fields, sizeof(fields));
        write(key, sizeof(key));
        write(static_cast<const uint8_t*>(data) + (size_t) r * row_bytes, (size_t) row_bytes);
    }
}

#ifdef DIVTRACE_DEVICE_CAPTURE
inline void device(const char* name, int64_t pos, int stage_visit, const void* data, int row_bytes, int rows,
                   void* stream, int kind = 0) {
    if (!tape) return;
    static std::vector<uint8_t> copy;
    copy.resize((size_t) row_bytes * rows);
    ck(cudaStreamSynchronize(cs(stream)), "divergence capture stream");
    ck(cudaMemcpy(copy.data(), data, copy.size(), cudaMemcpyDeviceToHost), "divergence capture copy");
    host(name, pos, stage_visit, copy.data(), row_bytes, rows, kind);
}
#endif

}  // namespace strata::poly::divtrace
