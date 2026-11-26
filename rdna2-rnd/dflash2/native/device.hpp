#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace dflash2 {

inline void hip_check(hipError_t error, const char* task) {
    if (error != hipSuccess) throw std::runtime_error(std::string(task) + ": " + hipGetErrorString(error));
}

template<class T> class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { if (data_) (void)hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(size_t elements) {
        if (data_) throw std::logic_error("device buffer already allocated");
        hip_check(hipMalloc(&data_, elements * sizeof(T)), "hipMalloc");
    }
    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

class Stream {
public:
    Stream() { hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate"); }
    ~Stream() { (void)hipStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    operator hipStream_t() const { return stream_; }
    void wait() const { hip_check(hipStreamSynchronize(stream_), "draft stream"); }

private:
    hipStream_t stream_ = nullptr;
};

} // namespace dflash2
