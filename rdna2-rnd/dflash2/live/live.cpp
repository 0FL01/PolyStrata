#include <cstddef>
#include "live.hpp"
#include "draft_abi.h"
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace df_live {
namespace {
void *so, *model;
decltype(&df2_close) release;
decltype(&df2_append) append;
decltype(&df2_propose) block;
int64_t committed;
int available;
size_t stride;
const float* planes;

template<typename F> F symbol(const char* key) {
    auto p = reinterpret_cast<F>(dlsym(so, key));
    if (!p) throw std::runtime_error(dlerror());
    return p;
}
}
void open(const char* target, const void* head) {
    const char *lib = std::getenv("DF2_LIBRARY"), *weights = std::getenv("DF2_WEIGHTS");
    if (!lib || !weights || !head) throw std::runtime_error("need DF2_LIBRARY, DF2_WEIGHTS and the target head");
    so = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!so) throw std::runtime_error(dlerror());
    release = symbol<decltype(release)>("df2_close");
    append = symbol<decltype(append)>("df2_append");
    block = symbol<decltype(block)>("df2_propose");
    model = symbol<decltype(&df2_open)>("df2_open")(weights, target, head);
    if (!model) throw std::runtime_error("df2_open failed");
}
void close() {
    if (model) release(model);
    if (so) dlclose(so);
    model = so = nullptr;
}
void pending(int64_t pos, int n, const float* base, size_t plane) {
    if (committed==0 && pos>0 && n>=2048 && std::getenv("DF2_WARM_TARGET")) committed=pos;
    if (!model || available || pos != committed || n <= 0) throw std::logic_error("non-contiguous DFlash2 taps");
    available = n;
    planes = base;
    stride = plane;
}
void commit(int n) {
    if (n <= 0 || n > available) throw std::logic_error("bad DFlash2 commit");
    if (append(model, committed, n, planes, planes + stride, planes + 2*stride, planes + 3*stride))
        throw std::runtime_error("df2_append failed at " + std::to_string(committed));
    available = 0;
    committed += n;
}
void propose(int32_t anchor, int k, int32_t* out) {
    if (available || !model || k < 1 || k > 7) throw std::logic_error("bad proposal boundary");
    if (block(model, committed, anchor, k, out)) throw std::runtime_error("df2_propose failed at " + std::to_string(committed));
    for (int i = 0; i < k; ++i)
        if (out[i] < 0 || out[i] >= 200058) throw std::runtime_error("invalid DFlash2 proposal");
}
}
