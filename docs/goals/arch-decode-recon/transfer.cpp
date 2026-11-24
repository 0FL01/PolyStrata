// CP-P1-03. Isolated optimistic follower service times, not decode or overlapped throughput.
// Build with the HIP runtime headers/library; run only under the RDNA2 benchmark guard.
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); }
}
using clock_type = std::chrono::steady_clock;
static double us(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double,std::micro>(b-a).count();
}
int main() {
    constexpr size_t matrix = 2304u*1152/32*18, slot = 3*matrix, experts = 32;
    std::vector<unsigned char> pages(slot*experts), out(slot);
    for (size_t i=0;i<pages.size();++i) pages[i] = static_cast<unsigned char>((i/slot)*17+i*31);
    unsigned char *host = nullptr, *gpu = nullptr;
    check(hipSetDevice(0)); check(hipHostMalloc((void**)&host,slot)); check(hipMalloc((void**)&gpu,slot));
    hipStream_t s; check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
    std::puts("mode,rep,bytes,pack_us,submit_us,complete_after_submit_start_us");
    // Small DMA measures the fixed host/driver/fence cost. Full slots bound best-case useful bytes/prediction.
    for (int round=0;round<2;++round) for (int order=0;order<3;++order) {
        int mode = round ? 2-order : order;
        size_t n = mode==0 ? 4 : slot;
        std::memcpy(host,pages.data(),slot);
        for (int i=-4;i<64;++i) {
            const size_t ix = size_t(i+4)%experts;
            auto a = clock_type::now();
            if (mode==2) std::memcpy(host,pages.data()+slot*ix,n);
            auto b = clock_type::now(); check(hipMemcpyAsync(gpu,host,n,hipMemcpyHostToDevice,s));
            auto c = clock_type::now(); check(hipStreamSynchronize(s)); auto d = clock_type::now();
            if (i>=0) std::printf("%s,%d,%zu,%.3f,%.3f,%.3f\n", mode==0?"small":mode==1?"pinned":"pack",round*64+i,n,us(a,b),us(b,c),us(b,d));
        }
        check(hipMemcpy(out.data(),gpu,n,hipMemcpyDeviceToHost));
        if (std::memcmp(out.data(),host,n)!=0) return 2;
    }
    check(hipStreamDestroy(s)); check(hipFree(gpu)); check(hipHostFree(host));
}
