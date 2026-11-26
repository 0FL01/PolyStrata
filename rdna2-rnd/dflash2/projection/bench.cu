// Timings exclude quantizing the eight input rows, host work, top-k and verification.
#include "q40.cuh"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace df_proj;
namespace k = strata::kernels;
void ck(hipError_t e) { if (e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template<typename T> struct Dev {
    T* p=nullptr; explicit Dev(size_t n){ck(hipMalloc(&p,n*sizeof(T)));} ~Dev(){(void)hipFree(p);}
};
__global__ void sweep(unsigned* u,int n){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) u[i]=u[i]*17+1;
}
int main(int argc,char** argv) try {
    if (argc!=7) throw std::invalid_argument("bench WEIGHTS offset GGML_TYPE n_out n_in surrogate.f32");
    const int type=std::stoi(argv[3]), rows=std::stoi(argv[4]), in=std::stoi(argv[5]), nc=8;
    if (in<=0 || in>9216 || rows<=0 || rows>201024 || in%256) throw std::invalid_argument("shape");
    const size_t bytes=k::native_mmvq_weight_bytes(type,in,rows), n=size_t(nc)*rows;
    if (!bytes) throw std::invalid_argument("unsupported quant");
    std::vector<char> data(bytes); std::vector<float> x(size_t(in)*nc), ref(n), got(n);
    std::ifstream f(argv[1],std::ios::binary), fx(argv[6],std::ios::binary);
    if (!f.seekg(std::stoull(argv[2])).read(data.data(),bytes) || !fx.read(reinterpret_cast<char*>(x.data()),x.size()*4))
        throw std::runtime_error("short input");
    hipDeviceProp_t prop{}; ck(hipGetDeviceProperties(&prop,0));
    if(std::string(prop.gcnArchName).find("gfx1031")!=0 || prop.warpSize!=32) throw std::runtime_error("gfx1031 only");
    Dev<char> w(bytes), q(k::native_q8_1_bytes(in)*nc); Dev<float> xd(x.size()),out(n);
    constexpr int ne=(128<<20)/4; Dev<unsigned> ev(ne);
    ck(hipMemcpy(w.p,data.data(),bytes,hipMemcpyHostToDevice)); ck(hipMemcpy(xd.p,x.data(),x.size()*4,hipMemcpyHostToDevice));
    hipStream_t st; ck(hipStreamCreate(&st)); hipEvent_t a,b;ck(hipEventCreate(&a));ck(hipEventCreate(&b));
    ck(hipMemsetAsync(ev.p,0,ne*4,st)); k::native_quantize_q8_1(xd.p,q.p,in*nc,1,st);
    const bool test_wave=(type==2 && in==2304), cols=(type==8 || type==14) && size_t(nc)*in/32*36<=65536;
    const int arms=test_wave?4:cols?2:1;
    auto launch=[&](int arm) {
        if (!arm) k::native_mmvq(type,w.p,q.p,out.p,in,rows,nc,st);
        else if (test_wave && arm==1) wave8<false><<<(rows+3)/4,128,0,st>>>((Q40*)w.p,(Q81*)q.p,out.p,rows);
        else if (test_wave && arm==2) wave8<true><<<(rows+3)/4,128,8*72*36,st>>>((Q40*)w.p,(Q81*)q.p,out.p,rows);
        else if (test_wave) wave8<false,true><<<(rows+3)/4,128,0,st>>>((Q40*)w.p,(Q81*)q.p,out.p,rows);
        else if (!(type==8 ? k::native_q8_0_mmvq_warps_cols : k::native_q6_k_mmvq_warps_cols)(w.p,q.p,out.p,in,rows,nc,st))
            throw std::runtime_error("no cols dispatch");
        ck(hipGetLastError());
    };
    for(int arm=0;arm<arms;++arm){
        launch(arm); ck(hipStreamSynchronize(st)); ck(hipMemcpy(got.data(),out.p,n*4,hipMemcpyDeviceToHost));
        if (!arm) ref=got;
        size_t diff=0; float peak=0;
        for (size_t i=0;i<n;++i){if(!std::isfinite(got[i])) throw std::runtime_error("non-finite output");
            diff+=std::memcmp(&ref[i],&got[i],4)!=0;peak=std::max(peak,std::abs(ref[i]-got[i]));}
        std::printf("CHECK arm=%d different=%zu/%zu max_abs=%.9g\n",arm,diff,n,peak);
    }
    for(int cold : {0,1}) {
        std::vector<std::vector<float>> t(arms);
        for(int r=0;r<34;++r)for(int j=0;j<arms;++j){
            const int arm=(j+r)%arms;
            if(cold) sweep<<<640,256,0,st>>>(ev.p,ne);
            ck(hipEventRecord(a,st));launch(arm);ck(hipEventRecord(b,st));ck(hipEventSynchronize(b));
            float ms;ck(hipEventElapsedTime(&ms,a,b));if(r>=3)t[arm].push_back(ms);
        }
        for(int arm=0;arm<arms;++arm){std::sort(t[arm].begin(),t[arm].end());const float ms=t[arm][15];
            std::printf("TIME type=%d rows=%d in=%d arm=%d cold=%d p50_us=%.3f p90_us=%.3f payload_GBs=%.2f\n",
                type,rows,in,arm,cold,1e3*ms,1e3*t[arm][27],double(bytes)/(1e6*ms));}
    }
    ck(hipEventDestroy(b));ck(hipEventDestroy(a));ck(hipStreamDestroy(st));
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
