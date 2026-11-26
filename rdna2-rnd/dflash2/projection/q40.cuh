// One wave per output row, eight activation columns, n_in=2304. Order of
// the five half-block visits emulates the stock four-wave reduction.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "strata/hip_compat/intrinsics.hpp"

namespace df_proj {
struct Q40 { __half d; unsigned short q[8]; };
struct Q81 { __half2 ds; int q[8]; };
static_assert(sizeof(Q40)==18 && sizeof(Q81)==36);

template<bool Stage, bool Split = false> __global__ __launch_bounds__(128)
void wave8(const Q40* __restrict__ w, const Q81* __restrict__ x, float* __restrict__ y, int rows) {
    constexpr int nb=72, nc=8;
    extern __shared__ int lds[];
    if constexpr (Stage) {
        for (int i=threadIdx.x;i<nb*nc*9;i+=128) lds[i]=reinterpret_cast<const int*>(x)[i];
        __syncthreads();
        x=reinterpret_cast<const Q81*>(lds);
    }
    const int row=int(blockIdx.x)*4+int(threadIdx.x)/32, lane=int(threadIdx.x)%32;
    if (row>=rows) return;
    const int part=2*(lane%2);
    float s[nc]={};
    #pragma unroll
    for (int step=0;step<5;++step) {
        const int b=lane/2 + (step==1 ? 64 : step>1 ? 16*(step-1) : 0);
        if (b>=nb) continue;
        const Q40& weight=w[row*nb+b];
        const float d=__half2float(weight.d);
        const unsigned short* p=weight.q+part*2;
        const unsigned v0=p[0]|(unsigned(p[1])<<16), v1=p[2]|(unsigned(p[3])<<16);
        #pragma unroll
        for (int c=0;c<nc;++c) {
            const Q81& a=x[c*nb+b];
            int z=__dp4a(v0&0x0f0f0f0f, a.q[part], 0);
            z=__dp4a((v0>>4)&0x0f0f0f0f, a.q[part+4], z);
            z=__dp4a(v1&0x0f0f0f0f, a.q[part+1], z);
            z=__dp4a((v1>>4)&0x0f0f0f0f, a.q[part+5], z);
            const float2 ds=__half22float2(a.ds);
            const float v=z*ds.x-4*ds.y;
            if (step<=1) s[c]+=d*v;
            else if constexpr (Split) {
                // hip's __fmul_rn alone did NOT prevent this multiply from
                // contracting into the subsequent v_fma_mix in the v1 ISA.
                float rounded; asm("v_mul_f32 %0, %1, %2" : "=v"(rounded) : "v"(d), "v"(v));
                s[c]+=rounded;
            } else s[c]+=d*v;
        }
    }
    #pragma unroll
    for (int c=0;c<nc;++c) {
        #pragma unroll
        for (int mask=16;mask;mask/=2) s[c]+=__shfl_xor(s[c],mask,32);
        if (!lane) y[c*rows+row]=s[c];
    }
}
}
