// A prompt batch's projections (Mmq::mul) on the matrix units: Metal 4's MetalPerformancePrimitives matmul2d (on the
// M5, its neural accelerators), as the routed experts of metal/kernels/moe_mm.metal do for the Strata engine.  A
// threadgroup takes TM rows of one matrix (an expert's rows lie between its bounds) and TN of its outputs; per TK
// values of a row it stages the rows' fp16 activations and the TN x TK weights, dequantized by ggml's decoders and
// rounded to fp16, in threadgroup memory, and matmul2d multiplies them with fp32 sums.  The tile goes out through
// threadgroup memory, row r to row ids[r] of y.  ggml's pinned dequantize.h comes first (quantized.cpp).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct MmArgs {
    device void* p[5];   // the first matrix, the rows' fp16 activations, y, ids (or null), bounds (or null)
    ulong stride;        // bytes from one matrix to the next
    ulong row_bytes;     // bytes of one matrix row
    ulong ld;            // floats from one row of y to the next
    int ni, no, total;   // a row's values, the matrices' rows, the call's rows
};

template<typename Block,int QK,void Deq(device const Block*,short,thread half4x4&),int TM,int TN,int TK,int SG>
kernel void poly_mm(constant MmArgs& a [[buffer(0)]],uint3 tg [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH=SG*32,STAGED=(TM+TN)*TK/2,MEM=STAGED>TM*TN?STAGED:TM*TN;
    threadgroup float mem[MEM];   // the staged activations and weights, then the finished tile in their place
    threadgroup half* xa=reinterpret_cast<threadgroup half*>(mem);
    threadgroup half* wb=xa+TM*TK;
    threadgroup float* out=mem;
    device const int* bounds=static_cast<device const int*>(a.p[4]);
    device const int* ids=static_cast<device const int*>(a.p[3]);
    const int e=int(tg.z),lo=bounds?bounds[e]:0,hi=bounds?bounds[e+1]:a.total,r0=lo+int(tg.y)*TM;
    if(r0>=hi) return;   // past this matrix's rows: the whole threadgroup leaves
    const int f0=int(tg.x)*TN;
    device const uchar* w0=static_cast<device const uchar*>(a.p[0])+ulong(e)*a.stride+ulong(f0)*a.row_bytes;
    device const half* x=static_cast<device const half*>(a.p[1]);
    constexpr auto desc=matmul2d_descriptor(TM,TN,TK,false,true,false);
    matmul2d<desc,execution_simdgroups<SG>> op;
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> tA(xa,dextents<int32_t,2>(TK,TM));
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> tW(wb,dextents<int32_t,2>(TK,TN));
    auto acc=op.template get_destination_cooperative_tensor<decltype(tA),decltype(tW),float>();
    for(uint16_t i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)) acc[i]=0.0f;
    for(int k0=0;k0<a.ni;k0+=TK) {
        for(int it=tid;it<TM*TK/4;it+=NTH) {   // four values at a time
            const int m=it/(TK/4),r=r0+m;
            reinterpret_cast<threadgroup half4*>(xa)[it]=
                r<hi?reinterpret_cast<device const half4*>(x+ulong(r)*a.ni+k0)[it%(TK/4)]:half4(0.0h);
        }
        for(int it=tid;it<TN*(TK/16);it+=NTH) {
            const int n=it/(TK/16),k=k0+(it%(TK/16))*16;
            half4x4 v=half4x4(0.0h);
            if(f0+n<a.no) Deq(reinterpret_cast<device const Block*>(w0+ulong(n)*a.row_bytes)+k/QK,short((k%QK)/16),v);
            threadgroup half4* dst=reinterpret_cast<threadgroup half4*>(wb+n*TK+(it%(TK/16))*16);
            dst[0]=v[0];dst[1]=v[1];dst[2]=v[2];dst[3]=v[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto part=op.template get_destination_cooperative_tensor<decltype(tA),decltype(tW),float>();
        op.run(tA,tW,part);
        for(uint16_t i=0;i<part.get_capacity();++i) if(part.is_valid_element(i)) acc[i]+=part[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);   // also: no one reads xa and wb past here, out takes them
    }
    tensor<threadgroup float,dextents<int32_t,2>,tensor_inline> tO(out,dextents<int32_t,2>(TN,TM));
    acc.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* y=static_cast<device float*>(a.p[2]);
    for(int it=tid;it<TM*TN;it+=NTH) {
        const int m=it/TN,n=it%TN,r=r0+m;
        if(r<hi && f0+n<a.no) y[ulong(ids?ids[r]:r)*a.ld+f0+n]=out[it];
    }
}

// poly_mm_<type>_<TM>_<TN>_<TK>_<SG>: TM rows of a matrix, TN outputs, TK values a step, SG SIMD groups.  Every type
// has the default tile; the others are for poly_mm_bench, on the types of LFM2.5's prompt batch.
#define MM_(name,QK,TM,TN,TK,SG) template [[host_name("poly_mm_" #name "_" #TM "_" #TN "_" #TK "_" #SG)]] kernel decltype(poly_mm<block_##name,QK,dequantize_##name<half4x4>,TM,TN,TK,SG>) poly_mm<block_##name,QK,dequantize_##name<half4x4>,TM,TN,TK,SG>;
#define MM(name,QK) MM_(name,QK,32,64,64,8)
#define MM_TRY(name,QK) MM_(name,QK,32,64,128,4) MM_(name,QK,32,64,64,4) MM_(name,QK,32,64,128,8) MM_(name,QK,32,128,64,8) \
                        MM_(name,QK,32,64,64,16) MM_(name,QK,64,64,64,16)
MM(q4_0,32) MM(q4_1,32) MM(q5_0,32) MM(q5_1,32) MM(q8_0,32) MM(q2_0,64) MM(mxfp4,32)
MM(q2_K,256) MM(q3_K,256) MM(q4_K,256) MM(q5_K,256) MM(q6_K,256)
MM(iq2_xxs,256) MM(iq2_xs,256) MM(iq3_xxs,256) MM(iq1_s,256) MM(iq4_nl,32)
MM(iq3_s,256) MM(iq2_s,256) MM(iq4_xs,256) MM(iq1_m,256)
MM_TRY(q3_K,256) MM_TRY(iq3_xxs,256) MM_TRY(iq4_xs,256)
