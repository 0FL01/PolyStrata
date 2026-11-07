#pragma once
#include "strata/gpu/metal.hpp"
#include <string>
namespace strata::poly::metal_detail {
struct QArgs {
    const void* p[24]{};
    uint64_t stride=0,xstride=0,ystride=0,offset=0;
    int ni=0,no=0,rows=0,count=0,every=1,kind=0;
};
static_assert(sizeof(QArgs)==248);

void qrun(const std::string&,const QArgs&,void*,gpu::metal::Grid);
const char* quant_name(int);
/// Mmq::quantize() keeps rows of `cols` values in fp16 and Mmq::mul() multiplies them on the matrix units (mm.metal):
/// with --metal-optimize, where matmul2d compiles (Metal 4) and cols is whole 64-value steps.  Else q8_1 rows and
/// poly_mul, as before.
bool mm_rows(int64_t cols);
/// Off: q8_1 rows and poly_mul even where mm_rows() could hold (STRATA_METAL_MM=0 sets it at start), for tests and
/// comparisons.
void allow_mm_rows(bool on);
/// mm.metal's tile as "rows,outputs,values,simdgroups" (an instance mm.metal has); null: the default.  For benchmarks.
void set_mm_tile(const char* tile);
}
