// CPU-only, read-only RECON. Approximate *hints*, never route target experts with this.
// Reads Tideglass's real F32 router and PSCPU01 normalized expert inputs.
#include "strata/artifact/gguf_reader.hpp"
#include <array>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <numeric>

namespace {
constexpr int W = 2304, E = 128, L = 24, K = 4;
struct Frame { int64_t pos; int32_t v,l,n,gpu,down,res,id[8]; float w[8], x[W], sum[W]; };
static_assert(sizeof(Frame) == 18528);
struct Net {
    const float *f, *b;
    std::vector<int8_t> q;
    std::array<float,E> scale;
    std::array<int,E> offset;
};
void require(bool b, const char* s) { if (!b) throw std::runtime_error(s); }
int layer(int v) { return v < 6 ? v : v >= 42 ? v - 24 : 6 + (v - 6) % 12; }
using Ids = std::array<int,K>;
using Scores = std::array<float,E>;
__attribute__((noinline)) Ids predict(const Net& a, const float* x, bool packed) {
    Scores scores;
    alignas(32) uint8_t qx[W];
    float xs = 1;
    if (packed) {
        float mx = 0;
        for (int j=0;j<W;++j) mx = std::max(mx, std::abs(x[j]));
        xs = mx ? mx / 127.f : 1.f;
        const float inv = 1.f / xs;
        for (int j=0;j<W;++j) qx[j] = uint8_t(int(std::nearbyint(x[j] * inv)) + 127);
    }
    for (int e=0;e<E;++e) {
        float dot = 0;
        if (packed) {
            // qx in [0,254], w in [-63,63]: maddubs' pair fits int16.
            const auto one = _mm256_set1_epi16(1);
            auto acc = _mm256_setzero_si256();
            for (int j=0;j<W;j+=32) {
                auto u = _mm256_load_si256((const __m256i*)(qx+j));
                auto v = _mm256_loadu_si256((const __m256i*)(a.q.data()+e*W+j));
                acc = _mm256_add_epi32(acc,_mm256_madd_epi16(_mm256_maddubs_epi16(u,v),one));
            }
            alignas(32) int lanes[8]; _mm256_store_si256((__m256i*)lanes,acc);
            dot = float(std::accumulate(lanes,lanes+8,-a.offset[e])) * (xs * a.scale[e]);
        } else {
            auto acc = _mm256_setzero_ps();
            for (int j=0;j<W;j+=8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x+j),_mm256_loadu_ps(a.f+e*W+j),acc);
            alignas(32) float lanes[8]; _mm256_store_ps(lanes,acc);
            for (auto f:lanes) dot += f;
        }
        scores[e] = 1.f/(1.f+std::exp(-dot)) + a.b[e];
    }
    Ids ids{};
    for (int& i:ids) { i = std::max_element(scores.begin(),scores.end())-scores.begin(); scores[i] = -INFINITY; }
    return ids;
}
}

int main(int argc,char** argv) try {
    require(argc==4,"router_sidecar MODEL.gguf PSCPU01 ROUNDS");
    strata::GgufFile gg(argv[1]); std::array<Net,L> nets;
    for (int l=0;l<L;++l) {
        auto& a=nets[l];
        auto tensor=[&](const char* key, std::vector<uint64_t> shape) {
            const auto* t=gg.find("blk."+std::to_string(l)+"."+key);
            require(t && t->type==0 && t->shape==shape,"unexpected router tensor");
            return reinterpret_cast<const float*>(gg.tensor_data(*t));
        };
        a.f=tensor("ffn_gate_inp.weight",{W,E}); a.b=tensor("exp_probs_b.bias",{E}); a.q.resize(E*W);
        for(int e=0;e<E;++e) {
            float mx=0; for(int j=0;j<W;++j) mx=std::max(mx,std::abs(a.f[e*W+j]));
            a.scale[e]=mx ? mx/63.f : 1.f; int sum=0;
            for(int j=0;j<W;++j) sum+=(a.q[e*W+j]=int8_t(std::nearbyint(a.f[e*W+j]/a.scale[e])));
            a.offset[e]=sum*127;
        }
    }
    std::ifstream in(argv[2],std::ios::binary); char magic[8]; uint32_t h[4];
    require(bool(in.read(magic,8).read((char*)h,16)) && !memcmp(magic,"PSCPU01\n",8) && h[1]==W && h[0] && h[0]<=h[3]*48,"bad capture header");
    std::vector<Frame> frames(h[0]); require(bool(in.read((char*)frames.data(),sizeof(Frame)*frames.size())) && in.peek()==EOF,"bad capture length");
    std::map<std::pair<int64_t,int>,const Frame*> by_visit;
    for (const auto& f:frames) {
        require(f.v>=0 && f.v<48 && layer(f.v)==f.l && f.n>=1 && f.n<=8 && f.gpu+f.n==8 && f.pos>=h[2] && f.pos<h[2]+h[3],"bad capture row");
        require(by_visit.emplace(std::make_pair(f.pos,f.v),&f).second,"duplicate row");
    }
    int rounds = std::stoi(argv[3]); require(rounds>0,"no rounds");
    std::cout<<"round,pos,visit,arm,ns,h0,h1,h2,h3,nmiss,hit\n";
    // First sweep populates both representations. Separate whole sweeps, alternating order.
    const auto bench=[&](int round,bool packed) {
        for (const auto& f:frames) if (f.v<47) {
            auto start=std::chrono::steady_clock::now();
            Ids ids=predict(nets[layer(f.v+1)],f.x,packed);
            auto dt=std::chrono::steady_clock::now()-start;
            int hits=0,n=0;
            if (auto it=by_visit.find({f.pos,f.v+1});it!=by_visit.end()) {
                const auto& next=*it->second; n=next.n;
                for(int i:ids) hits+=std::find(next.id,next.id+n,i)!=next.id+n;
            }
            if (round>=0) {
                std::cout<<round<<','<<f.pos<<','<<f.v<<','<<packed<<','<<std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();
                for (auto id:ids) std::cout<<','<<id;
                std::cout<<','<<n<<','<<hits<<'\n';
            }
        }
    };
    for (int r=-1;r<rounds;++r) {bench(r,(r&1)!=0);bench(r,(r&1)==0);}
    return 0;
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
