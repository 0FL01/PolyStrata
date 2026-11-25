// Included by the scratch build of prefetch/bench.cpp only. No GPU linkage.
struct Frame { int64_t p; int32_t v,l,n,g,d,r,id[8]; float w[8]; Vec x, sum; };
static_assert(sizeof(Frame)==18528);
using RefMap = std::map<std::pair<int64_t,int>,Frame>;
RefMap read_frames(const char* path) {
    require(path != nullptr,"set PSCPU_CAPTURE");
    std::ifstream f(path,std::ios::binary); char magic[8]; uint32_t h[4];
    require(bool(f.read(magic,8).read((char*)h,16)) && !memcmp(magic,"PSCPU01\n",8) && h[0] && h[0]<=48*h[3] && h[1]==width,"invalid CPU capture");
    RefMap result;
    for (uint32_t i=0;i<h[0];++i) {
        Frame r; require(bool(f.read((char*)&r,sizeof(r))),"short CPU capture");
        require(result.emplace(std::make_pair(r.p,r.v),r).second,"duplicate capture row");
    }
    require(f.peek()==EOF,"extra capture rows"); return result;
}
void use_capture(Case& c,const RefMap& refs,bool check) {
    auto apply=[&](int visit,Branch& branch,Vec& x,const Vec& value) {
        auto it=refs.find({c.pos,visit}); require((it==refs.end())==(branch.n==0),"capture missing/extra branch");
        if (!branch.n) return;
        const auto& r=it->second;
        require(r.n==branch.n && r.l==layer(visit),"capture layer or count mismatch");
        if(check) parity(c,value,r.sum,"captured sum"); else x=r.x;
    };
    apply(c.visit-1,c.prev,c.xp,c.refp); apply(c.visit,c.target,c.x,c.ref);
}
