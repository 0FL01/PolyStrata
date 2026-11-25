"""Relink scratch engine.cpp with an inline CPU-only read during the GPU router wait.

Run from the repo root, with the gfx1031 Release Ninja build already complete.
No GPU code, project source or build output is modified. Single-sequence W1 only.
"""
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True,exist_ok=True)
    root = Path.cwd(); build = root/'build-poly-hip'; src=root/'src/poly/engine.cpp'
    leaf=Path(__file__).with_name('router_sidecar.cpp').read_text()
    prefix=leaf[:leaf.index('int main(')]
    load=leaf[leaf.index('    for (int l=0;l<L;'):leaf.index('    std::ifstream in(argv[2]')]
    load=load.replace('int l=0;l<L','int l=6;l<18')
    # Keep only the tiny biases and packed weights once the load-time mapping is gone.
    load=load.replace('a.q.resize(E*W);','std::copy_n(a.b,E,bias[l].begin()); a.b=bias[l].data(); a.q.resize(E*W);')
    (out/'hint.cpp').write_text(prefix+'''static std::array<Net,L> nets;
static std::array<std::array<float,E>,L> bias;
void recon_load(const char* model) { strata::GgufFile gg(model);
'''+load+'''}
void recon_predict(int l,const float* x,int* out) {
    auto picks=predict(nets[l],x,true); std::copy(picks.begin(),picks.end(),out);
}
''')
    s=src.read_text().replace('namespace strata::poly {','extern void recon_load(const char*);\nextern void recon_predict(int,const float*,int*);\nnamespace strata::poly {',1)
    start=s.index('    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");',s.index('bool Engine::moe(Layer&'))
    s=s[:start]+'''    // Scratch protocol: a new process per model, an authoritative router every visit.
    // The predictor has no pointers into mutable GPU state. Snapshot the previous
    // mapped host row after its consumer and before the next head overwrites it.
    struct Probe {
        int v=-1, mode=0; bool valid=false; alignas(32) float x[2304];
        uint64_t scored=0, issued=0, hits=0, bytes=0;
        ~Probe(){if(mode) std::fprintf(stderr,"RECON scored=%llu issued=%llu cpu_hits=%llu bytes=%llu\\n",
            (unsigned long long)scored,(unsigned long long)issued,(unsigned long long)hits,(unsigned long long)bytes);}
    };
    static thread_local Probe h;
    const bool screen=serial_decode_ && n==1 && n_expert_==128 && n_embd_==2304 && looped_ && !draft_;
    bool on=false; int probe_id=-1;
    if(screen) {
        h.v=il==0 ? 0 : h.v+1;
        if((h.v<6 ? h.v : h.v<42 ? 6+(h.v-6)%12 : h.v-24)!=il) std::abort();
        static thread_local const bool init=[&](){
            const char* v=std::getenv("RECON_WAIT"); h.mode=v?std::atoi(v):0;
            recon_load(opt_.model.c_str()); return true;
        }(); (void)init;
        // Modes 1/2: opposite eight-position blocks; mode 3: always on.
        on=h.mode==3 || (h.mode>0 && (((pos_/8)&1)==h.mode-1));
        const auto waiting=[&](){const auto rc=cudaEventQuery((cudaEvent_t)ev_);
            if(rc==cudaErrorNotReady) return true;
            ck(rc,"RECON router query"); return false;};
        if(on && h.valid && h.v>=18 && h.v<42 && waiting()) {
            int top[4]; recon_predict(il,h.x,top); ++h.scored;
            for(int e:top) if(!L.slot[e]) {probe_id=e;break;}
            if(probe_id>=0) {
                const int e=probe_id; ++h.issued;
                const auto touch=[&](const uint8_t* p,size_t n) {
                    for(size_t i=0;i<n;i+=262144){ if(!waiting()) return false;
                        const size_t end=std::min(n,i+262144);
                        for(size_t j=i;j<end;j+=64) (void)*static_cast<const volatile uint8_t*>(p+j);
                        h.bytes+=end-i;
                    }return true;
                };
                if(touch(L.exp_gate+e*L.gu_stride,L.gu_bytes) && touch(L.exp_up+e*L.u_stride,L.u_bytes))
                    touch(L.exp_down+e*L.d_bytes,L.d_bytes);
            }
        }
    }
'''+s[start:]
    pos=s.index('        if (pool_) pool_take(L, il, host, n_host);',start)
    s=s[:pos]+'''        if(probe_id>=0 && std::find(host_e,host_e+n_host,probe_id)!=host_e+n_host) ++h.hits;
'''+s[pos:]
    s=s.replace('    return any_host;', '''    h.valid=screen && on && h.v>=17 && h.v<41 && any_host;
    if(h.valid) std::memcpy(h.x,h_x_,sizeof(h.x));
    return any_host;''',1)
    (out/'engine.cpp').write_text(s)
    lines=subprocess.check_output(['ninja','-t','commands','strata-poly'],cwd=build,text=True).splitlines()
    cc,= [shlex.split(l) for l in lines if l.endswith(' -c '+str(src))]
    link,=[shlex.split(l)[2:-2] for l in lines if ' -o strata-poly ' in l and ' --hip-link ' in l]
    k=next((i for i,a in enumerate(link) if a.startswith('--dependency-file=')),None)
    if k is not None:
        assert link[k-1]=='-Xlinker'; del link[k-1:k+1]
    i=next(i for i,a in enumerate(link) if a.endswith('/src/poly/engine.cpp.o'))
    link[i:i+1]=[str(out/'engine.o'),str(out/'hint.o')]; link[link.index('-o')+1]=str(out/'strata-poly')
    for cmd in (['/usr/bin/c++','-O3','-DNDEBUG','-std=c++20','-fPIE','-mavx2','-mfma','-I'+str(root/'include'),'-c',str(out/'hint.cpp'),'-o',str(out/'hint.o')],
                cc[:cc.index('-MD')]+['-I'+str(src.parent),'-c',str(out/'engine.cpp'),'-o',str(out/'engine.o')],link):
        subprocess.run(cmd,cwd=build,check=True)


if __name__=='__main__':
    main()
