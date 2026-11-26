#!/usr/bin/env python3
"""Link the one-shot target + DFlash2 research binary from an existing Release HIP Ninja build.

No source/header or object in the production build is replaced. The two overridden
translation units use the identical class layout and flags as the unmodified ones.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace(src, anchor, new):
    if src.count(anchor) != 1:
        raise ValueError(f'source no longer matches: {anchor!r}')
    return src.replace(anchor, new)


def build(b, out, *, extra_overlay=None):
    out.mkdir(parents=True, exist_ok=True)
    src = ROOT / 'src/poly'
    cpp = '#include "live.hpp"\n#include <fstream>\n' + (src / 'main.cpp').read_text()
    cpp = replace(cpp, '        loaded = Engine::open(o);', '''        if (want_serve || want_disk_sessions || o.batch || check ||
            !follow.empty() || !images.empty() || o.window < 1 || o.window > 8 || lookup != 0)
            throw std::runtime_error("live probe: single text prompt only; use --lookup 0 --window 1..8, no --follow");
        loaded = Engine::open(o);''')
    # Only swap the decode loop: keep the actual target sampler, history, prompt read and timing.
    begin = cpp.index('        if (max_new > 0) {')
    end = cpp.index('        return r;', cpp.index('        r.decode_ms = ms_since(t0);', begin))
    cpp = cpp[:begin] + '        #include "generate.inc"\n' + cpp[end + len('        return r;'):]
    cpp = replace(cpp, '            rc = once(ss, ids, pictures, image_pad, max_new, s, top_probs);', '''            if (const char* snap = std::getenv("DF2_WARM_TARGET")) {
                // This imports a target, NOT a draft state. Require a fresh complete SWA
                // prefill before proposing. Check engine.pos() as disk.run also reports
                // recoverable errors on stdout and returns true in that case.
                if (engine.uses_dspark() || ids.size() < 2049)
                    throw std::runtime_error("target restore needs target-only snapshot + >=2049 fresh IDs");
                strata::poly::DiskSessions disk(engine,true,(uint64_t)ss.flags.min_free_mib << 20);
                if (!disk.run(false,snap,ss.kept) || engine.pos()<=0 || (int64_t)ss.kept.seq().size()!=engine.pos())
                    throw std::runtime_error("failed to restore target prefix");
                if (engine.pos()+(int64_t)ids.size()+max_new>engine.max_context())
                    throw std::runtime_error("restored prefix exceeds context");
                std::fprintf(stderr,"DF2_RESTORED %lld fresh=%zu\\n",(long long)engine.pos(),ids.size());
                ids.insert(ids.begin(),ss.kept.seq().begin(),ss.kept.seq().end());
            }
            rc = once(ss, ids, pictures, image_pad, max_new, s, top_probs);''')

    hip = '#include "live.hpp"\n' + (src / 'inkling2.cu').read_text()
    hip = replace(hip, '    opt_.window = 1;', '''    if (!decode_ || k256_ || v4_ || opt_.prompt_batch < window)
        throw std::runtime_error("live probe: dense global target, STRATA_INKLING_DECODE=1, batch >= window");
    opt_.window = window;
    cols_kernels_ = window > 1;''')
    hip = replace(hip, '    if (!draft_) return;\n    ds_taps_ = fbuf(3 * width);', '''    ds_taps_ = fbuf(4 * width);
    if (scratch_) { // real allocation pass, BEFORE fill_expert_cache prices the remaining VRAM
        if (output_.type != GGML_TYPE_Q6_K) throw std::runtime_error("live probe requires the Tideglass Q6_K head");
        ck(cudaStreamSynchronize(cs(st_)), "share target head");
        df_live::open(opt_.model.c_str(), output_.p);
    }
    if (!draft_) return;''')
    hip = replace(hip, "        if (draft_)   // the drafter's taps: the residual entering visits 5, 23 and 41 (drafter_kept())", '        // DFlash2 consumes the same entering visits as DSpark.')
    hip = replace(hip, '        layer(physical(visit_), n);\n    }\n}', '''        layer(physical(visit_), n);
    }
    const size_t plane = (size_t) rows_ * kWidth;
    output_rows(0, n, ds_taps_ + 3 * plane);
    ck(cudaStreamSynchronize(cs(st_)), "DFlash2 committed taps");
    df_live::pending(pos_, n, ds_taps_, plane);
    if (!windowed_) df_live::commit(n); // speculative rows wait for the sampler's decision
}''')
    hip = replace(hip, 'Inkling2::~Inkling2() {', 'Inkling2::~Inkling2() {\n    df_live::close();')

    sources = {'main.cpp': cpp, 'inkling2.cu': hip}
    if extra_overlay is not None:
        for name, contents in extra_overlay(src, sources.copy()).items():
            if name == 'live.cpp' or Path(name).name != name or not (src / name).is_file():
                raise ValueError(f'invalid extra overlay unit: {name}')
            sources[name] = contents
    for name, contents in sources.items():
        (out / name).write_text(contents)
    units = [(name, name) for name in sources] + [('live.cpp', 'main.cpp')]
    cmds = [shlex.split(s) for s in subprocess.check_output(['ninja','-t','commands','strata-poly'], cwd=b, text=True).splitlines()]
    link, = (c[2:-2] for c in cmds if '--hip-link' in c and 'strata-poly' in c and c[:2] == [':','&&'] and c[-2:] == ['&&',':'])
    compile_units = []
    for name, original in units:
        c, = (s.copy() for s in cmds if s[-2:] == ['-c', str(src / original)])
        obj = out / (name + '.o')
        if name != 'live.cpp':
            link[link.index(c[c.index('-o')+1])] = str(obj)
        else:
            link.insert(link.index('-o'), str(obj))
        for flag in ['-MT','-MF']:
            if flag in c: del c[c.index(flag):c.index(flag)+2]
        if '-MD' in c: c.remove('-MD')
        c[c.index('-o')+1] = str(obj)
        c[-1] = str((HERE if name == 'live.cpp' else out) / name)
        compile_units.append((name, c + ['-I'+str(src), '-I'+str(HERE)]))
    if extra_overlay is None:
        for _, command in compile_units:
            subprocess.run(command, cwd=b, check=True)
    else:
        def compile_unit(unit):
            name, command = unit
            with (out / (name + '.build.log')).open('w') as log:
                subprocess.run(command, cwd=b, check=True, stdout=log, stderr=subprocess.STDOUT)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(compile_unit, compile_units))
    for i,x in enumerate(link):
        if x.startswith('--dependency-file=') and link[i-1] == '-Xlinker':
            del link[i-1:i+1]
            break
    exe = out / 'strata-poly-dflash2-live'
    link[link.index('-o')+1] = str(exe)
    subprocess.run(link + ['-ldl'], cwd=b, check=True)
    print(exe, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, default=ROOT/'build-poly-hip')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    build(a.build.resolve(), a.out.resolve())
