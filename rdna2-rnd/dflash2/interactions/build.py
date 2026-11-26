#!/usr/bin/env python3
"""Relink the existing live DF2 overlay with private prompt input and SWA/placement counters."""
import argparse
import importlib.util
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENTRY = HERE.parent / 'live/build.py'

COUNTERS = r'''
namespace {
long long ix_staged, ix_staged_rows, ix_window_staged, ix_windows;

template <class Layers>
void ix_placement(const Layers& ls, const char* stage) {
    // Read full expert membership and relative slot offsets, independent of VRAM base address.
    uint64_t h = 14695981039346656037ull;
    size_t resident = 0;
    for (const auto& l : ls) {
        uintptr_t first = UINTPTR_MAX;
        for (const auto* p : l.slot) if (p) first = std::min(first, (uintptr_t) p);
        for (const auto* p : l.slot) {
            if (p) ++resident;
            uint64_t x = p ? (uintptr_t) p - first : UINT64_MAX;
            for (int i = 0; i < 8; ++i, x >>= 8) h = (h ^ (x & 255)) * 1099511628211ull;
        }
    }
    std::fprintf(stderr, "IX_PLACEMENT %s %zu %016llx\n", stage, resident, (unsigned long long) h);
}
}
'''

PROMPT = '''    if (const char* path = std::getenv("IX_TOKEN_FILE")) {
        std::ifstream f(path);
        std::string line, extra;
        if (!ids.empty() || want_serve || !std::getline(f, line) || std::getline(f, extra) ||
            !parse_ids(line.c_str(), ids)) throw std::runtime_error("invalid IX_TOKEN_FILE");
    }
'''


def build(out, release):
    spec = importlib.util.spec_from_file_location('df2_live_builder', ENTRY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    replace = m.replace

    def inject(source, old, new):
        if old == '        loaded = Engine::open(o);':
            anchor = '    if (o.model.empty() || (!want_serve && ids.empty()) || max_new < 1) {'
            source = replace(source, anchor, PROMPT + anchor)
        elif old == 'Inkling2::~Inkling2() {':
            source = replace(source, old, COUNTERS + old)
            new += '''
    ix_placement(layers_, "exit");
    std::fprintf(stderr, "IX_SWA ring=%d stage=%d borrow=%d staged_visits=%lld staged_rows=%lld window_staged=%lld windows=%lld\\n",
                 swa_plan_.ring_rows, swa_plan_.stage_rows, int(swa_plan_.borrow_input),
                 ix_staged, ix_staged_rows, ix_window_staged, ix_windows);
'''
            anchor = 'void Inkling2::run_visits(int n) {'
            source = replace(source, anchor, anchor + '''
    if (!pos_) ix_placement(layers_, "start");
    if (windowed_) ++ix_windows;
''')
            anchor = '    if (staged) copy_swa(live,'
            source = replace(source, anchor, '''    if (staged) { ++ix_staged; ix_staged_rows += n; ix_window_staged += windowed_; }
''' + anchor)
        return replace(source, old, new)

    m.replace = inject
    def run(cmd, **kwargs):
        if '--hip-link' in cmd:
            obj = fixed_arena_object(out, release, (ROOT/'src/poly/engine.cpp').read_text())
            i = next(i for i, x in enumerate(cmd) if x.endswith('/src/poly/engine.cpp.o'))
            cmd[i] = str(obj)
        subprocess.run(cmd, **kwargs)
    m.subprocess = SimpleNamespace(check_output=subprocess.check_output, run=run)
    out.mkdir(parents=True, exist_ok=False)
    m.build(release, out)


def once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError(f'incompatible source: {anchor}')
    return source.replace(anchor, replacement)


def fixed_arena_source(eng):
    anchor = '    size_t arena = std::min({all, free_b - reserve, arena_.base ? arena_.reserved - 256 : most});'
    return once(eng, anchor, anchor + '''
    if (const char* fixed = std::getenv("IX_ARENA_MIB")) {
        const size_t budget = size_t(std::stoul(fixed)) << 20;
        if (arena < budget) throw std::runtime_error("fixed expert arena unavailable");
        arena = budget;
    }
''')


def compile_one(out, release, cmds, name, body):
    source = out/name
    source.write_text(body)
    src = ROOT/'src/poly'/name
    c, = (s.copy() for s in cmds if s[-2:] == ['-c', str(src)])
    obj = out/(name+'.o')
    original = c[c.index('-o')+1]
    for flag in ('-MT', '-MF'):
        if flag in c:
            del c[c.index(flag):c.index(flag)+2]
    if '-MD' in c:
        c.remove('-MD')
    c[c.index('-o')+1], c[-1] = str(obj), str(source)
    subprocess.run(c+['-I'+str(src.parent)], cwd=release, check=True)
    return obj, original


def ninja_commands(release):
    return [shlex.split(s) for s in subprocess.check_output(['ninja','-t','commands','strata-poly'], cwd=release, text=True).splitlines()]


def fixed_arena_object(out, release, eng):
    obj, _ = compile_one(out, release, ninja_commands(release), 'engine.cpp', fixed_arena_source(eng))
    return obj


def serial(out, release, probe):
    """Read just the existing DFT1 instrumentation; use stock main, kernel and attention objects."""
    out.mkdir(parents=True, exist_ok=False)
    ink = (probe/'inkling2.cu').read_text()
    first = ink.index('    const char* ow_env = ')
    last = ink.index('    load_shape();', first)
    ink = ink[:first] + '    opt_.window = 1;\n' + ink[last:]
    anchor = 'Inkling2::~Inkling2() {'
    ink = once(ink, anchor, COUNTERS + '\nstatic long long ix_sparse, ix_mixed;\n' + anchor + '''
    ix_placement(layers_, "exit");
    std::fprintf(stderr, "IX_POLICY k256=%lld v4=%lld\\n", ix_sparse, ix_mixed);
''')
    anchor = 'void Inkling2::run_visits(int n) {'
    ink = once(ink, anchor, anchor + '''
    static bool ix_first = true;
    if (ix_first) { ix_placement(layers_, "start"); ix_first = false; }
''')
    anchor = 'if (serial_decode_ && n == 1 && !windowed_ && inkling::attend_k256(a, mixed, c.index, chunks_, cs(st_), attq_))\n            return;'
    ink = once(ink, anchor, anchor.removesuffix('return;') + '{ ++ix_sparse; return; }')
    anchor = '        inkling::attend_v4(a, mixed, chunks_, cs(st_), q8_out);'
    ink = once(ink, anchor, '        ++ix_mixed;\n' + anchor)
    eng = fixed_arena_source((probe/'engine.cpp').read_text())
    main = '#include <fstream>\n' + (ROOT/'src/poly/main.cpp').read_text()
    anchor = '        r.decode_ms = ms_since(t0);\n        return r;'
    main = once(main, anchor, anchor.replace('        return r;', '''        if (const char* path = std::getenv("IX_STATE")) {
            std::vector<uint8_t> b(engine.state_bytes());
            engine.save_state(b.data());
            std::ofstream f;
            f.exceptions(std::ios::badbit | std::ios::failbit);
            f.open(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(b.data()), b.size());
        }
        return r;'''))
    units = {'inkling2.cu': ink, 'engine.cpp': eng, 'main.cpp': main}
    (out/'inkling2.hpp').write_bytes((probe/'inkling2.hpp').read_bytes())
    cmds = ninja_commands(release)
    link, = (c[2:-2] for c in cmds if '--hip-link' in c and 'strata-poly' in c and c[:2] == [':','&&'] and c[-2:] == ['&&',':'])
    if '--offload-arch=gfx1031' not in link:
        raise ValueError('expected gfx1031 Release HIP objects')
    for name, body in units.items():
        obj, original = compile_one(out, release, cmds, name, body)
        link[link.index(original)] = str(obj)
    for i, word in enumerate(link):
        if word.startswith('--dependency-file=') and link[i-1] == '-Xlinker':
            del link[i-1:i+1]
            break
    binary = out/'strata-poly-w1-dft1'
    link[link.index('-o')+1] = str(binary)
    subprocess.run(link, cwd=release, check=True)
    print(binary)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--release', type=Path, default=ROOT / 'build-poly-hip')
    p.add_argument('--serial-probe-source', type=Path, help='existing probe-bin generated sources; read only')
    a = p.parse_args()
    if a.serial_probe_source:
        serial(a.out.resolve(), a.release.resolve(), a.serial_probe_source.resolve())
    else:
        build(a.out.resolve(), a.release.resolve())
