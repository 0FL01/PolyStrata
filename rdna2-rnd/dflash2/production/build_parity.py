#!/usr/bin/env python3
"""Build a private diagnostic ELF from the native production Ninja build; do not time this ELF."""
import argparse
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one source anchor: {old}')
    return text.replace(old, new)


def main_probe(src):
    return '#include <fstream>\n' + once(src, '        const auto pick = [&](const float* logits) {', '''        std::ofstream tape;
        if (const char* path = std::getenv("DF2_VERIFY_TAPE")) {
            tape.exceptions(std::ios::badbit | std::ios::failbit);
            tape.open(path, std::ios::binary | std::ios::app);
        }
        const auto pick = [&](const float* logits) {
            if (tape.is_open()) tape.write(reinterpret_cast<const char*>(logits), engine.n_vocab()*sizeof(float));''')


def placement_probe(src):
    src = once(src, 'void Engine::run(const int32_t* tokens, int n) {', '''static void df2_boundary(int64_t at,
        const std::vector<Layer>& layers, const auto& uses, const auto& recency, int64_t kv_rows, int64_t swaps) {
    if (!std::getenv("DF2_VERIFY_TAPE")) return;
    uint64_t slots = 14695981039346656037ull, recent = slots, used = slots;
    const auto add = [](uint64_t& h, uint64_t v) {
        for (int i = 0; i < 8; ++i, v >>= 8) h = (h ^ (v & 255)) * 1099511628211ull;
    };
    uintptr_t base = UINTPTR_MAX;
    for (const auto& l : layers)
        for (const auto* p : l.slot) if (p) base = std::min(base, (uintptr_t) p);
    for (const auto& l : layers)
        for (const auto* p : l.slot) add(slots, p ? (uintptr_t)p - base : UINT64_MAX);
    for (float r : recency) add(recent, std::bit_cast<uint32_t>(r));
    for (const auto u : uses) add(used, (uint64_t) u);
    std::fprintf(stderr,"DF2_BOUNDARY %lld %016llx %016llx %016llx %lld %lld\\n",
                 (long long) at, (unsigned long long) slots, (unsigned long long) recent,
                 (unsigned long long) used, (long long) kv_rows, (long long) swaps);
}

void Engine::run(const int32_t* tokens, int n) {''')
    src = once(src, '    if (prepared_decode_ >= 0) {\n', '    const bool df2_unprepared = prepared_decode_ < 0;\n    if (prepared_decode_ >= 0) {\n')
    src = once(src, '        follow(exact_window_ ? 1 : n < kStageRows ? n : 0);\n    }\n    tok_', '''        follow(exact_window_ ? 1 : n < kStageRows ? n : 0);
    }
    if (df2_unprepared && (serial_decode_ || exact_window_))
        df2_boundary(pos_, layers_, use_, recent_, kv_rows_, stats_.swapped_experts);
    tok_''')
    src = once(src, '    decode_placement(pos_ + row);\n    for (size_t entry :', '''    decode_placement(pos_ + row);
    df2_boundary(pos_ + row, layers_, use_, recent_, kv_rows_, stats_.swapped_experts);
    for (size_t entry :''')
    src = once(src, '    if (row < verified_) return true;  // frozen layout or no swaps', '''    if (row < verified_) {
        while (counted_routes_ < row) commit_routes(counted_routes_++);
        df2_boundary(pos_ + row, layers_, use_, recent_, kv_rows_, stats_.swapped_experts);
        return true;
    }''')
    return src


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build-poly-hip')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(mode=0o700)
    src = ROOT / 'src/poly'
    commands = [shlex.split(s) for s in subprocess.check_output(['ninja', '-t', 'commands', 'strata-poly'],
                                                                cwd=a.build, text=True).splitlines()]
    link, = (s[2:-2] for s in commands if '--hip-link' in s and 'strata-poly' in s and s[:2] == [':', '&&'])
    for unit, transform in [('engine.cpp', placement_probe), ('main.cpp', main_probe)]:
        (out / unit).write_text(transform((src / unit).read_text()))
        args, = (s.copy() for s in commands if s[-2:] == ['-c', str(src / unit)])
        obj = out / (unit + '.o')
        oi = args.index('-o') + 1
        link[link.index(args[oi])] = str(obj)
        args[oi], args[-1] = str(obj), str(out / unit)
        for flag in ('-MT', '-MF'):
            if flag in args:
                j = args.index(flag)
                del args[j:j+2]
        if '-MD' in args:
            args.remove('-MD')
        subprocess.run(args + ['-I' + str(src)], cwd=a.build, check=True)
    for i, arg in enumerate(link):
        if arg.startswith('--dependency-file=') and link[i-1] == '-Xlinker':
            del link[i-1:i+1]
            break
    link[link.index('-o') + 1] = str(out / 'strata-poly-df2-parity')
    subprocess.run(link, cwd=a.build, check=True)


if __name__ == '__main__':
    main()
