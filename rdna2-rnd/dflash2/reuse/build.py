#!/usr/bin/env python3
"""Build a CPU-only comparison against the existing HIP Release's CPU object."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def instrument(src):
    def hook(a, b):
        nonlocal src
        if src.count(a) != 1:
            raise ValueError(f"upstream anchor changed: {a}")
        src = src.replace(a, b)

    hook('#include "cpu_experts.hpp"', '#include "cpu_experts.hpp"\n#include "q4_multi.hpp"')
    hook('void cpu_experts(Workers&', 'void cpu_experts_reuse(Workers&')
    hook('    float* hp = h.data();', '''    const bool group_gu = gu_type == GGML_TYPE_Q4_0 && u_type == GGML_TYPE_Q4_0 && !own_act;
    const bool group_d = d_type == GGML_TYPE_Q4_0 || d_type == GGML_TYPE_Q4_1;
    float* hp = h.data();''')
    hook('''            for (int i = i0; i < i1; ++i) {
                const int j = byp[i];
                const uint8_t* a = actp + (size_t) experts[j].row * a_bytes;
                const uint8_t* au = actup + (size_t) experts[j].row * au_step;
                float g = 0.f, u = 0.f;
                tg->vec_dot(n_embd, &g, 0, gate, 0, a, 0, 1);
                tu->vec_dot(n_embd, &u, 0, up, 0, au, 0, 1);
                g *= e.gate_scale;''', '''            for (int i = i0; i < i1;) {
                const int k = group_gu && i1 - i > 1 ? reuse::chunk(i1 - i) : 1;
                float gates[4]{}, ups[4]{};
                if (k > 1) {
                    const uint8_t* a[4];
                    for (int j = 0; j < k; ++j) a[j] = actp + (size_t) experts[byp[i + j]].row * a_bytes;
                    reuse::dot<false>(n_embd, gate, a, k, gates);
                    reuse::dot<false>(n_embd, up, a, k, ups);
                } else {
                    const int row = experts[byp[i]].row;
                    tg->vec_dot(n_embd, gates, 0, gate, 0, actp + (size_t) row * a_bytes, 0, 1);
                    tu->vec_dot(n_embd, ups, 0, up, 0, actup + (size_t) row * au_step, 0, 1);
                }
                for (int j0 = 0; j0 < k; ++j0) {
                const int j = byp[i++];
                float g = gates[j0], u = ups[j0];
                g *= e.gate_scale;''')
    hook('''                hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
            }''', '''                hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
                }
            }''')
    # Indent the copied per-use activation block without changing its floating-point expressions.
    b = src.index('                const int j = byp[i++];')
    e = src.index('                }\n            }', src.index('std::exp(-g)) * u;', b))
    src = src[:b] + ''.join('    ' + l for l in src[b:e].splitlines(keepends=True)) + src[e:]
    hook('''                for (int i = runp[u]; i < runp[u + 1]; ++i)
                    td->vec_dot(n_ff, s + byp[i], 0, down, 0, hqp + (size_t) byp[i] * h_bytes, 0, 1);''',
         '''                const int end = runp[u + 1];
                for (int i = runp[u]; i < end;) {
                    if (!group_d || end - i == 1) {
                        const int j = byp[i++];
                        td->vec_dot(n_ff, s + j, 0, down, 0, hqp + (size_t) j * h_bytes, 0, 1);
                    } else {
                        const int k = reuse::chunk(end - i);
                        const uint8_t* a[4];
                        float values[4];
                        for (int j = 0; j < k; ++j) a[j] = hqp + (size_t) byp[i + j] * h_bytes;
                        if (d_type == GGML_TYPE_Q4_1) reuse::dot<true>(n_ff, down, a, k, values);
                        else reuse::dot<false>(n_ff, down, a, k, values);
                        for (int j = 0; j < k; ++j) s[byp[i++]] = values[j];
                    }
                }''')
    # Use a separate TU for the candidate, but the exact same Workers implementation at link time.
    b = src.index('Workers::Workers(')
    e = src.index('void cpu_experts_reuse(')
    return (src[:b] + src[e:]).replace('thread_local bool in_task = false;   // this thread is running a task of a batch\n', '')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--build', type=Path, default=ROOT/'build-poly-hip')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = args.build.resolve(strict=True)
    upstream = ROOT/'src/poly/cpu_experts.cpp'
    stock = build/'CMakeFiles/strata-poly.dir/src/poly/cpu_experts.cpp.o'
    libs = [build/f'ggml/src/libggml-{lib}.a' for lib in ('cpu', 'base')]
    generated = out/'cpu_experts_reuse.cpp'
    generated.write_text(instrument(upstream.read_text()))
    includes = [ROOT/'src/poly', ROOT/'include', ROOT/'third_party/llama.cpp/ggml/include', HERE]
    cc = ['c++', '-O3', '-DNDEBUG', '-std=c++20', '-fPIE', '-pthread', '-Wall', '-Wextra',
          *[f'-I{p}' for p in includes]]
    with (out/'build.log').open('w') as log:
        cmd = [*cc, '-mclflushopt', str(HERE/'replay.cpp'), str(generated), str(stock), *map(str, libs), '-ldl',
               '-o', str(out/'replay')]
        print(shlex.join(cmd), file=log, flush=True)
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    paths = [upstream, stock, *libs, generated, HERE/'q4_multi.hpp', HERE/'replay.cpp', out/'replay']
    with (out/'build.sha256').open('w') as f:
        for path in paths:
            with path.open('rb') as src:
                print(hashlib.file_digest(src, 'sha256').hexdigest(), path, file=f)


if __name__ == '__main__':
    main()
