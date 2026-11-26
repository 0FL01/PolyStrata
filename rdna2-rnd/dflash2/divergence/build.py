#!/usr/bin/env python3
"""Link a private diagnostic program against the existing Release HIP build, without touching it."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R = Path.home() / '.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon/divergence'


def replace(s, anchor, new):
    if s.count(anchor) != 1:
        raise ValueError(f'expected one source anchor: {anchor!r}; got {s.count(anchor)}')
    return s.replace(anchor, new, 1)


def insert(s, anchor, new):
    return replace(s, anchor, new + anchor)


def after(s, anchor, new):
    return replace(s, anchor, anchor + new)


def main_cpp(s):
    s = '#include "trace.hpp"\n#include <fstream>\n' + s
    return insert(s, '        if (max_new > 0) {', '''        if (const char* path = std::getenv("DIV_TRACE")) {
            namespace d = strata::poly::divtrace;
            const char* width = std::getenv("DIV_WIDTH");
            const int w = width ? std::atoi(width) : 0;
            if (w < 1 || w > 8 || max_new != 9 || engine.max_window() != 8 || engine.pos() != n)
                throw std::runtime_error("diagnostic width / prompt / rows contract");
            d::Scope trace(path, engine.pos());
            const auto cap = [&](int64_t pos, const float* logits) {
                d::host("softcapped_logits", pos, 49, logits, engine.n_vocab() * sizeof(float), 1);
                return (int32_t) strata::poly::most_likely(logits, engine.n_vocab());
            };
            const int32_t first = cap(n - 1, engine.logits());
            std::vector<int32_t> force;
            if (const char* f = std::getenv("DIV_FORCE")) {
                std::ifstream file(f);
                for (int t; file >> t;) force.push_back(t);
                if (force.size() != 8 || force[0] != first)
                    throw std::runtime_error("diagnostic fixed ID file / prefix mismatch");
            } else if (w != 1) throw std::runtime_error("diagnostic window requires W1 IDs");
            const auto before = engine.stats();
            show(first);
            int32_t tok = first;
            if (w == 1) {
                for (int i = 0; i < 8; ++i) {
                    if (!force.empty()) tok = force[i];
                    engine.eval_decode(tok);
                    kept.read(&tok, 1);
                    tok = cap(n + i, engine.logits());
                    show(tok);
                }
            } else {
                engine.window(force.data(), w);
                for (int i = 0; i < w; ++i) show(cap(n + i, engine.logits_of(i)));
                engine.keep(w);  // no rejected rows; all trace bytes precede keep()
                kept.read(force.data(), w);
            }
            const auto end = engine.stats();
            std::fprintf(stderr, "DIV_RESULT width=%d rows=%d pos=%lld gpu_experts=%lld cpu_experts=%lld\\n",
                         w, w == 1 ? 8 : w, (long long) n, (long long) (end.gpu_experts - before.gpu_experts),
                         (long long) (end.cpu_experts - before.cpu_experts));
            r.decode_ms = ms_since(t0);
            return r;
        }
''')


def inkling2(s):
    s = after(s, '#include "runtime.hpp"\n', '#define DIVTRACE_DEVICE_CAPTURE\n#include "trace.hpp"\n')
    s = after(s, '    opt_.window = 1;\n', '''    if (std::getenv("DIV_TRACE") && !drafter) {
        opt_.window = 8;   // same capacity, scratch and expert slots in W1 and every verifier leg
        cols_kernels_ = true;
    }
''')
    # Tensor captures go after launches, before any in-place transforms or scratch reuse.
    s = after(s, 'void Inkling2::run_visits(int n) {\n', '''    if (divtrace::tape) {
        divtrace::device("embedding", pos_, -1, r_, kWidth * sizeof(float), n, st_);
        divtrace::host("input_ids", pos_, -1, tok_, sizeof(int32_t), n, 2);
        if (pos_ == divtrace::base) {
            // Hash the complete placement by expert ID and within-layer relative slot offsets, not GPU VA.
            uint64_t hash = 14695981039346656037ull;
            size_t resident = 0;
            for (const auto& l : layers_) {
                uintptr_t first = UINTPTR_MAX;
                for (const auto* ptr : l.slot) if (ptr) first = std::min(first, (uintptr_t) ptr);
                for (const auto* ptr : l.slot) {
                    if (ptr) ++resident;
                    uint64_t off = ptr ? (uintptr_t) ptr - first : UINT64_MAX;
                    for (int i = 0; i < 8; ++i, off >>= 8) hash = (hash ^ (off & 255)) * 1099511628211ull;
                }
            }
            std::fprintf(stderr, "DIV_PLACEMENT slots=%zu layout=%016llx swaps=%d mode=%s n=%d\\n", resident,
                         (unsigned long long) hash, n_swap_, serial_decode_ ? "decode" : "window", n);
        }
    }
''')
    s = after(s, '    if (first < n) norm_text(first, n - first);\n', '''    divtrace::device("token_norm", pos_, -1, r_, kWidth * sizeof(float), n, st_);
''')
    s = after(s, '        layer(physical(visit_), n);\n', '''        divtrace::device("visit_out", pos_, visit_, r_, kWidth * sizeof(float), n, st_);
''')
    s = after(s, 'void Inkling2::layer_head(Layer& L, int il, int n) {\n', '''    const auto stage = [&](const char* name, const void* data, int len, int kind = 0) {
        divtrace::device(name, pos_, visit_, data, kind ? len : len * sizeof(float), n, st_, kind);
    };
    divtrace::visit = visit_;
    stage("visit_in", r_, kWidth);
    if (divtrace::tape && pos_ == divtrace::base)
        divtrace::device("incoming_conv", pos_, visit_, state_ + (size_t) visit_ * kConvFloats,
                         kConvFloats * sizeof(float), 1, st_);
''')
    s = insert(s, '    if (fused(n) && n == 1) project(b,', '''    stage("attn_input_q", xq, kWidth);
    stage("attn_input_k", xk, kWidth);
    stage("attn_input_v", xv, kWidth);
    const int row8 = (int) kernels::native_q8_1_bytes(kWidth);
    if (shared) stage("attn_input_q_q8", shared, row8, 1);
    if (streams == 1 || kq) stage("attn_input_k_q8", streams == 1 ? shared : kq, row8, 1);
    if (streams == 1 || vq) stage("attn_input_v_q8", streams == 1 ? shared : vq, row8, 1);
''')
    s = after(s, '    lin(b.sink, xq, sink_, n);\n', '''    stage("proj_q", q_, kQuery);
    stage("proj_k", k_, kKey);
    stage("proj_v", v_, kKey);
    stage("proj_rel", rel_, kHeads * kRelative);
    stage("proj_gate", gate_, kQuery);
    stage("proj_sink", sink_, kHeads);
''')
    s = after(s, '            lin(b.vproj, xv, taps2_, n);\n', '''            stage("proj_k_dynamic_conv", taps_, kKv * kTaps);
            stage("proj_v_dynamic_conv", taps2_, kKv * kTaps);
''')
    s = insert(s, '    dbg("Qcur", visit_', '''    stage("head_q", q_, kQuery);
    stage("head_k", k_, kKey);
    stage("head_v", v_, kKey);
    stage("head_rel", rel_, kHeads * kRelative);
''')
    s = after(s, '    attention(b, n);\n', '''    stage("attend_f32", att_, kQuery);
    if (fused(n)) stage("attend_q8", attq_, (int) kernels::native_q8_1_bytes(kQuery), 1);
''')
    s = after(s, '    lin(b.o, att_, ao_, n, fused(n) ? attq_ : nullptr);   // attend_decode() quantized it\n', '''    stage("attention_out_projection", ao_, kWidth);
''')
    s = insert(s, '    depth_mix(fp(b.fquery), cur_, 0, n);\n', '''    stage("attention_residual", r_, kWidth);
    if (il % kBlock == 0) stage("depth_saved", depth_[depth_n_ - 1], kWidth);
''')
    s = insert(s, '    kernels::native_add_rms_norm_q8_1(cur_, nullptr, fp(L.ffn_norm)', '''    stage("ffn_control", cur_, kWidth);
''')
    s = after(s, '    if (!batched(L, n)) moe_router(L, n);\n', '''    stage("ffn_norm", xn_, kWidth);
    stage("ffn_norm_q8", xq_, row8, 1);
    if (L.moe && !batched(L, n)) stage("router_scores", rl_, n_expert_);
''')
    s = insert(s, '    inkling::Attend a;\n    a.q = q_;', '''    if (divtrace::tape) {
        const char* names[]{"cache_k", "cache_k_scale", "cache_v", "cache_v_scale"};
        for (int p = 0; p < 4; ++p) {
            const auto& plane = c.planes[p];
            for (int r = 0; plane.stride && r < n; ++r) {
                const int wrap = hybrid && p >= 2 ? hot_rows_ : ring;
                const int64_t at = wrap ? (pos + r) % wrap : pos + r;
                divtrace::device(names[p], pos_ + r, visit_, plane.data + at * plane.stride,
                                 (int) plane.stride, 1, st_, 1);
            }
        }
    }
''')
    s = after(s, 'void Inkling2::layer_tail(Layer&, int il, int n) {\n', '''    if (divtrace::tape) {
        divtrace::host("moe_cpu_sum", pos_, visit_, h_cpu_, kWidth * sizeof(float), n);
        if (!tail_record_) divtrace::device("moe_gpu_sum", pos_, visit_, fo_, kWidth * sizeof(float), n, st_);
    }
''')
    return s


def engine(s):
    s = after(s, '#include "runtime.hpp"\n', '#define DIVTRACE_DEVICE_CAPTURE\n#include "trace.hpp"\n')
    s = after(s, 'namespace strata::poly {\n', '''void divtrace_router_rows(const float* w, const float* x, float* y, int n_in, int n_out, int n,
                          void* stream);
''')
    s = after(s, 'void Engine::router_scores(const Layer& L, int n) {\n', '''    if (divtrace::tape && divtrace::enabled("DIV_FIX_ROUTER") && looped_ && n > 1 && n <= 8 &&
        L.gate_inp.type == kTypeF32 && L.gate_inp.scale == 1.f && wave_gemv_) {
        route_dual_done_ = false;
        divtrace_router_rows(fp(L.gate_inp), xn_, rl_, L.gate_inp.n_in, L.gate_inp.n_out, n, st_);
        return;
    }
''')
    s = replace(s, 'const bool replay = opt_.graphs && n == 1 && want_logits && !opt_.dump;',
                'const bool replay = opt_.graphs && n == 1 && want_logits && !opt_.dump && !divtrace::tape;')
    s = insert(s, '            float w_all = 0.f;\n', '''            if (divtrace::tape) {
                int32_t place[kMaxUsed];
                for (int k = 0; k < n_used_; ++k) place[k] = L.slot[(size_t) ids[k]] != nullptr;
                divtrace::host("route_ids", pos_ + at + t, divtrace::visit, ids, n_used_ * sizeof(int), 1, 2);
                divtrace::host("route_weights", pos_ + at + t, divtrace::visit, w, n_used_ * sizeof(float), 1);
                divtrace::host("route_on_gpu", pos_ + at + t, divtrace::visit, place, n_used_ * sizeof(int32_t), 1, 2);
            }
''')
    s = after(s, '            output_rows(n - m, m, h);\n', '''            divtrace::device("output_norm", pos_ + n - m, 48, h, n_embd_ * sizeof(float), m, st_);
''')
    s = after(s, '    output_rows(0, t, h);\n', '''    divtrace::device("output_norm", pos_, 48, h, n_embd_ * sizeof(float), t, st_);
''')
    return s


def kernels(s):
    return insert(s, 'void gemv_f32_waves(const float* w,', '''// The diagnostic verifier retains the W1 router's reduction order in a single batched launch.
void divtrace_router_rows(const float* w, const float* x, float* y, int n_in, int n_out, int n, void* stream) {
    if (n_in != 2304 || n_out != 128 || (reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(x)) % 16)
        throw std::logic_error("unexpected Inkling diagnostic router shape");
    gemv_rows_kernel<<<dim3(n_out, n), 256, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, n_out);
}

''')


def attend(s):
    s = after(s, '#include "inkling_kv_policy.hpp"\n', '#include "trace.hpp"\n')
    anchor = 'void attend_decode(const Attend& a, float* scratch, cudaStream_t st, void* q8, bool group_rows) {\n'
    return after(s, anchor, '''    if (a.n > 1 && !a.window && a.k.scales && divtrace::tape && divtrace::enabled("DIV_FIX_GLOBAL_ATTN")) {
        // Exactness control: reproduce each query's W1 chunk boundaries and combine order. Not a speed candidate.
        for (int r = 0; r < a.n; ++r) {
            Attend row = a;
            row.pos += r;
            row.n = 1;
            row.q += r * kQuery;
            row.rel += r * kHeads * kRelative;
            row.sink += r * kHeads;
            row.gate += r * kQuery;
            row.out += r * kQuery;
            void* row_q8 = q8 ? static_cast<Q81*>(q8) + r * kQuery / 32 : nullptr;
            attend_decode(row, scratch, st, row_q8, false);
        }
        return;
    }
''')


def build(folder):
    os.umask(0o077)
    out = R / 'bin'
    out.mkdir(parents=True, exist_ok=True)
    subprocesses = {}
    provenance = {}
    lines = sp.check_output(['ninja', '-t', 'commands', 'strata-poly'], cwd=folder, text=True).splitlines()
    link, = (shlex.split(line) for line in lines if ' -o strata-poly ' in line and ' --hip-link ' in line)
    assert link[:2] == [':', '&&'] and link[-2:] == ['&&', ':']
    link = link[2:-2]
    changed = {'main.cpp': main_cpp, 'engine.cpp': engine, 'inkling2.cu': inkling2,
               'kernels.cu': kernels, 'inkling_attend.cu': attend}
    for name, transform in changed.items():
        path = REPO / 'src/poly' / name
        text = path.read_text()
        generated = out / name
        generated.write_text(transform(text))
        provenance[name] = hashlib.sha256(text.encode()).hexdigest()
        cmd, = (shlex.split(line) for line in lines if line.endswith(' -c ' + str(path)))
        assert cmd[-2] == '-c'
        cmd[-1] = str(generated)
        old_obj = cmd[cmd.index('-o') + 1]
        obj = out / (name + '.o')
        cmd[cmd.index('-MF') + 1] = str(out / (name + '.d'))
        cmd[cmd.index('-o') + 1] = str(obj)
        cmd.extend(['-I' + str(HERE), '-I' + str(path.parent)])
        subprocesses[name] = cmd
        link[link.index(old_obj)] = str(obj)
    exe = out / 'strata-poly-divtrace'
    link[link.index('-o') + 1] = str(exe)
    def compile_unit(entry):
        name, cmd = entry
        with (out / (name + '.log')).open('w') as log:
            sp.run(cmd, cwd=folder, check=True, stdout=log, stderr=sp.STDOUT)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(compile_unit, subprocesses.items()))
    sp.run(link, cwd=folder, check=True)
    (out / 'build.json').write_text(json.dumps({
        'head': sp.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_hashes': provenance,
        'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
        'link': link, 'compile': subprocesses,
    }, indent=2) + '\n')
    print(exe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=REPO / 'build-poly-hip')
    build(parser.parse_args().release.resolve())
