#!/usr/bin/env python3
"""Build a new live DFlash2 ELF with exact target arithmetic for widths 1 through 8."""

import argparse
import importlib.util
import os
from pathlib import Path

from build import REPO, R, after, insert, replace


def engine(s):
    s = after(s, 'namespace strata::poly {\n', '''void df2_exact_router(const float* w, const float* x, float* y, int n_in, int n_out, int n, void* stream);
''')
    s = after(s, 'void Engine::router_scores(const Layer& L, int n) {\n', '''    if (looped_ && windowed_ && n > 1 && n <= 8 && wave_gemv_ && L.gate_inp.type == kTypeF32 &&
        L.gate_inp.scale == 1.f && L.gate_inp.n_in == 2304 && L.gate_inp.n_out == 128) {
        route_dual_done_ = false;
        df2_exact_router(fp(L.gate_inp), xn_, rl_, L.gate_inp.n_in, L.gate_inp.n_out, n, st_);
        return;
    }
''')
    # Keep W1's Q8_0 lane and sum order if a W8 tall projection would select warps_cols.
    # Tideglass's shared output is Q6_K, so leave the separately checked Q6_K path intact.
    s = insert(s, '    if (cols_kernels_ && n >= 2 && n <= 8 &&\n', '''    if (looped_ && windowed_ && cols_kernels_ && w.type == GGML_TYPE_Q8_0 && n == 8 && w.n_out >= 8192) {
        const size_t token = kernels::native_q8_1_bytes(w.n_in);
        for (int i = 0; i < n; ++i)
            kernels::native_mmvq((int) w.type, w.p, (const uint8_t*) xq + (size_t) i * token,
                                 y + (size_t) i * w.n_out, w.n_in, w.n_out, 1, st_);
        return;
    }
''')
    return placement_trace(s)


def commit_profile(s):
    # One-shot, single-engine overlay. Live swap cadence intentionally stays out of scope.
    s = after(s, 'namespace strata::poly {\n', '''namespace {
std::array<std::vector<size_t>, 8> df2_window_routes;
}
''')
    s = after(s, 'void Engine::window(const int32_t* tokens, int t) {\n', '''    if (looped_ && !n_swap_)
        for (auto& rows : df2_window_routes) rows.clear();
''')
    s = replace(s, '                    ++use_[(size_t) il * n_expert_ + e];', '''                    const size_t route_id = (size_t) il * n_expert_ + e;
                    if (looped_ && windowed_ && !n_swap_)
                        df2_window_routes.at(at + t).push_back(route_id);
                    else
                        ++use_[route_id];''')
    return after(s, '    if (next_) next_->keep(n);\n', '''    if (looped_ && !n_swap_)
        for (int row = 0; row < n; ++row)
            for (size_t route_id : df2_window_routes.at(row)) ++use_[route_id];
''')


def placement_trace(s):
    # Compare actual slot IDs and relative ordering, not only the count, at the first decode call.
    return insert(s, '    tok_ = tokens;\n    const size_t e_row = (size_t) n_embd_, r_row =', '''    if (looped_ && (serial_decode_ || windowed_)) {
        static bool recorded = false; // one-shot, one-target binary
        if (!std::exchange(recorded, true)) {
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
            std::fprintf(stderr, "DF2_TARGET_PLACEMENT slots=%zu layout=%016llx swaps=%d\\n", resident,
                         (unsigned long long) hash, n_swap_);
        }
    }
''')


def kernels(s):
    return insert(s, 'void gemv_f32_waves(const float* w,', '''// As gemv_f32_waves_dual's W1 dot, one 256-thread block per output, batched over tokens.
void df2_exact_router(const float* w, const float* x, float* y, int n_in, int n_out, int n, void* stream) {
    if ((reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(x)) % 16)
        throw std::logic_error("unaligned live exact router input");
    gemv_rows_kernel<<<dim3(n_out, n), 256, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, n_out);
}

''')


def attention(s):
    anchor = 'void attend_decode(const Attend& a, float* scratch, cudaStream_t st, void* q8, bool group_rows) {'
    # The proven serial exactness control still applies to each query. Its launch can be shared by nearby
    # rows ONLY when both their partial and combine geometries match W1, and existing scratch can hold them.
    return insert(s, anchor, '''void attend_exact_window(const Attend& a, float* scratch, cudaStream_t st, void* q8) {
    if (a.n <= 1 || a.n > 8) throw std::logic_error("Inkling exact multi-row window required");
    {
        const auto span_of = [&](int row) {
            const int64_t end = a.pos + row + 1;
            return (int) (a.window ? std::min<int64_t>(a.window, end) : end);
        };
        constexpr int serial_max = kDecodeBlocks / kKv;
        for (int r = 0; r < a.n;) {
            const size_t qr = (size_t) r * kQuery, hr = (size_t) r * kHeads;
            auto* out8 = q8 ? static_cast<Q81*>(q8) + qr / 32 : nullptr;
            if (!a.k.scales) {
                Attend row = a;
                row.pos += r;
                row.n = 1;
                row.q += qr;
                row.rel += hr * kRelative;
                row.sink += hr;
                row.gate += qr;
                row.out += qr;
                attend_decode(row, scratch, st, out8, false);
                ++r;
                continue;
            }
            const int span = span_of(r), chunk = q8_chunk(span, serial_max), chunks = div_up(span, chunk);
            int count = 1;
            while (r + count < a.n && count < serial_max / chunks) {
                const int next = span_of(r + count);
                if (q8_chunk(next, serial_max) != chunk || div_up(next, chunk) != chunks) break;
                ++count;
            }
            decode_q8_kernel<><<<dim3(chunks, count), kAttThreads, 0, st>>>(
                a.q + qr, a.rel + hr * kRelative, a.table, a.k, a.v, a.pos + r, a.window, a.ring, chunk, scratch);
            decode_combine_kernel<<<dim3(kHeads, count), kHead, 0, st>>>(
                scratch, chunks, a.sink + hr, a.gate + qr, a.out + qr, out8);
            r += count;
        }
        check("Inkling W1-ordered verifier attention");
    }
}

''')


def inkling(s):
    s = after(s, 'namespace strata::poly {\n', '''namespace inkling {
void attend_exact_window(const Attend& a, float* scratch, cudaStream_t st, void* q8);
}
''')
    return replace(s, '    else if (decode_ && n < kStageRows) inkling::attend_decode(a, chunks_, cs(st_), attq_, ds_attend_group_);', '''    else if (decode_ && n < kStageRows) {
        if (windowed_ && n > 1 && n <= 8) inkling::attend_exact_window(a, chunks_, cs(st_), attq_);
        else inkling::attend_decode(a, chunks_, cs(st_), attq_, ds_attend_group_);
}''')


def group4_overlay(src, sources):
    result = profile_overlay(src, sources)
    # Do not change serial chunk geometry. The extra capacity holds exactly four
    # cohorts at production's 160-chunk setting; fall back at larger chunk counts.
    marker = 'void attend_exact_window(const Attend& a, float* scratch, cudaStream_t st, void* q8) {'
    result['inkling_attend.cu'] = insert(result['inkling_attend.cu'], marker, '''namespace {
constexpr int group4_chunks = 160;
constexpr int group4_blocks = 4 * kKv * group4_chunks;
}
size_t exact_group4_extra_floats() {
    return (size_t) (group4_blocks - kDecodeBlocks) * kChunkFloats;
}

''')
    result['inkling_attend.cu'] = replace(result['inkling_attend.cu'],
        '            while (r + count < a.n && count < serial_max / chunks) {', '''            const bool group4 = !a.window && span >= 32768 && chunks <= group4_chunks;
            const int max_rows = group4 ? 4 : serial_max / chunks;
            while (r + count < a.n && count < max_rows) {''')
    result['inkling_attend.cu'] = replace(result['inkling_attend.cu'],
        '            decode_q8_kernel<><<<dim3(chunks, count), kAttThreads, 0, st>>>(', '''            if (group4 && count > 1)
                decode_q8_kernel<4><<<dim3(kKv / 4, count, chunks), 4 * 32, 0, st>>>(
                    a.q + qr, a.rel + hr * kRelative, a.table, a.k, a.v, a.pos + r, a.window, a.ring, chunk, scratch);
            else decode_q8_kernel<><<<dim3(chunks, count), kAttThreads, 0, st>>>(''')
    result['inkling2.cu'] = after(result['inkling2.cu'], 'namespace inkling {\n',
                                 'size_t exact_group4_extra_floats();\n')
    result['inkling2.cu'] = replace(result['inkling2.cu'], 'chunks_ = fbuf(inkling::decode_floats() +',
                                  'chunks_ = fbuf(inkling::decode_floats() + inkling::exact_group4_extra_floats() +')
    return result


def overlay(src, sources):
    items = {'engine.cpp': engine, 'kernels.cu': kernels, 'inkling_attend.cu': attention}
    return {**{name: transform((src / name).read_text()) for name, transform in items.items()},
            'inkling2.cu': inkling(sources['inkling2.cu'])}


def profile_overlay(src, sources):
    result = overlay(src, sources)
    result['engine.cpp'] = commit_profile(result['engine.cpp'])
    return result


def add_cpu_reuse(src, result):
    path = REPO / 'rdna2-rnd/dflash2/reuse/build.py'
    spec = importlib.util.spec_from_file_location('cpu_reuse_builder', path)
    reuse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reuse)
    stock = (src / 'cpu_experts.cpp').read_text()
    # Append the tested candidate to the stock translation unit. Reuse its includes and
    # anonymous-namespace helpers; don't build a second pool or change its work order.
    variant = reuse.instrument(stock).split('void cpu_experts_reuse(', 1)[1]
    candidate = stock.rsplit('}  // namespace strata::poly', 1)[0] + 'void cpu_experts_reuse(' + variant
    candidate = after(candidate, '#include "cpu_experts.hpp"\n',
                      f'#include "{path.with_name("q4_multi.hpp")}"\n')
    result['cpu_experts.cpp'] = candidate
    decl = 'void cpu_experts_reuse(' + variant.split(') {', 1)[0] + ');\n'
    result['engine.cpp'] = after(result['engine.cpp'], 'namespace strata::poly {\n', decl)
    result['engine.cpp'] = replace(result['engine.cpp'], '        cpu_experts(*workers_, host, n_host, m,',
                                   '''        const auto compute = looped_ && windowed_ && m >= 3 && m <= 8
                           ? cpu_experts_reuse : cpu_experts;
        compute(*workers_, host, n_host, m,''')
    return result


def stock_prefill(src, _sources):
    # The live main supplies the target tape in K0, but the actual Inkling engine
    # is unmodified and attaches no drafter. Only add the placement fingerprint.
    return {'inkling2.cu': (src / 'inkling2.cu').read_text(),
            'engine.cpp': placement_trace((src / 'engine.cpp').read_text())}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=REPO / 'build-poly-hip')
    p.add_argument('--out', type=Path, default=R / 'live-exact-bin-v2')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--stock-prefill', action='store_true', help='build a stock target K0-only parity control')
    mode.add_argument('--commit-profile', action='store_true', help='count only committed target routes when swaps are disabled')
    mode.add_argument('--group4', action='store_true', help='commit-only profile and four-wave W1-ordered Q8 attention at long spans')
    p.add_argument('--cpu-reuse', action='store_true', help='enable experimental Zen 3 wide Q4 CPU dots in verifier windows')
    args = p.parse_args()
    if args.stock_prefill and args.cpu_reuse:
        p.error('--stock-prefill cannot change verifier CPU dots')
    os.umask(0o077)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('live_builder', REPO / 'rdna2-rnd/dflash2/live/build.py')
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    base = (stock_prefill if args.stock_prefill else group4_overlay if args.group4 else
            profile_overlay if args.commit_profile else overlay)
    def transform(src, sources):
        result = base(src, sources)
        return add_cpu_reuse(src, result) if args.cpu_reuse else result
    live.build(args.build.resolve(), out, extra_overlay=transform)
