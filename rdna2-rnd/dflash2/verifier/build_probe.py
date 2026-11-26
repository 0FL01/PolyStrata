#!/usr/bin/env python3
"""Build a scratch HIP probe by replacing only the touched strata-poly objects."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one probe insertion: {old!r}")
    return source.replace(old, new, 1)


def overlay(out, trace):
    src = ROOT / "src/poly"

    main = (src / "main.cpp").read_text()
    main = "#include <fstream>\n" + replace(main, "        if (max_new > 0) {",
        (HERE / "oracle.inc").read_text() + "        if (max_new > 0) {")
    (out / "main.cpp").write_text(main)

    ink = (src / "inkling2.cu").read_text()
    ink = replace(ink, "    opt_.window = 1;", '''    opt_.window = 1;
    if (std::getenv("RECON_TARGET") && !drafter) {
        opt_.window = 8;  // same allocations for every arm, including unmodified serial GEN
        const char* f = std::getenv("RECON_VERIFIER_FIX");
        cols_kernels_ = !f || !(std::atoi(f) & 4);
    }''')
    if trace:
        entry = "constexpr int kThreads = 128;"
        ink = replace(ink, entry, (HERE / "trace.inc").read_text() + entry)

        def before(marker, stages):
            nonlocal ink
            calls = [f'    recon_trace("{name}", visit_, pos_, n, {data}, sizeof(float) * {cols}, cs(st_));\n'
                     for name, data, cols in stages]
            ink = replace(ink, marker, "".join(calls) + marker)

        layer = "void Inkling2::layer_head(Layer& L, int il, int n) {"
        ink = replace(ink, layer, layer + '\n    recon_trace("input", visit_, pos_, n, r_, sizeof(float) * kWidth, cs(st_));')
        before("    if (fused(n) && n == 1) project(b,", [("q_input", "xq", "kWidth"), ("kv_input_k", "xk", "kWidth"),
                                                           ("kv_input_v", "xv", "kWidth")])
        before("    float* hist = state_ + (size_t) visit_ * kConvFloats;", [("q_proj", "q_", "kQuery"),
            ("k_proj", "k_", "kKey"), ("v_proj", "v_", "kKey"), ("rel_proj", "rel_", "kHeads * kRelative"),
            ("gate_proj", "gate_", "kQuery"), ("sink_proj", "sink_", "kHeads")])
        before("    attention(b, n);", [("q_norm", "q_", "kQuery"), ("rel_norm", "rel_", "kHeads * kRelative")])
        before("    lin(b.o, att_, ao_, n,", [("attention", "att_", "kQuery")])
        before("    depth_mix(fp(b.fquery), cur_, 0, n);", [("attention_residual", "r_", "kWidth")])
        before("    if (!batched(L, n)) moe_router(L, n);", [("router_input", "xn_", "kWidth")])
        router = "    if (!batched(L, n)) moe_router(L, n);"
        ink = replace(ink, router, router + '\n    if (!batched(L, n))\n'
            '        recon_trace("router", visit_, pos_, n, rl_, sizeof(float) * n_expert_, cs(st_));')
        tail = '        dbg("l_out", visit_, r_, kWidth);'
        ink = replace(ink, tail, '        recon_trace("output", visit_, pos_, n, r_, sizeof(float) * kWidth, cs(st_));\n' + tail)
    (out / "inkling2.cu").write_text(ink)

    eng = (src / "engine.cpp").read_text()
    limit = "    size_t arena = std::min({all, free_b - reserve, arena_.base ? arena_.reserved - 256 : most});"
    eng = replace(eng, limit, limit + '''
    if (const char* m = std::getenv("RECON_ARENA_MIB")) {
        const size_t desired = size_t(std::stoul(m)) << 20;
        if (arena < desired) throw std::runtime_error("not enough VRAM for the fixed expert arena");
        arena = desired;
    }''')
    # Exact physical (layer, expert) IDs, not just matching counts or a hit rate.
    dump = '''
    if (std::getenv("RECON_TARGET")) {
        std::fprintf(stderr, "RECON_PLACEMENT %s");
        for (size_t il = 0; il < layers_.size(); ++il) {
            const auto& slot = layers_[il].slot;
            for (size_t e = 0; e < slot.size(); ++e)
                if (slot[e]) std::fprintf(stderr, " %zu:%zu", il, e);
        }
        std::fputc('\\n', stderr);
    }'''
    # Substitute only the stage label; keep the C++ formats intact.
    load_end = "    if (full_bytes_ == 0) {"
    eng = replace(eng, load_end, dump.replace("%s", "load") + "\n" + load_end)
    dtor = "Engine::~Engine() {"
    eng = replace(eng, dtor, dtor + dump.replace("%s", "exit"))
    (out / "engine.cpp").write_text(eng)

    kern = (src / "kernels.cu").read_text()
    router = "n_in >= 1024 && (n_out + 7) / 8 * rows < 80"
    if kern.count(router) != 3:
        raise ValueError("expected matching single, dual and segmented GEMV launch conditions")
    kern = kern.replace(router, "recon_f32_block(n_in, n_out, rows)")
    marker = "void gemv_f32_waves(const float* w,"
    kern = replace(kern, marker, '''static bool recon_f32_block(int n_in, int n_out, int rows) {
    static const int recon_fix = [] {
        const char* f = std::getenv("RECON_VERIFIER_FIX"); return f ? std::atoi(f) : 0;
    }();
    const int test_rows = (recon_fix & 1) && rows <= 8 ? 1 : rows;
    return n_in >= 1024 && (n_out + 7) / 8 * test_rows < 80;
}

''' + marker)
    (out / "kernels.cu").write_text(kern)

    attn = (src / "inkling_attend.cu").read_text()
    entry = "void attend_decode(const Attend& a, float* scratch, cudaStream_t st, void* q8, bool group_rows) {"
    attn = replace(attn, entry, entry + '''
    static const int recon_fix = [] {
        const char* f = std::getenv("RECON_VERIFIER_FIX"); return f ? std::atoi(f) : 0;
    }();
    if ((recon_fix & 2) && a.n > 1) {
        // Diagnostic exact control: reproduce each serial row's span, chunk size and combine order.
        for (int r = 0; r < a.n; ++r) {
            Attend row = a;
            row.n = 1;
            row.pos += r;
            row.q += (size_t) r * kQuery;
            row.rel += (size_t) r * kHeads * kRelative;
            row.sink += (size_t) r * kHeads;
            row.gate += (size_t) r * kQuery;
            row.out += (size_t) r * kQuery;
            auto* quant = q8 ? static_cast<Q81*>(q8) + (size_t) r * (kQuery / 32) : nullptr;
            attend_decode(row, scratch, st, quant, false);
        }
        return;
    }''')
    (out / "inkling_attend.cu").write_text(attn)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, default=ROOT / "build-poly-hip")
    ap.add_argument("--output", type=Path, required=True, help="new scratch build directory")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--trace", action="store_true", help="add optional RECON_TRACE_POS operation fingerprints")
    a = ap.parse_args()
    build, out = a.build.resolve(), a.output.resolve()
    subprocess.run(["ninja", "-j", str(a.jobs), "strata-poly"], cwd=build, check=True)
    commands = subprocess.check_output(["ninja", "-t", "commands", "strata-poly"], cwd=build, text=True)
    [link] = [shlex.split(l) for l in commands.splitlines() if " --hip-link " in l and " -o strata-poly " in l]
    if "--offload-arch=gfx1031" not in link:
        raise ValueError("expected RX 6700 XT (gfx1031) HIP build")
    out.mkdir(parents=True, exist_ok=False)
    overlay(out, a.trace)
    link = link[2:-2] if link[:2] == [":", "&&"] and link[-2:] == ["&&", ":"] else link
    for i, word in enumerate(link):
        if word.startswith("--dependency-file="):
            assert link[i-1] == "-Xlinker"
            del link[i-1:i+1]
            break
    objects = []
    for path in sorted(out.iterdir()):
        src = ROOT / "src/poly" / path.name
        [cmd] = [shlex.split(l) for l in commands.splitlines() if l.endswith(f" -c {src}")]
        obj = out / (path.name + ".o")
        # Leave dependency files (if any) in the private build directory.
        for flag, value in (("-o", obj), ("-c", path), ("-MT", obj), ("-MF", obj.with_suffix(".d"))):
            if flag in cmd:
                cmd[cmd.index(flag) + 1] = str(value)
        objects.append(cmd + ["-I" + str(src.parent)])
        i = next(i for i, word in enumerate(link) if word.endswith(f"/src/poly/{path.name}.o"))
        link[i] = str(obj)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(lambda cmd: subprocess.run(cmd, cwd=build, check=True), objects))
    binary = out / "strata-poly-probe"
    link[link.index("-o") + 1] = str(binary)
    subprocess.run(link, cwd=build, check=True)
    print(binary)


if __name__ == "__main__":
    main()
