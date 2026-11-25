// Host-only: c++ -std=c++20 -O2 src/poly/inkling_swa_layout_test.cpp -o inkling_swa_layout_test
#include "inkling_swa_layout.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
using namespace strata::poly::inkling;
constexpr int kWindow = 1024, kVisits = 40;
constexpr size_t kQ8Row = 2176, kF16Row = 4096, kInputRow = 9216;

constexpr SwaPlan plan(int context, int rows, int verify, size_t kv, bool on = true) {
    rows = std::min(rows, context);
    return plan_swa(context, kWindow, rows, std::min(rows, verify), kVisits, kv, kInputRow * rows, on);
}

static_assert(plan(200000, 2048, 1, kQ8Row, false).ring_rows == 3071);
static_assert(plan(200000, 2048, 1, kQ8Row).ring_rows == 1024);
static_assert(plan(200000, 2048, 8, kQ8Row).ring_rows == 1031);
static_assert(plan(200000, 2048, 1, kQ8Row).stage_rows * kQ8Row == 6715136);
static_assert(plan(200000, 2048, 1, kQ8Row).saved_bytes == kVisits * 2047 * kQ8Row);
static_assert(plan(200000, 2048, 1, kF16Row).stage_rows * kF16Row == 12640256);
static_assert(plan(200000, 320, 1, kQ8Row).scratch_bytes > 0 && plan(200000, 321, 1, kQ8Row).borrow_input);
static_assert(plan(200000, 830, 1, kF16Row).scratch_bytes > 0 && plan(200000, 831, 1, kF16Row).borrow_input);
static_assert(plan(1031, 2048, 8, kQ8Row).saved_bytes == 0);
static_assert(plan(1050, 32, 1, kQ8Row).saved_bytes == 0);   // dedicated stage is larger than the ring saving
static_assert(plan(1051, 32, 1, kQ8Row).saved_bytes > 0);

using Row = std::array<uint64_t, 4>;   // distinct sentinels for the four planes, not attention math
using Rows = std::vector<Row>;

void require(bool ok, const char* why) {
    if (ok) return;
    std::cerr << why << '\n';
    std::abort();
}

Row marker(int64_t p, int append = 0) {
    Row r{};
    for (size_t i = 0; i < r.size(); ++i) r[i] = 1 + p * 16 + append * 4 + i;
    return r;
}

int64_t oldest(int64_t pos) { return std::max<int64_t>(0, pos - (kWindow - 1)); }

Rows save(const Rows& ring, int context, int64_t pos) {
    Rows out(std::min(context, kWindow - 1));
    const int64_t begin = pos - (int64_t) out.size();
    for (int64_t p = std::max<int64_t>(0, begin); p < pos;) {
        const auto [row, count] = swa_ring_span(p, pos, (int) ring.size());
        std::copy_n(ring.begin() + row, count, out.begin() + (p - begin));
        p += count;
    }
    return out;
}

void restore(const Rows& image, Rows& ring, int64_t pos) {
    const int64_t begin = pos - (int64_t) image.size();
    for (int64_t p = std::max<int64_t>(0, begin); p < pos;) {
        const auto [row, count] = swa_ring_span(p, pos, (int) ring.size());
        std::copy_n(image.begin() + (p - begin), count, ring.begin() + row);
        p += count;
    }
}

void scenario(int context, int batch, int verify, size_t kv, int64_t pos, int n) {
    const auto stock = plan(context, batch, verify, kv, false), compact = plan(context, batch, verify, kv);
    const int r = compact.ring_rows;
    Rows a(stock.ring_rows), b(r);
    for (int64_t p = oldest(pos); p < pos; ++p) a[p % a.size()] = marker(p);
    const auto checkpoint = save(a, context, pos);
    restore(checkpoint, b, pos);
    require(save(b, context, pos) == checkpoint, "stock to compact checkpoint round trip");

    const auto layout = swa_stage_layout(pos, kWindow);
    const bool staged = !layout.fits_ring(n, r);
    const int64_t append = staged ? layout.append() : pos;
    const int64_t shift = pos - append;
    Rows stage(compact.stage_rows);
    if (staged) {
        require(shift % kSwaKeyTile == 0, "Flash tile phase changed");
        require(append + n <= (int64_t) stage.size(), "staging capacity exceeded");
        for (int t = 0; t < layout.history; ++t) stage[layout.head + t] = b[(layout.first + t) % r];
    }
    for (int t = 0; t < n; ++t) {
        a[(pos + t) % a.size()] = marker(pos + t, 1);
        (staged ? stage[append + t] : b[(pos + t) % r]) = marker(pos + t, 1);
    }
    for (int t = 0; t < n; ++t) {
        const int64_t end = pos + t;
        require(oldest(append + t) == oldest(end) - shift, "attention span or relative distance changed");
        for (int64_t p = oldest(end); p <= end; ++p)
            require(a[p % a.size()] == (staged ? stage[p - shift] : b[p % r]), "attention read the wrong row");
    }
    if (staged) {
        const int count = std::min(n, r), skip = n - count;
        for (int t = skip; t < n; ++t) b[(pos + t) % r] = stage[append + t];
    }
    for (int keep = n <= verify ? 1 : n; keep <= n; ++keep) {
        const auto canonical = save(b, context, pos + keep);
        require(canonical == save(a, context, pos + keep), "canonical state differs after append or partial keep");
        Rows restored(a.size());
        restore(canonical, restored, pos + keep);
        require(save(restored, context, pos + keep) == canonical, "compact to stock checkpoint round trip");
        if (keep < n) {
            Rows continued = b;
            continued[(pos + keep) % r] = marker(pos + keep, 2);
            restored[(pos + keep) % a.size()] = marker(pos + keep, 2);
            require(save(continued, context, pos + keep + 1) == save(restored, context, pos + keep + 1),
                    "continuation did not overwrite rejected row");
        }
    }
}
}  // namespace

int main() {
    int cases = 0;
    for (size_t kv : {kQ8Row, kF16Row})
        for (int verify : {1, 8})
            for (int batch : {32, 257, 2048, 4096})
                for (int context : {48, 1025, 200000})
                    for (int pos : {0, 17, 1017, 1023, 1024, 1031, 3070, 3071, 65569, 199997})
                        for (int n : {1, 2, 7, 8, 31, 32, 65, 257, 1023, 1024, 2048, 4096}) {
                            if (n > batch || pos + n > context) continue;
                            scenario(context, batch, verify, kv, pos, n);
                            ++cases;
                        }
    std::cout << "SWA layout: " << cases << " host scenarios passed\n";
}
