// src/poly/hot_cache_test.cpp - exhaustive host verification for Hot Session Cache & Exact SWA State Replay.
//
// Builds standalone:
//   g++ -O2 -std=c++20 src/poly/hot_cache_test.cpp src/poly/spill.cpp -lpthread -o hot_cache_test && ./hot_cache_test
#include "conversations.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using strata::poly::CacheOptions;
using strata::poly::Conversations;

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

uint64_t mix(uint64_t h, int32_t t) {
    return (h ^ (uint64_t) (uint32_t) t) * 0x9E3779B97F4A7C15ull + 0x7F4A7C15ull;
}

// A fake engine implementing the exact SWA ring + conv state + global KV semantics of dense Q8 Inkling2:
// - Global KV: 2 uint64_t planes per token (hash after token, token id), row_bytes() = 16.
// - SWA ring: ring_rows slots (default 64, history 63) storing 2 uint64_t planes per token, swa_row_bytes() = 16.
// - Conv state: 3 previous raw tokens + running hash (32 bytes = conv_state_bytes()).
// Verifies byte-exact reconstruction after every single operation, including provisional verifier overwrites,
// in-VRAM fast rollbacks, cancellations, branching, full-tip parking, component trims, and vision isolation.
struct FakeDelta {
    static constexpr int64_t kCtx = 1 << 16;
    static constexpr int64_t kHist = 63;
    static constexpr int64_t kRing = 64;   // compact ring = kHist + 1 (or +4 for verifier tests)

    int64_t ring_rows = kRing;
    uint64_t h = 1;
    int32_t conv_hist[3] = {0, 0, 0};
    int64_t cur_pos = 0;
    int64_t ring_valid_lo = 0, ring_valid_hi = 0, ring_hw = 0;
    int64_t swa_h2d_rows = 0, swa_d2h_rows = 0;
    int64_t global_h2d_rows = 0, global_d2h_rows = 0;
    bool allow_delta = true;

    std::vector<uint64_t> ga = std::vector<uint64_t>(kCtx), gb = std::vector<uint64_t>(kCtx);
    std::vector<uint64_t> sa = std::vector<uint64_t>(128), sb = std::vector<uint64_t>(128);

    struct SwaRange { int64_t lo, hi; };

    bool has_swa_delta() const { return allow_delta; }
    int64_t pos() const { return cur_pos; }
    size_t state_bytes() const { return conv_state_bytes() + (size_t) kHist * swa_row_bytes(); }
    size_t row_bytes() const { return 16; }
    size_t conv_state_bytes() const { return 32; }
    size_t swa_row_bytes() const { return 16; }
    int64_t swa_ring_rows() const { return ring_rows; }
    int64_t swa_history_rows() const { return kHist; }
    SwaRange resident_swa_rows() const {
        if (ring_valid_lo == INT64_MAX) return {0, 0};
        const int64_t hi = std::min(ring_valid_hi, cur_pos);
        return ring_valid_lo < hi ? SwaRange{ring_valid_lo, hi} : SwaRange{0, 0};
    }

    void reset() {
        h = 1;
        conv_hist[0] = conv_hist[1] = conv_hist[2] = 0;
        cur_pos = 0;
        ring_valid_lo = 0;
        ring_valid_hi = 0;
        ring_hw = 0;
    }

    void eval(const int32_t* t, int64_t n) {
        const int64_t prev_lo = ring_valid_lo, prev_hi = ring_valid_hi, prev_hw = ring_hw;
        ring_valid_lo = INT64_MAX;
        ring_valid_hi = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t p = cur_pos + i;
            h = mix(h, t[i]);
            conv_hist[0] = conv_hist[1];
            conv_hist[1] = conv_hist[2];
            conv_hist[2] = t[i];
            ga[(size_t) p] = h;
            gb[(size_t) p] = (uint64_t) (uint32_t) t[i];
            sa[(size_t) (p % ring_rows)] = h ^ 0x5A5A5A5A5A5A5A5Aull;
            sb[(size_t) (p % ring_rows)] = (uint64_t) p;
        }
        const int64_t new_hw = std::max(prev_hw, cur_pos + n);
        const int64_t ring_min = std::max<int64_t>(0, new_hw - ring_rows);
        if (prev_lo != INT64_MAX && prev_lo <= cur_pos && cur_pos <= prev_hi) {
            ring_valid_lo = std::max(prev_lo, ring_min);
            ring_valid_hi = cur_pos + n;
        } else if (cur_pos == 0) {
            ring_valid_lo = ring_min;
            ring_valid_hi = n;
        }
        ring_hw = new_hw;
        cur_pos += n;
    }

    // Simulates a speculative verifier window of `t` tokens where only `keep_n` are accepted.
    void spec_window_keep(const int32_t* t, int64_t width, int64_t keep_n) {
        check(width <= ring_rows - kHist, "verifier width fits ring headroom");
        const int64_t p0 = cur_pos;
        const uint64_t h0 = h;
        int32_t ch0[3] = {conv_hist[0], conv_hist[1], conv_hist[2]};
        // Tentatively write all `width` rows into the ring (advancing ring_hw to p0 + width)
        eval(t, width);
        // Rewind conv/hash/pos to p0 + keep_n, leaving rejected slots overwritten in the ring!
        h = h0;
        std::memcpy(conv_hist, ch0, sizeof(ch0));
        for (int64_t i = 0; i < keep_n; ++i) {
            h = mix(h, t[i]);
            conv_hist[0] = conv_hist[1];
            conv_hist[1] = conv_hist[2];
            conv_hist[2] = t[i];
        }
        cur_pos = p0 + keep_n;
        // Notice ring_hw stays at p0 + width, so ring_valid_lo reflects the overwritten oldest slots!
    }

    void save_conv_state(uint8_t* dst) const {
        std::memcpy(dst, &h, 8);
        std::memcpy(dst + 8, conv_hist, 12);
        std::memset(dst + 20, 0, 12);
    }
    void load_conv_state(const uint8_t* src, int64_t p) {
        std::memcpy(&h, src, 8);
        std::memcpy(conv_hist, src + 8, 12);
        cur_pos = p;
        ring_valid_lo = std::max<int64_t>(0, p - kHist);
        ring_valid_hi = p;
        ring_hw = p;
    }

    void save_swa_rows(int64_t first, int64_t n, uint8_t* dst) {
        const auto live = resident_swa_rows();
        check(first >= live.lo && first + n <= live.hi && n > 0 && n <= ring_rows, "save_swa_rows inside resident");
        swa_d2h_rows += n;
        for (int64_t i = 0; i < n; ++i) {
            const size_t slot = (size_t) ((first + i) % ring_rows);
            std::memcpy(dst + (size_t) i * 8, &sa[slot], 8);
            std::memcpy(dst + (size_t) (n + i) * 8, &sb[slot], 8);
        }
    }
    void load_swa_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n, int64_t target_pos) {
        check(first >= from && first + n <= from + count && n > 0 && n <= ring_rows, "load_swa_rows valid slice");
        check(first >= std::max<int64_t>(0, target_pos - ring_rows) && first + n <= target_pos, "load_swa_rows no alias");
        ring_valid_lo = INT64_MAX;
        ring_valid_hi = 0;
        swa_h2d_rows += n;
        for (int64_t i = 0; i < n; ++i) {
            const size_t slot = (size_t) ((first + i) % ring_rows);
            std::memcpy(&sa[slot], src + (size_t) (first - from + i) * 8, 8);
            std::memcpy(&sb[slot], src + (size_t) (count + first - from + i) * 8, 8);
        }
    }

    void save_state(uint8_t* dst) const {
        save_conv_state(dst);
        const int64_t first = std::max<int64_t>(0, cur_pos - kHist);
        uint8_t* swa = dst + conv_state_bytes();
        std::memset(swa, 0, (size_t) kHist * 16);
        for (int64_t r = first; r < cur_pos; ++r) {
            const size_t off = (size_t) (r - (cur_pos - kHist));
            const size_t slot = (size_t) (r % ring_rows);
            std::memcpy(swa + off * 8, &sa[slot], 8);
            std::memcpy(swa + (kHist + off) * 8, &sb[slot], 8);
        }
    }
    void load_state(const uint8_t* src, int64_t p) {
        const int64_t first = std::max<int64_t>(0, p - kHist);
        const uint8_t* swa = src + conv_state_bytes();
        for (int64_t r = first; r < p; ++r) {
            const size_t off = (size_t) (r - (p - kHist));
            const size_t slot = (size_t) (r % ring_rows);
            std::memcpy(&sa[slot], swa + off * 8, 8);
            std::memcpy(&sb[slot], swa + (kHist + off) * 8, 8);
        }
        load_conv_state(src, p);
    }

    void save_rows(int64_t first, int64_t n, uint8_t* dst) {
        global_d2h_rows += n;
        std::memcpy(dst, &ga[(size_t) first], (size_t) n * 8);
        std::memcpy(dst + (size_t) n * 8, &gb[(size_t) first], (size_t) n * 8);
    }
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
        global_h2d_rows += n;
        std::memcpy(&ga[(size_t) first], src + (size_t) (first - from) * 8, (size_t) n * 8);
        std::memcpy(&gb[(size_t) first], src + (size_t) (count + first - from) * 8, (size_t) n * 8);
    }

    // Strict bit-exact check of global KV [0, pos), SWA ring [max(0, pos - kHist), pos), and conv state!
    bool holds_exact(const std::vector<int32_t>& ids) const {
        if (cur_pos != (int64_t) ids.size()) return false;
        uint64_t x = 1;
        int32_t ch[3] = {0, 0, 0};
        const int64_t n = (int64_t) ids.size();
        const int64_t swa_lo = std::max<int64_t>(0, n - kHist);
        for (int64_t i = 0; i < n; ++i) {
            x = mix(x, ids[(size_t) i]);
            ch[0] = ch[1]; ch[1] = ch[2]; ch[2] = ids[(size_t) i];
            if (ga[(size_t) i] != x || gb[(size_t) i] != (uint64_t) (uint32_t) ids[(size_t) i]) return false;
            if (i >= swa_lo) {
                const size_t slot = (size_t) (i % ring_rows);
                if (sa[slot] != (x ^ 0x5A5A5A5A5A5A5A5Aull) || sb[slot] != (uint64_t) i) return false;
            }
        }
        return h == x && std::memcmp(conv_hist, ch, sizeof(ch)) == 0;
    }
};

std::vector<int32_t> text(int seed, int n) {
    std::vector<int32_t> v((size_t) n);
    std::mt19937 r((unsigned) seed);
    for (int32_t& t : v) t = 1 + (int32_t) (r() % 50000);
    return v;
}

std::vector<int32_t> operator+(std::vector<int32_t> a, const std::vector<int32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

struct HotChat {
    FakeDelta e;
    Conversations<FakeDelta> kept;
    std::mt19937 rng{99};
    explicit HotChat(const CacheOptions& o) : kept(e, o) {}

    int64_t ask(std::vector<int32_t>& ids, int reply = 10, int64_t turn = -1, int64_t pin = -1,
                bool has_vision = false, bool use_spec = false) {
        const int64_t n = (int64_t) ids.size();
        const int64_t from = kept.resume(ids, has_vision);
        kept.turn_at(turn);
        if (pin > 0 && pin < n) kept.pin_at(pin);
        check(from >= 0 && from < n && e.pos() == from, "resume leaves engine at exact position");
        for (int64_t i = from; i < n;) {
            const bool last = i == n - 1;
            const int64_t room = std::min<int64_t>(n - 1 - i, 128);
            const int64_t m = last ? 1 : room;
            kept.before(i, last);
            if (!last) {
                std::vector<std::pair<int64_t, uint8_t*>> due;
                kept.inside(i, m, due);
                int64_t done = 0;
                for (const auto& d : due) {
                    e.eval(ids.data() + i + done, d.first - done);
                    done = d.first;
                    kept.capture_mark_at(i + done);
                }
                e.eval(ids.data() + i + done, m - done);
            } else {
                e.eval(ids.data() + i, 1);
            }
            kept.read(ids.data() + i, m);
            i += m;
        }
        kept.prompt_was(n);
        for (int k = 0; k < reply;) {
            if (use_spec && reply - k >= 2 && e.ring_rows >= FakeDelta::kHist + 4) {
                int32_t w[4];
                for (int j = 0; j < 4; ++j) w[j] = 1000000 + (int32_t) (rng() % 5000);
                const int keep_n = 1 + (int) (rng() % std::min(4, reply - k));
                e.spec_window_keep(w, 4, keep_n);
                kept.read(w, keep_n);
                kept.on_decode_commit();
                for (int j = 0; j < keep_n; ++j) ids.push_back(w[j]);
                k += keep_n;
            } else {
                const int32_t t = 1000000 + (int32_t) (rng() % 5000);
                e.eval(&t, 1);
                kept.read(&t, 1);
                kept.on_decode_commit();
                ids.push_back(t);
                ++k;
            }
        }
        check(e.holds_exact(ids) && kept.seq() == ids, "engine holds exact global KV, SWA ring, and conv state");
        kept.release();
        return from;
    }
};

CacheOptions hot_opts(int64_t park_mib = 16, int slots = 4, int64_t delta_every = 16) {
    CacheOptions o;
    o.hot_cache = true;
    o.every = 128;
    o.checkpoints = 4;
    o.park_mib = park_mib;
    o.park_slots = slots;
    o.park_min = 64;
    o.swa_delta_every = delta_every;
    o.active_delta_rows = 512;
    o.parked_tail_rows = 64;
    return o;
}

void test_full_tip_and_decode_deltas() {
    HotChat c(hot_opts(16, 4, 16));
    std::vector<int32_t> a = text(1, 300), b = text(2, 200);
    check(c.ask(a, 50) == 0, "cold A");
    const int64_t tip_a = (int64_t) a.size();   // 350 (includes all 50 decoded tokens!)
    check(c.ask(b, 20) == 0, "switch to B parks A at full tip");
    check(c.kept.parked() == 1, "A is parked");

    // Return to A: must resume at FULL TIP (350), not prompt-1 (299)!
    std::vector<int32_t> a2 = a + text(3, 15);
    check(c.ask(a2, 40) == tip_a, "full-tip parking resumes at exact committed decode tip");

    // Late divergence (tool-call JSON canonicalization mutates last 6 tokens of the 40-token reply!):
    // With swa_delta_every=16, we have delta snaps at +16 and +32 -> resumes at tip_a + 15 + 32 (not prompt-1!)
    const int64_t swa_before = c.e.swa_h2d_rows;
    std::vector<int32_t> a3(a2.begin(), a2.end() - 6);
    a3 = a3 + text(4, 12);
    const int64_t resumed = c.ask(a3, 10);
    check(resumed >= (int64_t) a2.size() - 16, "decode delta checkpoint catches late divergence within stride");
    // In-VRAM fast rollback: rolling back <= 16 rows on the active slot skips most of the 63-row SWA H2D!
    check(c.kept.swa_fast_rollbacks() >= 1, "in-VRAM fast rollback engaged");
    check(c.e.swa_h2d_rows - swa_before < FakeDelta::kHist, "in-VRAM fast rollback skipped live ring rows");
}

void test_pinned_prefix_and_shared_sessions() {
    HotChat c(hot_opts(16, 6, 32));
    const std::vector<int32_t> sys = text(10, 220);
    std::vector<int32_t> a = sys + text(11, 800);
    check(c.ask(a, 20, 220, 220) == 0, "cold A with pinned sys=220");
    // Grow A long enough that drop_of() would normally evict 220 without pinning
    for (int step = 0; step < 6; ++step) {
        a = a + text(20 + step, 160);
        c.ask(a, 10, (int64_t) a.size() - 160, 220);
    }
    // New Session B sharing only sys (220): MUST hit exact 220!
    std::vector<int32_t> b = sys + text(30, 100);
    check(c.ask(b, 15, 220, 220) == 220, "pinned system prefix anchor survived growth and hit exactly");
    // New Session C sharing sys (220): MUST hit exact 220 too!
    std::vector<int32_t> d = sys + text(31, 90);
    check(c.ask(d, 15, 220, 220) == 220, "second new session hits pinned system prefix anchor");
    // Return to A at its FULL TIP!
    const int64_t a_tip = (int64_t) a.size();
    a = a + text(32, 20);
    check(c.ask(a, 5, a_tip, 220) == a_tip, "A survived B & C at full tip");
}

void test_spec_verifier_overwrite_and_rollback() {
    auto o = hot_opts(16, 4, 8);
    HotChat c(o);
    c.e.ring_rows = FakeDelta::kHist + 4;   // compact W4 ring = 67 rows
    std::vector<int32_t> a = text(40, 200);
    check(c.ask(a, 60, 100, 100, false, true) == 0, "speculative decode with partial accepts");
    // Now roll back 12 tokens into the speculative reply and verify exact SWA + conv parity!
    std::vector<int32_t> rb(a.begin(), a.end() - 12);
    rb = rb + text(41, 15);
    const int64_t at = c.ask(rb, 20, 100, 100, false, true);
    check(at >= (int64_t) a.size() - 20, "rollback across speculative partial-accept windows hit delta snap");
}

void test_vision_isolation_and_adopt() {
    HotChat c(hot_opts(16, 4, 16));
    std::vector<int32_t> vis = text(50, 180);
    vis[50] = -1;
    check(c.ask(vis, 10, -1, -1, true) == 0, "vision prompt runs cleanly");
    // Same tokens with vision MUST NOT reuse or cross-share
    std::vector<int32_t> vis2 = vis + text(51, 20);
    check(c.ask(vis2, 5, -1, -1, true) == 0, "vision requests stay isolated");

    // Simulate DiskSessions RESTORE -> adopt(), then switch away and back -> Full-Tip Parking preserves it!
    c.e.reset();
    std::vector<int32_t> restored = text(60, 250);
    c.e.eval(restored.data(), (int64_t) restored.size());
    c.kept.adopt(restored);
    std::vector<int32_t> other = text(61, 150);
    check(c.ask(other, 10) == 0, "switch after adopt parks adopted session");
    std::vector<int32_t> cont = restored + text(62, 15);
    check(c.ask(cont, 10) == 250, "adopted session resumes at exact restored tip after switch");
}

void test_component_eviction_and_fuzz() {
    // Tight budget (1 MiB, enough for many sessions if interior snaps are trimmed!)
    auto o = hot_opts(1, 6, 16);
    HotChat c(o);
    std::mt19937 rng(777);
    const std::vector<int32_t> sys = text(70, 150);
    std::vector<std::vector<int32_t>> chats;
    for (int step = 0; step < 60; ++step) {
        std::vector<int32_t> ids;
        const unsigned w = rng() % 10;
        if (chats.empty() || w == 0) {
            ids = sys + text((int) rng(), 80 + (int) (rng() % 300));
        } else if (w == 1) {
            ids = text((int) rng(), 30 + (int) (rng() % 25));   // short side request (< park_min)
        } else {
            ids = chats[rng() % chats.size()];
            if (w < 5 && ids.size() > 20) ids.resize(10 + rng() % (ids.size() - 10));
            ids = ids + text((int) rng(), 1 + (int) (rng() % 120));
        }
        const int64_t pin = ids.size() > 150 && std::equal(sys.begin(), sys.end(), ids.begin()) ? 150 : -1;
        const int64_t turn = ids.size() > 20 ? (int64_t) (ids.size() - 10) : -1;
        c.ask(ids, (int) (rng() % 45), turn, pin, false, (rng() % 3) == 0);
        chats.push_back(ids);
        check(c.kept.parked() <= (size_t) o.park_slots, "parked count within slot cap");
        check(c.kept.parked_bytes() <= ((size_t) o.park_mib << 20), "parked bytes within budget");
    }
}

void test_snapshot_memory_pressure() {
    auto o = hot_opts();
    int attempts = 0;
    bool room = false;
    o.admit = [&](size_t) { ++attempts; return room; };
    HotChat c(o);
    std::vector<int32_t> a = text(1, 90);
    check(c.ask(a, 50) == 0, "cold request can proceed without snapshot memory");
    check(c.e.swa_d2h_rows == 0, "declined snapshots do not copy SWA rows");
    check(attempts == 5, "one prefill snapshot and four decode attempts spaced 16 tokens apart");
    room = true;
    const auto tip = (int64_t) a.size();
    auto next = a + text(2, 10);
    check(c.ask(next, 20) == tip, "live extension still reuses the tip under pressure");
    check(c.e.swa_d2h_rows > 0, "snapshot capture resumes when memory is available");
}

}  // namespace

int main() {
    test_full_tip_and_decode_deltas();
    test_pinned_prefix_and_shared_sessions();
    test_spec_verifier_overwrite_and_rollback();
    test_vision_isolation_and_adopt();
    test_component_eviction_and_fuzz();
    test_snapshot_memory_pressure();
    std::printf("hot_cache_test: %d checks passed\n", checks);
    return 0;
}
