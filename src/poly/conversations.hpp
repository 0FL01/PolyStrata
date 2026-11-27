// src/poly/conversations.hpp - the conversations a server has read, kept so that a prompt which starts like one of
// them is not read again from its first token.
//
// The engine holds one sequence at a time.  Coming back to an earlier point of a sequence takes two things: the
// recurrent state at that point, which is a checkpoint, and the rows its tokens left in the attention caches.  The
// rows of the sequence in the engine are there already.  A conversation that makes room for another is parked:
// its rows are copied to host memory, up to its last checkpoint (or up to its committed tip under hot_cache).  What
// is saved is written once and never changed, so conversations that start alike share it, and a parked conversation
// comes back bit for bit.  Between requests cold blobs leave the process for the file cache (release(), spill.hpp).
//
// Nothing here knows the model.  E is the engine:
//     void reset();
//     size_t state_bytes() const;  void save_state(uint8_t*);  void load_state(const uint8_t*, int64_t pos);
//     size_t row_bytes() const;    void save_rows(int64_t first, int64_t n, uint8_t*);
//     void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n);
//
// Optional Hot Session Cache & Exact SWA State Replay interface (detected via C++20 requires + has_swa_delta()):
//     bool has_swa_delta() const;
//     int64_t pos() const;
//     size_t conv_state_bytes() const;
//     size_t swa_row_bytes() const;
//     int64_t swa_ring_rows() const;
//     int64_t swa_history_rows() const;
//     auto resident_swa_rows() const;   // {int64_t lo, hi;}
//     void save_conv_state(uint8_t*);
//     void load_conv_state(const uint8_t*, int64_t pos);
//     void save_swa_rows(int64_t first, int64_t n, uint8_t*);
//     void load_swa_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n, int64_t target_pos);
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "spill.hpp"

namespace strata::poly {

template<class E>
concept HasSwaDelta = requires(E& e, const E& ce, uint8_t* dst, const uint8_t* src, int64_t p) {
    { ce.has_swa_delta() } -> std::convertible_to<bool>;
    { ce.pos() } -> std::convertible_to<int64_t>;
    { ce.conv_state_bytes() } -> std::convertible_to<size_t>;
    { ce.swa_row_bytes() } -> std::convertible_to<size_t>;
    { ce.swa_ring_rows() } -> std::convertible_to<int64_t>;
    { ce.swa_history_rows() } -> std::convertible_to<int64_t>;
    { ce.resident_swa_rows().lo } -> std::convertible_to<int64_t>;
    { ce.resident_swa_rows().hi } -> std::convertible_to<int64_t>;
    e.save_conv_state(dst);
    e.load_conv_state(src, p);
    e.save_swa_rows(p, p, dst);
    e.load_swa_rows(src, p, p, p, p, p);
};

struct CacheOptions {
    int checkpoints = 6;       ///< kept per conversation; 0: a prompt is reread unless it goes on from the last token
    int64_t every = 2048;      ///< a checkpoint every so many prompt tokens, besides the one at a prompt's end and
                               ///< the one where its last turn begins (turn_at()); 0: none
    int64_t park_mib = 0;      ///< host memory for parked conversations; 0: none is parked
    int park_slots = 4;        ///< parked conversations at most
    int64_t park_min = 1024;   ///< a conversation with fewer tokens before its last checkpoint is not parked
    bool hot_cache = false;    ///< opt-in Hot Session Cache & Exact SWA State Replay (full-tip parking, SWA deltas,
                               ///< pinned prefix anchors, component-level eviction, hierarchical L2/L3 spill)
    int64_t swa_delta_every = 256;   ///< decode tokens between incremental SWA+conv delta checkpoints (0: off)
    int64_t active_delta_rows = 4096;///< max rolling delta span behind tip on the active conversation
    int64_t parked_tail_rows = 512;  ///< pre-tip delta span retained on parked slots before Tier-2 trim
    int64_t l2_ram_mib = -1;         ///< parked MiB kept hot in L2 RAM before spilling to L3 (-1: park_mib)
    std::function<bool(size_t)> admit;   ///< whether the host can spare this many bytes; unset: it can
    std::function<bool()> l2_ok;         ///< whether host RAM has headroom to keep L2 hot without spilling; unset: yes
    FILE* log = nullptr;       ///< where parking is reported; the server reads these lines
};

template<class E>
class Conversations {
public:
    Conversations(E& engine, CacheOptions o)
        : e_(engine), o_(std::move(o)), slots_((size_t) std::max(o_.park_slots, 0) + 1) {}

    /// The tokens the engine has read.
    const std::vector<int32_t>& seq() const { return slots_[cur_].seq; }

    /// Whether the opt-in Hot Session Cache & Exact SWA State Replay path is active on this engine.
    bool uses_swa_delta() const {
        if (!o_.hot_cache) return false;
        if constexpr (HasSwaDelta<E>) return e_.has_swa_delta();
        else return false;
    }

    /// Leaves the engine at the furthest point of `ids` that can be come back to, and returns that position.
    int64_t resume(const std::vector<int32_t>& ids, bool has_vision = false) {
        if (uses_swa_delta()) return resume_hot(ids, has_vision);
        const int64_t n = (int64_t) ids.size();
        turn_ = -1;
        pin_ = -1;
        size_t best = cur_;
        int64_t at = 0, same = 0;
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (has_vision || slots_[i].has_vision) continue;
            const int64_t c = common(slots_[i].seq, ids), r = reach(i, c, n);
            if (r > at || (r == at && i == cur_)) best = i, at = r, same = c;
        }
        ++clock_;
        if (at == 0) {
            // nothing kept starts like this
            if (!cur().seq.empty() && !cur().has_vision && park(slots_.size())) cur_ = take(cur_);
            slots_[cur_] = Slot{};
            slots_[cur_].has_vision = has_vision;
            e_.reset();
        } else if (best != cur_ || at < len(cur())) {
            if (best != cur_) enter(best, at);
            // The conversation is left at `at`.  If the prompt the client sent for it went on from there, the
            // client may send that again, so the conversation is parked and this one goes on in a copy.
            if (same >= cur().prompt || !branch(at)) cut(cur(), at);
            e_.load_state(state_at(cur(), at), at);
        }
        cur().stamp = clock_;
        return at;
    }

    /// One more checkpoint of the prompt resume() took up: where its chat's last turn begins, which the server
    /// knows from the template.  The chat's next request renders the conversation up to there as this one does,
    /// but this turn's header and reply not always in the same tokens (a template drops the reply's thinking), so
    /// the checkpoint before the prompt's last token is seldom where it goes on from.  A place the read does not
    /// pass adds none.
    void turn_at(int64_t at) { turn_ = at; }

    /// A semantic shared-prefix boundary (e.g., end of system prompt + tools) to pin against drop_of() and Tier-1 trim.
    void pin_at(int64_t at) {
        pin_ = at;
        if (!uses_swa_delta() || at < 1) return;
        for (ConvSnap& s : cur().snaps)
            if (s.pos == at) s.pinned = true;
    }

    /// Of `want` tokens from position `at`, how many to read before the next checkpoint is due.
    int64_t until_checkpoint(int64_t at, int64_t want) const {
        if (o_.checkpoints < 1) return want;
        if (o_.every > 0) want = std::min(want, (at / o_.every + 1) * o_.every - at);
        if (turn_ > at) want = std::min(want, turn_ - at);
        if (uses_swa_delta() && pin_ > at) want = std::min(want, pin_ - at);
        return want;
    }

    /// Before the engine reads the token at `at`; `last` says it is the prompt's last.  Takes a checkpoint if one is
    /// due.  One sits before the last token of every prompt, so coming back to it leaves a token to read and the
    /// logits that come with it.
    void before(int64_t at, bool last) {
        if (o_.checkpoints < 1 || at < 1) return;
        const bool want_pin = uses_swa_delta() && at == pin_;
        if (!last && at != turn_ && !want_pin && (o_.every < 1 || at % o_.every != 0)) return;
        if (uses_swa_delta()) {
            capture_snap_hot(cur().snaps, nullptr, at, want_pin, false);
            return;
        }
        if (const BlobPtr b = mark(at)) e_.save_state(b->data());
    }

    /// For an engine that reads past checkpoints in one go, of n tokens from `at`: the checkpoints due after the
    /// read's first token.  `out` gets, for each, how many of the read's tokens come before it and where its
    /// state_bytes() go; null where no state is wanted, and the engine still breaks its work there as if one
    /// were.  The engine writes a state when it has it.  The checkpoints are the conversation's once read() has
    /// the read's tokens, and not at all when the read is given up (rewind()).
    ///
    /// Under hot_cache (uses_swa_delta()), `out` carries null state pointers and the caller supplies `capture_mark_at`
    /// as Engine::read()'s on_mark callback so incremental SWA+conv snapshots are captured without pre-allocating
    /// full 88.57 MiB buffers.
    void inside(int64_t at, int64_t n, std::vector<std::pair<int64_t, uint8_t*>>& out) {
        out.clear();
        due_.clear();
        after_.clear();
        due_hot_.clear();
        pending_snaps_.clear();
        if (o_.checkpoints < 1) return;
        std::vector<int64_t> due;
        if (o_.every > 0)
            for (int64_t p = (at / o_.every + 1) * o_.every; p < at + n; p += o_.every) due.push_back(p);
        const auto add_due = [&](int64_t p) {
            if (p > at && p < at + n && std::find(due.begin(), due.end(), p) == due.end())
                due.insert(std::upper_bound(due.begin(), due.end(), p), p);
        };
        add_due(turn_);
        if (uses_swa_delta()) add_due(pin_);
        if (due.empty()) return;
        if (uses_swa_delta()) {
            // Compute which non-pinned coarse checkpoints survive drop_of_hot(); pinned anchors always stay.
            std::vector<int64_t> coarse;
            for (const ConvSnap& m : cur().snaps)
                if (!m.pinned && !m.is_delta) coarse.push_back(m.pos);
            for (const int64_t p : due) {
                if (p == pin_) continue;
                coarse.push_back(p);
                while (coarse.size() > (size_t) o_.checkpoints)
                    coarse.erase(coarse.begin() + (std::ptrdiff_t) drop_of_hot(coarse, pin_));
            }
            for (const int64_t p : due) {
                const bool stays = p == pin_ || std::find(coarse.begin(), coarse.end(), p) != coarse.end();
                if (stays) due_hot_.push_back(p);
                out.emplace_back(p - at, nullptr);
            }
            return;
        }
        for (const Checkpoint& m : cur().marks) after_.push_back(m.pos);
        for (const int64_t p : due) {
            after_.push_back(p);
            while (after_.size() > (size_t) o_.checkpoints) after_.erase(after_.begin() + (std::ptrdiff_t) drop_of(after_));
        }
        for (const int64_t p : due) {
            const bool stays = std::find(after_.begin(), after_.end(), p) != after_.end();
            due_.push_back({p, stays ? blob(e_.state_bytes()) : nullptr});
            out.emplace_back(p - at, due_.back().state ? due_.back().state->data() : nullptr);
        }
    }

    /// Invoked by Engine::read()'s on_mark callback at each mark inside a looped prefill read when uses_swa_delta().
    void capture_mark_at(int64_t pos) {
        if (!uses_swa_delta() || due_hot_.empty()) return;
        if (std::find(due_hot_.begin(), due_hot_.end(), pos) == due_hot_.end()) return;
        capture_snap_hot(pending_snaps_, &cur().snaps, pos, pos == pin_, false);
    }

    /// The engine has read these tokens.  A read that inside() was asked about is done with them, and its
    /// checkpoints take their places.
    void read(const int32_t* t, int64_t n) {
        Slot& s = cur();
        s.seq.insert(s.seq.end(), t, t + n);
        if (uses_swa_delta()) {
            if (!pending_snaps_.empty()) {
                for (ConvSnap& sn : pending_snaps_) insert_snap_hot(s.snaps, std::move(sn));
                prune_active_snaps_hot(s);
                pending_snaps_.clear();
            }
            due_hot_.clear();
            return;
        }
        if (due_.empty()) return;
        const auto stays = [&](const Checkpoint& m) {
            return m.state && std::find(after_.begin(), after_.end(), m.pos) != after_.end();
        };
        std::vector<Checkpoint> marks;
        for (const Checkpoint& m : s.marks)
            if (stays(m)) marks.push_back(m);
        for (const Checkpoint& m : due_)
            if (stays(m)) marks.push_back(m);
        s.marks = std::move(marks);
        due_.clear();
    }

    /// Call after committing decoded token(s) (eval_decode or keep(fed)) to capture periodic incremental SWA+conv deltas.
    void on_decode_commit() {
        if (!uses_swa_delta() || o_.swa_delta_every < 1 || o_.checkpoints < 1) return;
        if constexpr (HasSwaDelta<E>) {
            Slot& s = cur();
            if (s.has_vision) return;
            const int64_t tip = len(s);
            if (tip < 1 || e_.pos() != tip) return;
            const int64_t last_p = s.snaps.empty() ? 0 : s.snaps.back().pos;
            if (tip - last_p < o_.swa_delta_every) return;
            capture_snap_hot(s.snaps, nullptr, tip, false, true);
            prune_active_snaps_hot(s);
        }
    }

    /// A read was given up part way: what it left in the engine is not the conversation's, and the checkpoints
    /// inside it are not taken.  The conversation goes back to its last checkpoint, which the engine is left
    /// at, or is forgotten when it has none.  Returns where the conversation is then.
    int64_t rewind() {
        due_.clear();
        due_hot_.clear();
        pending_snaps_.clear();
        Slot& s = cur();
        if (uses_swa_delta()) {
            if (s.snaps.empty()) {
                slots_[cur_] = Slot{};
                e_.reset();
                return 0;
            }
            const int64_t at = s.snaps.back().pos;
            cut(s, at);
            restore_snap_hot(s.snaps.back(), false);
            return at;
        }
        if (s.marks.empty()) {
            slots_[cur_] = Slot{};
            e_.reset();
            return 0;
        }
        const int64_t at = s.marks.back().pos;
        cut(s, at);
        e_.load_state(s.marks.back().state->data(), at);
        return at;
    }

    /// The first n tokens of the conversation are the prompt the client sent.
    void prompt_was(int64_t n) { cur().prompt = std::min(n, len(cur())); }

    /// After an engine error: the conversation being read is lost.  The caller resets the engine.
    void drop() {
        slots_[cur_] = Slot{};
        due_.clear();
        due_hot_.clear();
        pending_snaps_.clear();
    }

    /// The engine's sequence is about to hold another conversation, read elsewhere (a batch slot's copy): the one
    /// it holds is parked, as resume() parks it, or forgotten.  adopt() then says what it holds.
    void leave() {
        if (!cur().seq.empty() && !cur().has_vision) {
            const bool ok = uses_swa_delta() ? park_hot(slots_.size()) : park(slots_.size());
            if (ok) cur_ = take(cur_);
        }
        slots_[cur_] = Slot{};
        due_.clear();
        due_hot_.clear();
        pending_snaps_.clear();
    }

    /// The engine holds `seq` now, with the state at its end: a prompt that goes on from there reads none of it.
    void adopt(std::vector<int32_t> seq) {
        Slot& s = cur();
        s = Slot{};
        s.seq = std::move(seq);
        s.prompt = len(s);
        s.stamp = ++clock_;
    }

    /// Between requests: what the parked conversations hold leaves the process (legacy), or stays hot in L2 RAM
    /// up to l2_ram_mib and spills colder blobs to L3 file cache/SSD (hot_cache).
    void release() const {
        if (!uses_swa_delta()) {
            for (size_t i = 0; i < slots_.size(); ++i) {
                if (i == cur_) continue;
                for (const Checkpoint& m : slots_[i].marks) m.state->release();
                for (const Rows& r : slots_[i].rows) r.blob->release();
            }
            return;
        }
        const bool keep_l2 = !o_.l2_ok || o_.l2_ok();
        const int64_t mib = o_.l2_ram_mib >= 0 ? o_.l2_ram_mib : o_.park_mib;
        const size_t l2_budget = keep_l2 && mib > 0 ? ((size_t) mib << 20) : 0;
        // Order parked slots hottest-first so the most recently active sessions stay in L2 RAM.
        std::vector<size_t> order;
        for (size_t i = 0; i < slots_.size(); ++i)
            if (i != cur_ && !slots_[i].seq.empty()) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return slots_[a].stamp > slots_[b].stamp; });
        std::vector<const Blob*> seen;
        // Blobs shared with the active slot stay in RAM.
        collect_blobs(slots_[cur_], seen);
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        size_t l2_used = 0;
        for (const size_t i : order) {
            std::vector<const Blob*> bs;
            collect_blobs(slots_[i], bs);
            std::sort(bs.begin(), bs.end());
            bs.erase(std::unique(bs.begin(), bs.end()), bs.end());
            for (const Blob* b : bs) {
                if (std::binary_search(seen.begin(), seen.end(), b)) continue;
                seen.insert(std::upper_bound(seen.begin(), seen.end(), b), b);
                if (l2_used + b->bytes() <= l2_budget) l2_used += b->bytes();
                else b->release();
            }
        }
    }

    /// The parked conversations, the bytes they hold, and how many were pushed out so far.
    size_t parked() const { return held(slots_.size()).count; }
    size_t parked_bytes() const { return held(slots_.size()).bytes; }
    size_t evictions() const { return evictions_; }
    size_t component_trims() const { return component_trims_; }
    size_t swa_fast_rollbacks() const { return swa_fast_rollbacks_; }

private:
    using Clock = std::chrono::steady_clock;

    using Blob = Spill;
    using BlobPtr = std::shared_ptr<const Blob>;
    static constexpr int64_t kHotRowChunk = 4096;   // ~68 MiB at Q8 global KV; fits 128 MiB narrow stage_ for packed DMA

    struct Checkpoint {
        int64_t pos;
        BlobPtr state;
    };
    struct SwaSlice {
        BlobPtr blob;
        int64_t from = 0, count = 0;   // backing blob holds [from, from + count)
        int64_t first = 0, n = 0;      // slice uses [first, first + n)
    };
    struct ConvSnap {
        int64_t pos = 0;
        bool pinned = false;
        bool is_delta = false;
        BlobPtr conv;
        std::vector<SwaSlice> swa;     // sorted, gap-free coverage of [max(0, pos - hist), pos)
    };
    // The blob holds `count` rows from row `first`; the first `keep` of them are this conversation's.
    struct Rows {
        BlobPtr blob;
        int64_t first, count, keep;
    };
    struct Slot {
        std::vector<int32_t> seq;
        int64_t prompt = 0;             // seq's first tokens that the client sent
        std::vector<Checkpoint> marks;  // legacy checkpoints by position
        std::vector<ConvSnap> snaps;    // hot-cache incremental SWA+conv checkpoints by position
        std::vector<Rows> rows;         // back to back from row 0; a parked conversation has all of its rows
        uint64_t stamp = 0;
        bool has_vision = false;
    };
    struct Held {
        size_t count = 0, bytes = 0;
    };

    Slot& cur() { return slots_[cur_]; }
    const Slot& cur() const { return slots_[cur_]; }
    static int64_t len(const Slot& s) { return (int64_t) s.seq.size(); }
    static int64_t saved(const Slot& s) { return s.rows.empty() ? 0 : s.rows.back().first + s.rows.back().keep; }

    static int64_t common(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return (int64_t) n;
    }

    static void collect_blobs(const Slot& s, std::vector<const Blob*>& out) {
        for (const Checkpoint& m : s.marks)
            if (m.state) out.push_back(m.state.get());
        for (const ConvSnap& sn : s.snaps) {
            if (sn.conv) out.push_back(sn.conv.get());
            for (const SwaSlice& sl : sn.swa)
                if (sl.blob) out.push_back(sl.blob.get());
        }
        for (const Rows& r : s.rows)
            if (r.blob) out.push_back(r.blob.get());
    }

    // How far into a prompt of n tokens slot i gets it, given that they start with `same` tokens alike.
    int64_t reach(size_t i, int64_t same, int64_t n) const {
        const Slot& s = slots_[i];
        if (i == cur_ && same > 0 && same == len(s) && n > same) return same;   // the engine is right there
        int64_t at = 0;
        for (const Checkpoint& m : s.marks)
            if (m.pos <= std::min(same, n - 1)) at = m.pos;
        return at;
    }

    static const uint8_t* state_at(const Slot& s, int64_t at) {
        for (const Checkpoint& m : s.marks)
            if (m.pos == at) return m.state->data();
        return nullptr;   // resume() only goes to positions reach() found
    }

    static BlobPtr blob(size_t bytes) {
        auto b = std::make_shared<Blob>(bytes);
        return b->data() ? b : nullptr;
    }

    // A checkpoint of the current conversation at `at`, its state still to be written.  Null: there is one there,
    // or no memory for it.
    BlobPtr mark(int64_t at) {
        Slot& s = cur();
        if (!s.marks.empty() && s.marks.back().pos == at) return nullptr;
        const BlobPtr b = blob(e_.state_bytes());
        if (!b) return nullptr;
        s.marks.push_back({at, b});
        while (s.marks.size() > (size_t) o_.checkpoints) {
            std::vector<int64_t> pos;
            for (const Checkpoint& m : s.marks) pos.push_back(m.pos);
            s.marks.erase(s.marks.begin() + (std::ptrdiff_t) drop_of(pos));
        }
        return b;
    }

    // Of more checkpoints than are kept, the one that goes: the one whose neighbours are closest together.  The
    // last two stay, the end of this prompt and of the one before being where a client most often comes back to.
    static size_t drop_of(const std::vector<int64_t>& pos) {
        size_t drop = 0;
        int64_t gap = INT64_MAX;
        for (size_t k = 0; k + 2 < pos.size(); ++k) {
            const int64_t g = pos[k + 1] - (k ? pos[k - 1] : 0);
            if (g < gap) gap = g, drop = k;
        }
        return drop;
    }

    // Hot-cache variant: once a semantic prefix anchor `pin` (e.g. sys_at) is pinned, shed interior checkpoints
    // strictly inside the static [0, pin) prefix before sacrificing fresher post-pin checkpoints near the live tip.
    static size_t drop_of_hot(const std::vector<int64_t>& pos, int64_t pin) {
        if (pin > 0) {
            size_t drop = pos.size();
            int64_t gap = INT64_MAX;
            for (size_t k = 0; k + 2 < pos.size(); ++k) {
                if (pos[k] >= pin) break;
                const int64_t g = pos[k + 1] - (k ? pos[k - 1] : 0);
                if (g < gap) gap = g, drop = k;
            }
            if (drop < pos.size()) return drop;
        }
        return drop_of(pos);
    }

    static void cut(Slot& s, int64_t at) {
        if (at >= len(s)) return;
        s.seq.resize((size_t) at);
        s.prompt = std::min(s.prompt, at);
        while (!s.marks.empty() && s.marks.back().pos > at) s.marks.pop_back();
        while (!s.snaps.empty() && s.snaps.back().pos > at) s.snaps.pop_back();
        while (!s.rows.empty() && s.rows.back().first >= at) s.rows.pop_back();
        if (!s.rows.empty()) s.rows.back().keep = std::min(s.rows.back().keep, at - s.rows.back().first);
    }

    // The parked conversations and what they hold; a blob several of them share counts once.  The current one
    // counts when `with` is its slot.
    Held held(size_t with) const {
        Held h;
        std::vector<const Blob*> seen;
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot& s = slots_[i];
            if (s.seq.empty() || (i == cur_ && i != with)) continue;
            ++h.count;
            collect_blobs(s, seen);
        }
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (const Blob* b : seen) h.bytes += b->bytes();
        return h;
    }

    // Makes the parked conversations, the current one among them and `extra` bytes more, fit the budget and the
    // slots.  The least recently used goes first, whatever its length: a long conversation can lose its place to
    // several short ones.  Slot `keep` stays; it is the one about to take the engine, if any, so it frees a slot.
    // False: it does not fit, or the host cannot spare the bytes.
    bool fit(size_t extra, size_t keep) {
        const size_t budget = (size_t) o_.park_mib << 20;
        for (;;) {
            const Held h = held(cur_);
            const size_t leaving = keep < slots_.size() && keep != cur_ && !slots_[keep].seq.empty() ? 1 : 0;
            if (h.bytes + extra <= budget && h.count - leaving <= (size_t) o_.park_slots) break;
            size_t out = slots_.size();
            for (size_t i = 0; i < slots_.size(); ++i)
                if (i != cur_ && i != keep && !slots_[i].seq.empty() &&
                    (out == slots_.size() || slots_[i].stamp < slots_[out].stamp))
                    out = i;
            if (out == slots_.size()) return false;
            slots_[out] = Slot{};
            ++evictions_;
        }
        return extra == 0 || !o_.admit || o_.admit(extra);
    }

    // Copies the rows the current conversation still has only in the engine, up to its last checkpoint; what it
    // read after that cannot be come back to and is dropped.  False: it cannot be kept.
    bool park(size_t keep) {
        Slot& s = cur();
        if (o_.park_mib < 1 || o_.park_slots < 1 || s.marks.empty() || s.marks.back().pos < o_.park_min) return false;
        const Clock::time_point t0 = Clock::now();
        const int64_t end = s.marks.back().pos, from = saved(s);
        cut(s, end);
        const size_t bytes = (size_t) (end - from) * e_.row_bytes();
        const BlobPtr b = from < end && fit(bytes, keep) ? blob(bytes) : nullptr;
        if (from < end && !b) {
            say("skipped", end, t0, slots_.size());
            return false;
        }
        if (b) {
            e_.save_rows(from, end - from, b->data());
            s.rows.push_back({b, from, end - from, end - from});
        }
        if (!fit(0, keep)) return false;
        say("parked", end, t0, cur_);
        return true;
    }

    // An empty slot other than `keep`.
    size_t take(size_t keep) {
        size_t out = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (i == keep) continue;
            if (slots_[i].seq.empty()) return i;
            if (out == slots_.size() || slots_[i].stamp < slots_[out].stamp) out = i;
        }
        slots_[out] = Slot{};   // park() left one empty, so this is not reached
        ++evictions_;
        return out;
    }

    // Slot i takes the engine, with its rows before `need` back in the caches.  The conversation that leaves is
    // parked if it can be.  Rows that both have from the same blob are in the caches already.
    void enter(size_t i, int64_t need) {
        const Clock::time_point t0 = Clock::now();
        const Slot& in = slots_[i];
        int64_t have = 0;
        for (size_t k = 0; k < in.rows.size() && k < cur().rows.size() && in.rows[k].blob == cur().rows[k].blob; ++k) {
            have = in.rows[k].first + std::min(in.rows[k].keep, cur().rows[k].keep);
            if (in.rows[k].keep != cur().rows[k].keep) break;
        }
        if (cur().seq.empty() || cur().has_vision || !park(i)) slots_[cur_] = Slot{};
        for (const Rows& r : in.rows) {
            const int64_t a = std::max(r.first, have), b = std::min(r.first + r.keep, need);
            if (a < b) e_.load_rows(r.blob->data(), r.first, r.count, a, b - a);
        }
        cur_ = i;
        say("restored", need, t0, slots_.size());
    }

    // Parks the current conversation whole and goes on at `at` in a copy of its start, which shares what is
    // saved.  False: it cannot be parked.
    bool branch(int64_t at) {
        const size_t from = cur_;
        if (!park(slots_.size())) return false;
        const size_t to = take(from);
        const Slot& a = slots_[from];
        Slot b;
        b.seq.assign(a.seq.begin(), a.seq.begin() + (std::ptrdiff_t) at);
        b.prompt = std::min(a.prompt, at);
        for (const Checkpoint& m : a.marks)
            if (m.pos <= at) b.marks.push_back(m);
        for (const Rows& r : a.rows)
            if (r.first < at) b.rows.push_back({r.blob, r.first, r.count, std::min(r.keep, at - r.first)});
        slots_[to] = std::move(b);
        cur_ = to;
        return true;
    }

    // ---- Hot Session Cache & Exact SWA State Replay path (opt-in, requires HasSwaDelta<E>) ----

    static bool swa_covers(const ConvSnap& sn, int64_t hist) {
        if (sn.pos < 1 || !sn.conv) return false;
        const int64_t need_lo = std::max<int64_t>(0, sn.pos - hist);
        int64_t covered = need_lo;
        for (const SwaSlice& sl : sn.swa) {
            if (!sl.blob || sl.n <= 0 || sl.first < sl.from || sl.first + sl.n > sl.from + sl.count) return false;
            if (sl.first > covered) return false;
            if (sl.first + sl.n > covered) covered = sl.first + sl.n;
        }
        return covered >= sn.pos;
    }

    int64_t reach_hot(size_t i, int64_t same, int64_t n) const {
        if constexpr (HasSwaDelta<E>) {
            const Slot& s = slots_[i];
            if (i == cur_ && same > 0 && same == len(s) && n > same) return same;
            const int64_t lim = std::min(same, n - 1);
            const int64_t hist = e_.swa_history_rows();
            int64_t at = 0;
            for (const ConvSnap& sn : s.snaps)
                if (sn.pos <= lim && sn.pos > at && swa_covers(sn, hist)) at = sn.pos;
            return at;
        } else {
            return 0;
        }
    }

    static const ConvSnap* snap_at(const Slot& s, int64_t at) {
        for (const ConvSnap& sn : s.snaps)
            if (sn.pos == at) return &sn;
        return nullptr;
    }

    static void insert_snap_hot(std::vector<ConvSnap>& snaps, ConvSnap sn) {
        for (ConvSnap& ex : snaps) {
            if (ex.pos == sn.pos) {
                ex.pinned = ex.pinned || sn.pinned;
                ex.is_delta = ex.is_delta && sn.is_delta;
                return;
            }
        }
        auto it = std::upper_bound(snaps.begin(), snaps.end(), sn.pos,
                                   [](int64_t p, const ConvSnap& m) { return p < m.pos; });
        snaps.insert(it, std::move(sn));
    }

    // Collects all existing SwaSlices from `snaps` (and optional `fallback`) that intersect [need_lo, pos),
    // reuses them without copying, and captures only the missing tail from VRAM into a new immutable SwaSeg.
    bool capture_snap_hot(std::vector<ConvSnap>& target, const std::vector<ConvSnap>* fallback,
                          int64_t pos, bool pinned, bool is_delta) {
        if constexpr (HasSwaDelta<E>) {
            if (pos < 1 || e_.pos() != pos) return false;
            for (ConvSnap& ex : target) {
                if (ex.pos == pos) {
                    ex.pinned = ex.pinned || pinned;
                    ex.is_delta = ex.is_delta && is_delta;
                    return true;
                }
            }
            const int64_t hist = e_.swa_history_rows();
            const int64_t need_lo = std::max<int64_t>(0, pos - hist);
            const auto live = e_.resident_swa_rows();

            // Gather reusable slices from the most recent prior snapshot (monotone capture order).
            std::vector<SwaSlice> slices;
            const auto reuse_from = [&](const std::vector<ConvSnap>& src) {
                for (auto it = src.rbegin(); it != src.rend(); ++it) {
                    if (it->pos > pos) continue;
                    if (it->pos <= need_lo) break;
                    for (const SwaSlice& sl : it->swa) {
                        const int64_t a = std::max(sl.first, need_lo);
                        const int64_t b = std::min(sl.first + sl.n, pos);
                        if (a >= b) continue;
                        const int64_t cur_end = slices.empty() ? need_lo : slices.back().first + slices.back().n;
                        const int64_t take_a = std::max(a, cur_end);
                        if (take_a < b) slices.push_back({sl.blob, sl.from, sl.count, take_a, b - take_a});
                    }
                    break;
                }
            };
            if (!target.empty() && target.back().pos <= pos) reuse_from(target);
            else if (fallback && !fallback->empty()) reuse_from(*fallback);

            // Verify reused slices start at need_lo without gaps; otherwise rebuild full window from live ring.
            int64_t covered = need_lo;
            bool contiguous = true;
            for (const SwaSlice& sl : slices) {
                if (sl.first != covered) {
                    contiguous = false;
                    break;
                }
                covered = sl.first + sl.n;
            }
            if (!contiguous) {
                slices.clear();
                covered = need_lo;
            }
            if (covered < pos) {
                if (covered < live.lo || pos > live.hi || pos - covered > e_.swa_ring_rows()) return false;
                const int64_t n_new = pos - covered;
                const BlobPtr swa_b = blob((size_t) n_new * e_.swa_row_bytes());
                if (!swa_b) return false;
                e_.save_swa_rows(covered, n_new, swa_b->data());
                slices.push_back({swa_b, covered, n_new, covered, n_new});
            }
            const BlobPtr conv_b = blob(e_.conv_state_bytes());
            if (!conv_b) return false;
            e_.save_conv_state(conv_b->data());
            ConvSnap sn{pos, pinned, is_delta, conv_b, std::move(slices)};
            if (!swa_covers(sn, hist)) return false;
            insert_snap_hot(target, std::move(sn));
            if (&target == &cur().snaps) prune_active_snaps_hot(cur());
            return true;
        } else {
            (void) target; (void) fallback; (void) pos; (void) pinned; (void) is_delta;
            return false;
        }
    }

    void prune_active_snaps_hot(Slot& s) {
        // Bound non-pinned coarse checkpoints via drop_of(), and bound rolling decode delta snapshots behind tip.
        const int64_t tip = len(s);
        if (o_.active_delta_rows > 0) {
            s.snaps.erase(std::remove_if(s.snaps.begin(), s.snaps.end(), [&](const ConvSnap& m) {
                return m.is_delta && !m.pinned && tip - m.pos > o_.active_delta_rows;
            }), s.snaps.end());
        }
        for (;;) {
            std::vector<int64_t> coarse;
            int64_t pinned_at = pin_;
            for (const ConvSnap& m : s.snaps) {
                if (m.pinned && m.pos > pinned_at) pinned_at = m.pos;
                if (!m.pinned && !m.is_delta) coarse.push_back(m.pos);
            }
            if (coarse.size() <= (size_t) std::max(o_.checkpoints, 2)) break;
            const int64_t victim = coarse[drop_of_hot(coarse, pinned_at)];
            s.snaps.erase(std::remove_if(s.snaps.begin(), s.snaps.end(),
                                         [&](const ConvSnap& m) { return !m.pinned && !m.is_delta && m.pos == victim; }),
                          s.snaps.end());
        }
    }

    void restore_snap_hot(const ConvSnap& sn, bool same_slot) {
        if constexpr (HasSwaDelta<E>) {
            const auto live = same_slot ? e_.resident_swa_rows() : decltype(e_.resident_swa_rows()){0, 0};
            const bool can_skip = same_slot && live.lo < live.hi;
            bool skipped_any = false;
            for (const SwaSlice& sl : sn.swa) {
                const int64_t a = sl.first, b = sl.first + sl.n;
                if (can_skip && a < live.hi && b > live.lo) {
                    skipped_any = true;
                    // Upload only the sub-intervals NOT already intact in the VRAM SWA ring!
                    if (a < live.lo)
                        e_.load_swa_rows(sl.blob->data(), sl.from, sl.count, a, live.lo - a, sn.pos);
                    if (b > live.hi)
                        e_.load_swa_rows(sl.blob->data(), sl.from, sl.count, live.hi, b - live.hi, sn.pos);
                } else {
                    e_.load_swa_rows(sl.blob->data(), sl.from, sl.count, a, sl.n, sn.pos);
                }
            }
            if (skipped_any) ++swa_fast_rollbacks_;
            e_.load_conv_state(sn.conv->data(), sn.pos);
        } else {
            (void) sn; (void) same_slot;
        }
    }

    bool trim_slot_snaps_hot(Slot& s, int64_t keep_tail_rows) {
        if (s.snaps.empty()) return false;
        const int64_t tip = s.snaps.back().pos;
        const size_t before = s.snaps.size();
        s.snaps.erase(std::remove_if(s.snaps.begin(), s.snaps.end(), [&](const ConvSnap& m) {
            if (m.pos == tip || m.pinned) return false;
            if (keep_tail_rows > 0 && tip - m.pos <= keep_tail_rows) return false;
            return true;
        }), s.snaps.end());
        if (s.snaps.size() < before) {
            ++component_trims_;
            return true;
        }
        return false;
    }

    // Component-level tiered eviction:
    //   Tier 1: trim non-pinned interior snaps of parked slots (keeping tip + pins + parked_tail_rows).
    //   Tier 2: trim pre-tip delta tail of parked slots (keeping tip + pins + all global KV!).
    //   Tier 3: evict whole parked slot minimizing recompute_tokens / age^1.2.
    bool fit_hot(size_t extra, size_t keep) {
        const size_t budget = (size_t) o_.park_mib << 20;
        const auto over_bytes = [&] { return held(cur_).bytes + extra > budget; };
        const auto over_slots = [&] {
            const Held h = held(cur_);
            const size_t leaving = keep < slots_.size() && keep != cur_ && !slots_[keep].seq.empty() ? 1 : 0;
            return h.count - leaving > (size_t) o_.park_slots;
        };
        if (over_bytes()) {
            // Order parked candidates oldest-first for component trimming.
            std::vector<size_t> order;
            for (size_t i = 0; i < slots_.size(); ++i)
                if (i != cur_ && i != keep && !slots_[i].seq.empty()) order.push_back(i);
            std::sort(order.begin(), order.end(),
                      [&](size_t a, size_t b) { return slots_[a].stamp < slots_[b].stamp; });
            // Tier 1: trim interior checkpoints while keeping parked_tail_rows behind tip.
            for (const size_t i : order) {
                if (!over_bytes()) break;
                trim_slot_snaps_hot(slots_[i], o_.parked_tail_rows);
            }
            // Tier 2: trim remaining pre-tip tails (still keeping every slot's tip + pinned anchors + global KV!).
            for (const size_t i : order) {
                if (!over_bytes()) break;
                trim_slot_snaps_hot(slots_[i], 0);
            }
            // Also trim current slot's redundant interior snaps (keeping its tip + pins) before evicting whole sessions.
            if (over_bytes()) trim_slot_snaps_hot(slots_[cur_], 0);
        }
        // Tier 3: value-aware whole-slot eviction.
        while (over_bytes() || over_slots()) {
            size_t out = slots_.size();
            double worst = 0.0;
            for (size_t i = 0; i < slots_.size(); ++i) {
                if (i == cur_ || i == keep || slots_[i].seq.empty()) continue;
                const double age = (double) std::max<uint64_t>(1, clock_ - slots_[i].stamp);
                const double val = (double) std::max<int64_t>(1, len(slots_[i])) / std::pow(age, 1.2);
                if (out == slots_.size() || val < worst || (val == worst && slots_[i].stamp < slots_[out].stamp)) {
                    worst = val;
                    out = i;
                }
            }
            if (out == slots_.size()) return false;
            slots_[out] = Slot{};
            ++evictions_;
        }
        return extra == 0 || !o_.admit || o_.admit(extra);
    }

    // Full-Tip Parking: captures the committed tip (len(s)) as an incremental SWA+conv checkpoint, then saves unsaved
    // global KV rows in stage_-friendly chunks split at pinned prefix anchors.
    bool park_hot(size_t keep) {
        if constexpr (HasSwaDelta<E>) {
            Slot& s = cur();
            if (o_.park_mib < 1 || o_.park_slots < 1 || s.has_vision) return false;
            const int64_t tip = len(s);
            if (tip < o_.park_min) return false;
            const Clock::time_point t0 = Clock::now();
            // Capture the exact committed tip if the engine holds it and no snapshot exists there yet.
            int64_t end = tip;
            if (e_.pos() == tip) {
                if (!capture_snap_hot(s.snaps, nullptr, tip, false, false)) {
                    if (s.snaps.empty()) return false;
                    end = s.snaps.back().pos;
                    if (end < o_.park_min) return false;
                    cut(s, end);
                }
            } else {
                if (s.snaps.empty()) return false;
                end = s.snaps.back().pos;
                if (end < o_.park_min) return false;
                cut(s, end);
            }
            const int64_t from = saved(s);
            const size_t total_bytes = from < end ? (size_t) (end - from) * e_.row_bytes() : 0;
            if (!fit_hot(total_bytes, keep)) {
                say("skipped", end, t0, slots_.size());
                return false;
            }
            // Split global KV rows at pinned anchors and at kHotRowChunk so branches drop dead tails cleanly
            // and every chunk fits inside Inkling2::stage_ for single-DMA packed transfer.
            std::vector<int64_t> splits;
            for (const ConvSnap& sn : s.snaps)
                if (sn.pinned && sn.pos > from && sn.pos < end) splits.push_back(sn.pos);
            splits.push_back(end);
            int64_t cur_pos = from;
            for (const int64_t stop : splits) {
                while (cur_pos < stop) {
                    const int64_t cnt = std::min<int64_t>(kHotRowChunk, stop - cur_pos);
                    const BlobPtr b = blob((size_t) cnt * e_.row_bytes());
                    if (!b) {
                        say("skipped", end, t0, slots_.size());
                        return false;
                    }
                    e_.save_rows(cur_pos, cnt, b->data());
                    s.rows.push_back({b, cur_pos, cnt, cnt});
                    cur_pos += cnt;
                }
            }
            if (!fit_hot(0, keep)) return false;
            say("parked", end, t0, cur_);
            return true;
        } else {
            (void) keep;
            return false;
        }
    }

    void enter_hot(size_t i, int64_t need) {
        const Slot& in = slots_[i];
        int64_t have = 0;
        for (size_t k = 0; k < in.rows.size() && k < cur().rows.size() && in.rows[k].blob == cur().rows[k].blob; ++k) {
            have = in.rows[k].first + std::min(in.rows[k].keep, cur().rows[k].keep);
            if (in.rows[k].keep != cur().rows[k].keep) break;
        }
        if (cur().seq.empty() || cur().has_vision || !park_hot(i)) slots_[cur_] = Slot{};
        const Clock::time_point t0 = Clock::now();
        for (const Rows& r : in.rows) {
            const int64_t a = std::max(r.first, have), b = std::min(r.first + r.keep, need);
            if (a < b) e_.load_rows(r.blob->data(), r.first, r.count, a, b - a);
        }
        cur_ = i;
        say("restored", need, t0, slots_.size());
    }

    // Branches from `cur_` at `at`.  If `already_parked` is true (we just entered a parked slot to branch from its
    // prefix), its immutable parked rows/snaps are already saved in RAM so we clone references without re-parking VRAM.
    bool branch_hot(int64_t at, bool already_parked) {
        const size_t from = cur_;
        if (!already_parked) {
            if (!park_hot(slots_.size())) return false;
        } else {
            if (o_.park_mib < 1 || o_.park_slots < 1 || !fit_hot(0, from)) return false;
        }
        const size_t to = take(from);
        const Slot& a = slots_[from];
        Slot b;
        b.seq.assign(a.seq.begin(), a.seq.begin() + (std::ptrdiff_t) at);
        b.prompt = std::min(a.prompt, at);
        for (const ConvSnap& sn : a.snaps)
            if (sn.pos <= at) b.snaps.push_back(sn);
        for (const Rows& r : a.rows)
            if (r.first < at) b.rows.push_back({r.blob, r.first, r.count, std::min(r.keep, at - r.first)});
        slots_[to] = std::move(b);
        cur_ = to;
        return true;
    }

    int64_t resume_hot(const std::vector<int32_t>& ids, bool has_vision) {
        const int64_t n = (int64_t) ids.size();
        turn_ = -1;
        pin_ = -1;
        size_t best = cur_;
        int64_t at = 0, same = 0;
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (has_vision || slots_[i].has_vision) continue;
            const int64_t c = common(slots_[i].seq, ids), r = reach_hot(i, c, n);
            if (r > at || (r == at && i == cur_)) best = i, at = r, same = c;
        }
        ++clock_;
        if (at == 0) {
            if (!cur().seq.empty() && !cur().has_vision && park_hot(slots_.size())) cur_ = take(cur_);
            slots_[cur_] = Slot{};
            slots_[cur_].has_vision = has_vision;
            e_.reset();
        } else if (best != cur_ || at < len(cur())) {
            const bool same_slot = best == cur_;
            // Hold strong references to the target snapshot before parking/trimming/branching can touch any slot.
            const ConvSnap* found = snap_at(slots_[best], at);
            const ConvSnap target_snap = *found;
            if (!same_slot) enter_hot(best, at);
            const bool already_parked = !same_slot && saved(cur()) >= len(cur());
            // Keep the existing conversation parked if the incoming prompt diverged before its prompt end
            // (or before its full tip when switching/branching from a longer conversation).
            const bool should_branch = same < cur().prompt || (!same_slot && at < len(cur()));
            if (!should_branch || !branch_hot(at, already_parked)) cut(cur(), at);
            insert_snap_hot(cur().snaps, target_snap);
            restore_snap_hot(target_snap, same_slot);
        }
        cur().stamp = clock_;
        return at;
    }

    // `with` is the current slot when its conversation is the one just parked.
    void say(const char* what, int64_t tokens, Clock::time_point t0, size_t with) const {
        if (!o_.log) return;
        const Held h = held(with);
        std::fprintf(o_.log,
                     "strata-poly: conversation cache: %s %lld tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu\n",
                     what, (long long) tokens, std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                     h.count, h.bytes, evictions_);
    }

    E& e_;
    CacheOptions o_;
    std::vector<Slot> slots_;   // the current conversation and the parked ones
    size_t cur_ = 0;
    uint64_t clock_ = 0;
    size_t evictions_ = 0;
    size_t component_trims_ = 0;
    size_t swa_fast_rollbacks_ = 0;
    std::vector<Checkpoint> due_;   // inside()'s checkpoints, until read() makes them the conversation's
    std::vector<int64_t> after_;    // where the conversation's checkpoints are once that read is done
    std::vector<int64_t> due_hot_;  // inside()'s staying mark positions under hot_cache
    std::vector<ConvSnap> pending_snaps_; // captured during read() under hot_cache, published in read()
    int64_t turn_ = -1;             // turn_at()'s place in the prompt being read; -1: none
    int64_t pin_ = -1;              // pin_at()'s shared-prefix anchor; -1: none
};

}  // namespace strata::poly
