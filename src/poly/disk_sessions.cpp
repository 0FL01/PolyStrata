#include "disk_sessions.hpp"

#include "conversations.hpp"
#include "runtime.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace strata::poly {
namespace {

using namespace strata::core;

// Inkling layouts within the STRSESS v1 container, with their own configuration identities. The position is
// exactly live.ids.size() (only tokens actually fed, not the last token emitted). live.gdn is the opaque
// save_state() image: convolution state followed by the chronological SWA tails. Each K/V record's k part
// is a save_rows() block (all full-attention planes of those rows, in checkpoint order). A block fits one host
// segment, so restore does not need to reassemble it before calling load_rows(). DSpark adds the last mixed
// context and its SWA tails to save_state(), and its full-attention plane to save_rows(). No guesses are stored.
constexpr int64_t kInklingTag = 0x494e4b4c494e4732;   // INKLING2
constexpr int64_t kLayoutVersion = 1;
constexpr int64_t kDsparkLayoutVersion = 2;
constexpr uint64_t kDiskFloor = uint64_t(4096) << 20;

struct Layout {
    int64_t context, vocab, width, layers, kv_format;
    size_t state_bytes, row_bytes;
    bool dspark;

    std::array<int64_t, 18> geometry() const {
        return {kInklingTag, dspark ? kDsparkLayoutVersion : kLayoutVersion, context, vocab, width, layers, kv_format,
                (int64_t) state_bytes, (int64_t) row_bytes, block_rows()};
    }
    int64_t block_rows() const { return (int64_t) (ConversationBuffer::segment_bytes / row_bytes); }
    int64_t blocks(int64_t n) const { return (n + block_rows() - 1) / block_rows(); }
    int64_t rows_at(int64_t pos, int64_t block) const { return std::min(block_rows(), pos - block * block_rows()); }
    size_t bytes_at(int64_t pos, int64_t block) const { return (size_t) rows_at(pos, block) * row_bytes; }
    bool valid_tokens(const std::vector<int32_t>& ids) const {
        return !ids.empty() && ids.size() <= (size_t) context &&
               std::all_of(ids.begin(), ids.end(), [&](int32_t t) { return t >= 0 && t < vocab; });
    }
    SessionReadLimits limits() const {
        SessionReadLimits l;
        l.max_tokens = (uint64_t) context;
        l.max_checkpoints = 0;
        l.max_kv_layers = (uint64_t) blocks(context);
        l.geometry = geometry();
        l.layer_range = {{0, layers}};
        l.max_state_bytes = {state_bytes, 0, 0, 0, 0};
        for (int64_t i = 0; i < blocks(context); ++i) l.max_kv_bytes.push_back({bytes_at(context, i), 0, 0, 0, 0});
        return l;
    }
    // The container has already checked identity, geometry, layer range, global bounds and payload checksum.
    // Check the specific lengths/shape against the actual token list before loading even one row on the GPU.
    bool validate(const SavedConversation& image) const {
        const auto& live = image.live;
        const int64_t pos = (int64_t) live.ids.size();
        if (!valid_tokens(live.ids) || live.gdn.size() != state_bytes || !live.imgs.empty() || live.used || image.cvec ||
            image.kv.size() != (size_t) blocks(pos))
            return false;
        for (int64_t i = 0; i < blocks(pos); ++i) {
            const auto& k = image.kv[(size_t) i];
            if (k.format != kv_format || k.cells != rows_at(pos, i) || k.heads || k.head_dim || k.page_size ||
                k.pooled_rows || k.idx_dim || k.k.size() != bytes_at(pos, i))
                return false;
        }
        return true;
    }
};

}  // namespace

DiskSessions::DiskSessions(Engine& e, bool enabled, uint64_t ram_floor)
    : engine_(e), ram_floor_(ram_floor), supported_(false) {
    if (!enabled) return;
    const auto* arch = e.gg_->get("general.architecture");
    const bool known_draft = !e.draft_ || e.uses_dspark();
    supported_ = arch && is_inkling2_architecture(arch->s) && known_draft && !e.split_ && !e.next_ &&
                 e.sequences() == 1 && e.lo_ == 0 && e.hi_ == e.n_layer_ && e.row_bytes() > 0 &&
                 e.row_bytes() <= ConversationBuffer::segment_bytes;
    if (!supported_)
        std::fputs("strata-poly: --disk-sessions requires single-sequence Inkling-2-Mini on one card\n", stderr);
}

bool DiskSessions::run(bool save, const std::string& path, Conversations<Engine>& kept) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const auto say_error = [&](std::string why, SessionError error = SessionError::invalid, bool published = false) {
        std::replace(why.begin(), why.end(), '\r', ' ');
        std::replace(why.begin(), why.end(), '\n', ' ');
        std::printf("SERR %s %d %s\n", session_error_name(error), (int) published, why.c_str());
        std::fflush(stdout);
        return true;
    };
    if (!supported_) return say_error("strata-poly keeps no disk sessions");
    if (path.empty() || path.find_first_of("\r\n") != std::string::npos || path.find('\0') != std::string::npos)
        return say_error("invalid session path");

    const auto phase = [](const char* name, uint64_t bytes) {
        std::printf("SWAIT %s %lld\n", name, (long long) session_phase_limit_s(bytes));
        std::fflush(stdout);
    };
    const auto progress = [](uint64_t done, uint64_t total) {
        std::printf("SESSION %llu %llu\n", (unsigned long long) done, (unsigned long long) total);
        std::fflush(stdout);
    };
    const auto admit = [&](uint64_t need, std::string& why) {
        if (conversation_memory_admit(available_host_bytes(), need, ram_floor_)) return true;
        why = "not enough RAM for the session (" + std::to_string(need >> 20) + " MiB plus a floor of " +
              std::to_string(ram_floor_ >> 20) + " MiB needed)";
        return false;
    };
    const auto fatal = [](const char* why) {
        std::fprintf(stderr, "strata-poly: session restore transfer failed: %s\n", why);
        std::puts("FATAL restoring the session failed after the device state may have changed");
        std::fflush(stdout);
        return false;
    };

    Engine& e = engine_;
    const Layout layout{e.max_context(), e.n_vocab(), e.n_embd(), e.n_layer_, (int) e.opt_.kv_format,
                        e.state_bytes(), e.row_bytes(), e.uses_dspark()};
    const auto& ids = kept.seq();
    int64_t pos = e.pos();
    if (save && (pos != (int64_t) ids.size() || !layout.valid_tokens(ids)))
        return say_error("no complete text session to save");

    size_t bytes = 0;
    bool transferring = false;
    try {
        SessionFileIdentity id;
        std::string why;
        // Sample the shards the engine actually opened. A new file installed at the same path must not
        // have its identity attached to state computed with the old mapping.
        const auto shards = e.gg_->views();
        constexpr size_t sample = 1u << 20;
        SessionIdentityBuilder model(0x5354524154414d44ull);
        model.u64("Inkling shards", shards.size());
        for (const auto& [data, size] : shards) {
            if (!data || !size) return say_error("cannot fingerprint an unmapped Inkling shard", SessionError::io);
            phase("fingerprint", 2 * sample);
            const size_t head = std::min(size, sample), tail = std::min(size - head, sample);
            model.u64("size", size);
            model.bytes("head", data, head);
            if (tail) model.bytes("tail", data + size - tail, tail);
        }
        if (layout.dspark) {
            if (!dspark_identity_) {
                SessionIdentityBuilder draft(0);
                const auto files = e.dg_->views();
                draft.u64("DSpark shards", files.size());
                for (const auto& [data, size] : files) {
                    if (!data || !size) return say_error("cannot fingerprint an unmapped DSpark shard", SessionError::io);
                    phase("fingerprint", size);
                    draft.bytes("shard", data, size);
                }
                dspark_identity_ = draft.digest();
            }
            model.u64("DSpark", *dspark_identity_);
        }
        id.model = model.digest();
        SessionIdentityBuilder config(0);
        config.str("engine", "strata-poly " STRATA_VERSION " HIP RDNA2");
        const auto geometry = layout.geometry();
        config.bytes("Inkling snapshot geometry", geometry.data(), geometry.size() * sizeof(geometry[0]));
        config.i64("prompt batch", e.max_rows());
        id.config = config.digest();
        SessionStatus status;
        if (save) {
            const size_t block_bytes = layout.bytes_at(pos, 0);
            const uint64_t need = session_save_peak_bytes(nullptr, {layout.state_bytes + block_bytes, ids.size(), 0,
                                                                     (uint64_t) layout.blocks(pos)});
            if (!admit(need, why)) return say_error(why, SessionError::memory);
            SavedConversation image;
            image.geometry = geometry;
            image.layer_hi = layout.layers;
            image.cvec = false;
            image.live.ids = ids;
            phase("capture", layout.state_bytes);
            image.live.gdn.resize(layout.state_bytes);
            e.save_state(image.live.gdn.data());

            auto block = std::make_unique_for_overwrite<uint8_t[]>(block_bytes);
            std::vector<SessionKvSource> sources;
            sources.reserve((size_t) layout.blocks(pos));
            for (int64_t i = 0; i < layout.blocks(pos); ++i) {
                auto& k = sources.emplace_back();
                k.format = (int) layout.kv_format;
                k.cells = layout.rows_at(pos, i);
                k.sizes[0] = layout.bytes_at(pos, i);
                k.read = [&, first = i * layout.block_rows(), rows = k.cells](size_t, size_t offset, void* dst, size_t n) {
                    // The file writer can split a source at an I/O block boundary. Capture it only once.
                    if (offset == 0) e.save_rows(first, rows, block.get());
                    std::memcpy(dst, block.get() + offset, n);
                    return true;
                };
            }
            SessionWriteOptions wo;
            wo.min_free_bytes = kDiskFloor;
            wo.progress = progress;
            wo.phase = phase;
            if (!session_file_write(path, image, sources, id, bytes, why, wo, &status))
                return say_error(why, status.error, status.published);
            if (status.dir_flush_unsupported)
                std::fputs("strata-poly: session saved; this filesystem does not flush the new name\n", stderr);
        } else {
            auto limits = layout.limits();
            limits.progress = progress;
            limits.admit = admit;
            SavedConversation image;
            if (!session_file_read(path, id, image, bytes, why, limits, &status)) return say_error(why, status.error);
            if (!layout.validate(image)) return say_error("session file: invalid Inkling state, token list or KV rows");

            pos = (int64_t) image.live.ids.size();
            phase("transfer", bytes);
            transferring = true;
            // Loading the deepest block first lets the existing KV grow path cover the whole extent in one go.
            for (int64_t i = layout.blocks(pos); i-- > 0;) {
                const int64_t first = i * layout.block_rows(), rows = layout.rows_at(pos, i);
                const auto& k = image.kv[(size_t) i].k;
                k.visit(0, k.size(), [&](const uint8_t* src, size_t n, size_t) {
                    if (n != k.size()) throw std::logic_error("Inkling snapshot block is not contiguous");
                    e.load_rows(src, first, rows, first, rows);
                    return true;
                });
            }
            e.load_state(image.live.gdn.data(), pos);
            kept.adopt(std::move(image.live.ids));
        }
    } catch (const std::bad_alloc&) {
        return transferring ? fatal("out of memory") : say_error("not enough RAM for the session", SessionError::memory);
    } catch (const std::exception& ex) {
        return transferring ? fatal(ex.what()) : say_error(ex.what(), SessionError::io);
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::printf("%s %lld %zu %.1f\n", save ? "SAVED" : "RESTORED", (long long) pos, bytes, ms);
    std::fflush(stdout);
    return true;
}

}  // namespace strata::poly
