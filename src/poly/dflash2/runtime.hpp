#pragma once

#include <array>
#include <cstdint>
#include <memory>

namespace dflash2 {

inline constexpr int hidden = 2304, block = 8, proposals = block - 1, top_k = 16;
inline constexpr int vocab = 200058, window = 2048, max_positions = 1048576;

struct Candidate {
    int32_t id;
    float logit;
};

struct Proposal {
    std::array<int32_t, proposals> ids{};
    std::array<std::array<Candidate, top_k>, proposals> candidates{};
    std::array<std::array<float, top_k>, proposals> scores{}, probs{};
    // Filled only when requested, for comparison with the independent CPU reference.
    std::array<float, proposals * hidden> hidden_states{};
};

// Native Inkling 2 Mini DFlash2 runtime, ported from rdna2-rnd/dflash2/native
// at fa000e5c with its opt/attention_exact.cuh overlay. One handle per sequence,
// on RDNA2 device 0; model paths are local single-file GGUFs. Calls throw on
// invalid inputs or device errors. A device failure poisons the handle: destroy it.
// Not thread-safe. Construction and use select device 0 on the calling thread.
//
// shared_head, if non-null, is the target's raw resident Q6_K output.weight;
// the caller owns it, keeps it alive, and finishes writing it before construction.
// A null head loads a private copy from target_path.
class Runtime {
public:
    Runtime(const char* draft_path, const char* target_path, const void* shared_head = nullptr);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // Discard committed history and resume collecting at an absolute position.
    // This does not recover a poisoned handle. A nonzero start normally requires
    // a full rebuilt window; the optional partial proposer uses only genuine rows.
    void reset(int64_t next_pos = 0);
    bool usable() const;
    bool ready(int64_t pos, bool full_window) const;
    // Four contiguous device FP32 [rows, hidden] planes, ordered target visits
    // 5, 23, 41, final norm 48. The producer must have finished writing them.
    // Positions must be contiguous from reset's next_pos. For compatibility,
    // the first append can also start nonzero when it provides >= window rows.
    // Returns after the planes can be reused.
    void append(int64_t pos, int rows, const std::array<const float*, 4>& taps);
    // pos is the next uncommitted position, anchor is its target token. Always
    // computes the full bidirectional block; count in [1, 7] limits host results.
    // No draft KV is committed by this call. out.ids contains greedy proposals.
    void propose(int64_t pos, int32_t anchor, int count, Proposal& out,
                 bool read_hidden = false, bool details = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dflash2
