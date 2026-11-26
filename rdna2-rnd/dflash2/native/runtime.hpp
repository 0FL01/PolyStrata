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

// Single sequence. Inputs to append are four separate device matrices on device 0.
// An initial nonzero absolute position is allowed only with a full >=2048-row warmup.
// The caller keeps any borrowed head alive, and makes the head/taps ready before use.
class Runtime {
public:
    Runtime(const char* draft_path, const char* target_path, const void* shared_head = nullptr);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void reset();
    void append(int64_t pos, int rows, const std::array<const float*, 4>& taps);
    void propose(int64_t pos, int32_t anchor, int count, Proposal& out,
                 bool read_hidden = false, bool details = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dflash2
