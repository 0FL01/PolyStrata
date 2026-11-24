// Inkling's opt-in disk snapshots over the existing SAVE/RESTORE protocol.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace strata::poly {

class Engine;
template<class E> class Conversations;

class DiskSessions {
public:
    DiskSessions(Engine& engine, bool enabled, uint64_t ram_floor);
    bool supported() const { return supported_; }

    // Prints the reply; false means restore may have changed device state and the process must exit.
    bool run(bool save, const std::string& path, Conversations<Engine>& kept);

private:
    Engine& engine_;
    uint64_t ram_floor_;
    bool supported_;
    std::optional<uint64_t> dspark_identity_;   // The drafter mapped by this engine, computed on first use.
};

}  // namespace strata::poly
