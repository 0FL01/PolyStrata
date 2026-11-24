// Host-only checksum helper; uses the actual STRSESS hash, not standard xxHash64.
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

static void hash_range(int fd, off_t offset, uint64_t bytes) {
    strata::core::SessionHasher hash;
    std::vector<unsigned char> buf(1 << 20);
    while (bytes) {
        const ssize_t n = pread(fd, buf.data(), std::min<uint64_t>(buf.size(), bytes), offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            std::fputs("short or failed snapshot read\n", stderr);
            std::exit(1);
        }
        hash.update(buf.data(), n);
        offset += n;
        bytes -= n;
    }
    std::printf("%016llx\n", (unsigned long long) hash.digest());
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;  // inherited descriptor, payload size from the bounded parser
    const int fd = std::atoi(argv[1]);
    hash_range(fd, 0, 56);
    hash_range(fd, 64, std::strtoull(argv[2], nullptr, 10));
}
