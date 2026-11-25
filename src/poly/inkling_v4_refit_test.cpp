// Standalone CPU-only byte parity against the symmetric refit rule; no model or GPU needed.
// c++ -O2 -std=c++20 src/poly/inkling_v4_refit{,_test}.cpp -o inkling_v4_refit_test
#include "inkling_v4_refit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using strata::poly::codec::pack_refit;
using strata::poly::codec::pack_refit_scalar;

constexpr size_t kCodes = 1024, kPacked = 512, kScales = 64, kBlock = 32;
constexpr uint8_t kSentinel = 0x5b;

uint8_t nibble(uint8_t byte, int max) {
    const int x = byte < 128 ? byte : byte - 256;
    // The quotient is small; using double keeps every non-tie far enough away from the half-way point.
    return !max ? 0 : static_cast<int>(std::nearbyint(7.0 * x / max)) & 15;
}

bool compare(const char* what, const std::vector<uint8_t>& ref, const std::vector<uint8_t>& got) {
    const auto [a, b] = std::mismatch(ref.cbegin(), ref.cend(), got.cbegin());
    if (a == ref.cend()) return true;
    std::fprintf(stderr, "%s differs at byte %zu: expected %02x, found %02x\n", what,
                 size_t(a - ref.cbegin()), unsigned(*a), unsigned(*b));
    return false;
}

bool check_rows(const std::vector<uint8_t>& code, const std::vector<uint8_t>& scales,
                const std::vector<int>& maxima) {
    const size_t rows = code.size() / kCodes;
    // Use unaligned planes and guard bytes. An empty call must not touch either output.
    std::vector<uint8_t> src(code.size() + 2), src_scales(scales.size() + 2);
    std::copy(code.begin(), code.end(), src.begin() + 1);
    std::copy(scales.begin(), scales.end(), src_scales.begin() + 1);
    std::vector<uint8_t> packed(rows * kPacked + 2, kSentinel), refitted(rows * kScales + 2, kSentinel);
    auto exact = packed, exact_scales = refitted;
    for (size_t b = 0; b < maxima.size(); ++b) {
        for (size_t j = 0; j < kBlock; j += 2) {
            const size_t i = b * kBlock + j;
            exact[1 + i / 2] = nibble(code[i], maxima[b]) | (nibble(code[i + 1], maxima[b]) << 4);
        }
        _Float16 scale;
        std::memcpy(&scale, scales.data() + 2 * b, 2);
        // FP32 divide + multiply, followed by one FP16 rounding, including signed zero and overflow.
        const _Float16 refit = static_cast<_Float16>((float)scale * ((float)maxima[b] / 7.f));
        std::memcpy(exact_scales.data() + 1 + 2 * b, &refit, 2);
    }
    pack_refit_scalar(src.data() + 1, src_scales.data() + 1, rows, packed.data() + 1, refitted.data() + 1);
    if (!compare("scalar codes", exact, packed) || !compare("scalar scales", exact_scales, refitted)) return false;
    pack_refit(src.data() + 1, src_scales.data() + 1, rows, packed.data() + 1, refitted.data() + 1);
    return compare("dispatched codes", exact, packed) && compare("dispatched scales", exact_scales, refitted);
}

}  // namespace

int main() {
    pack_refit(nullptr, nullptr, 0, nullptr, nullptr);
    pack_refit_scalar(nullptr, nullptr, 0, nullptr, nullptr);

    // Cover every Q8 input code legal at each maximum, in both nibble positions. Mix fallbacks and fast blocks
    // inside a row. Also check -128 coexisting with 127, not just blocks without +/-127.
    constexpr size_t kMaxRows = 129 * 8;
    std::vector<uint8_t> code(kMaxRows * kCodes), scales(kMaxRows * kScales);
    std::vector<int> maxima(code.size() / kBlock);
    for (size_t b = 0; b < maxima.size(); ++b) {
        const int amax = b % 129, wanted = int(b / 129) - 128;
        const int x = std::clamp(wanted, -amax, std::min(amax, 127));
        maxima[b] = amax;
        uint8_t* p = code.data() + b * kBlock;
        for (size_t i = 0; i < kBlock; i += 2) {
            p[i] = static_cast<uint8_t>(x);
            p[i + 1] = static_cast<uint8_t>(std::min(-x, 127));
        }
        p[b % kBlock] = static_cast<uint8_t>(amax == 128 || b % 2 ? -amax : amax);
        const uint16_t half_bits = static_cast<uint16_t>(b * 137);
        std::memcpy(scales.data() + b * 2, &half_bits, 2);
    }
    if (!check_rows(code, scales, maxima)) return 1;

    // All FP16 input bit patterns, including subnormals, signed zero, infinities and NaNs, on full-range Q8.
    constexpr size_t kScaleRows = 65536 / (kCodes / kBlock);
    code.assign(kScaleRows * kCodes, 0);
    scales.resize(kScaleRows * kScales);
    maxima.assign(code.size() / kBlock, 127);
    for (size_t b = 0; b < maxima.size(); ++b) {
        code[b * kBlock + b % kBlock] = static_cast<uint8_t>(b % 2 ? -127 : 127);
        const uint16_t half_bits = static_cast<uint16_t>(b);
        std::memcpy(scales.data() + b * 2, &half_bits, 2);
    }
    if (!check_rows(code, scales, maxima)) return 2;
    std::printf("V4 refit code and scale parity: %zu rows, %zu blocks\n", kMaxRows + kScaleRows,
                (kMaxRows + kScaleRows) * (kCodes / kBlock));
}
