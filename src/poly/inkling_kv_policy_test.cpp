#include "inkling_kv_policy.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
using namespace strata::poly::inkling;

void require(bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(msg);
}

// Model the two physical planes with integer row identities. A sealed value is deliberately noninvertible.
struct Cache {
    int capacity, pos = 0, sealed = 0, mapped = 0;
    std::vector<int> hot, cold;
    Cache(int context, int batch) : capacity(v4_hot_rows(context, batch)), hot(capacity), cold(context) {}
    void cover(int end) { mapped = std::max(mapped, end); }
    void seal() {
        const int stop = v4_seal(pos);
        require(stop >= sealed && pos - sealed <= capacity, "old hot rows before seal");
        while (sealed < stop) {
            cold[sealed] = hot[sealed % capacity] / 7;
            ++sealed;
        }
    }
    void append(int n, int tag) {
        cover(pos + n);
        seal();
        require(pos + n - sealed <= capacity, "append exceeds ring");
        for (int r = pos; r < pos + n; ++r) hot[r % capacity] = r + tag;
        pos += n;
    }
    std::vector<int> checkpoint() {
        seal();
        std::vector<int> tail;
        for (int r = sealed; r < pos; ++r) tail.push_back(hot[r % capacity]);
        return tail;
    }
    void rewind(int at, const std::vector<int>& tail) {
        sealed = v4_seal(at);
        require((size_t) (at - sealed) == tail.size(), "checkpoint tail length");
        for (int r = sealed; r < at; ++r) hot[r % capacity] = tail[r - sealed];
        pos = at;
    }
    void restore(const Cache& saved, const std::vector<int>& tail) {
        cover(saved.pos);
        std::copy_n(saved.cold.begin(), saved.sealed, cold.begin());
        rewind(saved.pos, tail);
    }
};

void lifecycle() {
    for (const int ctx : {16, 2048, 2175, 2176, 32768, 200000}) {
        for (const int batch : {1, 31, 128, 2048}) {
            const int capacity = v4_hot_rows(ctx, batch);
            for (int pos = 0; pos <= ctx; ++pos) {
                const int first = v4_seal(pos), n = std::min(batch, ctx - pos);
                require(first % kKvPage == 0 && first <= pos, "invalid seal");
                require(pos - first <= v4_checkpoint_rows(ctx), "invalid checkpoint size");
                require(pos + n - first <= capacity, "call would overwrite readable rows");
            }
        }
    }
    Cache live(28000, 2048);
    for (int i = 0; i < 4; ++i) live.append(2048, 9631);
    live.append(61, 9631);
    const int pos = live.pos;
    const auto hot = live.checkpoint();
    const Cache saved = live;
    for (int i = 0; i < 5; ++i) live.append(2048, 9631);
    Cache longer = live;
    const auto longer_tail = longer.checkpoint();
    live.rewind(pos, hot);
    require(live.checkpoint() == hot, "rewind recovered former cold rows as exact Q8");
    live.append(77, 17713);
    require(live.hot[pos % live.capacity] == pos + 17713, "branch overwrote the prefix tail slot");
    for (const Cache* src : std::array<const Cache*, 4>{&saved, &saved, &longer, &saved}) {
        const auto& tail = src == &saved ? hot : longer_tail;
        const int backing = std::max(live.mapped, src->pos);
        live.restore(*src, tail);
        require(live.mapped == backing, "repeat restore remapped or grew unnecessarily");
        require(live.checkpoint() == tail, "repeat restore lost the saved hot rows");
        require(std::equal(live.cold.begin(), live.cold.begin() + live.sealed, src->cold.begin()), "cold prefix");
    }
}

void selection() {
    const int start = kKvRecent + (kKvTop + 2) * kKvPage;
    require(!k256_worthwhile(start - 1) && k256_worthwhile(start), "K256 first profitable length");
    for (const int span : {start, start + 69, 64001, 199105, kKvPage * kKvMaxPages}) {
        const int old = k256_old_pages(span), pages = (span + kKvPage - 1) / kKvPage;
        std::array<int, kKvTop> picked;
        for (int p = 0; p < kKvTop; ++p) picked[p] = p + old - kKvTop;
        int prev = -1;
        for (int c = 0; c < 1 + kKvTop + pages - old; ++c) {
            const int page = k256_page(c, old, picked.data());
            require(prev < page && page < pages, "pages must be unique, chronological, and in range");
            if (c == 0) require(page == 0, "anchor");
            else if (c <= kKvTop) require(page < old, "selected page overlaps recent tail");
            else require(page == old + c - kKvTop - 1, "recent tail is contiguous");
            prev = page;
        }
        require(prev == pages - 1 && span - old * kKvPage >= kKvRecent, "last causal key and recent support");
    }
}
}  // namespace

int main() {
    lifecycle();
    selection();
    std::puts("Inkling hybrid KV policy: PASS");
}
