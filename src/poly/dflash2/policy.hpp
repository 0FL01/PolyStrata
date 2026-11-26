// Adaptive choice of target window, after the caller has admitted its placement.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace dflash2 {

// The budget is an estimate of work per round, not a deadline or an SLA. Even
// W1 can exceed it. This class does not allocate, place, propose, or verify.
class Policy {
public:
    static constexpr int kMaxK = 7; // guesses, excluding the anchor
    enum class Source { W1, Lookup, DFlash2 };
    struct Pick {
        Source source = Source::W1;
        int k = 0;
    };

    // A positive finite budget enables the cap; zero disables it. The caller
    // should validate user input rather than relying on this normalization.
    explicit Policy(double budget_ms = 0.0)
        : budget_ms_(std::isfinite(budget_ms) && budget_ms > 0.0 ? budget_ms : 0.0) {}

    // admitted_k is the maximum already allowed by placement, context, output
    // limits and the target window. lookup_k is the available exact suffix.
    // A returned k never exceeds either limit applicable to its source.
    // Call once per round and observe once after it; choose itself is pure.
    Pick choose(int admitted_k, int lookup_k, int match, bool dflash_available) const {
        const int limit = std::clamp(admitted_k, 0, kMaxK);
        const int lookup = std::clamp(lookup_k, 0, limit);
        if (!limit || rounds_ == 0 || rounds_ % 64 == 32) return {};

        // Probe a narrow unseen width first, then remeasure stale widths. The
        // estimate must fit the same margin as an ordinary speculative round.
        // Periodic probes let a formerly slow path recover as conditions change.
        if (dflash_available && (rounds_ <= 2 || rounds_ % 64 == 0)) {
            int candidate = 0;
            for (int k = 1; k <= limit; ++k) {
                if (!proposal_seen_[k]) { candidate = k; break; }
            }
            if (!candidate) {
                candidate = 1;
                for (int k = 2; k <= limit; ++k)
                    if (proposal_at_[k] < proposal_at_[candidate]) candidate = k;
            }
            if (fits(cost(candidate, Source::DFlash2))) return {Source::DFlash2, candidate};
        }

        // Require a modest improvement over W1, so marginal guesses do not
        // buy longer stalls. Compare expected emitted tokens per millisecond.
        Pick best;
        double rate = 1.03 / verify_cost(0);
        double expected = 1.0, survival = 1.0;
        if (dflash_available) {
            for (int k = 1; k <= limit; ++k) {
                survival *= (draft_ok_[k - 1] + 1.5) / (draft_trials_[k - 1] + 2.0);
                expected += survival;
                const double ms = cost(k, Source::DFlash2);
                // An unmeasured proposer width is tried only as a probe.
                if (proposal_seen_[k] && fits(ms) && expected / ms > rate) {
                    rate = expected / ms;
                    best = {Source::DFlash2, k};
                }
            }
        }
        expected = survival = 1.0;
        const double q = lookup_rate(match);
        for (int k = 1; k <= lookup; ++k) {
            survival *= q;
            expected += survival;
            const double ms = cost(k, Source::Lookup);
            if (fits(ms) && expected / ms > rate) {
                rate = expected / ms;
                best = {Source::Lookup, k};
            }
        }
        return best;
    }

    // actual is the source and actual verified width, which may be shorter
    // than chosen after a placement barrier. If the proposer actually ran at
    // another width, pass that as proposal_k so its time is not mispriced.
    // accepted is the matching prefix;
    // compared is the number of available labels in that prefix plus its
    // optional first mismatch. An EOS/cancel/barrier can leave a suffix
    // unlabeled: do not include it in compared. Proposal and verification
    // times are separate wall-clock milliseconds; pass 0 for unavailable
    // measurements, and exclude startup/diagnostic time when appropriate.
    void observe(Pick actual, int accepted, int compared, int match,
                 double proposal_ms, double verify_ms, int proposal_k = -1) {
        ++rounds_;
        const int k = std::clamp(actual.k, 0, kMaxK);
        const int n = std::clamp(compared, 0, k);
        const int hits = std::clamp(accepted, 0, n);
        if (valid_ms(verify_ms)) update(verify_ms_[k], verify_ms);
        if (actual.source == Source::DFlash2) {
            const int pk = proposal_k < 0 ? k : std::clamp(proposal_k, 0, kMaxK);
            if (pk && valid_ms(proposal_ms)) {
                update(proposal_ms_[pk], proposal_ms);
                proposal_seen_[pk] = true;
                proposal_at_[pk] = rounds_;
            }
            for (int i = 0; i < n && i <= hits; ++i) {
                draft_ok_[i] = .94 * draft_ok_[i] + (i < hits ? 1.0 : 0.0);
                draft_trials_[i] = .94 * draft_trials_[i] + 1.0;
            }
        } else if (actual.source == Source::Lookup) {
            const int b = bucket(match);
            lookup_ok_[b] = .97 * lookup_ok_[b] + hits;
            lookup_bad_[b] = .97 * lookup_bad_[b] + (n > hits ? 1.0 : 0.0);
        }
    }

private:
    static bool valid_ms(double x) { return x > 0.0 && std::isfinite(x); }
    static void update(double& old, double value) {
        old = old > 0.0 ? old + (value > old ? .5 : .15) * (value - old) : value;
    }
    static int bucket(int match) { return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3; }
    double lookup_rate(int match) const {
        // Same match buckets and four-observation priors as spec::DraftPolicy;
        // keep our own counts to distinguish an unlabeled suffix from a miss.
        constexpr double prior[] = {.75, .88, .93, .96};
        const int b = bucket(match);
        return (lookup_ok_[b] + 4.0 * prior[b]) / (lookup_ok_[b] + lookup_bad_[b] + 4.0);
    }
    double verify_cost(int k) const {
        if (verify_ms_[k] > 0.0) return verify_ms_[k];
        // Anchor estimates in measured milliseconds, not a unitless prior.
        // Use the nearest observed width and the same conservative shape as
        // the existing guesses policy. Before W1 is measured use 1 ms only
        // as a placeholder; the first round is always W1.
        int nearest = -1;
        for (int j = 0; j <= kMaxK; ++j)
            if (verify_ms_[j] > 0.0 && (nearest < 0 || std::abs(j - k) < std::abs(nearest - k))) nearest = j;
        if (nearest < 0) return 1.0 + .35 * k;
        return verify_ms_[nearest] * (1.0 + .35 * k) / (1.0 + .35 * nearest);
    }
    double proposal_cost(int k) const {
        if (proposal_seen_[k]) return proposal_ms_[k];
        // Full DFlash2 backbone runs for every width. Prefer an observed
        // proposer price to inventing a linear per-token cost.
        int nearest = -1;
        for (int j = 1; j <= kMaxK; ++j)
            if (proposal_seen_[j] && (nearest < 0 || std::abs(j - k) < std::abs(nearest - k))) nearest = j;
        return nearest < 0 ? .2 * verify_cost(0) : proposal_ms_[nearest];
    }
    double cost(int k, Source source) const {
        return verify_cost(k) + (source == Source::DFlash2 ? proposal_cost(k) : 0.0);
    }
    bool fits(double ms) const { return !budget_ms_ || ms * 1.10 <= budget_ms_; }

    double budget_ms_;
    std::uint64_t rounds_ = 0;
    std::array<double, kMaxK + 1> verify_ms_{}, proposal_ms_{};
    std::array<bool, kMaxK + 1> proposal_seen_{};
    std::array<std::uint64_t, kMaxK + 1> proposal_at_{};
    std::array<double, kMaxK> draft_ok_{}, draft_trials_{};
    std::array<double, 4> lookup_ok_{}, lookup_bad_{};
};

} // namespace dflash2
