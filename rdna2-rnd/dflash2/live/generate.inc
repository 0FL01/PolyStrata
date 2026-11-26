// Inside Session::generate, after the stock target pick/show closures.
// No draft RNG, no replacement of target scores with shortlisted scores.
const char* limit = std::getenv("DF2_K");
if (!limit || limit[0] < '0' || limit[0] > '7' || limit[1] || limit[0] - '0' >= engine.max_window())
    throw std::runtime_error("DF2_K must be 0..7 and less than --window");
const int K = limit[0] - '0';
const bool serial = std::getenv("DF2_SERIAL_VERIFY") && std::string(std::getenv("DF2_SERIAL_VERIFY")) == "1";
const bool adaptive = std::getenv("DF2_ADAPTIVE") && std::string(std::getenv("DF2_ADAPTIVE")) == "1";
double budget_ms = 0.;
if (const char* value = std::getenv("DF2_ADAPTIVE_BUDGET_MS")) {
    char* end = nullptr;
    budget_ms = std::strtod(value, &end);
    if (!adaptive || serial || end == value || *end || !(budget_ms > 0.) || !std::isfinite(budget_ms))
        throw std::runtime_error("DF2_ADAPTIVE_BUDGET_MS requires adaptive batch verification and a positive finite number");
    std::fprintf(stderr,"DF2_BUDGET %.17g\n",budget_ms);
}
strata::spec::DraftPolicy prices(K+1);
if (adaptive) { repeats.reset(); repeats.append(history.data(),history.size()); }
// Survival per position for the block, conditional on the preceding guesses matching.
// Never charge a failed tail as seven independent incorrect proposals.
const int sources=engine.uses_dspark() ? 2 : 1;
double ok[2][7]={}, trials[2][7]={}, block_ms[2]={};
double window_ms[8]={}, draft_k_ms[2][8]={};
int block_calls[2]={}, lookups=0, serial_calls=0;
struct Round { int64_t pos; int k, accepted, fed; double draft_ms, total_ms; };
std::vector<Round> rounds;
std::vector<double> shown_ms;
// Optional correctness tape. Compare whole distributions and canonical target
// state even if both modes happen to pick identical IDs. Never price these legs.
const char* tape = std::getenv("DF2_PARITY_TAPE");
std::ofstream diag;
if (tape) {
    diag.exceptions(std::ios::badbit | std::ios::failbit);
    diag.open(tape, std::ios::binary);
}
const auto emit = [&](int32_t t) {
    if (tape) diag.write(reinterpret_cast<const char*>(picked_from), engine.n_vocab()*sizeof(float));
    if (adaptive) repeats.append(t);
    shown_ms.push_back(ms_since(t0)); return show(t);
};
if (max_new > 0) {
    int32_t t = pick(engine.logits());
    bool more = emit(t);
    while (more) {
        const auto start = Clock::now();
        const int64_t pos = engine.pos();
        int k = (int) std::min<int64_t>({K, engine.max_context() - pos - 1, max_new - r.produced - 1});
        if (k < 0) throw std::logic_error("no room for anchor");
        int32_t w[8] = {t};
        int match=0, source=0; bool copied=false;
        if (adaptive && k) {
            int32_t lp[7]; const int have=repeats.propose(k,lp);
            match=repeats.last_match();
            const int max_k=k;
            const auto cost = [&](int j, int s) { // -1 = suffix lookup
                if (budget_ms) {
                    const double verify = window_ms[j] > 0. ? window_ms[j] : prices.cost_ms(j+1);
                    if (s < 0) return verify;
                    if (draft_k_ms[s][j] > 0.) return verify + draft_k_ms[s][j];
                    // Full 8-row draft backbone; smaller K only reduces head/selector work.
                    return verify + (block_ms[s] > 0. ? block_ms[s] : .2 * window_ms[0]);
                }
                return prices.cost_ms(j+1) + (s < 0 ? 0. : block_ms[s]);
            };
            // Periodically remeasure W1. Explore one more position than the source has tried,
            // falling back to the regular chooser if the candidate can't fit the budget.
            int probe = 0;
            const size_t step = rounds.size();
            if (budget_ms && step && (step<=size_t(2*sources) || step%64==0)) {
                source=(step<32 ? step-1 : step/64-1)%sources;
                for (int j=1;j<=max_k;++j) if (draft_k_ms[source][j]==0.) {
                    if (cost(j,source)*1.10 <= budget_ms) probe=j;
                    break;
                }
            }
            if (!step || (budget_ms && step%64==32)) k=0;
            else if (probe) k=probe;
            else if (!budget_ms && (step<=size_t(2*sources) || step%48==0)) {
                k=max_k; // Include the first cold DSpark head setup; remeasure its warm cost independently.
                source=(rounds.size()<=size_t(2*sources) ? rounds.size()-1 : rounds.size()/48)%sources;
            }
            else {
                k=0; double best=1.03/cost(0,-1);
                for(int s=0;s<sources;++s) {
                    double expected=1., survive=1.;
                    for (int j=1;j<=max_k;++j) {
                        survive*=(ok[s][j-1]+1.5)/(trials[s][j-1]+2.); expected+=survive;
                        const double ms=cost(j,s);
                        if (budget_ms && (draft_k_ms[s][j]==0. || ms*1.10>budget_ms)) continue;
                        const double r=expected/ms;
                        if (r>best) {best=r; k=j; source=s;}
                    }
                }
                double expected=1., survive=1.;
                for (int j=1;j<=have;++j) {
                    survive*=prices.lookup_rate(match); expected+=survive;
                    const double ms=cost(j,-1);
                    if (budget_ms && ms*1.10>budget_ms) continue;
                    const double r=expected/ms;
                    if (r>best) {best=r; k=j; copied=true;}
                }
                if (copied) std::copy_n(lp,k,w+1);
            }
        }
        if (k && !copied) {
            if (!source) df_live::propose(t, k, w + 1);
            else for (int j=0;j<k;++j) { float p; w[j+1]=engine.draft(w[j],j,&p); }
        }
        const double draft_ms = (k && !copied) ? ms_since(start) : 0.;
        int right = 0, fed = 0;
        if (!k || serial) {
            // Exact diagnostic. Never feed a guess until the W1 sampler actually picked it.
            for (;;) {
                engine.eval_decode(t);
                kept.read(&t, 1);
                ++fed;
                t = pick(engine.logits());
                const bool hit = right < k && t == w[right + 1];
                right += hit;
                more = emit(t);
                if (!more || !hit) break;
            }
        } else {
            engine.window(w, 1 + k);
            for (int i = 0; i <= k; ++i) {
                t = pick(engine.logits_of(i));
                const bool hit = i < k && t == w[i + 1];
                right += hit;
                ++fed;  // w[i] was fed, the sampled t has not been
                more = emit(t);
                if (!more || !hit) break;
            }
            engine.keep(fed);
            df_live::commit(fed);
            kept.read(w, fed);
        }
        r.guessed += k;
        r.guessed_right += right;
        const double elapsed=ms_since(start);
        if (adaptive) {
            const auto observe_ms = [](double& old, double value) {
                // Follow rising costs quickly; don't treat a single fast round as a new baseline.
                old = old > 0. ? old + (value > old ? .5 : .15)*(value-old) : value;
            };
            if (budget_ms) observe_ms(window_ms[k],elapsed-draft_ms);
            prices.observe(copied,k+1,right,match,elapsed-draft_ms);
            if (!k) ++serial_calls;
            else if (copied) ++lookups;
            else {
                // The first DSpark proposal stages its head. Count the observed acceptance
                // and stall, but don't seed a warm-price model with that one-time cost.
                const bool warm = !source || block_calls[source]>0;
                if (budget_ms && warm) observe_ms(draft_k_ms[source][k],draft_ms);
                if (!budget_ms || warm)
                    block_ms[source]=block_calls[source]>1 ? .8*block_ms[source]+.2*draft_ms : draft_ms;
                ++block_calls[source];
                for (int i=0;i<k && i<=right;++i) {
                    ok[source][i]=.94*ok[source][i]+(i<right); trials[source][i]=.94*trials[source][i]+1.;
                }
            }
        }
        rounds.push_back({pos, k, right, fed, draft_ms, elapsed});
    }
}
r.decode_ms = ms_since(t0);
if (adaptive) std::fprintf(stderr,"DF2_POLICY serial=%d lookup=%d dflash2=%d dspark=%d\n",serial_calls,lookups,block_calls[0],block_calls[1]);
if (tape) {
    const size_t nstate=engine.state_bytes();
    const int64_t pos=engine.pos();
    std::vector<uint8_t> state(nstate);
    engine.save_state(state.data());
    diag.write(reinterpret_cast<const char*>(state.data()), nstate);
    const size_t chunk=64, stride=engine.row_bytes();
    std::vector<uint8_t> rows(chunk*stride);
    for (int64_t i=0;i<pos;i+=chunk) {
        const size_t m=std::min<int64_t>(pos-i,chunk);
        engine.save_rows(i,m,rows.data());
        diag.write(reinterpret_cast<const char*>(rows.data()),m*stride);
    }
    diag.close();
    std::fprintf(stderr,"DF2_PARITY vocab=%d outputs=%lld state=%zu row_bytes=%zu committed=%lld\n",
                 engine.n_vocab(),(long long)r.produced,nstate,stride,(long long)pos);
}
// Buffer records during decoding; stderr I/O is not part of a round.
for (auto a : rounds) std::fprintf(stderr, "DF2_ROUND %lld %d %d %d %.5f %.5f\n",
    (long long)a.pos, a.k, a.accepted, a.fed, a.draft_ms, a.total_ms);
std::fprintf(stderr, "DF2_TIMES %.5f", r.prompt_ms);
for (double x : shown_ms) std::fprintf(stderr, " %.5f", x);
std::fputc('\n', stderr);
return r;
