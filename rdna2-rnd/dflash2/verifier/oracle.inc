        // The serial reference is a normal greedy GEN. This harness never stands in for that reference.
        if (const char* input = std::getenv("RECON_FOLLOW_FILE"); input && *input) {
            if (s.temperature != 0.f || engine.drafts() || look || max_new <= 0)
                throw std::logic_error("verifier probe needs plain greedy generation");
            std::ifstream file(input);
            const std::string csv((std::istreambuf_iterator<char>(file)), {});
            std::vector<int32_t> reference;
            if (!parse_ids(csv.c_str(), reference) || (int64_t) reference.size() != max_new ||
                max_new - 1 > engine.max_context() - engine.pos())
                throw std::logic_error("invalid verifier reference or insufficient context room");
            const char* width = std::getenv("RECON_ORACLE_W");
            const int W = width ? std::atoi(width) : 0;
            if (W < 1 || W > engine.max_window()) throw std::logic_error("invalid verifier probe width");
            const bool force = std::getenv("RECON_FORCE") != nullptr;
            const char* bad = std::getenv("RECON_BAD_DRAFT_INDEX");
            const int64_t bad_at = bad ? std::stoll(bad) : -1;
            if (bad && (force || W < 2 || bad_at < 1 || bad_at >= max_new - 1))
                throw std::logic_error("a bad draft must be a non-final output index of an accepted W>1 arm");
            bool injected = false;
            struct Check { int64_t index; int row; int32_t picked; float margin; };
            std::vector<Check> checks;
            checks.reserve(reference.size() + W);
            std::vector<double> windows;
            std::vector<int> sizes;
            auto inspect = [&](const float* logits, int64_t index, int row) {
                const int32_t t = strata::poly::most_likely(logits, engine.n_vocab());
                float other = -INFINITY;
                for (int v = 0; v < engine.n_vocab(); ++v)
                    if (v != t) other = std::max(other, logits[v]);
                checks.push_back({index, row, t, logits[t] - other});
                return t;
            };
            const int64_t start = engine.pos();
            const auto first = inspect(engine.logits(), 0, -1);
            bool more = show(force ? reference[0] : first);
            bool on_reference = first == reference[0];
            int32_t current = force ? reference[0] : first;
            int64_t rows = 0;
            const auto decode = Clock::now();
            while (more) {
                const auto idx = r.produced;    // predictions begin after the already emitted token
                const int w = (int) std::min<int64_t>(force || on_reference ? W : 1, max_new - idx);
                int32_t tokens[8];
                tokens[0] = current;
                for (int i = 1; i < w; ++i) {
                    const int64_t at = idx + i - 1;
                    tokens[i] = reference[(size_t) at];
                    if (at == bad_at && !injected) {
                        tokens[i] = (tokens[i] + 1) % engine.n_vocab();
                        injected = true;
                    }
                }
                const auto begin = Clock::now();
                if (w == 1) {
                    engine.eval_decode(tokens[0]);
                    kept.read(tokens, 1);
                    windows.push_back(ms_since(begin));
                    const auto t = inspect(engine.logits(), idx, 0);
                    current = force ? reference[(size_t) idx] : t;
                    on_reference = on_reference && current == reference[(size_t) idx];
                    more = show(current);
                } else {
                    engine.window(tokens, w);
                    int right = 1;
                    if (force) {
                        engine.keep(w);
                        kept.read(tokens, w);
                        windows.push_back(ms_since(begin));
                        for (int i = 0; i < w && more; ++i) {
                            inspect(engine.logits_of(i), idx + i, i);
                            current = reference[(size_t) idx + i];
                            more = show(current);
                        }
                    } else {
                        // Same accept/keep rule as generation. The bonus pick is emitted but not fed.
                        int32_t t = inspect(engine.logits_of(0), idx, 0);
                        while (right < w && t == tokens[right] && (more = show(t))) {
                            t = inspect(engine.logits_of(right), idx + right, right);
                            ++right;
                        }
                        engine.keep(right);
                        kept.read(tokens, right);
                        windows.push_back(ms_since(begin));
                        r.guessed += w - 1;
                        r.guessed_right += right - 1 + (more ? 0 : 1);
                        if (more) {
                            current = t;
                            on_reference = t == reference[(size_t) r.produced];
                            more = show(t);
                        }
                    }
                }
                sizes.push_back(w);
                rows += w;
            }
            r.decode_ms = ms_since(decode);
            if (bad && !injected) throw std::logic_error("requested bad draft index was not offered");
            const auto st = engine.stats();
            std::fprintf(stderr, "RECON_RESULT %s %d %lld %lld %lld %.6f %lld %lld\n",
                         force ? "forced" : "accepted", W, (long long) start, (long long) rows,
                         (long long) (engine.pos() - start), r.decode_ms,
                         (long long) st.gpu_experts, (long long) st.cpu_experts);
            for (size_t i = 0; i < windows.size(); ++i)
                std::fprintf(stderr, "RECON_WINDOW %d %.6f\n", sizes[i], windows[i]);
            for (const auto& c : checks)
                std::fprintf(stderr, "RECON_CHECK %lld %d %d %.9g\n", (long long) c.index, c.row, c.picked, c.margin);
            return r;
        }
