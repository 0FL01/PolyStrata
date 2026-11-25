# Can the host use the *next* router before the real MoE decision?

**Yes as a hint; not as the router.** We tested a narrower idea than donor #214. Never fetch an expert into a new GPU slot. Predict while the GPU computes the next head, and read one likely **CPU** expert into the host cache. Wrong predictions then cost time and memory bandwidth, not a different forward pass. The missing measurement after a CPU replay is what happens when the predictor competes with the real engine.

## From prediction to a timed CPU branch

Inputs: two earlier guarded Tideglass v2.1 runs on this RX 6700 XT / Ryzen 7 5700X, mixed and independent code at 65,536 prompt tokens, Q8 KV, 8 CPU workers and 64 live swaps. Read-only `PSCPU01` has the normalized FP32 expert input and final weighted CPU sum on every CPU-positive visit, 24 consecutive decode positions per workload. The corresponding `PSW642` timelines are from **separate** runs. Before using their timing or masks, `near_cases.py` checks *every* ordered CPU ID and weight bit (including empty branches) across all 1,152 visits per input. The captured and traced prompts and the first 33 output IDs match. Those older runs used a diagnostic 5,050 MiB cap, not today's production auto-placement.

1. Apply **the destination physical layer's** 128×2,304 router to the *previous logical visit's* normalized FFN input, including the 17→6 transitions at logical visits 18/30. Rank by `sigmoid(score)+destination_bias`; this never supplies weights to real experts. The first implementation uses AVX2 FP32 FMA (not a bit-exact reference for HIP). The second packs the weights once to signed 7-bit with one scale per output, the input to offset unsigned 8-bit with one scale per row. `vpmaddubsw` cannot saturate here (`254×63×2 < 32768`); subtract the zero-point contribution, then apply the scales. This is an intentionally lossy *ranking* kernel. No GGUF bytes or target projection changed.
2. On the same ordered inputs, including quantizing `x`, sigmoid, bias and top four, a single 5700X core took **37.21→6.59 µs** median mixed and **40.08→6.33 µs** code. One warmup per representation, four alternating whole sweeps; 1,006 / 1,010 predecessor rows per sweep (all CPU-positive sources except the last visit, not all 48 visits). Not a DRAM-cold or concurrent-engine timing. Top-four set retention relative to the host FP32 hint was **98.53% / 98.54%**. Without any residency filter, only **29.2% / 25.4%** of the four hints identified an actual target CPU use. High numerical agreement between predictors is not the same as useful preparation.
3. Now restrict the target to core passes 2 and 3. Admit at most one of the four, excluding resident and pending IDs **using the last same-layer visit's mask**, not the target labels. Abstain if the predecessor input is missing or there is no candidate. Use the real target assignment only to score. In these traces, no selected ID went resident/pending between mask observation and demand.

| 24 positions per workload; 576 eligible destinations | mixed | code |
|---|---:|---:|
| No predecessor input recorded / all four masked | 75 / 117 | 84 / 191 |
| Packed predictions that hit / issued | 306 / 384 | 269 / 301 |
| Precision / recall of **all** target CPU uses | 79.7% / 18.9% | 89.4% / 20.5% |
| FP32 sidecar, same policy: hit / issued | 304 / 380 | 269 / 304 |
| Previous pass's CPU router order instead: hit / issued | 199 / 469 | 280 / 446 |
| If every read finishes, total / false MB per token (decimal) | 71.66 / 14.56 | 56.17 / 5.97 |

The previous-pass comparison has different coverage. The ranked sidecar admits fewer requests, not universally more hits. These numbers are **not** donor prefetch precision, cache hit rate, recall at every visit, or a PCIe bandwidth measurement.

4. Test whether acting on those predictions actually helps. Reuse the checked `arch-decode-recon/prefetch/bench.cpp` paired CPU harness, substituting real `x` and validating both branch sums against the independent capture. Sample up to three visits per `(hit, CPU count)` stratum, deterministic seed 4242. Production CPU object, real GGUF, CLFLUSHOPT before each timed arm, pre-faulted pages, eight workers on logical CPUs 0–7, helper on 15. ABBA/BAAB within each case, three rounds; mixed/code have 43/37 cases. The late arm posts **after** the previous CPU branch, within the next router wait; the early arm posts during that CPU branch. Use a synthetic wait with the recorded length minus 8 µs for prediction on *both* arms (remaining p50 280/277 µs). That fixed debit is a model: packed calls exceed 8 µs on some rows. No real GPU, swap or launch activity in this CPU leaf.

| CPU-only measurement, positive = saving | mixed | code |
|---|---:|---:|
| Early read: median whole pair vs cold | −52.61 µs | −28.96 µs |
| Late read: median target CPU compute saved on hits | 61.23 µs | 62.86 µs |
| Late read: median whole pair vs cold, false hints | −3.90 µs | −5.27 µs |
| Late: finished all reads before synthetic deadline | 254 / 258 | 216 / 222 |
| Population-scaled late delta, clipped by archived target join proxy | 0.757 ms/token | 0.654 ms/token |
| Same diagnostic for cold/cold | −0.043 ms/token | +0.084 ms/token |

All 2,880 timed trials passed branch FP32 byte comparisons; no timed major faults. The last two rows are **projections**, not measured time saved in decode. `collect()` drains outside the synthetic critical path; one late helper per workload finished after target compute. An early helper made the current branch slower even on a machine with a spare SMT context. Rechecking slot ownership before use, and keeping the previous `h_x_` alive until a hint has read it, matter more than shaving another microsecond off the dot.

## Live probe design

`build_wait_probe.py` relinks one scratch `engine.cpp` and a separate AVX2 predictor object with the Release gfx1031 objects. Serial no-drafter only. Instead of asynchronous mutable state, snapshot the previous CPU-positive visit's 9,216-byte mapped input **before the next head can overwrite it**. At the next `Engine::moe`, the GPU is already computing the real router. If its event is pending, predict on the host thread, take the first top-four ID without a live VRAM slot and read that GGUF payload one cache line at a time. Query the event per 256 KiB and give way when ready. The probe touches no slot, defers no fence, posts no DMA, and never calls `policy.observe` or changes target routing/counters. Unlike the offline conservative mask, the live read tests current slots, not pending transfers: a pending expert can generate extra host traffic, but cannot become a fabricated GPU hit. No second helper to drain. Count bytes **attempted**, not memory-controller traffic. This experiment is deliberately a different schedule from the CPU leaf.

The scratch binary's modes 1/2 warm opposite eight-position blocks. Compare against unmodified Release and the other phase, drop the first two positions per block for timing, compare **all** naturally generated IDs. Init costs in the first decode step are not a production startup measurement. Code, fixtures, and no-change decisions are specific to W1; do not generalize to `moe_rows`, DSpark or multiple sequences.

**First live test (eight independent processes).** 2K fresh code / 65,600 mixed restored + one input, 160 serial steps per leg, capped production policy, code then long controls and opposite scratch phases interleaved. All four legs on each fixture emitted **161/161 identical IDs** per leg, including the first. The controls and candidates also reported identical aggregate expert counts within each fixture; no inference of byte-state or phase-by-phase placement from these counters. No fresh guard warnings or stops.

| Untraced W1, positive delta = **slower** | 2K code | Restored 64K mixed |
|---|---:|---:|
| Real CPU hits / issued, both phases pooled | 928 / 1,069 (86.8%) | 1,707 / 2,211 (77.2%) |
| Read extent per **active** token | 29.0 MB | 60.7 MB |
| Stock / stock repeat, ID/s | 38.30 / 41.35 | 27.74 / 25.31 |
| Scratch phase 1 / 2, ID/s (half on in each) | 37.24 / 41.91 | 29.27 / **8.48** |
| Same-position on−off, median of 20 six-position block means | +0.602 ms | −2.477 ms |
| Same diagnostic on two stock legs, arbitrary pseudo-treatment | +0.0004 ms | −0.422 ms |
| Blocks where on beat off | 9/20 | 11/20 |

In the slow long leg, *p90* was 282.8 ms vs 38.8 ms in the opposite phase, and **prompt-only** time before timed decode was 1,664 vs 111 ms. The slowdown cannot be assigned to the hint from these data. The large negative mean block delta on long (−15.26 ms) is unusable as a speed estimate; even the control/control mean delta was −5.79 ms. Treat the first long test as inconclusive, not a 2.48 ms/token win. `baseline/live/summary.json`, `wait_report.py`.

**Repeat and diagnose the outlier rather than deleting it.** Reversed the phase order, lengthened the same restored input to 256 steps, took process-tree rusage. All 257 greedy IDs again agreed in every leg. Both stock legs ran at 31.25/32.63 ID/s; alternating legs at 29.78/14.46. In the slow phase-2 leg `/usr/bin/time -v` recorded **1,244,746 major faults** over the whole process tree from startup through exit (vs 3,834 in phase 1 and 7/0 in the controls). The background GGUF mapper reported **5.1 GiB** registered only *after* generation vs **8.1 GiB** in the others; the original slow phase-2 leg had reported **0.0 GiB** at exit. Those are Linux VM counters, not GPU faults. During this repeat the paired-block median was −1.44 ms, but comparing one paging trajectory to one warm trajectory at alternating positions cannot distinguish treatment from systematic warmup. This changes the experiment: for the final warm diagnostic require the mapper's **full 8.1 GiB** receipt before sending GEN on *every* leg and count process faults during GEN. Do not subtract startup/registration from user-visible cold latency or change the actual service's startup policy.

**Final warm diagnostic:** `wait_leg.py --require-registered 8.1 --steps 256`, same 65,600-row import, order 2/stock/1/stock. All four emitted the same **257/257 IDs**, reported the same **70,388/98,688** hits, and recorded **zero major faults during GEN**; only 95–101 ms for the single new prompt row. Stock repeat **32.42/32.75 ID/s**, scratch half-active legs **32.57/32.04 ID/s**. Median of 32 matched block means **−0.342 ms/token** on−off vs **−0.075 ms** pseudo-effect on stock/stock (means −0.453/−0.056; 19/32 faster). Actual hints reached **2,551/3,381** CPU branches (75.5% precision); the probe read **59.0 MB per active token**. There is a small within-position signal under pinned inputs, but **neither a repeatable net engine win nor a payoff for moving weights across PCIe**. Keep the existing path. A cheaper selective preparation must beat both the no-op overhead and this warmed control on an independently held-out CPU-late task before meriting a runtime switch. `baseline/live/{repeat,resident}/summary.json`, each `metrics.json`, guard rusage and stderr.

**Does the entire optimization beat the actual release?** A final, independent warm **ABBA** used the unmodified ELF for A and mode **3 (always enabled)** for B, all other parameters and inputs as above. Stock / probe / probe / stock = **32.85 / 32.38 / 32.92 / 32.69 ID/s**; p90 **34.77 / 34.94 / 34.75 / 35.05 ms**. The always-on change lost 1.4% throughput in AB and gained 0.7% in BA; no consistent gain. All **257/257** greedy IDs matched, including the resumed first token, with the same lookup counts, 8.1 GiB registered and zero major faults during GEN. Both B legs issued 3,381 hints, of which exactly 2,551 hit, spanning **15.11/15.12 GB** of cache-line reads in 256 steps. **Close this implementation, not the question of CPU-late time.** Real data under `baseline/live/always/{A1,B1,B2,A2}/`; rerun these legs with `wait_leg.py --require-registered 8.1 --steps 256 --mode 0|3` on the matching executable.

## Reproduce on the profiling host

From the root; create a new `R`, reuse captures and model read-only. The original run is `native-donor-bc2b`. Do not run CPU leaf timing alongside a GPU leg. `prefetch/report.py` accepts only completed trials. Use `schedule`, not the older synthetic oracle's `ceiling` mode, for capture checks.

```sh
T=$HOME/.cache/opencode-tmp/opencode
R=$T/polystrata-runs/native-donor-repeat; mkdir -p "$R/baseline/live"
B=$R/baseline; D=docs/goals/native-donor-recon; P=docs/goals/arch-decode-recon/prefetch
M=$HOME/.lmstudio/models/unsloth/Inkling-2-Mini/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf
c++ -O3 -std=c++20 -mavx2 -mfma -Iinclude "$D/router_sidecar.cpp" -o "$B/router_sidecar"
patch -s -o "$B/helper.cpp" "$P/bench.cpp" < "$D/helper.patch"
c++ -O3 -DNDEBUG -std=c++20 -fPIE -pthread -mclflushopt -Iinclude -Isrc/poly -I"$D" \
  -Ithird_party/llama.cpp/ggml/include "$B/helper.cpp" build-poly-hip/CMakeFiles/strata-poly.dir/src/poly/cpu_experts.cpp.o \
  build-poly-hip/ggml/src/libggml-{cpu,base}.a -ldl -o "$B/helper"
for w in mixed code; do
  c=$T/tideglass-impl/capture/$w/run/cpu.bin
  taskset -c 6 "$B/router_sidecar" "$M" "$c" 4 > "$B/sidecar-$w.csv"
  python -B "$D/near_cases.py" "$B/sidecar-$w.csv" "$c" "$T/wait64-recon.dg0bMA/timeline/$w/run/visits.psv" "$B/$w.tsv"
  PSCPU_CAPTURE="$c" taskset -c 0-7 "$B/helper" schedule "$M" "$B/$w.tsv" 3 15 > "$B/helper-$w.csv" 2> "$B/helper-$w.log"
done
python -B "$P/report.py" --tokens 24 "$B"/helper-{mixed,code}.csv
python -B "$D/build_wait_probe.py" "$B/live"
```

For live tests `baseline/fixed.json` contains only the installed config's `env,args,tokenizer,lib_dirs`, plus `STRATA_INKLING_EXPERT_KV_MIB=4968` in `env`; do not copy credentials. Use `wait_leg.py OUT ELF FIXED_JSON PROFILE_SEED PROMPT_CSV --mode MODE --steps 160`, always inside `flock "$R/gpu.lock" python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output NEW_GUARD -- timeout 230 ...`. With `--restore READ_ONLY_SESSION` the prompt file contains the one additional token. Input snapshot: `swa-recon-20261125/cold64ref/session.bin`, suffix = last emitted ID in its `metrics.json`. Short code: `arch-campaign.XWM1yK/trace-code-2048/prompt.csv`; seed: `arch-campaign.XWM1yK/profile.seed`.
