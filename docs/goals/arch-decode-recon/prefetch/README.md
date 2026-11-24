# CP-P1-03/04 — CPU-only warmup of one routed expert

**Decision: no engine prototype on the present evidence.** Warming one real Tideglass Q4_0 expert works on this CPU,
but the best deadline-aware arm projects only **0.202–0.219 ms/token** in this CPU-only replay of the short trace
(roughly 1% of the CP-01 2K median interval). That is not a measured decode gain. Pushing the same read into the
preceding CPU expert branch is worse: it typically hurts the whole path, including on predictions later rejected by the
real router. Do not allocate a GPU slot or increase the prefetch cap for this candidate. Reopen only if a disjoint
long-context trace shows materially more *predictable exposed* cold work, then test contention before integration.

**64K follow-up: closed.** The [disjoint 64K replay](long-context.md) projects 0.338–0.356 ms/token for the measured
one-helper arm. Even granting 200 µs per correct prediction and free, always-on-time reads, the cap-one projection
is 1.072 ms/token and a cap of eight reaches only 1.813 ms/token. No further engine runs for this branch.

## Actual trace screen

CP-02 `arch-campaign.XWM1yK/trace-on/visits.jsonl`, SHA-256
`572ec695099f6efe50481c24e154e884adbe5eb89f94c318910b755bf421a526`; mixed label (prose at 2K), positions
2048–2303 inclusive, 48 visits per token, live swaps, 8 CPU workers. Decisions use the common route-screen suffix
2215–2303 (89 tokens); this is not independent workload validation. On each target visit in passes 2/3, predict the first
previous-pass, same-token, same-physical-layer expert whose source placement was CPU and whose source mask was not
resident or pending. Selection is in router order; use the target route only to score. No selected cap-one candidate
became resident or pending by target routing or was routed to the GPU. A live implementation would have to recheck that
fact before issuing, not assume it. Each read covers 4,478,976 B of GGUF payload.

| Cap | Issued | Actual CPU expert hits | Precision / CPU-miss recall | False-positive payload | Hit-visit join proxy per token |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,182 | 516 | 43.7% / 20.1% | 33.52 MB/token | 0.812 ms |
| 2 | 1,659 | 691 | 41.7% / 26.9% | 48.72 MB/token | 0.913 ms |
| 4 | 1,896 | 771 | 40.7% / 30.0% | 56.62 MB/token | 0.927 ms |

There are 2,568 target CPU expert uses and 1.743 ms/token of target join proxy in this slice. The last column grants *all*
observed join on visits with at least one correct prediction; it is an optimistic marker-gap envelope, not removable
CPU time. Cap one issues 59.48 MB/token in payload reads; cap two spends another 24.01 MB/token for at most another
0.101 ms/token of touched proxy. The full 256-token run gives cap one 1,673 / 3,654 useful predictions, 8,464 target
CPU experts, and a 1.125 ms/token hit-visit envelope. That does not warrant a wider sweep.

Same-layer source→target route-ready lead in the suffix is **4.50 / 5.18 / 6.30 ms** at p10/p50/p90. On correct picks
there are **5 / 12 / 24** intervening CPU expert uses (counts, not unique lines or DRAM reads). At the median this is
53.7 MB of weight touches, greater than the CPU's 32 MiB L3. Warming at the source is therefore not equivalent to
warming near the deadline. Target `host_begin`→`route_ready` is **268 / 296 / 391 µs** for issued picks, including
**239 / 269 / 365 µs** measured router wait. 803 of 1,182 issued visits follow a CPU-positive visit. The benchmark uses
the near window, not the full same-layer lead, and tests interference with preceding CPU work separately.

## CPU experiment and result

Ryzen 7 5700X, eight physical cores (`0–7`), 32 MiB shared L3, helper pinned to SMT sibling `15` of core `7`. The
unchanged `src/poly/cpu_experts.cpp` and ggml CPU archives from the production Release build were used; the relevant
sources match `c0dd0e52`. No GPU libraries are linked. The model is the campaign's Tideglass GGUF (manifest SHA-256
`7af63a2c9362f0ac6fc708443348ead5dbfcb7ecde8cd454ea3572c68864aa72`, 15,537,527,488 B; not rehashed in the slot).
The probe reads joined gate/up and down directly from the model mapping, physical layers 6–17, with real router order
and FP32 weight bits. SiTU uses the model metadata (`4/25`); activations are synthetic, fixed per captured visit. The
predictor neither adds experts to the input list nor changes backend, arithmetic, or weight aggregation. A no-op
placeholder preserves the production mixed visit's extra `meanwhile` task without calling the GPU.

38 rotating cases (19 hit, 19 false), chosen deterministically from strata `(hit, CPU expert count)`, at most three per
stratum, including false picks on zero-CPU targets. Each arm has an interleaved cold ABBA/BAAB control and a separate
cold/cold comparison. Cold means evicted with `CLFLUSHOPT` outside all timed intervals, after prefaulting model pages.
Nine read calibrators using eight workers in the ceiling run measured **97.3 µs cold / 62.3 µs warm** median (46.1 /
71.9 GB/s effective expert payload). This confirms a cache benefit, not a hardware counter measurement of LLC fills.
Only the oracle warms an actual hit outside the timer with eight workers. In schedule arms, all preparation is timed.
The helper stops issuing lines at the synthetic route deadline (check at most every 32 KiB), and target compute never
waits for it. `helper_t2` finishing means its hints were *issued*, not that fills retired.

Positive times in the table are savings versus paired cold. Two schedule legs used two and three rounds, respectively.

| Arm | Median hit compute saved, µs | Median whole path saved, all cases, µs | Median false-pick whole path saved, µs | Finished by synthetic deadline | Trace-scaled join-clipped diagnostic, µs/token |
|---|---:|---:|---:|---:|---:|
| Untimed hot-one oracle | 62.8 | — | — | n/a | 231.5 (cache-state ceiling proxy) |
| Inline demand at near-window start | 41.7 / 47.5 | 10.9 / 35.3 | −0.5 / −0.5 | 151/152, 228/228 | 158.8 / 193.5 |
| One helper, near-window demand | **54.3 / 57.7** | **40.3 / 26.0** | −0.1 / −1.3 | **147/152, 227/228** | **201.7 / 219.4** |
| One helper, near-window T2 hints | 16.7 / 16.9 | 9.2 / 3.1 | −1.6 / −0.7 | 151/152, 227/228 (issue only) | 72.9 / 60.8 |
| One helper, starts during predecessor | 54.9 / 52.5 | **−17.4 / −6.5** | **−47.1 / −40.6** | 151/152, 228/228 | Not applicable: predecessor interference |
| Cold/cold (schedule A/A) | 7.1 / 0.9 | 6.6 / 2.0 | 3.4 / 1.3 | n/a | 4.0 / 8.1 |

Whole path = preceding CPU branch + synthetic near window + target CPU branch, including issue cost, any inline
overrun, and any memory interference while those run. The near-window placeholder spins on the clock rather than
waiting on a HIP event. Compute = target CPU branch only. The median near-helper read took 128.0 / 129.4 µs once
started. Excluding trials without a CPU predecessor, false picks in the early-helper arm increased predecessor time
by **53.3 / 60.4 µs** median. The best near-window arm avoids that overlap but cannot avoid 33.52 MB/token of useless
reads on this trace if all lines are reached. The CPU-only run does not measure release of that bandwidth back to the
rest of the engine or the effect on concurrent GPU work.

The last column removes the predecessor interval from the paired delta for late arms, then caps positive target compute
savings by that visit's observed CP-02 join proxy; signed costs remain. `report.py` takes the median paired case delta
within each stratum and scales to its suffix population. One early schedule leg spuriously appeared to save predecessor
time for arms that did not start until after the predecessor (e.g. near-helper raw whole-path projection 584 µs/token
vs 294 on repeat). Do not treat that as scheduler gain. The 232 µs/token oracle number assumes a perfectly hot expert
for every correct prediction in an otherwise forced-cold replay; the 202–219 µs/token helper result additionally charges
the near-window scheduling and false predictions, but still assumes synthetic slack and no concurrent GPU activity.
Neither estimate extrapolates to naturally warm targets or to a new placement policy. For scale, CP-01's single
untraced 2K median is 22.64 ms/token; it is not an A/B control for this CPU replay.

**Parity:** all 4,104 timed trials passed an exact FP32 comparison with untimed production-reference outputs on the same
inputs. Both predecessor and target are compared in scheduled arms, 7,904 vectors total. There was no first divergence
and no timed major page fault. This is a CPU operator-chain check, not engine-level accuracy or greedy parity. CPU-only
wall for the three invocations was **0.25 + 2.11 + 3.06 = 5.42 seconds**, about 32.4 user CPU-seconds with a warm file
page cache. Peak RSS was ~738 MiB. No service, GPU, or Git commit was used for these runs.

## Reproduce

From the repo root, with the existing Release build's static ggml archives available. The build and trace summary are
single-command preparation; run **only the three `taskset` commands** in an exclusive CPU slot. With a warm page cache
allow about 10–15 seconds; each command has a 60-second hard limit for slow storage. Never run them beside a live GPU
benchmark. Raw measurements from this run are in ignored `prefetch/out/` (`ceiling`, `schedule`, `schedule-repeat`);
using a new output directory preserves them. The trace exporter intentionally refuses to overwrite an existing case file.

```sh
D=docs/goals/arch-decode-recon/prefetch
T="$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK/trace-on/visits.jsonl"
M="$HOME/.lmstudio/models/unsloth/Inkling-2-Mini/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf"
bash "$D/build.sh" build-poly-hip
O=$(mktemp -d "$D/out/replay.XXXXXX")
python -B "$D/trace_screen.py" "$T" --cases "$O/cases.tsv" > "$O/trace-screen.json"
B="$D/out/prefetch-bench"
FORMAT='elapsed_s=%e user_s=%U system_s=%S maxrss_kib=%M'
/usr/bin/time -f "$FORMAT" timeout 60s taskset -c 0-7 "$B" ceiling "$M" "$O/cases.tsv" 2 15 > "$O/ceiling.csv" 2> "$O/ceiling.log"
/usr/bin/time -f "$FORMAT" timeout 60s taskset -c 0-7 "$B" schedule "$M" "$O/cases.tsv" 2 15 > "$O/schedule.csv" 2> "$O/schedule.log"
/usr/bin/time -f "$FORMAT" timeout 60s taskset -c 0-7 "$B" schedule "$M" "$O/cases.tsv" 3 15 > "$O/schedule-repeat.csv" 2> "$O/schedule-repeat.log"
python -B "$D/report.py" --tokens 89 "$O/ceiling.csv" "$O/schedule.csv" "$O/schedule-repeat.csv" > "$O/results.json"
```

Measured executable SHA-256 `aa50400f80fb73014093d4245348a3d8d3ac5e46b4156079a023039a71c337b6`;
case TSV SHA-256 `0dc4323368f8b5f52cd5afe4c8d6a9a877b6fa6f45c798a99e4bde6116ed0063`.
