# CP-P1-03/04 — 64K follow-up: close CPU prefetch

**No-go; no further engine runs.** The fixed previous-pass, CPU-only prefetch policy falls short of 2 ms/token on the
disjoint 64K mixed trace. The measured one-helper recipe projects **0.338–0.356 ms/token**, conditionally on the
2K CPU calibration transferring. Giving each hit **200 µs** of saved compute—above the largest paired per-case
median of the 2K untimed hot oracle—and charging nothing for prep or false picks still gives just **1.072 ms/token**
at cap one. Allowing up to eight predictions per visit under the same generous assumptions gives **1.813 ms/token**.

## What was observed on the 64K trace

CP-02 `trace-mixed-65536/visits.psv`, SHA-256 `cb1b743fdb63e591ffb0b5d6e34d91f25b01ac375d172f085f2f80f4678a973f`.
Tideglass Q4_0, RX 6700 XT / Ryzen 7 5700X (8 CPU workers), Q8_0 KV, 200,000 context capacity, 2,048 MiB VRAM reserve,
64 swaps, no draft. The binary and configuration hashes match the 2K CP-02 trace, but the 64K mixed prompt differs
(SHA-256 `8f441694af24e9535905a5362957db6c77aad3dda83511786033d2eecbd4328a`). All 256 serial frames at positions
65536–65791 passed the CP-02 parser. No predictor fitting: rank the previous pass's actual CPU experts, exclude source
resident/pending IDs, take the first `cap` and use the next pass's actual route only to score. Only the 24 targets in
core passes 2 and 3 are considered. No pick was resident/pending by the target route or selected on the GPU.

| Cap | Predictions | True CPU hits / recall | False positives / share | Total / false payload, MB/token | Entire join proxy on hit visits, ms/token | Generous 200 µs/hit, ms/token |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,857 | 1,788 / 13.7% | 3,069 / 63.2% | 84.98 / 53.70 | 1.905 | 1.072 |
| 2 | 8,081 | 2,752 / 21.1% | 5,329 / 65.9% | 141.39 / 93.24 | 2.229 | 1.553 |
| 4 | 10,561 | 3,353 / 25.8% | 7,208 / 68.3% | 184.78 / 126.11 | 2.333 | 1.790 |
| 8 | 10,936 | 3,427 / 26.3% | 7,509 / 68.7% | 191.34 / 131.38 | 2.347 | 1.813 |

The targets had 13,017 CPU expert uses and 4.143 ms/token of exposed GPU join marker time. The entire join column
deletes this marker gap on any hit visit even if multiple CPU experts remain cold. It is a deliberately loose envelope
on that trace metric, not an assertion that all of it is removable. In particular the wide-cap envelope *does* cross
2 ms/token on the full capture. The last column instead assumes all correct predictions are hot and on time, adds
their full hypothetical compute savings and clips at the **visit's** measured join; it ignores every cost. This is a
sensitivity check, not a measured wide-cap hardware bound. Even 150 µs/hit in that check yields only 0.854 ms/token
at cap one and 1.547 at cap eight. In the final 89-token suffix, cap-one false payload stays at 53.95 MB/token; the
cap-one and cap-eight *entire join* envelopes fall to 1.520 and 1.870 ms/token respectively.

## What transfers from the CPU-only experiment

- **Hot-state oracle:** the short replay's one prewarmed expert saved 62.8 µs median compute (p90 80.8 µs, maximum
  181.5 µs across the 19 paired case medians). Apply 62.8 µs to every actual hit, clip per visit, and assume no prep
  overhead: **0.394 ms/token** on the full 64K capture, **0.360** on the suffix. This is a cold→hot estimate, not
  attainable scheduler gain.
- **Near-window helper:** transfer the two CPU legs' measured median hit savings (54.3/57.7 µs) and post→complete
  latencies (135.3/136.9 µs median, 150.8/151.3 µs p90 among completed reads, including wakeup). Launch at target
  `host_begin`, require a complete read before `route_ready`, assume no policy overhead or simultaneous engine
  contention. Either timing band would fit all 4,857 cap-one reads in this static replay: traced near-window
  p10/p50/p90 is 268/328/727 µs. Gross clipped gain is **0.341–0.362 ms/token**. Charging the existing near-helper
  false-pick median after-predecessor penalty (0.28/0.54 µs/false) leaves **0.338–0.356**; the suffix gives
  **0.309–0.324 ms/token**. This accounts only for the measured CPU-only penalty, not competing GPU or other work.
- **False traffic and overlap:** cap one makes about 12 useless reads per token (53.70 MB of payload, or about
  1.67 GB/s at this trace's 31.17 tok/s if fully read). Bytes are payload touched, not hardware-measured DRAM traffic.
  83.2% of issued visits have a CPU-positive predecessor versus 67.9% on the short suffix. Earlier 2K experiments
  already measured regressions when a helper reads during that CPU work. Issuing at the source is not a substitute:
  median lead is 6.92 ms with 22 intervening CPU expert uses on hits, about 98.5 MB of weight touches before reuse
  (not distinct lines or DRAM bytes; L3 capacity is 32 MiB).

The CPU timings come from a short-trace fixture using real weights, synthetic activations, forced-cold targets and
no GPU. The transfer to 64K is conditional: a different working set can change cache state, worker timing and actual
slack; the late-helper fixture does not measure GPU contention, live DDR traffic, or prefetching several experts.
No claim extends to a different predictor, tier migration, or 128K/200K. The measured recipe and the generous check
are both well under the gate, and the actual cap-one hit-visit marker envelope is under it even without calibration.
Close this branch rather than spending an engine slot to reduce uncertainty about a sub-threshold candidate.

## Reproduce offline

From the repo root; no build, benchmark invocation, GPU access or service change. This reuses the existing
`ceiling.csv`, `schedule.csv`, `schedule-repeat.csv` and corresponding `.log` files from the CPU screen.

```sh
P=docs/goals/arch-decode-recon/prefetch
D="$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK"
python -B "$P/long_context.py" "$D/trace-mixed-65536/visits.psv" "$P/out" > "$P/out/long-context.json"
```
