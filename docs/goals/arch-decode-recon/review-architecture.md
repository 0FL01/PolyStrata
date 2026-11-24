# CP-R-03 — Independent architecture review

**PASS for bounded serial trace evidence on the sampled RX 6700 XT (`gfx1031`); INCONCLUSIVE for untraced timing or
placement parity.** Reviewed the [CP-02 patch](trace/cp02-instrumentation.patch), [schema](trace/schema.md), and
[CP-04 capture patch](arithmetic/capture-xn-xq.patch) against `c0dd0e52`. CPU-only review; no GPU run. Neither patch is
a production optimization or an integration sign-off.

| Area | Finding |
|---|---|
| HIP lifetime and bounds | The trace rejects split/batch and unsupported geometry. One non-block `run_visits(1)` fits the 48 event/row entries, the top-8 records and the 128-bit masks; the swap limit fits the 16-bit totals. The 288 timing handles are lazily allocated, always recorded on `st_`, and read/reused after an existing stream sync (`run`, `eval`, `window`, or teardown). Teardown destroys them before the stream. No new per-visit host synchronization. |
| Threads | Only the inference thread mutates trace rows and writes the file. The optional GPU launch task records its two markers and publishes any HIP error before `Workers::run` returns (release/acquire on `left`); its caller checks that error before advancing the visit. The follower changes atomic `at` from Asked to Sent, not `L.slot` or pending slot identity; a scan is not an atomic snapshot of all swaps. No added race found on the supported path. |
| CP-02 off / guard | No extra HIP call, timestamp, trace allocation or file I/O with both trace variables unset; branches and the CPU-expert signature still change. The sampled ON and OFF legs used the same binary, prompt and seed, and all 257 greedy IDs also match the unmodified leg. That is not bitwise route, logits, state or performance parity. One ON/OFF pair observed 40.780 vs 42.328 t/s; the older unmodified leg had five fewer slots and long outliers, so it does not isolate disabled-path cost. |
| CP-04 copy and off path | The hook reads the fused norm's `xn_` and original `xq_` on the producer stream before routing consumes them or a later visit reuses scratch. One row is 9,216 + 2,592 = 11,808 bytes. The host vectors are pageable (the installed HIP header documents synchronous copies for unpinned host memory), and capture explicitly synchronizes before writing. Use only for bytes and match an ordered route **from the same capture run**; the blocking D2H copies, sync and file write change handoff timing. When unset, CP-04 still adds a `getenv` to each `layer_head`; do not use that binary as an unmodified throughput baseline. |

### Limits that affect the evidence

- **A join is not a CPU stall or a copy measurement.** `gpu_join_ms` is a gap between stream markers, not a reading of
  the CPU clock. With no routed GPU experts it starts before host routing/dispatch; with `STRATA_INKLING_TAIL_MOE=1`
  its end precedes the fused tail that sums the cached GPU rows and directly reads pinned `h_cpu_`. The cost of
  this mapped-host read is outside the span; there is no separate H2D copy. Marker submission, residency scans, and HIP default
  event fences can perturb the measured schedule; the installed HIP header documents the system-scope fence and its
  cache effects. The sample's **5.752 ms/token** summed join is diagnostic, not measured recoverable wall time or a
  speedup bound. Do not add it to CPU branch spans or interpret event spans as GPU busy time.
- **Observed uploads are not DMA timestamps.** On the sampled serial path `settle_swaps()` waits on the upload stream
  before claiming slots at the next token. All 1,044 asks were observed as arrivals one position later; timing the
  interval would include the inference cadence, not just transfer. At position 2048 the 17 `initial_pending` entries
  duplicate asks already in that position's event log because the snapshot follows `follow()`. Use the snapshot as the
  replay start. No KV growth or shrink occurred in this sample; it does not test those paths or long-context reserve.
- **Trace output failure is not recoverable in place.** CP-02 can throw from `fwrite` after removing a victim or
  assigning an arriving slot, but before publishing `kAsked` or completing the free/counter updates. A partial event
  allocation can also leave the first handle set and cause a retry to skip initialization. The pipe server's request
  reset clears token state, not that cache or the event table. Abort the research run on a trace error, don't continue
  on that engine. A close failure is logged only, and the format has no footer: check the engine log and verify the
  exact expected position range, not merely collector or child exit status.

### Independent CPU checks

- Read `arch-campaign.XWM1yK/trace-on/visits.psv`: 256 consecutive frames at `[2048,2304)`, 12,288 visits, all
  serial/non-frozen. Replay from the initial snapshot matched all resident masks and selected slot offsets.
- ON/OFF each recorded 22 **boot-only** AMDGPU memory-fault warnings, no stop reasons, and sampled free VRAM minima
  of 1,998.42 / 1,999.41 MiB with 2,048 MiB engine reserve and the unchanged 1,536 MiB guard floor. Warning mode
  does not excuse hard failures; sampled VRAM is not a continuous minimum. Use an explicit child `timeout` on future
  runs: the guard monitors faults and VRAM but sets no wall-clock deadline of its own.
- Both patches passed `git apply --check` (including CP-04 on the patched CP-02 worktree). The collector, capture-run,
  and guard CPU suites passed 4, 4, and 5 tests, respectively. No new capture, HIP validation or candidate parity test
  was performed.
