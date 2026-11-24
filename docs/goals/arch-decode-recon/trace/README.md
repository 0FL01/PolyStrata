# CP-02: Inkling logical-visit trace

The [instrumentation patch](cp02-instrumentation.patch) is based on `c0dd0e52`. It is a research patch, not part of the
normal engine build. With no trace variable, it makes no additional HIP calls, takes no additional timestamps,
allocates no trace structures and writes no files. It adds null checks and an optional timing argument to the CPU
expert path; it does not change the router, placement policy, expert math, launch order of model work, or state format.

## Build and run

Apply the patch in a detached worktree at `c0dd0e52`, not on the production source tree. Follow
[`docs/AMD_HIP.md`](../../../AMD_HIP.md) for the HIP toolchain and build `strata-poly` in Release for `gfx1031`.
The CP-02 build used the installed ROCm SDK, the main checkout's existing llama.cpp source, and these CMake options:

```sh
git apply --check /path/to/cp02-instrumentation.patch
git apply /path/to/cp02-instrumentation.patch
cmake -S . -B build-cp02 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1031 \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_CONVERSATION_TESTS=OFF
cmake --build build-cp02 --target strata-poly -j 6
```

Use one GPU job at a time and get the orchestrator's go-ahead first. The easiest way to preserve production settings
and the exact prompt and profile seed is the private CP-01 runner. From the main checkout, set `BIN` to the instrumented
Release binary, `SEED` to the same read-only expert profile seed as CP-01, and `OUT` to a new absolute directory:

```sh
python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output "$OUT" -- \
  python docs/goals/arch-decode-recon/baseline_run.py "$OUT/run" 2048 \
  --binary "$BIN" --profile-seed "$SEED" --steps 256 \
  --visit-trace "$OUT/run/visits.psv"
```

For a direct invocation, set `STRATA_INKLING_VISIT_TRACE=/absolute/new/path.psv` in the guarded child environment.
`STRATA_VISIT_TRACE` is the CP-01 runner's alias. A pre-existing output path is rejected. Optional
`STRATA_INKLING_VISIT_TRACE_FIRST` and `STRATA_INKLING_VISIT_TRACE_END` are inclusive/exclusive zero-based input
positions. Set `FIRST` to the prompt length and `END` to prompt length plus the desired number of decode steps to
exclude single-row prefill. For this to be a serial trajectory, disable speculation (`--window 1 --lookup 0 --no-draft`),
use greedy decoding (`--temperature 0`) and keep the other baseline settings. A last emitted token is sampled but not
fed back into the model, so it has no visit frame. Quit the process normally and inspect the guard status and warnings
before using the trace. The trace contains routing data for the prompt trajectory; keep it in the private run folder.

## Read it on CPU

```sh
python docs/goals/arch-decode-recon/trace/collect.py "$OUT/run/visits.psv" \
  --summary "$OUT/run/visit-summary.json" --jsonl "$OUT/run/visits.jsonl"
python -B docs/goals/arch-decode-recon/trace/test_collect.py -v
```

Without options, the collector validates the file and prints a summary by visit and CPU expert count. It accepts
gzip-compressed input with a `.gz` suffix. See the [schema](schema.md) for row fields, timings and cache events.
`python -B docs/goals/arch-decode-recon/trace/test_collect.py /new/path/sample.psv` writes a synthetic, one-token sample
for checking consumers; it is not RDNA2 evidence.

## Controls and limits

- Compare the unmodified c0dd0e52 binary to the trace binary with tracing **off** (changed code layout), then compare the
  trace binary **off/on** (instrumentation overhead). Repeat in interleaved order with the same GPU, software, model,
  prompt CSV, profile seed, greedy flags, and guard. Record full-run decode and steady-state results separately: the
  first traced token pays one-time event and slot-snapshot setup. The CP-01 runner emits raw greedy IDs and per-step
  arrival deltas. Compare IDs before analyzing throughput; if live swaps first diverge, report the index and repeat
  with matched placement, e.g. `--expert-swaps 0` as a separate control, not as a substitute production baseline.
- The trace has up to six HIP events per visit and a 13.5 KiB write per token frame. It waits for no extra per-visit
  GPU synchronization, but the HIP probes, host scans, deferred timing queries and file writes may perturb scheduling.
  Use traced timings for decomposition only; do not use them as an untraced E2E throughput result.
- This patch captures only single-row Inkling calls entering `run_visits()` with 24 physical layers and top-8 of 128.
  Multi-row prefill and speculative windows are not traced. It does not trace individual GPU kernels, hardware
  utilization, LM-head timing, PCIe DMA timestamps, or CPU worker intervals. Cache events are observed engine state;
  a swap can complete on the copy stream before its event is observed by `follow()`.
- Compare residency with the same request history. This trace does not log `grow_cache()`, `fill_now()`, arbitrary
  resize/refill internals, exact policy score counters, or explicit session/reset IDs. Use a fresh process and one
  serial answer for replay. The per-visit resident masks and selected offsets are ground truth at routing time; do not
  silently extrapolate the event ledger across unobserved cache changes.

Validation at handoff: HIP Release `gfx1031` built and the synthetic CPU collector tests passed; real-device sample,
token-ID parity, and traced/untraced overhead require the orchestrator's guarded GPU slot.
