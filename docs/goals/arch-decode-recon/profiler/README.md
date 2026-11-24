# CP-01 / CP-P3-02 — Serial decode trace

Source: `c0dd0e52` (`398027f8` changes no engine code). Use the campaign’s HIP Release build and Tideglass v2.1 on the RX 6700 XT (`gfx1031`). This is **traced attribution**, not the CP-01 untraced throughput test.

## Run one trace (orchestrator GPU queue only)

From the repository root, use the same untouched config, binary, profile seed, and exact prompt as the matched CP-01 leg. Use a new `trial` path. The guard creates its output and its parent; the harness creates `run`.

```sh
D="$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK"
trial="$D/profile-2048-r1"
config="$HOME/.config/strata/inkling2-mini/strata-inkling-2-mini.json"
python -B rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output "$trial/guard" -- \
  timeout 180s .venv/bin/python -B docs/goals/arch-decode-recon/profiler/collect.py \
  "$trial/run" 2048 --steps 32 --workload mixed --binary "$PWD/build-poly-hip/strata-poly" \
  --config "$config" --profile-seed "$D/profile.seed" --prompt-csv "$D/mixed-2048/prompt.csv"
```

The guard reports AMDGPU memory/page faults as warnings. A reset, hang, timeout, OOM, low free VRAM or failed command still invalidates the run. The orchestrator owns GPU serialization and service state; `baseline_run.py` refuses to run with the service active. Keep the 2,048 MiB engine reserve and the guard's 1,536 MiB floor.

`collect.py` launches `.venv/bin/rocprofv3` using `--hip-runtime-trace --kernel-trace --memory-copy-trace`. Do not use `--pmc`, PC sampling or `--sys-trace`. On this SDK, aggregate `--runtime-trace` also enables KFD probes, which are unnecessary here. `baseline_run.py` replaces the child's environment, so wrapping the outer Python process would lose rocprof setup. We wrap the engine call instead, with the campaign `lib_dirs` first in `LD_LIBRARY_PATH` and the matching `_rocm_sdk_core/lib` appended, following the archived local recipe. The actual engine binary remains the one hashed in `config.json`; the full wrapper arguments, effective library path, and rocprof version are in `trace/launch.json`. CPU-only preflight with this library order reported rocprofv3 1.3.0 / ROCm 7.13.0. Raw capture includes startup and prefill; the one request, fixed length, step cap and timeout bound collection. The CPU report discards startup, prefill and teardown rather than relying on an attach timer.

After the guard exits successfully, run on CPU (can overlap the next independent task):

```sh
.venv/bin/python -B docs/goals/arch-decode-recon/profiler/analyze.py \
  --run "$trial/run" --untraced "$D/mixed-2048" > "$trial/controller-cost-breakdown.json"
```

## Reading the output

- **Window:** The interpreter validates one logits stream and `decode_steps+1` final `softcap_kernel` dispatches. The first belongs to the prompt. Adjacent logits D2H completions give decode windows; the copy may be a trace record or the immediate `__amd_rocclr_copyBuffer` dispatch. If any copy boundary is missing, use all softcap ends instead and treat the endpoint as approximate. `baseline_run.py` separately checks that no prefill tokens were reused and no drafts were offered/accepted. Read the unmatched sequence counts, the 96 controller stream rows/token check, and `boundary_matches` before accepting quantitative results.
- **GPU:** Busy is the union of dispatch intervals across queues, not their sum. `kernel_idle_ms_per_token` is wall minus kernel union; `uncovered_by_kernel_or_copy_ms_per_token` also subtracts the copy union while counting copy/kernel overlap once. It is not a measurement of CPU expert join, driver launch gaps or whole-card idle. PCIe bytes are reported only where the copy CSV includes size; mapped host writes and untraced transfers are not attributed. Inspect `kernels_by_name` where a category is too broad. `conv_add_kernel` can include multiple roles and is not assigned entirely to MoE. Q4_0 expert slots and grouped dense segments are distinct symbols.
- **Controllers:** `control_in{,3}_kernel` includes bank loads/RMS, query dot, softmax/mix, message normalization **and live residual normalization**. `control_out{,3}_kernel` includes both gain/forget math and application. The matching rank-64 projection is attributed only inside the adjacent input→projection→output sequence on the logits stream. Unfused historical sequences allow separate message/current normalization and rank/output attribution but their mix still combines several steps. Do not sum the entries of `stage_sources_not_additive`: those are mappings into shared dispatches. Grid/workgroup, VGPR, SGPR, and LDS are available; hardware occupancy is not.
- **Host:** HIP launch/sync API durations are separate host-side unions; neither sync nor launch wall duration is a CPU cost to add to GPU spans. Graphs can group kernels and copy work under one correlation. `controller_only_correlated_launch_calls` is a conservative conditional count, not one host call per controller and not a measurement of exposed launch latency. Per-operation timing within a fused kernel, true occupancy and exposed CPU joins require separate evidence.
- **Ceiling:** `hoist_input_dispatch_union_ms_per_token` allots the **entire** fused input dispatch (including unrelated live normalization) to the hypothetical change, or the bank mix + message norm on the unfused path. `hoist_input_exclusive_device_ms_per_token` drops overlap with any other observed dispatch/copy. The factor `traced_window/(traced_window − hoist_input_union)` assumes that dispatch time disappears for free and translates 1:1 to traced wall savings. This is an optimistic bound **on the measured input-dispatch work**, not an untraced E2E prediction and not a bound on unrelated changes to launch, overlap or rank. The [source dependency graph](../loop/loop-dependency-graph.md) establishes what may move and which FP32 operations still need byte parity.
- **Baseline comparison:** `--untraced` requires matching config, binary, prompt and profile-seed hashes and reports common-prefix IDs and mean arrivals. This is an instrumentation perturbation check, not optimization parity; if IDs differ, report the first zero-based divergence and consider live placement variation. Do not apply the traced factor to the full untraced throughput number.

## §5 evidence handoff

Protocol: [goal §5](../../2026-11-24-inkling2-architecture-aware-decode-recon.md#5-context-pack-orchestration-protocol), [CP-01](../../2026-11-24-inkling2-architecture-aware-decode-recon.md#cp-01--fresh-baseline), [CP-P3-02](../../2026-11-24-inkling2-architecture-aware-decode-recon.md#cp-p3-02--baseline-controller-profiling), [campaign manifest](../research-manifest.md).

- **STATUS:** CPU method tested; fresh CP-01/P3-02 measurement pending the GPU queue. **HYPOTHESIS:** previous-pass bank work occupies part of the serial controller-input slot; use a dispatch ceiling before investing in an exact split.
- **METHOD / REPRODUCTION_COMMANDS / ARTIFACT_PATHS:** see above. Retain `$trial/guard/{warnings.json,stop-reasons.json,vram.jsonl}`, `$trial/run/{config.json,metrics.json,prompt.csv,engine.log,trace/}` and `$trial/controller-cost-breakdown.json` together in the private campaign directory. `trace/launch.json` records profiler provenance. Do not commit raw traces or user config.
- **MEASUREMENTS:** CPU replay of three archived RX 6700 XT CSV sets, each with 23 decode steps. Not Tideglass v2.1, not the pinned binary, and not fresh evidence; these validate parser behavior only. All had 24 logits-copy blits and exactly 96 controller stream rows/token, with no unmatched controller sequences.

  | Archive | Controller path | Traced window | Kernel busy (union) | Controller dispatch (union) | Input-removal budget | Traced factor |
  |---|---|---:|---:|---:|---:|---:|
  | `inkling-perf.g6YsYt9b/trace-csv` (Nov 21) | Unfused | 52.464 ms/token | 22.753 ms/token | 4.041 ms/token | 2.022 ms/token | 1.0401× |
  | `dspark-speed-1795285637/trace-0` (Nov 21) | Fused one-stream | 39.828 ms/token | 19.487 ms/token | 1.852 ms/token | 0.660 ms/token | 1.0169× |
  | `dspark-speed-1795285637/trace-1` (Nov 21) | Fused one-stream | 39.185 ms/token | 19.121 ms/token | 1.865 ms/token | 0.651 ms/token | 1.0169× |

  Reproduce each from `$HOME/.cache/opencode-tmp/opencode/<Archive>` with `analyze.py --steps 23 --kernel <archive>/trace-TG_kernel_trace.csv --copies <archive>/trace-TG_memory_copy_trace.csv --agent-info <archive>/trace-TG_agent_info.csv`. The first archive's launch recipe is `inkling-perf.g6YsYt9b/{trace.sh,bench.py}`; the second's historical decode-slicer is `dspark-speed-1795285637/trace_summary.py`. Neither archive contains HIP API records; a CPU fixture checks host API nesting and overlap instead.
- **BASELINE_COMPARISON / FIRST_DIVERGENCE / ACCURACY_EVIDENCE:** pending fresh trace. No engine/model changes or optimization parity claim. **MEASURED_UPPER_BOUND:** fresh dispatch ceiling pending; archived numbers above are not campaign sizing.
- **INTERPRETATION / NEXT_ATOMIC_EXPERIMENT:** run the matched 2K traced leg and confirm the expected fused QKV/FFN count and head boundaries. Decide from the fresh dispatch budget whether to invest a GPU slot in a separately instrumented bank-only vs live-only timing split. Do not attempt occupancy counters on gfx1031.

CPU checks: `python -B docs/goals/arch-decode-recon/profiler/test_analyze.py`.
