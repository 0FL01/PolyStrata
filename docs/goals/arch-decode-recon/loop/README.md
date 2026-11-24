# Loop message hoisting — CPU handoff

**CP-P3-01: PASS (source invariance). CP-P3-02: INCONCLUSIVE for the fresh baseline and individual fused phases.**

The historical RX 6700 XT traces bound the entire fused message/current-input dispatch budget at **0.302–0.325 ms/token** (median across full-token windows, per fixture). Only some of this work is invariant. Eliminating that whole busy time for free would turn the separate historical 2K Bartowski 51.325 t/s into at most **52.196 t/s (+1.70%) in this busy-only scenario**. Even deleting both historical, profiler-inflated gaps around every input dispatch for free maps to only +3.93%. Neither example predicts Tideglass performance. Rank and output accounted for roughly 0.83–0.85 ms/token of additional busy time in the historical controller, but their live dependence keeps it out of this bound.

Raw historical evidence, methodology, launch shape, and limitations: [controller-cost-breakdown.json](controller-cost-breakdown.json). Exact transition, dependency and FP details: [loop-dependency-graph.md](loop-dependency-graph.md).

## CPU reproduction

From the repository root, with the historical local traces still present:

```sh
p=docs/goals/arch-decode-recon/loop
python -B "$p/controller_trace.py" \
  --trace 'short94=~/.cache/claude-dec3/tr-r4/TG_kernel_trace.csv' \
  --trace 'trd2k=~/.cache/claude-fv/trd2k/T_kernel_trace.csv' \
  --trace '32K=~/.cache/claude-fv/final-tr32k-strata-poly-cedd3bf3/T_kernel_trace.csv'
```

The CSV parser validates 95/95, 95/95 and 3/3 LM-head-to-LM-head intervals, respectively; it deliberately refuses the older `tr-c16` dispatch shape. It ignores prefill and intervals with a different controller structure. `input_local_span` is calculated per token before taking a percentile; it need not equal the sum of other percentiles. `kernel_busy` is summed dispatch durations on that single stream, not whole-GPU utilization. The adjacent kernel gaps may contain tracing/queue effects and are not isolated launch measurements.

The generator reads the original `control_in_row`, `control_in{,3}_kernel` and `wave_reduce` from the pinned Git object. It emits a standalone A/D probe, not an engine patch. D prepares all 48 pass rows in one kernel and still pays the current norm and pair write at each visit. It compares raw messages and both halves of the pair byte-for-byte, then times A and D in alternated order plus D's prepass and application separately. This is a hot one-pass microbenchmark with only controller-input kernels and no dense/MoE interleave; it can screen the split, not prove an E2E speedup. Its reference executes QKV/FFN with baseline grid shape; any full-engine claim still requires byte comparisons against the actual baseline release. Offline compilation and linking with the repository's ROCm 7.13 Clang `-O3 -Wall -Wextra -Werror` passed. The generated assembly declares wave32, no private scratch, 117 `next_free_vgpr` in both reference kernels, 100 in grouped precompute and 33 in current norm/apply. These are compiler resource declarations, not measured occupancy.

```sh
# Compile/write only inside the pack while preparing. Do not run without a GPU slot.
p=docs/goals/arch-decode-recon/loop
sdk="$PWD/.venv/lib/python3.13/site-packages/_rocm_sdk_devel"
python -B "$p/generate_probe.py" > "$p/generated_probe.hip"
TMPDIR="$PWD/$p" "$sdk/llvm/bin/clang++" -x hip --offload-arch=gfx1031 \
  --rocm-path="$sdk" --rocm-device-lib-path="$sdk/lib/llvm/amdgcn/bitcode" \
  -O3 -DNDEBUG -std=c++20 -Wall -Wextra \
  "$p/generated_probe.hip" -Wl,-rpath,"$sdk/lib" -o "$p/generated_probe"
model="$HOME/.lmstudio/models/unsloth/Inkling-2-Mini/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf"
python -B "$p/generate_probe.py" --queries-from "$model" > "$p/queries.f32"
# Expected only for this Tideglass file: 442,368 bytes; SHA-256
# 583b63f469b4174ffc14e7daaef0c6bc21e29b7e2e36d5847a6878b411cbfe47
```

## Proposed order if a guarded GPU slot is assigned

1. **Reuse the orchestrator's CP-02 trace** on `c0dd0e52` + Tideglass first. Run the analyzer; this needs no extra GPU time once the CSV exists. Recompute the attributable envelope. Unless there is a substantially larger opportunity or a clear cross-vector reason, keep P3 below larger candidates. A simple pass precompute moves rather than deletes work, **adds two launches/token** and still leaves 48 current-norm launches. Even the more interesting producer fusion only removes 46 net launches and its benefit is unknown. Suggested screen before attempting integration: an isolated measured net saving of about **0.10 ms/token** or more, including prepass and apply, with a paired interval excluding zero on real fixtures; this is a prioritization threshold, not an E2E acceptance claim.
2. If warranted, first run the generated probe guarded with no arguments (synthetic smoke). For real fixtures, make a separate capture build: after each bank publication enqueue nonblocking D2H copies of both `bank*` arrays into per-pass pinned staging; capture the 3 `s3cur_` rows just before `control_in3` and FFN `cur_` before `control_in` for each physical layer. Preserve per-token/per-pass buffers until all copies complete, collect the bytes after the call, and do not use dump-run timings. File format (little-endian binary FP32, no header): `bank0[2304],bank1[2304]`, then 48 current rows in physical 6–17 × Q/K/V/FFN order; 460,800 bytes. Queries use the matching frozen GGUF in the same order. Capture both passes at multiple positions, e.g. 2K prose and 16K code, and before/after a restored continuation. Keep pos/pass/input/model hash in a separate manifest. Use a one-threaded timing harness only after fixture byte parity. Example invocation after assignment:

   ```sh
   python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output "$p/probe-guard" -- \
     timeout 120 "$p/generated_probe" "$p/banks-current.f32" "$p/queries.f32" 80
   ```

   Keep the existing free-VRAM floor and other stops; no `--pmc`. Do not commit fixtures, generated binaries or guard logs. To time individual phases only if needed, compare matched full and partial variants with the same workgroup/tensor shape and compiler flags, and report differential times separately (memory/ALU overlap and changed VGPR allocation make them non-additive). B adds a pass-level bank-scale reduction with the baseline tree, then compares A vs reuse; a boundary-separated `m`-only vs `m+mNorm` prep tests message RMS; a current-only normalization kernel isolates a candidate geometry, not its independent share of A. Rank and gain/controller-output already have unique chain points in CSV; split gain/tanh/apply only if it changes the decision. Keep events out of untraced E2E timings.
3. A positive stage result only justifies a small experimental engine candidate. A loss in standalone D alone does not rule out fusion: use its measured prepass cost against the fresh full-input budget, and only proceed if the remaining budget can cover added producer work and scratch loads. The actual launch-removing variant pairs D with current-norm-in-depth-mix and split-address rank reads; compare raw current, raw message, both pair halves, FP32 rank, controlled output, learned-norm FP32 and Q8_1. On SWA independently verify identical Q/K/V currents before trying RMS reuse. Run matched-placement forced logits and independent raw greedy tokens, then Q3/Q4 and untraced paired timing if Q0–Q2 pass. No integration or GPU run has happened here.
