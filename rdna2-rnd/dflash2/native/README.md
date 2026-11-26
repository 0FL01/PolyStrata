# Native Inkling 2 Mini DFlash2 replay (gfx1031)

This research executable proposes tokens from captured Inkling target taps. It does not run a target verifier.
Its `closed-loop` mode advances along prerecorded target tokens by `1 + accepted_prefix`, and `all-positions`
advances by one. Neither timing includes a live target. Keep profile and trace fixed when comparing proposals.

Build using a configured PolyStrata HIP build for `gfx1031`, with `strata_kernels` and `ggml-base` already built:

```sh
R="$HOME/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon"
bash rdna2-rnd/dflash2/native/build.sh "$R" build-poly-hip
```

This builds `$R/dflash2-replay` and `$R/libstrata-dflash2.so`. Example with a real target tap capture:

```sh
M="$HOME/.lmstudio/models/unsloth/Inkling-2-Mini"
out="$R/runs/native-bf16-check"
flock -x "$R/gpu.lock" python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults \
  --output "$out" -- "$R/dflash2-replay" \
  --target "$M/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf" \
  --model "$M/DFlash2/Inkling-2-Mini-DFlash2-BF16.gguf" \
  --replay "$R/taps/code_python.dft1" --mode both --max-steps 9 \
  --dump-json "$out/replay.json" --dump-steps "$out/steps.jsonl" --dump-hidden
```

`--dump-hidden` adds flattened FP32 `[7,2304]` hidden states to each step. Omit diagnostics for replay timing.
`compare_replay.py --native STEPS.jsonl --cpu-matrix PROFILE-DOMAIN.npz` compares candidates and greedy proposal IDs
by the same absolute position. Supply `--states DIR --domain DOMAIN --profile PROFILE` when corresponding CPU
intermediate dumps are available. Q4_0, Q8_0, Q4_0-mixed, Q4_K-mixed and BF16 are supported. Repo BF16 MMVF takes
FP32 activations; quantized projections, including the Q6_K target head, use the repo Q8_1 activation path.
Candidate IDs can differ from the reference, which projects quantized weights with unquantized F32 activations.
Q5_K, Q6_K, IQ4_NL and IQ4_XS projections use the existing MMVQ dispatch too.

A DFT1 suffix may start at a nonzero **absolute** position, provided it has at least 2048 contiguous warmup
records (`is_gen=0`) before the first proposal. Replay keeps the last 2048 prompt records and uses their original
absolute positions for RoPE. It neither zero-fills missing history nor renumbers the suffix.

## Runtime boundary

See [`../live/draft_abi.h`](../live/draft_abi.h) for the C ABI. One handle represents one sequence on device 0.
`df2_append` waits for its private stream before returning; callers must finish producing all four device tap planes
before calling. `df2_propose` is deterministic greedy with the full bidirectional eight-position noise block,
even when the caller requests fewer proposals. A non-null borrowed head is the target's raw resident Q6_K
`output.weight` and is neither duplicated nor freed. Model paths must be local single-file GGUFs.
The first `df2_append` may start at a nonzero position if it supplies at least 2048 contiguous warmup rows.

The native runtime reuses PolyStrata's HIP matrix kernels and owns a 2048-row FP32 K/V ring per layer. The noise
block K/V are ephemeral. It projects only the real 200058 target vocabulary rows. The GPU reduces unary logits
to the raw top 16 per position. By default the CPU walks the greedy path using sparse rows of the mmap-backed
selector codebooks. The post-correction softcap is also used for reported candidate probabilities at diagnostic
`T=1`. For opt-in GPU placement or a small decoded host row cache, see the [native codebook comparison](../codebooks/NATIVE.md).
