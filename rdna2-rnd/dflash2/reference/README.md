# Inkling 2 Mini DFlash2 CPU reference

This is an offline test of the **drafter**, not a decoder or throughput benchmark. It uses real committed
Tideglass target features and held-out next-token labels from a greedy width-one decode. The draft
weights come from `incoai/Inkling-2-Mini-DFlash2` revision `4f9088ea66c21f2b99ed657d1941c306b03e84d2`.
All arms borrow the same Tideglass target Q8_0 embedding, F32 embedding norm and Q6_K output head.
Draft weights and shared-head weights are dequantized for FP32 arithmetic. BF16 names the **checkpoint**,
not the arithmetic type. CPU replay does not model RDNA2's activation quantization or accumulation.

## Run

Use x86-64 with AVX2, FMA and F16C, CMake, Ninja, a C++ compiler and Python with NumPy. The bridge builds
only the vendored ggml CPU libraries; Python imports the vendored `gguf-py`. The full matrix reached
6.53 GiB RSS with four threads on Ryzen 7 5700X (the dense shared head alone uses 1.72 GiB). Keep the originals and all generated artifacts in
the private run directory, not in git.

From the repo root:

```sh
R="$HOME/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon"
REF=rdna2-rnd/dflash2/reference
OUT="$R/cpu-reference"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
umask 077
bash "$REF/build_bridge.sh" "$OUT/bridge"
BRIDGE="$OUT/bridge/libggml_qbridge.so"

# Full weight parity, head cross-check and synthetic SWA wraparound.
.venv/bin/python -B "$REF/dflash2_ref.py" --bridge "$BRIDGE" --selfcheck \
  --dump-json "$OUT/selfcheck.json"

# Real source-vs-GGUF tensors, serial-vs-batched proposals, and tap-48 check.
.venv/bin/python -B "$REF/capture_states.py" --bridge "$BRIDGE" \
  --domains code_python tool_calls --offsets 0 8 --output "$OUT/states"

# Smallest real replay, then the five-checkpoint matrix over seven domains.
.venv/bin/python -B "$REF/replay_matrix.py" --bridge "$BRIDGE" --profiles BF16 source \
  --domains code_python --max-anchors 4 --batch 4 --output "$OUT/smoke"
.venv/bin/python -B "$REF/replay_matrix.py" --bridge "$BRIDGE" --output "$OUT/matrix"

# Same-history DSpark from the separate forced-target captures.
.venv/bin/python -B "$REF/compare_dspark.py" --replay "$OUT/matrix" \
  --output "$OUT/paired-dspark.json"

# Change only one tensor family (or KV storage) at a time on common anchors.
.venv/bin/python -B "$REF/attribute_precision.py" --bridge "$BRIDGE" \
  --anchors 12 --output "$OUT/precision"
```

Use `--target`, `--taps` and `--draft-dir` (`--dflash2-dir` in the single-trace CLI) when the inputs
are not at the default paths. The prepared profiles are `BF16`, `Q8_0`, `Q4_0`, `Q4_0-mixed` and
`Q4_K-mixed`; [`tools/convert_inkling2_dflash2_to_gguf.py`](../../../tools/convert_inkling2_dflash2_to_gguf.py)
can recreate them from the safetensors source and the target GGUF. The one-trace CLI also has sparse
per-tensor overrides and a Q6_K/Q8_K head diagnostic; the main matrix keeps FP32 activations fixed.

## Alignment and checks

- DFT1 is `DFT1` followed by little-endian `(int64 absolute_pos, int32 token_id, int32 is_gen,
  float32 taps[4][2304])` records. The probe takes pre-visit partial-block residuals at **5, 23, 41**
  and the final post-norm state at **48**. These are visit indices, not physical layers or +1 offsets.
  Compare `Inkling2::run_visits`/`output_rows` in `src/poly/inkling2.cu` and the capture resolver in
  the checkpoint's `sglang_inkling_dflash2.py`. The three early taps' ordering is checked against source;
  `capture_states.py` additionally checks tap 48 against real next-token labels using the target head.
- The four taps go through `fc` and RMSNorm. Context K/V uses that result directly, without a layer's
  pre-attention norm or dynamic convolution. Context K, and block Q/K, use full-head half-split RoPE at
  absolute positions. The eight block rows (anchor plus seven masks) see one another bidirectionally,
  plus the valid part of the preceding 2048-slot context ring. Only committed target features enter
  the ring. The batched path projects multiple anchors together but masks out every true context row
  at or after each anchor. The single-step path is separately implemented for comparison.
- Borrowed token embeddings go through Inkling's embedding norm. The two-tap convolution (`prepare`
  and `finish`, see [`z-lab/dflash`](https://github.com/z-lab/dflash/blob/main/dflash/model.py)) uses the
  coefficients of the current **destination** row and a zero lag input at the anchor. After the five
  layers, only positions 1 through 7 go to the shared head, truncated to 200,058 valid tokens. Each
  top-16 unary list gets its learned predecessor/successor correction before the `30*tanh(s/30)` cap,
  as in the checkpoint's SGLang adapter. The next selector step uses the previous **proposed** ID.
  Exact unary ties favor lower token IDs, including across the top-16 boundary; the sequential
  selector keeps the first candidate on an exact corrected-score tie.
  This is an independent FP32 implementation of the donor math, not a bitwise donor/SGLang runtime check.
- The matrix validates prompt/gen flags and all recorded IDs against `taps/manifest.json`. The final
  emitted token has no DFT1 row because it is not fed back into the target. It can still be a truth
  label, never a tap. At anchor `p`, feature positions end at `p-1` and the labels start at `p+1`.
- The old `.dspark.log` files diverged from the DFT1 history in six domains: the old server probe did
  not use `--follow`. `matrix.json` labels comparison to those logs as prefix-only. For the fair
  comparator use `paired-dspark/<domain>/{result.json,proposals.log}` generated by
  [`../live/measure.py`](../live/measure.py) in `dspark-log` mode with all original generated IDs forced.
  `compare_dspark.py` verifies those IDs and every log anchor. The installed DSpark path narrows its
  offered width at the generation limit; comparison uses only the common full-width anchors.

`matrix.json` includes histograms, prefix survival, offset-wise matches, top-16 truth recall and
selector gains/losses versus unary-only. The paired BF16 diagnostics match candidate scores by token
ID rather than by rank. Proposal TVD compares the union of candidates and is reported both for each
arm's realized predecessor and over same-predecessor rows only. The per-domain NPZ files store anchor
positions, labels, unary and selected IDs, candidate IDs, and closed-loop row indices. `states/`
separately holds real single-anchor state bundles (both original weights and BF16 GGUF): fused context,
cached K/V, visibility mask, normalized embedding, layer deltas/Q/K/V/outputs, final hidden and raw
head logits, selected candidates/corrections/scores/probabilities/IDs. `parity.json` records differences.

Here `emitted_per_verify` means **1 + number of matching consecutive draft tokens**. `all_positions`
uses every full-width anchor; `closed_loop` advances by `1 + accepted` using each method's own proposals.
It excludes the partial-width tail and does not execute or time target verify. An advantage in either
metric is not necessarily a speedup. Only one capture represents each domain; the synthetic `agent_traj`
is a scripted multi-turn prefix, not an interactive agent run. Overlapping anchors are not independent
samples. Treat small differences descriptively; none of these checks establishes distributional or
sampled-token parity, or acceptance above the captured context lengths.

`attribute_precision.py` samples evenly spread anchors from each domain and keeps the truth, tap
positions and FP32 activation path fixed. The Q4_0 repairs borrow the named family from the Q8_0
checkpoint without double quantization. The Q4_K-mixed downgrades re-quantize only the named family
directly from BF16. Context KV tests change only the stored committed K/V rows (not the ephemeral
eight-row block). Report the paired prefix deltas and draft/hidden fidelity together; these small
samples do not rank near-tied formats reliably.

## Literal llama.cpp CPU graph check

The optional probe compiles the **actual vendored function bodies** for `build_dflash2_conv`,
`build_dflash2_selector`, `build_attn_mha` and `is_masked_swa` into a CPU ggml test library. The
extractor adds the exact post-correction transform from [llama.cpp PR #33049](https://github.com/ggml-org/llama.cpp/pull/33049)
to the older vendored selector. It can verify that fragment against a local copy of the PR diff.
The C++ adapter wires the Inkling checkpoint into those graph functions and runs the non-flash
attention branch. It recomputes fusion, layer K/V, all five layers, the shared head and selector.
It does not load or run a full llama.cpp session, PyTorch, or SGLang, and it does not test a real
donor KV cache or the target model. The backbone/head use dequantized F32 weights; the selector
gathers real BF16 codebook rows. Its inputs are the two domains and four positions captured above.

```sh
cmake -S "$REF" -B "$OUT/bridge" -G Ninja -DCMAKE_BUILD_TYPE=Release -DDFLASH2_DONOR_PROBE=ON
cmake --build "$OUT/bridge" --target ggml_qbridge ggml_dflash2_donor --parallel 4
.venv/bin/python -B "$REF/probe_donor.py" --states "$OUT/states" --output "$OUT/donor"

# Optional provenance check; no download required if the original diff is present.
.venv/bin/python -B "$REF/extract_donor.py" third_party/llama.cpp "$OUT/donor/provenance.inc" \
  --inkling-diff "$R/llamacpp-pr33049.diff"
cmp "$OUT/bridge/literal_donor.inc" "$OUT/donor/provenance.inc"
```

`donor/parity.json` has source and graph hashes, per-stage error, and lattice checks. The donor
top-16 order is not the reference's sorted order; compare candidate support by ID and compare
scores at the **same predecessor**, not the same slot or each run's potentially divergent path.
The NPZs save the packed lattices and full backbone states. The probe also scores frozen inputs
with unary-only softcapping and with no softcap as negative controls. Matching greedy IDs on those
controls do not establish score parity.
