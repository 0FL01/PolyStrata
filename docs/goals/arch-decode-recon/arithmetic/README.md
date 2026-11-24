# CP-P0-04: expert arithmetic across placement

**The unmodified CPU and HIP paths are not bit-identical on the model-derived proxy.** On the first full-Q4_0 Tideglass layer, the earliest difference was an activation code, not a gate or down accumulator. CPU emulation found it and a guarded RX 6700 XT run confirmed it. Neither result used a captured decode input. A separate capture engine has been linked for that check.

## Input and checks

- Engine source: `c0dd0e52f3975dbed3972deda2c2c75010b330c2`, with no changes to the examined source files at build time. CPU: Ryzen 7 5700X, GCC 16.2.1, Release ggml `GGML_NATIVE=ON` (AVX2/FMA). `STRATA_GGML_DIR` uses the extracted `third_party/llama.cpp` source with no separate Git metadata. The setup pin at the engine source is `3cf03257f219afbe7334045ff7c6a06ac68c627d`; the local x86 `quants.c` SHA-256 is `3c489fdc77e3ab484a5f188bff9c60c0e9483c951ed65c1a562a04c249d5628e`. `c74109e8` is a PolyStrata commit, not llama.cpp provenance.
- Production GGUF: `Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf` (15,537,527,488 bytes; adjacent manifest advertises SHA-256 `7af63a2c9362f0ac6fc708443348ead5dbfcb7ecde8cd454ea3572c68864aa72`). All 24 layers have packed Q4_0 gate/up; layers 0–2 have Q4_1 down, so this test uses layer 3, expert 0, with Q4_0 down. The fixture uses the real Q8_0 embedding of token 64 (`a`), dequantized and passed through layer 3's learned FFN RMSNorm (`eps=1e-6`). This is a **model-derived input proxy**, not an `xn_` captured mid-decode. The selected expert and test weight `0.125f` are not a router record. SiTU `beta=4`, `linear_beta=25` are loaded from the GGUF.
- CPU leaf functions come from ggml's actual `from_float` and `vec_dot` traits; the step-by-step CPU replay matches all 2,304 FP32 result words bit for bit with the actual weighted `cpu_experts()` call with eight workers (one expert, one row).
- GPU model follows the source and offline gfx1031 Release ISA: Q8_1 raw-sum and code packing, Q4_0's DP4A groups of 16 codes in two lanes, two rounds for 72 input blocks, four waves folded in order then the 32-lane XOR tree; the 36-block down keeps the fourth wave's zero add. The device code uses `v_fma_mix_f32` for the block correction and its accumulation. The fused input norm packs codes from rounded `xn_`, but the emitted `v_fmac_f32` combines this lane's final norm multiply with its rounded XOR-16 peer in the first raw-sum step. The model reproduces that instruction separately from standalone packing. No claim of reproducing OCML `tanhf`/`expf` bits: the host model uses host libm in the SiTU expression, so the reported downstream emulation numbers are diagnostic.

SHA-256 of the four GGUF byte slices consumed for the default replay (whole file not rehashed):

| Tensor slice | Byte count | SHA-256 |
| --- | ---: | --- |
| `blk.3.ffn_gate_up_exps.weight`, expert 0 | 2,985,984 | `61762fdc287e43ce02021fecfb9f2c0d81612a7fc2c8f4100d839e97e5ade5e1` |
| `blk.3.ffn_down_exps.weight`, expert 0 | 1,492,992 | `9f31ad99e7fbf7a8f842f7d7cb8abd3e7512eb955120b2c1b8c4993e1489c801` |
| `token_embd.weight`, token 64 | 2,448 | `2b26673137c42f643e171d49f2dcd310bb697931c6b85085c64a35b21eafb8e6` |
| `blk.3.ffn_norm.weight` | 9,216 | `a6afdf96b06faf861bbfec18d01614138c0698e44505414c0f2af629525d2f1c` |

## First divergence

The Q8_0/Q8_1 formats differ by construction: Q8_1 stores a half-rounded raw sum that Q8_0 does not. Comparing only their shared fields, all 72 scales matched and **4/2,304 codes differed**. The first is block 50, lane 3, index 1603. `xn = 0xbf5f5cd9` (`−0.872510493`) and `amax = 0x4090badb` (`4.52280951`); both scales pack to `0x288f`. CPU reciprocal-multiply and emulated GPU divide both produce exactly `−24.5` (`0xc1c40000`) for this lane. The AVX2 ties-to-even code is **`0xe8` (−24)**; Q8_1 `roundf` ties away from zero, hence **`0xe7` (−25)**. The format-only sum field is not counted as a differing code.

| Boundary | Different | First CPU vs CPU-emulated HIP bits |
| --- | ---: | --- |
| Gate FP32 | 1,152/1,152 | row 0 `0xbf19d0ac` vs `0xbf18fce2` |
| Up FP32 | 1,152/1,152 | row 0 `0x3fedd645` vs `0x3fec2b2e` |
| SiTU FP32 | 1,152/1,152 | row 0 `0xbec88ecd` vs `0xbec683b1` |
| Intermediate scales/codes | 36/36, 427/1,152 | block 0 scale `0x2985` vs `0x2986`; lane 2 `0xb1` vs `0xb0` |
| Down FP32 | 2,304/2,304 | row 0 `0xbed0aed9` vs `0xbec88956` |
| One weighted contribution | 2,304/2,304 | row 0 `0xbd50aed9` vs `0xbd488956` |

Copying CPU scales/codes into the GPU-model input **while retaining the GPU-model raw sums** still changes 1,152/1,152 gate rows: row 0 `0xbf19d0ac` vs `0xbf19623c`, max absolute difference `0.07166117`. CPU centers each nibble as an integer before the dot. The GPU model uses two half-block terms of the form `d_w * (dp4a(q_unsigned, q8) * d_x - 4 * s_raw)`. Even in block 0, with no differing codes, half `s_raw = 0x417c` (`2.7421875`) is not `d_x * sum(q8) = 2.73028564`. As a second counterfactual, center the integer nibbles in the **GPU** model and keep the GPU accumulator tree. The max difference falls to `9.53674316e-7`, but 956/1,152 gate rows still differ (row 0 `0xbf19d0ac` vs `0xbf19d0b4`). Matching codes is insufficient, and moving the offset into the integer dot does not reproduce the CPU reduction order. See [raw CPU evidence](cpu-proxy-token64.txt).

## Guarded actual HIP result, same proxy

The orchestrator ran the linked `parity-hip` on AMD Radeon RX 6700 XT (`gfx1031`). All 2,304 proxy RMSNorm FP32 values matched. CPU vs actual two-slot HIP confirmed the same activation code and all downstream first differing words and counts in the table above. Actual generic `native_mmvq` and specialized two-slot HIP matched at every reported boundary; fused vs separate HIP activation and intermediate packs also matched bit for bit on this proxy.

CPU emulation matched actual HIP in the entire input pack, gate and up. Its first mismatch with actual HIP was **SiTU index 14: `0x3ce02c46` vs `0x3ce02c49`**, with **90/1,152** values differing (max `9.53674316e-7`). No differences survived intermediate quantization for this input; the entire packed `hq`, down and single weighted contribution then matched actual HIP again. The SiTU difference rules out treating host libm as an exact reference for arbitrary GPU activations.

Local evidence: `~/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK/guard-proxy/command.log` (SHA-256 `1a1d099db5c1fdb287a57e3417b8c9c7cf2f86f23388f50d15df804c97d41d40`), `stop-reasons.json` was empty, and the 22 memory-fault warning records were all from earlier in the boot. Minimum sampled free VRAM was 10,428.5 MiB. No inference timing can be taken from this run.

## Reproduction

Run from repository root. `build.py` uses the existing Release gfx1031 Ninja build and ggml archives; standalone outputs are under `arithmetic/out/`. `--isa` emits assembly only. Neither `--hip` nor `--isa` calls the runtime or starts a GPU job.

```sh
A=docs/goals/arch-decode-recon/arithmetic
MODEL="$HOME/.lmstudio/models/unsloth/Inkling-2-Mini/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf"
python "$A/build.py"
"$A/out/parity-cpu" "$MODEL" 3 0 64 0.125
python "$A/build.py" --isa
python "$A/build.py" --hip
```

**Do not run the following without an orchestrator-granted slot.** The standalone HIP probe uses the actual fused norm and fused SiTU→Q8_1, `native_inkling_q4_0_slots` (two slots with the same expert to enter the production specialization), `native_mmvq` (one resident expert fallback), separate SiTU and Q8_1 diagnostic launches, down, and actual `add_scaled()` for one contribution. It compares proxy norm bits; then CPU and both device arms use the same FP32 input. It also compares both device arms and verifies the duplicate slot results. The local production config has both `STRATA_INKLING_Q4_0_SLOTS=1` and `STRATA_INKLING_TAIL_MOE=1`; this probe runs the slots path but not the tail convolution or the mixed top-8 sum. It does not touch the service. Probe binary SHA-256: `041fe9c8f5d8e86410da6224a02b3c1a9bdaa0a2c10f68cd6e03e1cca5578f92`.

```sh
python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults \
  --output "$A/out/guard-parity-1" -- \
  timeout 60 "$A/out/parity-hip" "$MODEL" 3 0 64 0.125
```

## Actual serial decode fixture: position 2048, visit 6

[`relink_capture.py`](relink_capture.py) verified the underlying detached worktree has exactly the published CP-02 trace patch on `c0dd0e52`, then applied [`capture-xn-xq.patch`](capture-xn-xq.patch) to a scratch copy of its `inkling2.cu`. It rebuilt that one HIP object and relinked the existing CP-02 Release objects into `arithmetic/out/capture/strata-poly`. Capture engine SHA-256 `7235a945270e6449d2905458dcad41c8c00a00786e663edda82a01123141974a`; original CP-02 executable SHA-256 unchanged (`93027e25d7f3723c69bec32e7efbaa0eab6c25344fa07bf3a3d644fb3ed1dc62`). Offline `ldd` with the production library paths resolved all libraries. No engine sources or objects were overwritten and the linked engine was not run here.

```sh
A="$PWD/docs/goals/arch-decode-recon/arithmetic"
WT="$HOME/.cache/opencode-tmp/opencode/arch-cp02-trace-wt"
BASE="$HOME/.cache/opencode-tmp/opencode/arch-campaign.XWM1yK"
python -B "$A/relink_capture.py" "$WT"  # already linked; repeat only when needed
D=$(mktemp -d "$HOME/.cache/opencode-tmp/opencode/arch-cp04-capture.XXXXXX")
python -B "$A/capture_run.py" prepare "$D"
```

The ready-to-use private directory for this handoff is `~/.cache/opencode-tmp/opencode/arch-cp04-capture.tBbJzJ` (directory `0700`, config `0600`). Its production config copy adds only `STRATA_CP04_CAPTURE=$D/run/input.bin`, `STRATA_CP04_POSITION=2048` and `STRATA_CP04_VISIT=6`. Use this dir as `D`, or create a fresh one as above. **For the orchestrator’s granted slot only:**

```sh
python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output "$D/guard" -- \
  timeout 300 .venv/bin/python docs/goals/arch-decode-recon/baseline_run.py "$D/run" 2048 \
  --binary "$A/out/capture/strata-poly" --profile-seed "$BASE/profile.seed" \
  --config "$D/capture-config.json" --steps 2 --prompt-csv "$BASE/mixed-2048/prompt.csv" \
  --visit-trace "$D/run/visits.psv"
```

After capture, [`capture_run.py`](capture_run.py) validates the guard, capture marker and exact 11,808-byte fixture (FP32 `xn_` then packed `xq_` from the fused kernel), verifies complete PSVST1 frames, and selects `ids[0]` and `weight_bits[0]` **in router order** from input position 2048, logical visit 6 (physical layer 6), regardless of the expert's original placement. It converts the binary32 word to an exact C hex float and derives the GGUF path from the completed run metadata. Both probe builds use the captured pack verbatim on the GPU comparison arm. The default is CPU-only command preparation; actual HIP requires a granted slot and always uses the guard.

```sh
python -B "$A/capture_run.py" replay "$D"                          # prints the guarded HIP command
python -B "$A/capture_run.py" replay "$D" --backend cpu --execute  # CPU replay of the captured fixture
python -B "$A/capture_run.py" replay "$D" --execute                # only with permission, guarded HIP
python -B "$A/test_capture_run.py" -v                              # CPU-only synthetic reader/config tests
```

This run is for numerical evidence only. The capture adds a blocking copy; do not treat its timing as a performance measurement.

## Placement decision

- **Do not assume a move preserves arithmetic.** Even on inputs with equal activation codes, the raw-sum correction and different accumulation trees can change results. Q0 and Q1 were checked on the proxy only; released placement changes are not qualified. No engine, sampling, state or token-ID parity was attempted here.
- A backend-neutral optimization preserving today's semantics needs a *shadow* reference assignment and the reference CPU and GPU arithmetic variants, including their individual ordered sums and original merge of CPU/GPU buckets. Preserving top-8 order but summing in new device buckets is insufficient. For GPU-reference work moved to CPU, consume the original fused `xq_` and validate both projection leaves before attempting an OCML-exact SiTU and its quantizer. For CPU-reference work moved to HIP, target the existing AVX2 integer-centering, eight FP32 accumulator lanes/FMA and horizontal reduction, plus CPU quantization and SiTU semantics. Neither is a one-line rounding fix.
- **Timing / placement upper bound: not measured.** No runnable cross-backend exactness variant or hardware timing has been qualified. The next atomic experiment is the guarded capture and Q0/Q1 replay on its first real selected expert. After localizing the first difference on that fixture, test a minimal CPU-reference HIP leaf: generate Q8_0 bytes with the AVX2 reciprocal-multiply and ties-to-even convention, then feed those bytes into integer-centering, eight independent FP32 FMA accumulators in block order and the AVX2 horizontal tree. Check input bytes and gate/up FP32 separately before accepting any leaf timing for CP-P0-05. Matching this leaf would not by itself qualify SiTU, down, ordered aggregation or end-to-end placement.
