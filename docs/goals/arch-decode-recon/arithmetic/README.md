# CP-P0-04: expert arithmetic across placement

**Verdict: the CPU-reference HIP leaves passed on both fixtures; exact expert placement is not qualified.** On RX 6700 XT, the prototype matched Ryzen AVX2 Q8_0, gate/up and isolated down bit for bit. Existing HIP SiTU still differed in 58/1,152 FP32 words on the captured row, even with exact gate/up. The differences did not survive Q8_0 on these fixtures; no weighted aggregation or end-to-end parity was tested. The captured-row pack + gate/up also cost **22.878 µs/expert** versus **14.190 µs** for native single-expert HIP (p50). No further prototypes or integration unless the causal placement screen passes.

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

### Guarded replay of the captured decode row

The orchestrator completed both the capture and the replay on RX 6700 XT (`gfx1031`). At position 2048, visit 6, physical layer 6, router rank 0 was **CPU expert 116**, with weight `0x3f94809c` (`0x1.290138p+0`). The 11,808-byte `run/input.bin` has SHA-256 `586b088a7e4cdcba8cb5a2e1bbed419f60de61cd7f0e9466606bcd7876a2466c`. The CPU and actual HIP agreed on all **72 shared scales and 2,304 codes**. The first mismatch was gate row 0:

| Boundary | Different | First CPU vs actual HIP bits |
| --- | ---: | --- |
| Gate FP32 | 1,152/1,152 | row 0 `0x3e49071c` vs `0x3e4aa5cc` (max abs `0.011149466`) |
| Up FP32 | 1,152/1,152 | row 0 `0xbb5ff2e0` vs `0xbbd6be80` (max abs `0.0130605102`) |
| SiTU FP32 | 1,152/1,152 | row 0 `0xb9c0e934` vs `0xba3a9a6f` |
| Intermediate scales/codes | 33/36, 411/1,152 | block 0 scale `0x13c2` vs `0x13bb`; lane 0 `0x00` vs `0xff` |
| Down FP32 | 2,304/2,304 | row 0 `0x3cf7dc54` vs `0x3cfe4e78` |
| One weighted contribution | 2,304/2,304 | row 0 `0x3d0fc7e6` vs `0x3d138520` |

The generic and two-slot HIP paths matched throughout. Even with centered nibbles, keeping the emulated GPU reduction order changed 904/1,152 gate words (max abs `2.38418579e-7`). Comparing emulation of the unmodified HIP path to actual HIP, SiTU first differed at row 1 (`0xbdbc1a56` vs `0xbdbc1a58`); 74/1,152 SiTU words differed, but the intermediate pack, down and one weighted contribution matched on this fixture. Local `replay-guard/command.log` SHA-256 `645aad5a432a8b964932ea093e807d08f5cdb29a7e5a2387fde58754e6a6d3ae`; both guard stop lists were empty, with only 22 earlier-boot memory-fault warnings each. Minimum sampled free VRAM was 2,002.6 MiB for capture and 10,429.8 MiB for replay.

## Bounded CPU-reference HIP leaf prototype

[`cpu_ref.cu`](cpu_ref.cu) is standalone. It packs GGML Q8_0 using FP32 `amax/127` for the half scale and `127/amax` multiplied by the input before ties-to-even rounding. Each group of eight GPU lanes reproduces the eight AVX2 dot lanes: center the Q4_0 nibbles as integers, signed dot of four bytes per lane, FP32 combined scale, one FMA per lane per block in order, then the AVX2 `4,2,1` horizontal tree. Offline gfx1031 ISA confirms correctly rounded division code, `v_rndne_f32`, signed `v_dot4c_i32_i8`, separate scale product and accumulation, and three FP32 horizontal adds. This targets finite model inputs and the measured Ryzen AVX2 / Q4_0 path only; not Q4_1 down or other input domains.

On **each** fixture the [scalar CPU model](cpu-ref-model.txt) and the [guarded HIP prototype](cpu-ref-rx6700xt.md) matched every packed AVX2 scale/code and every FP32 gate/up word: 72 scales, 2,304 codes, 1,152 gate and 1,152 up. The isolated down leaf fed exact CPU SiTU also matched all 36 scales, 1,152 codes and 2,304 down words. In the separate SiTU chain test, existing HIP SiTU differed in 115/1,152 words on the proxy and 58/1,152 on the real row. On the real row the first difference was index 51, AVX2 `0xbe122059` vs HIP `0xbe12205a`; the peak absolute difference was `2.98023224e-8`. The next Q8_0 pack and down output matched on both fixtures despite these SiTU differences.

```sh
A=docs/goals/arch-decode-recon/arithmetic
D="$HOME/.cache/opencode-tmp/opencode/arch-cp04-capture.tBbJzJ"
python -B "$A/build.py" --cpu-ref
python -B "$A/build.py" --cpu-ref --isa
python -B "$A/build.py" --cpu-ref --hip
python -B "$A/capture_run.py" cpu-ref "$D" --backend cpu --execute
```

**Reproduction only in an orchestrator-granted GPU slot:** the linked `out/parity-hip-ref` (SHA-256 `8a6222229f7cf3f1edf8b5710eccfda7a8c250a0bde6e1eca31c7b517572ee29`) checks both inputs in one guard, then times the leaves only if both pass. The original `cpu-ref-guard/` under `D` holds the completed run. Use a fresh `--guard-output` for a repro; omit `--execute` to print the command only.

```sh
python -B "$A/capture_run.py" cpu-ref "$D" --guard-output "$D/cpu-ref-guard-$(date +%s)" --execute
```

- **Leaf correctness:** compare candidate Q8_0 and gate/up to AVX2, then replay the dot with exact AVX2 bytes supplied separately. Also test the down leaf with exact **CPU** SiTU supplied separately. Any mismatch fails the probe before timing.
- **SiTU chain:** test the existing HIP SiTU with now-identical gate/up, then the candidate Q8_0/down. These diagnostics are independent of the leaf pass. No attempt to port host libm here, and no claim of FP32 SiTU parity if quantization happens to hide a mismatch.
- **Latency scope:** HIP event time per expert for prepacked gate/up and standalone pack + gate/up, candidate versus single-expert `native_mmvq`, same device-resident inputs and weights. Seven batches of 48 repetitions; all sample values and p50/min/max logged. Hot repeated weights, not a multi-slot or engine estimate. No transfers, SiTU, down, weighted sums or tail aggregation included in timing.

## Placement decision

- **Do not assume a move preserves arithmetic.** Even on inputs with equal activation codes, the raw-sum correction and different accumulation trees can change results. Q0/Q1 replay pins down these differences on the proxy and one real row, not arbitrary inputs. No candidate engine, state or token-ID parity was attempted here.
- A backend-neutral optimization preserving today's semantics needs a *shadow* reference assignment and the reference CPU and GPU arithmetic variants, including their individual ordered sums and original merge of CPU/GPU buckets. Preserving top-8 order but summing in new device buckets is insufficient. For GPU-reference work moved to CPU, consume the original fused `xq_` and validate both projection leaves before attempting an OCML-exact SiTU and its quantizer. For CPU-reference work moved to HIP, target the existing AVX2 integer-centering, eight FP32 accumulator lanes/FMA and horizontal reduction, plus CPU quantization and SiTU semantics. Neither is a one-line rounding fix.
- **Measured leaf cost, not a placement speedup bound.** On the captured row, the CPU-reference gate/up dot costs **+7.729 µs/expert** prepacked and **+8.688 µs/expert** with standalone packing versus native single-expert HIP, at p50. The proxy shows similar overhead. This is one hot resident expert at a time, not a cold transfer, multi-slot, or full-engine timing. Use the [measured ranges](cpu-ref-rx6700xt.md) in the CPU-only causal placement screen; full-expert cost and end-to-end gain remain unmeasured. Stop here unless the screen passes.
