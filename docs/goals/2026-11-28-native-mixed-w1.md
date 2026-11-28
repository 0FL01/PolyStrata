# Inkling-2-Mini: RDNA2 native mixed-quant MoE decode

Status: active. Source: [original request](mixed-w1/request.md), 2026-11-28.

## Finish line and initial plan (frozen before implementation)

Improve **time per generated token**, not nominal GEMV throughput, on the RX 6700 XT and the unmodified Tideglass-OrcaRouter-Inkling-2-Mini-UD-IQ4_XS-Ablit49 GGUF. Work on the smallest unresolved outcome; replan when a measurement changes the critical path. Do not ship a benchmark-only win.

| Outcome from the request | Acceptance / primary evidence | Status |
|---|---|---|
| R1. RECON, upstream and previous experiments | Identify the executed W1 chain and component times on this model, distinguish upstream formats and devices from ours. | Verified below |
| R2. Prototype, measure, review, iterate | Test competing device and hybrid execution hypotheses; compare same-prompt greedy IDs and uninstrumented decode. Record negative results and the next falsifiable experiment. | In progress |
| R3. Commit, push and deploy successful changes | A repeatable E2E win at comparable placement, qualifying coding/long or prefix-return runs, reviewed HIP Release, tested production and rollback. No new binary in production without evidence. | Pending |

1. Start from `3be84557`, the installed `0db4d974` text worker, production settings and private replay inputs. Trace W1 separately from prefill. Compare moving and fixed placement. Freeze the cache and total expert+KV budget for greedy comparisons.
2. Compare real routing and projection shapes to Strata's dedicated engine. Measure scale domain, memory traffic versus unpack/dot, ISA/resources, and a fused resident producer with unchanged mixed down. Do not assume SF4 and native NVFP4 have the same layout or math.
3. Use the exposed branch to select the next experiment (resident arithmetic, host misses, placement or overlap). Compare a candidate and stock in mirrored order; time without tracing. Repeat on a longer prefix/tool continuation. Use a general agent for independent review.
4. Integrate only a qualified change, gate across the RDNA2 family, rebuild, test and compare IDs, push to main and roll out with a rollback. Reprofile before the next experiment. Preserve GGUF, KV/session formats, other backends and serving auth.

Envelope: `strata-poly` and directly used HIP/host kernels, qualification notes and goal. Raw weights, traces, profiles, private sessions and experimental binaries stay outside Git. Guard serial GPU work with `rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults`; independent writable files in `~/.cache/opencode-tmp/opencode/polystrata-runs/mixed-w1-20261128/`.

## Starting evidence

The last two code rollouts targeted [prompt MMQ](2026-11-28-native-nvfp4-mixed-prefill.md) and [ROCm host memory](2026-11-28-rdna2-expert-upload-stall.md), not W1. The earlier [slot-select campaign](2026-11-24-inkling2-decode-implementation.md) covered a different GGUF, including NVFP4 down. Its select switch is not set in the Ablit49 production config. The [donor campaign](native-donor-recon/kernels.md) already rejected promoting a Q4 projection DPP leaf win and found multi-row reuse required actual shared expert rows. Neither is a W1 result on Ablit49.

## What the 5090 does, and what the Radeon actually runs

Donor: [Strata public source `4560d037`, v0.2.23](https://github.com/Niko1221/Strata/tree/4560d03793482ef367294cfbd86a4df8d5679e72/engine/src/inkling2). Do not compare default-policy speculative throughput to one-token decode. On the desktop 5090 the resident W1 fast path uses device top-eight, FP16 input, direct gate/up+SiTU, direct down, then reduction+ShortConv. FP4 tensor-core execution at `M=1` is not the default. Native FP4 conversion gave [179.0 → 203.2 tok/s fast, 139.4 → 153.5 exact](https://github.com/Niko1221/Strata/commit/4ee366e7152aabbad553591b5b78d8bf263e3d1c) drafter-off; the commit does not identify an Ablit49 GGUF. Selected graph edges [helped](https://github.com/Niko1221/Strata/commit/a5aa58b1ba2d06136e3857c287f93def3c07430e); starting every successor early hurt. On hybrid CUDA visits the device publishes cold jobs to the host after top-eight. Upstream HIP still synchronizes for host routing data on hybrid visits. Splitting CPU work by rows helped upstream [SQ-10 on R9700](https://github.com/Niko1221/Strata/commit/6cb548ffb1bf8c4589812802045b7089b5bdeae6); PolyStrata already does that.

`ink_gguf_loader.h` interleaves gate/up and separates codes and scales for SF4. The lossless adapter requires **both** joined GU and down to be NVFP4. `hip/ink_hip_codecs.hip.h`'s raw table does not cover NVFP4 and Q5_1. Fallback conversion to S_K on HIP or SF4 on CUDA changes weights. Its RDNA2 tiled `B >= 96` path is not W1. A separate lossless GU representation with unchanged down is possible here; below we measure rather than assume it needs slots, a loader or a new GGUF.

Ablit49 visits layers 0–5 once, 6–17 three times, then 18–23 once. The 48 visits share 24 physical layers' weights, not recurrent state. In `Engine::moe` the host waits on the router event; dual GEMV has already written scores and normalized FP32 input to mapped RAM. Host sigmoid, bias and top-eight decide route and normalized weights. `L.slot[e]` chooses the backend. The GPU uses a shared Q8_1 input, 16 lanes per output row for multi-resident NVFP4, multiplies gate/up by their expert scales **before** SiTU and packs 32 values per Q8_1. Down is IQ4_NL on 31 visits, Q5_1 on 17. In particular W1 `vec_dot_q5_1_q8_1` computes its min term from the **quantized codes**, not `ds.y`; do not borrow prompt MMQ's half-sum reasoning. CPU misses take ggml Q8_0 input for NVFP4; down uses CPU Q8_0 or Q8_1 according to its type. The CPU pool submits the resident branch as task zero of its GU batch. The tail joins the ordered resident outputs and the separate ordered mapped CPU sum. Moving an expert changes arithmetic even at identical top-eight.

## RX 6700 XT / Ryzen 7 5700X measurements

ROCm 7.13, production Ablit49, eight CPU threads, greedy. Fresh workers with independent copies of the same profile. `cold2048` is a **slice** of a saved coding conversation, not a graded task. Time between first and last wire ID excludes the prefill prediction. ROCm traces and host clocks below are diagnostic runs, not throughput samples.

| 2K baseline | Untraced (96 steps) | GPU hit share | Traced kernel sum / step | NVFP4 multi-slot GU | Gap immediately before MLP `conv_add`, summed / step |
|---|---:|---:|---:|---:|---:|
| Frozen swaps, 4800 MiB expert+KV cap | 26.86 tok/s | 54.5% | 14.04 ms | 2.02 ms | 20.61 ms |
| Moving slots, 2600 MiB reserve | 30.84 tok/s | 85.5% | 15.84 ms | 2.88 ms | 9.03 ms |

Frozen IQ4_NL/Q5_1 multi-slot down: 1.16/0.53 ms per token; SiTU/pack: 0.124 ms. In a separate 48-step moving run host NVFP4 GU+SiTU+launch callback occupied 6.16 ms/step, other GU 0.87, IQ4_NL and Q5_1 down+sum 2.61 and 1.12. These phases overlap the GPU. The gap includes driver/submission/scheduling; it is **not** a CPU FLOP count. A fully resident fusion cannot collect all of it.

### Memory, arithmetic and wave32

Full byte census (`scale-scan.json`): **849,346,560** GU scales, across all 20 physical NVFP4 tensors × 128 experts, all between 76 and 126 inclusive. This verifies the affine normal shortcut for **this file**, not the loader's general input. The exact path must handle exponent zero, `0x7f` and high bits according to the existing codec.

Seven-slot synthetic joined GU, 2304×2304, median of 31 HIP event samples (µs); useful weight payload 20.90 MB. The 192 MiB flush touches disjoint lines; not a hardware cache-miss counter.

| Control / full projection | Hot | After flush |
|---|---:|---:|
| Read/fold packed words | 30.12 | 69.80 |
| Unpack and signed dot, no FP32 weighting | 35.08 | 70.20 |
| Full branch / select | 44.04 / 40.48 | 74.24 / 70.72 |
| Affine, exceptional bytes checked | 39.00 | 70.64 |
| Shift into half, share ×128 in the activation scale | 37.48 | 70.52 |
| Unchecked normal affine (ceiling only) | 36.20 | 70.24 |
| SWAR sign application in place of permutes | 48.92 | 72.12 |

The GPU oracle checked all 256 scale encodings, 65,536 unpack inputs and 16,128 output words per full synthetic arm. Shifted-half uses subnormals and explicitly handles `0x7f`. In this leaf: select → shared-half 205 → 189 static instructions, 49 → 44 VGPR, zero scratch; SWAR 24 → 8 permutes but 62 VGPR and slower. The production L16 branch and old select kernels are different: 401 / 341 instructions and 37 / 62 VGPR. Hot loads leave room for shortening the arithmetic; after a flush the quickest full projection is within 1 µs of the read control. The controls change code generation and are not additive.

**Real weights and activations.** `layout.cu` reads all 128 joined matrices of physical layer 0. For each of 64 captured activation rows it routes eight experts; four candidates each matched all **1,179,648** stock FP32 words. The new direct RDNA2 decoder uses packed byte selects, signed dot4 and the original L16 accumulation order. A separate equal-size layout aligns 16 code bytes for each K/32 and stores the two scale bytes in a trailing plane. Down and GGUF are untouched. Median µs over six mirrored samples of 128 dispatches:

| Eight slots | Stock L16 | Raw, exact | Raw, normal only | Planes, exact | Planes, normal only |
|---|---:|---:|---:|---:|---:|
| Same eight | 71.41 | 41.45 | 36.92 | 39.11 | 35.00 |
| Rotate across 128 | 83.09 | 73.56 | 73.15 | 72.53 | 72.31 |
| Captured routes | 75.81 | 53.74 | 51.17 | 51.79 | 49.75 |

Pack once on GPU: 3.45 ms for the 128 matrices. Captured routing here repeatedly accesses **one** layer, not 48 visits of a forward pass. This does not justify paying repack on every cache insertion. Screen the exact raw path in the worker first.

### Whole-chain screens

- **Selective spill of a CPU miss**: one reusable 6 MiB device slot, copy one authoritative registered host expert on visits with ≥3 misses, keep permanent slots unchanged. Fixed control/candidate/candidate/control mean ms/step: **38.20 / 45.99 / 34.77 / 37.18**. Both candidate runs first differ at ID index **96** of 97. A resident/host split changes dot and sum order. Rejected for production.
- **Fused resident `NVFP4 GU → scaled SiTU → Q8_1`**: 256- and 512-thread CTAs produce complete 32-channel blocks, share activation loads and pass standard Q8_1 directly to unchanged down. 43/63 VGPR, 256 B LDS, no scratch versus 37 VGPR and no LDS for stock L16. Same-binary A/B/C/C/B/A with frozen slots: **36.135 / 35.749 / 36.058 / 35.689 / 35.736 / 35.677 ms/step**. Same 97 IDs and 20,570 resident rows. The faster-looking 256-thread path lost against the second control: removing one dispatch does not resolve the join.
- **Expert-ready CPU down**: pack each GU tile and let a CPU expert's down tiles start before other experts finish GU; keep original route-rank sum. Both down types, 216 CPU-only visits, 497,664 FP32 outputs exact. Frozen A/B/B/A **36.281 / 35.765 / 36.115 / 36.436 ms/step**, 97 IDs exact. Small effect; must hold with live placement, not just the low-hit case.
- **Exact raw RDNA2 GU alone**: at fixed 4800 MiB expert+KV cap, moving slots, 2K, deterministic prefill (`--stage-ahead 0 --cpu-share 0`, prompt prefetch off), three control/candidate pairs: **26.594/26.574, 26.734/26.490, 26.737/26.456 ms/step**. All 193 IDs and 58,949 resident rows equal. Much smaller gain than the isolated leaf. Hold out real task prompts and hot returns before considering a rollout.

Run root and specific artifacts: `mixed-w1-20261128/{guard-*,layout.cu,layout_build.py,raw.py,bench.py,diag.py,probe_spill.py,fused/,host/,tasksbench.py}`; standalone controls and disassembly: sibling `nvfp4-w1-micro.4vk9/`. The architecture and implementation review used independent general agents. Their proposed join, donor and arithmetic hypotheses above are distinguished from measured wins. No candidate in these screens requires changing weights on disk or the KV/session format.

**Holdout decision, raw GU:** moving slots, 4300 MiB cap, same stable prefill, four workers in A/B/B/A order. Mean ms per step on independent 16K source-review: **32.466/32.210/32.052/32.483**; on the following 160-token append (16,383 rows reused, 161 fresh): **33.428/35.098/34.237/34.228**. Python: 26.691/26.074/26.393/26.317, logic: 26.050/26.296/25.598/25.821, tool-result replay: 29.131/29.033/29.079/29.864. All 455 greedy IDs per arm matched (97 each on four requests, 67 on the last). The append is a captured token slice, not a graded tool workflow. The raw GPU path alone is **not** a deterministic production win across these workloads.

## Current checkpoint

R1 closed. For R2 compare the expert-ready CPU schedule under the same held-out traffic; test a simpler two-phase schedule which keeps the CPU dot exactly as installed, then a specialized CPU NVFP4 producer if it beats the linked ggml implementation. Leave production on the previous executable unless the untraced worker and actual serving workload qualify a successor. Keep the runnable artifacts and document the next action if none does.
