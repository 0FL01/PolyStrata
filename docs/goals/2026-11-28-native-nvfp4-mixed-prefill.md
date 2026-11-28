# Goal: Native NVFP4 and mixed-quant prefill on RDNA2

Status: complete for the measured recovery, research and qualified deployment
Source: 2026-11-28 user request, “Native NVFP4 & Mixed-Quant Prefill Acceleration RDNA2”
Last updated: 2026-11-28

Restore full prefill throughput on Tideglass-OrcaRouter-Inkling-2-Mini-UD-IQ4_XS-Ablit49, then improve cold and hot prefill without re-quantizing the weights. Use measurements to choose experiments, not a fixed order.

## Required outcomes

| ID | Source / acceptance | Primary evidence | Status |
|---|---|---|---|
| R1 | Verify the running binary, process settings, physical/logical dispatch, GGUF layout, and cause of 91.7 tok/s; recover the strongest existing baseline. | Same live ELF, same weights and token IDs, paired MMQ settings; per-layer eligibility; production incident and wait traces. | Verified dispatch cause and recovery; exact incident not traced |
| R2 | Profile the restored baseline and prototype native NVFP4 mixed-quant execution, including temporary lossless packing and gate/up reuse. Keep improvements only with reproducible gains. | Real-weights and activation microbenchmarks, resource/ISA inspection, wall/device timeline, cold 2K/8K/16K/64K, greedy IDs. | Verified on gfx1031; no new kernel qualified |
| R3 | Investigate adaptive narrow prefill under real agent workloads, including long tool returns and reused histories. | Hot appends 32–2048, placement/routing, cache and checkpoint correctness, TTFT, stream gaps, decode, memory/waits, time to correct result. Cover compact SWA and the interaction with DFlash2. | Verified on the guarded protocol and SSE fixtures; no new hot policy qualified |
| R4 | Build, commit, push, and deploy qualified changes with rollback. Consider safe automatic selection for supported RDNA2. | HIP Release, controlled greedy IDs/state parity and live agent smoke, commit, `origin/main`, installed executable/config and rollback. | Verified |

## Constraints and change envelope

- RDNA2 gfx1030–gfx1036 only; measure on RX 6700 XT gfx1031. Preserve source GGUF bytes and model math; report any greedy divergence. Do not expand complete experts to INT8 or requantize to Q4_0.
- Main surfaces: `src/poly/{engine,inkling2,mmq*,conversations}*`, directly used kernels, focused build/check integration and measured documentation. Leave Qwen and other architectures alone.
- Match token IDs, KV format, geometry, and placement where isolating a kernel change. Separate diagnostic timings from uninstrumented throughput. Isolate copies of the profile and session files from production.
- Run guarded experiments sequentially, using `--warn-amdgpu-faults`. Keep the production service loopback-only with existing authentication and preserve rollback files. No subagents.

## Production cause and restored baseline

The process ran `strata-poly-352beccd` (SHA-256 `246f94d30d55efe5e3ad801afae0cbbcb534f055ffb8895f12b6eba2c6f7b26e`). It used growing Q8_0 KV with 200K capacity, batch 2048, 2048 MiB reserve, 8 host threads, 64 swaps, exact compact SWA, 4 GiB hot parking, 6 slots, 8 checkpoints and a 256-token SWA delta interval. The actual environment had early prefetch, direct staging and reclaim-safe enabled, but none of the NVFP4 settings. Same RX 6700 XT (gfx1031), Ryzen 7 5700X, Linux 7.2.15 and TheRock ROCm 7.13 throughout.

The current GGUF has joined `[2304,2304,128]` gate/up in NVFP4 on physical layers 0–19, Q5_K on 20–21, Q4_K on 22 and IQ4_XS on 23. The first twenty have per-expert FP32 gate/up scales. Down is `[1152,2304,128]`, Q5_1 on layers 8–11, 14, 18–19, IQ4_NL elsewhere; no external down scale. The donor instead had NVFP4 down on layers 0–19 and different token and output quantization. The visit order is 0–5, three passes over 6–17, then 18–23. This gives **44 of 48** visits with an NVFP4 gate/up, not an NVFP4 down.

The live ELF contains the HIP NVFP4, IQ4_NL, IQ4_XS, Q5_1 and Q5_K MMQ instances. On a traced prompt batch with MMQ opt-in missing, `Mmq::fits()` rejected NVFP4 and `Layer::mmq` was false for those 44 visits. They used resident MMVQ and CPU experts instead of `moe_rows`; CPU expert work summed to 13.26 s in the 2K diagnostic. The other four visits still used MMQ. With MMQ on, all 48 passed for a full batch. Calls below 32 rows still use the small-call fallback regardless of weight type. The NVFP4 down-tail switch has no effect on this GGUF.

| Original ELF, same prompt IDs and profile seed, dynamic placement | 2,048 fresh tok/s | 10,865 fresh tok/s |
|---|---:|---:|
| MMQ off | 141–142 | 139–143 |
| Base MMQ | 1,026–1,044 | 908–910 |
| Base + lean | 1,052–1,064 | 923–926 |
| Base + lean + scale bits | 1,094–1,098 | 953–956 |

Arms ran in mirrored order. All 16 greedy IDs matched at 2K. At 10,865, the first difference between the first off/opt-in arms was output ID 6; two repeats of the fast arm also first differed at ID 6 under dynamic placement. Do not treat these runs as fixed-placement equivalence. The new Release with auto MMQ matched the original ELF with explicit MMQ on **all 72 greedy IDs and the entire 136 MiB STRSESS file** on the frozen matching-placement `A → B → A` replay, including a parked return with 2,698 reused and 129 fresh IDs. Checkpoint SHA-256 `fd178f83cc54b265ec681c263a255b7613e901ede980705c5f818e3cde2debf2` in both arms. The model file, arithmetic of each selected MMQ kernel, single-token decode, and KV/checkpoint formats did not change.

The live 118.461 s request at 13:43:12 logged 10,865 fresh tokens. The missing dispatch explains the reproducible throughput collapse, but there is no KFD/stack trace for that exact prompt and it was not replayed. In a separate 10,865-ID same-model replay with MMQ on, reclaim-safe off/on produced 11.006/10.875 s. It removed 68 USERPTR allocations (285 MiB), not SVM evictions: 4/9 evictions, longest 375/217 ms. One off-arm 199.897 ms `hipStreamSynchronize(up_)` in `settle_swaps()` overlapped 199.79 ms of an eviction. No on-arm HIP wait reached the trace's 150 ms cutoff. This does not justify claiming the policy prevents stalls. See [the earlier Q4_0 incident](2026-11-28-rdna2-expert-upload-stall.md) for the minutes-long stall, staging event wait, and previous KFD sample.

## Cold and hot timeline

With auto MMQ + lean + bits, uninstrumented prefill at 2K/8K/16K/64K was **1,098/1,010/980/796 tok/s**. The separate November 22 donor campaign measured 992/937/883/628 at those sizes. Different weights, token streams and attention settings make this a reference, not a paired gain.

In a separate traced 8K call, first token was at 7.994 s. GPU kernels occupied 7.343 s: NVFP4 gate/up 2.221 s, IQ4_NL down 0.838 s, Q5_1 down 0.505 s, other expert products 0.260 s, dense products 1.590 s, attention 0.814 s, activation packing and expert elementwise 0.395 s, other kernels 0.722 s. DMA occupied 2.486 s including 2.366 s overlapped with kernels: only 0.120 s was copy without compute. A further 0.532 s had no kernel or DMA in the trace. CPU expert work was zero during the full batches. Host event waits mostly cover already-counted GPU work (6.875 s in `hipEventSynchronize`, not additional latency). Prompt boundary events, routing, checkpoint work and profiling overhead are in the remaining wall time, not in GPU expert time.

A fixed-placement 8K run with a HIP copy hook submitted 58.52 GiB of H2D expert staging, and 354.29 MiB of D2H checkpoints (339.67 MiB SWA, 14.63 MiB convolution). SWA checkpoints also submitted 339.67 MiB of D2D. These are copy argument bytes, not physical PCIe counters; don't add them to the copy durations from the different trace. RSS was 10.60 GiB after the 8K call. The full cold/hot qualification guard stayed above 1,970 MiB free on the 12 GiB card. A separate frozen-placement 32K probe hit 1,450 MiB with the 2 GiB reserve and the guard stopped it; retry the experiment with a larger reserve, not a weaker guard.

| Nominal append after 32K reused history | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Time to first ID (ms) | 737 | 716 | 794 | 1104 | 1505 | 2155 | 3123 |
| Engine prompt read (ms) | 645 | 669 | 756 | 1070 | 1443 | 2120 | 3092 |

These were wire times, not SSE, and each request re-read one additional ID at the old boundary. The 32-token case split into 1+31+1 and did not get batched MMQ; its 31-row pass spent 356 ms on CPU experts in the diagnostic. At 63 rows (64-token request), 2,842 of 3,698 active expert visits had multiplicity at most eight, yet most routed rows ran on resident experts (17,557 resident, 4,729 staged, 1,906 CPU). DMA for the 64–1,024-token cases was mostly exposed: <1 ms of overlap in the trace. Forcing no CPU share raised TTFT from 716/794/1104 ms to 914/1009/1368 ms for 64/128/256. Predicted prefetch at 128 improved one 1,024-token run (2155 → 1778 ms) but regressed 256 (1104 → 1284 ms) and changed some IDs. Staged 8/16/32 classes were neutral, with some ID differences at dynamic placement. Keep the existing policy as the reference.

## Agent result via the loopback API

A guarded private OpenAI SSE server ran a three-request `A → B → A` workload at T=0/seed42. `A` reads a 34,359-token history and calls `lookup_card({"key":"blue"})` with a usable call ID. `B` asks for `MARBLE`. Returning to `A` reuses 34,358 IDs and reads 1,653 IDs from a long tool reply; the correct final answer is `cobalt-73`. Both the original ELF with MMQ off and the Release with auto + lean + bits passed. No external tool was called.

| SSE metric | Original, MMQ off | Release, auto + lean + bits |
|---|---:|---:|
| Cold lookup TTFT | 399.56 s | 36.96 s |
| Tool-return TTFT | 12.30 s | 3.25 s |
| Tool-return prompt read | 12.20 s | 3.09 s |
| Tool-return decode | 31.2 tok/s | 39.9 tok/s |
| Tool-return SSE gap p90 | 36.9 ms | 28.7 ms |
| Time to correct result, including the switch | 418.91 s | 46.17 s |

These are end-to-end workload results, not a per-token decode claim: the return generated 47 versus 46 IDs under dynamic expert placement. This is a controlled tool lookup with a long reference/tool payload, not a broad coding-agent grade.

## Kernel and placement experiments

R2/R3: the real blk.0 gate/up leaf benchmark used captured activation/routing from the same model. I32/J4 raw kept every FP32 output bit and improved a real 64-token routing slice (worst expert 51 rows) from 4.184 to 2.176 ms; at 2048 tokens it lost 12.218 → 46.590 ms. With the groups having at most 8 rows it won across all tested widths. A lossless same-size planar repack issued coalesced loads but took 3.721 ms for 382.2 MB; the 64-token slice was slower than the raw I32 tile (2.779 vs 2.176 ms). Neither requires changing an original weight. On gfx1031, wave32 raw J4 has 43 VGPRs, 11,072 B LDS and zero scratch; disassembly has `v_dot4c_i32_i8`, LDS reads and two barriers per K128 half. A small tile reuses decoded weights across fewer activation rows, which hurts on high-multiplicity calls despite lowering registers and LDS. A scratch launcher selecting raw I32/J4 by group maximum (≤64) had no stable full-inference win, e.g. 128 append 785 → 809–827 ms; 32 append improved 791 → 674–691 ms, but the control varied and all paths sometimes differed in greedy IDs. Do not promote this launcher.

Lowering the 32-row threshold in a scratch worker had mixed results (e.g. 32 append 644 → 543 ms, 64 append 669 → 689 ms) and changed some greedy IDs. First-round upload overlap after routing was not universally faster either. On one Zen 3 core and eight real activated rows, the bit-matching AVX2 probe reused E2M1 decode and a small UE4M3 table. W1 was neutral (~0.57 ms), W2 went from ~1.13 to 0.87 ms, W4 from 2.29 to 1.45 ms, W8 from 4.56 to 2.63 ms. The frozen-placement 32K worker probe used 3 GiB reserve after the stop at 2 GiB. At 24/32 append it moved TTFT from 518/597 to 503/587 ms, but at 64 it lost 632 → 639 ms and six of eight append cases differed in greedy IDs (first at output ID 2 on append64). Neither bitwise leaf parity nor these hot timings justify promotion.

The mixed SiTU prototype fused global gate/up scales, SiTU and transposed Q8_1 packing, using D4 for IQ4_NL and the original half scales/sums (DS4) for Q5_1. It skips the intermediate FP32 activation write/read, not the gate/up output. The frozen 8K ABBA took 7.552/7.540 s unfused and 7.544/7.500 s fused, matching all 16 greedy IDs in each arm. This was below 1% in full inference; it stays out of Release. The original down-tail and earlier K128 experiments were designed for NVFP4 down, not this layout.

With the DFlash2 Q4_0 drafter loaded, the same 8K history and 64/1,024/2,048 appends had loaded-W1/K3 TTFT of 625/627, 1779/1772 and 3001/2997 ms. All used compact SWA, zero swaps, a 3 GiB reserve and 64K context cap. K3 offered no guesses after the first two appends and accepted 9 of 18 after 2,048. Despite the same reserve and profile seed, the arms started with 923/947 resident experts. The first greedy differences on those appends were output IDs 3/10/15, so this is a compatibility and latency observation, not a matched-placement proof of speculative parity. Prefill uses the ordinary W4A8 dispatch; verifier windows are below the 32-row MMQ threshold. Decode residency epochs control swap boundaries, not MMQ tile selection. A restored target checkpoint remains target-only; the proposer can have to rebuild its features. Neither DFlash2 nor a different epoch is enabled in the deployment.

## Release and rollback

Code commit `0db4d974` is on `origin/main`. Rebuilt HIP Release for gfx1031 and passed `poly_mmq_tiles_test`, `poly_inkling_swa_layout_test`, `poly_hot_cache_test`, `conversation_memory_test` and `glm_conversations_test`. The fixed-placement 72-ID/SAVE comparison passed. No unqualified tile, repack, fusion, CPU reuse, staging policy or DFlash2 setting is in the deployed path.

The live user unit `strata` listens at `127.0.0.1:8080`. Its text worker is `~/.config/strata/inkling2-mini/strata-poly-0db4d974`, SHA-256 `252899e69889e71ccca98cd58cb526e1e2a5c42cef809a142d9d9cfe317de13a`, on Ablit49. MMQ is unset in the actual process environment and the engine logs active auto W4A8. Lean and scale bits are set to `1`; prefetch, direct staging, reclaim-safe, vision, KV, parking, profiles and existing session paths were kept. Live checks covered 401 without credentials, a tool call by call ID and a parked 124-token return (`cobalt-73`, 1.004 s TTFT), and unchanged/changed image replay (17/252 fresh tokens, correct red/blue answers). Both rollout and rollback passed the hardware guard. The backup was installed, the old worker was observed healthy with auth, and Release was then reinstalled and verified healthy.

Rollback (backup retains the original authenticated configuration; the old executable was kept):

```sh
d="$HOME/.config/strata/inkling2-mini"
t=$(mktemp --tmpdir="$d" .nvfp4-rollback.XXXXXXXX)
cp -p "$d/before-native-nvfp4-0db4d974.json.bak" "$t" &&
  mv -T "$t" "$d/strata-inkling-2-mini.json" &&
  systemctl --user restart strata
```

Private artifacts: `~/.cache/opencode-tmp/opencode/polystrata-runs/native-nvfp4.RBrq6D/` (`guard-*`, per-arm `metrics.jsonl`, `trace-summary.txt`, `layout.tsv`, exact states, leaf/prototype sources and ISA). No sessions, model data or private logs in Git.
