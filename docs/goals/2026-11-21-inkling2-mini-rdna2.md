# Goal: Inkling-2-Mini on RDNA2

Status: active

Source: user requests of 2026-11-21: native support in this fork, RDNA2 only, Q8_0 K/V, single-user agent use, commit and push, replace the production model after success.

Last updated: 2026-11-21

## Objective

Run `Inkling-2-Mini-UD-Q4_K_XL.gguf` natively in `strata-poly` on the local RX 6700 XT (gfx1031, 12 GiB) and Ryzen 7 5700X, then serve it in place of Qwen3.6 via the existing `strata.service`.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text. Finish when every required outcome is resolved and affected constraints remain satisfied. Stop substantive work at a proven external blocker, an approved budget boundary, or when no remaining in-scope action has a falsifiable expected result; record the exact evidence and smallest unlock.

## Frozen Contract

- **R1 — Native inference.** The exact local Unsloth XL GGUF loads and generates correctly on the current machine, with Q8_0 for both K and V. Evidence: guarded HIP gfx1031 build and real forced-token parity against a reference with the correct joined expert scales; compare Q8_0 to FP16, chunked prefill to sequential decode, and check the SWA boundary and checkpoint/resume. **Status: verified for the tested inputs.**
- **R2 — One agent at a time.** The existing HTTP server provides ordinary text, reasoning and JSON tool calls; the client calls a harmless tool and feeds its result back by the returned ID. Evidence: token ID and template replay parity, streaming tool arguments with a stable ID, and a live streamed tool round trip and cancel-then-fresh request on the exact GGUF, concurrency 1. **Status: verified.**
- **R3 — Production replacement.** After R1/R2, switch the existing user unit to Inkling only. Evidence: `systemctl --user is-active strata`, the authenticated model listing and one real request on the production port. Keep the current loopback bind and auth; keep old files outside the active configuration for rollback. **Status: pending.**
- **R4 — Publish.** Commit task-only changes and push them to this fork. Evidence: remote ref matches the checked local commit. **Status: pending.**

### Constraints and non-goals

- One HIP RDNA2 GPU and one active sequence. No CUDA/SYCL/Metal port, split cards, speculative decoding, multi-slot support, multimodal, new backend or dependency, or INT4 KV. No changes to defaults for other models. FP16 KV is only a selectable reference for parity.
- Do not start a second engine alongside a GPU check. Use `rdna2-rnd/scripts/run_guarded.py`, `DEBUG_HIP_DYNAMIC_QUEUES=0`, and no `--no-graphs`; stop on a fresh GPU/desktop fault or free VRAM below 1536 MiB. Do not change the production unit/config before acceptance.

## Change Envelope

- `src/poly/`, `CMakeLists.txt`: HIP-only Inkling loader, kernels, state and execution, reusing GGUF mmap, physical expert cache, CPU workers, native Q8_0/NVFP4 matvec and available prompt tiles. Shared code only for required integration; retain other models' paths.
- `tools/strata_tokenizer.py`, `serve/`: GPT4O BPE for `inkling2`, embedded chat template, parser wired to the existing response events and single-request FIFO; no server-side tool execution. Add only directly needed checks and setup notes.
- Existing local GGUF: `~/.lmstudio/models/unsloth/Inkling-2-Mini/Inkling-2-Mini-UD-Q4_K_XL.gguf`; no extra weight downloads, raw logs, secrets, binaries or model artifacts in Git. Production uses the current launcher and external config after verification.

## Reviewed Plan

1. Use 24 physical weight/slot owners with 48 logical visits: `0–5, (6–17) × 3, 18–23`. Each visit has independent K/V and causal convolution state. Validate the exact Mini geometry and use chunk-major prefill (the existing layer-major path only spills the residual). Crop the LM head to 200,058 valid IDs after validating/loading the padded 201,024-row tensor. Reuse the existing expert paths with joined NVFP4 gate/up scales before SiTU and down scaling in the expert sum.
2. Implement the Mini schedule and numerical path: RMS-normalized depth mix and previous-pass bank, rank-64 controllers, learned convolution history, relative bias and dynamic zero-value sink, then gated attention and MoE. Reuse Q8_0 storage primitives, with 40 SWA rings of window 1024 and 8 full caches; the 48 conv states remain FP32. Check a source-pinned model reference before using it: [loop graph #32695](https://github.com/ggml-org/llama.cpp/pull/32695) plus [joined scales #32657](https://github.com/ggml-org/llama.cpp/pull/32657). The loop PR head alone omits the joined scale arguments and is not an XL logits oracle.
3. Frontend review reproduced three issues: the current tokenizer silently uses Qwen35 instead of GPT4O (`123456` should be IDs `70904,81554`); the current parser emits native Inkling blocks as raw reasoning; and `reasoning_effort: none` and named/required tool choices would otherwise render the wrong format. Keep the official embedded template, classify native blocks, preserve `<|end_message|>` as a block boundary (not EOS), and prove tool result replay and cancellation. Use the [upstream parser format](https://github.com/ggml-org/llama.cpp/pull/32381) as reference.
4. Verify on the host, then publish and switch `strata.service` from the existing Qwen config, with no Qwen-specific expert knobs copied to Inkling. Keep the rollback config off the active model list.

Engine review confirmed the physical ownership, stale ring rows on rewind, and CPU/GPU/activation gates. It flagged a possible attention scale mismatch; the exact #32695 overload (`llama-graph.cpp:4481–4514` at `6196d980`) puts `bias − sink` in the mask, not in the scaled QK product. This agrees with vLLM: `QK/√128 + bias` with the projected sink as a separate denominator term. Do not patch a second scaling into that reference. The joined expert scale correction still applies. The normal single-agent path keeps checkpoints, so include the active SWA rows and all convolution histories in each snapshot.

## Current Checkpoint

- Closes: R3 and R4.
- Next action: publish the tested frontend, isolate the Inkling config from the Qwen configs, and switch the existing unit under the guard.
- Expected evidence: active service on the same authenticated loopback port, only Inkling in the model list, a real reply, and a matching remote ref.
- Stop or roll back if: the new service does not become ready or a guard fault appears.

## Current State

- Exact local file: 17,629,759,456 bytes. GGUF directory checked: 734 tensors, 20 NVFP4 joined expert layers and four Q8_0 expert layers. The old installed binary rejects `inkling2` before `READY`.
- `cmake --build build-poly-hip --target strata-poly -j 4` passed. First guarded run on the RX 6700 XT, default Q8_0, `--max-context 128 --prompt-batch 32 --vram-reserve-mib 3072 --kv-whole --expert-swaps 0 --threads 8 --tokens 70904,16,81554 --max-new 5 --top-probs 5`, generated IDs `91927,68026,198,68052,198`. It held 973/3072 experts in VRAM, observed 29.5% expert hits on GPU and 4.05 generated tokens/s in that single short run. No guard fault. This is execution evidence, not yet oracle parity or an agent workload measurement.
- FP16 and Q8_0 greedy top IDs matched for all five forced steps. The CPU oracle (`6196d980` plus `a1f42226`) traced 44 joined gate/up and 44 down scale applications per token; its four matching steps had the same top ID and contained all five native top-5 IDs in its top 10, in both KV formats. The largest top-relative logit gap on those candidates was 0.29 for FP16 and 0.53 for Q8_0, not bitwise agreement. The CPU Q8_0 reference uses flash attention, unlike its FP16 run.
- With Q8_0, 46 prompt tokens read one at a time or in chunks of 32 produced the same six IDs. A 1160-token prompt wrapped the 1055-row SWA ring; resuming a changed suffix at checkpoint 1056 and replaying it after a cold reset gave the same six IDs. Guard passed. This did not test the full configured production context.
- Guarded OpenAI Chat Completions SSE run at configured context 32,768, prompt batch 32, reserve 3072 MiB, one sequence: `lookup_card` streamed valid `{"key":"blue"}` with a stable call ID, tool-result replay preserved that ID and returned `cobalt-73`, and a separate arithmetic response had distinct reasoning and text. Closing a long stream after the first text delta logged `disconnect, cancel=True`; the next request returned `MARBLE`. Decode was 14.1–15.9 tokens/s for those five short requests, with 3042 MiB minimum free VRAM and no guard fault. `poly_workers_test` passed.
- The live tool and disconnect check passed again with hashes of the final engine and Python files recorded at both ends. The model can emit reasoning even with minimum requested effort. `tool_choice` accepts `auto` and `none`; named or required choices are rejected rather than injecting another model's call syntax.
- After resolving the HIP KV default before model geometry is loaded, the final Inkling binary repeated the original Q8_0 top-5 trace. On the old Qwen3.6 UD-IQ4_NL, default and explicit FP16 produced the same four greedy IDs and printed top-3 probabilities with all experts on CPU. The first comparison with different resident expert counts was not used as evidence.
- The native commit `ce9e9b28` was pushed and verified on `origin/feat/inkling2-rdna2`; R4 still includes the frontend and goal changes.
- Base: `8d0a6dea`. `strata.service` was inactive and disabled when checked. No production change made.

## Completion

Pending R1–R4 and the closure check.
