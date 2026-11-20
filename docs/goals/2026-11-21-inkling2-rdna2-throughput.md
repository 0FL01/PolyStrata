# Goal: faster Inkling-2-Mini on RDNA2

Status: active

Source: user requests of 2026-11-21 to implement and profile the Inkling latency fixes, substantially improve PP and TG while preserving coherent replies, then commit, push and deploy. Production downtime is allowed.

## Objective

Raise cold prefill and decode throughput on this RX 6700 XT / Ryzen 7 5700X with the existing `Inkling-2-Mini-UD-Q4_K_XL.gguf`, and deploy the measured configuration.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text. Finish when every required outcome is resolved and affected constraints remain satisfied. Stop on a fresh guard fault or a proven blocker; record the evidence and smallest unlock.

## Frozen Contract

- R1: Both cold PP and TG improve clearly beyond repeated-run variation over the currently installed engine/config on the same fixture. Measure real new tokens, not reused prompt tokens. Qwen's 600/50–60 tok/s is context, not a guaranteed Inkling target. Status: in progress.
- R2: The faster path keeps coherent text and tool-result replay with a stable call ID on the existing one-request HTTP server. Check fixed-token top predictions, chunk/single-row and SWA checkpoint behavior for any attention change, and a live agent round trip before closure. Status: verified on the candidate; see qualification below.
- R3: Commit and push task-only changes. Deploy the qualified binary/config through `strata.service`, check single-model API and one actual request after restart; keep a rollback copy. Status: pending.

## Constraints and Change Envelope

- One HIP RDNA2 card, one sequence, exact existing GGUF, Q8_0 K/V, 200,000 configured context and unchanged client input/output limits. No expert skipping, model swap, changed request protocol or speed claim based on reduced precision.
- Native Inkling attention and hot-path integration (`src/poly/inkling2.*` and existing shared helpers as needed), opt-in existing RDNA2 NVFP4 MMQ instantiation, benchmark-backed selection of existing KV and expert-cache controls. Other model defaults stay unchanged. Benchmark and deployment artifacts, credentials and profile data stay out of Git.
- Keep a single engine at a time; use `rdna2-rnd/scripts/run_guarded.py`, static queues, 1536 MiB guard floor and stop on fresh GPU/desktop faults. Benchmark outside the service before deployment; leave the old service available between experiments when no GPU run is active.

## Current Checkpoint

- R1/R3: finish the final timed A/B/B/A and separate phase trace with the qualified growing-Q8_0, batch-512, NVFP4 MMQ, eight-worker configuration with a separate cache seed. Then commit, push and deploy the same binary and settings.
- 200K Q8_0 A/B/B/A (RX 6700 XT, Ryzen 7 5700X, 2,048 cold PP tokens, then a separate 94-token prompt with 95 timed teacher-forced decode steps): installed baseline PP 28.1/30.4, TG 11.29/13.93 tok/s; adaptive attention alone PP 31.4/32.4, TG 13.53/14.29. One additional adaptive run with growing KV measured PP 41.2 and TG 16.96, with 919–920 rather than 242–279 resident experts. In the comparably warmed runs the CPU expert span was 32.5–32.6 ms/token with whole KV and 24.8 with growing KV on the decode fixture; these are host-reported spans, not isolated kernel times. Guard minimum 2,859 MiB free, no faults. The existing installed binary/config remains the baseline.
- On the same 2,048-token fixture and growing KV, switching only NVFP4 W4A8 MMQ from off to on took batch 128 from 47.7 to 115.2 tok/s and batch 256 from 48.5 to 159.3; batch 512 with MMQ reached 208.7. The enabled runs logged gfx1031 MMQ dispatch. Minimum 3,037 MiB free with no guard faults. These are one-run screens, not the final pair. The PP CLI cache-hit percentage is not comparable across dispatches: `moe_rows()` counts its CPU work but not its GPU assignments in that field.
- The candidate reuses one Q8_1 activation across the small-batch projections that share an input; the batch MMQ layout is unchanged. An expert-frequency profile trained on four separate synthetic prompts selected about 960 residents and measured TG 21.76/21.85/21.15 tok/s with 8/12/16 CPU workers. Keep 8: the screen did not demonstrate a gain from more workers. No guard faults; minimum 3,033 MiB free.
- Qualification: the four previously checked pinned CPU-oracle top-1 IDs match. One-row and 32-row reads with the same CPU expert placement give the same six generated IDs. An initial mixed-placement run had different resident experts and is not an equality check; at the fixed 1,064-token prefix with NVFP4 MMQ on, all eight top-1 IDs match the installed engine, while the largest change in top-1 probability is 0.0763. With batch 512, restoring at token 4,096 after SWA ring wrap and reading the prompt from scratch produced the same six IDs. A live single-slot HTTP agent streamed a JSON call, kept its call ID through tool replay, returned coherent text with separate reasoning, and served a fresh request after cancellation. Guard minimum 2,964 MiB free, no faults. The full HIP build and `poly_workers_test` passed; CUDA and SYCL toolchains were not available.
