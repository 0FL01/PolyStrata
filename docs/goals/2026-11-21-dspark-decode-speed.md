# Inkling DSpark decode speed on gfx1031

Status: unmet (speed); the requested first MoE experiment is complete.
Follow-up to `2026-11-21-inkling2-dspark.md`, base `936249e0`.

R1: the user wants a stable increase in decode speed. Compare emitted tokens/s
against today's plain production engine on tool calls, code, JSON and prose
(English and Russian), not just against the previous speculative engine.
Keep the target model, sampling and 200K capacity. The existing +25% agent mean,
no fixture below -3%, and 16K check remain the gate for production.

The user reprioritized the target MoE: first test fully resident repeated
layers on the original IQ4 GGUF, without speculation. Compare ordinary cache
placement (A), pinning with CPU routing (B), then identical placement with
device-only top-8, expert computation and sum (C). First milestone: 50 tok/s
of plain decode. Do not requantize before evidence for this placement.
Guard all card runs and use profile copies. Only keep a change that pays for
its cost; work on main and push task commits.

Initial screening: on sampled RU (seed 713, 256 tokens), plain / DSpark at
batch 2048 were 36.53 / 35.54 tok/s. DSpark at 256 was 37.35, but on the
4,281-token code prompt this took prefill from 633.6 to 324.4 tok/s. No batch
change. A proposal sampled in token-ID order with a copy of the target RNG
also lost (RU 33.90, 16/66 guesses right); prototype removed. Neither changes
the production service.

## The three-way MoE test

RX 6700 XT (gfx1031, 12 GB), Ryzen 7 5700X. Same Inkling-2-Mini-UD-IQ4_XS,
no drafter, Q8_0 KV, context limit 200,000, batch 2048, eight CPU workers,
64 expert swaps, 2048 MiB reserve. Decode, flash, direct staging and NVFP4
MMQ on. Fresh process and copy of the same frozen starting profile per run.
The short fixture is the earlier notebook benchmark: 94 prompt tokens, 96
fixed output IDs, 95 timed decode steps. Time for the prompt is not counted.

In an instrumented baseline, 6 and 7 started with 101 and 105 of their 128
experts resident; 9 and 13 had 97 each. These layers were cheap to complete.
Across **all** 36 repeated-core visits per one-row call the host computed 14.1
expert assignments; across the other 12 visits it computed **25.8**. Logical
bytes read from expert weights are not a measure of CPU misses. The four
selected layers accounted for 3.8 misses, 0.50 ms in the CPU-expert span and
4.46 ms waiting for router events per one-row call. The wait includes GPU work
queued before the event; it is not 4.46 ms of removable idle time.

B filled all 128 slots of each selected layer first, from the ordinary cache
budget, and excluded them from swaps. C used the same placement. It kept
router logits on the GPU, chose top-8 by sigmoid plus bias, normalized the
unbiased weights, then used device IDs for the NVFP4 products, scales, SiTU
and final sum. No expert was skipped. Other visits kept the CPU router.

| Mode | Production reserve only, tok/s | Expert arena capped at 5,600 MiB, tok/s |
|---|---|---|
| A: ordinary placement and router | 38.84 / 38.35 | 35.62 / 34.63 |
| B: pin 6, 7; CPU router | 38.63 / 37.63 | 35.28 / 34.36 |
| C: pin 6, 7; GPU router | 35.42 / 38.04 | 35.60 / 35.62 |
| B: pin 6, 7, 9, 13; CPU router | 37.19 | 34.71 / 35.01 |
| C: pin 6, 7, 9, 13; GPU router | 36.25 | 33.55 / 33.76 |

Left column: A/B2/C2/C2/B2/A/B4/C4. Right: A/B2/C2/B4/C4, then the reverse.
Compare **within** columns: the second series fixed the arena at exactly
5,872,025,600 bytes each time (1,300 / 1,302 / 1,303 slots, because slot sizes
differ). All runs included. Two layers sometimes beat pinning with the CPU
router, but did not consistently beat ordinary placement; four lost more.

**Path check, separate from timing:** rocprof, eight-token prompt and 23
timed decode steps, four pinned layers. Per step B → C: `route_one_kernel`
0 → 12, NVFP4 `MmvqIds` products 0 → 24, `copy2_kernel` 49 → 37. The three
visits per pinned layer did move off the host. At identical pinned placement,
swaps off, eight teacher-forced parity steps after 1,064 tokens had the same
top-1; largest absolute change in its probability was 0.0106. This is not a
claim of bitwise equality. A sampled RU screen (256 tokens, two layers) gave
A 35.64 / 36.42, C 35.55 / 35.43 tok/s; the local restore build overlapped
the start of the first C process, so the decision is based on the forced test.

**Decision:** no stable speedup and no 50 tok/s plain result. Do not change
placement in production or start the conditional requantization stage. No
16K/200K performance or IQ3/quality claim from this experiment. Revisit if a
different placement costs fewer hot slots or a trace shows **idle** gaps at
these visits large enough to change the full-token time.

The rejected engine changes and benchmark artifacts stayed out of Git. Local
reproduction directory: `~/.cache/opencode-tmp/opencode/dspark-speed-1795285637/`
(`tested-resident-moe.patch` against the base above, `build`, `forced`, `fixed`,
`moe-fixed-*`, `trace-{0,1}`). The test-only cap was `STRATA_PROTO_CACHE_MIB=5600`,
the pin masks were 0, 192 and 8896; `STRATA_INKLING_GPU_MOE=1` selected C.
The restored HIP `strata-poly` is byte-identical to the installed `32b3fc6f`
binary. The unchanged service was restarted; loopback `/health` returned 200
and unauthenticated `/v1/models` returned 401.
