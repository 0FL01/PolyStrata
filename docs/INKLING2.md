# Inkling 2 Mini: the inference graph

**24 stored layers, 48 visits per token.** A *layer* owns weights; a *visit* is one use of those weights at a
particular depth. For example, visits 11, 23 and 35 all use layer 11's weights. Each sees a different hidden
state, chooses its own experts and writes its own keys and values at the **same token position**.

The [release article](https://thinkingmachines.ai/news/inkling-2-mini/#architecture-and-training) reports 26.4B
total parameters, 3.4B active and 5.4B weight activations per token counting repeated passes. The last figure
describes work, not weights to allocate. Shapes below are from the released [model config][config]; storage and
execution details refer to one sequence on one card in `strata-poly`. Mini has a fixed pass count;
the variable-loop research checkpoint is a different model.

## Follow a token through the loop

| Stage | Visits (zero-based, inclusive) | Physical layers | Input at the start of the stage |
| --- | --- | --- | --- |
| Prelude | 0–5 | 0–5 | RMS-normalized token embedding |
| Core, pass 1 | 6–17 | 6–17 | Prelude output; no message |
| Core, pass 2 | 18–29 | 6–17 | Restart from prelude; read the two block outputs of pass 1 |
| Core, pass 3 | 30–41 | 6–17 | Restart from prelude; read the two block outputs of pass 2 |
| Coda | 42–47 | 18–23 | Continue from pass 3; no restart or message read |

**Attention over depth is not attention over tokens.** Visit 0's attention takes the input row directly.
Within each six-layer block, sublayer outputs accumulate in a partial residual. A learned depth query mixes
the original input row, completed blocks and the current partial block. The values are RMS-normalized;
the softmax runs over those states, not over token positions. The original input survives the core restarts.
The final output also uses a learned depth mix, then RMS norm, vocabulary projection and logit softcap (30).
Only 200,058 token IDs are valid; the embedding and head tensors are padded to 201,024 rows.

At entry to visits 18 and 30, `run_visits` saves the previous pass's two six-layer block outputs in `bank0_`
and `bank1_` and restores the prelude. Each core layer has separate message queries and controllers for Q, K,
V and FFN. Each query mixes the two banks into a message. A rank-64 projection of the normalized current and
message rows drives two gates, applied to the rows **before** that normalization:
`(1 − tanh(forget)) * current + tanh(gain) * message`. The banks are stable
during the consuming pass; the current row is not. In particular the FFN reads it **after** the attention
update. See [`Inkling2::run_visits`, `depth_mix` and `control`][graph] and the [bank lifetime diagram][banks].

## What one visit computes

1. **Causal attention.** From width 2,304, project 24 query heads and 8 KV heads of width 128 (three query
   heads per KV head). The first five visits of each six use a 1,024-token sliding window; the sixth attends
   to the whole prefix. K and V each pass through a four-tap causal convolution; full attention adds
   input-dependent taps. A window visit can share one input for Q/K/V outside the two message-reading passes;
   inside them, the three controllers produce separate inputs even on window visits. Clip Q and K by head
   norm, RMS-normalize V, and clip the relative query. Full attention also scales Q and the relative query
   logarithmically above 8,192 positions.
2. **Position and output gate.** The target does not use RoPE. Its score is `dot(Q,K)/sqrt(128)` plus a
   learned 16-component relative term for distances 0–1,023. Older keys in a full visit still contribute
   their content score. A projected sink enters the softmax denominator once per query head and has no
   value vector. A sigmoid gate scales the weighted values before output projection and post-attention
   convolution. See [the attention kernels](../src/poly/inkling_attend.cu).
3. **Routed FFN.** From the updated residual, mix over depth, apply the message controller when present,
   then normalize. `Inkling2::route` picks 8 of 128 experts by `sigmoid(logit) + bias`; only the **unbiased**
   sigmoid scores of those eight are normalized and multiplied by the route scale (8 in the release).
   Each expert has width 1,152: joined gate/up projections, SiTU (softcapped gate and up, not plain SwiGLU),
   then down projection. There is no shared expert. The weighted sum goes through a four-tap convolution
   before addition to the residual. There are `48 × 8 = 384` expert selections per token, not necessarily
   384 distinct experts. Route again on every visit; a previous pass's IDs are not the route.

## What can be shared in `strata-poly`?

| Owner | What it holds | Consequence for an engine change |
| --- | --- | --- |
| Physical layer (24) | Dense weights, router, expert weights and their VRAM slots | Reuse a resident expert across passes by layer and expert ID. On serial decode, resident experts run on the GPU; misses can run on the CPU from system RAM. Changing placement can change arithmetic even with identical router decisions. |
| Logical visit (48) | KV and three prior inputs per channel for each of the K, V, attention-output and FFN-output convolutions | Keep all 48 histories independent across token and prompt-chunk boundaries. Restore convolution history and sliding-window tails as well as full KV when resuming a prefix. |
| Current call | Prelude, depth residuals and the two message banks, one row per input position | Scratch, not committed sequence state. Prefill runs each prompt chunk through all 48 visits; spilling only the residual between visits would lose inputs. |

[`Inkling2::alloc_caches`][graph] allocates **40 window rings and 8 full caches**. By default a ring has room for up to
`1024 + rows_ − 1` positions (capped by configured context), so writing a chunk does not overwrite keys its
first query still needs. With default Q8_0 KV, one full visit stores `2 × (1024 codes + 32 two-byte scales)`
= 2,176 bytes per position. The eight together hold **17 KiB per stored position** (32 KiB with FP16). These are target KV
payloads, not total VRAM: they exclude rings, convolution state, scratch, allocation rounding and any drafter.
Beyond the window only eight attention scans keep getting longer, but all 48 routers still run per token.
The release config's 1,048,576 positions are a model limit, not a tested RDNA2 memory budget.

**Compact window storage** (`STRATA_INKLING_COMPACT_SWA=1`, off by default) retains the prior 1,023 rows plus
the verifier width in each target ring. A wide prompt uses a single staging area; it borrows `input_[0]` after
the sink projection if the area fits. The first staged key keeps its position modulo 16 so Flash reads the same
tiles. Serial decode and DSpark verification use the ring directly. At prompt batch 2,048 and Q8 KV without a
drafter, the ring payload falls by 169.92 MiB without a new staging allocation. The default placement policy
leaves that estimated saving free; `STRATA_INKLING_FUND_EXPERTS=1` allows the usual whole-expert cache to use it.
Changing the CPU/GPU split can change generated tokens; saving VRAM is not itself a speed or quality result.
The saved window state still contains the same oldest-first tail, not the physical ring or the stage.

For matched-budget runs, `STRATA_INKLING_EXPERT_KV_MIB` limits the expert arena plus charged global KV on
one card with VMM and no pool. It does **not** limit total process VRAM or replace `--vram-reserve-mib`.
Do not freeze a full startup arena and then try to grow into the same budget: frozen experts cannot be evicted.

### Conversation parking

`strata-poly --serve` uses the hot session cache automatically on supported dense Q8 Inkling configurations
when conversation parking is enabled. It parks a text conversation at its last committed token, so switching
back can reuse the reply as well as the prompt. `--conversation-cache-mib auto` sizes parking from available
RAM after model load; `0` keeps parking off. V4, K256 and DSpark use the legacy cache. When the hot cache is
selected, the startup log reports its mode and configured budget.

Automatic hot cache keeps prompt checkpoints, but does not take periodic snapshots during generation.
New snapshots are skipped if they would leave less than `--conversation-cache-min-free-mib` available.
For workloads that edit the end of a reply (for example, tool-call JSON), `--swa-delta-every 256` enables them.
`--hot-cache` forces the hot path on supported engines and retains the original 256-token interval unless set
explicitly; `--no-hot-cache` selects the legacy path. `STRATA_HOT_SESSION_CACHE=1` or `0` can set the same
override when neither flag is given. See the [RX 6700 XT benchmarks](goals/2026-11-27-hot-session-cache-swa-replay.md).

In a guarded RX 6700 XT run with Tideglass v3 Q4_0, dense Q8 KV, frozen experts (1,151 GPU slots),
4 GiB of parking, six slots and eight prompt checkpoints, six requests after the first cold request took
**8.72 s** total to first token with legacy parking, **7.68 s** with automatic hot cache, and **6.66 s**
with automatic hot cache and a 256-token decode interval. The prompts were 497 tokens to about 3.8K and covered
plain continuation, a late reply edit, a short side request and switches. For the first 576-token reply,
p99 token gaps were 41.2, 41.3 and 41.6 ms respectively; the maxima were 42.9, 49.7 and 72.5 ms.
On the first return after switching, legacy replayed 64 more tokens; both hot variants first differed
from its greedy output at token 37. With the 256-token interval, the late-edit reply also first differed
at token 64. The other replies matched in this run. A separate full-tip switch (`A → B → A`) in automatic
mode reused all 2,564 tokens and matched all 64 greedy IDs and the saved state byte for byte against
uninterrupted `A` on the same card and model.

### Optional global attention and storage

Dense Q8 remains the reference. K256 and V4 are independent of each other and of compact SWA. Both need Q8
KV, the HIP decode reader, a single sequence without DSpark, and a context of at most 262,144 positions.

| Control (off when unset) | What changes |
|---|---|
| `STRATA_INKLING_K256=1` | Serial generation only: keep page zero, 256 selected old 128-row pages per KV group and the full recent tail (at least 2,048 rows). If no old page would be skipped, read dense. Prompt and tool-return queries do not share a selection. The growing key index is rebuilt from K, never saved. |
| `STRATA_INKLING_V4=1` | Keep absolute Q8 K. Before an input batch, seal whole old pages of V to symmetric signed nibbles with one refitted FP16 scale per 32 values. A separate native Q8 ring holds the recent rows and the entire new batch. Needs growing VMM, without `--kv-whole` or `--vram-pool`. |
| `STRATA_INKLING_FUND_EXPERTS=1` | Allow the cache to use actual savings for ordinary whole-expert slots. A combined budget counts the index, packed planes and fixed hot ring. Without funding, charge saved KV bytes to placement instead. |
| `STRATA_INKLING_V4_SCALAR=1` | Reference CPU packer for V4. Otherwise use the byte-identical AVX2/F16C refit on eligible CPUs/blocks and scalar elsewhere. This is not a different attention policy. |
| `STRATA_INKLING_V4_STATE_PACK=1` | With V4 and `STRATA_INKLING_STATE_PACK=1`, capture the convolution state, SWA and exact Q8 V tail in one host copy when the existing expert stage fits. Otherwise use ordinary copies. Same checkpoint bytes; no wider stage or extra VRAM. |
| `STRATA_INKLING_SUFFIX_H2=1` | On RDNA2 with V4, use split Flash16 for global prefill batches of at least 32 rows once a sealed row is visible. The kernel reads old V4 and the wrapped Q8 tail directly. Its half-precision products and different reductions can change answers. It does not select pages for a prompt or change W1. |

A compact checkpoint saves the exact native V tail for its own position. Restoring a branch does not guess
the missing low bits from V4. Compact disk files have a **different format identity**; the compact reader can
import a compatible native file directly into packed backing, but the dense binary cannot open a compact file.
Keep the original native session if you need to roll back to the old binary. With K256 alone, the stored rows
are still native Q8; a native file describes storage, not which attention policy generated its hidden states.

## Reading a GGUF and comparing changes

- [`Engine::open` and `Engine::read`](../src/poly/engine.cpp) dispatch the model and chunk the prompt.
  [`Inkling2::load_geometry` and `load_layer`][graph] distinguish `inkling2_loop` (24 declared layers,
  per-layer message queries) from legacy `inkling2` (48 declared visits, global message-query tables).
  Both execute the [same visit-to-layer mapping](../src/poly/inkling2.hpp) with 24 physical weight owners.
- The release uses NVFP4 expert weights for layers 0–19 and MXFP8 for 20–23. Select weight kernels by GGUF
  tensor type and scales, not filename. In the [Q4_0 file](../README.md#inkling-2-mini-performance), all 24
  joined gate/up tensors are Q4_0, but the down tensors for layers 0–2 are Q4_1. The [tested Unsloth XL][xl]
  instead has NVFP4 experts followed by four Q8_0 expert layers. For NVFP4, gate/up scales apply **before**
  SiTU; the down scale belongs to the expert contribution. Weight format is independent of `--kv`.
- Optional paths are not extra target passes. The [vision helper](../tools/vision/INKLING.md) supplies
  projected image rows, which bypass the text embedding norm. The separate DSpark drafter reads the residuals
  entering visits 5, 23 and 41 plus the normalized final output; the target verifies its guesses. Test verifier
  windows separately from serial decode ([state and parity limits](DETAILS.md#using-it)).
- For attention changes, exercise both a wrapped ring and a long full visit ([`poly_inkling_attend_test`](../src/poly/inkling_attend_test.cu)).
  For routing or expert changes, compare ordered IDs, weights, CPU/GPU assignment and expert outputs before
  comparing final logits. For state changes, compare chunked prefill and save/restore against fresh replay.
- For measurements by phase, see [prefill](goals/2026-11-22-inkling2-prefill-e2e.md#final-measured-account) and
  [serial decode experiments, including rejected changes](goals/2026-11-24-inkling2-architecture-aware-results.md).
- On the RX 6700 XT, [exact SWA compaction](goals/2026-11-25-exact-swa-memory-compaction.md) reclaimed VRAM without
  changing expert placement. The [global attention and KV study](goals/2026-11-25-adaptive-global-kv/README.md)
  separates fewer KV reads from lower KV residency and measures both against time to a correct tool-assisted answer.
  The study numbers are from prototypes. [Production checks](goals/2026-11-25-adaptive-global-kv/production.md)
  distinguish current builds and actual deployment from the research measurements.

[config]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/c9d028302cd0f035fdf8e7e91f089a4ce94d2195/config.json
[graph]: ../src/poly/inkling2.cu
[banks]: goals/arch-decode-recon/loop/loop-dependency-graph.md
[xl]: goals/2026-11-21-inkling2-mini-rdna2.md#verification
