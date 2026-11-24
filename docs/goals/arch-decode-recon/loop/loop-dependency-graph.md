# CP-P3-01 — Bank lifetime and exact hoist boundaries

Source pin: `c0dd0e52f3975dbed3972deda2c2c75010b330c2`. `398027f8` did not change `src/poly/`. This is a source and GGUF-metadata analysis, not a GPU parity result.

## Visits and publication points

`src/poly/inkling2.hpp:45–62,99–110` and `src/poly/inkling2.cu:1284–1323,1424–1442,1503–1518`:

| Logical visits | Physical layers | Writers/readers of the banks |
|---|---|---|
| 0–5 (prelude) | 0–5 | No bank reads. After visit 5's tail, `r_` is copied to `prelude_` at entry to 6. |
| 6–17 (core pass 1) | 6–17 | No bank reads. Visit 12's attention residual update stores its entering `r_` in `depth_[2]` before replacing `r_` with the new attention output; that saved value is the finished **6–11** block. Visit 17's tail leaves 12–17's finished partial in `r_`. |
| 17→18 (publication) | 17→6 | In `st_` order: `bank0_ ← depth_[2]`, `bank1_ ← r_`, restore `r_ ← prelude_`, reset host `depth_n_ = 1` (retains slot 0). Consumers begin at visit 18. |
| 18–29 (core pass 2) | 6–17 | Read both banks for Q/K/V and FFN of every visit. No writes. The current pass overwrites `depth_[2]` in visit 24 (finished 18–23); this does not affect `bank0_`. `r_` changes each visit and ends with the 24–29 partial. |
| 29→30 (publication) | 17→6 | Same two copies and reset, now with pass 2 outputs. Must occur after **all** pass 2 consumers of old banks and old prepared rows. |
| 30–41 (core pass 3) | 6–17 | Read the new banks for Q/K/V and FFN of every visit. `depth_[2]` can be overwritten again in visit 36; banks do not change. Last consumer is visit 41's FFN controller. |
| 42–47 (coda) and output | 18–23 | No bank reads; coda inherits the live residual/depth results of pass 3, not the bank pointers. |

`depth_[2]` itself becomes a stable source after the first attention residual update of visits 12/24. The second source is not ready until the tail of 17/29 has finished (including experts and convolution). The simplest insertion point is directly **after the existing `bank1_` copy**, before or after the independent prelude restore, on the same stream. It is not safe to read pre-copy `bank1_`, to alias the constantly changing `r_`, or to use `depth_[2]` as a pass-long `bank0_` alias; `depth_[2]` is overwritten midway through the consumer pass. The stable `depth_[2]` source might allow an earlier bank0-only experiment, not earlier complete message prep. Copy elision must preserve baseline `axpy(...,1.f,false)` semantics (not assume raw copying is identical for every FP bit pattern).

`assign_scratch()` assigns independent `fbuf()` ranges of `rows_*2304` to `prelude_`, each bank, and the single-stream message (plus separate `s3msg_` for grouped QKV). Only the first `n` rows are valid for the current call. Neither bank nor prepared rows should enter committed state, snapshots or prefix reuse. Recompute at both boundaries of **each** `run_visits(n)`, including a restarted/replayed call; pass 1 must never read leftover scratch. For a same-stream prototype a single reusable slab is sufficient; a side stream would need both a producer event and a completion/lifetime edge before the same backing buffers are recycled. Same-stream placement after the boundary alone is not overlap. Avoid a grid-wide barrier.

## Exactly which queries are available

`load_layer()` uploads four controllers plus query rows for each physical layer 6–17 (`inkling2.cu:712–765`). CPU inspection of the Tideglass v2.1 GGUF header: `general.architecture=inkling2_loop`; all 12 `loop_msg_query` tensors are F32 `[2304,4]`; each controller's F32 `down`, `gain`, and `forget` is `[4608,64]`, `[64,2304]`, `[64,2304]` respectively (GGUF dimensions); 24 gate/up expert tensors are Q4_0. The historical Bartowski Q4_0 file has these same shapes/types, not an assumption of identical query weight values. The two layouts in `message_query()` (`inkling2.cu:1193–1198`) are:

| Stream | `inkling2_loop` per-layer row | Legacy layout (`r = il−6`) |
|---|---:|---|
| Q | 0 | `loop_query_[2*r]` |
| K | 1 | `loop_kv_query_[2*r]` |
| V | 2 | `loop_kv_query_[2*r+1]` |
| FFN | 3 | `loop_query_[2*r+1]` |

All 12×4 queries are known when the two banks are published. At `n=1` this is 48 message rows per pass, 96/token across the only two controlled passes. Physical 11 and 17 are full attention; the ten other physical layers are SWA. All four controllers run even on SWA in passes 2/3. On SWA, Q/K/V **live depth-mix** use the same `b.aquery` row and depth input sequence, so their raw currents and their own RMS are a separate shareable candidate after byte verification. They still have different *message* queries and controller weights. For full attention all three current queries differ; FFN obtains its current only **after** attention, output projection, and post-attention convolution/residual update. No sharing between these current paths or between the two controlled passes is implied by physical weight sharing.

## Dependency graph and numerical fences

```text
stable prior-pass b0,b1 + immutable per-layer per-stream q
  -> per-bank (square-sum -> scale, query dot) -> two weights -> raw m -> message RMS -> mNorm
changing depth[] and residual r_ + immutable depth query
  -> original depth_mix -> current x -> current RMS -> xNorm
             [xNorm || mNorm] + down weights -> rank64
   rank64 + gain/forget weights -> ga/fa -> tanh -> controlled(x, raw m)
  -> learned input norm + Q8_1 -> attention or FFN -> route, experts, convolution/residual
```

| Operation | Dependency / scheduling constraint |
|---|---|
| Bank loads, square sums and bank inverse RMS | Bank-only per call-row/pass. RMS terms can be reused over all 48 query rows; retain the same reduction and `rsqrtf`. |
| Query dot, two-way softmax/mix, raw message and its RMS/normalized row | Banks + immutable stream query. Exact candidates below. No explicit position, KV, or convolution inputs; the bank *values* already encode prior history. |
| Depth mix and current RMS | Live depth/residual at the stage; `mix3_kernel` in `heads()` is **not** a previous-pass bank mix. RMS may be shared only within the current equal SWA Q/K/V streams, or moved into their current producer with parity checks. |
| `down` rank projection | Live `xNorm` plus independently preparable `mNorm`, immutable weights. Address splitting can be exact; ordinary algebraic splitting into dots and adding is not. |
| Gain/forget projections and output | Live rank and raw current, immutable weights, plus raw `m`; cannot precompute outputs from only the banks. |
| Learned norms, quantization, projections, attention, routing, experts and conv | Descendants of the live output. Attention reads positions and visit-indexed caches; convolution uses visit-indexed recurrent state. Keep all as baseline. |

1. **`control_in_row()` (fused short rows):** 128 threads, 18 float elements/thread at `tid + k*128`, 4 wave32; explicit `__fmaf_rn` for bank squares and query dots. Each sum uses down shuffles `16,8,4,2,1`, then a wave32 reduction over four wave winners plus `+0.f` lanes. `reduce(0,5)` co-schedules the independent sums; sharing or moving terms must preserve each tree. Scale `rsqrtf(ss/2304 + 1.e-6f)`. Keep `dot*s*rsqrtf(2304.f)` with the written multiplication order, `fmaxf`, subtract, `expf`, `w0+w1`, and both per-channel FMAs (bank 0 starting at `0.f`, then bank 1) **before** `val/sum`. Hoisting rounded `(w0*s0, w1*s1, sum)` is possible; publishing `a0/sum` and `a1/sum`, or multiplying by a rounded reciprocal, is not the same arithmetic.
2. **Raw `m` and norm pair:** raw `m[k]=val/sum` is materialized and later read by `control_out_body()`. `ss += value*value` of raw `m` and `x` must keep the gfx1031 compiler's contraction and the same 128-thread tree. Norm uses `1.0f / sqrtf(ss / float(2304) + 1.e-6f)` (not `rsqrtf` or `__fsqrt_rn`) then one multiply. `pair+2*row` stores `xNorm[2304]` followed by `mNorm[2304]`. Preserve all three FP32 buffers as separate Q0 tests; check ISA as a diagnostic, not a replacement for byte parity on real activations. A new store or compile-time specialization can change contraction. Unfused/large-`n` fallback instead uses `mix_row(...,count=2)` followed by two `rms_norm()` calls; those norms use **1024** threads for width 2304 and `block_sum` with a different final reduction. Never substitute the fused current or message norm into this path without a separate proof. Initial prototype is `n==1 && fused(n)` with the plain code unchanged elsewhere (including short multi-row verifier and bulk prefill). `n==1` can also occur in a prompt tail.
3. **Rank FP fence (`kernels.cu:2847–2870,2897–2920,2938–2959`):** for aligned F32 `n==1`, `n_in=4608`, `n_out=64`, both QKV and FFN dispatch `gemv_rows_body()` with 256 threads/one block per rank output, QKV batched over three `blockIdx.z`. Each thread accumulates `float4` at `i=tid,tid+256,...<1152`; wave XOR `16,8,4,2,1`; thread 0 adds wave winners in index order, starting at `0.f`. Message half starts at float4 index 576: `tid<64` first encounters it at `tid+768`, others at `tid+512`. Simply replacing the contiguous `xr[i]` load with the correct addressing from two aligned FP32 rows can preserve this schedule, subject to original-build byte comparison. Hoisting a completed suffix, a per-lane sum, or a reduced message-only rank and adding later reorders FP32: `(1e20f + -1e20f) + 1.f` differs from `1e20f + (-1e20f + 1.f)`. A narrower possible hoist is each invariant float4 RHS **only if** the compiled HIP code rounds it separately and keeps the same subsequent adds into the live accumulator; verify ISA and bytes, and budget 64×576 floats per controller (6.75 MiB per pass for 48 controllers, plus readback traffic), so park this arm. For `n >= 10` the fused shape with F32 uses a different rank schedule (`8*n < 80` fails); do not silently batch short decode and change the tree.
4. **Controller output FP fence:** `control_out_body()` uses eight lanes per output, two `float4` iterations per lane at rank 64, and XOR `4,2,1`; preserve the group expressions, `tanhf` gates and the original contraction of `(1.f - tanhf(fa))*cur + tanhf(ga)*msg`. Verify rank and controlled output, then learned-norm FP32 and Q8_1 bytes. After that, apply goal Q2–Q5 with matched placement and separate forced/greedy runs, including canonical state, SWA wrap, rollback, saved/restarted sessions and prefix reuse.

## Arms and sizing (`n=1`)

| Arm | Pass-boundary result | FP32 storage for one pass, excluding existing scratch | Critical-path reality |
|---|---|---:|---|
| B | Two bank scales, or original square sums if testing the scale separately | 8 B (16 B if keeping both) | Only two reductions reused over the 48 streams; a new independent launch may outweigh this. |
| Coefficients | `w0*s0`, `w1*s1`, `sum`, still apply the two original FMAs and divide at the visit | 576 B | Smaller cache, leaves raw message generation and RMS serial. Keep this as a fallback scheduling arm. |
| C | Raw `m` for each stream | 432 KiB | Visit still needs both RMS outputs. |
| D / E | Group 48 raw `m` and normalized `mNorm` rows | 864 KiB; or 432 KiB + 192 B if storing raw rows and message scale only and doing the normalization multiply at consumption | A minimal split still has the per-visit current normalization dispatches. To eliminate them, compute the current norm with its original `depth_mix` producer and feed `down` the two separate halves (or write paired scratch). This differs from the older failed **post-controller** norm fusion: both current producer and controller-input RMS originally use 128-thread rows. Register pressure / throughput and byte parity still need proof. |

Storage scales with `n` if generalized but not pass count for same-stream reuse. Full D writes 1.6875 MiB/token across two boundaries, to be consumed later; these are logical bytes, not DRAM traffic. The bank pair is just 18 KiB/row and was already re-read in the baseline (likely some cache reuse). One 48-CTA prepass can fill a lot more of the 40-CU card than the baseline's 1-CTA FFN or 3-CTA QKV input, but it introduces clustered traffic and does not erase all controller work. Do not trade away an expert slot without measuring the full effect.
