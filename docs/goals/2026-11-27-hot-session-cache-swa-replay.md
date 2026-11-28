# Hot Session Cache & Exact SWA State Replay (Inkling 2 Mini on RDNA2)

Status: **complete and deployed to local production service** (27–28 November 2026, built from `80bac602`).
Tested on **AMD RX 6700 XT (`gfx1031`, 12 GiB)**, Ryzen 7 5700X, 32 GiB RAM, Btrfs, ROCm HIP Release build
(`build-poly-hip/strata-poly`), **Tideglass Inkling 2 Mini Unbound v3** (`Q4_0`, dense `Q8_0` global KV, exact compact SWA,
`--max-context 200000`, `--prompt-batch 2048`, `--threads 8`, `--expert-swaps 64`, `STRATA_INKLING_STATE_PACK=1`).

## Why this work was needed

Production audit of 885 real `strata.service` requests (plus `opencode.db` showing 40 session switches across 90 recent
agent turns) found **20.3% of total wall time (82.7 min)** spent in prefill: 55 cold misses (35 min) and 96 partial-prefix
drops (354,745 recomputed tokens). Five concrete causes accounted for almost all of that waste:

1. **Parking disabled (`--conversation-cache-mib 0`)**: short 497-token title side-requests and session switches wiped the
   active 16K–34K conversation.
2. **Stock `park()` truncated to `marks.back().pos`**: switching away discarded the entire last assistant turn (256–1,024+
   decoded tokens) because no checkpoint existed at the committed tip.
3. **Zero decode checkpoints**: chat-template canonicalization (e.g., `dictsort` on tool-call JSON mutating the last ~64
   tokens of a 1,024-token assistant reply) rolled all the way back to `prompt - 1`.
4. **`drop_of()` evicted shared prefix anchors and recent tail checkpoints**: the 13,316-token OpenCode system+tools prefix
   and recent rollback points (`22,528`) were dropped while stale checkpoints strictly inside `[0, sys_at)` remained.
5. **Slow state transfers and eager disk spill**: stock 197K parking took **1,074 ms** (single-threaded host page faults +
   unpacked transfers), and unconditional `Spill::release()` caused 8.5× slower restores (`2,093 ms` vs `246 ms` in RAM)
   even when 20+ GiB of host RAM remained free.

## Exact Inkling 2 Mini state geometry (dense Q8)

- **8 global attention visits**: `17,408 B/token` (`17 KiB/row`; `221.1 MiB` at 13,316 tokens, `3,270.9 MiB` at 197,021).
- **40 compact SWA visits**: `87,040 B/row` (`85 KiB/row`; canonical 1,023-row window = `89,041,920 B = 84.92 MiB`).
- **48 short-convolution states**: `48 × 3 × (2×1024 + 2×2304) × 4 B = 3,833,856 B` (`3.66 MiB`, channel-major).
  Per-token raw conv input is `1.22 MiB/token`, so discrete strided conv snapshots + deduplicated SWA row segments is far
  more compact than archiving per-token conv inputs.

## What changed

In this initial rollout the Hot Session Cache was opt-in (`--hot-cache` / `STRATA_HOT_SESSION_CACHE=1`).
For current defaults and overrides, see [conversation parking](../INKLING2.md#conversation-parking).

### Stage A — Fast packed transfers & exact SWA delta primitives (`src/poly/engine.{hpp,cpp}`, `src/poly/inkling2.{hpp,cu}`)
- Added `fence_stage()` (`up_` + `st_` sync) before borrowing `stage_`.
- Added `Inkling2::try_unpack_state()` (single H2D into `stage_` + async D2D scatter into `state_` and 40 SWA rings under
  `STRATA_INKLING_STATE_PACK=1`).
- Added parallel host page-faulting (`touch_host_pages()` over the worker pool) and `stage_`-packed/async `save_card_rows()`
  and `load_card_rows()` keeping serialized byte layout identical.
- Added virtual `Engine` SWA delta API (`has_swa_delta`, `conv_state_bytes`, `swa_row_bytes`, `swa_ring_rows`,
  `swa_history_rows`, `resident_swa_rows`, `save_conv_state`, `load_conv_state`, `save_swa_rows`, `load_swa_rows`) with
  conservative physical ring provenance tracking (`ring_valid_lo_`, `ring_valid_hi_`, `ring_high_water_`) across prefills,
  speculative `keep()`, `reset()`, `load_state()`, and `load_swa_rows()`.
- Added optional `on_mark` callback to `Engine::read()` for incremental snapshot capture inside looped Inkling prefills.

### Stage B — Hot Session Cache & Exact SWA State Replay (`src/poly/conversations.hpp`, `src/poly/main.cpp`, `serve/server.py`)
- **Full-Tip Parking (`park_hot`)**: captures the exact committed tip `len(s)` (never un-fed emitted tokens or rejected
  speculative draft rows) and splits global KV into `4,096`-row (`68 MiB`) chunks aligned at pinned anchors so every chunk
  fits inside narrow `stage_` (`128 MiB`) for packed DMA and shared prefixes are deduplicated by `shared_ptr<const Spill>`.
- **Deduplicated SWA Delta Checkpoints (`capture_snap_hot`, `on_decode_commit`)**: snapshots store `3.66 MiB` conv state +
  `SwaSlice` references reusing overlapping SWA rows from prior snapshots with zero row duplication; periodic decode deltas
  capture every `--swa-delta-every N` committed tokens (default 256: `21.25 MiB` SWA + `3.66 MiB` conv).
- **In-VRAM Fast SWA Rollback (`restore_snap_hot`)**: on same-slot rollbacks, skips H2D for every SWA row still proved
  resident by `resident_swa_rows()`.
- **Pinned Semantic Prefix Anchors (`sys_at=` / `pin=`)**: `serve/server.py` (`StrataEngine.turn_key`) emits `sys_at=<end>`
  when the first turn (system + tools) is `>= 2,048` tokens; `drop_of_hot()` preserves pinned anchors and sheds stale
  checkpoints strictly inside `[0, sys_at)` before dropping recent post-pin tail checkpoints.
- **Component-Level Tiered Eviction (`fit_hot`)**: Tier 1 trims non-pinned interior snapshots (keeping tip + pins +
  `parked_tail_rows = 512`); Tier 2 trims pre-tip tail (keeping tip + pins + global KV); Tier 3 evicts whole slots by
  `len / age^1.2`.
- **Hierarchical L1 VRAM / L2 RAM / L3 Spill (`release()`)**: keeps hottest parked blobs in L2 RAM up to
  `--conversation-cache-l2-mib` while `l2_ok()` confirms `min_free_mib + 1024 MiB` host headroom, spilling only colder
  excess blobs to L3 `Spill` so GGUF page cache is protected.
- **Vision isolation**: `resume(ids, has_vision)` excludes vision slots from cross-session parking/reuse.

## Bit-exact GPU verification (RX 6700 XT, Tideglass v3)

All host tests (`conversations_test`: 54,954 checks; `hot_cache_test`: 845 checks; `tools/test_setup_amd.py`: 61/61;
`tools/test_setup_bench_tips.py`: 9/9) and all 9 guarded GPU parity tests (`run_guarded.py --warn-amdgpu-faults`,
`STRATA_FREEZE_EXPERTS=1`, `STRATA_INKLING_EXPERT_KV_MIB=5000`, greedy `T=0`, comparing 64 greedy token IDs and full
`STRSESS` SHA-256 of all 48 conv states + 40 SWA rings + 8 global KV caches) passed:

| Test | Scenario | Reuse | Greedy IDs | `STRSESS` SHA-256 vs Cold Ref |
| --- | --- | ---: | ---: | --- |
| T1 | Full-Tip Parking (`A -> B -> A`) | 2,564 / 2,564 | 64 / 64 | Byte-identical (`b986dad10bf6be94...`) |
| T2 | In-VRAM Fast SWA Rollback (`2649 -> 2563`) | 2,563 | 64 / 64 | Byte-identical (`237e45eed1998098...`) |
| T3 | Multi-Segment Parked SWA Delta (`swa_delta_every=32`, `A -> B -> A@2563`) | 2,563 | 64 / 64 | Byte-identical (`237e45eed1998098...`) |
| T4 | Packed (`STATE_PACK=1`) ↔ Legacy (`0`) cross-`RESTORE`/`SAVE` | 2,564 | — | Byte-identical (`b986dad10bf6be94...`) |
| E1 | Mid-decode `STOP` (+31 emitted, tip=2,530) + switch `B` + return `A` | 2,530 | 64 / 64 | Byte-identical (`d594668255bc40ce...`) |
| E2a | Speculative `--lookup 6 --lookup-epoch` (E4, swaps=64) full-tip + rollback | 2,455 / 2,424 | Step 1: 96/96 | `placement_replays=0`, in-VRAM fast rollback |
| E2b | `DFlash2 Q8_0` + `STRATA_DFLASH2_PARTIAL=1` full-tip + rollback | 2,455 / 2,424 | 64 / 64 both | Byte-identical (`f0418811e0efc185...`) |
| E3 | Tight budget (`768 MiB`, `l2=192 MiB`) + L2→L3 Spill + Tier-1/2 trims | 2,649 | 64 / 64 | Byte-identical (`e3fec14bc2447137...`) |
| E4 | Production natural placement (`T=1.0`, `--expert-swaps 64`, `4 GiB`) | 2,627 | 64 generated | Clean (`49.3 tok/s`) |

## Measured RX 6700 XT end-to-end benchmarks

### 1. 12-Step OpenCode Multi-Session Suite (16K–25K, shared 13,316-tok system prefix, title side-request, switches, rollback, branch)

| Arm | Reused | Wasted | Hot TTFT Sum | vs Prod | Avg Decode | Max RSS | Min MemAvail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prod-mib0` (current prod, `--conversation-cache-mib 0`) | 60.5% | 46,219 | **74.57 s** | 1.00× | 43.86 t/s | 15,568 MiB | 23,318 MiB |
| `stock-park4g-spill` (stock 4 GiB, 4 slots, default Spill) | 80.0% | 7,972 | 37.83 s | 1.97× | 44.30 t/s | 16,001 MiB | — |
| `stock-park4g-nospill` (stock 4 GiB, `STRATA_NO_SPILL=1`) | 80.0% | 7,972 | 35.73 s | 2.09× | 45.38 t/s | 17,088 MiB | 21,923 MiB |
| `hot-4g-d256` (Stage B v1, `d=256`, `16K` chunks) | 81.8% | 4,456 | 29.69 s | 2.51× | 45.12 t/s | 17,225 MiB | 22,506 MiB |
| `hot-4g-d128` (Stage B v1, `d=128`, `16K` chunks) | 82.9% | 2,280 | 27.24 s | 2.74× | 43.48 t/s | 17,150 MiB | 22,515 MiB |
| **`hot-4g-d256-v2` (final: `drop_of_hot` + `4K` chunks)** | **82.8%** | **2,408** | **27.69 s** | **2.69×** | **43.21 t/s** | **16,904 MiB** | **22,412 MiB** |

Key per-step measurements on `hot-4g-d256-v2`:
- Shared-prefix global KV restore across sessions `A / B / C`: **1.2–2.9 ms** (only the unshared post-`sys_at` suffix is uploaded).
- Full 16,734-token restore after unshared 497-token title request: dropped from `272.3 ms` (`16K` chunks) to **32.3 ms** (`4K` chunks fitting narrow `stage_`).
- Step 9 rollback (`-1,200` tokens): `drop_of_hot` kept `22,528`, cutting TTFT from `4,650 ms` to **2,431 ms** and Max RSS by 321 MiB.

### 2. Hard Agentic Suite (1,024-tok reply + last-64-tok tool-call canonicalization + 16K tool return + multi-session A/B/C + 34K rollback)

| Arm | Reused | Wasted | Hot TTFT Sum | vs Prod | Avg Decode | Max RSS | Min MemAvail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hard-prod-mib0` (current prod) | 49.0% | 62,841 | **100.96 s** | 1.00× | 40.71 t/s | 15,580 MiB | 23,462 MiB |
| `hard-stock-park4g` (stock 4 GiB, 4 slots) | 74.7% | 11,480 | 49.40 s | 2.04× | 40.93 t/s | 16,229 MiB | 23,146 MiB |
| **`hard-hot-4g-d256` (final Hot Cache)** | **78.7%** | **3,416** | **37.74 s** | **2.68×** | **42.44 t/s** | **17,125 MiB** | **21,931 MiB** |

Notable per-step wins on Hard Suite:
- Late tool-call JSON canonicalization (last 64 of 1,024 tokens edited): `2,487 ms` (960 wasted) → **1,202 ms** (192 wasted, **2.07× faster**).
- Return to 34K session `A` tip after switching through `B` and `C`: `24,684 ms` (prod) → **1,795 ms** (**13.75× faster**, 0 wasted).
- Return to `B` (512-tok reply tip): `9,819 ms` (prod) / `2,360 ms` (stock) → **812 ms** (**12.1× / 2.91× faster**, 0 wasted).
- 34K mid-interval rollback (`-600` tokens): `11,743 ms` (stock) → **4,903 ms** (**2.40× faster**).

### 3. Deep 197K Context Switch (197,021-ID Tideglass v3 `history.session` ↔ 4K session `B`)

| Arm | `RESTORE 197K` | `park 197K` (1st / inc) | `restore 197K` | 197K Return TTFT | 197K Decode | Min MemAvail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline `deep-spill` | 3,813 ms | 1,074 ms / — | 2,093 ms | 3,780 ms (32 wasted) | 22.10 t/s | — |
| Baseline `deep-nospill` | 3,813 ms | 1,074 ms / — | 246 ms | 1,849 ms (32 wasted) | 22.10 t/s | 19,943 MiB |
| **`stageB-deep-hot-v2` (L2=6G)** | **3,465 ms** | **601 ms / 115 ms** | **143.0 ms** (24.4 GB/s) | **1,023–1,106 ms (0 wasted)** | **22.5–23.6 t/s** | **20,562 MiB** |
| `stageB-deep-hot-l2spill` (L2=1G) | 3,215 ms | 582 ms / 114 ms | 2,112 ms (cold L3) / 244 ms (warm) | 3,350 ms / 1,104 ms (0 wasted) | 21.8–23.6 t/s | 20,994→23,329 MiB |

Cold 197K prefill on this GPU takes ~357.5 s; hot L2 switch returns in **1.02–1.11 s** (**320–350× faster** than cold)
with **zero decode slowdown** and **zero wasted tokens**.

## Separate analysis: Cold-tier lossy KV parking synergy (V6R / LM5)

Evaluated analytically from the exact Inkling 2 Mini geometry and prior measurements in
[2026-11-26-kv-quant-v3.md](2026-11-26-kv-quant-v3.md) without mixing any lossy codec into the exact dense Q8 path:

| Depth | Dense Q8 Global KV | +V6R (`-14.7%`) | +LM5 (`-20.6%`) | +K6/LM5 (`-33.8%`) | Unique+Tip in 4 GiB L2 (Q8 → V6R → K6/LM5) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 20K coding | 332 MiB | 283 MiB | 264 MiB | 220 MiB | 19 → 21 → 24 parked sessions |
| 64K coding | 1,088 MiB | 928 MiB | 864 MiB | 720 MiB | 4 → 4 → 5 parked sessions |
| 128K coding | 2,176 MiB | 1,856 MiB | 1,728 MiB | 1,440 MiB | **1 → 2 → 2** parked sessions |
| 197K coding | 3,271 MiB | 2,790 MiB | 2,597 MiB | 2,165 MiB | 1 → 1 → 1 (`-481 MiB` RAM / `-18 ms` H2D) |

Because SWA deduplication + shared `4K` `BlobPtr` chunks already fit **19× 20K** or **4× 64K** sessions in 4 GiB L2 RAM
with 100% bit-exact dense Q8, lossy V6R only changes session slot capacity at `>=128K` multi-session depth (1 → 2 slots).
Because `kv-quant-v3.md` showed double quantization Q8 → WHT128 → V6R is nearly free (+0.08% L2) and our parked `Rows`
chunks are already immutable `4,096`-row blobs, cold-tier V6R compression can be layered cleanly onto evicted L2/L3 chunks
in the future once full quality gating (~200 graded runs/arm) is performed. Production remains 100% exact dense Q8.

## RDNA2 & UMA portability notes

- All GPU paths target RDNA2 (`gfx1030–gfx1036`) without single-card gating; `stage_` packing automatically adapts to
  `stage_bytes_` and `4,096`-row chunks (`68 MiB`) fit `stage_narrow_` (`128 MiB`) across all RDNA2 cards.
- On iGPU/UMA systems sharing one physical memory pool, `--conversation-cache-l2-mib` + `l2_ok()` (`min_free_mib + 1024 MiB`)
  provides the exact knob to cap resident L2 RAM and spill cold sessions to NVMe/file cache before starving GPU/GGUF allocations.

## Known remaining opportunity (observed, deferred)

- `STRATA_HOT_CACHE_TRACE=1` showed that `save_swa_rows()` takes `3.5 ms` for 114 rows during prefill, `7.8–9.8 ms` right
  after 4K decode, `113 ms` after 197K decode, and `253–260 ms` right after 16K decode (while `save_conv_state()` takes
  `0.2–1.3 ms` and incremental `save_rows()` takes `0.9–2.0 ms`). Investigating that post-decode first-`save_swa_rows` stall
  could cut 16K park latency from ~255 ms down to ~15 ms in a follow-up.

## Production deployment receipt

- Config backup preserved: `~/.config/strata/inkling2-mini/before-hot-session-cache-20261127.json.bak`
- Installed Release binary: `~/.config/strata/inkling2-mini/strata-poly-<commit>`
- Production flags added: `--conversation-cache-mib 4096 --conversation-cache-slots 6 --prompt-cache 8 --hot-cache --swa-delta-every 256`
  (replacing `--conversation-cache-mib 0`), keeping Tideglass Unbound v3 GGUF, tokenizer, vision, API key, `127.0.0.1:8080`,
  and slot directory unchanged.
