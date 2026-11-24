# PSVST1 schema

The format is little-endian with no checksum or footer. A token frame contains 48 contiguous visit records (13,824 B,
or 13.5 KiB); cache events are additional 64 B records. Byte sizes include any prefix. Consumer: [`collect.py`](collect.py).

## Header (48 B)

| Type | Field |
|---|---|
| `char[8]` | `PSVST1\r\n` |
| `u64, u64` | `first`, `end` input position filter, `[first, end)` |
| `u64` | `epoch_ns`, a host monotonic clock reading at trace open; subtract it when displaying relative times |
| `u32 × 4` | `layers`, `visits`, `experts`, `top`: `24, 48, 128, 8` |

Each following record has an 8 B prefix: `u16 kind`, `u16 total_bytes` (including this prefix), `u32 reserved` (=0).

## Visit (`kind=1`, 288 B)

| Type, in order | Fields / meaning |
|---|---|
| `u64 × 4` | `pos`, `serial`, `host_begin_ns`, `host_end_ns`. `pos` is the input row, `serial` begins at 1 and distinguishes repeats. Host begin/end span visits including pass transitions, from before the first launch to after the last marker enqueue, not GPU completion. |
| `u64 × 4` | `router_wait_ns` (host time from just before the existing router event wait through its return, plus stats accounting), `route_ready_ns` (ordered routing and weights are ready on the host), `cpu_begin_ns`, `cpu_ready_ns` (around the CPU expert call, including setup and intermediate quantization). |
| `u64 × 4` | `gate_begin_ns`, `gate_end_ns`, `down_begin_ns`, `down_end_ns`. Host wall spans of the two worker groups. Gate/up includes activation, late expert reads waited for by workers, the concurrent GPU launch task and its worker barrier. Down includes weighted reduction and its barrier. CPU timestamps are 0 when no CPU experts run. |
| `u64[2] × 3` | `resident_mask`, `asked_mask`, `sent_mask` for this physical layer at routing. Expert `e` is bit `e % 64` of word `e // 64`. Asked is an upload not issued yet; sent means the copy and done event were queued, not that the copy finished. Residency requires `L.slot[e]`, not merely a pending upload. |
| `i64[8]` | `slot_offset`: bytes from the cache arena start for each selected GPU expert (`-1` otherwise). These are offsets, not raw pointers. |
| `u32[8]` | `weight_bits`: exact IEEE binary32 bits in router order, before optional model-specific down scaling. |
| `f32 × 4` | `gpu_visit_ms`, `gpu_experts_ms`, `gpu_join_ms`, `gpu_dispatch_ms`. HIP stream-event elapsed times, described below. |
| `u16 × 2` | `total_asked`, `total_sent` across the whole cache. Counts and masks come from a scan of atomic states, not a simultaneous snapshot across workers. |
| `u8[8], u8[8]` | `ids`, `placement`. Router order, never sorted or grouped by backend. Placement: 0 CPU, 1 GPU, 2 skipped by opt-in tail skipping. |
| `u8, u8, i8, u8` | `visit`, `layer`, `core_pass`, `flags`. Core pass is 0–2, or -1 outside the core. Flags bit 0: `window()`, not `eval()`; bit 1: frozen placement. |

Hosts use one steady clock for all `*_ns` timestamps, except `router_wait_ns`, which is a duration. They are not
aligned with HIP events. JSONL emits masks as 32-digit hex (low word at the right), weight bits as unsigned integers,
and placement as strings. Visits 0–5 map to layer 0–5; visits 6–41 map to `6 + (visit - 6) % 12`; visits 42–47 map
to `visit - 24`. Visits with `visit % 6 == 5` use full attention, the others use SWA.

### HIP time boundaries

All markers are queued on `st_`; they are resolved after an existing stream synchronization. They are not a GPU/CPU
clock correlation or a kernel residency profile.

- `gpu_visit_ms`: start of the visit (before pass-transition work, if any) to after layer-tail work, including stream
  gaps. This is elapsed service time for the measured visit, not token wall time. It excludes embedding, logits and
  inter-visit host work outside the markers.
- `gpu_experts_ms`: before the first cached routed expert launches to after the last expert operation enqueued by
  `moe()`. It may include aggregation or internal launch gaps, not the separate shared-expert phase or a fused tail.
  Zero if no GPU routed experts run.
- `gpu_dispatch_ms`: from the marker after shared-expert work to the start of cached routed expert work, reflecting
  dispatch/handoff gaps. Zero if no GPU routed experts run.
- `gpu_join_ms`: end of cached routed expert work (or shared work if there are no cached routed experts) to a marker
  just before enqueueing use of the CPU expert result. This is an **exposed stream-gap proxy**, not a CPU-only stall
  precisely isolated from host overhead. If all routed experts are on CPU it includes routing/dispatch after the
  shared marker. Zero if the visit has no CPU branch. Do not add host CPU spans to GPU spans as though serial.

## Events (`kind=2..9`, 64 B each)

After the prefix are `u64 host_ns`, `u64 pos`, `i64 a,b,c,d,e`. Event timestamp is when the inference thread logged
the observation; `pos` is the engine's current position, not necessarily the row requested by the event.

| Kind | Name | `a,b,c,d,e` |
|---|---|---|
| 2 | `cover` | requested exclusive end, KV rows before, KV rows after, extra KV bytes requested, growth path taken (0/1) |
| 3 | `shrink_cache` | bytes requested, resident count before slot moves (after settling older uploads), resident count after, mapped arena bytes before, mapped arena bytes after |
| 4 | `follow` | written rows (0 is arrival-only), swap capacity, resident count, frozen (0/1), occupied arena tail offset |
| 5 | `swap_asked` | incoming layer/expert, victim layer/expert (`-1,-1` for an empty refill slot), target slot offset |
| 6 | `upload_observed` | incoming layer/expert, adds a previously empty slot (0/1), swap queue index, slot offset. `follow()` saw success from its existing completion query and attached the new slot. Not the DMA timestamp; frozen or unclaimed uploads might not be logged as complete. |
| 7 | `initial_resident` | layer/expert, arena slot offset, allocated slot bytes, trade class. The initial map is logged before the first captured visit, after any follow or shrink events at that position. |
| 8 | `shrink_slot` | source layer/expert, evicted destination layer/expert, new source offset. `-1,-1,-1` means the source expert was dropped instead of moved. The summary kind 3 follows the individual slot changes. |
| 9 | `initial_pending` | incoming layer/expert, queue index, status (1 asked, 2 sent), target slot offset. Logged alongside kind 7. |

File order preserves entire token frames, not a total timestamp order across events and visits. A frame from an
`eval()` without output logits may be written at the next call's existing synchronization, after that call's
`cover()` records. Use timestamps for host ordering. Reads reject partial frames but cannot detect a lost whole frame
or whole records at EOF, so verify the expected position range and successful run exit.
