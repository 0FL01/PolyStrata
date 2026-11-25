# Cold 64K code: dense vs GPU centroid K256

Both full-agent arms passed all 34 independent checks and returned the same visible answer. K256 allocated 48.81 MiB of index memory and finished in 81.890 s versus dense's 83.486 s in this single dense→K256 pair. There was no consistent TG improvement: K256 wrote 35 fewer reasoning IDs. Its first TTFT was 0.571 s lower, not higher, but the pair cannot isolate index time from normal cold-prefill variance. Treat this as a single cold observation, not a throughput or prefill-speedup claim.

RX 6700 XT / gfx1031; Tideglass Inkling-2 Mini Unbound v2.1; pinned `select/centroid/strata-poly` SHA-256 `5e9a0beebf700c151bf0fe05ab894234bf47af15c94a2d5d202e210cfe8743c8` in both arms. Fresh process and identical clean profile for each leg; same Q8_0, batch 2,048, `--cpu-share 0.6`, 4,900-MiB expert-cache cap (actual both 1,143 slots / 4,896 MiB), nonzero low reasoning, T=1 full-vocabulary sampling and seed 42. Identical 65,598 first-prompt IDs, all read natively with resume 0; no restore, no snapshots, no attention or logit dumps, no forced model picks. Both arms reused 65,662 live rows on the actual linked tool return. Guard returned zero with no stop reasons; minimum observed free VRAM 3,055 MiB.

| Metric | Dense | Centroid K256 |
|---|---:|---:|
| Cold first-turn TTFT, s | 77.684 | 77.113 |
| Cold native prompt time, s | 77.677 | 77.106 |
| Cold native prefill, IDs/s | 844.50 | 850.75 |
| Post-tool TTFT, ms | 862.5 | 877.5 |
| TG T1 / T2, IDs/s | 30.42 / 33.61 | 30.62 / 33.32 |
| Inter-token p90 T1 / T2, ms | 38.57 / 34.46 | 38.71 / 36.56 |
| Output IDs T1 / T2 | 65 / 93 | 65 / 58 |
| Total emitted / reasoning IDs | 158 / 84 | 123 / 49 |
| Passed checks | 34/34 | 34/34 |
| Wall to successful grade incl. tool, s | 83.486 | 81.890 |

Each made one valid live `read_case` call and ended with native EOS on T2. Neither retried or reread. Both T1 outputs (65 IDs), both T2 inputs (65,721 IDs) and the visible final function matched. The first output divergence is T2 index 2 (`33175` → `72358`, reasoning); the only phase count that changed was reasoning. This is not token parity. Clock starts just before request render; READY is separate (1.623 / 1.594 s).

All 1,632 first-prompt prefill group records matched byte-for-byte, including the position, routing fingerprint, expert-slot mask and CPU selection; no map was replayed against naturally divergent state. In the live post-tool prefill, only the first 6 of 48 group records matched, with the first differing route fingerprint at record 1,639 (`pos=65663`, `n=57`, layer 6). Matching T2 input IDs does not imply matching K/V after approximate T1 generation. The whole first-turn combined prefill+decode lookup count was 30,292 in both; its VRAM-hit counts were 16,704 / 16,707.

The K256 startup log confirms `mode=centroid2-z top=256 recent=2048 min=8192 page=128 index_bytes=51183616`. It builds the index incrementally from new K rows in cold prompt prefill: 512 completed pages per global visit, 4,096 across eight visits (16 MiB initially populated; 48.81 MiB allocated for the configured context). There is no restore rebuild. The candidate's 64 / 57 ordinary serial generation evals per turn meet the sparse dispatch gate; both first prompt logits are dense. These are source-derived work counts, not per-kernel event times. The existing separate real-KV leaf reports a 0.243–0.315 ms one-visit rebuild and 0.022–0.031 ms per 2,048-row batch at ~65.8K, but those are not this agent pair's index timing. The measured cold price here is allocation plus the aggregate native prefill/TTFT observation; a precise index-time delta is unresolved.

The single-use invocation was `cold_selector.py run`. Raw measurements, policy logs, map records, calls and native IDs are in `cold-centroid-k256-code/`. No repository, previous experiment or service was changed.
