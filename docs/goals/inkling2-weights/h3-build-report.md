# H3: Q8_0 projections in the loop layers

**CPU build verified; GPU smoke deferred.** On the verified H2 copy, the writer replaced `attn_q`, `attn_gate` and `attn_rel` in physical layers 6–17 with exact Q8_0 payloads from pinned U (36 tensors, 191,766,528 bytes fetched). Their shapes were unchanged. All other 668 payloads and the complete typed metadata matched H2 in the independent output reread.

| Field | Value |
| --- | --- |
| Parent H2 SHA-256 | `751608e4aa584969f087aa71ae4d24054ddf227767a645d84729775a3fd9d9b4` |
| H3 size (parent delta) | 15,382,256,320 bytes (+82,944,000) |
| H3 SHA-256 | `18cf874867a436c2157d5abbab1963f90b5059472afc512ff8f064e3077d56c3` |
| Sidecar SHA-256 | `5f0a90d359c159d52c7b021abf8eade564e7eabbc89ea1f6488c8bc243a6efed` |

CPU build elapsed 1m 55s, peak RSS 138,532 KiB. Private files are under `~/.cache/opencode-tmp/opencode/inkling-weights/h3/` (`H3-Inkling-2-Mini-Q4_0-BF16-Q8_0.gguf`, `h3-manifest.json`, `build.stdout`, `build.stderr`). H3 is an offline parent for the fixed A1 transplant, not a promotion verdict.
