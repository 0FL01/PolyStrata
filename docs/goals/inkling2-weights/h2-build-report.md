# H2: Full Q8_0 token embedding

**CPU build verified; GPU smoke deferred.** The writer used the verified H1 workcopy, not the original GGUF. It copied the exact pinned U `token_embd.weight` payload from Q8_0, retaining all 201,024 rows (including padding) and width 2,304. This is a raw tensor copy, not dequantization or requantization.

| Field | Value |
| --- | --- |
| Parent H1 SHA-256 | `09cdcd64d5f47b75cd20deb745f1cf6aca22a9d76ab6b6de678d71efbe4104a1` |
| H2 file size (parent delta) | 15,299,312,320 bytes (+231,579,648) |
| H2 SHA-256 | `751608e4aa584969f087aa71ae4d24054ddf227767a645d84729775a3fd9d9b4` |
| Embedding payload SHA-256 | `a88f12997684abb81416349b8e03cae29cf2ea111d0f2ad0e725bf3d87d138b1` |
| Sidecar SHA-256 | `dbf0aa1a4ae91a81000b9b8223d15891a6b5622de35a2206146c95b4ee59dfd0` |

The writer independently reread the output header and all tensor payloads. The other 703 payloads (including `output.weight`) match H1 by name and hash; all shapes and typed metadata are unchanged. CPU build elapsed 3m 1.5s, peak RSS 138,672 KiB. Private files are under `~/.cache/opencode-tmp/opencode/inkling-weights/h2/` (`H2-Inkling-2-Mini-Q4_0-Q8_0-Embedding.gguf`, `h2-manifest.json`, `build.stdout`, `build.stderr`). No engine was loaded and this is not a quality or performance verdict.
