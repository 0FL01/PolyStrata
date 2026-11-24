# A0 — Authorized OrcaRouter Mix-16G RECON

2026-11-24. **A0 passed.** The exact gated donor is accessible through the active Hugging Face login. D0 has no known structural blocker in the locked engine; A1's 24 projections pass the static contract. Neither has run. See the [inventory](orca-inventory.json) for exact spans, types and hashes, and the [experiment lock](experiment-lock.json) for H0, the shipped Release HIP `gfx1031` ELF (`c014d6ec`), config and corpus pins. No model was edited.

## Access and byte evidence

The donor is [`orcarouter/Inkling-2-Mini-abliterated-GGUF` at `20b83559fef0b641145603d089210269ae721a74`](https://huggingface.co/orcarouter/Inkling-2-Mini-abliterated-GGUF/tree/20b83559fef0b641145603d089210269ae721a74), file `Inkling-2-Mini-abliterated-Mix-16G.gguf`, **15,721,831,424 bytes**. The Hub advertised LFS SHA-256 `0a8784aa059d6dba83ff99780371180b5376f2056af893c521b37792c64ffa02`; this is not a locally verified full-file hash. Mix-12G was not used.

Using the active Hugging Face token, `GET /api/models/<repo>/auth-check` returned **200**. The authenticated pinned `HEAD /<repo>/resolve/<revision>/<filename>` returned **302** with the expected commit and size. For each range, the origin confirmed that commit, size and advertised LFS digest; the final response had one **206** `Content-Range` with the exact span and total, and the local byte count matched:

| Body | Inclusive range | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| First 12 MiB (header probe, not a whole-file hash) | `0–12582911` | 12,582,912 | `cd27475de0fc987a366576bc7a80256f9dd20db885f8da403156837f0a52bf5e` |
| `blk.0.attn_output.weight` | `7980820736–7988340991` | 7,520,256 | `c93412e658398e8fbc3c46afe75b44a9e1c295084fe5b29673b1dee2c796abbf` |

To repeat a probe, resolve the same pinned URL with `curl --disable --fail --location --proto '=https' --proto-redir '=https' --range "$start-$end" --max-filesize "$bytes" --max-time 120 --suppress-connect-headers -H @-`. Supply the `Authorization` header on stdin from `huggingface_hub.get_token()`; do not put the token in argv or print headers or signed redirect URLs. Use a private output directory, keep only allowlisted response fields, and check the first origin commit, final 206, exact range/length and SHA-256 against the inventory. For an authorized local file, verify the full byte count and SHA-256 and read the same spans locally. The existing `transplant inventory` command covers H0/U only; this check used its bounded `parse_header` and `IndependentHeader` readers on the acquired data.

## Header and runtime compatibility

Both decoders agreed on the 724 descriptors. The bounded parser checked 32-byte alignment, row geometry, no overlapping payloads, and file bounds. The donor's data starts at byte 11,978,752. All 704 H0 names and shapes are present; all 12 typed tokenizer records match. Architecture is `inkling2_loop`. Of the typed architecture keys, the donor adds `expert_gating_func=2` (sigmoid, which Inkling already uses) and omits `feed_forward_length=1152`. The loader uses `expert_feed_forward_length=1152`, present in both. The donor's `split.no=0` and `split.count=0` take the single-file path in `src/poly/runtime.hpp`.

| Path | Verified statically | Still unverified |
| --- | --- | --- |
| **D0** — full control | Full model geometry matches; the token table and output head are both Q6_K `[2304, 201024]`. Fused gate/up is `[2304, 2304, 128]`: 20 NVFP4 (blocks 0–19), 3 IQ4_XS (20, 22, 23), 1 Q5_K (21). Down is `[1152, 2304, 128]`: 17 IQ4_NL and 7 Q5_1. All 20 required joined NVFP4 gate/up scales are present as F32 `[128]`; there are no down scales. A host-only probe linked the existing Release `cpu_experts.cpp` object and ggml CPU libraries and called `cpu_expert_types_ok(gu, down, 2304, 1152)` for all four actual D0 pairs; all passed. | Full SHA-256, scale values, actual RAM/VRAM/cache fit, finite load/run and behavior under the locked flags. `STRATA_HIP_NVFP4_MMQ` is not enabled in the lock; keep the existing path for non-MMQ experts, not a new flag to make D0 pass. |
| **A1** — only attention output | Exactly `blk.0.attn_output.weight` through `blk.23.attn_output.weight`, all Q8_0 `[3072, 2304]` in GGUF order, already supported by the native path. None has or needs a separate scale. Total donor payload **180,486,144 bytes**. Applying those descriptors over planned H3 projects **15,467,190,976 bytes** on disk, **84,934,656 bytes** above H3. The other 680 H3 payloads and H3 metadata must remain unchanged. | Verified H3 parent, remaining 23 selected payload reads or verified full local donor, output hash, guarded runtime fit and measured effect. |

Loader paths checked: `src/poly/inkling2.cu` (`load_geometry`, `load_layer`, `route`), `src/poly/engine.cpp` (`load_shape`, `load_moe`, `upload_as`, `global_scale`, `lin_raw`), and `src/kernels/cuda/native_mmvq.cu` (`native_mmvq_supported`). **A2 remains unopened.** The down formats and absent scale dependencies are inventory facts, not reasons to import the MoE or to attribute a result uniquely to abliteration rather than quantization.

## Handoff

- No full-file download. Scratch had 128,972,587,008 available bytes at RECON; that is enough for the 15.72 GB donor on disk, not proof of runtime fit. D0-B should acquire the exact complete file in private scratch without changing the original, check size and full LFS SHA-256, then attempt a guarded load and finite smoke with the locked reserve, KV and kernels. Compare D0 and verified H3 at the frozen `low` effort. If D0 cannot run under the lock, state why and mark any bounded A1 fallback **whole-donor control not performed**. The manifest lists the 24 ranges for that fallback.
- At A0 the next GPU checkpoint was P3 no-op parity after a clean boot; see the [execution status](../2026-11-24-inkling2-weights-execution.md) for current progress. No GPU run, reboot, production stop or config change occurred at A0; `strata.service` was still active on H0 with the same PID and config SHA-256.
- Private scratch `~/.cache/opencode-tmp/opencode/inkling-weights/a0-recon.shlOw9/` holds the two bounded bodies, sanitized receipts and CPU check output for verification. Removed the temporary CPU probe executable. No credentials, redirect URLs or model bytes are in git.
