# Tideglass Unbound v3 on RDNA2

Status: active. Source: 2026-11-27 Tideglass v3 deployment request (§§1–13).

| Outcome | Acceptance and primary evidence | Status |
|---|---|---|
| R1 (§§3–7,9) | Reproducible compact GGUF beside v2.1; pinned provenance, independent reader checks, full-file and per-payload hashes; 144 Q8 attention + 48 Q4 experts + 512 byte-identical inherited tensors. | Pending |
| R2 (§§9–10) | Guarded gfx1031 load and paired 2K/16K/64K/197K forced timings, including resource and expert metrics, on one Release binary. | Pending |
| R3 (§11) | Frozen tasks and multi-step coding/tool/long-context workloads: success, thinking, iterations and elapsed time to a correct answer across seeds/efforts. | Pending |
| R4 (§12) | Same production API, vision and context; v3-only sessions and profile; verify argv, chat, reasoning, actual tool round-trip, client and SAVE/RESTORE. | Pending |
| R5 (§13) | Publish pipeline, tests, measured results and report; clean private intermediates. | Pending |

Scope: tools/inkling_weights, benchmark results, README, local model artifacts and production config. Preserve parent tokenizer, metadata and shapes, W1/W3 and expert order, Q4_1 down in blocks 0–2 only. No engine optimization or unrelated changes. Prefer a verified high-precision merged source for experts; use pinned Q8 if unavailable. Recreate KV, never share writable state between models.

Complete the frozen outcomes within this envelope. Stop at a proven external blocker and record the smallest unlock.

R2 envelope: a 197K comma-separated ID argument exceeds Linux's per-argument limit. Allow `--tokens -` to read the same ID list from stdin for standalone forced runs; no engine math or HTTP changes.

## Source and host checks

- On `phoenix`, `origin` is `0FL01/PolyStrata`; the requested `0FL01/RedStrata` URL and `origin` resolved to the same HEAD, `8e3762be`. The user unit `strata.service` was **inactive** before this work. Its launcher reads `~/.config/strata/server.env`, not the similarly named experimental GSQ config.
- The installed config points to `strata-poly-35c899cb`, v2.1, `inkling-2-mini` on `127.0.0.1:8080`, 200,000 context, Q8_0 KV, 2048-token prompt batch, 2048 MiB reserve, 8 CPU threads, 64 swaps and disk sessions. Vision is a separate CPU encoder (`strata-vision-690b87d8`, original Q4_0 model, `mmproj-F16.gguf`); keep it and the external tokenizer unchanged. Preserve API authentication, flags and bind address.
- Full parent SHA-256: `7af63a2c9362f0ac6fc708443348ead5dbfcb7ecde8cd454ea3572c68864aa72`. Btrfs had 159 GiB available; a same-filesystem `cp --reflink=always` succeeded. No production profile or slot is needed to build the candidate.
- [Pinned donor inventory](https://huggingface.co/api/models/nokeepai/Inkling-2-Mini-Unbound-GGUF/revision/accd51aadc5aaef0d16f9a3e5d71f0b5dab02e9d?blobs=true): v3 Q8_0 is 27,820,099,264 bytes, advertised SHA-256 `a497670d6d9ac3be262953c0679dc064bc92ba8af198e6f7073bc3d41f7efd2a`. HTTPS returned 206 and the correct total size. The card lists `-v3-BF16.gguf`, but the pinned file inventory does **not**; the unversioned BF16 is not a verified v3 artifact.
- [Merged checkpoint](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/tree/14d5c3e1ced6140d2f150feb74e4c60f65be7bf5): the 148,408-byte safetensors header describes packed U8 + F8 scales in layers 0–19, F8 + U8 scales in layers 20–23, not BF16 master expert matrices. The Prime adapter header contains 432 entries. Upstream's [converter](https://github.com/nokeepai/inkling-2-mini-unbound/blob/3ea6c5b296a2aacfcbbf24822dd977b5d4cdf4bd/scripts/v3stack/gguf_v3.py) proves the stock checkpoint against an official BF16 GGUF *before* applying changed values. It is not a BF16 payload extractor for this single merged file. Use the permitted pinned Q8 fallback rather than inferring a row permutation from a square matrix.
- Parent/donor aligned header lengths: 11,977,792 / 11,977,408 bytes; alignment 32. The 704 names/shapes and the architecture/tokenizer typed values agree. Joined gate/up `ne=[2304,2304,128]`, down `ne=[1152,2304,128]`; each expert is contiguous in GGUF order (innermost dimension first). Preserve that physical order for every row, including W1/W3. Reference GGML blocks: 32 elements in 34 / 18 / 20 bytes for Q8_0 / Q4_0 / Q4_1. No adapter merge or second gate/up join on GGUF data.
- Planned compact size from the two headers: 15,542,835,904 bytes (not a measured output yet). Only 12 relative projections grow. Planned type counts: F32 478, BF16 32, Q8_0 145, Q6_K 1, Q4_0 45, Q4_1 3. [Build commands](../../tools/inkling_weights/README.md#compact-tideglass-unbound-v3).

## Measurement protocol

Use one gfx1031 HIP Release executable per pair. The standalone `STRATA_TOKEN_TIMING=1` callback measures TTFT from generation entry and nearest-rank p50/p90 between emitted IDs (no load, tokenizer or HTTP). All forced IDs must match. Report max RSS from GNU time and board-wide used VRAM sampled by the guard, not a fictitious per-process allocation. Each arm starts with a fresh 2K warm-up, then uses its own frozen profile copy per process. CPU expert time and hit rate are the engine counters. Short inputs match the [same-day benchmark](../../bench/results/2026-11-27-rx6700xt-inkling/README.md); long inputs are slices of the original archived CPython coding prompt IDs, not an old KV snapshot.

Natural tests separately use the frozen task validators, two fixture-backed debugging cases, seeds 42/79 and low/high effort. No generated code is executed. Count reasoning from native token markers; repeated 16-token windows are a repetition proxy, not a semantic loop detector. Time to correct is null on failure; wall time and budget exhaustion remain part of the result. The long task restores a per-model SAVE built from the actual 197,021-ID history and asks about the archived source.

## Checkpoint

R1/R2 in progress. The parent hash and the two headers were checked; the Q8 donor is being fetched to `~/.cache/opencode-tmp/opencode/polystrata-runs/tideglass-v3`. Only a completed donor matching its full SHA can be used to build. The HIP target built; 18 focused CPU tests passed including actual GGML Q4_0/Q4_1 blocks. A 2K parent-only preflight on gfx1031 returned all 256 forced IDs (860.7 prefill, 48.67 decode tok/s; 1247/3072 experts resident at load). This is **not** an interleaved A/B or a claim of improvement. [Inputs, preflight and paired-run script](../../bench/results/2026-11-27-tideglass-unbound-v3/). Next: finish full-file verification and compact build, run paired performance and natural quality, then verify production. No production change at this checkpoint.

The first parent-only 197K preflight correctly rejected a truncated continuation: the archived history's 21st suffix ID was EOS (200006), so the engine emitted only 21 of 128. No guard stop. Do not score this run. At 64K/197K keep the original prefix but force the 128 ordinary C++ IDs used at 16K. This leaves natural EOS enabled for both arms.
