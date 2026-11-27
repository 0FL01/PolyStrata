# Tideglass Unbound v3 on RDNA2

Status: active (candidate built; post-reboot qualification). Source: 2026-11-27 Tideglass v3 deployment request (§§1–13).

| Outcome | Acceptance and primary evidence | Status |
|---|---|---|
| R1 (§§3–7,9) | Reproducible compact GGUF beside v2.1; pinned provenance, independent reader checks, full-file and per-payload hashes; 144 Q8 attention + 48 Q4 experts + 512 byte-identical inherited tensors. | Verified |
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
- Measured compact size: 15,542,835,904 bytes (14.4754 GiB). Only 12 relative projections grow. Actual type counts: F32 478, BF16 32, Q8_0 145, Q6_K 1, Q4_0 45, Q4_1 3. [Build commands](../../tools/inkling_weights/README.md#compact-tideglass-unbound-v3).

## Measurement protocol

Use one gfx1031 HIP Release executable per pair. The standalone `STRATA_TOKEN_TIMING=1` callback measures TTFT from generation entry and nearest-rank p50/p90 between emitted IDs (no load, tokenizer or HTTP). All forced IDs must match. Report max RSS from GNU time and board-wide used VRAM sampled by the guard, not a fictitious per-process allocation. Each arm starts with a fresh 2K warm-up, then uses its own frozen profile copy per process. CPU expert time and hit rate are the engine counters. Short inputs match the [same-day benchmark](../../bench/results/2026-11-27-rx6700xt-inkling/README.md); long inputs are slices of the original archived CPython coding prompt IDs, not an old KV snapshot.

Natural tests separately use the frozen task validators, two fixture-backed debugging cases, seeds 42/79 and low/high effort. No generated code is executed. Count reasoning from native token markers; repeated 16-token windows are a repetition proxy, not a semantic loop detector. Time to correct is null on failure; wall time and budget exhaustion remain part of the result. The long task restores a per-model SAVE built from the actual 197,021-ID history and asks about the archived source.

## Checkpoint

The parent hash and the two headers were checked. Only a completed donor matching its full SHA can be used to build. The HIP target built; 18 focused CPU tests passed including actual GGML Q4_0/Q4_1 blocks. A 2K parent-only preflight on gfx1031 returned all 256 forced IDs (860.7 prefill, 48.67 decode tok/s; 1247/3072 experts resident at load). This is **not** an interleaved A/B or a claim of improvement. [Inputs, preflight and paired-run script](../../bench/results/2026-11-27-tideglass-unbound-v3/). No production change.

The first parent-only 197K preflight correctly rejected a truncated continuation: the archived history's 21st suffix ID was EOS (200006), so the engine emitted only 21 of 128. No guard stop. Do not score this run. At 64K/197K keep the original prefix but force the 128 ordinary C++ IDs used at 16K. This leaves natural EOS enabled for both arms.

Parent quality checkpoint: 54/58 correct, 57/58 natural EOS, 10,147 generated / 8,183 reasoning IDs. The 197,021-ID history saved and restored 3,523,425,436 bytes and both repository questions passed. The harder repair task requires reading two fixtures and testing an expression on 88 vectors. It passed 1/4 on v2.1: at high/79 the model used 1,053 thinking IDs across turns, exhausted the last turn and never submitted a test. This provides a concrete stalled baseline; compare the candidate on the same budgets and tools. The parent-only totals do not establish a speedup.

## Stop and resume

At 18:50 MSK all 3,317 HTTP ranges had arrived (27,820,099,264 bytes). Reading the local `.part` for whole-file SHA-256 returned `OSError: [Errno 5] Input/output error`; no donor digest was obtained. The kernel reported a Btrfs data checksum failure for that exact inode (61962607, offset 506822656) on the same filesystem as the production models. Just before that, at 18:48, the kernel logged three general-protection Oopses (#10–12). These are host errors, not HTTP errors or permitted AMDGPU page-fault warnings. No conclusion about which hardware component caused them.

R1–R4 were stopped on that boot. Neither waiter changed the production model or config. Retained the partial donor and model-specific baseline session under `~/.cache/opencode-tmp/opencode/polystrata-runs/tideglass-v3`. Do not treat `ranges.ok` as an integrity check.

The host rebooted at 18:58 MSK and the operator removed the unstable RAM overclock. On the new boot the journal initially showed no Btrfs checksum error, kernel Oops or GPU reset. The cumulative corruption counter remained 888 (not cleared). Rechecked the executable, GGML library, template, corpus and both archived history hashes; 18 focused tests passed. Recover the 8 MiB range `503316480–511705087`, then require the **whole-file** donor and parent hashes before transplant. Use `tideglass-v3/postboot/` for all A/B runs, including fresh **v2.1** quality runs and independently rebuilt long sessions. Do not use pre-reboot wall times to compare models with different RAM settings.

**Recovery and build verified, 19:03 MSK.** Replacing that single range sufficed: the full donor and parent matched their pins. The compact candidate is beside v2.1, SHA-256 `0797a36af531aec9788544123c90c95138f49414530101c6d3dfeed7aeb08bf4`. [Manifest](../../bench/results/2026-11-27-tideglass-unbound-v3/v3.manifest.json) SHA-256 `0ea0943e37f47e1f16a0510cad67c66a87f736c73942ef1b04e3e060cbbcf10d`. Exactly the 192 selected hashes differ; the other 512 match. All 144 attention payloads match the donor. No non-finite expert inputs or residuals; worst absolute error from Q8 was 0.0469971 and median of the 48 per-tensor median expert RMSEs was 0.002084. The writer reread and hashed every output tensor. A third reader (`gguf-py`) agreed on all 704 compact spans and shapes, all 63 typed metadata records against the parent, and all 24 sampled quantized row hashes against the published row probes.
