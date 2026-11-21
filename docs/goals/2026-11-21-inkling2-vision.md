# Goal: serve Inkling pictures with the projector in RAM

Status: complete. Source: November 21 request for Inkling-2-Mini vision, RAM-resident encoder, production deployment, subagent review, then commit and push.

## Frozen contract

- R1 (verified): the local F16 `inkling2v` projector runs on CPU in system RAM, stays loaded, and supplies 2304-wide rows to the existing HIP model. A clean CPU build of `6196d980` encoded four PNGs in one helper process, from 4 to 256 rows of 2304 finite floats. Production kept the same encoder PID across requests (712 MiB RSS afterwards; no GPU device open). See [the build recipe](../../tools/vision/INKLING.md).
- R2 (verified): the production service answers based on real image content. Authenticated Chat Completions on `127.0.0.1:8080` returned `blue`, `red`, `blue` for same-size contrasting pictures, `red, blue` for both in order, and `blue` for a tool's picture. `/v1/models` lists text and image. Streaming reasoning, JSON call and tool reply by ID, disconnect and fresh text all passed; 200K context, one slot, and unauthenticated 401 unchanged.
- R3 (verified): independent subagents reviewed the native code, the HTTP path and the CPU helper recipe/config. No blocking findings; aligned the standalone recipe's token limit with its measured run.
- R4 (verified): after the live checks, implementation commit `0696b280` was pushed to `origin/main` and the remote SHA was checked with `git ls-remote`.

## Change envelope and execution

Complete the frozen outcomes with the smallest changes. Reuse `tools/vision`, SVE1/GENI, the embedded template and `Engine::input`. Change Inkling row normalization/capability, server marker handling, and the CPU helper dependency/build as needed. Deployment artifacts, models and credentials stay out of git. No new API, installer catalog, GPU vision backend or change to text defaults. Downtime authorized. Keep a rollback config and binaries; run one HIP engine at a time with the RDNA2 guard.

## Validation and deployment

- RX 6700 XT (gfx1031), Ryzen 7 5700X. On a 16-token prompt containing four real projected rows, the HIP Q8_0 decoder and the CPU reference (`6196d980` with the joined expert scale fix) agreed on the next token and the set of top ten. Largest top-relative logit difference there: 0.725, not bitwise parity.
- Text: for eight forced steps the old and new binaries printed identical top-five probabilities with fixed CPU expert placement. Dynamic placement varied (1,200 vs 1,211 resident experts); the eight top IDs still matched. Single cold 2K/8K runs were 542/649 tok/s before and 573/688 after; these are smoke checks, not evidence of a speedup.
- Deployed `strata-poly-90b7ec72b863` and `strata-vision-d30bac3f0e7a` (binary hash suffixes); CPU helper uses four threads, `max_tokens: 1024`, `gpu: false`, `elastic: false`. Only `exe` and `vision` changed in the active config. Existing binary retained; rollback config: `~/.config/strata/inkling2-mini/before-inkling-vision.json.bak`. Guard minimum over service startup and live checks: 2,875 MiB free VRAM, no faults. A first smoke request capped at 45 output tokens ran out during reasoning; with 256 available, both images returned their colors (42 and 57 completion tokens).
