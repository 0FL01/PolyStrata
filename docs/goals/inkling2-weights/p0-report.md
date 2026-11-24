# P0 — Experiment lock

Parent: original `Inkling-2-Mini-Q4_0.gguf`, SHA-256 `e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37`, 15,064,977,088 bytes. `sha256sum` read the full local file; its containing directory name is not model provenance. No model was edited or downloaded.

The [experiment lock](experiment-lock.json) records the source pins, live binary and config SHA-256, PCI GPU, compiler, exact comparable standalone flags, locked profile seed, corpus and budgets. The shipped Q4_0 flags are on and applicable to eligible shapes on gfx1031; the actual grouping of launches remains to be measured. Greedy, no-lookup standalone work is separate from the sampled production endpoint. This report contains no measurements of model quality or speed.

The [corpus manifest](corpus-manifest.json) freezes eight actual prefix/target windows (four per context length, 256 targets each), two stop-free 512-ID suffixes, 32 quality tasks and 24 benign Orca diagnostic/control prompts. Source/template, corpus and artifact hashes are pinned. `python -B tools/inkling_weights/corpus.py verify` checks the materialized data; the six CPU tests and a byte-identical offline rebuild passed. Python validators interpret only a bounded expression subset; Rust validators check token structure, not execution. The critical smoke subset has not yet been selected from H0 passes. Details: [corpus recipe](../../../tools/inkling_weights/README.md).

## Checkout drift and new comparison lock

`git diff --stat 0051297 c014d6ec -- src/poly` lists ten changed files: `disk_sessions.{cpp,hpp}`, `engine.hpp`, `inkling2.{cu,hpp}`, `inkling_attend.{cu,hpp}`, `inkling_dspark.{cu,hpp}` and `main.cpp`. Changes are DSpark session state, its proposal width and target/drafter attention dispatch. The grouped attention gate is set only with an attached drafter; the no-drafter, no-lookup CLI still uses single-token `eval`. The H1 loader, H2 embedding and H3 projection dispatch, tokenizer and CMake did not change. This is a code audit, not binary parity evidence. `git diff --name-only c014d6ec HEAD` reports only the DSpark results document. Retain `0051297` as RECON provenance and compare all variants against a new H0 on the exact `c014d6ec` ELF in the lock. Do not mix with the previous report's speed numbers.

## Preflight blockers

- **GPU**: CPU-only `Guard(...).boot_clean()` failed with `the GPU took a fault or a reset earlier in this boot; reboot first: … [gfxhub] page fault`. The current boot has 12 matching lines at `2026-11-23T19:15:23+03:00`. `card1` is the intended GPU. No guard bypass, driver action, service restart or inference was attempted. The smallest unlock is a clean boot and unmodified guard preflight before stopping production for the first run.
- **Orca A0 (at P0)**: an authenticated HEAD of the pinned HF `resolve` URL returned `403` / `x-error-code: GatedRepo`; no local donor in the searched model directory or HF cache matched. The later [A0 RECON](a0-report.md) confirmed entitlement and read the pinned header. U returned `206` for a 16-byte probe with the expected commit and total size, so the H inventory was independent of this blocker.

Production was active on loopback and unauthenticated `/v1/models` returned 401. Its files, sessions, binary, process and configuration were not changed. Only a read-only seed copy of the expert profile was created under the experiment directory; subsequent standalone runs must receive writable copies, never the seed or the live profile.
