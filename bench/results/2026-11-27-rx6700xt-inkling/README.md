# Inkling-2-Mini on RX 6700 XT — 2026-11-27

RX 6700 XT (12 GiB, `gfx1031`), Ryzen 7 5700X, 32 GiB RAM, Linux and TheRock HIP
`7.13.0a20260515`. `main` means `d6913676`; `deployed` means `35c899cb`, the installed
Tideglass engine. No changes were made to the installed service or its profiles.

## Workload

- One process per run. No HTTP, chat template, prefix reuse, disk sessions, tracing or drafter.
- The input is `git show 35c899cb:src/poly/main.cpp`, encoded with the Inkling tokenizer with
  `parse_special=False`. Use the first 2,048 or 16,384 IDs for `--tokens` and the immediately
  following 256 or 128 IDs for `--follow`. Set `--max-new` to the continuation length and
  `--temperature 0`. The timing run only counts if the engine reads and emits the requested
  token counts and the emitted IDs match the continuation.
- Engine options and HIP environment are the ones in the [run example](../../../README.md#run-inkling-2-mini),
  with `--expert-profile` pointing to a fresh copy of the weight-specific profile seed per run.
  Both Tideglass engines use the same seed. The settings include dense Q8_0 global KV, compact
  sliding-window attention, a 200,000-token capacity and a 2,048 MiB VRAM reserve.
- Model groups ran consecutively (Tideglass first). Each group started with one unmeasured 2K run
  on `main` to warm the OS file cache. The two Tideglass builds ran back-to-back at each length,
  reversing their order in the second round. The prompt length order also reversed in round two.
- Every launch ran under `rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults`. Rates are the
  engine's reported tokens/s, excluding model load and HTTP. Decode timing excludes the first
  output token, which the engine picks during prefill.

## Results

[Per-run rates](runs.csv), in tokens/s, include the two warmup runs for reference. The
[headline table](../../../README.md#inkling-2-mini-performance) reports medians of three forced
runs per build, GGUF and prompt length, excluding warmup. All 18 forced runs returned the
expected output IDs; the guard recorded no warnings or stops in any run.

The separate Tideglass greedy check used the same 2K prompt with `--max-new 160` and no
`--follow`. All 160 generated IDs matched across the two builds (SHA-256 of comma-separated
IDs: `525d56733f69fbed1cbae4a3f5d5f73bea4eda87e814ee74fc68eb00829919ab`). Neither check
scores answer quality or measures HTTP latency.

Use the same-day Tideglass rows to compare builds, rather than comparing against previous
README numbers. Expert profiles and host cache state can differ between sessions.

## Inputs

The GGUFs are the [pinned downloads in the run example](../../../README.md#run-inkling-2-mini).
The measured files match the pinned SHA-256 values:
`7af63a2c9362f0ac6fc708443348ead5dbfcb7ecde8cd454ea3572c68864aa72` (Tideglass v2.1) and
`e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37` (Bartowski Q4_0).
The current binary was built with `CMAKE_BUILD_TYPE=Release`, `STRATA_ENABLE_HIP=ON`,
`STRATA_ENABLE_CUDA=OFF` and `CMAKE_HIP_ARCHITECTURES=gfx1031`.

| Binary | SHA-256 |
| --- | --- |
| `d6913676` | `15686e5ba3af7bba3cf62f0633cc8f1e59aeb06a810abb1a79b8c908bf29909c` |
| `35c899cb` | `aa3aab2d4c13efe9c7f60b79f146447c87d7888f50fcfeeb2805c766eb9b5c6e` |

The input checksums below hash the comma-separated decimal token IDs with no trailing newline.

| Context | Prompt SHA-256 | Continuation SHA-256 |
| --- | --- | --- |
| 2K | `2ef214a7d142601ec402d757c1fe2eb70a8689e9e9c0f1431b366781270b14be` | `31b28c1a2fc13219eac6ac37a2f9a876c059fc767080a3ba85f8241c9001f95e` |
| 16K | `d50eb0f1049f2fe7cc42660b540e32597634735320bcedc950713eb9d39a29ed` | `417baa39612443fb31d7ed7c0a672a5155fe227095b7d6078e32772eb2889395` |

Corrected 2026-11-28: the 2K prompt hash was missing `9c`; the value now matches
`baseline-2k-preflight.json` and `measurements.json` in the Tideglass v3 results.

The local profile seeds were not published. Their SHA-256 values were
`0e1f086d16b66a399192cc964a80678b9e5f6c2ec80122eb6abfb1a3ca5a074b` (Tideglass) and
`1419dc0cf14795e6611959491bf0e4de58b906666d888e9a1d39180f7101e2a9` (Q4_0).
Different seeds can change which experts start in VRAM and affect throughput.
