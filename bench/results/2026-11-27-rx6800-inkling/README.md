# Inkling-2-Mini on RX 6800 — 2026-11-27

Community run on a Radeon RX 6800 (16 GiB, `gfx1030`), Intel Core i5-14600K, and 30.6 GiB of OS-visible RAM.
The card reports PCIe 4.0 ×16 capability; negotiated link speed and memory speed were not recorded. The model
was on an ext4 SSD. Ubuntu 26.04.1 LTS, Linux `7.0.0-44-generic`, amdgpu, ROCm 10.1.0,
HIP `7.16.26385-0000000`. No competing GPU inference ran during the benchmark.

## Results

| Fresh prompt | Forced output | Prefill tok/s, median (range) | Decode tok/s, median (range) |
| ---: | ---: | ---: | ---: |
| 2,048 | 256 | 1467.0 (1463.1–1482.3) | 66.78 (66.77–66.88) |
| 16,384 | 128 | 1406.3 (1398.3–1411.1) | 62.10 (61.92–62.40) |

[Per-run CSV](runs.csv), in execution order, includes the unscored warm-up. All runs read the requested
prompt length and emitted exactly the forced IDs; no tokens were proposed by a drafter or prompt lookup.
Rates are from the engine, excluding model loading and HTTP overhead. Decode timing excludes the first
output token, which is accounted for during prefill. The guard recorded no warnings or stops; minimum sampled free VRAM
was about 1999 MiB. Measured model loads took 2.4–2.5 s.

The [RX 6700 XT Q4_0 report](../2026-11-27-rx6700xt-inkling/README.md) is the reference workload.
That system measured 1027.7 / 51.68 tok/s at 2K and 1051.9 / 46.07 tok/s at 16K. The hosts differ in GPU,
CPU, ROCm, engine revision and expert-profile seed, so this comparison does not isolate the GPU or the
engine changes. No free-generation quality, HTTP serving, disk sessions, vision, or filled 200K context
was tested.

## Build and model

- RedStrata `b2a9f3281fedd06683b3bf784ed6260ac79d45d6`; GGML from the pinned llama.cpp revision
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- Native `strata-poly` Release build with Ninja, `STRATA_ENABLE_HIP=ON`, `STRATA_ENABLE_CUDA=OFF`,
  `CMAKE_HIP_ARCHITECTURES=gfx1030`, and `STRATA_BUILD_TESTS=OFF`. HIP was compiled with
  `/opt/rocm/llvm/bin/clang++` (AMD Clang 24.0.0); host compiler GCC 15.2.0.
- Binary SHA-256: `f151aaa33b556634ef29b58b1dbbc22696f12c26b8e68f6f60279838a32d6d82`.
- [Bartowski Inkling-2-Mini-Q4_0.gguf at `ad66863f`](https://huggingface.co/bartowski/Inkling-2-Mini-GGUF/resolve/ad66863fd3b7f3236c0b7a0862412c9e00f78a6e/Inkling-2-Mini-Q4_0.gguf),
  14.03 GiB; SHA-256 `e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37`.

## Reproduce the run

Use the `env` block verbatim from the [README preset at the tested revision](https://github.com/0FL01/RedStrata/blob/b2a9f3281fedd06683b3bf784ed6260ac79d45d6/README.md#run-inkling-2-mini),
with `LD_LIBRARY_PATH=/opt/rocm/lib` for this installation. Clear inherited tuning overrides.
The preset uses dense Q8_0 global KV, compact sliding-window attention, a 200,000-token context limit,
a 2048-token prompt batch, a 2048 MiB VRAM reserve, eight CPU threads, a 64-swap budget, and no
conversation cache. No draft model was passed.

Export the tokenizer from the measured GGUF using `tools/strata_tokenizer.py`. Encode
`git show 35c899cb722093fd526797decd03956b8e11d6d9:src/poly/main.cpp` using
`Tokenizer.from_dir(...).encode(source, parse_special=False)` without a chat template.
For each size, save the first 2,048 or 16,384 IDs to `PROMPT_IDS`, and the next 256 or 128 IDs,
respectively, to `FOLLOW_IDS`. Files contain comma-separated decimal IDs.

```sh
build/strata-poly --model "$MODEL" --max-context 200000 --kv q8_0 \
  --prompt-batch 2048 --vram-reserve-mib 2048 --threads 8 --expert-swaps 64 \
  --conversation-cache-mib 0 --expert-profile "$PROFILE" --temperature 0 \
  --tokens - --follow "$(cat "$FOLLOW_IDS")" --max-new "$OUTPUT_TOKENS" < "$PROMPT_IDS"
```

Start with a nonexistent profile path for one unmeasured 2K + 256 run. This warms the OS file cache and
produces the local expert-profile seed (SHA-256
`99496ffc401742dfb0c86814c7ddd34dd292f8a9df32d122761f4d1383cdf109`). Use a fresh copy of that seed
and a new engine process for each measured trial. Alternate length order by round: 2K/16K, 16K/2K,
then 2K/16K. No prefix or KV state is reused. The seed is local to this run; the RX 6700 XT used
a different seed. Run under `rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults` with the guard's
device path adjusted for the RX 6800 on this host. The free-VRAM threshold was left at 1536 MiB.

Fixture SHA-256 values below exclude any trailing newline. The earlier RX 6700 XT report's 2K
prompt hash was missing `9c`; it now carries the corrected 64-character value, which matches the
fixture used here. The other three fixture hashes match the reference report exactly.

| Prompt length | Prompt SHA-256 | Continuation SHA-256 |
| ---: | --- | --- |
| 2,048 | `2ef214a7d142601ec402d757c1fe2eb70a8689e9e9c0f1431b366781270b14be` | `31b28c1a2fc13219eac6ac37a2f9a876c059fc767080a3ba85f8241c9001f95e` |
| 16,384 | `d50eb0f1049f2fe7cc42660b540e32597634735320bcedc950713eb9d39a29ed` | `417baa39612443fb31d7ed7c0a672a5155fe227095b7d6078e32772eb2889395` |
