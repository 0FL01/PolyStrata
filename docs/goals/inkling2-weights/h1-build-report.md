# H1-B — BF16 small-projection transplant

**Status: passed.** Container, guarded load and finite-logit smoke verified. No deploy. See the [experiment lock](experiment-lock.json) and [P5 baseline](p5-report.md).

- Parent is the separate [P3 no-op copy](p3-report.md), with the pinned H0 SHA-256 `e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37`.
- H1 file: `$WORK/h1/H1-Inkling-2-Mini-Q4_0-BF16.gguf`, **15,067,732,672 bytes**, SHA-256 `09cdcd64d5f47b75cd20deb745f1cf6aca22a9d76ab6b6de678d71efbe4104a1`. Sidecar `$WORK/h1/h1-manifest.json`, SHA-256 `433c495a6fedb5cc8e9b8b3fb02b61dd1468cbb7d48e3416d3618e3f3412f8f4`.
- All 24 sink and eight K/V short-conv projection payloads changed from Q4_0 to the pinned U BF16 bytes, **3,833,856 donor payload bytes**, file delta **+2,755,584 bytes**. The other **672** payload hashes, all names and shapes, and the entire typed metadata block matched the no-op parent. The original named model was not written.
- `Inkling2::load_layer` asks for F32 on all these small matrices; the engine still uploads them as F32. BF16 is the on-disk donor type, not a new runtime path. MoE, embedding and output head remain the same as H0. No full U download.

```sh
WORK="$HOME/.cache/opencode-tmp/opencode/inkling-weights"
python -B -m tools.inkling_weights.transplant build --variant H1 \
  --base "$WORK/p3/H0-noop-Inkling-2-Mini-Q4_0.gguf" --parent-manifest "$WORK/p3/noop-manifest.json" \
  --output "$WORK/h1/H1-Inkling-2-Mini-Q4_0-BF16.gguf" --manifest "$WORK/h1/h1-manifest.json"
```

The writer compared both source pins and reread every output payload before publication. A second manifest/descriptor comparison verified the exact changed set and the 32 expected donor digests. The offline build used 138,656 KiB peak RSS and took 88 s (writer only; not model timing). Private build logs are in `$WORK/h1/`.

The pinned observer loaded H1 on the RX 6700 XT with 1,237/3,072 expert slots and eight CPU threads. It scored all 256 targets on frozen `2k-en` with finite active logits and exact positions/IDs. Minimum sampled free VRAM was 2,008.3 MiB; the unchanged guard reported no stop reasons. Private evidence: `$WORK/h1/smoke/2k-en/`, NLL TSV SHA-256 `4a4e006efc1c0e04dfe0bbdb6addf04235d2e0b0b5429d91040bfa1ad0f31a6b`. H1-E reuses this document and runs the other seven with the same observer, plus the locked task pilot and H0↔H1 timing. No quality or speed conclusion from the smoke alone.
