# Native selector placement on RX 6700 XT

The default `DF2_BOOKS=host` uses the draft GGUF's mmap directly and leaves the selector weights out of VRAM.
`DF2_BOOKS=gpu` uploads the pair before the live binary sizes its target expert arena. It supports F16 and Q4_0
selector tables, independent of the draft backbone. `DF2_BOOKS=host-lru` keeps the mmap and caches 256 decoded
rows per book (64 sets, four-way LRU). All three modes still score all 16 candidates at every realized
predecessor; they do not reduce the candidate set or quantize the activations.

The live C ABI returns IDs on the host. It downloads two buffers (gate and top-16) for `host` and `host-lru`,
or just an ID buffer for `gpu`: **8,064 vs 28 bytes D2H** at a full seven-position walk, with no selector H2D
per proposal. Both paths still upload the normal draft embeddings and target verifier inputs and run the same
backbone, head and top-k.
The host walk retains the original softmax even when the C caller only consumes IDs; the device walk skips
the unused softmax in that path. Native replay with diagnostics computes it on both paths.
`DF2_BOOKS_TIMING=1` records HIP events around the actual D2H commands and around the GPU selector kernel,
and a host monotonic clock around the post-copy work. `DF2_BOOKS_PINNED=1` stages selector readback in a
persistent `hipHostMalloc` allocation, with the same transfer sizes and number of copies; the host walk
reads the gates directly and copies the candidate list into the proposal buffer. Event elapsed time excludes
final host wakeup and does not equal a CPU wall-clock `hipStreamSynchronize`. Do not price a live throughput
change from instrumented legs alone.

## Measured results

RX 6700 XT (`gfx1031`), Ryzen 7 5700X, Tideglass v2.1 Q6_K target head borrowed in the live runs, greedy W8
with up to seven proposals. The inputs were the same JSON prompt and expert-profile seed, and the target
ELF was the same throughout and each comparison used one opt-in DSO. The fixed-layout control capped the
expert + KV arena at 3072 MiB, froze placement and disabled swaps. The same seed profile and KV charge
produce the same deterministic resident expert ordering; absolute GPU addresses can differ. F16 books are
from the supplied Q4_0-mixed model; Q4_0 books are from the supplied Q4_0 model. Backbone weights and
precision are held fixed *within* each comparison only.

The timing tables cover full seven-position walks only; acceptance also includes the last short walk.
`Selector` is D2H event elapsed time plus the correction (HIP event for GPU, wall clock for CPU) and
the timed host post-copy work. The pinned allocation is 8,092 payload bytes in all three modes.

### F16 pinned live comparison

| Layout | Book placement | D2H p50 (µs) | Walk p50 (µs) | Selector p50 / p90 (µs) | Resident experts | tok/s |
|---|---|---:|---:|---:|---:|---:|
| Fixed | Host mmap | 12.2 | 125.1 | 137.1 / 1,822.2 | 701 | 24.60 |
| Fixed | GPU | 9.0 | 50.0 | 59.2 / 60.9 | 701 | 23.69 |
| Fixed | Host LRU | 12.3 | 120.4 | 133.0 / 788.0 | 701 | 22.53 |
| Natural | Host mmap | 12.3 | 131.9 | 144.2 / 5,862.9 | 1,270 | 36.77 |
| Natural | GPU | 9.3 | 50.5 | 59.8 / 62.7 | 1,230 | 54.27 |
| Natural | Host LRU | 12.9 | 201.0 | 213.8 / 432.5 | 1,274 | 53.95 |

The fixed-layout comparison matched all **115 offered IDs**, **47 accepted**, and **65 emitted IDs** across
all three placements. With pinned buffers, the smaller GPU handoff saves only **3.2 µs** at p50 here. Most
of the GPU arm's selector win comes from the walk, not transfer. It did not win measured end-to-end
throughput in this short fixed-layout run. The F16 LRU hit **51.3%** of predecessor reads and **75.9%** of
successor reads; here it reduced the measured tail but barely changed the median host walk.

Under natural placement (regular reserve and swaps), the GPU F16 pair occupied **195.37 MiB** of VRAM
and the target loaded **40 fewer experts** than with plain mmap, or **44 fewer** than with the LRU. Minimum
sampled free VRAM was **2,000 / 2,026 / 2,022 MiB** for host / GPU / LRU. Both GPU and LRU diverged from
plain mmap at emitted index **18** (zero-based); they matched each other's 65 emitted IDs but only **46/55**
offers at shared anchors and widths. Host accepted **47/115**, versus **52/76** on each of GPU and LRU.
These natural runs measure residency, but the different placements, proposals and round boundaries make
throughput and acceptance-rate comparisons inconclusive. Keep plain mmap as the default; the GPU walk is
faster on the fixed-layout control, but these runs do not establish an end-to-end win for the VRAM cost.

### Retained pageable live and replay results

| Fixed placement, pageable readback | D2H bytes | D2H p50 (µs) | Correction p50 (µs) | Selector p50 / p90 (µs) | Accepted / offered |
|---|---:|---:|---:|---:|---:|
| F16 mmap | 8,064 | 35.7 | 132.3 | 170.9 / 1,727.0 | 47 / 115 |
| F16 device | 28 | 20.9 | 50.4 | 71.7 / 76.7 | 47 / 115 |
| F16 mmap LRU | 8,064 | 35.2 | 92.6 | 131.3 / 190.5 | 47 / 115 |
| Q4_0 mmap | 8,064 | 34.9 | 72.2 | 109.1 / 135.2 | 47 / 115 |
| Q4_0 device | 28 | 21.4 | 45.3 | 66.5 / 70.4 | 47 / 115 |
| Q4_0 mmap LRU | 8,064 | 36.1 | 71.5 | 109.1 / 136.2 | 47 / 115 |

The measured handoff is much larger than the bulk-payload floor discussed in the [CPU study](README.md).
In the F16 leg, eliminating one copy and moving 8 KiB less saved 14.8 µs in median event time; the kernel
itself cost 50.4 µs. The live mmap times include row gathering, the original CPU dot/softmax, and any page
faults or scheduling delays during that region. They are not the prefaulted AVX2 CPU microbenchmark.

For F16, the LRU hit **51.3%** of predecessor reads and **75.9%** of successor reads. Its selector p90 was
**0.19 ms** at **518 KiB** of heap storage, with no added VRAM. These earlier pageable legs were not randomized;
the GPU arm uploads the tables and can warm the file cache before the LRU leg. Each leg has 16 full walks.

In the retained pageable live receipts, all three placements matched all **115 offered IDs**, **47 accepted**,
and **65 emitted IDs**, separately for each profile. F16 replay at 32 aligned Python anchors matched all
**224 proposed IDs**, all accepted-prefix lengths, and all candidate IDs across stock, host, GPU and LRU.
The normal host and LRU replay score/probability values matched stock exactly; GPU FP32 accumulation and
softcap differed by up to **2.48e-5** in score and **1.16e-6** in probability. This is measured ID parity, not a
bit-identical GPU reduction or a near-tie guarantee.

The original successful pageable and replay receipts are under `$R/books-native/`. The resumed run is
under `$R/books-native-resumed/`; use `fixed-pinned/books-results.json` and `natural-pinned/books-results.json`.
All six resumed legs passed the guard. `retained-pageable-summary.json` rechecks the guarded original
receipts within each exact weight profile. The old interrupted series summary should not be used for
cross-arm parity. The original `$R/libstrata-dflash2.so` checksum did not change.

## Reproduce

Run from the repo root. Use a second output directory; do not overwrite the live campaign's DSO. The live
runner holds `$R/gpu.lock` for each guarded leg and creates separate profile copies from one input seed.

```bash
R=$HOME/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon
B=$R/books-native-repeat
M=$HOME/.lmstudio/models/unsloth/Inkling-2-Mini
CFG=$HOME/.config/strata/inkling2-mini/strata-inkling-2-mini.json
PROFILE=$R/live-cost-v1/json-fixed-production/profile
bash rdna2-rnd/dflash2/native/build.sh "$B" build-poly-hip
live=(--run "$R" --binary "$R/live-bin/strata-poly-dflash2-live" --library "$B/libstrata-dflash2.so"
      --config "$CFG" --profile "$PROFILE" --model-dir "$M/DFlash2" --domain json_struct --steps 65)
python rdna2-rnd/dflash2/codebooks/native_matrix.py "${live[@]}" --out "$B/fixed-pinned" \
  --weight Q4_0-mixed --pinned --cap 3072
python rdna2-rnd/dflash2/codebooks/native_matrix.py "${live[@]}" --out "$B/natural-pinned" \
  --weight Q4_0-mixed --pinned
python rdna2-rnd/dflash2/codebooks/native_matrix.py "${live[@]}" --out "$B/fixed-pageable" --cap 3072

export LD_LIBRARY_PATH=$(python -c 'import json,sys; print(":".join(json.load(open(sys.argv[1]))["lib_dirs"]))' "$CFG")
weight=Q4_0-mixed
for mode in host gpu host-lru; do
  out=$B/replay/$weight-$mode
  flock -x "$R/gpu.lock" env DF2_BOOKS="$mode" DF2_BOOKS_TIMING=0 DF2_BOOKS_PINNED=0 \
    python rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults --output "$out" -- \
    "$B/dflash2-replay" --target "$M/Tideglass-Inkling-2-Mini-Unbound-v2.1.gguf" \
    --model "$M/DFlash2/Inkling-2-Mini-DFlash2-$weight.gguf" --replay "$R/taps/code_python.dft1" \
    --mode all-positions --max-steps 32 --dump-json "$out/replay.json" --dump-steps "$out/steps.jsonl" --dump-hidden
done
python rdna2-rnd/dflash2/codebooks/compare_native.py "$B/replay/$weight-host/steps.jsonl" \
  "$B/replay/$weight-"{gpu,host-lru}/steps.jsonl
```
