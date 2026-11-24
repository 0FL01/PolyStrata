# CP-P0-04: guarded CPU-reference HIP measurements

**Leaf parity passed on both fixtures; FP32 SiTU parity failed.** RX 6700 XT (`gfx1031`), Release probe at engine pin `c0dd0e52`; reference is Ryzen 7 5700X ggml AVX2. Inputs were Tideglass layer 3/expert 0 from the token-64 proxy and actual serial decode at position 2048, visit 6, layer 6, CPU expert 116 (router weight `0x3f94809c`, not applied).

On each fixture the candidate on HIP matched all AVX2 activation Q8_0 words (72 scales and 2,304 codes) and all 1,152 gate and 1,152 up FP32 words, both with candidate packing and with the AVX2 pack supplied. It also matched all 36 scales, 1,152 codes and 2,304 down FP32 words with exact CPU SiTU supplied.

With the existing HIP SiTU fed exact gate/up, the differences were:

| Fixture | SiTU FP32 words different | First (AVX2 → HIP) | Max absolute difference | Subsequent Q8_0 scales/codes and down FP32 |
| --- | ---: | --- | ---: | --- |
| Proxy | 115/1,152 | `[11] 0x3e08ac8b → 0x3e08ac8a` | `9.53674316e-07` | 0/36, 0/1,152, 0/2,304 differ |
| Captured row | 58/1,152 | `[51] 0xbe122059 → 0xbe12205a` | `2.98023224e-08` | 0/36, 0/1,152, 0/2,304 differ |

HIP event time, **µs/expert p50 [min–max]**, from seven batches of 48 repetitions with fixed device-resident weights and input, one expert per launch:

| Fixture | Gate/up arm | CPU-reference HIP | Native HIP single-expert `native_mmvq` | Added p50 cost |
| --- | --- | ---: | ---: | ---: |
| Proxy | Prepacked dot | 18.013 [17.969–18.358] | 10.269 [10.227–10.310] | +7.744 (+75.4%) |
| Proxy | Standalone pack + dot | 22.896 [22.815–23.009] | 14.195 [14.157–14.277] | +8.701 (+61.3%) |
| Captured row | Prepacked dot | 18.008 [17.931–18.296] | 10.279 [10.248–10.308] | +7.729 (+75.2%) |
| Captured row | Standalone pack + dot | 22.878 [22.850–22.931] | 14.190 [14.165–14.218] | +8.688 (+61.2%) |

Candidate packs Q8_0; native packs Q8_1. Timing excludes transfers, fused engine normalization, SiTU, down and aggregation. These are costs for the causal placement screen, not a full-expert cost, placement speedup bound or end-to-end result. No further prototype unless the screen passes.

Source: `~/.cache/opencode-tmp/opencode/arch-cp04-capture.tBbJzJ/cpu-ref-guard/command.log`, SHA-256 `1871b66bcadff97c23a56b8b76057c54a59ad6c3cd08c1f58aeab897ce2ad958`. Guard stop list `[]`; 22 earlier-boot memory-fault warnings and none during the run. Minimum sampled free VRAM 10,429.1 MiB. [Reproduction and fixture provenance](README.md#bounded-cpu-reference-hip-leaf-prototype).
