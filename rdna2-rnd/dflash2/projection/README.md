# Eight draft activation columns, gfx1031

`run.py` compiles `bench.cu` using the configured HIP toolchain, links the actual engine's
projection archive, and emits assembly and guard receipts outside the repository. It reads
packed checkpoint bytes and uses eight real **surrogate** activation rows from a private DFT1.
It doesn't claim that target tap 5 is the FFN gate input. HIP event times exclude Q8_1
activation preparation, host copies, append, selector and target. The cold arm reads a disjoint
128 MiB allocation before timing. Bandwidth is matrix payload/time, not a hardware counter.

```sh
.venv/bin/python rdna2-rnd/dflash2/projection/run.py --run "$R" --out "$R/new-projections" \
  --weights "$M/DFlash2/Inkling-2-Mini-DFlash2-Q4_0.gguf" --tensors blk.0.ffn_gate.weight blk.0.attn_q.weight
```

On a 6700 XT the FFN gate is 6912 rows of width 2304. One wave per output row traverses the
stock four virtual waves' half-block contributions in reduction order. Loading activations
via LDS lost: in the first run stock / no-LDS / LDS took **107.8 / 69.1 / 137.3 µs** warm.
The no-LDS prototype used 158 VGPRs vs 144 with LDS (wave32, `v_dot4c_i32_i8`, no spills).
Both prototypes silently contracted intermediate products: 33,843/55,296 FP32 outputs differed.
Even `__fmul_rn` wasn't a barrier in this generated ISA. `wave8<false,true>` retains an
explicit rounded `v_mul_f32` for the other virtual waves before adding. In the second run:

| W8 projection | stock → explicitly rounded, warm µs | cold µs | FP32 mismatches |
| --- | ---: | ---: | ---: |
| FFN gate | 107.44 → 83.40 | 129.16 → 112.48 | 0/55,296 |
| attention Q | 70.68 → 57.12 | 85.28 → 77.36 | 0/32,768 |

`DF2_WAVE8=1` selects this last kernel in a newly built **draft DSO only** (8 columns, Q4_0,
width 2304). No target arithmetic changes. Compare full proposals, not just these two matrices;
microseconds here must not be interpreted as a speedup on generated tokens. For Q8_0 the
stock `warps_cols` arm improved warm gate time but lost on small conv and on FFN down.
See `$R/projection-{v2,round-v1,residual-v1}` and `$R/wave-full-v1`.
