# KVQ V3 research tools (Inkling global KV quantization)

Results and context: [docs/goals/2026-11-26-kv-quant-v3.md](../../docs/goals/2026-11-26-kv-quant-v3.md).
All scripts expect the private run directory `~/.cache/opencode-tmp/opencode/polystrata-runs/kvq-v3/` and a research
build of `strata-poly` with `capture-follow.patch` (STRATA_KVQ_CAPTURE capture; per-request `kvq_follow=` /
`kvq_logits=` / `kvq_top=` GEN keys). Run GPU steps under `rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults`.

| File | Role |
| --- | --- |
| `capture.py`, `worker.py` | prefill the nested 197K history with capture, SAVE the historical cuts; serve-mode driver |
| `kvq.py` | capture loaders and the CPU global-attention reference (validated against GPU output) |
| `kvcodec.py`, `fastq.c` | CPU reference codecs (Q8_0, V4 refit, q4_0, IQ4_NL, ExLlama H32, TurboQuant, Octave, Lloyd-Max, uniform, KVarN); `gcc -O3 -march=native -fopenmp -shared -fPIC fastq.c -o libfastq.so` |
| `screen.py`, `agg.py`, `dist.py`, `prot.py` | equal-byte codec screen (V or K side), aggregation, distributions, importance protection |
| `mkfixture.py` | write a dense-Q8 SAVE whose cold global V went through a codec (checksum refreshed) |
| `tf.py` | teacher-forced logit comparison on recorded three-turn trajectories |
| `fr.py`, `fragg.py` | free-running graded agent tasks after RESTORE, and their aggregation |
| `vread.hip` | native gfx1031 W1 reader for Q8 K with packed rotated V (uniform / LDS Lloyd-Max / v_perm) |
