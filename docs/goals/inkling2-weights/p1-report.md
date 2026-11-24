# P1 — Pinned header inventory

Parent H0: `e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37`. Authority: [experiment lock](experiment-lock.json) and [RECON pins](recon-pins.json), extracted from the plan's appendix. No production, inference or source model changes.

```sh
python -B -m tools.inkling_weights.transplant inventory --base "$H0" > transplant-plan.json
```

The [plan](transplant-plan.json) was regenerated from a local H0 header and U's first 12 MiB. Their header digests match the RECON pins. U responded with `206`, the pinned commit and the exact requested inclusive range and total length. No donor payload outside that bounded read was fetched. All 704 shared shapes and all 12 typed tokenizer records match. Typed architecture records also match. The 40 U-only scales are not imported. `output.weight` remains Q6_K. Spans are row-validated and bounded by their file, and the manifest gives both GGUF-relative and absolute offsets.

| Stage | New tensors | Total replaced relative to H0 | Added payload bytes | Calculated file bytes |
| --- | ---: | ---: | ---: | ---: |
| H1 | 32 | 32 | 2,755,584 | 15,067,732,672 |
| H2 | 1 | 33 | 231,579,648 | 15,299,312,320 |
| H3 | 36 | 69 | 82,944,000 | 15,382,256,320 |

The selected U payload budget is 687,707,136 bytes. The embedding retains 201,024 rows. These are header and layout checks, not estimates of speed, quality or actual VRAM. Verdict: **validated**; proceed to P2's writer fixtures. The scratch JSON used for atomic result collection was removed; the manifest remains.
