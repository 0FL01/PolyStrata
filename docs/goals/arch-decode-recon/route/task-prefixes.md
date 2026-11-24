# P1 follow-up: four short external task prefixes

On the same Tideglass/RX 6700 XT setup, the one-pick previous-pass rule recalled **23.5–28.1%** of the observed
CPU misses in these short prefixes, with **26.6–34.3 MB/scored token** of false-positive full-expert payload.
This supports the earlier routing signal, but does not change the decision against full-weight speculative H2D.

Each capture has 16 computed serial decode steps after a **49–94 token prompt** (the `2k-*` names are fixture IDs,
not context lengths). Score steps 2–16 to give every arm the same first-token warm-up: **15 tokens and 360 target
visits per trace**. The only training input for learned arms was the fixed 256-row mixed 2K trace; no task rows
were used for training or tuning. Same-token previous-pass and previous-token rules do not learn parameters.
All arms filter source-resident and source-pending experts. Score only the two distinct adjacent targets
(`1→2` and `2→3`), against actual CPU placement; no paired `1→3` double-counting.

Cells show **CPU miss recall; false-positive MB per scored token**. False-positive bytes count a full 4,478,976-byte
core expert for each issued candidate that is *not* an observed CPU miss at the target visit. These are proposed
bytes, not measured transfers. Cap is per source visit.

| External trace | Prompt tokens | Previous pass, cap 1 | Previous pass, cap 4 | Previous token, cap 1 |
| --- | ---: | ---: | ---: | ---: |
| English reasoning (`2k-logic-01`) | 49 | 158/563 (28.1%); 26.6 MB | 235/563 (41.7%); 54.9 MB | 20/563 (3.6%); 50.5 MB |
| Python (`2k-py-01`) | 56 | 149/635 (23.5%); 31.4 MB | 228/635 (35.9%); 65.4 MB | 31/635 (4.9%); 52.9 MB |
| Tool call (`2k-tool-01`) | 70 | 129/517 (25.0%); 26.9 MB | 188/517 (36.4%); 47.2 MB | 34/517 (6.6%); 43.9 MB |
| Tool-result replay | 94 | 126/524 (24.0%); 34.3 MB | 179/524 (34.2%); 60.9 MB | 26/524 (5.0%); 49.9 MB |
| Pooled, 60 scored tokens | — | **562/2,239 (25.1%); 29.8 MB** | 830/2,239 (37.1%); 57.1 MB | 111/2,239 (5.0%); 49.3 MB |

For comparison, mixed-2K-trained target-pass frequency and conditional transitions at cap 1 recalled **2.3%** and
**3.2%** pooled, with **103.6** and **102.2 MB/token** of false-positive payload. Increasing the previous-pass cap
from 1 to 4 raised pooled recall by 12 percentage points while nearly doubling false-positive payload. These are
four short, correlated decode prefixes; later tokens, task quality, tool success, numerical parity, transfer
interference, and E2E throughput were not evaluated. The replay is a fixed prompt, not evidence of a tool loop.

**Provenance and reproduction:** The prompt lengths and SHA-256 values in `$D/task-prompts/index.json` match the
run metadata and prompt files; all task guards completed without a stop. Use the private paths in the
[main report](tideglass-recon.md): `$D/task-ID/run/visits.psv`, `$S/task-prefixes/ID.jsonl`, and
`$S/task-prefixes/ID-analysis.json` (includes raw trace hash and both model and prompt pins). The sole training
file is `$S/tideglass-2k.jsonl`, SHA-256 `a6b30c6899017ed8c674758fe0ffa9bd2cfb3dd3a3ee2c5f9bfc9635517811b0`.
To rerun from normalized rows:

```sh
nice -n 19 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B docs/goals/arch-decode-recon/route/analyze_routes.py "$S/task-prefixes/$ID.jsonl" \
  --train "$S/tideglass-2k.jsonl" --budgets 1 4 --output "$OUT/$ID-analysis.json"
```
