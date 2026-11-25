# Query-aware K256 versus dense Q8 on the RX 6700 XT

## board, 197,021-row restore (seed 42)

| Arm | Grade | IDs / reasoning | Raw weighted TG IDs/s | Grade / success s | Restore s | Correct call? / errors / retries | VRAM free ready / end MiB | Index + extra scratch bytes |
|---|---|---|---:|---|---:|---|---|---|
| dense | pass | 47/99 / 61 | 18.946 | 10.206 / 10.206 | 3.226 | True / 0 / 0 | 3119 / 3174 | 0 + 0 |
| centroid | pass | 47/103 / 69 | 22.968 | 8.958 / 8.958 | 3.120 | True / 0 / 0 | 3113 / 3130 | 51183616 + 204800 |

Both arms: 4,900 MiB expert allowance, 4,968 MiB combined limit; 1,143 resident slots at READY.
- dense post-profile: `bb420bd272a92d319c626a19fc641bc2d882ccb664570e1d4831298086b063ff`, 3072 rows, 13889280 total counts, 32733 bytes.
- centroid post-profile: `786b71208c8d23c3053d0bf1163dfc044bed75dc06b3bd073b280660043f9b63`, 3072 rows, 13890816 total counts, 32735 bytes.
- T1: same input=True; first output difference: identical.
- T2: same input=True; first output difference: 19, 45609→47783, phases=['reasoning', 'reasoning'].
- dense: passed all four checks and loaded the required evidence; expert hits/lookups by turn [(7462, 23406), (14578, 41604)].
- centroid: passed all four checks and loaded the required evidence; expert hits/lookups by turn [(7479, 23044), (15594, 43163)].
- K256 geometry: 1184 / 1184 eligible serial full visits dispatch sparse, 34992.7 attended rows/group. Prefill remains dense.

## current, 197,021-row restore (seed 42)

| Arm | Grade | IDs / reasoning | Raw weighted TG IDs/s | Grade / success s | Restore s | Correct call? / errors / retries | VRAM free ready / end MiB | Index + extra scratch bytes |
|---|---|---|---:|---|---:|---|---|---|
| dense | pass | 38/105 / 61 | 19.529 | 9.721 / 9.721 | 3.110 | True / 0 / 0 | 3095 / 3107 | 0 + 0 |
| centroid | pass | 38/105 / 61 | 23.097 | 8.579 / 8.579 | 3.054 | True / 0 / 0 | 3047 / 3059 | 51183616 + 204800 |

Both arms: 4,900 MiB expert allowance, 4,968 MiB combined limit; 1,143 resident slots at READY.
- dense post-profile: `82b9477ed8ca33d5be92823d896d88ed51dd5b7dce59ea891e95163b2196f2bb`, 3072 rows, 13890816 total counts, 32738 bytes.
- centroid post-profile: `58b4277ebdbc0a83dd823fb77605923e2e4519e568545a695fc178f0e397c38f`, 3072 rows, 13890816 total counts, 32738 bytes.
- T1: same input=True; first output difference: identical.
- T2: same input=True; first output difference: identical.
- dense: passed all four checks and loaded the required evidence; expert hits/lookups by turn [(6205, 19549), (16573, 44084)].
- centroid: passed all four checks and loaded the required evidence; expert hits/lookups by turn [(6303, 19456), (16511, 43975)].
- K256 geometry: 1128 / 1128 eligible serial full visits dispatch sparse, 34996.7 attended rows/group. Prefill remains dense.

## Native mixed 199,104-row restore, 129 emitted IDs (one pair, greedy)

| Arm | TG IDs/s | p90 ms | TTFT ms | Ready / restored / final process VRAM bytes | Index + extra scratch bytes | Expert hits / lookups |
|---|---:|---:|---:|---|---|---|
| dense | 20.481 | 54.608 | 150.797 | 8407646208 / 8369799168 / 8377516032 | 0 + 0 | 23398 / 49536 |
| centroid | 24.384 | 46.515 | 133.606 | 8457986048 / 8420421632 / 8427864064 | 51183616 + 204800 | 22537 / 49536 |

Allocator (allowance, mapped, resident slots, combined cap; READY slots and MiB, end VRAM free MiB): [((5138022400, 5138026496, 1143, 5209325568), '1143', '4896', 3190), ((5138022400, 5138026496, 1143, 5209325568), '1143', '4896', 3154)]
Greedy first difference: None ; TG 1.1906×; p90 0.8518×.
- dense: `RESTORED 199104 3559694824 3151.3`, `DONE 129 199105 130.1 6250.0 length 0 0 199104 23398 49536 0 0 0.0 1 0`; post-profile `12bfe5538e015dc2a4162b3bf6ce0133425a488e68f6b900d3948f0b0b66db9d`, 3072 rows, 13827456 total counts, 32728 bytes.
- centroid: `RESTORED 199104 3559694824 3123.6`, `DONE 129 199105 111.7 5249.6 length 0 0 199104 22537 49536 0 0 0.0 1 0`; post-profile `6c0162301aaf9579edbfee7deebc9f1cca45f1750867684214f6679108f03ceb`, 3072 rows, 13827456 total counts, 32728 bytes.
