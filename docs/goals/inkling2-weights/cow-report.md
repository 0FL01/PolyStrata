# Btrfs reflink migration

The active workcopies are under `~/.cache/opencode-tmp/opencode/inkling-weights/cow.SJvJg9/`. Each clone shares its parent's existing extents. The writer appended only the replacement tensors and updated the header; it did not download the donors again. Runtime pilots reuse these paths, not a fresh GGUF for each run. The original H0 still has its pinned SHA-256 and was not written to.

| Variant | File in the workcopy directory | Logical bytes | Verified SHA-256 |
| --- | --- | ---: | --- |
| H0-noop | `H0-noop-Inkling-2-Mini.gguf` | 15,064,977,088 | `e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37` |
| H1 | `H1-Inkling-2-Mini.gguf` | 15,068,810,944 | `5726455d03545708154d218765a04009aaae292903f510aed7c6f52a260941ac` |
| H2 | `H2-Inkling-2-Mini.gguf` | 15,560,917,696 | `a4156a693d2dd3720b077e512a4bad0b974ecb2b8e43424e4bb4f04e83959ea8` |
| H3 | `H3-Inkling-2-Mini.gguf` | 15,752,684,224 | `35f0da378b69aa5d4e249f72a08fe7e2725a1a7a377f25b15039495025d736a2` |
| A1 | `Tideglass-OrcaRouter-Inkling-2-Mini-A1.gguf` | 15,933,170,368 | `eb2a4fb30cd4fafdc92d42d25b09f52c97ad1d2189f89dc95933998cfa25da82` |

The independent reread and manifest check matched all **704 payload hashes per variant**, all tensor shapes and types, and all typed metadata against the old compact versions. The 93 changed tensors in A1 relative to H0 are the same payloads as before the migration. Modified variants have new full-file hashes because the reflink layout retains parent offsets instead of repacking the payloads. The no-op clone is byte-identical to H0. The locked C++ reader uses recorded offsets, but no GPU was run to validate the new files.

`btrfs filesystem du -s --raw` reported zero exclusive data bytes for H0-noop and 192,471,040 exclusive bytes for A1 in the retained family. The family adds 868,193,280 bytes of replacement payloads to the original, plus rewritten headers and filesystem metadata. The five old copies had 76,281,475,072 exclusive bytes. After verifying all replacements, removed only those five task-created GGUFs; their manifests and logs remain. `df -B1` showed a net gain of 59,902,259,200 bytes on the NVMe and 15,456,882,688 bytes on the SATA disk (75.36 GB combined). Device error counters did not increase during the migration. The usage samples and removal log are in the workcopy directory.

All 15 CPU tests passed. The new files are not deployed. See the [subsequent same-boot GPU screening](same-boot-gpu-report.md) for runtime evidence and warning policy.
