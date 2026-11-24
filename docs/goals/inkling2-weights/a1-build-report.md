# A1: Orca Mix-16G attention output projections

The fixed A1 build replaces only `blk.0.attn_output.weight` through `blk.23.attn_output.weight` on the verified H3 copy. It takes the Q8_0 payloads as stored and imports no scales or metadata. This is a partial transplant, not a reproduction of the donor's abliteration. The whole-donor control (D0) is pending.

## First attempt: destination checksum error

The attempt under `~/.cache/opencode-tmp/opencode/inkling-weights/a1/` stopped with `EIO`. At 20:54:57 local, Btrfs on the root NVMe filesystem reported three checksum failures for inode `61013234` at offset `13735079936`. This inode is not H3 or any of the other retained GGUFs. The journal's device corruption counter rose from 880 at mount to 883; the underlying cause was not established. No GGUF or sidecar was published, and the writer removed its staging file. The original H0 had independently matched its full pin before the attempt.

## Retry: compact CPU build verified on separate storage

The retry used a new private directory on `/mnt/kston` (`/dev/sda1`, separate SATA SSD). All Btrfs device error counters were zero before and after the build. The root filesystem was not repaired or reconfigured.

| Field | Value |
| --- | --- |
| Historical compact candidate | `/mnt/kston/.polystrata-inkling-a1.DPuRxV/Tideglass-OrcaRouter-Inkling-2-Mini-A1.gguf` |
| Parent H3 SHA-256 | `18cf874867a436c2157d5abbab1963f90b5059472afc512ff8f064e3077d56c3` |
| A1 size (parent delta) | 15,467,190,976 bytes (+84,934,656) |
| A1 SHA-256 | `8b8590302a32f9bce291fcba8b749ffbd9cf9aa5cda2e8cd01eea728259f1318` |
| `a1-manifest.json` SHA-256 | `e6ae17afe302490e11217f9c66b03332865e04d752623a33ce1e3e7af3e0650f` |

The writer verified the full H3 hash, accepted the pinned Orca header and 24 bounded payloads (180,486,144 bytes), verified the A0 block-0 payload hash, and independently reread the staged output. The 24 selected projections are the only changed payloads versus H3; the other 680, all shapes, and the typed metadata are unchanged. Relative to H0, the manifest chain records exactly 93 planned replacements and 611 unchanged payloads. Full-file SHA-256 is verified for A1, **not** for the Orca source. Build elapsed 3m 15s, peak RSS 125,240 KiB.

No inference or deployment was run. The current-boot guard still refuses GPU launches after the H0 page fault. The experiment needs a clean boot for H2/H3/A1 smoke and the low-effort baseline. Defer the full D0 download until there is a baseline symptom to measure. Before any promotion, verify the full file hash at the final destination next to H0; if that filesystem cannot reliably verify a copy, keep the current production configuration and resolve storage first.

This compact file was subsequently replaced by the [verified Btrfs reflink](cow-report.md), with matching payloads and metadata. Use the path and SHA-256 in that report for further runs. The SATA compact file was removed after the comparison; its sidecar and logs are retained.
