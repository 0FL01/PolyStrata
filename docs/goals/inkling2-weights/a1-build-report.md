# A1: Orca Mix-16G attention output projections

The fixed A1 build replaces only `blk.0.attn_output.weight` through `blk.23.attn_output.weight` on the verified H3 copy. It takes the Q8_0 payloads as stored and imports no scales or metadata. This is a partial transplant, not a reproduction of the donor's abliteration. The whole-donor control (D0) is pending.

## First attempt: destination checksum error

The attempt under `~/.cache/opencode-tmp/opencode/inkling-weights/a1/` stopped with `EIO`. At 20:54:57 local, Btrfs on the root NVMe filesystem reported three checksum failures for inode `61013234` at offset `13735079936`. This inode is not H3 or any of the other retained GGUFs. The journal's device corruption counter rose from 880 at mount to 883; the underlying cause was not established. No GGUF or sidecar was published, and the writer removed its staging file. The original H0 had independently matched its full pin before the attempt.

The retry uses a new private temporary directory on `/mnt/kston` (`/dev/sda1`, separate SATA SSD, all device error counters zero before the retry). It fully verifies H3 before copying, then verifies the staged destination again before publication. The root filesystem was not repaired or reconfigured.
