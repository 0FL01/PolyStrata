"""Lossless CPU reference codec. Each independently indexed tile is 32 rows of one KV head."""

import numpy as np

from snapshot import BLOCKS, TILE, WIDTH, require

N = TILE * BLOCKS
ALIGN = 16
RAW_BYTES = N * (WIDTH + 2)
BITS = np.array([i.bit_length() for i in range(512)], dtype=np.uint8)
SHIFTS = np.arange(8, dtype=np.uint8)
WEIGHTS = 1 << SHIFTS


def aligned(n, grain=ALIGN):
    return (n + grain - 1) // grain * grain


def pack(v, bits):
    # Low bitplane first, low lane first in each byte. All lanes in a stream are a multiple of eight.
    return np.packbits((v.astype(np.uint8) >> SHIFTS[:bits, None]) & 1, axis=1, bitorder="little").tobytes()


def unpack(raw, n, bits):
    if not bits:
        return np.zeros(n, dtype=np.uint8)
    planes = np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(bits, n // 8), axis=1, bitorder="little")
    return np.bitwise_or.reduce(planes * WEIGHTS[:bits, None], axis=0)


def previous(a):
    return np.concatenate((np.zeros_like(a[:1]), a[:-1]), axis=0)


def refs(q, s, tag, reference):
    if tag == 1:
        return previous(q), previous(s)
    if tag >= 2:
        return reference
    return np.zeros_like(q), np.zeros_like(s)


def encode(q, s, tag, reference):
    rq, rs = refs(q, s, tag, reference)
    q, s = q.reshape(N, WIDTH), s.reshape(N, 2)
    raw = b"\xff" + q.tobytes() + s.tobytes()
    out = bytearray([tag])  # 0: zero reference, 1: previous token, 2+: earlier visit
    xor = q ^ rq.reshape(q.shape)
    d = q.view(np.int8).astype(np.int16) - rq.reshape(q.shape).view(np.int8)
    zig = np.where(d < 0, -2 * d - 1, 2 * d)
    db = BITS[zig.max(axis=1)]
    xb = BITS[xor.max(axis=1)]
    outliers = np.abs(d) > 3
    runs = xor != 0
    choices = np.argmin(np.array([np.full(N, WIDTH), 4 * db, 4 * xb, 12 + outliers.sum(axis=1),
                                  1 + 2 * runs.sum(axis=1)]), axis=0)
    for j, method in enumerate(choices):
        if method == 0:
            out.append(0)
            out.extend(q[j].tobytes())
        elif method == 1:
            out.append(1 + db[j])
            out.extend(pack(zig[j], db[j]))
        elif method == 2:
            out.append(10 + xb[j])
            out.extend(pack(xor[j], xb[j]))
        elif method == 3:
            out.append(19)
            out.extend(pack(np.where(outliers[j], 7, d[j] + 3), 3))
            out.extend(q[j, outliers[j]].tobytes())
        else:
            ix = np.flatnonzero(runs[j])
            out.extend((20, len(ix)))
            out.extend(np.column_stack((np.diff(ix, prepend=-1) - 1, xor[j, ix])).astype(np.uint8).tobytes())
    # Scale bytes are exact FP16 bits. Code low/high bytes independently, never perform FP operations.
    for byte in range(2):
        v = s[:, byte]
        base = int(v.min())
        f = v - base
        x = v ^ rs.reshape(N, 2)[:, byte]
        fb, xb = int(BITS[f.max()]), int(BITS[x.max()])
        if N // 8 * xb < min(N, 1 + N // 8 * fb):
            out.append(9 + xb)
            out.extend(pack(x, xb))
        elif 1 + N // 8 * fb < N:
            out.extend((1 + fb, base))
            out.extend(pack(f, fb))
        else:
            out.append(0)
            out.extend(v.tobytes())
    if aligned(len(out)) >= aligned(len(raw)):
        out = raw
    return bytes(out) + bytes(aligned(len(out)) - len(out))


def decode(encoded, references):
    tag = encoded[0]
    qshape, sshape = (TILE, BLOCKS, WIDTH), (TILE, BLOCKS, 2)
    if tag == 255:
        require(len(encoded) == aligned(1 + RAW_BYTES), "bad raw tile length")
        return (np.frombuffer(encoded, dtype=np.uint8, count=N * WIDTH, offset=1).reshape(qshape),
                np.frombuffer(encoded, dtype=np.uint8, count=N * 2, offset=1 + N * WIDTH).reshape(sshape))
    off = 1

    def take(n):
        nonlocal off
        require(n <= len(encoded) - off, "codec stream ends early")
        out = encoded[off:off + n]
        off += n
        return out

    def literal(n):
        return np.frombuffer(take(n), dtype=np.uint8)

    rq, rs = references[tag] if tag >= 2 else (np.zeros(qshape, np.uint8), np.zeros(sshape, np.uint8))
    q = np.zeros(qshape, np.uint8)
    s = np.empty((N, 2), np.uint8)
    for j in range(N):
        row, block = divmod(j, BLOCKS)
        ref = q[row - 1, block] if tag == 1 and row else rq[row, block]
        mode = take(1)[0]
        if mode == 0:
            out = literal(WIDTH)
        elif mode <= 9:
            w = mode - 1
            zig = unpack(take(4 * w), WIDTH, w).astype(np.int16)
            out = ref.view(np.int8) + ((zig >> 1) ^ -(zig & 1))
        elif mode <= 18:
            w = mode - 10
            out = ref ^ unpack(take(4 * w), WIDTH, w)
        elif mode == 19:
            codes = unpack(take(12), WIDTH, 3).astype(np.int16)
            escapes = codes == 7
            out = (ref.view(np.int8) + codes - 3).astype(np.uint8)
            out[escapes] = literal(int(escapes.sum()))
        else:
            require(mode == 20, "unknown code block mode")
            pairs = literal(2 * take(1)[0]).reshape(-1, 2)
            out = ref.copy()
            out[np.cumsum(pairs[:, 0].astype(np.int16) + 1) - 1] ^= pairs[:, 1]
        q[row, block] = out
    for byte in range(2):
        mode = take(1)[0]
        if mode == 0:
            out = literal(N)
        elif mode <= 8:
            w, base = mode - 1, take(1)[0]
            out = unpack(take(w * N // 8), N, w) + base
        else:
            w = mode - 9
            require(w <= 7, "unknown scale mode")
            out = unpack(take(w * N // 8), N, w).reshape(TILE, BLOCKS)
            out = np.bitwise_xor.accumulate(out, axis=0) if tag == 1 else out ^ rs[:, :, byte]
        s[:, byte] = out.reshape(N)
    require(len(encoded) == aligned(off) and not any(encoded[off:]), "bad tile padding")
    return q, s.reshape(sshape)
