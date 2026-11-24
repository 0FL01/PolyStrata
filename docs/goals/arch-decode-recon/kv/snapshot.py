"""Read-only Inkling STRSESS v1 and a bounded, independently reread KV sample."""

import hashlib
import mmap
import os
import stat
import struct
import subprocess

import numpy as np

VISITS = tuple(range(5, 48, 6))
PHYSICAL = dict(zip(VISITS, (5, 11, 17, 11, 17, 11, 17, 23)))
PASS = dict(zip(VISITS, (None, 1, 1, 2, 2, 3, 3, None)))
STRIDES = (1024, 64, 1024, 64)  # K codes, K scale bits, V codes, V scale bits
VISIT_ROW = sum(STRIDES)
TILE = 32
HEADS = 8
BLOCKS = 4
WIDTH = 32


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read_at(fd, n, offset):
    data = os.pread(fd, n, offset)
    require(len(data) == n, f"short read at {offset}")
    return data


def identity(fd):
    st = os.fstat(fd)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def parse(fd, checker):
    size = os.fstat(fd).st_size
    require(size >= 80, "short STRSESS container")
    magic, version, header, model, config, payload, r0, r1, hhash = struct.unpack(
        "<8sII6Q", read_at(fd, 64, 0))
    require((magic, version, header, r0, r1) == (b"STRSESS\x01", 1, 64, 0, 0), "not STRSESS v1")
    require(payload == size - 80, "payload size does not match file size")
    end = size - 16
    phash, end_magic = struct.unpack("<Q8s", read_at(fd, 16, end))
    require(end_magic == b"STRSEND\x01", "bad end marker")
    at = 64

    def take(fmt):
        nonlocal at
        n = struct.calcsize(fmt)
        require(n <= end - at, "record crosses end of payload")
        out = struct.unpack(fmt, read_at(fd, n, at))
        at += n
        return out

    def skip(n):
        nonlocal at
        require(0 <= n <= end - at, "data crosses end of payload")
        old, at = at, at + n
        return old

    tag, layout, ctx, vocab, width, layers, qfmt, state, row_bytes, rows_per, *zeros = take("<18q")
    require(tag == 0x494e4b4c494e4732 and layout in (1, 2) and not any(zeros), "not Inkling layout 1/2")
    require(ctx > 0 and 0 < vocab <= 2**31 and (width, layers, qfmt) == (2304, 24, 1), "not Inkling Q8_0")
    expected_state = 48 * 3 * (2 * 1024 + 2 * 2304) * 4 + 40 * min(ctx, 1023) * VISIT_ROW
    if layout == 2:
        expected_state += width * 4 + 2 * min(ctx, 1024) * VISIT_ROW
    require(state == expected_state and row_bytes == (8 + (layout == 2)) * VISIT_ROW and
            rows_per == (16 << 20) // row_bytes, "unexpected Inkling state/row geometry")
    require(take("<qqQ") == (0, layers, 0), "unexpected layer range or control vector")
    pos, = take("<Q")
    require(0 < pos <= ctx, "token count out of bounds")
    token_offset = skip(pos * 4)
    tokens = hashlib.sha256()
    for start in range(0, pos, 1 << 16):
        data = read_at(fd, min(1 << 16, pos - start) * 4, token_offset + start * 4)
        ids = np.frombuffer(data, dtype="<i4")
        require(int(ids.min()) >= 0 and int(ids.max()) < vocab, "token ID out of bounds")
        tokens.update(data)
    require(take("<QQ") == (0, state), "unexpected image count or state size")
    skip(state)
    require(take("<6Q") == (0,) * 6, "unexpected state arrays/used/checkpoints")
    nrec, = take("<Q")
    require(nrec == (pos + rows_per - 1) // rows_per, "unexpected KV record count")
    records = []
    for first in range(0, pos, rows_per):
        rows = min(rows_per, pos - first)
        require(take("<7qQ") == (1, rows, 0, 0, 0, 0, 0, rows * row_bytes), "unexpected KV record shape")
        records.append((rows, skip(rows * row_bytes)))
        require(take("<4Q") == (0,) * 4, "unexpected extra KV arrays")
    require(at == end, "unparsed payload")
    got = subprocess.check_output([checker, str(fd), str(payload)], pass_fds=(fd,)).decode().splitlines()
    checks = [f"{hhash:016x}", f"{phash:016x}"]
    require(got == checks, f"STRSESS checksum mismatch: computed {got}, stored {checks}")
    return dict(bytes=size, context_capacity=ctx, committed=pos, inkling_layout=layout, kv_record_rows=rows_per,
                kv_records=nrec, model=f"{model:016x}", config=f"{config:016x}", header_xxh_style=checks[0],
                payload_xxh_style=checks[1], committed_token_ids_sha256=tokens.hexdigest()), records


def sample(path, checker, prefix=None, count=12):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "snapshot is not a regular file")
        before = identity(fd)
        meta, records = parse(fd, checker)
        limit = meta["committed"] if prefix is None else prefix
        require(TILE <= limit <= meta["committed"], "prefix must cover at least one tile and be committed")
        tiles = limit // TILE
        count = min(count, tiles)
        starts = [TILE * (i * (tiles - 1) // max(1, count - 1)) for i in range(count)]
        samples = {visit: [[], [], [], []] for visit in VISITS}
        rows_per = meta["kv_record_rows"]
        with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as mm:
            for start in starts:
                pos = start
                while pos < start + TILE:
                    record, local = divmod(pos, rows_per)
                    rows, offset = records[record]
                    take = min(rows - local, start + TILE - pos)
                    cursor = offset
                    for i, visit in enumerate(VISITS):
                        for plane, stride in enumerate(STRIDES):
                            # The primary offset formula does not depend on the walk used by the reread.
                            at = offset + rows * (i * VISIT_ROW + sum(STRIDES[:plane])) + local * stride
                            raw = read_at(fd, take * stride, at)
                            # Separate mmap read, advancing through the planes in writer order.
                            reread = mm[cursor + local * stride:cursor + (local + take) * stride]
                            require(raw == reread, f"reread mismatch at token {pos}, visit {visit}, plane {plane}")
                            samples[visit][plane].append(raw)
                            cursor += rows * stride
                    pos += take
        require(before == identity(fd), "snapshot changed while reading")
        digest = {}
        for visit, planes in samples.items():
            raw = [b"".join(chunks) for chunks in planes]
            digest[visit] = [hashlib.sha256(p).hexdigest() for p in raw]
            samples[visit] = [np.frombuffer(p, dtype=np.uint8).reshape(count, TILE, HEADS, BLOCKS, sample_width)
                              for p, sample_width in zip(raw, (WIDTH, 2, WIDTH, 2))]
        meta.update(prefix=limit, tile_rows=TILE, tile_starts=starts, sample_rows=count * TILE,
                    sampled_bytes=count * TILE * VISIT_ROW * len(VISITS), reread_equal=True,
                    samples_sha256_K_codes_K_scale_V_codes_V_scale=digest)
        return meta, samples
    finally:
        os.close(fd)
