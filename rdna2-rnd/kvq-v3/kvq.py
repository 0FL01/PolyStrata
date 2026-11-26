"""KVQ V3 lab: capture loaders and the Inkling global-attention reference (numpy, CPU only)."""
import os
from pathlib import Path

import numpy as np

VISITS = tuple(range(5, 48, 6))
PHYS = dict(zip(VISITS, (5, 11, 17, 11, 17, 11, 17, 23)))
PASS = dict(zip(VISITS, ('prelude', 'core1', 'core1', 'core2', 'core2', 'core3', 'core3', 'coda')))
QH, G, D, REL = 24, 8, 128, 16
QREC = 8 + 4 * (QH * D * 3 + QH * REL + QH)   # pos, q, rel, sink, gate, out


class Capture:
    """Memory-mapped capture of one run. kv(visit) -> (K, V) float32 [rows, 8, 128], last write wins per row."""

    def __init__(self, root):
        self.root = Path(root)
        self._kv = {}

    def kv(self, visit, upto=None):
        if visit not in self._kv:
            idx = [tuple(map(int, l.split())) for l in (self.root / f'kv.{visit}.idx').read_text().split('\n') if l]
            raw = np.memmap(self.root / f'kv.{visit}.f32', dtype='<f4', mode='r')
            rows = max(p + n for p, n in idx)
            # map each absolute row to (offset of its K row); later writes override earlier ones
            where = np.full(rows, -1, dtype=np.int64)
            off = 0
            for p, n in idx:
                where[p:p + n] = off + np.arange(n) * 1024
                off += 2 * n * 1024
            # V row is n*1024 after the K row of the same call
            vwhere = np.full(rows, -1, dtype=np.int64)
            off = 0
            for p, n in idx:
                vwhere[p:p + n] = off + n * 1024 + np.arange(n) * 1024
                off += 2 * n * 1024
            assert (where >= 0).all()
            self._kv[visit] = (raw, where, vwhere)
        raw, where, vwhere = self._kv[visit]
        n = len(where) if upto is None else upto
        K = np.stack([raw[w:w + 1024] for w in where[:n]]) if n < 4096 else _gather(raw, where[:n])
        V = np.stack([raw[w:w + 1024] for w in vwhere[:n]]) if n < 4096 else _gather(raw, vwhere[:n])
        return K.reshape(n, G, D), V.reshape(n, G, D)

    def queries(self, visit):
        raw = np.fromfile(self.root / f'q.{visit}.f32', dtype=np.uint8)
        n = len(raw) // QREC
        assert n * QREC == len(raw)
        rec = raw.reshape(n, QREC)
        pos = rec[:, :8].copy().view('<i8')[:, 0]
        f = rec[:, 8:].copy().view('<f4')
        o = 0
        out = {}
        for name, size, shape in (('q', QH * D, (QH, D)), ('rel', QH * REL, (QH, REL)), ('sink', QH, (QH,)),
                                  ('gate', QH * D, (QH, D)), ('out', QH * D, (QH, D))):
            out[name] = f[:, o:o + size].reshape(n, *shape)
            o += size
        out['pos'] = pos
        return out

    def table(self, visit):
        return np.fromfile(self.root / f'table.{visit}.f32', dtype='<f4').reshape(1024, REL)


def _gather(raw, where):
    # contiguous runs -> fast slices
    out = np.empty((len(where), 1024), dtype=np.float32)
    i = 0
    while i < len(where):
        j = i
        while j + 1 < len(where) and where[j + 1] == where[j] + 1024:
            j += 1
        out[i:j + 1] = np.asarray(raw[where[i]:where[j] + 1024]).reshape(-1, 1024)
        i = j + 1
    return out


def q8_roundtrip(x):
    """Native Q8_0 over the last axis in 32-blocks: fp16 scale amax/127, rintf codes clamped to +-127."""
    s = x.shape
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.max(np.abs(b), axis=1, keepdims=True)
    d = (amax / np.float32(127)).astype(np.float32)
    inv = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d > 0)
    q = np.clip(np.rint(b * inv), -127, 127)
    dh = d.astype(np.float16).astype(np.float32)
    return (q * dh).reshape(s).astype(np.float32)


def attend(qrec, i, K, V, table, cold=None, Vcold=None):
    """Reference output of one captured query i over K/V rows [0, pos]. Returns dict with gated out, pre-gate o,
    per-head max m, denominator l (incl. sink), and probabilities p [24, rows]. If cold is given, rows < cold use
    Vcold in place of V (same weights: V-only codec)."""
    pos = int(qrec['pos'][i])
    q = qrec['q'][i].astype(np.float64)       # [24,128]
    rel = qrec['rel'][i].astype(np.float64)   # [24,16]
    sink = qrec['sink'][i].astype(np.float64)
    gate = qrec['gate'][i].astype(np.float64)
    n = pos + 1
    k = K[:n].astype(np.float64)              # [n,8,128]
    kh = np.repeat(k, 3, axis=1)              # [n,24,128] head h uses group h//3
    s = np.einsum('hd,nhd->hn', q, kh) / np.sqrt(D)
    dist = pos - np.arange(n)
    near = dist < 1024
    s[:, near] += rel @ table[dist[near]].T.astype(np.float64)
    m = np.maximum(s.max(axis=1), sink)
    e = np.exp(s - m[:, None])
    l = e.sum(axis=1) + np.exp(sink - m)
    p = e / l[:, None]
    Vu = V[:n].astype(np.float64)
    if cold is not None and cold > 0:
        Vu = Vu.copy()
        c = min(cold, n)
        Vu[:c] = Vcold[:c]
    vh = np.repeat(Vu, 3, axis=1)
    o = np.einsum('hn,nhd->hd', p, vh)
    g = 1 / (1 + np.exp(-gate))
    return dict(out=g * o, o=o, m=m, l=l, p=p, gate=g, pos=pos)
