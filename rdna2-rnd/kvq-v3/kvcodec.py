"""KVQ V3 CPU reference codecs for one 128-wide Inkling KV head row. x: float32 [N,128] -> x_hat float32 [N,128].

Every codec reports exact stored bytes per row (codes + scales/metadata), excluding page tables. Rotations are
orthonormal; decode applies the inverse. Codebooks are normalized; the scale rule maps them to the data.
"""
import math

import numpy as np

D = 128

# --------------------------------------------------------------------------------------------- rotations
TQ_S1 = (0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9)   # TurboQuant+ s1 (bit=1 means -1)
TQ_S2 = (0x16ACEE90, 0x3F628FDC, 0xB7A5357A, 0xBEA56562)


def _signs(words):
    s = np.ones(D, np.float32)
    for i in range(D):
        if (words[i >> 5] >> (i & 31)) & 1:
            s[i] = -1
    return s


S1, S2 = _signs(TQ_S1), _signs(TQ_S2)
_rng = np.random.default_rng(0x5E17)
RS = np.where(_rng.random(D) < 0.5, -1.0, 1.0).astype(np.float32)


def fwht(x, width):
    """Orthonormal Sylvester WHT over consecutive blocks of `width` on the last axis."""
    y = np.array(x, dtype=np.float32, copy=True)
    sh = y.shape
    y = y.reshape(-1, width)
    h = 1
    while h < width:
        y = y.reshape(-1, width // (2 * h), 2, h)
        a = y[:, :, 0, :].copy()
        b = y[:, :, 1, :]
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h *= 2
    return (y.reshape(sh) * np.float32(1 / math.sqrt(width))).astype(np.float32)


def rot(x, kind):
    if kind == 'none':
        return x
    if kind == 'h128s':     # TurboQuant+: D(s2) H D(s1)
        return fwht(x * S1, 128) * S2
    if kind == 'h128r':
        return fwht(x * RS, 128)
    w = int(kind[1:].rstrip('sr'))
    if kind.endswith('r'):
        return fwht(x * RS, w)
    return fwht(x, w)


def unrot(y, kind):
    if kind == 'none':
        return y
    if kind == 'h128s':
        return fwht(y * S2, 128) * S1
    if kind == 'h128r':
        return fwht(y, 128) * RS
    w = int(kind[1:].rstrip('sr'))
    if kind.endswith('r'):
        return fwht(y, w) * RS
    return fwht(y, w)


# --------------------------------------------------------------------------------------------- codebooks
def uniform_mid(b):          # ExLlama linear midpoint grid on [-1,1]
    n = 1 << b
    return ((2 * np.arange(n) + 1) / n - 1).astype(np.float64)


def uniform_sym(b):          # symmetric integers +-(2^(b-1)-1), zero included (current V4 refit at b=4)
    m = (1 << (b - 1)) - 1
    return (np.arange(-m, m + 1) / m).astype(np.float64)


def cubic(b, a=0.65):
    t = uniform_mid(b)
    return t * (a + (1 - a) * t * t)


def lloyd(samples, b, iters=200, init=None):
    """1-D Lloyd-Max (k-means) on samples; returns sorted centroids."""
    x = np.sort(np.asarray(samples, np.float64).ravel())
    n = 1 << b
    c = np.quantile(x, (np.arange(n) + 0.5) / n) if init is None else np.array(init, np.float64)
    for _ in range(iters):
        edges = (c[1:] + c[:-1]) / 2
        idx = np.searchsorted(edges, x)
        sums = np.bincount(idx, weights=x, minlength=n)
        cnt = np.bincount(idx, minlength=n)
        new = np.where(cnt > 0, sums / np.maximum(cnt, 1), c)
        if np.max(np.abs(new - c)) < 1e-9:
            c = new
            break
        c = new
    return np.sort(c)


_GAUSS = {}


def lloyd_gauss(b):
    """Lloyd-Max for N(0,1), symmetric, via a dense deterministic sample."""
    if b not in _GAUSS:
        from math import erf  # noqa
        u = (np.arange(1, 400001) - 0.5) / 400000
        xs = np.sqrt(2) * _erfinv(2 * u - 1)
        c = lloyd(xs, b, iters=500)
        c = (c - c[::-1]) / 2          # enforce symmetry
        _GAUSS[b] = c
    return _GAUSS[b]


def _erfinv(y):
    # Giles' approximation refined by two Newton steps
    w = -np.log((1 - y) * (1 + y))
    x = np.where(w < 5, _p1(w - 2.5), _p2(np.sqrt(w) - 3)) * y
    for _ in range(2):
        x = x - (np.vectorize(math.erf)(x) - y) / (2 / math.sqrt(math.pi) * np.exp(-x * x))
    return x


def _p1(w):
    p = 2.81022636e-08
    for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503, -0.00417768164,
              0.246640727, 1.50140941):
        p = c + p * w
    return p


def _p2(w):
    p = -0.000200214257
    for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613, 0.00943887047,
              1.00167406, 2.83297682):
        p = c + p * w
    return p


IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float64) / 127
OCT3 = np.array([-127, -79, -45, -14, 14, 45, 79, 127], np.float64) / 59      # N(0,1) units
OCT4 = np.array([-126, -95, -74, -58, -43, -30, -18, -6, 6, 18, 30, 43, 58, 74, 95, 126], np.float64) / 46
_TQ_POS = {
    4: [0.011349, 0.034299, 0.058050, 0.083292, 0.111036, 0.143016, 0.182877, 0.241529],
    5: [0.005827, 0.017515, 0.029304, 0.041269, 0.053491, 0.066064, 0.079097, 0.092730,
        0.107141, 0.122569, 0.139353, 0.158003, 0.179348, 0.204892, 0.237892, 0.288236],
    6: [0.002953, 0.008863, 0.014787, 0.020733, 0.026710, 0.032729, 0.038798, 0.044929,
        0.051133, 0.057422, 0.063809, 0.070310, 0.076940, 0.083717, 0.090663, 0.097801,
        0.105157, 0.112765, 0.120663, 0.128895, 0.137517, 0.146600, 0.156230, 0.166522,
        0.177626, 0.189753, 0.203203, 0.218435, 0.236198, 0.257865, 0.286417, 0.330935],
}


def turbo_table(b):          # in N(0,1) units (donor tables are N(0,1/128) for unit vectors)
    p = np.array(_TQ_POS[b], np.float64) * math.sqrt(128)
    return np.concatenate([-p[::-1], p])


def nearest(y, c):
    """Indices of nearest centroid (sorted c) for y."""
    edges = (c[1:] + c[:-1]) / 2
    return np.searchsorted(edges, y)


def fp16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


# --------------------------------------------------------------------------------------------- generic scalar codec
def scalar(x, *, b, rot_kind='none', group=128, book='gauss', norm='rms', fit='none', scale_bytes=2, iters=0,
           book_arr=None):
    """Rotate, then per `group` coordinates: s0 from `norm` ('absmax'|'rms'|'fixed'), codes = nearest(y/s0, book),
    then stored scale from `fit`: 'none' (s0), 'ls' <y,z>/<z,z>, 'nc' ||y||/||z||, 'oct' <y,y>/<y,z>,
    'nc_free' (sqrt(group)*rms_known/||z||, nothing stored; needs known group energy: only valid with norm='fixed').
    iters>0 alternates LS scale / re-encode. scale_bytes 2 = fp16, 1 = 8-bit log2 step 1/16 around 1, 0 = none."""
    c = book_arr if book_arr is not None else {
        'gauss': lambda: lloyd_gauss(b), 'mid': lambda: uniform_mid(b), 'sym': lambda: uniform_sym(b),
        'cubic': lambda: cubic(b), 'iq4nl': lambda: IQ4NL, 'oct': lambda: {3: OCT3, 4: OCT4}[b],
        'turbo': lambda: turbo_table(b)}[book]()
    c = np.asarray(c, np.float64)
    y = rot(x.astype(np.float32), rot_kind).astype(np.float64)
    N = y.shape[0]
    g = y.reshape(N, D // group, group)
    if norm == 'absmax':
        s0 = np.max(np.abs(g), axis=2, keepdims=True) / np.max(np.abs(c))
    elif norm == 'rms':
        s0 = np.sqrt(np.mean(g * g, axis=2, keepdims=True))
    elif norm == 'fixed':
        s0 = np.ones((N, D // group, 1))
    else:
        raise ValueError(norm)
    s0 = np.where(s0 > 0, s0, 1e-12)
    if scale_bytes == 2 and fit == 'none' and norm != 'fixed':
        s0 = fp16(s0).astype(np.float64)
    idx = nearest(g / s0, c)
    s = s0
    for it in range(max(1, iters + 1)):
        z = c[idx]
        if fit == 'none':
            s = s0
        elif fit == 'ls':
            s = np.sum(g * z, axis=2, keepdims=True) / np.maximum(np.sum(z * z, axis=2, keepdims=True), 1e-30)
        elif fit == 'nc':
            s = np.sqrt(np.sum(g * g, axis=2, keepdims=True) / np.maximum(np.sum(z * z, axis=2, keepdims=True), 1e-30))
        elif fit == 'oct':
            s = np.sum(g * g, axis=2, keepdims=True) / np.maximum(np.sum(g * z, axis=2, keepdims=True), 1e-30)
        elif fit == 'nc_free':
            s = np.sqrt(group / np.maximum(np.sum(z * z, axis=2, keepdims=True), 1e-30))
        else:
            raise ValueError(fit)
        if it < iters:
            idx = nearest(g / np.maximum(s, 1e-12), c)
    if fit != 'nc_free' and not (fit == 'none' and norm == 'fixed'):
        if scale_bytes == 2:
            s = fp16(s).astype(np.float64)
        elif scale_bytes == 1:
            # 8-bit log scale: s = 2^((q-128)/32), covers [2^-4, 2^4)
            q = np.clip(np.rint(np.log2(np.maximum(s, 1e-30)) * 32) + 128, 0, 255)
            s = 2.0 ** ((q - 128) / 32)
    xh = (c[idx] * s).reshape(N, D)
    out = unrot(xh.astype(np.float32), rot_kind)
    stored_scales = 0 if (fit == 'nc_free' or (norm == 'fixed' and fit == 'none')) else (D // group) * scale_bytes
    return out, b * D // 8 + stored_scales


# --------------------------------------------------------------------------------------------- named baselines
def q8_0(x):
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.max(np.abs(b), axis=1, keepdims=True) / np.float32(127)
    inv = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d > 0)
    q = np.clip(np.rint(b * inv), -127, 127)
    return (q * fp16(d)).reshape(x.shape), 136


def v4_refit(x):
    """Current PolyStrata V4: native Q8_0 codes refit to symmetric nibbles (+-7) with fp16 scale*amax/7."""
    blk = x.reshape(-1, 32).astype(np.float32)
    d = np.max(np.abs(blk), axis=1, keepdims=True) / np.float32(127)
    inv = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d > 0)
    q8 = np.clip(np.rint(blk * inv), -127, 127).astype(np.float64)
    amax = np.max(np.abs(q8), axis=1, keepdims=True)
    q4 = np.clip(np.sign(q8) * np.rint(np.abs(q8) * 7 / np.maximum(amax, 1)), -8, 7)
    s = fp16(fp16(d) * (amax / 7).astype(np.float32))
    return (q4 * s).reshape(x.shape).astype(np.float32), 72


def q4_0(x, rot_kind='none'):
    y = rot(x, rot_kind).reshape(-1, 32).astype(np.float32)
    j = np.argmax(np.abs(y), axis=1)
    mx = y[np.arange(len(y)), j][:, None]
    d = fp16(mx / -8)
    inv = np.divide(1, d, out=np.zeros_like(d), where=d != 0)
    q = np.clip(np.floor(y * inv + 8.5), 0, 15) - 8
    return unrot((q * d).reshape(x.shape), rot_kind), 72


def iq4_nl(x, rot_kind='none'):
    return scalar(x, b=4, rot_kind=rot_kind, group=32, book='iq4nl', norm='absmax', fit='ls', iters=2)


def iq4_nl_ggml(x, rot_kind='none'):
    """ggml-style IQ4_NL: 16-level table, fp16 scale per 32, best of 15 scale candidates with LS refit."""
    y = rot(x, rot_kind).reshape(-1, 32).astype(np.float64)
    c = IQ4NL
    j = np.argmax(np.abs(y), axis=1)
    mx = y[np.arange(len(y)), j][:, None]
    best_err = np.full((len(y), 1), np.inf)
    best = np.zeros_like(y)
    for k in range(-7, 8):
        d0 = mx / (c[0] * (1 + 0.02 * k))
        d0 = np.where(np.abs(d0) > 0, d0, 1e-12)
        z = c[nearest(y / d0, c)]
        d = fp16(np.sum(y * z, axis=1, keepdims=True) / np.maximum(np.sum(z * z, axis=1, keepdims=True), 1e-30))
        r = z * d
        err = np.sum((r - y) ** 2, axis=1, keepdims=True)
        better = err < best_err
        best = np.where(better, r, best)
        best_err = np.where(better, err, best_err)
    return unrot(best.reshape(x.shape).astype(np.float32), rot_kind), 72


def turbo(x, b):
    """TurboQuant+ turboN: WHT128 with fixed signs, unit-norm N(0,1/128) Lloyd-Max, fp16 corrected L2 scale."""
    return scalar(x, b=b, rot_kind='h128s', group=128, book='turbo', norm='rms', fit='nc', scale_bytes=2)


def exl3(x, b, cubic_a=0.0):
    """ExLlamaV3: plain H32, absmax per 32 (fp16), linear midpoint grid or cubic companding."""
    return scalar(x, b=b, rot_kind='h32', group=32, book='cubic' if cubic_a else 'mid', norm='absmax', fit='none',
                  scale_bytes=2, book_arr=cubic(b, cubic_a) if cubic_a else None)


def octave(x, b):
    """Octave: signed Hadamard over the whole head, RMS-standardized int8 Lloyd-Max LUT, fp16 dot-preserving scale."""
    return scalar(x, b=b, rot_kind='h128r', group=128, book='oct', norm='rms', fit='oct', scale_bytes=2)


def kvarn_page(x, b, iters=8, page=128):
    """KVarN-style V tile: per-row H128, Sinkhorn std balancing over each [page rows, 128] tile, affine RTN per
    row at b bits. Metadata: fp16 column scale per tile column (amortized 2 B/row at page 128) + fp16 row scale and
    offset (4 B/row)."""
    y = fwht(x.astype(np.float32), 128).astype(np.float64)
    out = np.empty_like(y)
    for p0 in range(0, len(y), page):
        T = y[p0:p0 + page]
        a = np.zeros((len(T), 1))
        bb = np.zeros((1, D))
        Z = T.copy()
        best, best_score = (a.copy(), bb.copy()), np.inf
        for _ in range(iters):
            bb = np.clip(bb + np.log(np.clip(Z.std(axis=0, keepdims=True), 1e-3, 1e3)), -0.3, 10)
            Z = T * np.exp(-a - bb)
            a = a + np.log(np.clip(Z.std(axis=1, keepdims=True), 1e-3, 1e3))
            Z = T * np.exp(-a - bb)
            cs, rs = Z.std(axis=0), Z.std(axis=1)
            score = cs.max() / max(cs.min(), 1e-8) + rs.max() / max(rs.min(), 1e-8)
            if score <= best_score:
                best, best_score = (a.copy(), bb.copy()), score
        a, bb = best
        bb = fp16(np.exp(bb)).astype(np.float64)
        Z = T / bb
        lo, hi = Z.min(axis=1, keepdims=True), Z.max(axis=1, keepdims=True)
        n = (1 << b) - 1
        s = fp16(np.maximum((hi - lo) / n, 1e-10)).astype(np.float64)
        lo = fp16(lo).astype(np.float64)
        q = np.clip(np.rint((Z - lo) / s), 0, n)
        out[p0:p0 + page] = (q * s + lo) * bb
    return fwht(out.astype(np.float32), 128), b * 16 + 6


# --------------------------------------------------------------------------------------------- fast C path
import ctypes as _ct
import os as _os
_lib = _ct.CDLL(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'libfastq.so'))
_lib.scalar_codec.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int64, _ct.c_int, _ct.c_int, _ct.c_void_p, _ct.c_int,
                              _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p]
_ROT = {'none': 0, 'h32': 1, 'h64': 2, 'h128': 3, 'h128s': 4, 'h128r': 5}
_NORM = {'absmax': 0, 'rms': 1, 'fixed': 2}
_FIT = {'none': 0, 'ls': 1, 'nc': 2, 'oct': 3, 'nc_free': 4}
_BOOKS = {'gauss': lambda b: lloyd_gauss(b), 'mid': uniform_mid, 'sym': uniform_sym, 'cubic': lambda b: cubic(b),
          'iq4nl': lambda b: IQ4NL, 'oct': lambda b: {3: OCT3, 4: OCT4}[b], 'turbo': turbo_table}
_py_scalar = scalar


def scalar(x, *, b, rot_kind='none', group=128, book='gauss', norm='rms', fit='none', scale_bytes=2, iters=0,
           book_arr=None, codes=False):
    c = np.ascontiguousarray(book_arr if book_arr is not None else _BOOKS[book](b), np.float32)
    xx = np.ascontiguousarray(x, np.float32).reshape(-1, 128)
    out = np.empty_like(xx)
    cod = np.empty(xx.shape, np.uint8) if codes else None
    s1 = np.ascontiguousarray(RS if rot_kind == 'h128r' else S1, np.float32)
    s2 = np.ascontiguousarray(S2, np.float32)
    _lib.scalar_codec(xx.ctypes.data, out.ctypes.data, len(xx), _ROT[rot_kind], group, c.ctypes.data, len(c),
                      _NORM[norm], _FIT[fit], scale_bytes, iters, s1.ctypes.data, s2.ctypes.data,
                      cod.ctypes.data if codes else None)
    stored = 0 if (fit == 'nc_free' or (norm == 'fixed' and fit == 'none')) else (128 // group) * scale_bytes
    nb = b * 128 // 8 + stored
    return (out.reshape(x.shape), nb, cod) if codes else (out.reshape(x.shape), nb)
