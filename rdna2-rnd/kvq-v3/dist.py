#!/usr/bin/env python3
"""Distribution study of captured Inkling global K/V (FP32, post conv + head norm)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvq
import kvcodec as C

cap = kvq.Capture(sys.argv[1])
upto = int(sys.argv[2])
out = Path(sys.argv[3])
rng = np.random.default_rng(0)
CLIP = 22.627417


def kurt(x, axis=None):
    m = x.mean(axis=axis, keepdims=True)
    v = ((x - m) ** 2).mean(axis=axis)
    return ((x - m) ** 4).mean(axis=axis) / v ** 2


res = {}
Vs = {}
for v in kvq.VISITS:
    K, V = cap.kv(v, upto)
    K = np.asarray(K, np.float32)
    V = np.asarray(V, np.float32)
    Vs[v] = V
    r = {}
    for name, X in (('K', K), ('V', V)):
        n = np.linalg.norm(X, axis=2)                       # [rows, 8]
        flat = X.reshape(-1, 128)
        sub = flat[rng.choice(len(flat), min(len(flat), 200000), replace=False)]
        rot = C.rot(sub, 'h128s')
        h32 = C.rot(sub, 'h32')
        rms = np.sqrt((sub ** 2).mean(axis=1))
        crest_row = np.abs(sub).max(axis=1) / rms
        b = sub.reshape(-1, 4, 32)
        crest32 = (np.abs(b).max(axis=2) / np.sqrt((b ** 2).mean(axis=2) + 1e-30)).ravel()
        rb = rot.reshape(-1, 4, 32)
        crest32r = (np.abs(rb).max(axis=2) / np.sqrt((rb ** 2).mean(axis=2) + 1e-30)).ravel()
        # channel structure per head: mean and std of each channel across rows
        ch_mean = X.mean(axis=0)            # [8,128]
        ch_std = X.std(axis=0)
        ch_rms = np.sqrt((X ** 2).mean(axis=0))
        r[name] = dict(
            norm_p=[float(np.percentile(n, q)) for q in (0, 1, 50, 99, 100)],
            at_clip=float(np.mean(n > CLIP * 0.9999)) if name == 'K' else None,
            kurt_native=float(kurt(sub)), kurt_h128s=float(kurt(rot)), kurt_h32=float(kurt(h32)),
            crest_row_p50_p99=[float(np.percentile(crest_row, 50)), float(np.percentile(crest_row, 99))],
            crest32_native_p50_p99=[float(np.percentile(crest32, 50)), float(np.percentile(crest32, 99))],
            crest32_h128s_p50_p99=[float(np.percentile(crest32r, 50)), float(np.percentile(crest32r, 99))],
            chan_rms_max_over_median=[float(ch_rms[g].max() / np.median(ch_rms[g])) for g in range(8)],
            chan_absmean_over_rms=float(np.mean(np.abs(ch_mean) / (ch_rms + 1e-30))),
            mean_vector_energy_share=[float(np.sum(ch_mean[g] ** 2) / np.mean(np.sum(X[:, g] ** 2, axis=1)))
                                      for g in range(8)],
            lag1_cos=float(np.mean(np.sum(X[1:] * X[:-1], axis=2) / (n[1:] * n[:-1] + 1e-30))),
            lag64_cos=float(np.mean(np.sum(X[64:] * X[:-64], axis=2) / (n[64:] * n[:-64] + 1e-30))),
        )
        # depth dependence: stats by 16K bucket
        buck = []
        for s0 in range(0, upto, 16384):
            s1 = min(upto, s0 + 16384)
            Xi = X[s0:s1].reshape(-1, 128)
            ri = C.rot(Xi[::7], 'h128s')
            buck.append(dict(start=s0, norm_med=float(np.median(np.linalg.norm(Xi, axis=1))),
                             kurt_native=float(kurt(Xi[::7])), kurt_rot=float(kurt(ri)),
                             at_clip=float(np.mean(np.linalg.norm(Xi, axis=1) > CLIP * 0.9999)) if name == 'K' else None))
        r[name]['by_depth'] = buck
    res[v] = r
    print(v, kvq.PASS[v], 'K at clip %.3f' % r['K']['at_clip'], 'K kurt %.2f->%.2f' % (r['K']['kurt_native'], r['K']['kurt_h128s']),
          'V kurt %.2f->%.2f(h128s) %.2f(h32)' % (r['V']['kurt_native'], r['V']['kurt_h128s'], r['V']['kurt_h32']),
          'V crest32 p99 %.2f->%.2f' % (r['V']['crest32_native_p50_p99'][1], r['V']['crest32_h128s_p50_p99'][1]),
          'V lag1 %.3f lag64 %.3f' % (r['V']['lag1_cos'], r['V']['lag64_cos']),
          'V mean-energy %s' % np.round(r['V']['mean_vector_energy_share'], 3), flush=True)

# cross-pass similarity: same physical layer, same row, same KV head
pairs = [(11, 23), (11, 35), (23, 35), (17, 29), (17, 41), (29, 41), (5, 11), (41, 47)]
cross = {}
for a, b in pairs:
    A, B = Vs[a], Vs[b]
    cos = np.sum(A * B, axis=2) / (np.linalg.norm(A, axis=2) * np.linalg.norm(B, axis=2))
    # linear predictability: per head, least squares B ~ A W on a subsample
    r2 = []
    for g in range(8):
        X = A[::13, g].astype(np.float64)
        Y = B[::13, g].astype(np.float64)
        W, *_ = np.linalg.lstsq(X[: len(X) // 2], Y[: len(Y) // 2], rcond=None)
        E = Y[len(Y) // 2:] - X[len(X) // 2:] @ W
        r2.append(float(1 - np.sum(E ** 2) / np.sum((Y[len(Y) // 2:] - Y[len(Y) // 2:].mean(0)) ** 2)))
    cross[f'{a}-{b}'] = dict(cos_mean=float(cos.mean()), heldout_r2_by_head=r2)
    print('cross', a, b, 'cos %.3f' % cos.mean(), 'R2', np.round(r2, 3), flush=True)
res['cross'] = cross
out.write_text(json.dumps(res, indent=1))
