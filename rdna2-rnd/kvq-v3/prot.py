#!/usr/bin/env python3
"""Mixed precision by row importance: keep the most-attended cold rows (per KV group) native Q8.
Importance = max attention probability over calibration queries (pos <= cal); error measured on later queries."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvq
import screen

cap = kvq.Capture(sys.argv[1])
upto, cal, ev0 = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
visits = [int(v) for v in sys.argv[5].split(',')]
codecs = sys.argv[6].split(',')
fracs = (0, 0.002, 0.005, 0.01, 0.02, 0.05)
table = screen.codec_table()
for v in visits:
    K, V = cap.kv(v, upto)
    K, V = np.asarray(K, np.float32), np.asarray(V, np.float32)
    Kq, Vq = kvq.q8_roundtrip(K), kvq.q8_roundtrip(V)
    Q = cap.queries(v)
    T = cap.table(v).astype(np.float32)
    allq = np.where((Q['pos'] % 256 == 0) & (Q['pos'] >= 4096) & (Q['pos'] < upto))[0]
    pos = Q['pos'][allq]
    n = len(V)
    seal = np.maximum((pos - 2048) // 128 * 128, 0)
    gate = 1 / (1 + np.exp(-Q['gate'][allq].astype(np.float64)))
    P = []
    ref = np.zeros((len(allq), 24, 128))
    for g in range(8):
        q = Q['q'][allq][:, 3 * g:3 * g + 3].reshape(-1, 128)
        rel = Q['rel'][allq][:, 3 * g:3 * g + 3].reshape(-1, 16)
        S = (q @ Kq[:, g].T) / np.float32(np.sqrt(128))
        pr = np.repeat(pos, 3)
        for r in range(len(pr)):
            lo = max(0, pr[r] - 1023)
            S[r, lo:pr[r] + 1] += T[pr[r] - np.arange(lo, pr[r] + 1)] @ rel[r]
            S[r, pr[r] + 1:] = -np.inf
        sk = Q['sink'][allq][:, 3 * g:3 * g + 3].reshape(-1)
        m = np.maximum(S.max(1), sk)
        E = np.exp(S - m[:, None])
        Pg = (E / (E.sum(1) + np.exp(sk - m))[:, None]).astype(np.float32)
        ref[:, 3 * g:3 * g + 3] = (Pg @ Vq[:, g]).reshape(-1, 3, 128)
        Pg[np.arange(n)[None, :] >= np.repeat(seal, 3)[:, None]] = 0
        P.append(Pg)
        del S, E
    calq = np.repeat(pos <= cal, 3)
    evq = pos > ev0
    imp = [P[g][calq].max(0) for g in range(8)]          # [n] per group, only rows seen by calibration queries
    gref = gate[evq] * ref[evq]
    for name in codecs:
        fn, mean = table[name]
        Xh, nb = screen.with_mean(fn, V[:int(seal.max())], mean)
        dV = np.asarray(Xh, np.float32) - Vq[:len(Xh)]
        for frac in fracs:
            do = np.zeros((evq.sum(), 24, 128))
            kept = 0
            for g in range(8):
                d = dV[:, g].copy()
                if frac:
                    cand = np.arange(min(cal, len(d)))
                    k = int(frac * len(d))
                    top = cand[np.argsort(imp[g][cand])[::-1][:k]]
                    d[top] = 0
                    kept += k
                Pe = P[g].reshape(len(pos), 3, -1)[evq].reshape(-1, P[g].shape[1])[:, :len(d)]
                do[:, 3 * g:3 * g + 3] = (Pe @ d).reshape(-1, 3, 128)
            relo = np.linalg.norm(gate[evq] * do, axis=2) / np.linalg.norm(gref, axis=2)
            extra = frac * (136 - nb + 4)   # bytes per row: native rows + a 4-byte index entry
            print(f"v{v} {name:12s} protect {frac:5.3f} (+{extra:.2f} B/row) att p50 {np.median(relo):.5f} p95 "
                  f"{np.percentile(relo, 95):.5f} max {relo.max():.4f}", flush=True)
        # page 0 only (system prompt / sink-like rows)
        do = np.zeros((evq.sum(), 24, 128))
        for g in range(8):
            d = dV[:, g].copy()
            d[:128] = 0
            Pe = P[g].reshape(len(pos), 3, -1)[evq].reshape(-1, P[g].shape[1])[:, :len(d)]
            do[:, 3 * g:3 * g + 3] = (Pe @ d).reshape(-1, 3, 128)
        relo = np.linalg.norm(gate[evq] * do, axis=2) / np.linalg.norm(gref, axis=2)
        print(f"v{v} {name:12s} page0 native            att p50 {np.median(relo):.5f} p95 {np.percentile(relo, 95):.5f}",
              flush=True)
