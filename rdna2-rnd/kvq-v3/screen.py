#!/usr/bin/env python3
"""V-only codec screen on captured Inkling global KV: reconstruction and gated attention-output error.

Production reference: K native Q8, V native Q8 for every row (dense). Candidate: rows below the page-aligned seal
(pos - 2048 rounded down to 128) use codec(V); the newest >= 2048 rows stay native Q8. Weights are unchanged
(V-only), so delta_out = gate * sum_{cold i} p_i (V_codec_i - V_q8_i).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvq
import kvcodec as C

HOT = 2048


def with_mean(fn, X, mode):
    """X [n,8,128]. Subtract per-(page,head) or calibration mean, code the residual, add it back."""
    if mode is None:
        out, nb = fn(X.reshape(-1, 128))
        return out.reshape(X.shape), nb
    if mode == 'page':
        n = len(X) // 128 * 128
        P = X[:n].reshape(-1, 128, 8, 128)
        mu = C.fp16(P.mean(axis=1, keepdims=True))
        R = (P - mu).reshape(-1, 128)
        out, nb = fn(R)
        out = out.reshape(P.shape) + mu
        res = np.concatenate([out.reshape(-1, 8, 128), X[n:]], axis=0)
        return res, nb + 2
    raise ValueError(mode)


def codec_table():
    t = {}
    t['q8_0'] = (C.q8_0, None)
    _rj = np.random.default_rng(12345)
    t['jitter2e-3'] = (lambda x: (x + _rj.normal(0, 2e-3, x.shape).astype(np.float32), 136), None)
    t['jitter2e-2'] = (lambda x: (x + _rj.normal(0, 2e-2, x.shape).astype(np.float32), 136), None)
    t['v4refit'] = (C.v4_refit, None)
    t['q4_0'] = (C.q4_0, None)
    t['q4_0+h64'] = (lambda x: C.q4_0(x, 'h64'), None)
    t['iq4nl'] = (C.iq4_nl, None)
    t['iq4nl+h128s'] = (lambda x: C.iq4_nl(x, 'h128s'), None)
    for b in (3, 4, 5, 6):
        t[f'exl3-lin{b}'] = (lambda x, b=b: C.exl3(x, b), None)
    t['exl3-cub4'] = (lambda x: C.exl3(x, 4, 0.65), None)
    for b in (4, 5, 6):
        t[f'turbo{b}'] = (lambda x, b=b: C.turbo(x, b), None)
    for b in (3, 4):
        t[f'octave{b}'] = (lambda x, b=b: C.octave(x, b), None)
    for b in (3, 4, 5, 6):
        # known-norm V: rotated coordinates have exact mean square 1 -> no stored scale
        t[f'lm{b}-fixed'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', norm='fixed', fit='none'), None)
        t[f'lm{b}-ncfree'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', norm='fixed', fit='nc_free'), None)
        t[f'lm{b}-ls1'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', norm='fixed', fit='ls', scale_bytes=1,
                                                   iters=1), None)
        t[f'lm{b}-oct'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', norm='rms', fit='oct'), None)
        t[f'lm{b}-ls'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', norm='rms', fit='ls', iters=2), None)
        t[f'lm{b}-g32ls'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', group=32, norm='rms', fit='ls',
                                                     iters=2), None)
        t[f'lm{b}-pmean-ls'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', norm='rms', fit='ls', iters=2),
                                'page')
    UST = {4: 0.3286, 5: 0.1834, 6: 0.1008}
    for b in (4, 5, 6):
        bk = (np.arange(1 << b) - ((1 << b) - 1) / 2) * UST[b]
        t[f'u{b}-fixed'] = (lambda x, b=b, bk=bk: C.scalar(x, b=b, rot_kind='h128s', norm='fixed', fit='none',
                                                          book_arr=bk), None)
        t[f'u{b}-pmean'] = (lambda x, b=b, bk=bk: C.scalar(x, b=b, rot_kind='h128s', norm='rms', fit='ls', iters=1,
                                                          book_arr=bk / UST[b] * UST[b]), 'page')
    for b in (4, 5, 6):
        t[f'k-lm{b}-ls'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', norm='rms', fit='ls', iters=2), None)
        t[f'k-lm{b}-ls1'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', norm='rms', fit='ls', iters=2,
                                                     scale_bytes=1), None)
        t[f'k-lm{b}-pmean'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='h128s', norm='rms', fit='ls', iters=2), 'page')
        t[f'k-q{b}-g32'] = (lambda x, b=b: C.scalar(x, b=b, rot_kind='none', group=32, book='sym', norm='absmax'), None)
    for b in (4, 5):
        t[f'kvarn{b}'] = (lambda x, b=b: C.kvarn_page(x, b), None)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', required=True)
    ap.add_argument('--upto', type=int, required=True)
    ap.add_argument('--visits', default=','.join(map(str, kvq.VISITS)))
    ap.add_argument('--codecs', default='')
    ap.add_argument('--qevery', type=int, default=2048, help='use captured queries at multiples of this')
    ap.add_argument('--qmin', type=int, default=8192)
    ap.add_argument('--source', default='fp32', choices=('fp32', 'q8'), help='codec input: direct or requantized')
    ap.add_argument('--side', default='v', choices=('v', 'k'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cap = kvq.Capture(a.cap)
    table = codec_table()
    names = [n for n in (a.codecs.split(',') if a.codecs else table)]
    res = dict(args=vars(a), visits={})
    for v in map(int, a.visits.split(',')):
        t0 = time.time()
        K, V = cap.kv(v, a.upto)
        K = np.asarray(K, np.float32)
        V = np.asarray(V, np.float32)
        n = len(V)
        Kq = kvq.q8_roundtrip(K)
        Vq = kvq.q8_roundtrip(V)
        Q = cap.queries(v)
        T = cap.table(v).astype(np.float32)
        sel = np.where((Q['pos'] % a.qevery == 0) & (Q['pos'] >= a.qmin) & (Q['pos'] < n))[0]
        dec = np.where((Q['pos'] >= n - 64) & (Q['pos'] < n))[0]
        sel = np.unique(np.concatenate([sel, dec]))
        pos = Q['pos'][sel]
        seal = np.maximum((pos - HOT) // 128 * 128, 0)
        nq = len(sel)
        # probabilities per group, cold-masked; and the reference outputs (all-Q8) per head
        Pc = []
        ref = np.zeros((nq, 24, 128), np.float64)
        ref32 = np.zeros((nq, 24, 128), np.float64)
        gate = 1 / (1 + np.exp(-Q['gate'][sel].astype(np.float64)))
        sink = Q['sink'][sel].astype(np.float64)
        idx = np.arange(n)
        for g in range(8):
            q = Q['q'][sel][:, 3 * g:3 * g + 3].reshape(nq * 3, 128).astype(np.float32)
            S = (q @ Kq[:, g].T) / np.float32(np.sqrt(128))                  # [nq*3, n]
            rel = Q['rel'][sel][:, 3 * g:3 * g + 3].reshape(nq * 3, 16).astype(np.float32)
            pr = np.repeat(pos, 3)
            for r in range(nq * 3):
                lo = max(0, pr[r] - 1023)
                S[r, lo:pr[r] + 1] += (T[pr[r] - np.arange(lo, pr[r] + 1)] @ rel[r])
                S[r, pr[r] + 1:] = -np.inf
            sk = sink[:, 3 * g:3 * g + 3].reshape(-1)
            m = np.maximum(S.max(axis=1), sk)
            E = np.exp(S - m[:, None])
            l = E.sum(axis=1, dtype=np.float64) + np.exp(sk - m)
            P = (E / l[:, None]).astype(np.float32)
            ref[:, 3 * g:3 * g + 3] = (P @ Vq[:, g]).reshape(nq, 3, 128)
            ref32[:, 3 * g:3 * g + 3] = (P @ V[:, g]).reshape(nq, 3, 128)
            cold = idx[None, :] < np.repeat(seal, 3)[:, None]
            Pc.append(np.where(cold, P, 0).astype(np.float32))
            del S, E, P, cold
        gref = gate * ref
        cold_mass = np.array([[Pc[g][r].sum() for r in range(nq * 3)] for g in range(8)])   # [8, nq*3]
        vres = dict(n=n, queries=pos.tolist(), cold_mass_mean=float(cold_mass.mean()),
                    cold_mass_p05_p95=[float(np.percentile(cold_mass, 5)), float(np.percentile(cold_mass, 95))],
                    q8_vs_fp32=None, codecs={})
        d32 = gate * (ref32 - ref)
        rel32 = np.linalg.norm(d32, axis=2) / np.linalg.norm(gref, axis=2)
        vres['q8_vs_fp32'] = dict(p50=float(np.median(rel32)), p95=float(np.percentile(rel32, 95)), max=float(rel32.max()))
        if a.side == 'k':
            ksrc = K if a.source == 'fp32' else Kq
            cold_rows = int(seal.max())
            qs = Q['q'][sel].astype(np.float32)
            rels = Q['rel'][sel].astype(np.float32)
            for name in names:
                fn, mean = table[name]
                Kh, nb = with_mean(fn, ksrc[:cold_rows], mean)
                Kh = np.concatenate([np.asarray(Kh, np.float32), Kq[cold_rows:]], axis=0)
                out_c = np.zeros((nq, 24, 128))
                kl, top1 = [], []
                for g in range(8):
                    q = qs[:, 3 * g:3 * g + 3].reshape(nq * 3, 128)
                    rel = rels[:, 3 * g:3 * g + 3].reshape(nq * 3, 16)
                    pr = np.repeat(pos, 3)
                    Sq = (q @ Kq[:, g].T) / np.float32(np.sqrt(128))
                    Sh = (q @ Kh[:, g].T) / np.float32(np.sqrt(128))
                    cold = idx[None, :] < np.repeat(seal, 3)[:, None]
                    S = np.where(cold, Sh, Sq)
                    for r in range(nq * 3):
                        lo = max(0, pr[r] - 1023)
                        bias = T[pr[r] - np.arange(lo, pr[r] + 1)] @ rel[r]
                        S[r, lo:pr[r] + 1] += bias
                        Sq[r, lo:pr[r] + 1] += bias
                        S[r, pr[r] + 1:] = -np.inf
                        Sq[r, pr[r] + 1:] = -np.inf
                    sk = sink[:, 3 * g:3 * g + 3].reshape(-1)
                    for X, keep in ((S, 'c'), (Sq, 'q')):
                        m = np.maximum(X.max(axis=1), sk)
                        E = np.exp(X - m[:, None])
                        l = E.sum(axis=1, dtype=np.float64) + np.exp(sk - m)
                        if keep == 'c':
                            Pc_ = E / l[:, None]
                        else:
                            Pq_ = E / l[:, None]
                    out_c[:, 3 * g:3 * g + 3] = (Pc_.astype(np.float32) @ Vq[:, g]).reshape(nq, 3, 128)
                    lp = np.log(np.maximum(Pq_, 1e-30)); lc = np.log(np.maximum(Pc_, 1e-30))
                    kl.extend(np.sum(Pq_ * (lp - lc), axis=1).tolist())
                    top1.extend((Pq_.argmax(axis=1) == Pc_.argmax(axis=1)).tolist())
                    del Sq, Sh, S, Pc_, Pq_
                relo = np.linalg.norm(gate * (out_c - ref), axis=2) / np.linalg.norm(gref, axis=2)
                err = Kh[:cold_rows] - K[:cold_rows]
                vres['codecs'][name] = dict(bytes=int(nb), bpv=nb * 8 / 128,
                    rec_rel=float(np.sqrt(np.sum(err.astype(np.float64) ** 2) / np.sum(K[:cold_rows].astype(np.float64) ** 2))),
                    att_p50=float(np.median(relo)), att_p95=float(np.percentile(relo, 95)), att_max=float(relo.max()),
                    att_kl_mean=float(np.mean(kl)), att_top1=float(np.mean(top1)), rec_rel_p99=0.0, norm_ratio_mean=1.0)
                c = vres['codecs'][name]
                print(f"v{v} K {name:16s} {nb:4d}B rec {c['rec_rel']:.4f} att p50 {c['att_p50']:.5f} p95 {c['att_p95']:.5f} "
                      f"max {c['att_max']:.4f} attKL {c['att_kl_mean']:.5f} top1 {c['att_top1']:.4f}", flush=True)
            res['visits'][v] = vres
            Path(a.out).write_text(json.dumps(res, indent=1))
            continue
        src = V if a.source == 'fp32' else Vq
        cold_rows = int(seal.max())
        for name in names:
            fn, mean = table[name]
            t1 = time.time()
            Xh, nb = with_mean(fn, src[:cold_rows], mean)
            Xh = Xh.astype(np.float32)
            X0 = V[:cold_rows]
            err = Xh - X0
            rel_row = np.linalg.norm(err, axis=2) / np.linalg.norm(X0, axis=2)
            ratio = np.linalg.norm(Xh, axis=2) / np.linalg.norm(X0, axis=2)
            dV = (Xh - Vq[:cold_rows]).astype(np.float32)
            do = np.zeros((nq, 24, 128))
            for g in range(8):
                do[:, 3 * g:3 * g + 3] = (Pc[g][:, :cold_rows] @ dV[:, g]).reshape(nq, 3, 128)
            relo = np.linalg.norm(gate * do, axis=2) / np.linalg.norm(gref, axis=2)
            vres['codecs'][name] = dict(
                bytes=int(nb), bpv=nb * 8 / 128,
                rec_rel=float(np.sqrt(np.sum(err.astype(np.float64) ** 2) / np.sum(X0.astype(np.float64) ** 2))),
                rec_rel_p99=float(np.percentile(rel_row, 99)), norm_ratio_mean=float(ratio.mean()),
                norm_ratio_p01_p99=[float(np.percentile(ratio, 1)), float(np.percentile(ratio, 99))],
                att_p50=float(np.median(relo)), att_p95=float(np.percentile(relo, 95)), att_max=float(relo.max()),
                att_by_query_p95=[float(np.percentile(relo[i], 95)) for i in range(nq)], secs=time.time() - t1)
            c = vres['codecs'][name]
            print(f"v{v} {name:16s} {nb:4d}B rec {c['rec_rel']:.4f} p99 {c['rec_rel_p99']:.4f} norm {c['norm_ratio_mean']:.4f} "
                  f"att p50 {c['att_p50']:.5f} p95 {c['att_p95']:.5f} max {c['att_max']:.4f}", flush=True)
        res['visits'][v] = vres
        print(f"visit {v}: n={n} nq={nq} cold mass {vres['cold_mass_mean']:.3f} q8-vs-fp32 att p95 {vres['q8_vs_fp32']['p95']:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        del Pc, K, V, Kq, Vq


if __name__ == '__main__':
    main()
