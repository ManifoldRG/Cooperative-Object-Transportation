"""Monotone tent deconfliction: the mass-weighted tent push (Jowett's
relax_tent_experimental) made monotone with the method-of-chains invariant.

A move for pair (j, l) displaces both whole trajectories by a tent profile
B(k) (peak 1 at the worst step k*, zero at the pinned steps 0, 1, K, and
optionally limited to a half-width W), each in its own 3-D direction:

    x_j[k] += c lam_j s B(k) g_j,   x_l[k] += c lam_l s B(k) g_l,
    lam_j = m_l / (m_j + m_l)   (the lighter payload yields).

Every affected pair's relative displacement at step k is linear in
(g_j, g_l), so the first-order "approach nothing near its target" condition
is a cone in R^6 (one half-space per near-target same-step pair or segment
touched by the tent). (g_j, g_l) starts at (g0, -g0)/sqrt 2, g0 the
separating direction at k*, and is projected into that cone. A single shared
g would intersect both payloads' cones and is empty whenever either sits in
a cluster. The move is then
checked exactly with step halving: no same-step truncated distance
min(d, tau) and no segment truncated distance min(s, delta + seg_h) of any
pair may decrease. If the full tent is blocked, narrower tents are tried;
the one-step tent (W = 1) is the method of chains' single-point move.

Invariant (as for the method of chains): every pair's truncated same-step
and segment distances are non-decreasing over the whole run. Termination:
every same-step deficit < tol and every segment >= delta; on a jam (no
progress over a window of sweeps) the grid is refined, K <- 2K.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

import chain_kernels as ck
from chain_kernels import _seg


@njit(cache=True)
def _profile(K, ks, W):
    """Tent over steps 0..K peaked (=1) at ks, zero at 0, 1, K; W > 0 limits
    the support to |k - ks| < W."""
    B = np.zeros(K + 1)
    for k in range(2, K):
        a = (k - 1) / (ks - 1)
        b = (K - k) / (K - ks)
        w = min(a, b)
        if W > 0:
            w = min(w, max(0.0, 1.0 - abs(k - ks) / W))
        B[k] = max(w, 0.0)
    return B


@njit(cache=True)
def _norm3(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@njit(cache=True)
def _project_cone_nd(u, N, nn):
    """Cyclic projection of unit u (any dimension) onto {v : v . n >= 0 for
    the nn rows n of N} (chain_kernels._project_cone in R^n).
    Returns (ok, unit v, index of the row still violated on failure or -1)."""
    v = u.copy()
    if nn == 0:
        return True, v, -1
    for _ in range(24):
        imin, dmin = 0, np.inf
        for i in range(nn):
            d = 0.0
            for c in range(v.shape[0]):
                d += N[i, c] * v[c]
            if d < dmin:
                dmin, imin = d, i
        if dmin >= -1e-9:
            return True, v, -1
        nrm2 = 0.0
        for c in range(v.shape[0]):
            nrm2 += N[imin, c] * N[imin, c]
        v = v - (dmin / max(nrm2, 1e-18)) * N[imin]
        nv = np.sqrt(np.sum(v * v))
        if nv < 1e-9:
            return False, v, imin
        v = v / nv
    for i in range(nn):
        d = 0.0
        for c in range(v.shape[0]):
            d += N[i, c] * v[c]
        if d < -1e-6:
            return False, v, i
    return True, v, -1


@njit(cache=True)
def _tent_move(P, j, l, ks, g0, s0, W, T, S, lam_j, lam_l, use_seg, info):
    """One exact, monotone tent move for pair (j, l). Each payload moves by
    the tent B(k) in its own direction, x_j += c lam_j s B g_j and
    x_l += c lam_l s B g_l (c = sqrt 2, so the symmetric start u0 = (g0, -g0)
    / sqrt 2 opens the pair by s at k*). (g_j, g_l) is projected into the
    first-order cone in R^6, then checked exactly with halving.

    Returns a code: 1 moved, 0 empty tent support, -1 cone empty, -2 every
    halving rejected by the exact check. info[0:4] = (p, q, k, kind) of the
    blocking constraint (kind 0 same-step, 1 segment): for -1 the row the
    cone projection could not satisfy, for -2 the first failed check at the
    smallest step."""
    M, K = P.shape[0], P.shape[1] - 1
    B = _profile(K, ks, W)
    kmin, kmax = K, 0
    for k in range(K + 1):
        if B[k] > 0.0:
            kmin = min(kmin, k)
            kmax = max(kmax, k)
    if kmax < kmin:
        return 0
    s_lo, s_hi = max(kmin - 1, 0), min(kmax, K - 1)   # segments touching support
    r2 = np.sqrt(2.0)
    Nrm = np.zeros((4 * M * (K + 1) + 4, 6))
    src = np.zeros((4 * M * (K + 1) + 4, 4), np.int64)
    nn = 0
    lams = (lam_j, lam_l)
    movers = (j, l)
    # ---- first-order cone in R^6 -------------------------------------------
    for a in range(2):
        p = movers[a]
        for q in range(M):
            if q == p or (a == 1 and q == j):      # pair (j, l) handled at a = 0
                continue
            pair = (a == 0 and q == l)
            c = r2 * s0 * (1.0 if pair else lams[a])
            for k in range(kmin, kmax + 1):
                d = P[p, k] - P[q, k]
                dn = _norm3(d)
                if dn < T[p, q, k] + c * B[k] + 1e-9:
                    n = d / max(dn, 1e-9)
                    if pair:      # d(x_j - x_l) = c B (lam_j g_j - lam_l g_l)
                        Nrm[nn, 0:3] = lam_j * n
                        Nrm[nn, 3:6] = -lam_l * n
                    else:
                        Nrm[nn, 3 * a:3 * a + 3] = n
                    src[nn, 0], src[nn, 1], src[nn, 2], src[nn, 3] = p, q, k, 0
                    nn += 1
            if use_seg:
                for k in range(s_lo, s_hi + 1):
                    bw = max(B[k], B[k + 1])
                    if bw <= 0.0:
                        continue
                    sd, cvec = _seg(P[p, k], P[p, k + 1], P[q, k], P[q, k + 1])
                    if sd < S[p, q] + c * bw + 1e-9:
                        if pair:
                            Nrm[nn, 0:3] = lam_j * cvec
                            Nrm[nn, 3:6] = -lam_l * cvec
                        else:
                            Nrm[nn, 3 * a:3 * a + 3] = cvec
                        src[nn, 0], src[nn, 1], src[nn, 2], src[nn, 3] = p, q, k, 1
                        nn += 1
    u0 = np.zeros(6)
    u0[0:3] = g0 / r2
    u0[3:6] = -g0 / r2
    ok, v, ib = _project_cone_nd(u0, Nrm, nn)
    if not ok:
        for c in range(4):
            info[c] = src[ib, c] if ib >= 0 else -1
        return -1
    vj, vl = v[0:3].copy(), v[3:6].copy()
    # ---- exact check with halving ------------------------------------------
    s = s0
    newj = P[j].copy()
    newl = P[l].copy()
    for _ in range(6):
        for k in range(kmin, kmax + 1):
            newj[k] = P[j, k] + r2 * lam_j * s * B[k] * vj
            newl[k] = P[l, k] + r2 * lam_l * s * B[k] * vl
        good = True
        for a in range(2):
            p = movers[a]
            newp = newj if a == 0 else newl
            for q in range(M):
                if q == p or (a == 1 and q == j):
                    continue
                if q == l and a == 0:
                    newq = newl
                else:
                    newq = P[q]
                for k in range(kmin, kmax + 1):
                    d_old = _norm3(P[p, k] - P[q, k])
                    d_new = _norm3(newp[k] - newq[k])
                    if d_new < d_old - 1e-9 and d_new < T[p, q, k]:
                        good = False
                        info[0], info[1], info[2], info[3] = p, q, k, 0
                        break
                if not good:
                    break
                if use_seg:
                    for k in range(s_lo, s_hi + 1):
                        if max(B[k], B[k + 1]) <= 0.0:
                            continue
                        so, _ = _seg(P[p, k], P[p, k + 1], P[q, k], P[q, k + 1])
                        sn, _ = _seg(newp[k], newp[k + 1], newq[k], newq[k + 1])
                        if sn < so - 1e-9 and sn < S[p, q]:
                            good = False
                            info[0], info[1], info[2], info[3] = p, q, k, 1
                            break
                if not good:
                    break
            if not good:
                break
        if good:
            for k in range(kmin, kmax + 1):
                P[j, k] = newj[k]
                P[l, k] = newl[k]
            return 1
        s *= 0.5
    return -2


@njit(cache=True)
def _cluster(P, j, l, ks, T, alpha):
    """Payloads connected to j or l at step ks through pairs within 2 alpha of
    their targets (breadth-first closure). Returns the member indices."""
    M = P.shape[0]
    inq = np.zeros(M, np.bool_)
    order = np.empty(M, np.int64)
    inq[j] = True
    inq[l] = True
    order[0], order[1] = j, l
    n, head = 2, 0
    while head < n:
        p = order[head]
        head += 1
        for q in range(M):
            if not inq[q] and _norm3(P[p, ks] - P[q, ks]) < T[p, q, ks] + 2.0 * alpha:
                inq[q] = True
                order[n] = q
                n += 1
    return order[:n].copy()


@njit(cache=True)
def _cluster_move(P, mem, ks, W, alpha, T, S, mass, use_seg, info):
    """Monotone tent dilation of a cluster: member i moves by
    eps B(k) (x_i(ks) - c), c the mass-weighted centroid at ks, so every
    in-cluster distance at ks grows by the factor (1 + eps) and the heavier
    payloads (near c) move least. Peak displacement capped at alpha, then the
    exact check with halving over every pair touching a member (same-step and
    segment, truncated distances non-decreasing). Returns 1 moved or -2."""
    M, K = P.shape[0], P.shape[1] - 1
    B = _profile(K, ks, W)
    kmin, kmax = K, 0
    for k in range(K + 1):
        if B[k] > 0.0:
            kmin = min(kmin, k)
            kmax = max(kmax, k)
    if kmax < kmin:
        return 0
    s_lo, s_hi = max(kmin - 1, 0), min(kmax, K - 1)
    n = mem.shape[0]
    c = np.zeros(3)
    mt = 0.0
    for a in range(n):
        c += mass[mem[a]] * P[mem[a], ks]
        mt += mass[mem[a]]
    c /= mt
    rmax = 0.0
    for a in range(n):
        rmax = max(rmax, _norm3(P[mem[a], ks] - c))
    if rmax < 1e-9:
        return -2
    ismem = np.zeros(M, np.bool_)
    for a in range(n):
        ismem[mem[a]] = True
    eps = alpha / rmax
    NEW = P.copy()
    for _ in range(6):
        for a in range(n):
            i = mem[a]
            dvec = P[i, ks] - c
            for k in range(kmin, kmax + 1):
                NEW[i, k] = P[i, k] + eps * B[k] * dvec
        good = True
        for a in range(n):
            p = mem[a]
            for q in range(M):
                if q == p or (ismem[q] and q < p):      # member pairs once
                    continue
                for k in range(kmin, kmax + 1):
                    d_old = _norm3(P[p, k] - P[q, k])
                    d_new = _norm3(NEW[p, k] - NEW[q, k])
                    if d_new < d_old - 1e-9 and d_new < T[p, q, k]:
                        good = False
                        info[0], info[1], info[2], info[3] = p, q, k, 0
                        break
                if not good:
                    break
                if use_seg:
                    for k in range(s_lo, s_hi + 1):
                        if max(B[k], B[k + 1]) <= 0.0:
                            continue
                        so, _ = _seg(P[p, k], P[p, k + 1], P[q, k], P[q, k + 1])
                        sn, _ = _seg(NEW[p, k], NEW[p, k + 1], NEW[q, k], NEW[q, k + 1])
                        if sn < so - 1e-9 and sn < S[p, q]:
                            good = False
                            info[0], info[1], info[2], info[3] = p, q, k, 1
                            break
                if not good:
                    break
            if not good:
                break
        if good:
            for a in range(n):
                i = mem[a]
                for k in range(kmin, kmax + 1):
                    P[i, k] = NEW[i, k]
            return 1
        eps *= 0.5
    return -2


@njit(cache=True)
def _pair_deficit(P, j, l, T, S, Dl, use_seg):
    """Worst deficit of pair (j, l): same-step (tau - d at free steps) or
    segment (S - s where s < delta). Returns (deficit, k*, unit direction g0,
    worst same-step deficit, worst segment surplus over delta)."""
    K = P.shape[1] - 1
    best, kbest = 0.0, -1
    g = np.zeros(3)
    ss_worst = 0.0
    for k in range(2, K):
        d = P[j, k] - P[l, k]
        dn = _norm3(d)
        df = T[j, l, k] - dn
        ss_worst = max(ss_worst, df)
        if df > best:
            best, kbest = df, k
            g = d / max(dn, 1e-9)
    seg_surplus = np.inf
    if use_seg:
        for k in range(1, K):
            sd, cvec = _seg(P[j, k], P[j, k + 1], P[l, k], P[l, k + 1])
            seg_surplus = min(seg_surplus, sd - Dl)
            if sd < Dl:                           # unsafe segment: drive to S
                df = S[j, l] - sd
                if df > best:
                    # tent centre at the free endpoint nearer the closest point
                    d0 = P[j, k] - P[l, k]
                    d1 = P[j, k + 1] - P[l, k + 1]
                    kk = k if _norm3(d0) < _norm3(d1) else k + 1
                    kk = min(max(kk, 2), K - 1)
                    best, kbest, g = df, kk, cvec.copy()
    return best, kbest, g, ss_worst, seg_surplus


@njit(cache=True)
def tent_sweep(P, T, S, D, LAM, Ws, use_seg, tol, alpha, rep, mass):
    """One Gauss-Seidel sweep over all pairs. rep (rows, 10) is filled with one
    row per active pair that could not move at any width:
    (j, l, k*, deficit, code, blocker p, q, k, kind, #widths whose cone was
    empty). A pair blocked at every width falls back to a cluster dilation
    (_cluster_move) of everything near target with it at k*; code -3 means
    that failed too, and the blocker is then the cluster move's. Returns
    (moved, total deficit, worst same-step deficit, min segment surplus over
    delta, #stuck rows, #cluster moves)."""
    M = P.shape[0]
    moved = 0
    nst = 0
    ncl = 0
    info = np.full(4, -1, np.int64)
    for j in range(M):
        for l in range(j + 1, M):
            df, ks, g0, ssw, segs = _pair_deficit(P, j, l, T, S, D[j, l], use_seg)
            if ks < 0 or (ssw < tol and segs >= 0.0):
                continue
            code, ncone = 0, 0
            for wi in range(len(Ws)):
                info[:] = -1
                code = _tent_move(P, j, l, ks, g0, min(df, alpha), Ws[wi], T, S,
                                  LAM[j, l], LAM[l, j], use_seg, info)
                if code == 1:
                    moved += 1
                    break
                if code == -1:
                    ncone += 1
            if code != 1:
                mem = _cluster(P, j, l, ks, T, alpha)
                for wi in range(len(Ws)):
                    info[:] = -1
                    if _cluster_move(P, mem, ks, Ws[wi], alpha, T, S, mass,
                                     use_seg, info) == 1:
                        code = 1
                        moved += 1
                        ncl += 1
                        break
                if code != 1:
                    code = -3
            if code != 1 and nst < rep.shape[0]:
                rep[nst, 0], rep[nst, 1], rep[nst, 2], rep[nst, 3] = j, l, ks, df
                rep[nst, 4] = code
                for c in range(4):
                    rep[nst, 5 + c] = info[c]
                rep[nst, 9] = ncone
                nst += 1
    total, worst, segmin = 0.0, 0.0, np.inf
    for j in range(M):
        for l in range(j + 1, M):
            df, ks, g0, ssw, segs = _pair_deficit(P, j, l, T, S, D[j, l], use_seg)
            total += max(ssw, 0.0) + max(-segs, 0.0)
            worst = max(worst, ssw)
            segmin = min(segmin, segs)
    return moved, total, worst, segmin, nst, ncl


def _post(P, R, T, S, rounds, beta):
    """Gated retraction + smoothing after separation (the method of chains'
    projected smoothing, plus a pull toward the nominal R): each point tries
    Delta = 0.25 (G(U) - U) - beta U, U = P - R, G a Gaussian blend along time,
    accepted only through the same exact check (no truncated same-step or
    segment distance may decrease), so separation is kept. Returns #moves."""
    from scipy.ndimage import gaussian_filter1d
    M = P.shape[0]
    Lbig = np.full(M, 1e12)                    # no link limit for tents
    n = 0
    for _ in range(rounds):
        U = P - R
        Delta = 0.25 * (gaussian_filter1d(U, sigma=1.5, axis=1) - U) - beta * U
        m = int(ck.smooth_round(P, Delta, T, Lbig, S, True))
        n += m
        if m == 0:
            break
    return n


_KIND = ("same-step", "segment")
_CODE = {0: "no tent support", -1: "cone empty", -2: "exact check rejected all halvings",
         -3: "pair and cluster moves both blocked"}


def _jam_report(rep, nst, top=8):
    """Readable stuck-pair rows (largest deficit first)."""
    rows = sorted(rep[:nst].tolist(), key=lambda r: -r[3])[:top]
    out = []
    for r in rows:
        j, l, ks, df, code, p, q, k, kind, ncone = r
        blk = (f"{_KIND[int(kind)]} ({int(p)},{int(q)}) at step {int(k)}"
               if p >= 0 else "unknown")
        out.append(dict(pair=(int(j), int(l)), k_star=int(ks), deficit=round(df, 2),
                        reason=_CODE.get(int(code), str(code)), blocker=blk,
                        widths_cone_empty=int(ncone)))
    return out


def format_jam(report):
    return "; ".join(f"{r['pair']}@{r['k_star']} short {r['deficit']} m: "
                     f"{r['reason']}, blocked by {r['blocker']}" for r in report)


def deconflict_tent(rbar, pairs, tgt, deltas, masses, seg_h=2.5,
                    widths=(0, 8, 4, 2, 1), tol=0.1, max_sweeps=3000, window=40,
                    alpha=5.0, post_rounds=300, beta=0.0, deadline=None,
                    log=None):
    """Monotone tent deconfliction on one grid. tgt: {(j,l): (K+1,) targets};
    deltas (M,M). Each move's peak relative displacement is capped at alpha
    (as the method of chains caps its point steps), so the first-order cone
    only collects constraints from pairs within alpha of their targets.

    Jam guards (so a run never spins): a sweep with no accepted move is a
    fixed point (the sweep is deterministic, so every later sweep would be
    identical) and stops at once; otherwise the run stops when the total
    deficit has not improved by 0.05 m over `window` sweeps, or at the
    wall-clock `deadline` (time.perf_counter() value). On a jam the stats
    carry `jam_report`: every stuck pair with its deficit, why its moves
    failed (cone empty / exact check) and the blocking constraint.
    Returns (P or None, sweeps, stats)."""
    P = np.array(rbar, float)
    R = np.array(rbar, float)
    M, K = P.shape[0], P.shape[1] - 1
    T = np.zeros((M, M, K + 1))
    for (j, l) in pairs:
        T[j, l] = T[l, j] = np.asarray(tgt[(j, l)], float)
    D = np.asarray(deltas, float)
    S = D + seg_h
    m = np.asarray(masses, float)
    LAM = m[None, :] / (m[:, None] + m[None, :])     # LAM[j,l]: j's share
    Ws = np.asarray(widths, float)
    rep = np.zeros((len(pairs), 10))
    best, best_it, moves, clusters = np.inf, 0, 0, 0
    why = "max sweeps"
    for it in range(max_sweeps):
        n, total, worst, segmin, nst, ncl = tent_sweep(P, T, S, D, LAM, Ws, True, tol,
                                                       alpha, rep, m)
        moves += n
        clusters += ncl
        if worst < tol and segmin >= 0.0:
            return P, it + 1, dict(moves=moves, cluster_moves=clusters, worst=worst,
                                   segmin=segmin,
                                   post_moves=_post(P, R, T, S, post_rounds, beta))
        if n == 0:
            why = "fixed point (no move accepted in a full sweep)"
            break
        if total < best - 0.05:
            best, best_it = total, it
        elif it - best_it >= window:
            why = f"no progress in {window} sweeps"
            break
        if deadline is not None and time.perf_counter() > deadline:
            why = "time budget"
            break
    report = _jam_report(rep, nst)
    stuck = sorted(r["pair"] for r in _jam_report(rep, nst, top=len(pairs)))
    # stuck but safe (as in the method of chains): same-step headroom partly
    # used, every segment still clears delta, which is all safety needs
    hmin = float(np.min([T[j, l].max() - D[j, l] for (j, l) in pairs] or [0.0]))
    if worst <= 0.5 * hmin and segmin >= 0.0:
        return P, it + 1, dict(moves=moves, worst=worst, segmin=segmin, jam_accepted=True,
                               jam_reason=why,
                               post_moves=_post(P, R, T, S, post_rounds, beta))
    if log is not None:
        log(f"  jam on K={K} after {it + 1} sweeps ({why}); worst deficit "
            f"{worst:.2f} m, {len(stuck)} stuck pairs: {format_jam(report)}")
    return None, it + 1, dict(moves=moves, cluster_moves=clusters, worst=worst,
                              segmin=segmin, jam=True,
                              jam_reason=why, jam_report=report, stuck_pairs=stuck)


def run_instance(sc0, deltas, masses, h=10.0, K0=30, Kmax=960, seg_h=2.5,
                 widths=(0, 8, 4, 2, 1), nominals=None, budget_s=120.0,
                 log=None, **kw):
    """Monotone tent + segment-sized tubes, refining K on a jam.
    nominals: callable K -> (sc, psis, rbar, LB). Returns a dict.

    Refinement guards: stop at Kmax, at the wall-clock budget, or when the
    refined grid jams on the same stuck pairs as the previous grid (the jam
    is then not a resolution problem, and doubling again only costs time).
    Every jam's report is kept in out["attempts"][K]["jam_report"]."""
    import sys
    import expand_method as em
    from chain_method import step_targets
    if log is None:
        log = lambda msg: print(msg, file=sys.stderr, flush=True)
    M = sc0.M
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc0.payloads]
    margin = sc0.delta - 2 * max(ext)
    rho = [e + margin / 2 + h / 2 for e in ext]
    out = dict(attempts={})
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    K, prev_stuck = K0, None
    while True:
        sc, psis, rbar, LB = (nominals or (lambda k: em.nominals_fast(sc0, k)))(K)
        tg = step_targets(rbar, pairs, deltas, rho, h)
        td = time.perf_counter()
        P, sweeps, st = deconflict_tent(rbar, pairs, tg, deltas, masses, seg_h=seg_h,
                                        widths=widths, deadline=deadline, log=log, **kw)
        out["attempts"][K] = dict(sweeps=sweeps, t=round(time.perf_counter() - td, 2),
                                  **{k: (round(float(v), 3) if isinstance(v, float) else v)
                                     for k, v in st.items()})
        if P is not None:
            break
        stop = None
        if st.get("jam_reason") == "time budget" or time.perf_counter() > deadline:
            stop = f"time budget ({budget_s:.0f} s) exhausted"
        elif prev_stuck is not None and st["stuck_pairs"] == prev_stuck:
            stop = (f"persistent jam: same stuck pairs on K={K // 2} and K={K}, "
                    "refinement is not helping")
        elif 2 * K > Kmax:
            stop = "jammed up to Kmax"
        if stop:
            out.update(status=stop, jam_report=st.get("jam_report"),
                       t=round(time.perf_counter() - t0, 1))
            log(f"  STOP: {stop}")
            return out
        prev_stuck = st["stuck_pairs"]
        K *= 2
    out["t_deconflict"] = round(time.perf_counter() - t0, 2)
    t1 = time.perf_counter()
    J, rs = em.seg_tubes(sc, psis, [P[j] for j in range(M)], pairs, deltas, K, M)
    out["t_tubes"] = round(time.perf_counter() - t1, 2)
    if J is None:
        out.update(status=f"tubes failed: {rs}", K=K)
        return out
    w, b, n = em.cont_margin(rs, pairs, deltas)
    out.update(status="ok", K=K, gap_pct=round(100 * (J - LB) / LB, 2),
               seg_bad=b, seg_margin=round(w, 3),
               t=round(time.perf_counter() - t0, 2))
    return out
