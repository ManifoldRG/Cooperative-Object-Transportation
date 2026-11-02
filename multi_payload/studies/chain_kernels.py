"""numba kernels for the method of chains (expand_method.deconflict).

Each kernel is a line-by-line port of the per-point moves try_move (expansion)
and try_shift (projected smoothing), including the segment check, applied in
the same order (payload by payload, step by step, in place). Every accepted
move passes the same exact test, so the monotone invariants are unchanged; the
results match the reference Python loop up to floating-point round-off.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _seg(a0, a1, b0, b1):
    """Closest approach of two points moving linearly over one step:
    (distance, unit direction from b to a at the closest point)."""
    d0 = a0 - b0
    e = (a1 - b1) - d0
    ee = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
    t = 0.0
    if ee > 1e-12:
        t = -(d0[0] * e[0] + d0[1] * e[1] + d0[2] * e[2]) / ee
        t = min(1.0, max(0.0, t))
    c = d0 + t * e
    n = np.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
    return n, c / max(n, 1e-9)


@njit(cache=True)
def _project_cone(u, N, nn):
    """chain_method._project_cone: cyclic projection of unit u onto the most
    violated half-space {d . n >= 0}. Returns (ok, v)."""
    v = u.copy()
    if nn == 0:
        return True, v
    for _ in range(12):
        imin, dmin = 0, np.inf
        for i in range(nn):
            d = N[i, 0] * v[0] + N[i, 1] * v[1] + N[i, 2] * v[2]
            if d < dmin:
                dmin, imin = d, i
        if dmin >= -1e-9:
            return True, v
        v = v - dmin * N[imin]
        nv = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if nv < 1e-9:
            return False, v
        v = v / nv
    for i in range(nn):
        if N[i, 0] * v[0] + N[i, 1] * v[1] + N[i, 2] * v[2] < -1e-6:
            return False, v
    return True, v


@njit(cache=True)
def _move(P, j, k, T, Lj, S, use_seg, alpha, delta, expand):
    """One exact, monotone move of point (j, k). expand=True: try_move with
    step alpha; expand=False: try_shift by delta. Returns True if it moved."""
    M = P.shape[0]
    p = P[j, k].copy()
    dist = np.empty(M)
    Nrm = np.empty((3 * M + 2, 3))
    nn = 0
    if expand:
        mag = alpha
    else:
        mag = np.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
        if mag < 1e-6:
            return False
    imin, dmin = -1, np.inf
    all_clear = True
    for l in range(M):
        if l == j:
            continue
        d = p - P[l, k]
        dl = np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        dist[l] = dl
        if dl < dmin:
            dmin, imin = dl, l
        if dl < T[j, l, k]:
            all_clear = False
        if dl < T[j, l, k] + mag:
            Nrm[nn] = d / max(dl, 1e-9)
            nn += 1
    if expand:
        u = p - P[imin, k]
        nu = np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        if nu < 1e-9:
            return False
        u = u / nu
    else:
        u = delta / mag
    for q in (k - 1, k + 1):                          # Rigid Link half-spaces
        e = p - P[j, q]
        en = np.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
        if en > Lj - mag and en > 1e-9:
            Nrm[nn] = -e / en
            nn += 1
    sd_old = np.empty((2, M))
    skip = np.zeros((2, M), np.bool_)
    if use_seg:
        worst, useg = 0.0, np.zeros(3)
        found = False
        for s in range(2):
            q = k - 1 + s
            for l in range(M):
                if l == j:
                    continue
                # Exact culling. The relative segment runs from da to db, so its
                # closest approach is >= max(|da|, |db|) - |db - da|. Moving
                # p by <= mag changes that bound by <= 2 mag, so if it exceeds
                # S + 3 mag the segment adds no normal or deficit now and
                # stays above S after any candidate move: skip it.
                if s == 0:
                    da = P[j, q] - P[l, q]
                    db = p - P[l, q + 1]
                else:
                    da = p - P[l, q]
                    db = P[j, q + 1] - P[l, q + 1]
                ed = db - da
                lb = (max(np.sqrt(da[0] ** 2 + da[1] ** 2 + da[2] ** 2),
                          np.sqrt(db[0] ** 2 + db[1] ** 2 + db[2] ** 2))
                      - np.sqrt(ed[0] ** 2 + ed[1] ** 2 + ed[2] ** 2))
                if lb > S[j, l] + 3.0 * mag:
                    skip[s, l] = True
                    continue
                if s == 0:
                    sd, sc = _seg(P[j, q], p, P[l, q], P[l, q + 1])
                else:
                    sd, sc = _seg(p, P[j, q + 1], P[l, q], P[l, q + 1])
                sd_old[s, l] = sd
                if sd < S[j, l] + mag:
                    Nrm[nn] = sc
                    nn += 1
                if S[j, l] - sd > worst:
                    worst, useg, found = S[j, l] - sd, sc, True
        if expand and found and all_clear:
            u = useg
    ok, v = _project_cone(u, Nrm, nn)
    if not ok:
        return False
    if expand:
        step = alpha
    else:
        step = mag * max(v[0] * u[0] + v[1] * u[1] + v[2] * u[2], 0.0)
        if step < 1e-6:
            return False
    for _ in range(4):                                # exact checks, halving
        new = p + step * v
        good = True
        for l in range(M):
            if l == j:
                continue
            d = new - P[l, k]
            nd = np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            if nd < dist[l] - 1e-9 and nd < T[j, l, k]:
                good = False
                break
        if good:
            for q in (k - 1, k + 1):
                a = new - P[j, q]
                b = p - P[j, q]
                ln_new = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
                ln_old = np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
                if ln_new > max(Lj, ln_old) + 1e-9:
                    good = False
        if good and use_seg:
            for s in range(2):
                q = k - 1 + s
                for l in range(M):
                    if l == j or skip[s, l]:
                        continue
                    if s == 0:
                        sd, _ = _seg(P[j, q], new, P[l, q], P[l, q + 1])
                    else:
                        sd, _ = _seg(new, P[j, q + 1], P[l, q], P[l, q + 1])
                    if sd < sd_old[s, l] - 1e-9 and sd < S[j, l]:
                        good = False
                        break
                if not good:
                    break
        if good:
            P[j, k] = new
            return True
        step *= 0.5
    return False


@njit(cache=True)
def expansion_round(P, T, L, S, use_seg, alpha):
    """One expansion round (try_move for every free point). Returns #moved."""
    M, K = P.shape[0], P.shape[1] - 1
    moved = 0
    dummy = np.zeros(3)
    for j in range(M):
        for k in range(2, K):
            if _move(P, j, k, T, L[j], S, use_seg, alpha, dummy, True):
                moved += 1
    return moved


@njit(cache=True)
def smooth_round(P, Delta, T, L, S, use_seg):
    """One projected smoothing round (try_shift for every free point), with the
    per-point displacements Delta (M, K+1, 3) computed from the displacement
    field at the start of the round. Returns #moved."""
    M, K = P.shape[0], P.shape[1] - 1
    moved = 0
    for j in range(M):
        for k in range(2, K):
            if _move(P, j, k, T, L[j], S, use_seg, 0.0, Delta[j, k], False):
                moved += 1
    return moved
