"""Expansion method (user's): gradual, monotone, link-limited repulsion.

Every round, while any overlap remains, every free point (j,k) tries to step
alpha directly away from its nearest same-timestep point of another payload.
The step direction is projected into the point's feasible set:
  - approach no nearby point (other payloads' same-step points that this step
    could bring below their target), and
  - lengthen neither of its chain links if that link is within alpha of the
    Rigid Link limit L_j.
No feasible direction -> the point holds. There is no pulling: the Rigid Link
constraint only blocks moves. Each move is checked exactly before it is
applied (step halved up to 3 times, else hold), and points are updated in
place one at a time, so no gap below its target ever shrinks and no link
ever exceeds L_j: the method is monotone by construction.

Stagnation (total overlap not improving over a window) doubles the timestep
count and restarts. Endpoints and step 1 are pinned; near them the targets
are tapered to what is reachable (never below the true clearance delta).
Hand-off: half-split tubes + decoupled convex re-plans, certified gap vs LB.

Usage: py expand_method.py [--alphas 2 5] [--link-rho 1.0] [--Ms 20 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_methods import relax_iso  # noqa: E402
from benchmark_tubes import tubes_detail  # noqa: E402
from chain_method import _project_cone, links_ok, nominals, step_targets  # noqa: E402
from multi_payload import scenario, translation  # noqa: E402

HEADROOM = 60.0


def _adj_seg(P, j, k, pj):
    """Segment check: closest approach of the two segments adjacent to (j,k),
    with (j,k) placed at pj, to every other payload's same-step segments.
    Segment q is q -> q+1; segment 0 -> 1 is pinned at both ends and skipped.
    Returns a list of (dist (M-1,), unit direction (M-1,3)) per segment."""
    K = P.shape[1] - 1
    oth = np.arange(P.shape[0]) != j
    out = []
    for q in (k - 1, k):
        if q < 1 or q + 1 > K:
            continue
        a0 = pj if q == k else P[j, q]
        a1 = pj if q + 1 == k else P[j, q + 1]
        d0 = a0[None] - P[oth, q]
        e = (a1[None] - P[oth, q + 1]) - d0
        ee = np.sum(e * e, axis=1)
        t = np.where(ee > 1e-12, np.clip(-np.sum(d0 * e, axis=1) / np.maximum(ee, 1e-12),
                                         0.0, 1.0), 0.0)
        c = d0 + t[:, None] * e
        dist = np.linalg.norm(c, axis=1)
        out.append((dist, c / np.maximum(dist, 1e-9)[:, None]))
    return out


def _seg_monotone(old, new, S_j):
    """Truncated segment distances min(d, S) must not decrease."""
    return all(np.all((nd >= od - 1e-9) | (nd >= S_j))
               for (od, _), (nd, _) in zip(old, new))


def try_shift(P, j, k, delta, T, L, S=None):
    """Move (j,k) by delta projected into its feasible cone (approach no
    nearby point, lengthen no near-limit link), with exact checks (step
    halved up to 3 times, else hold). Monotone like try_move. S (M,M):
    segment targets; if given, adjacent segments are checked too."""
    mag = float(np.linalg.norm(delta))
    if mag < 1e-6:
        return False
    M = P.shape[0]
    p = P[j, k]
    oth = np.arange(M) != j
    O = P[oth, k]
    D = p[None] - O
    dist = np.linalg.norm(D, axis=1)
    tg = T[j, oth, k]
    near = dist < tg + mag
    normals = [d / max(dn, 1e-9) for d, dn in zip(D[near], dist[near])]
    for q in (k - 1, k + 1):
        e = p - P[j, q]
        en = np.linalg.norm(e)
        if en > L - mag and en > 1e-9:
            normals.append(-e / en)
    if S is not None:
        S_j = S[j, oth]
        segs = _adj_seg(P, j, k, p)
        for sd, se in segs:
            normals += list(se[sd < S_j + mag])
    v = _project_cone(delta / mag, np.array(normals) if normals else np.zeros((0, 3)))
    if v is None:
        return False
    step = mag * max(float(v @ (delta / mag)), 0.0)  # projected length
    if step < 1e-6:
        return False
    for _ in range(4):
        new = p + step * v
        nd = np.linalg.norm(new[None] - O, axis=1)
        ok = np.all((nd >= dist - 1e-9) | (nd >= tg))
        for q in (k - 1, k + 1):
            if np.linalg.norm(new - P[j, q]) > max(L, np.linalg.norm(p - P[j, q])) + 1e-9:
                ok = False
        if ok and S is not None:
            ok = _seg_monotone(segs, _adj_seg(P, j, k, new), S_j)
        if ok:
            P[j, k] = new
            return True
        step *= 0.5
    return False


def smooth_round(P, R, T, L, K, sigma=1.5, alpha_s=0.25, S=None):
    """One projected Gaussian smoothing round: each payload's displacement
    from its nominal u is blended toward its Gaussian-smoothed version, and
    every resulting point move goes through the feasibility projection and
    exact checks (so smoothing can spread pushes but never give back
    clearance or stretch a link past its limit)."""
    from scipy.ndimage import gaussian_filter1d
    moved = 0
    for j in range(P.shape[0]):
        u = P[j] - R[j]
        us = gaussian_filter1d(u, sigma=sigma, axis=0)
        target = (1 - alpha_s) * u + alpha_s * us
        for k in range(2, K):
            if try_shift(P, j, k, target[k] - u[k], T, L[j], S):
                moved += 1
    return moved


def force_map(psis, dt, m, K):
    """Dense A with F = A u: the extra thrust (3K) needed to fly a position
    displacement field u (3(K+1)) under the discrete Tschauner-Hempel
    dynamics r[k+1] = r[k] + dt v[k], v[k+1] = v[k] + dt(Psi_k [r;v] + F/m),
    with the terminal velocity held fixed (boundary condition)."""
    n = 3 * (K + 1)
    I3 = np.eye(3)

    def V(k):                       # dv_k as a row-block map on u
        Z = np.zeros((3, n))
        if k < K:
            Z[:, 3 * (k + 1):3 * (k + 2)] += I3 / dt
            Z[:, 3 * k:3 * (k + 1)] -= I3 / dt
        return Z                    # dv_K = 0 (terminal velocity fixed)

    A = np.zeros((3 * K, n))
    for k in range(K):
        Ur = np.zeros((3, n))
        Ur[:, 3 * k:3 * (k + 1)] = I3
        Ps = np.asarray(psis[k])
        A[3 * k:3 * (k + 1)] = m * ((V(k + 1) - V(k)) / dt
                                    - Ps[:, :3] @ Ur - Ps[:, 3:] @ V(k))
    return A


def smooth_round_dyn(P, R, T, L, K, H, eta):
    """One projected gradient step on the dynamics-aware deviation cost
    ||A u||^2 per payload (H = A^T A, eta = step), each point move passed
    through the same feasibility projection and exact checks."""
    moved = 0
    for j in range(P.shape[0]):
        u = (P[j] - R[j]).reshape(-1)
        delta = (-eta[j] * (H[j] @ u)).reshape(K + 1, 3)
        for k in range(2, K):
            if try_shift(P, j, k, delta[k], T, L[j]):
                moved += 1
    return moved


def dilate_round(P, T, L, K, alpha, S=None):
    """Failsafe for jams: simultaneous cluster dilation. At each free step,
    group the payload points into clusters (connected components of 'closer
    than target + 2 alpha'); every cluster with an overlap is scaled about its
    centroid by one common factor, so EVERY pairwise distance inside it grows
    at once (no individual point has to find a free direction). Points move at
    most alpha, so pairs with an outside point (>= target + 2 alpha away)
    stay >= target + alpha. The factor is shrunk until no Rigid Link exceeds
    its limit; if the links leave no room, the cluster holds (more timesteps
    give the links more slack). Returns the number of clusters dilated."""
    M = P.shape[0]
    moved = 0
    for k in range(2, K):
        X = P[:, k]
        D = np.linalg.norm(X[:, None] - X[None], axis=2)
        Tk = T[:, :, k]
        adj = (D < Tk + 2 * alpha) & ~np.eye(M, dtype=bool)
        seen = np.zeros(M, bool)
        for s in range(M):
            if seen[s]:
                continue
            comp, stack = [], [s]
            seen[s] = True
            while stack:
                a = stack.pop()
                comp.append(a)
                for b in np.nonzero(adj[a] & ~seen)[0]:
                    seen[b] = True
                    stack.append(b)
            if len(comp) < 2:
                continue
            c = np.array(comp)
            sub = D[np.ix_(c, c)] - Tk[np.ix_(c, c)]
            np.fill_diagonal(sub, np.inf)
            if sub.min() >= 0:
                continue                       # no overlap in this cluster
            cen = X[c].mean(axis=0)
            rad = np.linalg.norm(X[c] - cen, axis=1).max()
            if rad < 1e-9:
                continue
            beta = alpha / rad
            for _ in range(6):
                new = cen + (1 + beta) * (X[c] - cen)
                ok = True
                for i, j in enumerate(c):
                    for q in (k - 1, k + 1):
                        old_len = np.linalg.norm(X[j] - P[j, q])
                        if np.linalg.norm(new[i] - P[j, q]) > max(L[j], old_len) + 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and S is not None:
                    P2 = P.copy()
                    P2[c, k] = new
                    oth = np.arange(M)
                    ok = all(_seg_monotone(_adj_seg(P, j, k, P[j, k]),
                                           _adj_seg(P2, j, k, P2[j, k]),
                                           S[j, oth != j]) for j in c)
                if ok:
                    P[c, k] = new
                    moved += 1
                    break
                beta *= 0.5
    return moved


def qp_polish(P, R, pairs, tg, As, K, prune=150.0, rounds=5, deltas=None,
              seg=False):
    """One joint convex QP over positions: minimize sum_j ||A_j u_j||^2
    (dynamic cost of the displacement from the nominal) subject to
    eta.(x_j - x_l) >= target at every free step, with eta frozen from the
    current configuration (a unit vector, so the linear constraint implies
    true separation). Endpoints and step 1 fixed. The current configuration
    is feasible, so the result is safe and never costlier. Distant pairs are
    left out, then added back if the solution brings them below target."""
    import cvxpy as cp
    M = P.shape[0]
    scale = 1.0 / np.mean([np.linalg.norm(A) for A in As]) ** 2

    # seg=True: one wall per SEGMENT k->k+1 (k = 1..K-1), shared by both
    # ends. eta = direction of the reference segment's closest point; the
    # reference satisfies eta.d >= |c| at both ends, so RHS = min(target, |c|)
    # keeps it feasible, and a straight segment whose ends satisfy one linear
    # constraint satisfies it everywhere: the QP output clears its RHS (>=
    # delta) at all times. If a reference segment dips below delta, the RHS
    # is delta and the QP may be infeasible (then the input is returned).
    walls = {}
    if seg:
        for (j, l) in pairs:
            dist, c = _seg_closest(P[j] - P[l])
            for k in range(1, K):
                tk = min(tg[(j, l)][k], tg[(j, l)][k + 1])
                if dist[k] < tk + prune:
                    eta = c[k] / max(dist[k], 1e-9)
                    rhs = max(deltas[j][l], min(tk, dist[k]))
                    walls[(j, l, k)] = (eta, rhs)
    active = set()
    if not seg:
        for (j, l) in pairs:
            d = np.linalg.norm(P[j] - P[l], axis=1)
            for k in range(2, K):
                if d[k] < tg[(j, l)][k] + prune:
                    active.add((j, l, k))
    for _ in range(rounds):
        X = [cp.Variable((K + 1, 3)) for _ in range(M)]
        cons = []
        for j in range(M):
            cons += [X[j][0] == P[j, 0], X[j][1] == P[j, 1], X[j][K] == P[j, K]]
        if seg:
            for (j, l, k), (eta, rhs) in walls.items():
                cons.append(eta @ (X[j][k] - X[l][k]) >= rhs)
                cons.append(eta @ (X[j][k + 1] - X[l][k + 1]) >= rhs)
        for (j, l, k) in active:
            d = P[j, k] - P[l, k]
            eta = d / max(np.linalg.norm(d), 1e-9)
            cons.append(eta @ (X[j][k] - X[l][k]) >= tg[(j, l)][k])
        obj = sum(cp.sum_squares(As[j] @ cp.reshape(X[j] - R[j], (3 * (K + 1),),
                                                    order="C"))
                  for j in range(M)) * scale
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return P, f"qp {prob.status}"
        Xn = np.array([x.value for x in X])
        added = 0
        if seg:
            for (j, l) in pairs:
                dn, _ = _seg_closest(Xn[j] - Xn[l])
                dist, c = _seg_closest(P[j] - P[l])
                for k in range(1, K):
                    if (j, l, k) in walls:
                        continue
                    tk = min(tg[(j, l)][k], tg[(j, l)][k + 1])
                    if dn[k] < min(tk, dist[k]) - 1e-3:
                        eta = c[k] / max(dist[k], 1e-9)
                        walls[(j, l, k)] = (eta, max(deltas[j][l], min(tk, dist[k])))
                        added += 1
        else:
            for (j, l) in pairs:
                d = np.linalg.norm(Xn[j] - Xn[l], axis=1)
                for k in range(2, K):
                    if d[k] < tg[(j, l)][k] - 1e-3 and (j, l, k) not in active:
                        active.add((j, l, k))
                        added += 1
        if added == 0:
            return Xn, f"ok ({len(active) + 2 * len(walls)} constraints)"
    return P, "qp did not settle"


def try_move(P, j, k, alpha, T, L, K, reach=np.inf, S=None):
    """Attempt one feasible expansion step for point (j,k). Returns True if
    it moved. Only points within `reach` of some target move (inf = all).
    S (M,M): segment targets. If given, the two adjacent segments must not
    lower any truncated segment distance min(d_seg, S), and a point with no
    same-step deficit but a segment deficit steps along that segment's
    closest-approach direction instead of away from its nearest point."""
    M = P.shape[0]
    p = P[j, k]
    oth = np.arange(M) != j
    O = P[oth, k]                                   # (M-1,3)
    D = p[None] - O
    dist = np.linalg.norm(D, axis=1)
    tg = T[j, oth, k]
    if np.min(dist - tg) > reach:
        return False
    u = D[int(np.argmin(dist))]
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        return False
    u = u / nu
    normals = []
    near = dist < tg + alpha
    for d, dn in zip(D[near], dist[near]):
        normals.append(d / max(dn, 1e-9))
    for q in (k - 1, k + 1):                       # Rigid Link halfspaces
        e = p - P[j, q]
        en = np.linalg.norm(e)
        if en > L - alpha and en > 1e-9:
            normals.append(-e / en)                 # v . e <= 0
    if S is not None:
        S_j = S[j, oth]
        segs = _adj_seg(P, j, k, p)
        worst_seg, u_seg = 0.0, None
        for sd, se in segs:
            normals += list(se[sd < S_j + alpha])
            gap = S_j - sd
            i = int(np.argmax(gap))
            if gap[i] > worst_seg:
                worst_seg, u_seg = float(gap[i]), se[i]
        if u_seg is not None and np.all(dist >= tg):
            u = u_seg
    v = _project_cone(u, np.array(normals) if normals else np.zeros((0, 3)))
    if v is None:
        return False
    step = alpha
    for _ in range(4):                              # exact checks
        new = p + step * v
        nd = np.linalg.norm(new[None] - O, axis=1)
        ok_sep = np.all((nd >= dist - 1e-9) | (nd >= tg))
        ok_link = True
        for q in (k - 1, k + 1):
            ln_new = np.linalg.norm(new - P[j, q])
            ln_old = np.linalg.norm(p - P[j, q])
            if ln_new > max(L, ln_old) + 1e-9:
                ok_link = False
        if ok_sep and ok_link and S is not None:
            ok_link = _seg_monotone(segs, _adj_seg(P, j, k, new), S_j)
        if ok_sep and ok_link:
            P[j, k] = new
            return True
        step *= 0.5
    return False


def _project_cone_batch(v, Nrm, msk, iters=12):
    """Row-wise _project_cone: v (B,3) unit vectors, Nrm (B,n,3) unit normals,
    msk (B,n) which normals are active. Cyclic projection onto the most
    violated half-space. Returns (v, ok); ok False where the projection
    vanished or did not settle."""
    v = v.copy()
    ok = np.ones(len(v), bool)
    if Nrm.shape[1] == 0:
        return v, ok
    rows = np.arange(len(v))
    for _ in range(iters):
        dots = np.where(msk, np.einsum("bnd,bd->bn", Nrm, v), np.inf)
        i = np.argmin(dots, axis=1)
        dmin = dots[rows, i]
        act = ok & (dmin < -1e-9)
        if not act.any():
            break
        v[act] -= dmin[act, None] * Nrm[rows, i][act]
        nv = np.linalg.norm(v, axis=1)
        ok &= ~(act & (nv < 1e-9))
        v[ok] /= nv[ok, None]
    final = np.where(msk, np.einsum("bnd,bd->bn", Nrm, v), np.inf)
    ok &= final.min(axis=1) >= -1e-6
    return v, ok


def _adj_seg_batch(P, j, ks, pj):
    """_adj_seg for many points at once: points (j, ks[b]) placed at pj[b].
    ks are free steps (2..K-1), so both adjacent segments exist. Returns
    (dist (2,B,M-1), unit direction (2,B,M-1,3)) for segments k-1 and k."""
    O = P[np.arange(P.shape[0]) != j]                  # (M-1, K+1, 3)
    out_d, out_c = [], []
    for a0, a1, q in ((P[j, ks - 1], pj, ks - 1), (pj, P[j, ks + 1], ks)):
        d0 = a0[:, None] - O[:, q].transpose(1, 0, 2)   # (B, M-1, 3)
        e = (a1[:, None] - O[:, q + 1].transpose(1, 0, 2)) - d0
        ee = np.sum(e * e, axis=2)
        t = np.where(ee > 1e-12, np.clip(-np.sum(d0 * e, axis=2) / np.maximum(ee, 1e-12),
                                         0.0, 1.0), 0.0)
        c = d0 + t[..., None] * e
        d = np.linalg.norm(c, axis=2)
        out_d.append(d)
        out_c.append(c / np.maximum(d, 1e-9)[..., None])
    return np.stack(out_d), np.stack(out_c)


def move_batch(P, j, ks, T, Lj, S=None, alpha=None, delta=None):
    """Vectorized try_move (alpha given) / try_shift (delta (B,3) given) for
    the points (j, ks) at once. The ks are at least 2 steps apart, so no two
    share a link or a segment, and every other point is fixed while they move:
    each accepted move passes exactly the per-point test of try_move /
    try_shift, so the monotone invariants are unchanged. Returns #moved."""
    M = P.shape[0]
    B = len(ks)
    rows = np.arange(B)
    oth = np.arange(M) != j
    p = P[j, ks]
    O = P[oth][:, ks].transpose(1, 0, 2)               # (B, M-1, 3)
    D = p[:, None] - O
    dist = np.linalg.norm(D, axis=2)
    tg = T[j][oth][:, ks].T                             # (B, M-1)
    if delta is None:
        mag = np.full(B, float(alpha))
        u = D[rows, dist.argmin(axis=1)]
        nu = np.linalg.norm(u, axis=1)
        live = nu > 1e-9
    else:
        mag = np.linalg.norm(delta, axis=1)
        live = mag > 1e-6
        u, nu = delta, mag
    u = u / np.maximum(nu, 1e-12)[:, None]
    Nrm = [D / np.maximum(dist, 1e-9)[..., None]]
    msk = [dist < tg + mag[:, None]]
    for q in (ks - 1, ks + 1):                          # Rigid Link half-spaces
        e = p - P[j, q]
        en = np.linalg.norm(e, axis=1)
        Nrm.append((-e / np.maximum(en, 1e-9)[:, None])[:, None])
        msk.append(((en > Lj - mag) & (en > 1e-9))[:, None])
    if S is not None:
        S_j = S[j, oth]
        sd0, sc0 = _adj_seg_batch(P, j, ks, p)
        for s in range(2):
            Nrm.append(sc0[s])
            msk.append(sd0[s] < S_j + mag[:, None])
        if delta is None:                               # segment-deficit direction
            gap = (S_j - sd0).transpose(1, 0, 2).reshape(B, -1)
            i = gap.argmax(axis=1)
            use = (gap[rows, i] > 0) & np.all(dist >= tg, axis=1)
            dirs = sc0.transpose(1, 0, 2, 3).reshape(B, -1, 3)[rows, i]
            u = np.where(use[:, None], dirs, u)
    v, ok = _project_cone_batch(u, np.concatenate(Nrm, axis=1),
                                np.concatenate(msk, axis=1))
    ok &= live
    if delta is None:
        step = mag.copy()
    else:
        step = mag * np.maximum(np.sum(v * u, axis=1), 0.0)
        ok &= step >= 1e-6
    acc = np.zeros(B, bool)
    newp = p.copy()
    for _ in range(4):                                  # exact checks, halving
        cand = p + step[:, None] * v
        nd = np.linalg.norm(cand[:, None] - O, axis=2)
        good = ok & ~acc & np.all((nd >= dist - 1e-9) | (nd >= tg), axis=1)
        for q in (ks - 1, ks + 1):
            ln_new = np.linalg.norm(cand - P[j, q], axis=1)
            ln_old = np.linalg.norm(p - P[j, q], axis=1)
            good &= ln_new <= np.maximum(Lj, ln_old) + 1e-9
        if S is not None and good.any():
            sd1, _ = _adj_seg_batch(P, j, ks, cand)
            good &= np.all((sd1 >= sd0 - 1e-9) | (sd1 >= S_j), axis=(0, 2))
        newp[good] = cand[good]
        acc |= good
        step = step * 0.5
    P[j, ks[acc]] = newp[acc]
    return int(acc.sum())


def deconflict(rbar, pairs, tgt, L, alpha, max_rounds=6000, window=300,
               reach=np.inf, h=HEADROOM, smooth=False, post_smooth=100,
               dyn=None, failsafe=False, Dm=None, seg_h=0.25, engine="numba"):
    """dyn = (H list, eta list): use the dynamics-aware smoothing step
    instead of the Gaussian blend. Dm (M,M) = clearances delta: if given,
    segment check on -- every segment k -> k+1 (k >= 1) is also driven to
    target delta + seg_h with a monotone truncated segment distance, and the
    phase only ends once every segment clears delta (so the chain is a
    feasible start for the split-plane walls)."""
    P = np.array(rbar, float)
    R = np.array(rbar, float)
    S = None if Dm is None else np.asarray(Dm, float) + seg_h

    M, K = P.shape[0], P.shape[1] - 1
    free = np.arange(2, K)
    Larr = np.asarray(L, float)
    Sarr = np.zeros((M, M)) if S is None else np.ascontiguousarray(S, float)
    if engine == "numba":
        try:
            import chain_kernels as ck
        except ImportError:          # numba not installed: same moves, slower
            import warnings
            warnings.warn("numba not available; deconflict falls back to "
                          "engine='numpy' (pip install numba for the fast path)")
            engine = "numpy"
    batches = [free[free % 2 == 0], free[free % 2 == 1]]

    def smooth_step():
        if dyn is not None:
            return smooth_round_dyn(P, R, T, L, K, dyn[0], dyn[1])
        if engine == "python":
            return smooth_round(P, R, T, L, K, S=S)
        if engine == "numba":
            from scipy.ndimage import gaussian_filter1d
            U = P - R
            Delta = 0.25 * (gaussian_filter1d(U, sigma=1.5, axis=1) - U)
            return int(ck.smooth_round(P, Delta, T, Larr, Sarr, S is not None))
        from scipy.ndimage import gaussian_filter1d
        n = 0
        for j in range(M):
            u = P[j] - R[j]
            delta = 0.25 * (gaussian_filter1d(u, sigma=1.5, axis=0) - u)
            for ks in batches:
                if len(ks):
                    n += move_batch(P, j, ks, T, L[j], S, delta=delta[ks])
        return n

    T = np.zeros((M, M, K + 1))
    for (j, l), v in tgt.items():
        T[j, l] = T[l, j] = v
    sl = slice(2, K)
    segw = [0.0]                      # worst segment deficit below delta
    PI = np.array([p[0] for p in pairs])
    PJ = np.array([p[1] for p in pairs])
    Tp = T[PI, PJ]                                       # (npairs, K+1)
    if S is not None:
        Dp = np.asarray(Dm, float)[PI, PJ][:, None]
        Sp = S[PI, PJ][:, None]

    def overlap():
        need = Tp[:, sl] - np.linalg.norm(P[PI, sl] - P[PJ, sl], axis=2)
        pos = need[need > 0]
        worst = float(pos.max()) if pos.size else 0.0
        total = float(pos.sum())
        if S is not None:
            d = P[PI] - P[PJ]                            # (npairs, K+1, 3)
            d0, e = d[:, :-1], d[:, 1:] - d[:, :-1]
            ee = np.sum(e * e, axis=2)
            t = np.where(ee > 1e-12, np.clip(-np.sum(d0 * e, axis=2)
                                             / np.maximum(ee, 1e-12), 0.0, 1.0), 0.0)
            sd = np.linalg.norm(d0 + t[..., None] * e, axis=2)[:, 1:]  # 0 -> 1 pinned
            segw[0] = float(np.max(Dp - sd))
            need = Sp - sd
            total += float(need[need > 0].sum())
        return worst, total

    def done(worst, slack=0.1):
        return worst < slack and segw[0] <= 0.0

    def clipped():
        return np.minimum(np.linalg.norm(P[PI, sl] - P[PJ, sl], axis=2),
                          Tp[:, sl]).ravel()

    worst, total = overlap()
    best, best_it = total, 0
    holds = moves = decreases = smooth_moves = 0
    stats_dil = [0]

    def finish(it):
        nonlocal decreases, smooth_moves
        if smooth:
            for _ in range(post_smooth):        # shape-only rounds, still safe
                c0 = clipped()
                n = smooth_step()
                decreases += int(np.sum(clipped() < c0 - 1e-6))
                smooth_moves += n
                if n == 0:
                    break
        res = overlap()[0]
        return P, it, res, dict(holds=holds, moves=moves,
                                decreases=decreases,
                                smooth_moves=smooth_moves,
                                dilations=stats_dil[0],
                                seg_worst=round(segw[0], 3))

    for it in range(max_rounds):
        if done(worst):
            return finish(it)
        c0 = clipped()
        if engine == "numba" and not np.isfinite(reach):
            n = int(ck.expansion_round(P, T, Larr, Sarr, S is not None, float(alpha)))
            moves += n
            holds += M * (K - 2) - n
        for j in range(M if engine != "numba" or np.isfinite(reach) else 0):
            if engine == "numpy" and not np.isfinite(reach):
                for ks in batches:
                    if len(ks):
                        n = move_batch(P, j, ks, T, L[j], S, alpha=alpha)
                        moves += n
                        holds += len(ks) - n
                continue
            for k in range(2, K):
                if try_move(P, j, k, alpha, T, L[j], K, reach, S=S):
                    moves += 1
                else:
                    holds += 1
        if smooth:
            smooth_moves += smooth_step()
        decreases += int(np.sum(clipped() < c0 - 1e-6))
        worst, total = overlap()
        if total < best - 0.05:
            best, best_it = total, it
        elif it - best_it >= window:
            dil = 0
            if failsafe and worst >= 0.1:
                # jam: simultaneous cluster dilations until clear or blocked
                for _ in range(4000):
                    c0 = clipped()
                    n = dilate_round(P, T, L, K, alpha, S=S)
                    decreases += int(np.sum(clipped() < c0 - 1e-6))
                    dil += n
                    worst, total = overlap()
                    if worst < 0.1 or n == 0:
                        break
                stats_dil[0] += dil
            st = dict(holds=holds, moves=moves, decreases=decreases,
                      smooth_moves=smooth_moves, dilations=stats_dil[0],
                      seg_worst=round(segw[0], 3))
            # stuck but safe: same-step headroom partly used, every segment
            # still clears delta
            if done(worst) or (worst <= h / 2 and segw[0] <= 0.0):
                return finish(it)
            return None, it, worst, st
    return None, max_rounds, worst, dict(holds=holds, moves=moves, decreases=decreases)


def _seg_closest(d):
    """Closest approach to the origin of each segment d[k] -> d[k+1]:
    returns (distances (K,), closest points (K,3))."""
    d0, e = d[:-1], d[1:] - d[:-1]
    ee = np.sum(e * e, axis=1)
    t = np.where(ee > 1e-12, np.clip(-np.sum(d0 * e, axis=1) / np.maximum(ee, 1e-12),
                                     0.0, 1.0), 0.0)
    c = d0 + t[:, None] * e
    return np.linalg.norm(c, axis=1), c


def _dyn_sparse(sc, psis, j, K):
    """Payload j's discrete dynamics and boundary conditions as one sparse
    equality A z = b over z = [r (K+1)x3 | v (K+1)x3 | F Kx3] (row-major),
    the same discretization as route2_multi._dyn / translation.solve_reference:
        r_{k+1} = r_k + dt v_k,  v_{k+1} = v_k + dt (Psi_k [r_k; v_k] + F_k / m).
    One sparse block compiles in a fraction of the time of 2K small cvxpy
    constraints."""
    import scipy.sparse as spr
    nz = 9 * K + 6
    ri = lambda k: 3 * k                 # noqa: E731
    vi = lambda k: 3 * (K + 1) + 3 * k   # noqa: E731
    fi = lambda k: 6 * (K + 1) + 3 * k   # noqa: E731
    I3 = np.eye(3)
    m, dt = sc.payloads[j].m, sc.dt
    rows, cols, vals, b = [], [], [], []

    def put(row0, col0, blk):
        rr, cc = np.nonzero(blk)
        rows.extend(row0 + rr)
        cols.extend(col0 + cc)
        vals.extend(blk[rr, cc])
    x0, xf = sc.payloads[j].x0, sc.payloads[j].xf
    for n_, (c0, val) in enumerate(((ri(0), x0.r), (vi(0), x0.v),
                                    (ri(K), xf.r), (vi(K), xf.v))):
        put(3 * n_, c0, I3)
        b.extend(np.asarray(val, float))
    row = 12
    for k in range(K):
        Pk = psis[k]
        put(row, ri(k + 1), I3)                   # r+ = r + dt v
        put(row, ri(k), -I3)
        put(row, vi(k), -dt * I3)
        b.extend([0.0] * 3)
        row += 3
        put(row, vi(k + 1), I3)                   # v+ = v + dt (Psi [r;v] + F/m)
        put(row, vi(k), -I3 - dt * Pk[:, 3:])
        put(row, ri(k), -dt * Pk[:, :3])
        put(row, fi(k), -(dt / m) * I3)
        b.extend([0.0] * 3)
        row += 3
    return spr.csr_matrix((vals, (rows, cols)), shape=(row, nz)), np.array(b)


def nominals_fast(sc0, K):
    """chain_method.nominals with the sparse formulation: every payload's
    fuel-optimal reference (no collision constraint) on a K-step grid.
    Returns (sc, psis, rbar list, LB = sum of fuels)."""
    import dataclasses
    import cvxpy as cp
    sc = dataclasses.replace(sc0, N=K)
    psis = translation.psi_list(sc)
    rbar, LB = [], 0.0
    for j in range(sc.M):
        A, b = _dyn_sparse(sc, psis, j, K)
        z = cp.Variable(9 * K + 6)
        Fz = cp.reshape(z[6 * (K + 1):], (K, 3), order="C")
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(Fz, 2, axis=1))), [A @ z == b])
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"nominal solve failed at K={K} (payload {j})")
        rbar.append(np.asarray(z.value[:3 * (K + 1)]).reshape(K + 1, 3))
        LB += float(np.sum(np.linalg.norm(np.asarray(Fz.value), axis=1)))
    return sc, psis, rbar, LB


def seg_tubes(sc, psis, x, pairs, deltas, N, M, cap=150.0):
    """Corridor re-plan that is safe at ALL times, not just at timesteps.
    Ball radius at step k = half the smallest surplus over delta of the
    reference's closest approach on the two segments touching k (min over
    partners). Deviations inside the balls interpolate linearly over a step,
    so the actual relative path stays within (eps_j + eps_l) of the reference
    relative path, which clears delta + that amount on every segment.
    Returns (fuel, rs) or (None, reason)."""
    import cvxpy as cp
    import route2_multi as r2
    eps = [np.full(N + 1, cap) for _ in range(M)]
    for (j, l) in pairs:
        dist, _ = _seg_closest(x[j] - x[l])
        sur = dist - deltas[j][l]                       # per segment
        pt = np.full(N + 1, np.inf)
        pt[:-1] = np.minimum(pt[:-1], sur)
        pt[1:] = np.minimum(pt[1:], sur)
        half = 0.5 * pt
        eps[j] = np.minimum(eps[j], half)
        eps[l] = np.minimum(eps[l], half)
    worst = min(float(e[1:N].min()) for e in eps)
    if worst < -1e-6:
        return None, f"reference segment dips below delta ({worst:.2f} m)"
    fuel, rs = 0.0, []
    for j in range(M):
        A, b = _dyn_sparse(sc, psis, j, N)
        z = cp.Variable(9 * N + 6)
        R = cp.reshape(z[:3 * (N + 1)], (N + 1, 3), order="C")
        cons = [A @ z == b]
        ks = [k for k in range(1, N) if eps[j][k] < cap]
        if ks:
            ks = np.array(ks)
            cons.append(cp.norm(R[ks] - np.asarray(x[j])[ks], 2, axis=1)
                        <= np.maximum(eps[j][ks], 0.0))
        Fz = cp.reshape(z[6 * (N + 1):], (N, 3), order="C")
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(Fz, 2, axis=1))), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, f"payload {j} {prob.status}"
        fuel += float(prob.value)
        rs.append(np.asarray(z.value[:3 * (N + 1)]).reshape(N + 1, 3))
    return fuel, rs


def split_plane_rounds(sc, psis, P, pairs, tg, deltas, K, M, rounds=10,
                       D=75.0):
    """Split-plane re-plan with re-centered midpoints. The wall directions
    eta and clearances T' are fixed once from the inflation output P; each
    round, every pair's split point is re-centered at the midpoint of the
    previous round's positions and all payloads re-solve independently (in
    parallel; only paired payloads exchange midpoints). The previous round's
    trajectories satisfy the new half-walls, so every round is feasible and
    safe at all times, and each payload's fuel can only decrease. Returns a
    list of (fuel, rs) per round."""
    import cvxpy as cp
    import route2_multi as r2
    W = []                                   # (j, l, k, eta, T')
    for (j, l) in pairs:
        dist, c = _seg_closest(P[j] - P[l])
        for k in range(1, K):
            tk = min(tg[(j, l)][k], tg[(j, l)][k + 1])
            if D is not None and dist[k] >= tk + 2 * D:
                continue
            W.append((j, l, k, c[k] / max(dist[k], 1e-9),
                      max(deltas[j][l], min(tk, dist[k]))))
    # Wall entries: one per (pair, segment, free end). Their normals and
    # clearances are fixed, only the split offsets change between rounds, so
    # every payload's problem is built once with its offsets as a Parameter
    # and re-solved each round (no re-canonicalization).
    ent = [(j, l, e, eta, tp) for (j, l, k, eta, tp) in W
           for e in (k, k + 1) if 2 <= e <= K - 1]
    Ej = np.array([x[0] for x in ent], int)
    El = np.array([x[1] for x in ent], int)
    Ee = np.array([x[2] for x in ent], int)
    Eeta = np.array([x[3] for x in ent], float).reshape(-1, 3)
    Etp = np.array([x[4] for x in ent], float)
    rows_j = [np.nonzero(Ej == j)[0] for j in range(M)]   # payload j: +eta side
    rows_l = [np.nonzero(El == j)[0] for j in range(M)]   # payload j: -eta side
    import scipy.sparse as spr
    nz = 9 * K + 6
    fi0 = 6 * (K + 1)
    probs = []
    for j in range(M):
        A, b = _dyn_sparse(sc, psis, j, K)
        z = cp.Variable(nz)
        cons = [A @ z == b]
        if D is not None:          # keep-in only needed when pairs are pruned
            cons += [cp.norm(z[3 * k:3 * k + 3] - P[j, k]) <= D for k in range(2, K)]
        idx = np.concatenate([Ee[rows_j[j]], Ee[rows_l[j]]])
        rhs = None
        if len(idx):
            Et = np.vstack([Eeta[rows_j[j]], -Eeta[rows_l[j]]])
            G = spr.csr_matrix((Et.ravel(), (np.repeat(np.arange(len(idx)), 3),
                                             (3 * idx[:, None] + np.arange(3)).ravel())),
                               shape=(len(idx), nz))
            rhs = cp.Parameter(len(idx))
            cons.append(G @ z >= rhs - 1e-6)      # eta_i . r[e_i] >= rhs_i
        Fz = cp.reshape(z[fi0:], (K, 3), order="C")
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(Fz, 2, axis=1))), cons)
        r = z[:3 * (K + 1)]
        probs.append((prob, r, rhs))

    cur = np.array(P, float)
    hist = []
    for _ in range(rounds):
        # split points at the midpoints of the current positions
        o = 0.5 * np.sum(Eeta * (cur[Ej, Ee] + cur[El, Ee]), axis=1)
        b_j = o + Etp / 2                 # eta . r_j >= o + T'/2
        b_l = -o + Etp / 2                # -eta . r_l >= -(o - T'/2)
        fuel, new = 0.0, []
        for j, (prob, r, rhs) in enumerate(probs):
            if rhs is not None:
                rhs.value = np.concatenate([b_j[rows_j[j]], b_l[rows_l[j]]])
            prob.solve(solver="CLARABEL")
            if prob.status not in ("optimal", "optimal_inaccurate"):
                return hist if hist else None
            fuel += float(prob.value)
            new.append(np.asarray(r.value).reshape(K + 1, 3))
        cur = np.array(new)
        hist.append((fuel, new))
    return hist


def split_plane_replan(sc, psis, P, pairs, tg, deltas, K, M, D=75.0):
    """Decoupled replacement for the joint QP + ball corridors: every pair
    wall is split at the pair's midpoint into two one-sided halfspaces, so
    each payload re-plans ALONE (true fuel, true dynamics) inside its own
    polytope corridor. For each nearby pair and segment k->k+1: eta = the
    reference segment's closest-point direction, T' = max(delta, min(target,
    |c|)); at each end e, o_e = eta.(x_j[e] + x_l[e])/2 and
        payload j: eta.r_j[e] >= o_e + T'/2,  payload l: eta.r_l[e] <= o_e - T'/2.
    Both ends satisfy the pair wall, positions are linear over the step, so
    the pair clears T' >= delta at all times. The reference meets every wall
    (eta.d >= |c| at both ends), so each solve is feasible. Every payload also
    stays within D of its reference; pairs left out (closest approach >=
    target + 2D) therefore cannot violate. Returns (fuel, rs, t_max) or
    (None, reason, None)."""
    import cvxpy as cp
    import route2_multi as r2
    walls = [[] for _ in range(M)]          # (e, eta, rhs, sign)
    for (j, l) in pairs:
        dist, c = _seg_closest(P[j] - P[l])
        for k in range(1, K):
            tk = min(tg[(j, l)][k], tg[(j, l)][k + 1])
            if dist[k] >= tk + 2 * D:
                continue
            eta = c[k] / max(dist[k], 1e-9)
            tp = max(deltas[j][l], min(tk, dist[k]))
            for e in (k, k + 1):
                o = float(eta @ (P[j, e] + P[l, e])) / 2
                walls[j].append((e, eta, o + tp / 2, 1.0))
                walls[l].append((e, -eta, -(o - tp / 2), 1.0))
    fuel, rs, tmax = 0.0, [], 0.0
    for j in range(M):
        t0 = time.perf_counter()
        r = cp.Variable((K + 1, 3))
        v = cp.Variable((K + 1, 3))
        F = cp.Variable((K, 3))
        cons = r2._dyn(sc, psis, r, v, F, j)
        cons += [cp.norm(r[k] - P[j, k]) <= D for k in range(2, K)]
        for (e, eta, rhs, _) in walls[j]:
            if 2 <= e <= K - 1:
                cons.append(eta @ r[e] >= rhs - 1e-6)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
        prob.solve(solver="CLARABEL")
        tmax = max(tmax, time.perf_counter() - t0)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, f"payload {j} {prob.status}", None
        fuel += float(prob.value)
        rs.append(np.asarray(r.value))
    return fuel, rs, tmax


def cont_margin(rs, pairs, deltas):
    """Exact between-timestep check. Positions are piecewise linear between
    samples (r[k+1] = r[k] + dt v[k]), i.e. a degree-1 B-spline with shared
    knots, so the relative path of a pair over a step is the segment from
    d_k to d_{k+1}; its closest approach to the origin is computed exactly.
    Returns (min continuous margin, #segments below delta, #pairs x steps)."""
    worst, bad, n = np.inf, 0, 0
    for (j, l) in pairs:
        d = rs[j] - rs[l]
        d0, e = d[:-1], d[1:] - d[:-1]
        ee = np.sum(e * e, axis=1)
        t = np.where(ee > 1e-12, np.clip(-np.sum(d0 * e, axis=1) / np.maximum(ee, 1e-12),
                                         0.0, 1.0), 0.0)
        mind = np.linalg.norm(d0 + t[:, None] * e, axis=1) - deltas[j][l]
        worst = min(worst, float(mind.min()))
        bad += int(np.sum(mind < -1e-6))
        n += len(mind)
    return worst, bad, n


def run(M, seed, alpha, link_rho, K0=30, tf=500.0, Kmax=240, reach=np.inf,
        h=HEADROOM, h_conf=None, smooth=False, dyn_smooth=False,
        post_smooth=100, qp=False, failsafe=False, seg_safe=False,
        gauss_h=None, split=False, split_rounds=10, split_D=75.0,
        seg_chain=False, layout="random_seq", ring_radius=None,
        link_floor=False, tent_h=None, engine="numba"):
    """h_conf: pairs whose nominals overlap (closer than delta + h at some
    timestep) get the larger target delta + h_conf, creating smoothing margin
    exactly at the kinks; all other pairs keep delta + h. ring_radius None =
    the constant-density scaling 500 (M/20)^(1/3) m."""
    ring = (500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0) if ring_radius is None
            else float(ring_radius))
    sc0 = scenario.generate(layout, M=M, n_points=8, n_agents=3 * M,
                            seed=seed, nu_deg=60, N=K0, tf=tf,
                            ring_radius=ring, max_tries=5000)
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc0.payloads]
    margin = sc0.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    rho = [e + margin / 2 + h / 2 for e in ext]
    L = [link_rho * r for r in rho]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    row = dict(M=M, seed=seed, alpha=alpha, link_rho=link_rho)

    _nom_cache = {}

    def nom(K):
        """Nominals on a K-step grid, solved once per K (sparse formulation)."""
        if K not in _nom_cache:
            _nom_cache[K] = nominals_fast(sc0, K)
        return _nom_cache[K]

    sc, psis, rbar, LB0 = nom(K0)
    if link_floor:
        # Rigid Link limit never below the nominal's own longest link, so long
        # nominal paths do not force a finer grid before deconfliction starts
        L = [max(L[j], float(np.max(np.linalg.norm(np.diff(rbar[j], axis=0), axis=1))))
             for j in range(M)]
    row["link_floor"] = bool(link_floor)

    def baseline(tag, hb, profile):
        rho_b = [e + margin / 2 + hb / 2 for e in ext]
        tgb = {(j, l): rho_b[j] + rho_b[l] for (j, l) in pairs}
        row[f"{tag}_h"] = hb
        tb = time.perf_counter()
        xb, _, _ = relax_iso(rbar, pairs, tgb, K0, M, profile=profile)
        if xb is None:
            row[f"{tag}_status"] = "inflation did not clear"
            return
        if seg_safe:
            Jb, rsb = seg_tubes(sc, psis, [np.asarray(v) for v in xb], pairs,
                                deltas, K0, M)
            if Jb is None:
                row[f"{tag}_status"] = rsb
        else:
            Jb, rsb, _ = tubes_detail(sc, psis, pairs, deltas, xb, K0, M)
        row[f"t_{tag}"] = round(time.perf_counter() - tb, 1)
        row[f"gap_{tag}_pct"] = round(100 * (Jb - LB0) / LB0, 2) if Jb else None
        if Jb:
            cw, cb, cn = cont_margin(rsb, pairs, deltas)
            row[f"{tag}_cont_min_margin"] = round(cw, 3)
            row[f"{tag}_cont_violations"] = f"{cb}/{cn}"

    baseline("gauss", h if gauss_h is None else gauss_h, "gauss")
    if tent_h is not None:
        baseline("tent", tent_h, "tent")

    t0 = time.perf_counter()
    K = K0
    while not links_ok(nom(K)[2], L):
        K *= 2
    attempts = {}
    while True:
        sck, psk, rbk, LBk = nom(K)
        tg = step_targets(rbk, pairs, deltas, L, h)
        if h_conf is not None:
            tgc = step_targets(rbk, pairs, deltas, L, h_conf)
            nconf = 0
            for (j, l) in pairs:
                dmin = float(np.min(np.linalg.norm(rbk[j] - rbk[l], axis=1)))
                if dmin < deltas[j][l] + h:
                    tg[(j, l)] = tgc[(j, l)]
                    nconf += 1
            row["conflicted_pairs"] = nconf
        dyn = None
        if dyn_smooth:
            Hs, etas = [], []
            for j in range(M):
                A = force_map(psk, sck.dt, sc0.payloads[j].m, K)
                Hm = A.T @ A
                fz = np.zeros(K + 1, bool)
                fz[[0, 1, K]] = True
                mask = np.repeat(~fz, 3).astype(float)
                Hm = Hm * mask[:, None] * mask[None, :]   # pinned rows/cols out
                lam = float(np.linalg.eigvalsh(Hm).max())
                Hs.append(Hm)
                etas.append(0.5 / max(lam, 1e-12))
            dyn = (Hs, etas)
        P, it, res, st = deconflict(rbk, pairs, tg, L, alpha, reach=reach, h=h,
                                    smooth=smooth or dyn_smooth, dyn=dyn,
                                    post_smooth=post_smooth, failsafe=failsafe,
                                    Dm=(deltas if seg_chain else None),
                                    engine=engine)
        attempts[K] = [it, round(res, 1), st]
        if P is not None or K >= Kmax:
            break
        K *= 2
    row["attempts"] = attempts
    row["t"] = round(time.perf_counter() - t0, 1)
    if P is None:
        row["status"] = f"stuck up to Kmax (resid {res:.1f})"
        return row
    if split:
        t1 = time.perf_counter()
        hist = split_plane_rounds(sck, psk, P, pairs, tg, deltas, K, M,
                                  rounds=split_rounds, D=split_D)
        row["t_split_total"] = round(time.perf_counter() - t1, 1)
        if not hist:
            row["split_status"] = "infeasible"
        else:
            row["gap_split_by_round"] = [round(100 * (f - LBk) / LBk, 2)
                                         for f, _ in hist]
            worst_c, viol = np.inf, 0
            for _, rs_ in hist:
                cw, cb, cn = cont_margin(rs_, pairs, deltas)
                worst_c = min(worst_c, cw)
                viol += cb
            row["split_cont_min_margin_all_rounds"] = round(worst_c, 3)
            row["split_cont_violations_all_rounds"] = viol
    if qp:
        x0 = [P[j].copy() for j in range(M)]
        J0, _, _ = tubes_detail(sck, psk, pairs, deltas, x0, K, M)
        row["gap_before_qp_pct"] = round(100 * (J0 - LBk) / LBk, 2) if J0 else None
        As = [force_map(psk, sck.dt, sc0.payloads[j].m, K) for j in range(M)]
        t1 = time.perf_counter()
        P, qstat = qp_polish(P, np.array(rbk), pairs, tg, As, K,
                             deltas=deltas, seg=seg_safe)
        row["t_qp"] = round(time.perf_counter() - t1, 1)
        row["qp"] = qstat
    x = [P[j].copy() for j in range(M)]
    if seg_safe:
        Jc, rs = seg_tubes(sck, psk, x, pairs, deltas, K, M)
        if Jc is None:
            row["status"] = f"segment tubes: {rs}"
            return row
    else:
        Jc, rs, _ = tubes_detail(sck, psk, pairs, deltas, x, K, M)
    if Jc is None:
        row["status"] = "tube re-plan infeasible"
        return row
    row["K_final"] = K
    row["gap_expand_pct"] = round(100 * (Jc - LBk) / LBk, 2)
    cw, cb, cn = cont_margin(rs, pairs, deltas)
    row["cont_min_margin"] = round(cw, 3)
    row["cont_violations"] = f"{cb}/{cn}"
    row["max_dev"] = round(max(float(np.max(np.linalg.norm(x[j] - rbk[j], axis=1)))
                               for j in range(M)), 0)
    row["max_link_over_limit"] = round(max(
        float(np.max(np.linalg.norm(np.diff(x[j], axis=0), axis=1))) / L[j]
        for j in range(M)), 3)
    row["min_margin"] = round(min(
        float(np.min(np.linalg.norm(rs[j] - rs[l], axis=1))) - deltas[j][l]
        for (j, l) in pairs), 4)
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 5.0])
    ap.add_argument("--link-rho", type=float, default=1.0)
    ap.add_argument("--Ms", type=int, nargs="+", default=[20, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--reaches", type=float, nargs="+", default=[float("inf")],
                    help="only points within this distance of a target move")
    ap.add_argument("--h", type=float, default=HEADROOM)
    ap.add_argument("--h-confs", type=float, nargs="+", default=[None],
                    help="larger headroom for pairs that conflict in the nominal")
    ap.add_argument("--smooth", action="store_true",
                    help="projected (monotone) Gaussian smoothing each round")
    ap.add_argument("--dyn-smooth", action="store_true",
                    help="projected gradient step on ||A u||^2 (TH dynamics)")
    ap.add_argument("--post-smooth", type=int, default=100)
    ap.add_argument("--qp-polish", action="store_true",
                    help="final joint convex QP on dynamic cost, frozen sides")
    ap.add_argument("--failsafe", action="store_true",
                    help="simultaneous cluster dilation when the expansion jams")
    ap.add_argument("--split-planes", action="store_true",
                    help="also run the decoupled split-plane re-plan")
    ap.add_argument("--split-rounds", type=int, default=10)
    ap.add_argument("--split-D", type=float, default=75.0,
                    help="keep-in radius around the inflation output; "
                         "0 = walls for every pair and step, no keep-in")
    ap.add_argument("--gauss-h", type=float, default=None,
                    help="headroom for the gauss baseline (default: same as --h)")
    ap.add_argument("--seg-safe", action="store_true",
                    help="safety between timesteps: per-segment QP walls and "
                         "segment-based corridor radii (gauss uses the radii too)")
    ap.add_argument("--seg-chain", action="store_true",
                    help="segment check in the chains: every segment must "
                         "clear delta before the phase ends")
    ap.add_argument("--layout", choices=("random_seq", "crossing", "random_joint"),
                    default="random_seq")
    ap.add_argument("--ring", type=float, default=None,
                    help="start/goal ring radius [m]; default: constant-density "
                         "scaling 500 (M/20)^(1/3)")
    ap.add_argument("--N", type=int, default=30,
                    help="starting number of time steps (doubled on a jam)")
    ap.add_argument("--Kmax", type=int, default=240)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--link-floor", action="store_true",
                    help="Rigid Link limit = max(link_rho*rho, longest nominal link): "
                         "no grid refinement forced by long nominal paths")
    ap.add_argument("--engine", choices=("numba", "numpy", "python"), default="numba",
                    help="deconfliction moves: numba kernels (default), numpy batches "
                         "(red-black), or the original per-point Python loop")
    ap.add_argument("--tent-h", type=float, default=None,
                    help="also run the tent-profile inflation baseline at this headroom")
    ap.add_argument("--workers", type=int, default=1,
                    help="instances run in parallel processes")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    jobs = []
    for hc in a.h_confs:
        for reach in a.reaches:
            for alpha in a.alphas:
                for M in a.Ms:
                    for s in a.seeds:
                        jobs.append(dict(
                            M=M, seed=s, alpha=alpha, link_rho=a.link_rho,
                            K0=a.N, tf=a.tf, Kmax=a.Kmax, reach=reach, h=a.h,
                            h_conf=hc, smooth=a.smooth, dyn_smooth=a.dyn_smooth,
                            post_smooth=a.post_smooth, qp=a.qp_polish,
                            failsafe=a.failsafe, seg_safe=a.seg_safe,
                            gauss_h=a.gauss_h, split=a.split_planes,
                            split_rounds=a.split_rounds,
                            split_D=(None if a.split_D == 0 else a.split_D),
                            seg_chain=a.seg_chain, layout=a.layout,
                            ring_radius=a.ring, link_floor=a.link_floor,
                            tent_h=a.tent_h, engine=a.engine))
    name = (f"expand_method_rho{a.link_rho:g}_h{a.h:g}"
            + ("_hconf" if a.h_confs != [None] else "")
            + ("_smooth" if a.smooth else "")
            + (f"_dyn{a.post_smooth}" if a.dyn_smooth else "")
            + ("_qp" if a.qp_polish else "") + ".json")
    p = Path(__file__).with_name("results") / (a.out or name)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = []

    def record(row):
        out.append(row)
        print(json.dumps(row), flush=True)
        p.write_text(json.dumps(out, indent=1))

    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_run_job, job) for job in jobs]
            for f in as_completed(futs):
                record(f.result())
    else:
        for job in jobs:
            record(_run_job(job))


def _run_job(job):
    """One instance, never raising (errors become a status row)."""
    job = dict(job)
    M, s, alpha, link_rho = job.pop("M"), job.pop("seed"), job.pop("alpha"), job.pop("link_rho")
    try:
        row = run(M, s, alpha, link_rho, **job)
        row["smooth"] = "dyn" if job["dyn_smooth"] else job["smooth"]
        row["post_smooth"] = job["post_smooth"]
        row["h"] = job["h"]
        row["h_conf"] = job["h_conf"]
    except Exception as exc:
        row = dict(M=M, seed=s, alpha=alpha, status=f"error: {str(exc)[:90]}")
    row["layout"] = job["layout"]
    row["N_start"] = job["K0"]
    row["reach"] = job["reach"] if np.isfinite(job["reach"]) else "all"
    return row


if __name__ == "__main__":
    main()
