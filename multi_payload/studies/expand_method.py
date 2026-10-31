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
from multi_payload import scenario  # noqa: E402

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


def deconflict(rbar, pairs, tgt, L, alpha, max_rounds=6000, window=300,
               reach=np.inf, h=HEADROOM, smooth=False, post_smooth=100,
               dyn=None, failsafe=False, Dm=None, seg_h=0.25):
    """dyn = (H list, eta list): use the dynamics-aware smoothing step
    instead of the Gaussian blend. Dm (M,M) = clearances delta: if given,
    segment check on -- every segment k -> k+1 (k >= 1) is also driven to
    target delta + seg_h with a monotone truncated segment distance, and the
    phase only ends once every segment clears delta (so the chain is a
    feasible start for the split-plane walls)."""
    P = np.array(rbar, float)
    R = np.array(rbar, float)
    S = None if Dm is None else np.asarray(Dm, float) + seg_h

    def smooth_step():
        if dyn is None:
            return smooth_round(P, R, T, L, K, S=S)
        return smooth_round_dyn(P, R, T, L, K, dyn[0], dyn[1])
    M, K = P.shape[0], P.shape[1] - 1
    T = np.zeros((M, M, K + 1))
    for (j, l), v in tgt.items():
        T[j, l] = T[l, j] = v
    sl = slice(2, K)
    segw = [0.0]                      # worst segment deficit below delta

    def overlap():
        worst, total = 0.0, 0.0
        for (j, l) in pairs:
            need = tgt[(j, l)][sl] - np.linalg.norm(P[j, sl] - P[l, sl], axis=1)
            if (need > 0).any():
                worst = max(worst, float(need.max()))
                total += float(need[need > 0].sum())
        if S is not None:
            sw = -np.inf
            for (j, l) in pairs:
                sd, _ = _seg_closest(P[j] - P[l])
                sd = sd[1:]                       # segment 0 -> 1 is pinned
                sw = max(sw, float(np.max(Dm[j][l] - sd)))
                need = S[j, l] - sd
                total += float(need[need > 0].sum())
            segw[0] = sw
        return worst, total

    def done(worst, slack=0.1):
        return worst < slack and segw[0] <= 0.0

    def clipped():
        return np.concatenate([np.minimum(np.linalg.norm(P[j, sl] - P[l, sl], axis=1),
                                          tgt[(j, l)][sl]) for (j, l) in pairs])

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
        for j in range(M):
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
        r = cp.Variable((N + 1, 3))
        v = cp.Variable((N + 1, 3))
        F = cp.Variable((N, 3))
        cons = r2._dyn(sc, psis, r, v, F, j)
        for k in range(1, N):
            if eps[j][k] < cap:
                cons.append(cp.norm(r[k] - x[j][k]) <= max(eps[j][k], 0.0))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, f"payload {j} {prob.status}"
        fuel += float(prob.value)
        rs.append(np.asarray(r.value))
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
    cur = np.array(P, float)
    hist = []
    for _ in range(rounds):
        walls = [[] for _ in range(M)]
        for (j, l, k, eta, tp) in W:
            for e in (k, k + 1):
                if 2 <= e <= K - 1:
                    o = float(eta @ (cur[j, e] + cur[l, e])) / 2
                    walls[j].append((e, eta, o + tp / 2))
                    walls[l].append((e, -eta, -(o - tp / 2)))
        fuel, new = 0.0, []
        for j in range(M):
            r = cp.Variable((K + 1, 3))
            v = cp.Variable((K + 1, 3))
            F = cp.Variable((K, 3))
            cons = r2._dyn(sc, psis, r, v, F, j)
            if D is not None:      # keep-in only needed when pairs are pruned
                cons += [cp.norm(r[k] - P[j, k]) <= D for k in range(2, K)]
            if walls[j]:
                idx = np.array([e for (e, _, _) in walls[j]])
                Et = np.array([eta for (_, eta, _) in walls[j]])
                rhs = np.array([b for (_, _, b) in walls[j]])
                # one vectorized block: eta_i . r[e_i] >= rhs_i
                sel = np.zeros((len(idx), K + 1))
                sel[np.arange(len(idx)), idx] = 1.0
                cons.append(cp.sum(cp.multiply(Et, sel @ r), axis=1) >= rhs - 1e-6)
            prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
            prob.solve(solver="CLARABEL")
            if prob.status not in ("optimal", "optimal_inaccurate"):
                return hist if hist else None
            fuel += float(prob.value)
            new.append(np.asarray(r.value))
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
        seg_chain=False):
    """h_conf: pairs whose nominals overlap (closer than delta + h at some
    timestep) get the larger target delta + h_conf, creating smoothing margin
    exactly at the kinks; all other pairs keep delta + h."""
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    sc0 = scenario.generate("random_seq", M=M, n_points=8, n_agents=3 * M,
                            seed=seed, nu_deg=60, N=K0, tf=tf,
                            ring_radius=ring, max_tries=5000)
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc0.payloads]
    margin = sc0.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    rho = [e + margin / 2 + h / 2 for e in ext]
    L = [link_rho * r for r in rho]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    row = dict(M=M, seed=seed, alpha=alpha, link_rho=link_rho)

    sc, psis, rbar, LB0 = nominals(sc0, K0)
    hg = h if gauss_h is None else gauss_h
    rho_g = [e + margin / 2 + hg / 2 for e in ext]
    tg0 = {(j, l): rho_g[j] + rho_g[l] for (j, l) in pairs}
    row["gauss_h"] = hg
    xg, _, _ = relax_iso(rbar, pairs, tg0, K0, M)
    if xg is not None:
        if seg_safe:
            Jg, rsg = seg_tubes(sc, psis, [np.asarray(v) for v in xg], pairs,
                                deltas, K0, M)
            if Jg is None:
                row["gauss_status"] = rsg
        else:
            Jg, rsg, _ = tubes_detail(sc, psis, pairs, deltas, xg, K0, M)
        row["gap_gauss_pct"] = round(100 * (Jg - LB0) / LB0, 2) if Jg else None
        if Jg:
            cw, cb, cn = cont_margin(rsg, pairs, deltas)
            row["gauss_cont_min_margin"] = round(cw, 3)
            row["gauss_cont_violations"] = f"{cb}/{cn}"

    t0 = time.perf_counter()
    K = K0
    while not links_ok(nominals(sc0, K)[2], L):
        K *= 2
    attempts = {}
    while True:
        sck, psk, rbk, LBk = nominals(sc0, K)
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
                                    Dm=(deltas if seg_chain else None))
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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = []
    for hc in a.h_confs:
     for reach in a.reaches:
      for alpha in a.alphas:
        for M in a.Ms:
            for s in a.seeds:
                try:
                    row = run(M, s, alpha, a.link_rho, reach=reach, h=a.h,
                              h_conf=hc, smooth=a.smooth,
                              dyn_smooth=a.dyn_smooth,
                              post_smooth=a.post_smooth, qp=a.qp_polish,
                              failsafe=a.failsafe, seg_safe=a.seg_safe,
                              gauss_h=a.gauss_h, split=a.split_planes,
                              split_rounds=a.split_rounds,
                              split_D=(None if a.split_D == 0 else a.split_D),
                              seg_chain=a.seg_chain)
                    row["smooth"] = "dyn" if a.dyn_smooth else a.smooth
                    row["post_smooth"] = a.post_smooth
                    row["h"] = a.h
                    row["h_conf"] = hc
                except Exception as exc:
                    row = dict(M=M, seed=s, alpha=alpha,
                               status=f"error: {str(exc)[:90]}")
                row["reach"] = reach if np.isfinite(reach) else "all"
                out.append(row)
                print(json.dumps(row), flush=True)
                name = (f"expand_method_rho{a.link_rho:g}_h{a.h:g}"
                        + ("_hconf" if a.h_confs != [None] else "")
                        + ("_smooth" if a.smooth else "")
                        + (f"_dyn{a.post_smooth}" if a.dyn_smooth else "")
                        + ("_qp" if a.qp_polish else "") + ".json")
                if a.out:
                    name = a.out
                p = Path(__file__).with_name("results") / name
                p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
