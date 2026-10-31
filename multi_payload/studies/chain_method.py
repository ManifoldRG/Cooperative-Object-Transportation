"""Chain method (user's): deconfliction as a system of rigid-link chains.

Spheres: payload j carries radius rho_j = ext_j + margin/2 + h/2 at every
timestep, so two spheres are disjoint iff same-step separation >= delta_jl + h
(the same target as the gauss relaxation, for a like-for-like comparison).

Rigid Link constraint: consecutive points of one chain are at most one sphere
diameter apart, ||x_j[k+1] - x_j[k]|| <= 2 rho_j (a maximum, so links may
shorten): every chain stays a connected sausage.

1. Nominal resolution: plan nominals at K steps; double K until every nominal
   satisfies the links, then bisect for the smallest such K.
2. Deconfliction at a given K: repeat
     - repulsion: every same-step overlapping pair is pushed apart by exactly
       its overlap (half each), sequentially over pairs;
     - links: red-black projection of every over-long link back to 2 rho
       (half each; a pinned end makes the free end take it all), iterated
       to convergence so the Rigid Link constraint holds after every round;
   until no overlap (success) or the total overlap stops decreasing (stuck).
3. Stuck -> replan nominals at 2K, 4K, ... until deconfliction succeeds, then
   bisect between the last failure and the first success.
4. Hand-off: half-split tube radii from the deconflicted chains, decoupled
   convex re-plans on the same K grid; certified gap vs the LB at that K.

Fixed points (avoiding trivial violations): steps {0, 1, K} are pinned by the
boundary conditions (v0 = 0 makes r1 = r0). The grid is shared, so at any
step both points of a pair are pinned or both are free: pinned steps are
excluded from repulsion and from the residual, and are instead checked once
against the true clearance delta_jl (start/goal placement guarantees it). The
link from step 0 to step 1 has length zero in every nominal.

Usage: py chain_method.py [--Ms 20 30] [--seeds 0 1]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_methods import relax_iso  # noqa: E402
from benchmark_tubes import tubes_detail  # noqa: E402
from multi_payload import scenario, translation  # noqa: E402

HEADROOM = 60.0


def nominals(sc0, K):
    sc = dataclasses.replace(sc0, N=K)
    psis = translation.psi_list(sc)
    refs = translation.solve_all(sc, psis=psis)
    if not all(r["ok"] for r in refs):
        raise RuntimeError(f"nominal solve failed at K={K}")
    LB = sum(r["fuel"] for r in refs)
    return sc, psis, [np.asarray(r["r"]) for r in refs], LB


def links_ok(x, L, tol=1e-6):
    return all(np.max(np.linalg.norm(np.diff(x[j], axis=0), axis=1)) <= L[j] + tol
               for j in range(len(x)))


def enforce_links(xj, Lj, free, passes=60, tol=1e-3):
    """Red-black projection of over-long links back to length Lj."""
    K = len(xj) - 1
    for _ in range(passes):
        worst = 0.0
        for par in (0, 1):
            i = np.arange(par, K, 2)
            d = xj[i + 1] - xj[i]
            ln = np.linalg.norm(d, axis=1)
            ex = ln - Lj
            m = ex > 0
            if not m.any():
                continue
            worst = max(worst, float(ex[m].max()))
            i, d, ln, ex = i[m], d[m], ln[m], ex[m]
            corr = (ex / np.maximum(ln, 1e-12))[:, None] * d
            fa, fb = free[i], free[i + 1]
            wa = np.where(fa & fb, 0.5, np.where(fa, 1.0, 0.0))
            wb = np.where(fa & fb, 0.5, np.where(fb, 1.0, 0.0))
            xj[i] += wa[:, None] * corr
            xj[i + 1] -= wb[:, None] * corr
        if worst < tol:
            break
    return xj


def _blocked(Pj, others_pos, thr2, idx, D):
    """For chain points idx moved by D (n,3): True where the move would bring
    the point closer to some other payload's same-step point AND leave it
    below that pair's target (create or deepen an overlap)."""
    cur = Pj[idx]
    oth = others_pos[:, idx]                       # (M-1,n,3)
    before = np.sum((cur[None] - oth) ** 2, axis=2)
    after = np.sum((cur[None] + D[None] - oth) ** 2, axis=2)
    return np.any((after < before - 1e-9) & (after < thr2[:, idx]), axis=0)


def enforce_links_guarded(xj, Lj, free, others_pos, thr2, passes=60,
                          tol=1e-3):
    """Rigid Link projection that never creates or deepens an overlap. Each
    over-long link is shortened half by each end; an end whose move is
    blocked hands its share to the other end; if both are blocked the link
    stays over-long this pass. Returns (xj, worst remaining excess)."""
    K = len(xj) - 1
    worst = 0.0
    for _ in range(passes):
        worst, moved = 0.0, False
        for par in (0, 1):
            i = np.arange(par, K, 2)
            d = xj[i + 1] - xj[i]
            ln = np.linalg.norm(d, axis=1)
            ex = ln - Lj
            m = ex > tol
            if not m.any():
                continue
            for a, dd, l_, e_ in zip(i[m], d[m], ln[m], ex[m]):
                b = a + 1
                corr = (e_ / max(l_, 1e-12)) * dd   # move a by +corr, b by -corr
                fa, fb = free[a], free[b]
                opts = []
                if fa and fb:
                    opts.append((0.5, 0.5))
                if fb:
                    opts.append((0.0, 1.0))
                if fa:
                    opts.append((1.0, 0.0))
                done = False
                for wa, wb in opts:
                    ok = True
                    if wa and _blocked(xj, others_pos, thr2, np.array([a]),
                                       (wa * corr)[None])[0]:
                        ok = False
                    if ok and wb and _blocked(xj, others_pos, thr2, np.array([b]),
                                              (-wb * corr)[None])[0]:
                        ok = False
                    if ok:
                        xj[a] += wa * corr
                        xj[b] -= wb * corr
                        done = moved = True
                        break
                if not done:
                    worst = max(worst, float(e_))
        if not moved:
            break                       # nothing left to fix, or all blocked
    return xj, worst


def turn_angles(xj):
    """Turning angle at each joint k=1..K-1 (0 = straight); zero-length
    links (the pinned 0-1 link) count as straight."""
    a = xj[1:-1] - xj[:-2]
    b = xj[2:] - xj[1:-1]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 1e-9) & (nb > 1e-9)
    c = np.ones(len(a))
    c[ok] = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return np.arccos(np.clip(c, -1.0, 1.0))


def enforce_bends(xj, theta, free, passes=20, tol=1e-3):
    """Hard maximum turning angle per joint, as a rod-style bend correction.
    The joint point's deflection from its neighbors' chord is h = a b sin(t)/c
    (t = turning angle, a, b = link lengths, c = chord); the angle is within
    theta iff h <= a b sin(theta) / c_theta with c_theta^2 = a^2 + b^2 +
    2 a b cos(theta). An excess d is removed by moving the joint toward the
    chord and its free neighbors the opposite way, with corrections summing
    to zero (the three points' mean stays put), so straightening a joint does
    not fling the chain outward. Joints two apart share a point: two colors."""
    K = len(xj) - 1
    cth, sth = np.cos(theta), np.sin(theta)
    ks = np.arange(1, K)
    groups = [ks[(ks % 4) < 2], ks[(ks % 4) >= 2]]
    for _ in range(passes):
        worst = 0.0
        for k in groups:
            k = k[free[k]]
            if not len(k):
                continue
            p, q, r = xj[k - 1], xj[k], xj[k + 1]
            ch = r - p
            c = np.linalg.norm(ch, axis=1)
            ok = c > 1e-9
            u = np.zeros_like(ch)
            u[ok] = ch[ok] / c[ok][:, None]
            off = (q - p) - np.sum((q - p) * u, axis=1)[:, None] * u
            h = np.linalg.norm(off, axis=1)
            a = np.linalg.norm(q - p, axis=1)
            b = np.linalg.norm(r - q, axis=1)
            ct = np.sqrt(a * a + b * b + 2 * a * b * cth)
            hmax = np.where(ct > 1e-9, a * b * sth / np.maximum(ct, 1e-9), 0.0)
            ex = h - hmax
            m = ok & (h > 1e-9) & (ex > 1e-9)
            if not m.any():
                continue
            worst = max(worst, float(ex[m].max()))
            k, ex, n = k[m], ex[m], off[m] / h[m][:, None]
            fa, fb = free[k - 1], free[k + 1]
            nf = 1.0 + fa + fb                      # free points sharing it
            wq = np.where(nf > 1, (fa + fb) / nf, 1.0)   # joint share
            wn = 1.0 / nf                           # each free neighbor share
            xj[k] -= (wq * ex)[:, None] * n
            xj[k - 1] += (np.where(fa, wn, 0.0) * ex)[:, None] * n
            xj[k + 1] += (np.where(fb, wn, 0.0) * ex)[:, None] * n
        if worst < tol:
            break
    return xj


def step_targets(rbar, pairs, deltas, L, h):
    """Per-step targets that never demand the unreachable near pinned steps.
    The headroom is only for corridor width; safety needs delta_jl, which the
    pinned start/goal already satisfy. A pair can open its gap by at most
    L_j + L_l per link away from a pinned step, so the target at step k is
    min(delta+h, max(delta, reachable from start), max(delta, from goal))."""
    K = len(rbar[0]) - 1
    k = np.arange(K + 1)
    out = {}
    for (j, l) in pairs:
        d1 = float(np.linalg.norm(rbar[j][1] - rbar[l][1]))
        dK = float(np.linalg.norm(rbar[j][K] - rbar[l][K]))
        grow = L[j] + L[l]
        reach_s = d1 + grow * np.maximum(k - 1, 0)
        reach_g = dK + grow * (K - k)
        dl = deltas[j][l]
        out[(j, l)] = np.minimum(dl + h, np.minimum(np.maximum(dl, reach_s),
                                                    np.maximum(dl, reach_g)))
    return out


def _project_cone(v, normals):
    """Project unit v into {d : d.n >= 0 for all n} (cyclic halfspace
    projections). Unit vector, or None if the projection vanishes."""
    if not len(normals):
        return v
    for _ in range(12):
        dots = normals @ v
        i = int(np.argmin(dots))
        if dots[i] >= -1e-9:
            return v
        v = v - dots[i] * normals[i]
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            return None
        v = v / nv
    return v if (normals @ v).min() >= -1e-6 else None


def _cone_push(x, j, k, g, mag, T, M):
    """Push (j,k) by mag along g projected into its 'away from every nearby
    point' cone. Nearby = other payloads' same-step points that this step
    could bring below their target. No eligible direction: hold."""
    p = x[j][k]
    normals = []
    for m in range(M):
        if m == j:
            continue
        d = p - x[m][k]
        dn = float(np.linalg.norm(d))
        if dn < T[j, m, k] + mag:
            normals.append(d / max(dn, 1e-9))
    v = _project_cone(g, np.array(normals) if normals else np.zeros((0, 3)))
    if v is None:
        return False
    x[j][k] = p + mag * v
    return True


def _clipped(x, pairs, tgt, sl):
    """min(distance, target) for every pair at every free step."""
    return np.concatenate([np.minimum(np.linalg.norm(x[j][sl] - x[l][sl], axis=1),
                                      tgt[(j, l)][sl]) for (j, l) in pairs])


def deconflict(rbar, pairs, tgt, L, max_iters=4000, window=200, theta=None,
               h=HEADROOM, guard=False, stats=None, cone=False):
    """Chain dynamics at a fixed K. Returns (x or None, iters, residual).
    guard=True: link projections never create or deepen an overlap.
    stats (dict, optional) collects monotonicity counts: how many (pair,
    step) clipped distances min(d, target) decreased during the repulsion
    step and during the link step."""
    M = len(rbar)
    K = len(rbar[0]) - 1
    x = [r.copy() for r in rbar]
    free = np.ones(K + 1, bool)
    free[[0, 1, K]] = False
    sl = slice(2, K)
    best, best_it = np.inf, 0
    if guard or cone:
        T = np.zeros((M, M, K + 1))
        for (j, l), v in tgt.items():
            T[j, l] = T[l, j] = v
        T2 = T ** 2
    holds = 0
    if stats is not None:
        stats.update(dec_repulsion=0, dec_links=0, max_dec=0.0,
                     blocked_excess=0.0)
    for it in range(max_iters):
        if stats is not None:
            c0 = _clipped(x, pairs, tgt, sl)
        worst, total = 0.0, 0.0
        for (j, l) in pairs:
            d = x[j][sl] - x[l][sl]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            need = tgt[(j, l)][sl] - dist
            m = need > 0
            if m.any():
                worst = max(worst, float(need[m].max()))
                total += float(need[m].sum())
                if cone:
                    # each side moves along the repulsion direction projected
                    # into its 'away from every nearby point' cone; no
                    # eligible direction -> that point holds this round
                    for i in np.nonzero(m)[0]:
                        k = i + 2
                        dd = x[j][k] - x[l][k]
                        dn = max(float(np.linalg.norm(dd)), 1e-9)
                        nd = tgt[(j, l)][k] - dn
                        if nd <= 0:
                            continue
                        g = dd / dn
                        holds += not _cone_push(x, j, k, g, 0.5 * nd, T, M)
                        holds += not _cone_push(x, l, k, -g, 0.5 * nd, T, M)
                else:
                    g = d[m] / dist[m][:, None]
                    x[j][sl][m] += 0.5 * need[m][:, None] * g
                    x[l][sl][m] -= 0.5 * need[m][:, None] * g
        if stats is not None:
            stats["holds"] = holds
            c1 = _clipped(x, pairs, tgt, sl)
            stats["dec_repulsion"] += int(np.sum(c1 < c0 - 1e-6))
            stats["max_dec"] = max(stats["max_dec"], float(np.max(c0 - c1)))
        if guard:
            P = np.array(x)
        for j in range(M):
            if guard:
                oth = np.arange(M) != j
                x[j], ex = enforce_links_guarded(x[j], L[j], free, P[oth],
                                                 T2[j][oth])
                P[j] = x[j]
                if stats is not None:
                    stats["blocked_excess"] = max(stats["blocked_excess"], ex)
            elif theta is None:
                x[j] = enforce_links(x[j], L[j], free)
            else:
                for _ in range(6):   # links and bends, alternated
                    x[j] = enforce_bends(x[j], theta, free)
                    x[j] = enforce_links(x[j], L[j], free)
        if stats is not None:
            c2 = _clipped(x, pairs, tgt, sl)
            stats["dec_links"] += int(np.sum(c2 < c1 - 1e-6))
            stats["max_dec"] = max(stats["max_dec"], float(np.max(c1 - c2)))
        # residual measured AFTER the chain constraints are restored
        worst, total = 0.0, 0.0
        for (j, l) in pairs:
            dist = np.linalg.norm(x[j][sl] - x[l][sl], axis=1)
            need = tgt[(j, l)][sl] - dist
            if (need > 0).any():
                worst = max(worst, float(need.max()))
                total += float(need[need > 0].sum())
        if worst < 0.1:
            return x, it + 1, worst
        if total < best - 0.05:
            best, best_it = total, it
        elif it - best_it >= window:
            # equilibrium with overlap. If every gap still keeps at least half
            # the headroom, the corridors are merely narrower there: accept.
            # Only jams that squeeze the corridors trigger refinement.
            if worst <= h / 2:
                return x, it + 1, worst
            return None, it + 1, worst
    return None, max_iters, worst


def run(M, seed, K0=30, tf=500.0, Kmax=480, Kmin=30, link_scale=None,
        link_rho=2.0, bend_deg=None, h=HEADROOM, guard=False, cone=False):
    theta = None if bend_deg is None else np.deg2rad(bend_deg)
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    sc0 = scenario.generate("random_seq", M=M, n_points=8, n_agents=3 * M,
                            seed=seed, nu_deg=60, N=K0, tf=tf,
                            ring_radius=ring, max_tries=5000)
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc0.payloads]
    margin = sc0.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    rho = [e + margin / 2 + h / 2 for e in ext]
    # Rigid Link: ||x[k+1]-x[k]|| <= link_rho * rho_j (2 = spheres touching,
    # 1 = each sphere reaches its neighbor's center, 0.5 = heavy overlap)
    L = [link_rho * r for r in rho]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    tgt = {(j, l): rho[j] + rho[l] for (j, l) in pairs}
    row = dict(M=M, seed=seed)
    t_start = time.perf_counter()

    # pinned steps: true clearance must already hold (placement)
    _, _, rb0, _ = nominals(sc0, K0)
    for (j, l) in pairs:
        for k in (0, K0):
            if np.linalg.norm(rb0[j][k] - rb0[l][k]) < deltas[j][l]:
                row["status"] = "pinned endpoints closer than delta"
                return row

    # gauss baseline on the same instance at K0
    sc, psis, rbar, LB0 = nominals(sc0, K0)
    xg, _, _ = relax_iso(rbar, pairs, tgt, K0, M)
    if xg is not None:
        Jg, _, _ = tubes_detail(sc, psis, pairs, deltas, xg, K0, M)
        row["gap_gauss_pct"] = round(100 * (Jg - LB0) / LB0, 2) if Jg else None

    # 1. nominal resolution for the Rigid Link constraint
    cache = {}

    def nom(K):
        if K not in cache:
            cache[K] = nominals(sc0, K)
        return cache[K]

    def chain_ok(x):
        if not links_ok(x, L):
            return False
        if theta is None:
            return True
        return all(turn_angles(xj).max() <= theta + 1e-6 for xj in x)

    K = K0
    if chain_ok(nom(K)[2]):
        lo, hi = Kmin - 1, K    # search downward, never below Kmin
        if Kmin >= K:
            lo = K - 1
    else:
        while not chain_ok(nom(K)[2]):
            if K >= Kmax:
                row["status"] = "links unsatisfiable up to Kmax"
                return row
            K *= 2
        lo, hi = K // 2, K
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if chain_ok(nom(mid)[2]):
            hi = mid
        else:
            lo = mid
    K_link = hi
    row["K_link"] = K_link

    # 2-3. deconfliction with doubling, then bisection
    attempts = {}
    mono = {}
    Lused = {}

    def attempt(K):
        if K not in attempts:
            sck, psk, rbk, LBk = nom(K)
            Lk = L
            if link_scale is not None:
                # tight chain: links at most link_scale x the nominal's own
                # longest link, so pushes propagate along the chain
                Lk = [link_scale * float(np.max(np.linalg.norm(
                    np.diff(rbk[j], axis=0), axis=1))) for j in range(M)]
            tg = step_targets(rbk, pairs, deltas, Lk, h)
            st = {}
            x, iters, res = deconflict(rbk, pairs, tg, Lk, theta=theta, h=h,
                                       guard=guard, stats=st, cone=cone)
            mono[K] = st
            Lused[K] = Lk
            attempts[K] = (x, iters, res)
        return attempts[K]

    K = K_link
    fail_K = None
    while attempt(K)[0] is None:
        fail_K = K
        if K >= Kmax:
            row["status"] = f"stuck up to Kmax (resid {attempt(K)[2]:.1f})"
            row["attempts"] = {k: [v[1], round(v[2], 1)] for k, v in attempts.items()}
            return row
        K *= 2
    if fail_K is not None:
        lo, hi = fail_K, K
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if attempt(mid)[0] is not None:
                hi = mid
            else:
                lo = mid
        K = hi
    x = attempt(K)[0]
    row["K_final"] = K
    row["attempts"] = {k: [v[1], round(v[2], 1)] for k, v in sorted(attempts.items())}
    row["t_chain"] = round(time.perf_counter() - t_start, 1)

    # 4. hand-off: half-split tubes + decoupled convex re-plans at K
    sck, psk, rbk, LBk = nom(K)
    Jc, rs, _ = tubes_detail(sck, psk, pairs, deltas, x, K, M)
    if Jc is None:
        row["status"] = "tube re-plan infeasible"
        return row
    row["gap_chain_pct"] = round(100 * (Jc - LBk) / LBk, 2)
    row["monotone"] = {k: (round(v, 2) if isinstance(v, float) else v)
                       for k, v in mono[K].items()}
    row["max_link_over_limit"] = round(max(
        float(np.max(np.linalg.norm(np.diff(x[j], axis=0), axis=1))) / Lused[K][j]
        for j in range(M)), 2)
    row["max_dev"] = round(max(float(np.max(np.linalg.norm(x[j] - rbk[j], axis=1)))
                               for j in range(M)), 0)
    row["min_margin"] = round(min(
        float(np.min(np.linalg.norm(rs[j] - rs[l], axis=1))) - deltas[j][l]
        for (j, l) in pairs), 1)
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+", default=[20, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--Kmin", type=int, default=30,
                    help="floor for the downward bisection (base resolution)")
    ap.add_argument("--link-scale", type=float, default=None,
                    help="tight-chain variant: link max = scale x nominal link")
    ap.add_argument("--link-rho", type=float, default=2.0,
                    help="Rigid Link max = link_rho x sphere radius rho_j")
    ap.add_argument("--bend-deg", type=float, default=None,
                    help="hard max turning angle per joint (degrees)")
    ap.add_argument("--Kmax", type=int, default=480)
    ap.add_argument("--h", type=float, default=HEADROOM,
                    help="headroom beyond delta_jl (corridor width >= h/2)")
    ap.add_argument("--guard-links", action="store_true",
                    help="link projections never create or deepen an overlap")
    ap.add_argument("--cone-repulsion", action="store_true",
                    help="repulsion only along directions away from every "
                         "nearby point; none eligible -> hold")
    a = ap.parse_args()
    out = []
    for M in a.Ms:
        for s in a.seeds:
            try:
                r = run(M, s, Kmin=a.Kmin, link_scale=a.link_scale,
                        link_rho=a.link_rho, bend_deg=a.bend_deg,
                        Kmax=a.Kmax, h=a.h, guard=a.guard_links,
                        cone=a.cone_repulsion)
                r["h"] = a.h
                r["guard"] = a.guard_links
                r["cone"] = a.cone_repulsion
                r["bend_deg"] = a.bend_deg
                r["link_scale"] = a.link_scale
                r["link_rho"] = a.link_rho
            except Exception as exc:
                r = dict(M=M, seed=s, status=f"error: {str(exc)[:100]}")
            out.append(r)
            print(json.dumps(r), flush=True)
    tag = (f"scale{a.link_scale}" if a.link_scale is not None
           else f"rho{a.link_rho}")
    if a.bend_deg is not None:
        tag += f"_bend{a.bend_deg:g}"
    tag += (f"_h{a.h:g}" + ("_guard" if a.guard_links else "")
            + ("_cone" if a.cone_repulsion else ""))
    p = Path(__file__).with_name("results") / f"chain_method_{tag}.json"
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
