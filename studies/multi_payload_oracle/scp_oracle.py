"""Full-pose SCP oracle: successive convexification over translation AND
attitude, with pairwise separating planes updated per iteration.

Per payload the body-frame convention makes cones and the torque map
attitude-independent; the attitude enters only the translational force
coupling R_k sum(u) and the attitude kinematics. Each outer iteration:
  1. update pairwise collision planes from the current positions,
  2. per payload, solve one SOCP linearized about the current attitude
     iterate (error state dphi, frozen-u cross terms, trust region),
  3. re-roll the attitude nonlinearly from the solved torques,
  4. keep the best independently rollout-verified solution.
Warm start: prescribed attitude + alternation translational references +
allocator thrusts. Certificate: gap to the collision-free translational
lower bound LB = sum ell_j.

Usage: py scp_oracle.py [--M 5] [--seeds 3 5 7] [--N 30] [--tf 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import route2_multi as r2  # noqa: E402  (adds COT repo to sys.path)
import sandwich_check as sw  # noqa: E402

import cvxpy as cp  # noqa: E402

from multi_payload import allocation, attitude  # noqa: E402
from spacecraft_libraries.new_opts import (  # noqa: E402
    so3_exp, so3_log, tau_proj_nonlin_new)


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0.0]])


def att_rollout(sc, j, tau):
    """Exact nonlinear attitude rollout from x0 under torques tau (N,3)."""
    P0 = sc.payloads[j]
    I = np.asarray(P0.I, float)
    Ii = np.linalg.inv(I)
    R = so3_exp(np.asarray(P0.x0.phi, float))
    w = np.asarray(P0.x0.omega, float).copy()
    Rs, ws = [R], [w.copy()]
    for k in range(sc.N):
        R = R @ so3_exp(sc.dt * w)
        w = w + sc.dt * (Ii @ (tau[k] - np.cross(w, I @ w)))
        Rs.append(R)
        ws.append(w.copy())
    return Rs, np.asarray(ws)


def payload_scp_step(sc, psis, j, pts, Rbar, wbar, Ubar, etas_j, tr_phi=0.3):
    """One linearized SOCP for payload j about (Rbar, wbar, Ubar).

    etas_j: list of (eta (N+1,3), other_r (N+1,3)) separating-plane rows.
    Returns dict(U, r, v, tau, dphi, fuel) or None.
    """
    N, dt = sc.N, sc.dt
    P0 = sc.payloads[j]
    I = np.asarray(P0.I, float)
    Ii = np.linalg.inv(I)
    A = len(pts)
    U = [cp.Variable((N, 3)) for _ in range(A)]
    r = cp.Variable((N + 1, 3))
    v = cp.Variable((N + 1, 3))
    dphi = cp.Variable((N + 1, 3))
    w = cp.Variable((N + 1, 3))
    cons = [r[0] == P0.x0.r, v[0] == P0.x0.v, r[N] == P0.xf.r, v[N] == P0.xf.v,
            dphi[0] == 0, dphi[N] == 0,
            w[0] == P0.x0.omega, w[N] == P0.xf.omega,
            cp.norm(dphi, axis=1) <= tr_phi]
    cosnu = float(np.cos(sc.nu))
    taus = []
    for k in range(N):
        P = psis[k]
        thrust = sum(U[p][k] for p in range(A))
        ubar = Ubar[:, k, :].sum(axis=0)
        # translation: R_k f ~ Rbar f + Rbar (dphi x ubar)
        force = Rbar[k] @ thrust - (Rbar[k] @ skew(ubar)) @ dphi[k]
        cons += [r[k + 1] == r[k] + dt * v[k],
                 v[k + 1] == v[k] + dt * (P[:, :3] @ r[k] + P[:, 3:] @ v[k]
                                          + force / P0.m)]
        # attitude error-state kinematics about (Rbar, wbar)
        cons.append(dphi[k + 1] == (np.eye(3) - dt * skew(wbar[k])) @ dphi[k]
                    + dt * (w[k] - wbar[k]))
        # omega dynamics, gyroscopic term linearized about wbar
        tau = sum(skew(pts[p]) @ U[p][k] for p in range(A))
        taus.append(tau)
        gyro = (np.cross(wbar[k], I @ wbar[k])
                + (skew(wbar[k]) @ I - skew(I @ wbar[k])) @ (w[k] - wbar[k]))
        cons.append(w[k + 1] == w[k] + dt * (Ii @ tau - Ii @ gyro))
        # exact body-frame cones
        for p in range(A):
            ph = pts[p] / np.linalg.norm(pts[p])
            cons.append(cosnu * cp.norm(U[p][k]) <= -(ph @ U[p][k]))
    for eta, other in etas_j:
        for k in range(N + 1):
            cons.append(eta[k] @ (r[k] - other[k]) >= sc.delta)
    prob = cp.Problem(cp.Minimize(sum(cp.sum(cp.norm(Uu, 2, axis=1)) for Uu in U)),
                      cons)
    try:
        prob.solve(solver="CLARABEL")
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    Uv = np.stack([np.asarray(Uu.value) for Uu in U])
    return dict(U=Uv, r=np.asarray(r.value), v=np.asarray(v.value),
                tau=np.stack([np.asarray(sum(skew(pts[p]) @ Uv[p][k]
                                             for p in range(A)))
                              for k in range(sc.N)]),
                dphi=np.asarray(dphi.value), fuel=float(prob.value))


def joint_polish(sc, psis, pts_all, Rbars, taus, pairs, etas):
    """Exact convex fuel polish, ALL payloads jointly: attitude FIXED per
    payload (torque pinned to taus[j] so the nonlinear rollout Rbars[j] stays
    valid), translation exact with the known Rbars, cones exact, separating
    planes between the VARIABLE positions of both payloads (so the Euclidean
    separation >= delta holds by construction, ||eta|| = 1).
    Returns list of dict(U, r) or None.
    """
    N, dt = sc.N, sc.dt
    cosnu = float(np.cos(sc.nu))
    var, cons = [], []
    for j in range(sc.M):
        P0 = sc.payloads[j]
        pts = pts_all[j]
        A = len(pts)
        U = [cp.Variable((N, 3)) for _ in range(A)]
        r = cp.Variable((N + 1, 3))
        v = cp.Variable((N + 1, 3))
        var.append((U, r))
        cons += [r[0] == P0.x0.r, v[0] == P0.x0.v,
                 r[N] == P0.xf.r, v[N] == P0.xf.v]
        for k in range(N):
            P = psis[k]
            thrust = sum(U[p][k] for p in range(A))
            cons += [r[k + 1] == r[k] + dt * v[k],
                     v[k + 1] == v[k] + dt * (P[:, :3] @ r[k] + P[:, 3:] @ v[k]
                                              + (Rbars[j][k] @ thrust) / P0.m),
                     sum(skew(pts[p]) @ U[p][k] for p in range(A)) == taus[j][k]]
            for p in range(A):
                ph = pts[p] / np.linalg.norm(pts[p])
                cons.append(cosnu * cp.norm(U[p][k]) <= -(ph @ U[p][k]))
    for (j, l) in pairs:
        for k in range(N + 1):
            cons.append(etas[(j, l)][k] @ (var[j][1][k] - var[l][1][k])
                        >= sc.delta)
    obj = sum(cp.sum(cp.norm(Uu, 2, axis=1)) for U, _ in var for Uu in U)
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver="CLARABEL")
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return [dict(U=np.stack([np.asarray(Uu.value) for Uu in U]),
                 r=np.asarray(r.value)) for U, r in var]


def verify(sc, psis, allocs_pts, Us):
    """Independent rollout of ALL payloads; returns (fuel, maxV, min_sep)."""
    hist, fuel, maxV = [], 0.0, 0.0
    for j, (pts, Uv) in enumerate(zip(allocs_pts, Us)):
        P0 = sc.payloads[j]
        I = np.asarray(P0.I, float)
        Ii = np.linalg.inv(I)
        rj = np.asarray(P0.x0.r, float).copy()
        vj = np.asarray(P0.x0.v, float).copy()
        R = so3_exp(np.asarray(P0.x0.phi, float))
        w = np.asarray(P0.x0.omega, float).copy()
        h = [rj.copy()]
        for k in range(sc.N):
            thrust = Uv[:, k, :].sum(axis=0)
            tau = sum(np.cross(pts[p], Uv[p, k]) for p in range(len(pts)))
            P = np.asarray(psis[k])
            rj, vj = rj + sc.dt * vj, vj + sc.dt * (P[:, :3] @ rj + P[:, 3:] @ vj
                                                    + (R @ thrust) / P0.m)
            R = R @ so3_exp(sc.dt * w)
            w = w + sc.dt * (Ii @ (tau - np.cross(w, I @ w)))
            h.append(rj.copy())
        Rf = so3_exp(np.asarray(P0.xf.phi, float))
        V = (np.linalg.norm(rj - P0.xf.r) + np.linalg.norm(vj - P0.xf.v)
             + np.linalg.norm(so3_log(R.T @ Rf)) + np.linalg.norm(w - P0.xf.omega))
        maxV = max(maxV, float(V))
        fuel += float(np.linalg.norm(Uv, axis=2).sum())
        hist.append(np.asarray(h))
    min_sep = min(np.linalg.norm(hist[j][k] - hist[l][k])
                  for j in range(sc.M) for l in range(j + 1, sc.M)
                  for k in range(sc.N + 1))
    return fuel, maxV, min_sep, hist


def run(seed, N, tf, M, iters=40, V_tol=0.05, collision=True, return_sol=False):
    sc, psis, refs0, pairs, gaps, n_conf = r2.crossing(seed, N, tf, M, n_points=8)
    LB = sum(r["fuel"] for r in refs0)
    row = dict(seed=seed, M=M, agents=sc.n_agents, planes=collision,
               conflicting_pairs=f"{n_conf}/{len(pairs)}", LB=round(LB, 2))
    if not collision:
        pairs = []  # diagnostic: same SCP with the collision constraints off

    t0 = time.time()
    # closure allocation, alternation references, prescribed attitudes
    al_basis = allocation.build_allocators(sc, [sw.BASIS] * M)
    occ = [sorted(o) for o in allocation.greedy(al_basis, sc.n_agents)["occupied"]]
    pts = [[sc.payloads[j].points[p] for p in occ[j]] for j in range(M)]
    if collision:
        best_alt = sw.alternate_full(sc, psis, pairs, refs0)
        if best_alt is None:
            row["status"] = "alternation failed"
            return row
        _, rs, Fs = best_alt
    else:
        rs = [np.asarray(refs0[j]["r"]) for j in range(M)]
        Fs = [np.asarray(refs0[j]["F"]) for j in range(M)]
    atts = [attitude.attitude_reference(sc, j, seed=seed) for j in range(M)]
    W_act = [attitude.required_wrenches(dict(F=Fs[j]), atts[j]) for j in range(M)]
    al_act = allocation.build_allocators(sc, W_act)
    Us = []
    for j in range(M):
        _, e, Uall = al_act[j].evaluate(set(occ[j]))
        if e > 0:
            row["status"] = f"payload {j} not closed"
            return row
        Us.append(Uall[:, : sc.N, :])
    Rbars = [atts[j]["Rs"] for j in range(M)]
    wbars = [atts[j]["omes"] for j in range(M)]
    pos = [np.asarray(r) for r in rs]
    row["warm_time"] = round(time.time() - t0, 1)

    def planes_from(positions):
        etas = {}
        for (j, l) in pairs:
            e = np.zeros((N + 1, 3))
            for k in range(N + 1):
                d = positions[j][k] - positions[l][k]
                n = np.linalg.norm(d)
                e[k] = d / n if n > 1e-9 else np.array([1.0, 0, 0])
            etas[(j, l)] = e
        return etas

    t0 = time.time()
    best = None
    hist_iter = []
    tr, no_improve = 0.3, 0
    for it in range(iters):
        # linearize about the last ACCEPTED point (planes from its positions)
        etas = planes_from(pos)
        sols = []
        for j in range(M):
            ej = ([(etas[(j, l)], pos[l]) for (jj, l) in pairs if jj == j]
                  + [(-etas[(l, j)], pos[l]) for (l, jj) in pairs if jj == j])
            s = payload_scp_step(sc, psis, j, pts[j], Rbars[j], wbars[j], Us[j],
                                 ej, tr_phi=tr)
            if s is None:
                sols = None
                break
            sols.append(s)
        if sols is None:
            break
        # candidate: project tau onto the attitude BCs, re-roll, joint polish
        cR, cw = [None] * M, [None] * M
        for j in range(M):
            sp, bc = sc.sys_params_for(j)
            proj, _ = tau_proj_nonlin_new(sols[j]["tau"], sc.N, sc.epsilon, sp, bc)
            if proj is not None:
                sols[j]["tau"] = np.asarray(proj).reshape(sc.N, 3)
            cR[j], cw[j] = att_rollout(sc, j, sols[j]["tau"])
        cUs = [s["U"] for s in sols]
        cpos = [s["r"] for s in sols]
        pol = joint_polish(sc, psis, pts, cR, [s["tau"] for s in sols],
                           pairs, planes_from(cpos))
        if pol is not None:
            cUs = [p["U"] for p in pol]
            cpos = [p["r"] for p in pol]
        fuel, maxV, min_sep, _ = verify(sc, psis, pts, cUs)
        ok = bool(maxV < V_tol and (not collision or min_sep >= sc.delta - 1e-3))
        improved = ok and (best is None or fuel < best[0] - 1e-6)
        hist_iter.append(dict(it=it, fuel=round(fuel, 2), maxV=round(maxV, 4),
                              min_sep=round(min_sep, 1), tr=round(tr, 4),
                              pol_fail=0 if pol is not None else 1, acc=improved))
        if improved:
            best = (fuel, maxV, min_sep,
                    [U.copy() for U in cUs],
                    [[np.array(R) for R in Rb] for Rb in cR])
            Us, pos, Rbars, wbars = cUs, cpos, cR, cw
            tr = min(tr * 1.4, 0.4)
            no_improve = 0
        else:
            # reject: keep the accepted linearization point, shrink the region
            tr *= 0.5
            no_improve += 1
        if tr < 5e-3 or no_improve >= 6:
            break
    row["scp_time"] = round(time.time() - t0, 1)
    row["iters"] = hist_iter
    if best:
        row["J_final"] = round(best[0], 2)
        row["maxV"] = round(best[1], 4)
        row["min_sep"] = round(best[2], 1)
        row["delta"] = round(sc.delta, 1)
        row["gap"] = round((best[0] - LB) / best[0], 4)
    else:
        row["status"] = "no verified iterate"
    if return_sol:
        sol = None if best is None else dict(sc=sc, psis=psis, pts=pts,
                                             Us=best[3], Rbars=best[4])
        return row, sol
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--no-planes", action="store_true",
                    help="diagnostic: collision constraints off")
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        row = run(s, a.N, a.tf, a.M, collision=not a.no_planes)
        rows.append(row)
        print(json.dumps(row), flush=True)
    tag = "_noplanes" if a.no_planes else ""
    out = Path(__file__).with_name("results") / f"scp_oracle_M{a.M}{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
