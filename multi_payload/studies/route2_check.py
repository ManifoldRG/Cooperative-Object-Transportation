"""Route-2 tightness check: position-dependent separating plane INSIDE the
optimizer, on the translational CoM-force layer, 2 payloads.

Collision between payloads 1 and 2 at step k is enforced exactly by
    eta_k^T (r_{1,k} - r_{2,k}) >= delta,   ||eta_k|| <= 1,
with the plane normal eta_k a decision variable (bilinear, nonconvex).

On a crossing scenario whose nominals conflict, computes:
  LB0      unconstrained (nominal) fuel
  MC(box)  McCormick+SOC relaxation of the bilinear terms, with the relative
           position d = r1 - r2 boxed either globally or in a box of radius R
           centered on the NOMINAL d (warm-start certificate: valid lower
           bound over all plans staying within R of the nominals)
  UB_eta   route-1 collapse: eta fixed to nominal normals, one SOCP
  UB_alt   alternation eta <-> trajectories from the nominal warm start
  J*       exact bilinear NLP, IPOPT multi-start, feasibility re-checked
  UB_rep   waypoint-repair baseline fuel (reference)

Usage: py route2_check.py [--seeds 3 5 7] [--N 30] [--tf 500]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

COT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, COT)

import casadi as ca  # noqa: E402
import cvxpy as cp   # noqa: E402

from multi_payload import conflicts, scenario, translation  # noqa: E402

FEAS_TOL = 1e-6


def crossing(seed, N, tf):
    sc = scenario.generate("crossing", M=2, n_points=4, n_agents=6, seed=seed,
                           nu_deg=60, N=N, tf=tf)
    for P in sc.payloads:          # swap sides so the paths cross
        P.xf.r = -P.x0.r
    psis = translation.psi_list(sc)
    refs = translation.solve_all(sc, psis=psis)
    # orbital bowing can separate the paths; shift payload 2 so the nominal
    # midpoints coincide, which forces a head-on conflict
    shift = refs[0]["r"][N // 2] - refs[1]["r"][N // 2]
    sc.payloads[1].x0.r = sc.payloads[1].x0.r + shift
    sc.payloads[1].xf.r = sc.payloads[1].xf.r + shift
    refs = translation.solve_all(sc, psis=psis)
    gap = min(np.linalg.norm(refs[0]["r"][k] - refs[1]["r"][k]) for k in range(N + 1))
    assert gap < sc.delta, f"no conflict after shift (gap {gap:.1f} >= {sc.delta:.1f})"
    return sc, psis, refs, gap


def nominal_normals(refs, N):
    """Unit normals along the nominal relative position (undefined -> e_x)."""
    etas = np.zeros((N + 1, 3))
    for k in range(N + 1):
        d = refs[0]["r"][k] - refs[1]["r"][k]
        n = np.linalg.norm(d)
        etas[k] = d / n if n > 1e-9 else np.array([1.0, 0, 0])
    return etas


# --------------------------------------------------------------- convex pieces
def _dyn(sc, psis, r, v, F, j):
    cons = [r[0] == sc.payloads[j].x0.r, v[0] == sc.payloads[j].x0.v,
            r[sc.N] == sc.payloads[j].xf.r, v[sc.N] == sc.payloads[j].xf.v]
    for k in range(sc.N):
        P = psis[k]
        cons += [r[k + 1] == r[k] + sc.dt * v[k],
                 v[k + 1] == v[k] + sc.dt * (P[:, :3] @ r[k] + P[:, 3:] @ v[k]
                                             + F[k] / sc.payloads[j].m)]
    return cons


def _vars(sc):
    return [(cp.Variable((sc.N + 1, 3)), cp.Variable((sc.N + 1, 3)),
             cp.Variable((sc.N, 3))) for _ in range(2)]


def solve_fixed_eta(sc, psis, etas):
    """Route-1 collapse: eta fixed -> SOCP. Returns (fuel, d (N+1,3)) or None."""
    (r1, v1, F1), (r2, v2, F2) = _vars(sc)
    cons = _dyn(sc, psis, r1, v1, F1, 0) + _dyn(sc, psis, r2, v2, F2, 1)
    for k in range(sc.N + 1):
        cons.append(etas[k] @ (r1[k] - r2[k]) >= sc.delta)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F1, 2, axis=1))
                                  + cp.sum(cp.norm(F2, 2, axis=1))), cons)
    try:
        prob.solve(solver="CLARABEL")
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return float(prob.value), np.asarray(r1.value) - np.asarray(r2.value)


def alternate(sc, psis, refs, iters=15):
    """eta <- d/||d||, then SOCP, repeated from the nominal warm start."""
    etas = nominal_normals(refs, sc.N)
    best = None
    for _ in range(iters):
        out = solve_fixed_eta(sc, psis, etas)
        if out is None:
            return best
        fuel, d = out
        if best is not None and fuel > best - 1e-9:
            return min(best, fuel)
        best = fuel if best is None else min(best, fuel)
        for k in range(sc.N + 1):
            n = np.linalg.norm(d[k])
            if n > 1e-9:
                etas[k] = d[k] / n
    return best


def mccormick(sc, psis, dL, dU):
    """McCormick+SOC relaxation; d boxed in [dL, dU] (N+1,3). Lower bound."""
    (r1, v1, F1), (r2, v2, F2) = _vars(sc)
    eta = cp.Variable((sc.N + 1, 3))
    w = cp.Variable((sc.N + 1, 3))
    cons = _dyn(sc, psis, r1, v1, F1, 0) + _dyn(sc, psis, r2, v2, F2, 1)
    for k in range(sc.N + 1):
        d = r1[k] - r2[k]
        cons += [d >= dL[k], d <= dU[k], cp.norm(eta[k]) <= 1,
                 cp.sum(w[k]) >= sc.delta]
        for i in range(3):
            x, y, ww = eta[k][i], d[i], w[k][i]
            xL, xU, yL, yU = -1.0, 1.0, dL[k][i], dU[k][i]
            cons += [ww >= xL * y + x * yL - xL * yL,
                     ww >= xU * y + x * yU - xU * yU,
                     ww <= xU * y + x * yL - xU * yL,
                     ww <= xL * y + x * yU - xL * yU]
    prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F1, 2, axis=1))
                                  + cp.sum(cp.norm(F2, 2, axis=1))), cons)
    try:
        prob.solve(solver="CLARABEL")
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None  # box too small for the pins -> infeasible certificate
    return float(prob.value)


# ------------------------------------------------------------------- exact NLP
def exact_nlp(sc, psis, refs, n_random=5, seed=0):  # noqa: C901
    """Bilinear NLP via IPOPT multi-start. Returns (best fuel, n_feasible)."""
    N, dt = sc.N, sc.dt
    S = ca.SX
    F = [S.sym(f"F{j}", N, 3) for j in range(2)]
    r = [S.sym(f"r{j}", N + 1, 3) for j in range(2)]
    v = [S.sym(f"v{j}", N + 1, 3) for j in range(2)]
    eta = S.sym("eta", N + 1, 3)
    g, lbg, ubg = [], [], []

    def pin(e):
        g.append(ca.reshape(e, -1, 1)); lbg.extend([0] * e.numel()); ubg.extend([0] * e.numel())

    obj = 0
    for j in range(2):
        P0 = sc.payloads[j]
        pin(r[j][0, :].T - ca.DM(P0.x0.r)); pin(v[j][0, :].T - ca.DM(P0.x0.v))
        pin(r[j][N, :].T - ca.DM(P0.xf.r)); pin(v[j][N, :].T - ca.DM(P0.xf.v))
        for k in range(N):
            P = ca.DM(psis[k])
            pin(r[j][k + 1, :].T - (r[j][k, :].T + dt * v[j][k, :].T))
            pin(v[j][k + 1, :].T - (v[j][k, :].T + dt * (
                P[:, :3] @ r[j][k, :].T + P[:, 3:] @ v[j][k, :].T + F[j][k, :].T / P0.m)))
            obj += ca.sqrt(ca.sumsqr(F[j][k, :]) + 1e-8)
    for k in range(N + 1):
        d = (r[0][k, :] - r[1][k, :]).T
        g.append(ca.dot(eta[k, :].T, d) - sc.delta); lbg.append(0); ubg.append(ca.inf)
        g.append(1 - ca.sumsqr(eta[k, :])); lbg.append(0); ubg.append(ca.inf)

    x = ca.veccat(*F, *r, *v, eta)
    solver = ca.nlpsol("r2", "ipopt", {"x": x, "f": obj, "g": ca.vertcat(*g)},
                       {"print_time": False, "ipopt": {"print_level": 0, "sb": "yes",
                                                       "max_iter": 2000}})
    def rollout_ok(F1, F2):
        fuel = 0.0
        rr = []
        for j, Fj in enumerate((F1, F2)):
            P0 = sc.payloads[j]
            rj = np.asarray(P0.x0.r, float).copy()
            vj = np.asarray(P0.x0.v, float).copy()
            hist = [rj.copy()]
            for k in range(N):
                P = np.asarray(psis[k])
                rj, vj = rj + dt * vj, vj + dt * (P[:, :3] @ rj + P[:, 3:] @ vj
                                                  + Fj[k] / P0.m)
                hist.append(rj.copy())
            if (np.linalg.norm(rj - P0.xf.r) > 1e-3
                    or np.linalg.norm(vj - P0.xf.v) > 1e-4):
                return None
            fuel += float(np.linalg.norm(Fj, axis=1).sum())
            rr.append(np.asarray(hist))
        if min(np.linalg.norm(rr[0][k] - rr[1][k]) for k in range(N + 1)) < sc.delta - 1e-4:
            return None
        return fuel

    rng = np.random.default_rng(seed)
    starts = [nominal_normals(refs, N)]
    for _ in range(n_random):
        e = rng.standard_normal((N + 1, 3))
        starts.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    best, n_ok = np.inf, 0
    for etas0 in starts:
        x0 = np.concatenate([np.asarray(refs[j]["F"]).flatten(order="F") for j in range(2)]
                            + [np.asarray(refs[j]["r"]).flatten(order="F") for j in range(2)]
                            + [np.asarray(refs[j]["v"]).flatten(order="F") for j in range(2)]
                            + [etas0.flatten(order="F")])
        sol = solver(x0=x0, lbg=np.array(lbg, float), ubg=np.array(ubg, float))
        wv = np.asarray(sol["x"]).flatten()
        nF = N * 3
        F1 = wv[:nF].reshape(N, 3, order="F")
        F2 = wv[nF:2 * nF].reshape(N, 3, order="F")
        fuel = rollout_ok(F1, F2)   # independent check: dynamics, pins, separation
        if fuel is None:
            continue
        n_ok += 1
        best = min(best, fuel)
    return best, n_ok


def run(seed, N, tf, radii):
    sc, psis, refs, gap = crossing(seed, N, tf)
    LB0 = sum(r["fuel"] for r in refs)
    dbar = np.asarray(refs[0]["r"]) - np.asarray(refs[1]["r"])
    row = dict(seed=seed, N=N, delta=sc.delta, nominal_min_gap=gap, LB0=LB0)

    # waypoint-repair reference
    reps = copy.deepcopy(refs)
    reps, _, _, ok = conflicts.repair(sc, reps, [dict(), dict()], conflicts.zero_eps(sc),
                                      psis=psis)
    row["UB_repair"] = sum(r["fuel"] for r in reps) if ok else None

    out = solve_fixed_eta(sc, psis, nominal_normals(refs, sc.N))
    row["UB_eta_nominal"] = out[0] if out else None
    row["UB_alternation"] = alternate(sc, psis, refs)

    big = float(np.abs(dbar).max() + 2000.0)
    row["MC_global"] = mccormick(sc, psis, np.full((sc.N + 1, 3), -big),
                                 np.full((sc.N + 1, 3), big))
    for R in radii:
        row[f"MC_R{R:g}"] = mccormick(sc, psis, dbar - R, dbar + R)

    t0 = time.time()
    row["J_star"], row["nlp_feasible_starts"] = exact_nlp(sc, psis, refs, seed=seed)
    row["nlp_time"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--radii", type=float, nargs="+", default=[25, 50, 100, 200, 400])
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        row = run(s, a.N, a.tf, a.radii)
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)
    out = Path(__file__).with_name("results") / "route2_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
