"""Route-2 check, M payloads: pairwise position-dependent separating planes
INSIDE the optimizer, on the translational CoM-force layer.

Pairwise collision is enforced exactly by
    eta_{jl,k}^T (r_{j,k} - r_{l,k}) >= delta,   ||eta_{jl,k}|| <= 1,
with the normals decision variables (bilinear, nonconvex).

Scenario: M payloads crossing one central region simultaneously (each nominal
midpoint steered to ~20 m from the center, so most pairs conflict).

Per seed: LB0 (unconstrained fuel), conflicting pairs, UB_eta (normals fixed
to nominal directions; may be jointly infeasible), UB_alt (alternation from
that warm start), J_star (bilinear NLP, IPOPT multi-start, rollout-verified),
UB_repair (waypoint-repair baseline). McCormick dropped: vacuous at M=2.

Usage: py route2_multi.py [--M 5] [--seeds 3 5 7] [--N 30] [--tf 500]
"""
from __future__ import annotations

import argparse
import copy
import itertools
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


def crossing(seed, N, tf, M, mid_radius=20.0, n_points=4):
    sc = scenario.generate("crossing", M=M, n_points=n_points, n_agents=3 * M, seed=seed,
                           nu_deg=60, N=N, tf=tf)
    for P in sc.payloads:              # swap sides so all paths cross the center
        P.xf.r = -P.x0.r
    psis = translation.psi_list(sc)
    refs = translation.solve_all(sc, psis=psis)
    for j, P in enumerate(sc.payloads):   # steer nominal midpoints to the center
        target = mid_radius * P.x0.r / np.linalg.norm(P.x0.r)
        shift = target - refs[j]["r"][N // 2]
        P.x0.r = P.x0.r + shift
        P.xf.r = P.xf.r + shift
    refs = translation.solve_all(sc, psis=psis)
    pairs = list(itertools.combinations(range(sc.M), 2))
    gaps = {pl: min(np.linalg.norm(refs[pl[0]]["r"][k] - refs[pl[1]]["r"][k])
                    for k in range(N + 1)) for pl in pairs}
    n_conf = sum(g < sc.delta for g in gaps.values())
    assert n_conf >= 1, "no conflicting pair"
    return sc, psis, refs, pairs, gaps, n_conf


def nominal_normals(refs, pairs, N):
    etas = {}
    for (j, l) in pairs:
        e = np.zeros((N + 1, 3))
        for k in range(N + 1):
            d = refs[j]["r"][k] - refs[l]["r"][k]
            n = np.linalg.norm(d)
            e[k] = d / n if n > 1e-9 else np.array([1.0, 0, 0])
        etas[(j, l)] = e
    return etas


def _dyn(sc, psis, r, v, F, j):
    cons = [r[0] == sc.payloads[j].x0.r, v[0] == sc.payloads[j].x0.v,
            r[sc.N] == sc.payloads[j].xf.r, v[sc.N] == sc.payloads[j].xf.v]
    for k in range(sc.N):
        P = psis[k]
        cons += [r[k + 1] == r[k] + sc.dt * v[k],
                 v[k + 1] == v[k] + sc.dt * (P[:, :3] @ r[k] + P[:, 3:] @ v[k]
                                             + F[k] / sc.payloads[j].m)]
    return cons


def solve_fixed_eta(sc, psis, pairs, etas):
    """All normals fixed -> one joint SOCP. Returns (fuel, d by pair) or None."""
    rv = [(cp.Variable((sc.N + 1, 3)), cp.Variable((sc.N + 1, 3)),
           cp.Variable((sc.N, 3))) for _ in range(sc.M)]
    cons = []
    for j, (r, v, F) in enumerate(rv):
        cons += _dyn(sc, psis, r, v, F, j)
    for (j, l) in pairs:
        for k in range(sc.N + 1):
            cons.append(etas[(j, l)][k] @ (rv[j][0][k] - rv[l][0][k]) >= sc.delta)
    prob = cp.Problem(cp.Minimize(sum(cp.sum(cp.norm(F, 2, axis=1))
                                      for _, _, F in rv)), cons)
    try:
        prob.solve(solver="CLARABEL")
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    ds = {pl: np.asarray(rv[pl[0]][0].value) - np.asarray(rv[pl[1]][0].value)
          for pl in pairs}
    return float(prob.value), ds


def alternate(sc, psis, pairs, refs, iters=20):
    etas = nominal_normals(refs, pairs, sc.N)
    best = None
    for _ in range(iters):
        out = solve_fixed_eta(sc, psis, pairs, etas)
        if out is None:
            return best
        fuel, ds = out
        if best is not None and fuel > best - 1e-9:
            return min(best, fuel)
        best = fuel if best is None else min(best, fuel)
        for pl in pairs:
            for k in range(sc.N + 1):
                n = np.linalg.norm(ds[pl][k])
                if n > 1e-9:
                    etas[pl][k] = ds[pl][k] / n
    return best


def exact_nlp(sc, psis, refs, pairs, n_random=5, seed=0):
    M, N, dt = sc.M, sc.N, sc.dt
    S = ca.SX
    F = [S.sym(f"F{j}", N, 3) for j in range(M)]
    r = [S.sym(f"r{j}", N + 1, 3) for j in range(M)]
    v = [S.sym(f"v{j}", N + 1, 3) for j in range(M)]
    eta = {pl: S.sym(f"e{pl[0]}_{pl[1]}", N + 1, 3) for pl in pairs}
    g, lbg, ubg = [], [], []

    def pin(e):
        g.append(ca.reshape(e, -1, 1))
        lbg.extend([0] * e.numel())
        ubg.extend([0] * e.numel())

    obj = 0
    for j in range(M):
        P0 = sc.payloads[j]
        pin(r[j][0, :].T - ca.DM(P0.x0.r))
        pin(v[j][0, :].T - ca.DM(P0.x0.v))
        pin(r[j][N, :].T - ca.DM(P0.xf.r))
        pin(v[j][N, :].T - ca.DM(P0.xf.v))
        for k in range(N):
            P = ca.DM(psis[k])
            pin(r[j][k + 1, :].T - (r[j][k, :].T + dt * v[j][k, :].T))
            pin(v[j][k + 1, :].T - (v[j][k, :].T + dt * (
                P[:, :3] @ r[j][k, :].T + P[:, 3:] @ v[j][k, :].T
                + F[j][k, :].T / P0.m)))
            obj += ca.sqrt(ca.sumsqr(F[j][k, :]) + 1e-8)
    for (j, l) in pairs:
        for k in range(N + 1):
            d = (r[j][k, :] - r[l][k, :]).T
            g.append(ca.dot(eta[(j, l)][k, :].T, d) - sc.delta)
            lbg.append(0)
            ubg.append(ca.inf)
            g.append(1 - ca.sumsqr(eta[(j, l)][k, :]))
            lbg.append(0)
            ubg.append(ca.inf)

    x = ca.veccat(*F, *r, *v, *[eta[pl] for pl in pairs])
    solver = ca.nlpsol("r2", "ipopt", {"x": x, "f": obj, "g": ca.vertcat(*g)},
                       {"print_time": False,
                        "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 3000}})

    def rollout_ok(Fs):
        fuel, hist = 0.0, []
        for j in range(M):
            P0 = sc.payloads[j]
            rj = np.asarray(P0.x0.r, float).copy()
            vj = np.asarray(P0.x0.v, float).copy()
            h = [rj.copy()]
            for k in range(N):
                P = np.asarray(psis[k])
                rj, vj = rj + dt * vj, vj + dt * (P[:, :3] @ rj + P[:, 3:] @ vj
                                                  + Fs[j][k] / P0.m)
                h.append(rj.copy())
            if (np.linalg.norm(rj - P0.xf.r) > 1e-3
                    or np.linalg.norm(vj - P0.xf.v) > 1e-4):
                return None
            fuel += float(np.linalg.norm(Fs[j], axis=1).sum())
            hist.append(np.asarray(h))
        for (j, l) in pairs:
            if min(np.linalg.norm(hist[j][k] - hist[l][k])
                   for k in range(N + 1)) < sc.delta - 1e-4:
                return None
        return fuel

    rng = np.random.default_rng(seed)
    starts = [nominal_normals(refs, pairs, N)]
    for _ in range(n_random):
        st = {}
        for pl in pairs:
            e = rng.standard_normal((N + 1, 3))
            st[pl] = e / np.linalg.norm(e, axis=1, keepdims=True)
        starts.append(st)
    best, n_ok = np.inf, 0
    nF = N * 3
    for etas0 in starts:
        x0 = np.concatenate(
            [np.asarray(refs[j]["F"]).flatten(order="F") for j in range(M)]
            + [np.asarray(refs[j]["r"]).flatten(order="F") for j in range(M)]
            + [np.asarray(refs[j]["v"]).flatten(order="F") for j in range(M)]
            + [etas0[pl].flatten(order="F") for pl in pairs])
        sol = solver(x0=x0, lbg=np.array(lbg, float), ubg=np.array(ubg, float))
        wv = np.asarray(sol["x"]).flatten()
        Fs = [wv[j * nF:(j + 1) * nF].reshape(N, 3, order="F") for j in range(M)]
        fuel = rollout_ok(Fs)
        if fuel is None:
            continue
        n_ok += 1
        best = min(best, fuel)
    return best, n_ok


def run(seed, N, tf, M):
    sc, psis, refs, pairs, gaps, n_conf = crossing(seed, N, tf, M)
    row = dict(seed=seed, M=M, N=N, delta=round(sc.delta, 1),
               conflicting_pairs=f"{n_conf}/{len(pairs)}",
               min_pair_gap=round(min(gaps.values()), 1),
               LB0=sum(r["fuel"] for r in refs))

    reps = copy.deepcopy(refs)
    t0 = time.time()
    try:
        reps, _, rlog, ok = conflicts.repair(sc, reps, [dict() for _ in range(M)],
                                             conflicts.zero_eps(sc), psis=psis)
        row["UB_repair"] = sum(r["fuel"] for r in reps) if ok else None
        row["repair_iters"] = len(rlog)
    except (RuntimeError, ValueError) as exc:
        row["UB_repair"] = None
        row["repair_error"] = str(exc)[:60]
    row["repair_time"] = round(time.time() - t0, 1)

    t0 = time.time()
    out = solve_fixed_eta(sc, psis, pairs, nominal_normals(refs, pairs, sc.N))
    row["UB_eta_nominal"] = out[0] if out else None
    row["UB_alternation"] = alternate(sc, psis, pairs, refs)
    row["alt_time"] = round(time.time() - t0, 1)

    t0 = time.time()
    row["J_star"], row["nlp_feasible_starts"] = exact_nlp(sc, psis, refs, pairs,
                                                          seed=seed)
    row["nlp_time"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        row = run(s, a.N, a.tf, a.M)
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)
    out = Path(__file__).with_name("results") / f"route2_check_M{a.M}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
