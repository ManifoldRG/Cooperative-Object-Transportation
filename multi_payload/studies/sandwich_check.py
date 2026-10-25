"""Sandwich test of the closure-based allocation methodology, M payloads.

Pipeline under a required-wrench-closure formulation:
  LB   = sum_j ell_j: collision-FREE translational fuel (allocation-independent)
  refs = collision-free translational trajectories from pairwise-plane
         alternation (stand-in for the future collision-avoidance MISOCP)
  att  = prescribed attitude trajectory per payload (allocation-independent)
  UB(O) = sum_j gamma_j(w_k; O_j): agent fuel to follow refs + att exactly
         (feasible for the full Oracle whenever payload j has wrench closure)

Allocations compared, same refs and bounds:
  closure-greedy  agents maximize the wrench closure they add (graded by the
                  residual on the 12 basis wrenches +-e_i, our plateau fix)
  fuel-greedy     the existing rule (graded by residual then fuel on the
                  ACTUAL required wrenches)

Reported: LB, translational collision cost, UB and band (UB-LB)/LB for both
allocations, closure status per payload.

Usage: py sandwich_check.py [--M 5] [--seeds 3 5 7] [--N 30] [--tf 500]
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

import cvxpy as cp  # noqa: E402, F401

from multi_payload import allocation, attitude  # noqa: E402
from spacecraft_libraries.new_opts import so3_exp, so3_log  # noqa: E402

BASIS = np.vstack([np.eye(6), -np.eye(6)])  # 12 basis wrenches +-e_i


def lb_cone(sc, psis, j, occupied):
    """Cone-aware per-payload lower bound (collision-free, attitude-free).

    fuel_k >= max(||F_k||, ||tau_k|| / (ell_j sin nu)) with ell_j the farthest
    OCCUPIED lever arm, and the maneuver requires total torque
    sum_k ||tau_k|| >= 2 lambda_min(J) theta N / tf^2 (inertial angular
    momentum must rise to lambda_min * theta/tf and return to zero; geodesic
    angle theta). Scheduled jointly with the translational dynamics as one
    SOCP (torque cost placed where ||F|| is small), so the bound is additive
    rather than max(ell, torque floor). Exact in continuous time; the Euler
    discretization adds minor slack.
    """
    P0 = sc.payloads[j]
    ell = max(np.linalg.norm(P0.points[p]) for p in occupied)
    c = ell * np.sin(sc.nu)
    Ra = so3_exp(np.asarray(P0.x0.phi, float))
    Rb = so3_exp(np.asarray(P0.xf.phi, float))
    theta = float(np.linalg.norm(so3_log(Ra.T @ Rb)))
    T_tot = 2.0 * float(np.min(np.linalg.eigvalsh(np.asarray(P0.I)))) \
        * theta * sc.N / sc.tf ** 2
    r = cp.Variable((sc.N + 1, 3))
    v = cp.Variable((sc.N + 1, 3))
    F = cp.Variable((sc.N, 3))
    m = cp.Variable(sc.N)
    t = cp.Variable(sc.N, nonneg=True)
    cons = r2._dyn(sc, psis, r, v, F, j)
    cons += [m >= cp.norm(F, 2, axis=1), m >= t / c, cp.sum(t) >= T_tot]
    prob = cp.Problem(cp.Minimize(cp.sum(m)), cons)
    prob.solve(solver="CLARABEL")
    return float(prob.value), ell, theta, T_tot


def solve_fixed_eta_full(sc, psis, pairs, etas):
    """Joint SOCP with fixed normals; returns (fuel, rs, Fs, ds) or None."""
    rv = [(cp.Variable((sc.N + 1, 3)), cp.Variable((sc.N + 1, 3)),
           cp.Variable((sc.N, 3))) for _ in range(sc.M)]
    cons = []
    for j, (r, v, F) in enumerate(rv):
        cons += r2._dyn(sc, psis, r, v, F, j)
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
    rs = [np.asarray(r.value) for r, _, _ in rv]
    Fs = [np.asarray(F.value) for _, _, F in rv]
    ds = {pl: rs[pl[0]] - rs[pl[1]] for pl in pairs}
    return float(prob.value), rs, Fs, ds


def alternate_full(sc, psis, pairs, refs, iters=20):
    """Alternation keeping the best full solution. Returns (fuel, rs, Fs)."""
    etas = r2.nominal_normals(refs, pairs, sc.N)
    best = None
    for _ in range(iters):
        out = solve_fixed_eta_full(sc, psis, pairs, etas)
        if out is None:
            break
        fuel, rs, Fs, ds = out
        if best is not None and fuel > best[0] - 1e-9:
            break
        best = (fuel, rs, Fs)
        for pl in pairs:
            for k in range(sc.N + 1):
                n = np.linalg.norm(ds[pl][k])
                if n > 1e-9:
                    etas[pl][k] = ds[pl][k] / n
    return best


def run(seed, N, tf, M):
    sc, psis, refs0, pairs, gaps, n_conf = r2.crossing(seed, N, tf, M, n_points=8)
    LB = sum(r["fuel"] for r in refs0)
    row = dict(seed=seed, M=M, agents=sc.n_agents, delta=round(sc.delta, 1),
               conflicting_pairs=f"{n_conf}/{len(pairs)}", LB=round(LB, 2))

    t0 = time.time()
    best = alternate_full(sc, psis, pairs, refs0)
    if best is None:
        row["status"] = "alternation failed"
        return row
    fuel_trans, rs, Fs = best
    row["trans_collision_free"] = round(fuel_trans, 2)
    row["alt_time"] = round(time.time() - t0, 1)

    # prescribed attitudes and the actual required wrenches along the refs
    atts = [attitude.attitude_reference(sc, j, seed=seed) for j in range(M)]
    W_act = [attitude.required_wrenches(dict(F=Fs[j]), atts[j]) for j in range(M)]

    # allocators: closure (basis wrenches) and fuel (actual wrenches)
    al_basis = allocation.build_allocators(sc, [BASIS] * M)
    al_act = allocation.build_allocators(sc, W_act)

    t0 = time.time()
    occ_c = allocation.greedy(al_basis, sc.n_agents)["occupied"]
    occ_f = allocation.greedy(al_act, sc.n_agents)["occupied"]
    row["alloc_time"] = round(time.time() - t0, 1)
    row["alloc_closure"] = occ_c
    row["alloc_fuel"] = occ_f

    for name, occ in (("closure", occ_c), ("fuel", occ_f)):
        res = [al_act[j].evaluate(set(occ[j])) for j in range(M)]
        feas = all(r[1] == 0.0 for r in res)
        closure = [round(al_basis[j].evaluate(set(occ[j]))[1], 4) for j in range(M)]
        J_feas = float(sum(r[0] for r in res))   # yardstick only, not a method piece
        row[f"J_feas_{name}"] = round(J_feas, 2) if feas else None
        # cone-aware LB for this allocation (allocation-aware through ell_j)
        lbs = [lb_cone(sc, psis, j, occ[j]) for j in range(M)]
        LBc = float(sum(x[0] for x in lbs))
        row[f"LB_cone_{name}"] = round(LBc, 2)
        if feas:
            row[f"gap_old_{name}"] = round((J_feas - LB) / J_feas, 4)
            row[f"gap_cone_{name}"] = round((J_feas - LBc) / J_feas, 4)
        row[f"closure_resid_{name}"] = closure
    row["theta_deg"] = [round(np.degrees(lb_cone(sc, psis, j, occ_c[j])[2]), 1)
                        for j in range(M)]
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
        print(json.dumps(row), flush=True)
    out = Path(__file__).with_name("results") / f"sandwich_check_M{a.M}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
