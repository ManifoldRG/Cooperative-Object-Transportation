"""Full-pipeline scaling benchmark with certificate.

Per instance (random scenarios): allocation first, then oracle.
  1 nominals    M translational SOCPs -> LB = sum ell_j (collision-free bound)
  2 allocation  closure-greedy on the 12 basis wrenches (trajectory-free)
  3 deconflict  sphere-inflation relaxation + ONE joint SOCP (fixed normals)
  4 attitude    min-torque-energy references + nonlinear projection
  5 oracle      per-payload agent SOCP: inward cones, wrench-pinned to (3)+(4)
J_final = sum_j gamma_j (agent-level fuel incl. attitude); certified gap
(J_final - LB)/LB. Pairwise separation verified exactly on the final positions.

Usage: py benchmark_pipeline.py [--Ms 4 8 ... 40] [--seeds 0 1 2 3 4]
                                [--N 120] [--tf 1200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

COT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, COT)
sys.path.insert(0, str(Path(__file__).parent))

import cvxpy as cp  # noqa: E402

from multi_payload import allocation, attitude, scenario, translation  # noqa: E402
from multi_payload.allocation import PayloadAllocator  # noqa: E402
import route2_multi as r2  # noqa: E402
from benchmark_methods import relax_iso  # noqa: E402
from sandwich_check import BASIS  # noqa: E402

PRUNE = 300.0  # keep a plane pair if its relaxed clearance is below delta+PRUNE


def planes_solve(sc, psis, pairs, x, deltas):
    """One joint SOCP with normals fixed from the relaxed references.
    Returns (fuel, rs, Fs) or None."""
    etas = r2.nominal_normals([dict(r=x[j]) for j in range(sc.M)], pairs, sc.N)
    keep = [(j, l) for (j, l) in pairs
            if float(np.min(np.linalg.norm(x[j] - x[l], axis=1)))
            < deltas[j][l] + PRUNE]
    rv = [(cp.Variable((sc.N + 1, 3)), cp.Variable((sc.N + 1, 3)),
           cp.Variable((sc.N, 3))) for _ in range(sc.M)]
    cons = []
    for j, (r, v, F) in enumerate(rv):
        cons += r2._dyn(sc, psis, r, v, F, j)
    for (j, l) in keep:
        for k in range(sc.N + 1):
            cons.append(etas[(j, l)][k] @ (rv[j][0][k] - rv[l][0][k])
                        >= deltas[j][l])
    prob = cp.Problem(cp.Minimize(sum(cp.sum(cp.norm(F, 2, axis=1))
                                      for _, _, F in rv)), cons)
    prob.solve(solver="CLARABEL")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    rs = [np.asarray(r.value) for r, _, _ in rv]
    Fs = [np.asarray(F.value) for _, _, F in rv]
    return float(prob.value), rs, Fs, len(keep)


def run_instance(M, seed, N, tf, headroom=60.0):
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    sc = scenario.generate("random", M=M, n_points=8, n_agents=3 * M,
                           seed=seed, nu_deg=60, N=N, tf=tf,
                           ring_radius=ring, max_tries=5000)
    psis = translation.psi_list(sc)
    row = dict(M=M, seed=seed, ring_radius=round(ring, 0))

    # 1 nominals -> LB
    t0 = time.perf_counter()
    refs = translation.solve_all(sc, psis=psis)
    row["t_nominals"] = round(time.perf_counter() - t0, 1)
    if not all(r["ok"] for r in refs):
        row["status"] = "nominal failed"
        return row
    LB = sum(r["fuel"] for r in refs)
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]

    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc.payloads]
    margin = sc.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    n_conf = sum(min(np.linalg.norm(rbar[j] - rbar[l], axis=1))
                 < deltas[j][l] for (j, l) in pairs)
    row.update(n_pairs=len(pairs), n_conflicts=int(n_conf), LB=round(LB, 2))

    # 2 allocation (closure-greedy, trajectory-free)
    t0 = time.perf_counter()
    al_basis = allocation.build_allocators(sc, [BASIS] * M)
    alloc = allocation.greedy(al_basis, sc.n_agents)
    occ = [sorted(o) for o in alloc["occupied"]]
    row["t_allocation"] = round(time.perf_counter() - t0, 1)
    row["agents_per_payload"] = [len(o) for o in occ]
    row["max_closure_residual"] = round(max(alloc["residual"]), 6)

    # 3 deconfliction: inflation relaxation + one-shot planes
    tgt = {(j, l): deltas[j][l] + headroom for (j, l) in pairs}
    t0 = time.perf_counter()
    x, sweeps, resid = relax_iso(rbar, pairs, tgt, N, M)
    row["t_relax"] = round(time.perf_counter() - t0, 1)
    row["relax_sweeps"] = sweeps
    if x is None:
        row["status"] = f"relax stalled (resid {resid:.2f})"
        return row
    t0 = time.perf_counter()
    out = planes_solve(sc, psis, pairs, x, deltas)
    row["t_planes"] = round(time.perf_counter() - t0, 1)
    if out is None:
        row["status"] = "planes solve failed"
        return row
    J_tr, rs, Fs, n_keep = out
    row.update(J_translational=round(J_tr, 2), planes_pairs_kept=n_keep)

    # exact separation check on the final CoM positions
    min_margin = min(float(np.min(np.linalg.norm(rs[j] - rs[l], axis=1)))
                     - deltas[j][l] for (j, l) in pairs)
    row["min_margin"] = round(min_margin, 1)

    # 4 attitude references + required wrenches
    t0 = time.perf_counter()
    atts = [attitude.attitude_reference(sc, j, seed=seed) for j in range(M)]
    W_act = [attitude.required_wrenches(dict(F=Fs[j]), atts[j]) for j in range(M)]
    row["t_attitude"] = round(time.perf_counter() - t0, 1)
    row["att_terminal_error_max"] = round(max(a["terminal_error"] for a in atts), 4)

    # 5 oracle: per-payload agent SOCP on the occupied points only
    t0 = time.perf_counter()
    J_final = 0.0
    for j in range(M):
        pts = np.asarray([sc.payloads[j].points[p] for p in occ[j]])
        al = PayloadAllocator(pts, W_act[j], sc.nu)
        fuel, e, _ = al.evaluate(set(range(len(pts))))
        if e > 0:
            row["status"] = f"payload {j} not closed (e={e:.3g})"
            return row
        J_final += fuel
    row["t_oracle"] = round(time.perf_counter() - t0, 1)
    row.update(J_final=round(J_final, 2),
               gap_pct=round(100 * (J_final - LB) / LB, 1),
               gap_translational_pct=round(100 * (J_tr - LB) / LB, 1),
               status="ok")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+",
                    default=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--tf", type=float, default=1200.0)
    a = ap.parse_args()
    rows = []
    for M in a.Ms:
        for s in a.seeds:
            t0 = time.perf_counter()
            try:
                row = run_instance(M, s, a.N, a.tf)
            except Exception as exc:
                row = dict(M=M, seed=s, status=f"error: {str(exc)[:100]}")
            row["t_total"] = round(time.perf_counter() - t0, 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
            p = Path(__file__).with_name("results") / "benchmark_pipeline.json"
            p.write_text(json.dumps(rows, indent=1))
    print("done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
