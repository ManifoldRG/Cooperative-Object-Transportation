"""Large-M scaling benchmark: alternation vs inflation+tubes vs
inflation+planes (all isotropic), random scenarios, M = 4..100 step 8.

N = 30 timesteps, tf = 500 s. The placement ball grows as 500*(M/20)^(1/3)
above M=20 (constant congestion density). Alternation gets a wall-clock
budget per instance; its best-so-far value is a valid feasible solution
(every iteration is), flagged alt_capped. The one-shot planes solve prunes
pairs whose relaxed clearance exceeds delta+PRUNE and the final separation
is verified exactly over ALL pairs (min_margin).

Usage: py benchmark_scale100.py [--Ms 4 12 ... 100] [--seeds 0 1 2]
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

import cvxpy as cp  # noqa: E402

from benchmark_methods import relax_iso, tubes_solve  # noqa: E402
from multi_payload import scenario, translation  # noqa: E402
import route2_multi as r2  # noqa: E402

PRUNE = 300.0
ALT_BUDGET = 600.0


def alternate_capped(sc, psis, pairs, refs, deltas, iters=20, budget=ALT_BUDGET):
    """alternate_pp with a wall-clock budget; returns (best, n_iters, capped)."""
    etas = r2.nominal_normals(refs, pairs, sc.N)
    best, capped = None, False
    t0 = time.perf_counter()
    for it in range(iters):
        rv = [(cp.Variable((sc.N + 1, 3)), cp.Variable((sc.N + 1, 3)),
               cp.Variable((sc.N, 3))) for _ in range(sc.M)]
        cons = []
        for j, (r, v, F) in enumerate(rv):
            cons += r2._dyn(sc, psis, r, v, F, j)
        for (j, l) in pairs:
            for k in range(sc.N + 1):
                cons.append(etas[(j, l)][k] @ (rv[j][0][k] - rv[l][0][k])
                            >= deltas[j][l])
        prob = cp.Problem(cp.Minimize(sum(cp.sum(cp.norm(F, 2, axis=1))
                                          for _, _, F in rv)), cons)
        try:
            prob.solve(solver="CLARABEL")
        except cp.error.SolverError:
            break
        if prob.status not in ("optimal", "optimal_inaccurate"):
            break
        if best is not None and prob.value > best - 1e-9:
            return best, it + 1, False
        best = float(prob.value)
        if time.perf_counter() - t0 > budget:
            capped = it + 1 < iters
            return best, it + 1, capped
        for (j, l) in pairs:
            for k in range(sc.N + 1):
                d = np.asarray(rv[j][0].value[k]) - np.asarray(rv[l][0].value[k])
                n = np.linalg.norm(d)
                if n > 1e-9:
                    etas[(j, l)][k] = d / n
    return best, iters, capped


def planes_pruned(sc, psis, pairs, x, deltas):
    """One joint SOCP, normals from the relaxed refs, distant pairs pruned.
    Returns (fuel, min_margin over ALL pairs, n_kept) or None."""
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
    mm = min(float(np.min(np.linalg.norm(rs[j] - rs[l], axis=1)))
             - deltas[j][l] for (j, l) in pairs)
    return float(prob.value), mm, len(keep)


def run_instance(M, seed, N, tf, headroom=60.0):
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    sc = scenario.generate("random", M=M, n_points=8, n_agents=3 * M,
                           seed=seed, nu_deg=60, N=N, tf=tf,
                           ring_radius=ring, max_tries=5000)
    psis = translation.psi_list(sc)

    t0 = time.perf_counter()
    refs = translation.solve_all(sc, psis=psis)
    t_nom = time.perf_counter() - t0
    if not all(r["ok"] for r in refs):
        return dict(M=M, seed=seed, status="nominal failed")
    LB = sum(r["fuel"] for r in refs)
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]

    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc.payloads]
    margin = sc.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    n_conf = sum(min(np.linalg.norm(rbar[j] - rbar[l], axis=1))
                 < deltas[j][l] for (j, l) in pairs)
    row = dict(M=M, seed=seed, ring_radius=round(ring, 0), n_pairs=len(pairs),
               n_conflicts=int(n_conf), LB=round(LB, 2),
               t_nominals=round(t_nom, 2))

    # alternation (budgeted)
    t0 = time.perf_counter()
    J_alt, alt_iters, alt_capped = alternate_capped(sc, psis, pairs, refs, deltas)
    row["t_alternation"] = round(time.perf_counter() - t0, 2)
    row["J_alternation"] = round(J_alt, 2) if J_alt else None
    row["alt_iters"] = alt_iters
    row["alt_capped"] = bool(alt_capped)

    # inflation relaxation (isotropic)
    tgt = {(j, l): deltas[j][l] + headroom for (j, l) in pairs}
    t0 = time.perf_counter()
    x, sweeps, resid = relax_iso(rbar, pairs, tgt, N, M)
    row["t_relax"] = round(time.perf_counter() - t0, 2)
    row["relax_sweeps"] = sweeps
    if x is None:
        row["status"] = f"relax stalled (resid {resid:.2f})"
        return row

    t0 = time.perf_counter()
    J_tubes = tubes_solve(sc, psis, pairs, deltas, x, N, M)
    row["t_tubes"] = round(time.perf_counter() - t0, 2)
    row["J_tubes"] = round(J_tubes, 2) if J_tubes else None

    t0 = time.perf_counter()
    out = planes_pruned(sc, psis, pairs, x, deltas)
    row["t_planes"] = round(time.perf_counter() - t0, 2)
    if out is None:
        row["status"] = "planes solve failed"
        return row
    J_pl, mm, n_keep = out
    row.update(J_planes=round(J_pl, 2), min_margin=round(mm, 1),
               planes_pairs_kept=n_keep, status="ok")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+",
                    default=[4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    a = ap.parse_args()
    rows = []
    for M in a.Ms:
        for s in a.seeds:
            try:
                row = run_instance(M, s, a.N, a.tf)
            except Exception as exc:
                row = dict(M=M, seed=s, status=f"error: {str(exc)[:100]}")
            rows.append(row)
            print(json.dumps(row), flush=True)
            p = Path(__file__).with_name("results") / "benchmark_scale100.json"
            p.write_text(json.dumps(rows, indent=1))
    print("done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
