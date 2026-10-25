"""Planes-only scaling benchmark with certificate: inflation (isotropic) +
ONE joint SOCP with fixed normals, constant-density random scenarios,
M = 10..100 step 10. Mirrors benchmark_tubes.py for the planes hand-off.

Per instance: nominals -> LB = sum ell_j; vectorized isotropic relaxation;
one joint SOCP with separating-plane normals frozen from the relaxed gap
directions (distant pairs pruned, final separation verified exactly over ALL
pairs). Reports LB, J_planes, certified gap, per-stage times, min margin.

Usage: py benchmark_planes.py [--Ms 10 20 ... 100] [--seeds 0 ... 9]
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

from benchmark_methods import relax_iso  # noqa: E402
from benchmark_scale100 import planes_pruned  # noqa: E402
from multi_payload import scenario, translation  # noqa: E402


def run_instance(M, seed, N, tf, headroom=60.0):
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    sc = scenario.generate("random_seq", M=M, n_points=8, n_agents=3 * M,
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

    tgt = {(j, l): deltas[j][l] + headroom for (j, l) in pairs}
    t0 = time.perf_counter()
    x, sweeps, resid = relax_iso(rbar, pairs, tgt, N, M)
    row["t_relax"] = round(time.perf_counter() - t0, 2)
    row["relax_sweeps"] = sweeps
    if x is None:
        row["status"] = f"relax stalled (resid {resid:.2f})"
        return row

    t0 = time.perf_counter()
    out = planes_pruned(sc, psis, pairs, x, deltas)
    row["t_planes"] = round(time.perf_counter() - t0, 2)
    if out is None:
        row["status"] = "planes solve failed"
        return row
    J_pl, mm, n_keep = out
    row.update(J_planes=round(J_pl, 2),
               gap_pct=round(100 * (J_pl - LB) / LB, 2),
               min_margin=round(mm, 1), planes_pairs_kept=n_keep, status="ok")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+",
                    default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--out", default="benchmark_planes_fair.json")
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
            p = Path(__file__).with_name("results") / a.out
            p.write_text(json.dumps(rows, indent=1))
    print("done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
