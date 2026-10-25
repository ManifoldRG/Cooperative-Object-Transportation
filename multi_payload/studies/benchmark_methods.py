"""Compute-time scaling benchmark: alternation vs inflation+{tubes,planes}
x {isotropic, anisotropic}, random scenarios, M = 2..20.

Per instance: shared nominals (timed separately; also give LB), then
  alternation  iterated per-pair-delta plane re-orientation (joint SOCP/iter)
  relax        vectorized sphere-inflation PBD (while-loop + polish);
               isotropic pushes, or fuel-metric W^-2 g pushes with axis
               prices from +-50 m waypoint probes (timed as t_prices)
  tubes        relax + M decoupled ball-corridor SOCPs
  planes       relax + ONE joint SOCP with fixed normals from the relaxation
Times reported per stage; inflation method time = relax (+ prices) + hand-off.

Usage: py benchmark_methods.py [--Ms 2 6 10 14 18 20] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

COT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, COT)
sys.path.insert(0, str(Path(__file__).parent))

import cvxpy as cp  # noqa: E402

from multi_payload import scenario, translation  # noqa: E402
import route2_multi as r2  # noqa: E402
from corridor_oracle import alternate_pp, axis_prices  # noqa: E402


def relax_iso(rbar, pairs, tgt, N, M, max_sweeps=5000, alpha=0.25, sigma=1.5,
              tol=0.1, Winv2=None, profile="gauss"):
    """Vectorized inflation PBD; isotropic pushes, or fuel-metric W^-2 g
    pushes when Winv2 (per-payload axis weights) is given.

    profile selects how a push spreads along the trajectory:
      "gauss" (default): per-step exact-gap projections + Gaussian smoothing
        of the displacement field (the original two-phase scheme);
      "tent": per pair and sweep, ONE whole-trajectory push peaked at the
        worst violated step, linearly decaying to zero at the pinned ends
        (the elastic-string / point-load Green's function of min int(u')^2);
      "beam": the same with a smoothstep profile, zero slope at the ends
        and the peak (beam-like, closer to the min int(u'')^2 fuel metric);
      "bump<W>" (e.g. "bump6"): compactly supported hybrid — a smoothstep
        bump of half-width W steps around the worst violated step, smooth
        (zero value AND slope) at the window borders, so the push keeps
        tent's no-self-fighting speed while deviation stays local.

    Step k=1 is frozen (no pushes, no smoothing displacement, excluded from
    the residual): v(0)=0 pins r_1 = r_0 for every payload, so step-1
    separation equals the start separation, which placement guarantees. An
    inflated step-1 target would be both unattainable downstream (the tube
    ball moves away from where the payload must be) and unnecessary."""
    x = [r.copy() for r in rbar]

    if profile != "gauss":
        ks = np.arange(N + 1, dtype=float)
        B = np.zeros((N + 1, N + 1))
        W = float(profile[4:] or 6) if profile.startswith("bump") else None
        for k in range(2, N):
            if W is not None:
                u = np.clip(1.0 - np.abs(ks - k) / W, 0.0, None)
                w = u * u * (3.0 - 2.0 * u)
            else:
                w = np.minimum((ks - 1) / (k - 1), (N - ks) / (N - k))
                w = np.clip(w, 0.0, None)
                if profile == "beam":
                    w = w * w * (3.0 - 2.0 * w)
            w[0] = w[1] = w[N] = 0.0
            B[k] = w

        def sweep_field():
            worst = 0.0
            for (j, l) in pairs:
                d = x[j][2:N] - x[l][2:N]
                dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                need = tgt[(j, l)] - dist
                kw = int(np.argmax(need))
                worst_need = float(need[kw])
                if worst_need <= 0:
                    continue
                worst = max(worst, worst_need)
                g = d[kw] / dist[kw]
                F = np.outer(B[kw + 2], 0.5 * worst_need * g)
                x[j] += F
                x[l] -= F
            return worst

        def sweep_point():
            worst = 0.0
            for (j, l) in pairs:
                d = x[j][2:N] - x[l][2:N]
                dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                need = tgt[(j, l)] - dist
                m = need > 0
                if m.any():
                    worst = max(worst, float(need[m].max()))
                    g = d[m] / dist[m][:, None]
                    x[j][2:N][m] += 0.5 * need[m][:, None] * g
                    x[l][2:N][m] -= 0.5 * need[m][:, None] * g
            return worst

        sweeps, worst = 0, np.inf
        while sweeps < max_sweeps and worst >= tol:
            worst = sweep_field()
            sweeps += 1
        if profile == "anneal" and worst < tol:
            # retraction: shrink the displacement field toward the nominal,
            # re-project real violations, and keep it SMOOTH while doing so
            # (shrink + point projections alone collapse to jagged spikes)
            for _ in range(120):
                for j in range(M):
                    x[j] = rbar[j] + 0.93 * (x[j] - rbar[j])
                sweep_point()
                for j in range(M):
                    u = x[j] - rbar[j]
                    us = gaussian_filter1d(u, sigma=sigma, axis=0)
                    u = (1 - alpha) * u + alpha * us
                    u[0] = 0.0
                    u[1] = 0.0
                    u[N] = 0.0
                    x[j] = rbar[j] + u
                sweeps += 1
            worst = np.inf
            while sweeps < max_sweeps and worst >= tol:
                worst = sweep_point()
                sweeps += 1
        return (x if worst < tol else None), sweeps, float(worst)

    def sweep(smooth):
        worst = 0.0
        for (j, l) in pairs:
            d = x[j][2:N] - x[l][2:N]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            need = tgt[(j, l)] - dist
            m = need > 0
            if m.any():
                worst = max(worst, float(need[m].max()))
                g = d[m] / dist[m][:, None]
                if Winv2 is None:
                    x[j][2:N][m] += 0.5 * need[m][:, None] * g
                    x[l][2:N][m] -= 0.5 * need[m][:, None] * g
                else:
                    pj = g * Winv2[j]
                    pl = g * Winv2[l]
                    x[j][2:N][m] += (0.5 * need[m] / np.sum(g * pj, axis=1)
                                     )[:, None] * pj
                    x[l][2:N][m] -= (0.5 * need[m] / np.sum(g * pl, axis=1)
                                     )[:, None] * pl
        if smooth:
            for j in range(M):
                u = x[j] - rbar[j]
                us = gaussian_filter1d(u, sigma=sigma, axis=0)
                u = (1 - alpha) * u + alpha * us
                u[0] = 0.0
                u[1] = 0.0
                u[N] = 0.0
                x[j] = rbar[j] + u
        return worst

    sweeps, worst = 0, np.inf
    while sweeps < 800 and worst >= 1.0:
        worst = sweep(True)
        sweeps += 1
    while sweeps < max_sweeps and worst >= tol:
        worst = sweep(False)
        sweeps += 1
    return (x if worst < tol else None), sweeps, float(worst)


def tubes_solve(sc, psis, pairs, deltas, x, N, M):
    CAP = 150.0
    fuel = 0.0
    for j in range(M):
        e = np.full(N + 1, CAP)
        for l in range(M):
            if l != j:
                e = np.minimum(e, 0.5 * (np.linalg.norm(x[j] - x[l], axis=1)
                                         - deltas[j][l]))
        r = cp.Variable((N + 1, 3))
        v = cp.Variable((N + 1, 3))
        F = cp.Variable((N, 3))
        cons = r2._dyn(sc, psis, r, v, F, j)
        for k in range(1, N):
            if e[k] < CAP:
                cons.append(cp.norm(r[k] - x[j][k]) <= e[k])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        fuel += float(prob.value)
    return fuel


def run_instance(M, seed, N, tf, headroom=60.0):
    sc = scenario.generate("random", M=M, n_points=8, n_agents=3 * M,
                           seed=seed, nu_deg=60, N=N, tf=tf)
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
    row = dict(M=M, seed=seed, n_pairs=len(pairs), n_conflicts=int(n_conf),
               LB=round(LB, 2), t_nominals=round(t_nom, 2))

    # alternation
    t0 = time.perf_counter()
    J_alt = alternate_pp(sc, psis, pairs, refs, deltas)
    row["t_alternation"] = round(time.perf_counter() - t0, 2)
    row["J_alternation"] = round(J_alt, 2) if J_alt else None

    # inflation relaxation, isotropic and anisotropic, each -> tubes + planes
    tgt = {(j, l): deltas[j][l] + headroom for (j, l) in pairs}

    t0 = time.perf_counter()
    Ws = [axis_prices(sc, psis, refs, j) for j in range(M)]
    row["t_prices"] = round(time.perf_counter() - t0, 2)
    Winv2 = [1.0 / np.asarray(w) ** 2 for w in Ws]

    for tag, W in (("iso", None), ("aniso", Winv2)):
        t0 = time.perf_counter()
        x, sweeps, resid = relax_iso(rbar, pairs, tgt, N, M, Winv2=W)
        row[f"t_relax_{tag}"] = round(time.perf_counter() - t0, 2)
        row[f"sweeps_{tag}"] = sweeps
        if x is None:
            row["status"] = f"relax {tag} stalled (resid {resid:.2f})"
            return row

        t0 = time.perf_counter()
        J_tubes = tubes_solve(sc, psis, pairs, deltas, x, N, M)
        row[f"t_tubes_{tag}"] = round(time.perf_counter() - t0, 2)
        row[f"J_tubes_{tag}"] = round(J_tubes, 2) if J_tubes else None

        t0 = time.perf_counter()
        J_planes = alternate_pp(sc, psis, pairs,
                                [dict(r=x[j]) for j in range(M)],
                                deltas, iters=1)
        row[f"t_planes_{tag}"] = round(time.perf_counter() - t0, 2)
        row[f"J_planes_{tag}"] = round(J_planes, 2) if J_planes else None
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+",
                    default=[2, 6, 10, 14, 18, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    a = ap.parse_args()
    rows = []
    for M in a.Ms:
        for s in a.seeds:
            try:
                row = run_instance(M, s, a.N, a.tf)
            except Exception as exc:  # scenario placement failures etc.
                row = dict(M=M, seed=s, status=f"error: {str(exc)[:80]}")
            rows.append(row)
            print(json.dumps(row), flush=True)
    p = Path(__file__).with_name("results") / "benchmark_methods5.json"
    p.write_text(json.dumps(rows, indent=1))
    print(f"wrote {p}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
