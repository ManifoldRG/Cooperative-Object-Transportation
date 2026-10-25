"""Price diagnostic along the verified SCP solutions.

Per payload and step, with the solution's body thrusts u_{p,k} and attitude
R_k known, compare three per-step floors on the actual fuel Sigma||u_p||:
  ||G_k||          G = Sigma u_p (body force; = inertial force norm)  -- what
                   the current LB prices
  pi(G_k)          min fuel to make body force G under the cones, torque
                   free (exact SOCP)                                   -- what
                   a perfectly-tight attitude-aware price could charge
  max(pi, ||tau||/c)  adds the lever-arm torque floor
Aggregates show how much of the gap J_final - Sigma_ell an ideal
(known-attitude) directional price could certify; the geodesic angles from
R_a/R_b at the expensive steps show whether endpoint reachability balls can
reach it.

Usage: py measure_price.py [--M 5] [--seeds 3 5 7]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import scp_oracle as so  # noqa: E402

import cvxpy as cp  # noqa: E402

from spacecraft_libraries.new_opts import so3_exp, so3_log  # noqa: E402


class ForcePrice:
    """pi(G) = min Sigma||u_p|| s.t. Sigma u_p = G, cones; torque free."""

    def __init__(self, pts, nu):
        self.A = len(pts)
        self.G = cp.Parameter(3)
        self.U = [cp.Variable(3) for _ in range(self.A)]
        cons = [sum(self.U) == self.G]
        cosnu = float(np.cos(nu))
        for p, pt in enumerate(pts):
            ph = np.asarray(pt) / np.linalg.norm(pt)
            cons.append(cosnu * cp.norm(self.U[p]) <= -(ph @ self.U[p]))
        self.prob = cp.Problem(
            cp.Minimize(sum(cp.norm(u) for u in self.U)), cons)

    def __call__(self, G):
        self.G.value = np.asarray(G, float)
        self.prob.solve(solver="CLARABEL")
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return float("nan")
        return float(self.prob.value)


def run(seed, N, tf, M):
    row, sol = so.run(seed, N, tf, M, return_sol=True)
    if sol is None:
        return dict(seed=seed, status="no verified SCP solution"), None
    sc, pts, Us, Rbars = sol["sc"], sol["pts"], sol["Us"], sol["Rbars"]
    out = dict(seed=seed, LB=row["LB"], J_final=row["J_final"])
    per_step = []  # summed over payloads: [fuel, ||G||, pi, max(pi,tau/c)]
    ang_a, ang_w = [], []  # per step: min over payloads angle-to-nearer-BC,
    #                        fuel-weighted via the price excess
    tot = dict(fuel=0.0, nG=0.0, pi=0.0, pimax=0.0)
    step = np.zeros((sc.N, 4))
    excess_w_angle = []
    for j in range(M):
        P0 = sc.payloads[j]
        price = ForcePrice(pts[j], sc.nu)
        ell = max(np.linalg.norm(p) for p in pts[j])
        c = ell * np.sin(sc.nu)
        Ra = so3_exp(np.asarray(P0.x0.phi, float))
        Rb = so3_exp(np.asarray(P0.xf.phi, float))
        for k in range(sc.N):
            u = Us[j][:, k, :]
            fuel = float(np.linalg.norm(u, axis=1).sum())
            G = u.sum(axis=0)
            nG = float(np.linalg.norm(G))
            pi = price(G) if nG > 1e-9 else 0.0
            tau = sum(np.cross(pts[j][p], u[p]) for p in range(len(pts[j])))
            pim = max(pi, float(np.linalg.norm(tau)) / c)
            step[k] += [fuel, nG, pi, pim]
            tot["fuel"] += fuel
            tot["nG"] += nG
            tot["pi"] += pi
            tot["pimax"] += pim
            R = Rbars[j][k]
            th = min(np.linalg.norm(so3_log(Ra.T @ R)),
                     np.linalg.norm(so3_log(Rb.T @ R)))
            excess_w_angle.append((pi - nG, float(np.degrees(th)), k))
    out["totals"] = {k: round(v, 2) for k, v in tot.items()}
    gap_total = out["J_final"] - out["LB"]
    out["price_excess"] = round(tot["pi"] - tot["nG"], 2)
    out["price_excess_frac_of_gap"] = round(
        (tot["pi"] - tot["nG"]) / gap_total, 3)
    out["torque_floor_adds"] = round(tot["pimax"] - tot["pi"], 2)
    # where does the excess live? fuel-excess-weighted angle to nearest BC
    ex = np.array([e for e, _, _ in excess_w_angle])
    th = np.array([t for _, t, _ in excess_w_angle])
    kk = np.array([k for _, _, k in excess_w_angle])
    w = np.maximum(ex, 0)
    if w.sum() > 1e-9:
        out["excess_weighted_angle_deg"] = round(float((w * th).sum() / w.sum()), 1)
        out["excess_weighted_step"] = round(float((w * kk).sum() / w.sum()), 1)
    out["per_step"] = step.round(3).tolist()
    return out, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        out, _ = run(s, a.N, a.tf, a.M)
        rows.append(out)
        print(json.dumps({k: v for k, v in out.items() if k != "per_step"}),
              flush=True)
    out = Path(__file__).with_name("results") / f"price_diag_M{a.M}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
