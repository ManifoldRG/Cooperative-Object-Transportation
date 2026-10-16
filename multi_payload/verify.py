"""Independent verification: rollout of each payload's controls through the
paper's discrete dynamics (same as scripts/csf/baseline_comparison.py
_rollout_verify), pairwise separation, and the fuel certificate."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from spacecraft_libraries.new_opts import state_attitude_to_phi, th_psi_matrix


def rollout(sys_params, bc, U):
    """Returns dict(r (N+1,3), fuel, V) for controls U (agents, N, 3) in the
    body frame. V = ||dr|| + ||dv|| + attitude geodesic error + ||domega||."""
    N = sys_params.N
    dt = bc.tf / N
    I = np.asarray(sys_params.I, float)
    I_inv = np.linalg.inv(I)
    rs = [np.asarray(x, float) for x in sys_params.rs]
    r = np.asarray(bc.x0.r, float).copy()
    v = np.asarray(bc.x0.v, float).copy()
    R = Rotation.from_rotvec(state_attitude_to_phi(bc.x0)).as_matrix()
    w = np.asarray(bc.x0.omega, float).copy()
    rh = [r.copy()]
    for k in range(N):
        Uk = U[:, k, :]
        torque = sum(np.cross(rs[i], Uk[i]) for i in range(len(rs)))
        Psi = np.asarray(th_psi_matrix(sys_params.mu, sys_params.a, sys_params.e, k * dt))
        r_new = r + dt * v
        v = v + dt * (Psi[3:6, :] @ np.concatenate([r, v]) + (R @ Uk.sum(axis=0)) / sys_params.m)
        R = R @ Rotation.from_rotvec(dt * w).as_matrix()
        w = w + dt * (I_inv @ (torque - np.cross(w, I @ w)))
        r = r_new
        rh.append(r.copy())
        if not (np.isfinite(r).all() and np.isfinite(v).all() and np.isfinite(w).all()):
            return dict(r=np.asarray(rh), fuel=float("nan"), V=float("inf"))
    Rf = Rotation.from_rotvec(state_attitude_to_phi(bc.xf)).as_matrix()
    att = np.linalg.norm(Rotation.from_matrix(R.T @ Rf).as_rotvec())
    V = (np.linalg.norm(r - bc.xf.r) + np.linalg.norm(v - bc.xf.v) + att
         + np.linalg.norm(w - bc.xf.omega))
    return dict(r=np.asarray(rh), fuel=float(np.linalg.norm(U, axis=2).sum()), V=float(V))


def min_separation(paths, substeps=10):
    """Pairwise minimum distance, at the steps and on the linear segments
    between them. Returns (min_dist, (j, l, k_continuous))."""
    best = (np.inf, None)
    s = np.linspace(0.0, 1.0, substeps + 1)[:, None, None]
    for j in range(len(paths)):
        for l in range(j + 1, len(paths)):
            d = paths[j] - paths[l]                       # (N+1,3)
            seg = d[:-1][None] * (1 - s) + d[1:][None] * s  # (substeps+1, N, 3)
            dist = np.linalg.norm(seg, axis=2)
            i = np.unravel_index(np.argmin(dist), dist.shape)
            if dist[i] < best[0]:
                best = (float(dist[i]), (j, l, float(i[1] + s[i[0], 0, 0])))
    return best


def verify(sc, occupied, oracle_out, lower_bound, J_hat, V_tol=1e-6):
    payloads = []
    paths = []
    for j, res in enumerate(oracle_out["payloads"]):
        sp, bc = sc.sys_params_for(j, occupied[j])
        ro = rollout(sp, bc, res["control"])
        paths.append(ro["r"])
        payloads.append(dict(fuel=ro["fuel"], V=ro["V"], max_slack=float(res["slack"].max())))
    dmin, where = min_separation(paths)
    J_final = float(sum(p["fuel"] for p in payloads))
    return dict(
        payloads=payloads, J_final=J_final, lower_bound=float(lower_bound),
        J_hat=float(J_hat), min_separation=dmin, min_separation_at=where,
        delta=sc.delta, collision_free=bool(dmin >= sc.delta),
        all_terminal_ok=bool(all(p["V"] < V_tol for p in payloads)),
        certificate_ok=bool(lower_bound <= J_final + 1e-6
                            and (not np.isfinite(J_hat) or J_final <= J_hat * (1 + 1e-6) + 1e-6)),
        paths=paths)
