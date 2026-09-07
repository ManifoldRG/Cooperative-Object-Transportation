"""Solve ONE (scenario_id, method) at 10xT/60s, rollout-verify, write JSON,
then os._exit(0) to skip CasADi's segfaulting interpreter teardown.
Usage: python task_worker.py <scenario_id> <method> <out_dir>
Repo location comes from env COT_REPO."""
import sys, os, json, time

import numpy as np

# repo root inferred from this file's location (scripts/tenxT/ -> repo);
# COT_REPO env var overrides.
REPO = os.environ.get("COT_REPO") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "csf"))

BUDGET = 60.0
T_MULT = 10


def main():
    sid, method, out_dir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    import pandas as pd
    import baseline_comparison as bcmod
    from spacecraft_libraries.solvers.gradient_descent import (
        solve_centralized_gd, solve_decentralized_gd)
    from spacecraft_libraries.solvers.centralized_nlp_th import solve_centralized_nlp_th
    from scipy.spatial.transform import Rotation
    from spacecraft_libraries.new_opts import th_psi_matrix, state_attitude_to_phi

    r = pd.read_csv(os.path.join(REPO, "CSF Runs", "results_20260823",
                                 "baseline_large_20_scenarios_th.csv"))
    r = r[r.global_scenario_id == sid].iloc[0]
    scenario = dict(scenario_id=sid, seed=int(r.seed),
                    thrust_angle_deg=float(r.thrust_angle_deg),
                    mu=float(r.mu), a=float(r.a), e=float(r.e), nu=float(r.nu),
                    I_diag=json.loads(r.I_diag), m=float(r.m),
                    N=int(r.N) * T_MULT,
                    tf=float(r.tf), epsilon=float(r.epsilon), rs=json.loads(r.rs),
                    x0_r=json.loads(r.x0_r), x0_v=json.loads(r.x0_v),
                    x0_phi=json.loads(r.x0_phi), x0_omega=json.loads(r.x0_omega),
                    xf_r=json.loads(r.xf_r), xf_v=json.loads(r.xf_v),
                    xf_phi=json.loads(r.xf_phi), xf_omega=json.loads(r.xf_omega))
    sys_params, bc, epsilon = bcmod.make_sys_bc(scenario)
    seed = bcmod.derive_solver_seed(scenario, BUDGET, 42)

    row = dict(scenario_id=sid, thrust_angle_deg=scenario["thrust_angle_deg"],
               method=method, n_steps=scenario["N"], time_limit_s=BUDGET)
    t0 = time.perf_counter()
    try:
        if method == "centralized_gd":
            res = solve_centralized_gd(sys_params, bc, epsilon, seed=seed,
                                       tau_init_scale=0.1, rel_step=0.03,
                                       max_runtime_s=BUDGET)
            status, conv = "", ""
        elif method == "decentralized_gd":
            res = solve_decentralized_gd(sys_params, bc, epsilon, base_seed=seed,
                                         tau_init_scale=0.1, rel_step=0.03,
                                         max_runtime_s=BUDGET)
            status, conv = "", ""
        else:
            res = solve_centralized_nlp_th(sys_params, bc, epsilon,
                                           max_runtime_s=BUDGET)
            status, conv = res["ipopt_status"], str(res["converged"])
    except Exception as e:
        row.update(wall_s=round(time.perf_counter() - t0, 1),
                   cost_reported=None, cost_rollout=None, V_rollout=None,
                   ipopt_status=f"ABSTAIN:{type(e).__name__}", converged="")
        _write(out_dir, sid, method, row)
        os._exit(0)

    wall = time.perf_counter() - t0
    U = np.asarray(res["control"], dtype=float)

    # independent rollout
    N = sys_params.N
    dt = bc.tf / N
    m = sys_params.m
    I = np.asarray(sys_params.I, dtype=float)
    I_inv = np.linalg.inv(I)
    rs = [np.asarray(x, float) for x in sys_params.rs]
    rr = np.asarray(bc.x0.r, float).copy()
    v = np.asarray(bc.x0.v, float).copy()
    R = Rotation.from_rotvec(state_attitude_to_phi(bc.x0)).as_matrix()
    w = np.asarray(bc.x0.omega, float).copy()
    for k in range(N):
        Uk = U[:, k, :]
        thrust_body = Uk.sum(axis=0)
        torque = sum(np.cross(rs[i], Uk[i]) for i in range(len(rs)))
        Psi = np.asarray(th_psi_matrix(sys_params.mu, sys_params.a,
                                       sys_params.e, k * dt))
        r_new = rr + dt * v
        v_new = v + dt * (Psi[3:6, :] @ np.concatenate([rr, v]) + (R @ thrust_body) / m)
        R = R @ Rotation.from_rotvec(dt * w).as_matrix()
        w = w + dt * (I_inv @ (torque - np.cross(w, I @ w)))
        rr, v = r_new, v_new
    Rf = Rotation.from_rotvec(state_attitude_to_phi(bc.xf)).as_matrix()
    att_err = np.linalg.norm(Rotation.from_matrix(R.T @ Rf).as_rotvec())
    V = (np.linalg.norm(rr - np.asarray(bc.xf.r, float))
         + np.linalg.norm(v - np.asarray(bc.xf.v, float))
         + att_err + np.linalg.norm(w - np.asarray(bc.xf.omega, float)))

    row.update(wall_s=round(wall, 1), cost_reported=float(res["cost"]),
               cost_rollout=float(np.sum(U ** 2)), V_rollout=float(V),
               ipopt_status=status, converged=conv)
    _write(out_dir, sid, method, row)
    os._exit(0)  # skip CasADi teardown (segfaults at N~900)


def _write(out_dir, sid, method, row):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f".{sid}_{method}.tmp")
    dst = os.path.join(out_dir, f"{sid}_{method}.json")
    with open(tmp, "w") as f:
        json.dump(row, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


if __name__ == "__main__":
    main()
