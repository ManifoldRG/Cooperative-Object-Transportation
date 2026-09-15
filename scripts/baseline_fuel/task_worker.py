"""Solve ONE (scenario_id, method, budget) under the FUEL objective,
rollout-verify, write JSON, then os._exit(0) (skips CasADi's segfaulting
interpreter teardown).
Usage: python task_worker.py <scenario_id> <method> <budget_s> <t_mult> <out_dir>
Repo root inferred from this file's location; COT_REPO env overrides."""
import sys, os, json, time

import numpy as np

REPO = os.environ.get("COT_REPO") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "csf"))


def main():
    sid = int(sys.argv[1])
    method = sys.argv[2]
    budget = float(sys.argv[3])
    t_mult = int(sys.argv[4])
    out_dir = sys.argv[5]

    import pandas as pd
    import baseline_comparison as bcmod
    from spacecraft_libraries.solvers.gradient_descent import (
        solve_centralized_gd, solve_decentralized_gd)
    from spacecraft_libraries.solvers.centralized_nlp_th import solve_centralized_nlp_th
    from spacecraft_libraries.solvers.greedy_sampler import (
        solve_centralized_gs, solve_decentralized_gs)
    from spacecraft_libraries.solvers.centralized_mppi import solve_centralized_mppi
    from spacecraft_libraries.solvers.decentralized_mppi import solve_decentralized_mppi
    from scipy.spatial.transform import Rotation
    from spacecraft_libraries.new_opts import th_psi_matrix, state_attitude_to_phi

    r = pd.read_csv(os.path.join(REPO, "CSF Runs", "results_20260823",
                                 "baseline_large_20_scenarios_th.csv"))
    r = r[r.global_scenario_id == sid].iloc[0]
    scenario = dict(scenario_id=sid, seed=int(r.seed),
                    thrust_angle_deg=float(r.thrust_angle_deg),
                    mu=float(r.mu), a=float(r.a), e=float(r.e), nu=float(r.nu),
                    I_diag=json.loads(r.I_diag), m=float(r.m),
                    N=int(r.N) * t_mult,
                    tf=float(r.tf), epsilon=float(r.epsilon), rs=json.loads(r.rs),
                    x0_r=json.loads(r.x0_r), x0_v=json.loads(r.x0_v),
                    x0_phi=json.loads(r.x0_phi), x0_omega=json.loads(r.x0_omega),
                    xf_r=json.loads(r.xf_r), xf_v=json.loads(r.xf_v),
                    xf_phi=json.loads(r.xf_phi), xf_omega=json.loads(r.xf_omega))
    sys_params, bc, epsilon = bcmod.make_sys_bc(scenario)
    seed = bcmod.derive_solver_seed(scenario, budget, 42)

    row = dict(scenario_id=sid, thrust_angle_deg=scenario["thrust_angle_deg"],
               method=method, n_steps=scenario["N"], time_limit_s=budget)
    t0 = time.perf_counter()
    try:
        if method == "centralized_gd":
            res = solve_centralized_gd(sys_params, bc, epsilon, seed=seed,
                                       tau_init_scale=0.1, rel_step=0.03,
                                       max_runtime_s=budget)
            status, conv = "", ""
        elif method == "decentralized_gd":
            res = solve_decentralized_gd(sys_params, bc, epsilon, base_seed=seed,
                                         tau_init_scale=0.1, rel_step=0.03,
                                         max_runtime_s=budget)
            status, conv = "", ""
        elif method == "centralized_nlp_th":
            res = solve_centralized_nlp_th(sys_params, bc, epsilon,
                                           max_runtime_s=budget)
            status, conv = res["ipopt_status"], str(res["converged"])
        elif method == "centralized_ga":
            # harness defaults (baseline_comparison run_one_solver)
            res = bcmod.run_centralized_ga_seeded(
                sys_params, bc, epsilon, pop_size=10, generations=5000,
                max_runtime_s=budget, seed=seed)
            status, conv = "", ""
        elif method in ("centralized_gs", "decentralized_gs"):
            # harness CLI defaults: sigma .05, step 1.5, tau_init .1,
            # batch 4, white noise
            kw = dict(n_samples=4, sigma=0.05, tau_init_scale=0.1,
                      noise_mode="white", step_size=1.5, max_runtime_s=budget)
            if method == "centralized_gs":
                res = solve_centralized_gs(sys_params, bc, epsilon,
                                           seed=seed, **kw)
            else:
                res = solve_decentralized_gs(sys_params, bc, epsilon,
                                             base_seed=seed, **kw)
            status, conv = "", ""
        elif method in ("centralized_mppi", "decentralized_mppi"):
            # harness defaults: sigma 1.0, lambda .5 (c) / .9 (d),
            # deadline-driven iterations, batch from SAMPLE_SCHEDULE
            ns = bcmod.SAMPLE_SCHEDULE.get(budget, 10)
            if method == "centralized_mppi":
                res = solve_centralized_mppi(
                    sys_params, bc, epsilon, seed=seed,
                    n_iter=bcmod.MPPI_DEADLINE_ITERS, n_samples=ns,
                    sigma=1.0, lambda_=0.5, max_runtime_s=budget)
            else:
                res = solve_decentralized_mppi(
                    sys_params, bc, epsilon, base_seed=seed,
                    n_iter=bcmod.MPPI_DEADLINE_ITERS, n_samples=ns,
                    sigma=1.0, lambda_=0.9, max_runtime_s=budget)
            status, conv = "", ""
        else:
            raise ValueError(f"unknown method {method}")
    except Exception as e:
        row.update(wall_s=round(time.perf_counter() - t0, 1),
                   cost_reported=None, cost_rollout=None, V_rollout=None,
                   ipopt_status=f"ABSTAIN:{type(e).__name__}", converged="")
        _write(out_dir, sid, method, budget, row)
        os._exit(0)

    wall = time.perf_counter() - t0
    # GD/NLP return a raw (agents, N, 3) array; GA/GS/MPPI wrap the same
    # layout in a ControlHistory dataclass under .U
    ctrl = res["control"]
    U = np.asarray(getattr(ctrl, "U", ctrl), dtype=float)

    # independent rollout through the discrete dynamics
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
        if not (np.isfinite(rr).all() and np.isfinite(v).all()
                and np.isfinite(w).all()):
            break
    try:
        Rf = Rotation.from_rotvec(state_attitude_to_phi(bc.xf)).as_matrix()
        att_err = np.linalg.norm(Rotation.from_matrix(R.T @ Rf).as_rotvec())
        V = (np.linalg.norm(rr - np.asarray(bc.xf.r, float))
             + np.linalg.norm(v - np.asarray(bc.xf.v, float))
             + att_err + np.linalg.norm(w - np.asarray(bc.xf.omega, float)))
    except Exception:
        # divergent controls blow up the rollout numerically: honest failure
        V = float("inf")

    # exact fuel of the returned controls (solvers minimize the smoothed
    # epigraph version of the same quantity)
    row.update(wall_s=round(wall, 1), cost_reported=float(res["cost"]),
               cost_rollout=float(np.linalg.norm(U, axis=2).sum()),
               V_rollout=float(V), ipopt_status=status, converged=conv)
    _write(out_dir, sid, method, budget, row)
    os._exit(0)


def _write(out_dir, sid, method, budget, row):
    os.makedirs(out_dir, exist_ok=True)
    name = f"{sid}_{method}_{int(budget)}"
    tmp = os.path.join(out_dir, f".{name}.tmp")
    dst = os.path.join(out_dir, f"{name}.json")
    with open(tmp, "w") as f:
        json.dump(row, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


if __name__ == "__main__":
    main()
