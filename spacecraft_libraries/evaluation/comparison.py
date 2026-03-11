from __future__ import annotations

import numpy as np

from ..data_structures import BoundaryConditions, StateVector, SystemParams
from ..solvers import solve_centralized_ga, solve_centralized_nlp, solve_decentralized_island_ga
from .metrics import quaternion_aware_violation, terminal_violation


def default_scenario() -> tuple[SystemParams, BoundaryConditions, float]:
    rs = [np.array([0.5, 1.0, 1.5]), np.array([0.0, 0.5, 2.0]), np.array([-0.5, 1.0, -1.5])]
    sys_params = SystemParams(
        mu=3.98e14,
        a=8e6,
        e=0.2,
        nu=np.pi / 4,
        I=1000 * np.diag([1, 2, 3]),
        m=100,
        rs=rs,
        N=20,
    )
    bc = BoundaryConditions(
        x0=StateVector(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), eps=np.array([0, 0, 0, 1]), omega=np.array([0, 0, 0])),
        xf=StateVector(r=np.array([5, 5, 5]), v=np.array([0, 0, 0]), eps=np.array([0.5, 0.5, 0.5, 0.5]), omega=np.array([0, 0, 0])),
        tf=50,
    )
    return sys_params, bc, 1e-5


def _extract_terminal_state(result, method: str):
    if method == "centralized_nlp":
        x = result["state"]
        return {"r": x[-1, 0:3], "v": x[-1, 3:6], "eps": x[-1, 6:10], "omega": x[-1, 10:13]}
    traj = result["trajectory"]
    s = traj.states[-1]
    return {"r": s.r, "v": s.v, "eps": s.eps, "omega": s.omega}


def run_method_comparison(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    max_runtime_s: float | None = None,
):
    results = [
        solve_centralized_nlp(sys_params, bc, max_iters=30, max_runtime_s=max_runtime_s),
        solve_centralized_ga(sys_params, bc, epsilon, pop_size=8, generations=5, max_runtime_s=max_runtime_s),
        solve_decentralized_island_ga(
            sys_params,
            bc,
            epsilon,
            pop_size=8,
            migration_rounds=3,
            max_runtime_s=max_runtime_s,
        ),
    ]

    table = []
    for result in results:
        terminal = _extract_terminal_state(result, result["method"])
        violation = (
            terminal_violation(terminal["r"], bc.xf.r)
            + terminal_violation(terminal["v"], bc.xf.v)
            + quaternion_aware_violation(terminal["eps"], bc.xf.eps)
            + terminal_violation(terminal["omega"], bc.xf.omega)
        )
        table.append(
            {
                "method": result["method"],
                "cost": float(result["cost"]),
                "terminal_violation": float(violation),
                "runtime_s": float(result["runtime"]),
            }
        )
    return table
