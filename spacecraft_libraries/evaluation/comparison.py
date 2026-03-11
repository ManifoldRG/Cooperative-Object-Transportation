from __future__ import annotations

import numpy as np

from ..data_structures import BoundaryConditions, StateVector, SystemParams
from ..solvers import solve_centralized_ga, solve_centralized_nlp, solve_decentralized_island_ga
from .metrics import quaternion_aware_violation, terminal_violation


def scenario_1() -> tuple[SystemParams, BoundaryConditions, float]:
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


def scenario_2() -> tuple[SystemParams, BoundaryConditions, float]:
    rs = [np.array([1.0, -0.5, 1.0]), np.array([-1.0, 0.75, 1.5]), np.array([0.25, 1.2, -1.25])]
    sys_params = SystemParams(
        mu=3.98e14,
        a=8e6,
        e=0.35,
        nu=np.pi / 3,
        I=800 * np.diag([1.0, 1.5, 2.5]),
        m=120,
        rs=rs,
        N=25,
    )
    bc = BoundaryConditions(
        x0=StateVector(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), eps=np.array([0, 0, 0, 1]), omega=np.array([0, 0, 0])),
        xf=StateVector(r=np.array([7, 4, 6]), v=np.array([0, 0, 0]), eps=np.array([0.0, 0.707, 0.0, 0.707]), omega=np.array([0, 0, 0])),
        tf=60,
    )
    return sys_params, bc, 1e-4


def scenario_3() -> tuple[SystemParams, BoundaryConditions, float]:
    rs = [np.array([-0.75, 1.5, 0.5]), np.array([0.5, -1.0, 2.2]), np.array([1.0, 0.75, -1.8])]
    sys_params = SystemParams(
        mu=3.98e14,
        a=8.5e6,
        e=0.1,
        nu=np.pi / 6,
        I=1200 * np.diag([0.8, 1.4, 2.0]),
        m=90,
        rs=rs,
        N=30,
    )
    bc = BoundaryConditions(
        x0=StateVector(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), eps=np.array([0, 0, 0, 1]), omega=np.array([0, 0, 0])),
        xf=StateVector(r=np.array([4, 6, 3]), v=np.array([0, 0, 0]), eps=np.array([0.5, -0.5, 0.5, 0.5]), omega=np.array([0, 0, 0])),
        tf=45,
    )
    return sys_params, bc, 5e-5


def get_scenario(scenario_id: int) -> tuple[SystemParams, BoundaryConditions, float]:
    scenarios = {1: scenario_1, 2: scenario_2, 3: scenario_3}
    if scenario_id not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario_id}'. Valid options are: 1, 2, 3.")
    return scenarios[scenario_id]()


def default_scenario() -> tuple[SystemParams, BoundaryConditions, float]:
    return scenario_1()


def _extract_terminal_state(result, method: str):
    if method == "centralized_nlp":
        x = result["state"]
        return {"r": x[-1, 0:3], "v": x[-1, 3:6], "eps": x[-1, 6:10], "omega": x[-1, 10:13]}
    traj = result["trajectory"]
    s = traj.states[-1]
    return {"r": s.r, "v": s.v, "eps": s.eps, "omega": s.omega}


def run_method_comparison(sys_params: SystemParams, bc: BoundaryConditions, epsilon: float, progress_callback=None):
    methods = [
        ("centralized_nlp", lambda: solve_centralized_nlp(sys_params, bc, max_iters=30)),
        ("centralized_ga", lambda: solve_centralized_ga(sys_params, bc, epsilon, pop_size=8, generations=5)),
        ("decentralized_island_ga", lambda: solve_decentralized_island_ga(sys_params, bc, epsilon, pop_size=8, migration_rounds=3)),
    ]

    results = []
    total_methods = len(methods)
    for index, (method_name, solver) in enumerate(methods, start=1):
        if progress_callback is not None:
            progress_callback(index - 1, total_methods, method_name)
        results.append(solver())
        if progress_callback is not None:
            progress_callback(index, total_methods, method_name)

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
