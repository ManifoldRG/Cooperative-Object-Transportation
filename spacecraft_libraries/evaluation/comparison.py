from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import os

from ..data_structures import BoundaryConditions, StateVectorLie, SystemParams
from ..solvers import (
    solve_centralized_ga,
    solve_centralized_mppi,
    solve_centralized_nlp,
    solve_decentralized_island_ga,
    solve_decentralized_mppi,
)
from .metrics import lie_attitude_violation, terminal_violation
from ..solver_logger import set_scenario_context, clear_scenario_context, log_nlp_failure
import random
from ..new_opts import so3_log

@contextmanager
def _suppress_solver_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def _render_loading_bar(completed: int, total: int) -> None:
    width = 30
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rRunning solvers [{bar}] {completed}/{total}", end="", flush=True)
    if completed == total:
        print()

def many_agent_scenario_gen(numagents):
    rsfull = [np.array([1 / 2, 1, 3 / 2]),
              np.array([0, 1 / 2, 2]),
              np.array([-1 / 2, 1, -3 / 2]),
              np.array([-1, -1, -1]),
              np.array([0, 1, 0]),
              np.array([0, -1, 0]),
              np.array([0, 1, -1]),
              np.array([1, -1, 0]),
              np.array([1 / 2, 1, -5 / 2]),
              np.array([0, 1 / 2, 2]),
              np.array([1 / 2, 1, -3 / 2]),
              np.array([-1, 1, -1]),
              np.array([0.5, -2., -2.]),
              np.array([-2., -1., 0.]),
              np.array([-0.5, -1.5, 0.5]),
              np.array([-2., -1., -1.]),
              np.array([-0.5, 1.5, -2.]),
              np.array([0., 0.5, -2.5]),
              np.array([-1., -2.5, 1.]),
              np.array([-0.5, -2.5, -0.]),
              np.array([1., -1., 0.]),
              np.array([-0.5, -1.5, 2.]),
              np.array([-1., -1.5, 0.5]),
              np.array([0.5, -1., -2.5]),
              np.array([-1., -1., 1.5]),
              np.array([1., 2.5, -0.]),
              np.array([0.5, 1.5, -0.]),
              np.array([-1., 0., 2.5]),
              np.array([-1.5, -0.5, 1.5]),
              np.array([1., 2., 2.]),
              np.array([-2.5, -1.5, -0.5]),
              np.array([1.53, -1.92, 1.78]),
              np.array([3.04, 1.48, -2.01]),
              np.array([-1.47, 2.54, -0.43]),
              np.array([-2.48, -0.97, 1.09]),
              np.array([1.32, 0.08, -2.67]),
              np.array([-1.02, 1.94, 1.83]),
              np.array([2.43, -1.51, 0.39]),
              np.array([-1.56, 2.39, 1.07]),
              np.array([0.53, -1.42, 2.45]),
              np.array([-2.08, 1.48, -1.56]),
              np.array([2.59, 1.06, -1.58]),
              np.array([-1.98, -1.09, 2.34]),
              np.array([1.01, 1.97, -1.46]),
              np.array([-1.57, -2.46, 1.04]),
              np.array([2.02, 1.54, 1.63]),
              np.array([1.07, -2.36, -1.57]),
              np.array([-1.56, -1.64, -2.37]),
              np.array([2.45, -1.93, 0.54]),
              np.array([-2.53, 0.49, 1.98]),
              np.array([1.47, 2.04, 1.56]),
              np.array([-0.49, 2.47, -1.57]),
              np.array([1.94, -2.42, 0.53]),
              np.array([-1.01, -1.94, 2.04]),
              np.array([1.54, 1.03, 2.36]),
              np.array([2.04, 2.46, -0.56]),
              np.array([-2.42, -1.59, 1.02]),
              np.array([0.58, 1.59, 2.43]),
              np.array([-2.01, 2.45, -1.02]),
              np.array([1.06, -1.57, 2.42]),
              np.array([2.07, -0.49, -2.54]),
              np.array([-1.49, 1.04, -2.53]),
              np.array([1.94, -1.89, -1.03]),
              np.array([0.49, -2.36, 1.54]),
              np.array([-2.01, 0.56, 2.49]),
              np.array([1.57, 2.47, -0.52]),
              np.array([2.09, 1.58, -2.46]),
              np.array([-1.59, -2.46, -1.01]),
              np.array([2.47, -1.01, 1.56]),
              np.array([0.59, -2.01, -2.47]),
              np.array([-1.02, 1.93, -2.46])
              ]
    rand_rs = random.sample(rsfull, numagents)
    rand_sys = SystemParams(mu=3.98e14, a=8e6, e=0.2, nu=np.pi/4, I=1000*np.diag([1,2,3]),m=100, rs=rand_rs, N=20)
    rand_bc = BoundaryConditions(x0=StateVectorLie(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), #changed: quaternion to Twist
        phi=np.array([0, 0, 0]),omega=np.array([0, 0, 0])),
         xf=StateVectorLie(r=np.array([5, 5, 5]), v=np.array([0, 0, 0]), phi=np.array([1.20919958, 1.20919958, 1.20919958]),#note: check angle
                        omega=np.array([0, 0, 0])),tf=50)

    print("This is the scenario generator for random agents for scaling tests")

    # rand_bc = BoundaryConditions(~)
    return rand_sys,rand_bc, 1e-5

def random_scenario_generator():
    """
    Params:
    - a : semi-major axis - 1.1 - 1.3
    - e : Eccentricity - 0.01 - 0.3
    - J : Inertia Tensor - random diagonal matrix with a maximum of 500 #change to 500
    - m : Payload Mass - random from 1 - 500 Kgs #change to 500
    - rb : Final position - random location within 1Km radius to ensure linearised dyanmics hold constant
    - epsilon_b : Final attitude quaternion - any random quaternion
    - tf : final time : random time within 1 - 10 mins # change to larger (10-60 mins)
    - agent placement : random within 10m radius sphere
    - number of agents  : 3 - 30, random value with max 30 min 3.
    """

    # Semi-major axis: 1.1–1.3 × Earth radius
    R_earth = 6.371e6
    a = random.uniform(1.1, 1.3) * R_earth

    # Eccentricity
    e = random.uniform(0.01, 0.3)

    # Inertia tensor: random diagonal, each principal moment in (1, 5000)
    diag_vals = np.array([random.uniform(1, 5000) for _ in range(3)]) # change to 500
    # Ensure valid inertia (triangle inequality: each < sum of other two)
    diag_vals = np.sort(diag_vals)  # sort so triangle inequality is easier to satisfy
    while diag_vals[2] >= diag_vals[0] + diag_vals[1]:
        diag_vals = np.sort(np.array([random.uniform(1, 5000) for _ in range(3)]))
    J = np.diag(diag_vals)

    # Payload mass (kg)
    m = random.uniform(1, 5000) # change to 500

    # Final position: random point within 1km radius sphere
    r_mag = random.uniform(0, 1000)  # metres
    r_dir = np.random.randn(3)
    r_dir /= np.linalg.norm(r_dir)
    rb = r_dir * r_mag

    # Final attitude: random unit quaternion [x, y, z, w] via Gaussian sampling
    #q_raw = np.random.randn(4)
    # epsilon_b = q_raw / np.linalg.norm(q_raw)


    # generating the quaternion first and then converting to twist
    q_raw = np.random.randn(4)
    q_raw /= np.linalg.norm(q_raw)
    x, y, z, w = q_raw
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ])
    phi_b = so3_log(R)

    # Final time: 1–10 minutes in seconds
    tf = random.uniform(60, 300)

    # Number of agents and their positions (within 10m radius sphere)
    n_agents = random.randint(3, 10) #! currently set to small vals for testing
    rs = []
    for _ in range(n_agents):
        mag = random.uniform(0, 10)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        rs.append(direction * mag)

        # Number of timesteps — scale loosely with tf so discretisation stays reasonable
    N = max(20, int(tf / 2))

    epsilon = random.uniform(1e-6, 1e-4)

    sys_params = SystemParams(mu=3.98e14, a=a,e=e,nu=np.pi / 4, I=J, m=m, rs=rs, N=N)

    bc = BoundaryConditions( x0=StateVectorLie(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), #changed: quaternion to Twist
                                            phi=np.array([0, 0, 0]),omega=np.array([0, 0, 0]) ),
         xf=StateVectorLie(r=rb, v=np.array([0, 0, 0]), ##note: check twist angle for final state
                        phi=phi_b, omega=np.array([0, 0, 0]) ), tf=tf)

    return sys_params, bc, epsilon



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
        x0=StateVectorLie(r=np.array([0,0,0]), v=np.array([0,0,0]), phi=np.zeros(3), omega=np.zeros(3)),
        xf=StateVectorLie(r=np.array([5,5,5]), v=np.array([0,0,0]), phi=np.array([1.20919958, 1.20919958, 1.20919958]), omega=np.zeros(3)),
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
        x0=StateVectorLie(r=np.array([0,0,0]), v=np.array([0,0,0]), phi=np.zeros(3), omega=np.zeros(3)),
        xf=StateVectorLie(r=np.array([7,4,6]), v=np.array([0,0,0]), phi=np.array([0.0, np.pi/2, 0.0]), omega=np.zeros(3)),
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
        x0=StateVectorLie(r=np.array([0,0,0]), v=np.array([0,0,0]), phi=np.zeros(3), omega=np.zeros(3)),
        xf=StateVectorLie(r=np.array([4,6,3]), v=np.array([0,0,0]), phi=np.array([np.pi/2, -np.pi/2, np.pi/2]), omega=np.zeros(3)),
        tf=45,
    )
    return sys_params, bc, 5e-5


def scaling_base_scenario() -> tuple[SystemParams, BoundaryConditions, float]:
    """
    Generate a randomised base scenario for the agent-scaling study.
    Calls random_scenario_generator() once to fix all orbital and dynamics
    parameters (a, e, I, m, BCs, tf, epsilon, N).

    The returned sys_params has rs=[] — attachment vectors are intentionally
    empty.  Pass the returned values to scaling_inject_agents() to produce a
    fully populated SystemParams for a specific agent count.

    This separation guarantees that a, e, I, m, BCs, tf, epsilon are
    identical across every agent count in the sweep.
    """
    sys_params, bc, epsilon = random_scenario_generator()
    sys_params.rs = []   # strip agents — caller injects them per sweep step
    return sys_params, bc, epsilon


def scaling_inject_agents(base_sys: SystemParams,base_bc: BoundaryConditions,epsilon: float,n_agents: int) -> tuple[SystemParams, BoundaryConditions, float]:
    """
    Clone base_sys and populate rs with n_agents freshly randomised attachment
    vectors using the same placement rule as random_scenario_generator:
    uniform magnitude in [0, 10] m, random unit direction.

    All other parameters are taken unchanged from base_sys.
    """
    rs = []
    for _ in range(n_agents):
        mag = random.uniform(0, 10)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        rs.append(direction * mag)

    sys_params = SystemParams(
        mu=base_sys.mu,
        a=base_sys.a,
        e=base_sys.e,
        nu=base_sys.nu,
        I=base_sys.I,
        m=base_sys.m,
        rs=rs,
        N=base_sys.N,
    )
    return sys_params, base_bc, epsilon


def get_scenario(scenario_id: int, numagents=3) -> tuple[SystemParams, BoundaryConditions, float]:
    scenarios = {1: scenario_1, 2: scenario_2, 3: scenario_3}
    if scenario_id == 4:
        return many_agent_scenario_gen(numagents=numagents)
    if scenario_id == 5:
        return random_scenario_generator()
    if scenario_id not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario_id}'. Valid options are: 1, 2, 3.")
    return scenarios[scenario_id]()


def default_scenario() -> tuple[SystemParams, BoundaryConditions, float]:
    return scenario_1()


def _extract_terminal_state(result, method: str):
    if method == "centralized_nlp":
        x = result["state"]
        # 12-col layout: [r(3), v(3), phi(3), omega(3)]
        return {"r": x[-1, 0:3], "v": x[-1, 3:6], "phi": x[-1, 6:9], "omega": x[-1, 9:12]}
    traj = result["trajectory"]
    s = traj.states[-1]
    return {"r": s.r, "v": s.v, "phi": s.phi, "omega": s.omega}


def run_method_comparison(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    max_runtime_s: float | None = None,
    show_progress: bool = False,
    silence_solver_output: bool = True,
    mppi_iterations: int = 5,
    mppi_samples: int = 10,
    mppi_sigma: float = 1e-1,
    mppi_lambda: float = 1.0,
    mppi_base_seed: int = 42,
    include_mppi: bool = True,
):
    solver_calls = [
        lambda: solve_centralized_nlp(sys_params, bc, max_iters=3000, max_runtime_s=max_runtime_s),
        lambda: solve_centralized_ga(sys_params, bc, epsilon, pop_size=10, generations=5000, max_runtime_s=max_runtime_s),
        lambda: solve_decentralized_island_ga(
            sys_params,
            bc,
            epsilon,
            pop_size=5,
            migration_rounds=5000,
            max_runtime_s=max_runtime_s,
        ),
    ]
    method_names = ["centralized_nlp", "centralized_ga", "decentralized_island_ga"]

    if include_mppi:
        solver_calls.extend([
            lambda: solve_centralized_mppi(
                sys_params,
                bc,
                epsilon,
                n_iter=mppi_iterations,
                n_samples=mppi_samples,
                sigma=mppi_sigma,
                lambda_=mppi_lambda,
                seed=mppi_base_seed,
                max_runtime_s=max_runtime_s,
            ),
            lambda: solve_decentralized_mppi(
                sys_params,
                bc,
                epsilon,
                n_iter=mppi_iterations,
                n_samples=mppi_samples,
                sigma=mppi_sigma,
                lambda_=mppi_lambda,
                base_seed=mppi_base_seed,
                max_runtime_s=max_runtime_s,
            ),
        ])
        method_names.extend(["centralized_mppi", "decentralized_mppi"])

    results = []
    total = len(solver_calls)
    if show_progress:
        _render_loading_bar(0, total)

    for idx, (solver, method_name) in enumerate(zip(solver_calls, method_names), start=1):
        set_scenario_context(
            method=method_name,
            n_agents=len(sys_params.rs),
            a=sys_params.a, e=sys_params.e, m=sys_params.m, tf=bc.tf,
        )
        try:
             with _suppress_solver_output(silence_solver_output):
                    results.append(solver())
        except Exception as exc:
            import traceback
            log_nlp_failure(
                function_name=method_name,
                solver_status=type(exc).__name__,
                context={"error": str(exc)[:120]},
            )
            print(f"  [{method_name}] failed — {exc}")
            traceback.print_exc()
            results.append({
                "method": method_name,
                "cost": float("nan"),
                "runtime": float("nan"),
                "trajectory": None,
                "state": None,
            })
        finally:
            clear_scenario_context()
        if show_progress:
            _render_loading_bar(idx, total)

    table = []
    for result in results:
        if result.get("cost") is None or (isinstance(result["cost"], float) and np.isnan(result["cost"])):
            table.append({
                "method": result["method"],
                "cost": float("nan"),
                "terminal_violation": float("nan"),
                "runtime_s": float("nan"),
            })
            continue
        terminal = _extract_terminal_state(result, result["method"])
        violation = (
            terminal_violation(terminal["r"], bc.xf.r)
            + terminal_violation(terminal["v"], bc.xf.v)
            + lie_attitude_violation(terminal["phi"], bc.xf.phi)
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
