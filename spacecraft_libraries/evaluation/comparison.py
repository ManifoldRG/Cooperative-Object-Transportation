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
from ..solvers.centralized_nlp import solve_centralized_nlp_warm  # changed: added import for warm-start NLP solver
from .metrics import lie_attitude_violation, terminal_violation
from ..solver_logger import set_scenario_context, clear_scenario_context, log_nlp_failure
import random
from ..new_opts import so3_log
from ..closed_loop.faults import FaultEvent


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

def many_agent_scenario_gen(numagents, seed: int | None = None):
    rng = random.Random(seed) if seed is not None else random
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
    rand_rs = rng.sample(rsfull, numagents)
    rand_sys = SystemParams(mu=3.98e14, a=8e6, e=0.2, nu=np.pi/4, I=1000*np.diag([1,2,3]),m=100, rs=rand_rs, N=20)
    rand_bc = BoundaryConditions(x0=StateVectorLie(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), #changed: quaternion to Twist
        phi=np.array([0, 0, 0]),omega=np.array([0, 0, 0])),
         xf=StateVectorLie(r=np.array([5, 5, 5]), v=np.array([0, 0, 0]), phi=np.array([1.20919958, 1.20919958, 1.20919958]),#note: check angle
                        omega=np.array([0, 0, 0])),tf=50)

    print("This is the scenario generator for random agents for scaling tests")

    # rand_bc = BoundaryConditions(~)
    return rand_sys,rand_bc, 1e-5


def sample_inertia_tensor(m: float, L: float, max_tries: int = 10000, rng: random.Random = None) -> np.ndarray:
    """
    Generating physically valid inertia tensor for a body of mass m
    that fits within a sphere of radius L (sampled per call).

    Constraints enforced (all necessary for a real rigid body):
    - 0 <= I_i <= m * L^2            (no mass element can sit beyond radius L)
    - I_1 + I_2 + I_3 <= 2 * m * L^2  (sum bound from integrating over the body)
    - triangle inequality: I_i < I_j + I_k for all permutations
    """
    I_max = m * L ** 2 # setting the max limit
    for _ in range(max_tries):
        _rng = rng if rng is not None else random
        vals = np.sort([_rng.uniform(0, I_max) for _ in range(3)]) #automatically ensuring first cond satisfied
        if vals.sum() > 2 * I_max: # second conditions
            continue
        if vals[2] >= vals[0] + vals[1]: #third condition
            continue
        return np.diag(vals)
    raise RuntimeError("sample_inertia_tensor: failed to find valid sample in max_tries")

def random_scenario_generator(fixed_agents_num: int = -1, 
                              seed: int | None = 42, 
                              thrust_angle: float = np.pi / 3):
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
        - number of agents  : 3 - 6, random value with max 6 min 3. [capped for baseline comparison tractability]
        - seed : if given, all draws in this call come from a local RNG scoped to
          this call. If None, falls back to the global random/np.random state
          (unchanged behavior — existing scripts that seed globally still work).
        """
    rng    = random.Random(seed) if seed is not None else random
    np_rng = np.random.default_rng(seed) if seed is not None else np.random

    # Semi-major axis: 1.1–1.3 × Earth radius
    R_earth = 6.371e6
    a = rng.uniform(1.1, 1.3) * R_earth

    # Eccentricity
    e = rng.uniform(0.01, 0.3)

    # # Inertia tensor: random diagonal, each principal moment in (1, 5000)
    # diag_vals = np.array([random.uniform(1, 500) for _ in range(3)])
    # # Ensure valid inertia (triangle inequality: each < sum of other two)
    # diag_vals = np.sort(diag_vals)  # sort so triangle inequality is easier to satisfy
    # while diag_vals[2] >= diag_vals[0] + diag_vals[1]:
    #     diag_vals = np.sort(np.array([random.uniform(1, 5000) for _ in range(3)]))
    #J = np.diag(diag_vals)
    m = rng.uniform(1, 500)
    #J = sample_inertia_tensor(m)

    # Payload mass (kg)


    # Final position: random point within 1km radius sphere
    r_mag = rng.uniform(0, 1000)  # metres
    r_dir = np_rng.standard_normal(3)
    r_dir /= np.linalg.norm(r_dir)
    rb = r_dir * r_mag

    # Final attitude: random unit quaternion [x, y, z, w] via Gaussian sampling
    # q_raw = np.random.randn(4)
    # epsilon_b = q_raw / np.linalg.norm(q_raw)

    # generating the quaternion first and then converting to twist
    q_raw = np_rng.standard_normal(4)
    q_raw /= np.linalg.norm(q_raw)
    x, y, z, w = q_raw
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ])
    phi_b = so3_log(R)

    # Final time: 1–10 minutes in seconds
    tf = rng.uniform(300, 900)

    # Number of agents and their positions (within 10m radius sphere)
    if fixed_agents_num <= 0:
        n_agents = rng.randint(3, 6)  # changed: agent count cap reduced from 30 to 6 for baseline tractability
    else:
        n_agents = fixed_agents_num

    rs = []
    for _ in range(n_agents):
        mag = rng.uniform(5, 50)
        direction = np_rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        rs.append(direction * mag)

    L = max(np.linalg.norm(r) for r in rs)

    # Inertia tensor
    J = sample_inertia_tensor(m, L, rng=rng)

    # Number of timesteps — scale loosely with tf so discretisation stays reasonable
    N = max(20, int(tf / 10)) #change to tf/10

    epsilon = rng.uniform(1e-6, 1e-4)

    sys_params = SystemParams(mu=3.98e14, a=a, e=e, nu=thrust_angle, I=J, m=m, rs=rs, N=N)  # changes to pi/2

    bc = BoundaryConditions(x0=StateVectorLie(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]),
                                              phi=np.array([0, 0, 0]), omega=np.array([0, 0, 0])),
                            xf=StateVectorLie(r=rb, v=np.array([0, 0, 0]),
                                              phi=phi_b, omega=np.array([0, 0, 0])), tf=tf)

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
        x0=StateVectorLie(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), phi=np.zeros(3), omega=np.zeros(3)),
        xf=StateVectorLie(r=np.array([5, 5, 5]), v=np.array([0, 0, 0]),
                          phi=np.array([1.20919958, 1.20919958, 1.20919958]), omega=np.zeros(3)),
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
        x0=StateVectorLie(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), phi=np.zeros(3), omega=np.zeros(3)),
        xf=StateVectorLie(r=np.array([7, 4, 6]), v=np.array([0, 0, 0]), phi=np.array([0.0, np.pi / 2, 0.0]),
                          omega=np.zeros(3)),
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
        x0=StateVectorLie(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), phi=np.zeros(3), omega=np.zeros(3)),
        xf=StateVectorLie(r=np.array([4, 6, 3]), v=np.array([0, 0, 0]),
                          phi=np.array([np.pi / 2, -np.pi / 2, np.pi / 2]), omega=np.zeros(3)),
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
    sys_params.rs = []  # strip agents — caller injects them per sweep step
    return sys_params, bc, epsilon

def scaling_inject_agents(base_sys: SystemParams, base_bc: BoundaryConditions, epsilon: float, n_agents: int, seed: int | None = None) -> tuple[SystemParams, BoundaryConditions, float]:
    """
    Clone base_sys and populate rs with n_agents freshly randomised attachment
    vectors using the same placement rule as random_scenario_generator:
    uniform magnitude in [0, 10] m, random unit direction.

    All other parameters are taken unchanged from base_sys.
    """
    rng    = random.Random(seed) if seed is not None else random
    np_rng = np.random.default_rng(seed) if seed is not None else np.random

    rs = []
    for _ in range(n_agents):
        mag = rng.uniform(0, 10)
        direction = np_rng.standard_normal(3)
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


# it output a map of { agent_id : delay_time }
# each agent can have a unique random delay time for mode="random", and non-zero extra_time_step
def comms_delay_generator(sys_params: SystemParams,
                          mode: str = "fixed",
                          delay_time_step: int = 2,
                          extra_time_step: int = 0) -> dict[int, int]:
    if mode not in {"fixed", "random"}:
        raise ValueError("mode must be either fixed or random")

    if delay_time_step < 0 or extra_time_step < 0:
        raise ValueError("delay_time_step and extra_time_step must be non-negative")

    num_agents = len(sys_params.rs)
    agents_comms_delay_step_map: dict[int, int] = {}

    for aid in range(num_agents):
        if (mode == "random"):
            agents_comms_delay_step_map[aid] = delay_time_step + random.randint(0, extra_time_step)
        else:
            agents_comms_delay_step_map[aid] = delay_time_step

    return agents_comms_delay_step_map


#    Generate random fault scenarios for testing swarm recovery simulations.
#
#    Supports three fault models:
#        - random: randomly selected failed agents
#        - localized: agents nearest a random seed fail
#        - clustered: agents within a radius of one or more seeds fail
#    The first returned entry is always a no-fault scenario ([]).
#
#
#    Usage:
#        - Intened to be used after get_scenario. Here are some examples
#           all_fault_events = random_dropout_fault_generator( sys_params, bc.tf, 10, "clustered", at_least_n_survivors=2, affected_radius=1.0 )
#           all_fault_events = random_dropout_fault_generator( sys_params, bc.tf, 10, "random", at_least_n_survivors=2 )
#        - then we pass the events into run_recovery_sim
#            result = run_recovery_episode(sys_params, bc, epsilon, fault_events=fevent, cfg=cfg, verbose=True)
#
#    Note:
#        - num_seeds and affected_radius are only used by the clustered model.
#        - trigger time is chosen randomly between [0.1 - 0.5] *tf
def random_dropout_fault_generator(sys_params: SystemParams,
                                   tf: float,
                                   num_of_events: int = 10,
                                   fault_model: str = "random",
                                   _fault_type: str = "both",
                                   at_least_n_survivors: int = 2,
                                   num_seeds: int = 2,
                                   affected_radius: float=3.0,
                                   trigger_time: float=0.5,
                                   rng_seed: int | None = 42) -> list[list[FaultEvent]]:
    print(
        f"[Fault Generator] "
        f"model={fault_model}, "
        f"events={num_of_events}, "
        f"fault_type={_fault_type}, "
        f"num_seeds={num_seeds}, "
        f"affected_radius={affected_radius:.2f} m"
    )

    rng = random.Random(rng_seed) if rng_seed is not None else random

    all_fault_model = ["random", "localized", "clustered"]  # specific to fault event
    all_fault_type = ["actuation", "comms", "both"]  # specific to each faulted agent

    # intialize with a no fault model first
    num_agents = len(sys_params.rs)
    agent_ids = list(range(num_agents))
    rs = np.asarray(sys_params.rs)
    all_fault_events = []

    # check selected fault_model
    if (fault_model not in all_fault_model):
        print(f"Unknown fault model: {fault_model}. It must be one of the three \"random, localized, clustered.\"")
        return all_fault_events

    # check number of agents
    if (len(sys_params.rs) <= at_least_n_survivors):
        return all_fault_events

    for i in range(num_of_events):
        events = []

        # choose how many agents fail. At least n agents remain functional
        max_faults = max(1, num_agents - at_least_n_survivors)
        n_faults = rng.randint(1, max_faults)

        # obtain the faulted_agenets based on the fault_model
        if (fault_model == "random"):
            # randomly choose which agents fail
            faulted_agents = rng.sample(agent_ids, n_faults)

        elif (fault_model == "localized"):
            # fault based on physical distance
            # Choose one random agent as the center/seed, then fault nearest agents.
            center_agent_id = rng.sample(agent_ids, 1)[0]
            center_pos = rs[center_agent_id]
            sorted_agents = sorted(agent_ids,
                                   key=lambda i: np.linalg.norm(rs[i] - center_pos))
            faulted_agents = sorted_agents[:n_faults]

        elif (fault_model == "clustered"):
            # similar to localized model, but instead of one random seed, multiple seeds are allowed.
            # Consequently, the nearby agents of the seeds will also be faulted
            # defined by num_seed , and affected_radius
            seed_ids = rng.sample(agent_ids, min(num_seeds, max_faults))

            faulted_set = set(seed_ids)
            for seed_id in seed_ids:
                seed_pos = rs[seed_id]

                for aid in agent_ids:
                    dist = np.linalg.norm(rs[aid] - seed_pos)

                    if dist <= affected_radius:
                        faulted_set.add(aid)
                        if len(faulted_set) >= max_faults:
                            break

                if len(list(faulted_set)) >= max_faults:
                    break

            faulted_agents = list(faulted_set)

        # generate the FaultEvent based on the collected faulted_agents
        for agent_id in faulted_agents:
            # choose when failure happens
            if  trigger_time != 0 :
                trigger_time = rng.uniform(0.1* tf, trigger_time * tf)

            # choose fault_type
            if (_fault_type in all_fault_type):
                chosen_fault_type = _fault_type
            else:
                chosen_fault_type = all_fault_type[rng.randint(0, 2)]
            events.append(FaultEvent(agent_id=agent_id,
                                     trigger_time=trigger_time,
                                     fault_type=chosen_fault_type))
        all_fault_events.append(events)
    return all_fault_events


def _extract_terminal_state(result, method: str):
    # changed: now also matches centralized_nlp_warm, which shares the same 12-col state layout
    if method in ("centralized_nlp", "centralized_nlp_warm"):
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
        include_warm_nlp: bool = False,  # changed: added flag to optionally include warm-start NLP
        include_decentralized_ga: bool = True,  # changed: added flag to optionally exclude decentralized island GA
):
    solver_calls = [
        lambda: solve_centralized_nlp(sys_params, bc, max_iters=3000, max_runtime_s=max_runtime_s),
    ]
    method_names = ["centralized_nlp"]

    if include_warm_nlp:  # changed: conditionally appends warm-start NLP solver call
        solver_calls.append(
            lambda: solve_centralized_nlp_warm(sys_params, bc, max_iters=3000, max_runtime_s=max_runtime_s)
        )
        method_names.append("centralized_nlp_warm")

    solver_calls.append(
        lambda: solve_centralized_ga(sys_params, bc, epsilon, pop_size=10, generations=5000,
                                     max_runtime_s=max_runtime_s)
    )
    method_names.append("centralized_ga")

    if include_decentralized_ga:  # changed: decentralized island GA now optional
        solver_calls.append(
            lambda: solve_decentralized_island_ga(
                sys_params,
                bc,
                epsilon,
                pop_size=5,
                migration_rounds=5000,
                max_runtime_s=max_runtime_s,
            )
        )
        method_names.append("decentralized_island_ga")

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