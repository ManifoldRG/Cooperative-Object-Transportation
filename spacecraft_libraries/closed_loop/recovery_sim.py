"""Closed-loop recovery orchestrator.

Runs a full episode: plan with decentralized MPPI, track the consensus
trajectory through the payload simulator, and on a fault-induced deviation drive
the swarm through detumble -> fault-identification -> replan -> resume, repeating
(capped) until the target is reached or recovery is exhausted.

The orchestrator shuttles messages over the delayed CommsBus and coordinates the
identify/replan bookkeeping; per-agent control + sensing live in AgentController.
"""
from __future__ import annotations

import dataclasses
import time

import networkx as nx
import numpy as np

from ..data_structures import BoundaryConditions, StateVectorLie, SystemParams
from ..evaluation.metrics import quaternion_aware_violation, terminal_violation, lie_attitude_violation
from ..solvers.decentralized_mppi import _build_line_of_sight_graph_with_degree, solve_decentralized_mppi
from .comms import CommsBus
from .config import RecoveryConfig
from .controller import AgentController, TRACKING, DETUMBLE, IDENTIFY, DONE, FAILED
from .faults import FaultModel
from .simulator import PayloadSimulator


def _distribute_plan(plan: dict, controllers: list[AgentController], active_ids: list[int]) -> None:
    U = np.array(plan["control"].U)  # (len(active_ids), N', 3)
    tau = plan["tau"]
    ref = plan["trajectory"]
    for pos, aid in enumerate(active_ids):
        controllers[aid].adopt_plan(tau, U[pos], ref)


def _terminal_violation(state: StateVectorLie, xf: StateVectorLie) -> float:
    return (
        terminal_violation(state.r, xf.r)
        + terminal_violation(state.v, xf.v)
       # + quaternion_aware_violation(state.eps, xf.eps)
        + lie_attitude_violation(state.phi, xf.phi)
        + terminal_violation(state.omega, xf.omega)
    )


def run_recovery_episode(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    fault_events: list | None = None,
    cfg: RecoveryConfig | None = None,
    verbose: bool = False,
):
    if cfg is None :
        raise ValueError("cfg cannot be None")
        return

    cfg = cfg or RecoveryConfig()
    t_start = time.perf_counter()
    dt = bc.tf / sys_params.N
    num_agents = len(sys_params.rs)

    fault_model = FaultModel(num_agents, fault_events)
    graph = _build_line_of_sight_graph_with_degree(np.asarray(sys_params.rs), 100.0, cfg.graph_degree)
    bus = CommsBus(graph, cfg.agents_comms_delay_step_map)

    controllers = [AgentController(i, sys_params.rs[i], sys_params.nu, cfg) for i in range(num_agents)]
    active_ids = list(range(num_agents))

    if verbose:
        print(f"[recovery] initial plan: {num_agents} agents, N={sys_params.N}, dt={dt:.3f}")
    plan = solve_decentralized_mppi(
        sys_params, bc, epsilon,
        n_iter=cfg.mppi_iterations, n_samples=cfg.mppi_samples,
        sigma=cfg.mppi_sigma, lambda_=cfg.mppi_lambda, base_seed=cfg.mppi_base_seed,
        graph_degree=cfg.graph_degree
    )
    _distribute_plan(plan, controllers, active_ids)

    sim = PayloadSimulator(sys_params, bc.x0, dt)

    log = {"transitions": [], "recovery_cycles": 0, "faults_injected": [], "removed_agents": []}
    fuel = 0.0
    step = 0
    t = 0.0
    cycles = 0
    status = "running"
    identify_deadline = None  # step by which the identify round completes
    max_comms_delay_steps = max(cfg.agents_comms_delay_step_map.values(), default=0)
    max_steps = (sys_params.N + cfg.max_detumble_steps + max_comms_delay_steps + cfg.id_timeout) * (cfg.max_recovery_cycles + 2)

    def _log(msg):
        log["transitions"].append((step, round(t, 3), msg))
        if verbose:
            print(f"  step {step:4d} t={t:7.2f}  {msg}")
    
    state_history=[]

    while step < max_steps:
        for aid in fault_model.update(t):
            log["faults_injected"].append((step, aid))
            _log(f"FAULT injected on agent {aid}")

        sensed = sim.state_vector()
        inbox = bus.deliver(step)
        state_history.append(np.asarray(sensed.as_array(), dtype=float).reshape(-1))
        active_controllers = [controllers[a] for a in active_ids]

        # --- DEVIATION message propagation: receiving it pulls a tracker into detumble.
        for aid in active_ids:
            c = controllers[aid]
            if c.mode == TRACKING and any(m.kind == "DEVIATION" for m in inbox.get(aid, [])):
                c.enter_detumble()
                _log(f"agent {aid} -> DETUMBLE (received DEVIATION)")

        # --- TRACKING: deviation checks (self-detection).
        for aid in active_ids:
            c = controllers[aid]
            if c.mode == TRACKING and c.check_tracking_deviation(sensed):
                bus.broadcast(aid, "DEVIATION", None, step, fault_model.comms_alive(aid))
                _log(f"agent {aid} -> DETUMBLE (deviation detected)")

        # --- DETUMBLE: rest checks; on reaching rest, enter IDENTIFY and announce status.
        for aid in active_ids:
            c = controllers[aid]
            if c.mode == DETUMBLE and c.update_detumble(sensed):
                status_payload = "FAULTED" if not fault_model.actuation_alive(aid) else "HEALTHY"
                bus.broadcast(aid, "STATUS", (aid, status_payload), step,
                              fault_model.comms_alive(aid))
                _log(f"agent {aid} -> IDENTIFY (at rest, announced {status_payload})")
                if identify_deadline is None:
                    identify_deadline = step + max_comms_delay_steps + cfg.id_timeout

        # --- IDENTIFY: collect STATUS into each agent's belief.
        for aid in active_ids:
            for m in inbox.get(aid, []):
                if m.kind == "STATUS":
                    src, st = m.payload
                    controllers[aid]._heard = getattr(controllers[aid], "_heard", {})
                    controllers[aid]._heard[src] = st

        # --- IDENTIFY complete: reconcile fault set, replan over the reduced swarm.
        if (identify_deadline is not None and step >= identify_deadline
                and all(c.mode == IDENTIFY for c in active_controllers)):
            faulted = _reconcile_faults(controllers, active_ids, graph)
            survivors = [a for a in active_ids if a not in faulted]
            log["removed_agents"].append((step, sorted(faulted)))
            _log(f"IDENTIFY consensus: faulted={sorted(faulted)}, survivors={survivors}")

            cycles += 1
            log["recovery_cycles"] = cycles
            remaining = bc.tf - t
            if cycles > cfg.max_recovery_cycles or len(survivors) == 0 or remaining < 2 * dt:
                for a in active_ids:
                    controllers[a].mode = FAILED
                status = "failed"
                _log(f"RECOVERY EXHAUSTED (cycles={cycles}, survivors={len(survivors)}, "
                     f"remaining={remaining:.2f})")
                break

            for a in faulted:
                controllers[a].mode = FAILED

            replan_state = sim.state_vector()
            N_prime = max(2, int(round(remaining / dt)))
            rs_reduced = [sys_params.rs[j] for j in survivors]
            sys_reduced = dataclasses.replace(sys_params, rs=rs_reduced, N=N_prime)
            bc_replan = BoundaryConditions(x0=replan_state, xf=bc.xf, tf=remaining)
            _log(f"REPLAN: survivors={survivors}, N'={N_prime}, remaining={remaining:.2f}s")
            plan = solve_decentralized_mppi(
                sys_reduced, bc_replan, epsilon,
                n_iter=cfg.mppi_iterations, n_samples=cfg.mppi_samples,
                sigma=cfg.mppi_sigma, lambda_=cfg.mppi_lambda, base_seed=cfg.mppi_base_seed,
                graph_degree=cfg.graph_degree
            )
            active_ids = survivors
            _distribute_plan(plan, controllers, active_ids)

            # Rebuild comms over the surviving subgraph and reset the identify cycle.
            graph = _build_line_of_sight_graph_with_degree(np.asarray(rs_reduced), 100.0, cfg.graph_degree)
            graph = _relabel_graph(graph, survivors)
            bus = CommsBus(graph, cfg.agents_comms_delay_step_map)
            identify_deadline = None
            for a in active_ids:
                controllers[a]._heard = {}
            continue  # restart loop on the fresh plan without stepping this iteration

        # --- commands + dynamics step ---
        thrusts = [np.zeros(3) for _ in range(num_agents)]
        for aid in active_ids:
            thrusts[aid] = controllers[aid].command(sensed)
        active_mask = fault_model.actuation_mask()
        for aid in active_ids:
            if active_mask[aid]:
                fuel += float(np.dot(thrusts[aid], thrusts[aid]))
        sim.step(thrusts, active_mask, t)

        for aid in active_ids:
            if controllers[aid].mode == TRACKING:
                controllers[aid].advance_tracking()

        step += 1
        t += dt

        if active_ids and all(controllers[a].mode == DONE for a in active_ids):
            status = "done"
            break

    final_state = sim.state_vector()
    state_history.append(np.asarray(final_state.as_array(), dtype=float).reshape(-1))
    viol = _terminal_violation(final_state, bc.xf)
    if status == "running":
        status = "timeout"
    if status == "done" and viol > cfg.success_tol:
        status = "done_offtarget"

    return {
        "status": status,
        "terminal_violation": float(viol),
        "fuel": float(fuel),
        "steps": step,
        "sim_time": float(t),
        "recovery_cycles": cycles,
        "final_active_agents": list(active_ids),
        "removed_agents": log["removed_agents"],
        "final_state": final_state,
        "state_history": state_history,
        "log": log,
        "runtime_s": time.perf_counter() - t_start,
    }


def _reconcile_faults(controllers: list[AgentController], active_ids: list[int], graph: nx.Graph) -> set[int]:
    """Union, across surviving agents, of agents believed faulted: an active
    agent is removed if any survivor either heard a FAULTED self-report from it
    or never heard from it within the timeout (unresponsive). Single-hop over the
    line-of-sight graph (sufficient for the near-complete LoS graphs in these
    scenarios; multi-hop gossip is a documented extension)."""
    faulted: set[int] = set()
    for aid in active_ids:
        heard = getattr(controllers[aid], "_heard", {})
        expected_neighbours = ( set(graph.neighbors(aid)) & set(active_ids) )

        # check if agent(aid) receirves messages from its neighbours
        for other in expected_neighbours:
            if other == aid:
                continue
            st = heard.get(other)
            if st is None or st == "FAULTED":
                faulted.add(other)
    return faulted


def _relabel_graph(graph, survivors):
    """The LoS graph built from reduced rs uses positional indices 0..len-1;
    relabel its nodes back to the original survivor agent ids."""
    import networkx as nx
    mapping = {pos: survivors[pos] for pos in range(len(survivors))}
    return nx.relabel_nodes(graph, mapping)
