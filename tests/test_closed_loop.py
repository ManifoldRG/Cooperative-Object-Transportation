"""Smoke tests for the closed-loop fault-recovery layer. Tiny scenarios so the
suite stays fast despite the inner IPOPT solves during (re)planning.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from spacecraft_libraries.data_structures import BoundaryConditions, StateVectorLie, SystemParams
from spacecraft_libraries.closed_loop import RecoveryConfig, run_recovery_episode
from spacecraft_libraries.closed_loop.cone import project_into_cone
from spacecraft_libraries.closed_loop.controller import AgentController, DETUMBLE
from spacecraft_libraries.closed_loop.simulator import PayloadSimulator
from spacecraft_libraries.closed_loop.faults import FaultEvent


def _scenario(n_agents=3, N=10, tf=20.0):
    rs_all = [
        np.array([0.5, 1.0, 1.5]),
        np.array([0.0, 0.5, 2.0]),
        np.array([-0.5, 1.0, -1.5]),
        np.array([1.0, -0.5, 1.0]),
        np.array([-1.0, 0.75, 1.5]),
    ]
    rs = rs_all[:n_agents]
    sys_params = SystemParams(
        mu=3.98e14, a=8e6, e=0.2, nu=np.pi / 4,
        I=1000 * np.diag([1, 2, 3]), m=100, rs=rs, N=N,
    )
    bc = BoundaryConditions(
        x0=StateVectorLie(r=np.zeros(3), v=np.zeros(3), phi=np.zeros(3), omega=np.zeros(3)),
        xf=StateVectorLie(r=np.array([2., 2., 2.]), v=np.zeros(3), phi=np.array([1.20919958, 1.20919958, 1.20919958]), omega=np.zeros(3)),
        tf=tf,
    )
    return sys_params, bc, 1e-4


def _fast_cfg(**kw):
    base = dict(mppi_iterations=2, mppi_samples=3, rest_v_tol=1e-2, rest_w_tol=1e-2,
                rest_hold=2, max_detumble_steps=8, dev_tol=0.3, dev_hold=2,
                comms_delay_steps=2, id_timeout=3)
    base.update(kw)
    return RecoveryConfig(**base)


def test_cone_projection_respects_cone_and_cap():
    rng = np.random.default_rng(0)
    nu = np.pi / 4
    axis = np.array([1.0, 0.5, -0.3])
    u_max = 4.0
    for _ in range(200):
        d = rng.normal(0, 3, size=3)
        out = project_into_cone(d, axis, nu, u_max)
        n = np.linalg.norm(out)
        assert n <= u_max + 1e-9
        if n > 1e-9:
            cos_ang = np.dot(out, axis) / (n * np.linalg.norm(axis))
            assert cos_ang >= np.cos(nu) - 1e-6


def test_healthy_rollout_does_not_trigger_recovery():
    sys_params, bc, eps = _scenario() #fixme: also
    result = run_recovery_episode(sys_params, bc, eps, fault_events=None, cfg=_fast_cfg())
    assert result["recovery_cycles"] == 0
    assert result["status"] in ("done", "done_offtarget")
    # No agent should have been removed and no detumble should have fired.
    assert result["removed_agents"] == []
    assert not any("DETUMBLE" in msg for _, _, msg in result["log"]["transitions"])


def test_detumble_reduces_translational_and_angular_motion():
    # Use a fine dt so the explicit PD damping is stable; 5 agents give the
    # ensemble enough cone-legal authority to damp a general (v, omega).
    sys_params, bc, _ = _scenario(n_agents=5)
    cfg = _fast_cfg(k_v=20.0, k_w=20.0, u_max=10.0)
    dt = 0.5
    x0 = StateVectorLie(r=np.zeros(3), v=np.array([0.1, -0.05, 0.08]),
                        phi=np.zeros(3), omega=np.array([0.02, -0.03, 0.05]))
    sim = PayloadSimulator(sys_params, x0, dt)
    controllers = [AgentController(i, sys_params.rs[i], sys_params.nu, cfg)
                   for i in range(len(sys_params.rs))]
    for c in controllers:
        c.mode = DETUMBLE

    v0 = np.linalg.norm(sim.v)
    w0 = np.linalg.norm(sim.omega)
    for _ in range(120):
        sensed = sim.state_vector()
        thrusts = [c.command(sensed) for c in controllers]
        mask = [True] * len(controllers)
        sim.step(thrusts, mask)
    # Translational motion damps strongly; angular damping about the high-inertia
    # axis is slower (Izz=3000) and cone-limited, so assert a clear reduction.
    assert np.linalg.norm(sim.v) < 0.5 * v0
    assert np.linalg.norm(sim.omega) < 0.7 * w0


def test_single_actuation_fault_triggers_recovery():
    # Generous horizon so the detumble + identify rounds don't exhaust the clock.
    sys_params, bc, eps = _scenario(n_agents=3, N=20, tf=40.0)  #fixme: change
    cfg = _fast_cfg()
    faults = [FaultEvent(agent_id=2, trigger_time=6.0, fault_type="actuation")]
    result = run_recovery_episode(sys_params, bc, eps, fault_events=faults, cfg=cfg)
    assert result["recovery_cycles"] >= 1
    removed = {a for _, agents in result["removed_agents"] for a in agents}
    assert 2 in removed
    assert 2 not in result["final_active_agents"]


def test_comms_dead_agent_marked_unresponsive():
    sys_params, bc, eps = _scenario(n_agents=3, N=20, tf=40.0)
    cfg = _fast_cfg(comms_delay_steps=2)
    # Agent 1 loses BOTH (dead + silent): must be detected as unresponsive.
    faults = [FaultEvent(agent_id=1, trigger_time=6.0, fault_type="both")]
    result = run_recovery_episode(sys_params, bc, eps, fault_events=faults, cfg=cfg)
    removed = {a for _, agents in result["removed_agents"] for a in agents}
    assert 1 in removed