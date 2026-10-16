"""Tests for the multi-payload approach (multi_payload/)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_payload import allocation, attitude, conflicts, pipeline, scenario, translation  # noqa: E402
from multi_payload.socp_corridor import envelope_grad_socp, solve_inner_socp  # noqa: E402
from spacecraft_libraries.solvers import socp_inner  # noqa: E402


def _small(M=2, points=4, agents=None, seed=1, N=10):
    return scenario.generate("crossing", M=M, n_points=points, n_agents=agents,
                             seed=seed, nu_deg=60, N=N, tf=300)


def test_reference_meets_boundary_conditions():
    sc = _small()
    refs = translation.solve_all(sc)
    for P, ref in zip(sc.payloads, refs):
        assert ref["ok"]
        assert np.allclose(ref["r"][0], P.x0.r, atol=1e-6)
        assert np.allclose(ref["r"][-1], P.xf.r, atol=1e-6)
        assert np.allclose(ref["v"][-1], 0.0, atol=1e-6)
        assert ref["fuel"] > 0


def test_repair_resolves_crossing_conflicts():
    sc = _small(M=3, seed=2)
    refs = translation.solve_all(sc)
    eps = conflicts.zero_eps(sc)
    assert conflicts.detect(sc, refs, eps), "crossing layout should start with conflicts"
    refs, wps, log, ok = conflicts.repair(sc, refs, [dict() for _ in range(sc.M)], eps)
    assert ok and not conflicts.detect(sc, refs, eps)
    assert len(log) >= 1


def test_no_repair_when_far_apart():
    sc = _small()
    off = np.array([0.0, 0.0, 5000.0])
    sc.payloads[1].x0.r = sc.payloads[1].x0.r + off
    sc.payloads[1].xf.r = sc.payloads[1].xf.r + off
    refs = translation.solve_all(sc)
    assert not conflicts.detect(sc, refs, conflicts.zero_eps(sc))


def test_misocp_no_worse_than_greedy_and_residual_consistent():
    sc = _small(points=4, agents=5, N=8)
    refs = translation.solve_all(sc)
    att = [attitude.attitude_reference(sc, j) for j in range(sc.M)]
    W = [attitude.required_wrenches(refs[j], att[j]) for j in range(sc.M)]
    al = allocation.build_allocators(sc, W)
    g = allocation.greedy(al, sc.n_agents)
    m = allocation.misocp(al, sc.n_agents, time_limit=120)
    for out in (g, m):
        assert sum(len(o) for o in out["occupied"]) == sc.n_agents
        for gam, e in zip(out["gamma"], out["residual"]):
            assert np.isfinite(gam) == (e == 0.0)
    if g["case"] == "fuel":
        assert m["case"] == "fuel"
        assert m["J_hat"] <= g["J_hat"] * (1 + 1e-4) + 1e-6
    else:
        assert max(m["residual"]) <= max(g["residual"]) * (1 + 1e-4) + 1e-6


def _single(seed=4, N=8):
    sc = _small(M=1, points=4, seed=seed, N=N)
    sp, bc = sc.sys_params_for(0, range(4))
    return sc, sp, bc


def test_corridor_socp_reduces_to_socp_inner():
    sc, sp, bc = _single()
    tau = np.random.default_rng(0).normal(0, 0.05, (sp.N, 3))
    base = socp_inner.solve_inner_socp(sp, bc, tau)
    rbar = np.linspace(bc.x0.r, bc.xf.r, sp.N + 1)
    wide = solve_inner_socp(sp, bc, tau, corridor=dict(rbar=rbar, eps=np.full(sp.N + 1, 1e6), rho=None))
    assert base["ok"] and wide["ok"]
    assert abs(base["J"] - wide["J"]) <= 1e-5 * max(1.0, abs(base["J"]))


def test_corridor_envelope_gradient_matches_finite_differences():
    sc, sp, bc = _single()
    rng = np.random.default_rng(1)
    tau = rng.normal(0, 0.05, (sp.N, 3))
    free = solve_inner_socp(sp, bc, tau)
    # an active elastic corridor around a straight line
    cor = dict(rbar=np.linspace(bc.x0.r, bc.xf.r, sp.N + 1), eps=np.full(sp.N + 1, 1.0), rho=10.0)
    sol = solve_inner_socp(sp, bc, tau, corridor=cor)
    assert sol["ok"] and free["ok"]
    assert sol["J"] > free["J"] + 1e-6, "corridor should be active"
    g = envelope_grad_socp(sp, bc, tau, sol)
    d = rng.normal(0, 1, tau.shape)
    d /= np.linalg.norm(d)
    h = 1e-4 * max(1.0, float(np.abs(tau).max()))
    Jp = solve_inner_socp(sp, bc, tau + h * d, corridor=cor)["J"]
    Jm = solve_inner_socp(sp, bc, tau - h * d, corridor=cor)["J"]
    fd = (Jp - Jm) / (2 * h)
    an = float(np.sum(g * d))
    assert abs(fd - an) <= 2e-2 * max(abs(fd), abs(an), 1e-3), (fd, an)


def test_pipeline_end_to_end_small():
    sc = _small(M=2, points=4, agents=6, seed=5)
    out = pipeline.run(sc, budget_s=10, max_restarts=1, verbose=False)
    v = out["verify"]
    assert out["status"] == "ok"
    assert v["collision_free"] and v["all_terminal_ok"] and v["certificate_ok"]


def test_docking_captures_all_agents():
    from multi_payload import docking
    sc = _small(M=2, points=4, agents=4, seed=6)
    occupied = [[0, 1], [0, 1]]
    starts = docking.agent_starts(sc, seed=0)
    dp = docking.DockingProblem(sc, starts, T_dock=300.0, N_dock=15)
    # payload drift is integrated backward so docking ends exactly at x0
    for j, P in enumerate(sc.payloads):
        assert np.allclose(dp.drift[j][0][-1], P.x0.r) and np.allclose(dp.drift[j][1][-1], P.x0.v)
    asg, _ = docking.assign(dp, occupied)
    assert sorted(asg) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    sim = docking.simulate(dp, asg, noise=0.02, seed=1)
    assert sim["all_captured"]
