"""Tunable configuration for the closed-loop recovery simulation.

Defaults are first-cut and will need a calibration pass (same workflow as the
MPPI sigma sweep). Gains/tolerances scale with the scenario, so expect to tune.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecoveryConfig:
    # --- PD detumble gains (local attachment-point damping) ---
    # NOTE: gains are scenario-scaled (mass / inertia / dt). These defaults are
    # calibrated for the comparison.py-class scenarios (m~100, dt~2); a different
    # payload needs a re-tune. Stability heuristic: k_v ~ m/dt for fast velocity
    # damping. Angular damping is inherently slower (high Izz, cone-limited torque).
    k_v: float = 0.05
    k_w: float = 0.1 #changed from 0.02 to 0.1
    u_max: float = 1 #0.5 to 1
    dt_detumble_fac: float = 0.0001 # during detumbling, dt smaller. so dt_detumble = dt * dt_detumble_fac

    # --- deviation detection ---
    dev_tol: float = 0.1    # weighted deviation threshold (see controller.deviation)
    dev_hold: int = 1       # consecutive steps over tol before triggering detumble
    w_r: float = 1.0        # deviation weights
    w_v: float = 1.0
    w_q: float = 1.0
    w_w: float = 1.0

    # --- detumble exit (rest) ---
    rest_v_tol: float = 1e-2
    rest_w_tol: float = 1e-2
    rest_hold: int = 2
    max_detumble_steps: int = 4000 # safety cap so a bad-gain run can't hang

    # --- comms / fault identification ---
    agents_comms_delay_step_map: dict[int, int] = field(default_factory=dict)
    id_timeout: int = 3     # steps to wait for a HEALTHY message before marking silent
    graph_degree: int = 3

    # --- recovery loop ---
    max_recovery_cycles: int = 3
    success_tol: float = 1.5   # terminal violation under which the episode is DONE

    # --- planner selection ---
    # "dgd" (default): decentralized gradient descent with random island
    # starts — deterministic replans, machine-zero terminal violations,
    # basin exploration at tight cones. "dmppi": legacy decentralized MPPI.
    planner: str = "dgd"

    # --- dGD replanning knobs (forwarded to solve_decentralized_gd) ---
    # Per-island wall budget. Replans must be timely, so the default caps
    # each island at 15 s (serial islands: plan wall ~= n_agents * budget);
    # islands with leftover time random-restart and keep the best. None =
    # descend to stationarity — can take minutes per island on aggressive
    # maneuvers (measured 42 s/341 iters on the tiny closed-loop test
    # scenario), so use None only when plan quality outranks latency.
    gd_budget_s: float | None = 15.0
    gd_rel_step: float = 0.03
    gd_tau_init_scale: float = 0.1
    gd_random_restart_scale: float = 1.0
    gd_base_seed: int = 42

    # --- MPPI replanning knobs (forwarded to solve_decentralized_mppi;
    # used only when planner="dmppi") ---
    mppi_iterations: int = 10
    mppi_samples: int = 5
    mppi_sigma: float = 7e-1
    mppi_lambda: float = 1.0
    mppi_base_seed: int = 42
