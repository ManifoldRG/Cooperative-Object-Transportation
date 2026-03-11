from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from spacecraft_libraries.data_structures import BoundaryConditions, SystemParams
from spacecraft_libraries.og_opts import full_nlp


@dataclass
class SolverRun:
    method: str
    trajectory: np.ndarray
    control: list[np.ndarray]
    q_history: np.ndarray
    cost: float
    runtime_s: float


def solve_centralized_nlp(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    max_iters: int = 5000,
) -> SolverRun:
    t0 = time.perf_counter()
    X_opt, U_opt, Q_opt, cost = full_nlp(
        bc.x0.as_array(),
        bc.xf.as_array(),
        bc.tf,
        sys_params.mu,
        sys_params.a,
        sys_params.e,
        sys_params.nu,
        sys_params.I,
        sys_params.m,
        sys_params.rs,
        sys_params.N,
        max_iters=max_iters,
    )
    runtime_s = time.perf_counter() - t0
    return SolverRun(
        method="centralized_nlp",
        trajectory=np.asarray(X_opt),
        control=U_opt,
        q_history=np.asarray(Q_opt),
        cost=float(cost),
        runtime_s=runtime_s,
    )
