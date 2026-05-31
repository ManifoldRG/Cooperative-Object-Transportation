from __future__ import annotations

import time

import numpy as np

from .. import og_opts
from ..data_structures import BoundaryConditions, SystemParams


def solve_centralized_nlp(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    max_iters: int = 100,
    max_runtime_s: float | None = None,
):
    start = time.perf_counter()
    U, X, Q = og_opts.full_nlp(
        x0=bc.x0.as_array(),
        xf=bc.xf.as_array(),
        tf=bc.tf,
        mu=sys_params.mu,
        a=sys_params.a,
        e=sys_params.e,
        nu=sys_params.nu,
        I=sys_params.I,
        m=sys_params.m,
        rs=sys_params.rs,
        N=sys_params.N,
        max_iters=max_iters,
        max_runtime_s=max_runtime_s,
    )
    runtime = time.perf_counter() - start
    return {
        "method": "centralized_nlp",
        "control": U,
        "state": X,
        "attachment": Q,
        "cost": float(np.sum(np.square(U))),
        "runtime": runtime,
    }
