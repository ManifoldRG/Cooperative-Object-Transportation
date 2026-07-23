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
        "cost": cost,
        "runtime": runtime,
    }


#changed: added warm-start NLP variant using linear-interpolation initial guess
def solve_centralized_nlp_warm(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    max_iters: int = 100,
    max_runtime_s: float | None = None,
):
    """Warm-started centralized NLP.

    Initial guess: r/v/phi/omega each linearly interpolated between x0 and xf,
    Q (attachment vectors) tiled at rs — matches full_nlp_warm's internal flat
    layout (block per variable, not interleaved per-step).
    Baseline against cold-start (solve_centralized_nlp) to isolate the effect
    of initial guess quality.
    """
    start = time.perf_counter()
    x0 = bc.x0.as_array()
    xf = bc.xf.as_array()
    N = sys_params.N

    r0, v0, phi0, ome0 = x0[0:3], x0[3:6], x0[6:9], x0[9:12]
    rf, vf, phif, omef = xf[0:3], xf[3:6], xf[6:9], xf[9:12]

    #changed: build X_guess as block-concatenated flat vector (r-block, v-block,
    #phi-block, omega-block, Q-block) — matches full_nlp_warm's unpack() layout,
    #not a naive (N+1,12) row-interleaved flatten
    X_guess = np.concatenate([
        np.linspace(r0, rf, N + 1).flatten(),
        np.linspace(v0, vf, N + 1).flatten(),
        np.linspace(phi0, phif, N + 1).flatten(),
        np.linspace(ome0, omef, N + 1).flatten(),
        np.tile(np.array(sys_params.rs).flatten(), N + 1),
    ])
    U, X, Q = og_opts.full_nlp_warm(
        x0=x0,
        xf=xf,
        tf=bc.tf,
        mu=sys_params.mu,
        a=sys_params.a,
        e=sys_params.e,
        nu=sys_params.nu,
        I=sys_params.I,
        m=sys_params.m,
        rs=sys_params.rs,
        N=N,
        max_iters=max_iters,
        max_runtime_s=max_runtime_s,
        X_guess=X_guess,
    )
    runtime = time.perf_counter() - start
    cost = float("nan") if _stalled(U) else float(np.sum(np.square(U)))  #changed: flag stalled solve
    return {
        "method": "centralized_nlp_warm",
        "control": U,
        "state": X,
        "attachment": Q,
        "cost": cost,
        "runtime": runtime,
    }
