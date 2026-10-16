"""Allocation-independent attitude references and required wrenches.

The attitude reference of payload j is the paper's warm start: the
multiple-shooting minimum-torque-energy maneuver (make_nominal_tau), projected
onto the attitude-feasible set with the same projector the COT solver uses
(tau_proj_nonlin_new). Both depend only on the inertia and the attitude
boundary conditions, not on the allocation.

Together with the translational reference, it fixes the body-frame wrench the
agents of payload j must produce at every step:
    w_k = (R_k^T F_k, tau_k),
with R_k from the same numeric attitude rollout the inner SOCP uses.
"""
from __future__ import annotations

import numpy as np

from spacecraft_libraries.new_opts import so3_log, state_attitude_to_phi, tau_proj_nonlin_new
from spacecraft_libraries.solvers.mppi_core import make_nominal_tau
from spacecraft_libraries.solvers.socp_inner import attitude_rollout


def attitude_reference(sc, j, seed=0, tau_init_scale=0.1):
    """Returns dict(tau (N,3), Rs, omes, terminal_error)."""
    sp, bc = sc.sys_params_for(j)
    tau = make_nominal_tau(sp, bc, sc.epsilon, np.random.default_rng(seed),
                           tau_init_scale=tau_init_scale)
    proj, _ = tau_proj_nonlin_new(tau, sc.N, sc.epsilon, sp, bc)
    if proj is not None:
        tau = np.asarray(proj, float).reshape(sc.N, 3)
    Rs, omes = attitude_rollout(sp, bc, tau)
    Rf = _rot(state_attitude_to_phi(bc.xf))
    err = float(np.linalg.norm(so3_log(Rs[-1].T @ Rf))
                + np.linalg.norm(omes[-1] - bc.xf.omega))
    return dict(tau=tau, Rs=Rs, omes=omes, terminal_error=err)


def _rot(phi):
    from spacecraft_libraries.new_opts import so3_exp
    return so3_exp(np.asarray(phi, float))


def required_wrenches(ref, att):
    """(N,6) body-frame wrenches [R_k^T F_k, tau_k]."""
    F = ref["F"]
    return np.array([np.concatenate([att["Rs"][k].T @ F[k], att["tau"][k]])
                     for k in range(len(F))])
