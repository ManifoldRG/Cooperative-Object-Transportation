"""Forward payload simulator that mirrors the planner's dynamics exactly.

CRITICAL: this uses the SAME discrete dynamics as the inner solver
``new_opts.opt_given_tau_ipopt_new`` (new_opts.py:600-652): Tschauner-Hempel
translational + SO(3) Rodrigues rotational + BODY-frame wrench. It deliberately
does NOT use ``dynamics.forward_pass_dynamics`` (which uses a Clohessy-Wiltshire
translational model + inertial-frame torque) -- if it did, a perfectly healthy
rollout would already diverge from the plan and the deviation threshold would be
meaningless.
"""
from __future__ import annotations

import numpy as np

from ..data_structures import StateVectorLie, SystemParams
from ..new_opts import so3_exp, so3_log, state_attitude_to_phi
from ..orbital_helpers import th_psi_matrix


def _skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


class PayloadSimulator:
    """Steps the rigid payload under per-agent body-frame thrusts.

    State carried internally: position r (inertial), velocity v (inertial),
    rotation matrix R (body->inertial), angular velocity omega (body frame).
    """

    def __init__(self, sys_params: SystemParams, x0: StateVectorLie, dt: float):
        self.sys_params = sys_params
        self.dt = float(dt)
        self.I = np.asarray(sys_params.I, dtype=float)
        self.I_inv = np.linalg.inv(self.I)
        self.m = float(sys_params.m)
        self.rs_body = [np.asarray(r, dtype=float).reshape(3) for r in sys_params.rs]

        self.r = np.asarray(x0.r, dtype=float).reshape(3).copy()
        self.v = np.asarray(x0.v, dtype=float).reshape(3).copy()
        self.R = so3_exp(state_attitude_to_phi(x0))
        self.omega = np.asarray(x0.omega, dtype=float).reshape(3).copy()
        self.t = 0.0

    def state_vector(self) -> StateVectorLie:
        return StateVectorLie(
            r=self.r.copy(),
            v=self.v.copy(),
            phi=so3_log(self.R),
            omega=self.omega.copy(),
        )

    def step(self, agent_thrusts, active_mask, t: float | None = None) -> StateVectorLie:
        """Advance one dt under the given per-agent body-frame thrusts.

        agent_thrusts: list/array of (3,) body-frame thrusts, one per agent
            (length == number of agents this simulator was built with).
        active_mask: iterable of bools; thrust from an inactive (actuation-dead)
            agent contributes zero wrench regardless of the commanded value.
        """
        if t is None:
            t = self.t

        thrust_body = np.zeros(3)
        tau_body = np.zeros(3)
        for i, (u_i, active) in enumerate(zip(agent_thrusts, active_mask)):
            if not active:
                continue
            u_i = np.asarray(u_i, dtype=float).reshape(3)
            thrust_body += u_i
            tau_body += _skew(self.rs_body[i]) @ u_i

        # Rotational update (body frame, Euler equation + Rodrigues kinematics).
        omega_dot = self.I_inv @ (tau_body - np.cross(self.omega, self.I @ self.omega))
        omega_next = self.omega + self.dt * omega_dot
        R_next = self.R @ so3_exp(self.dt * self.omega)

        # Translational update (Tschauner-Hempel, body->inertial thrust via R).
        Psi = th_psi_matrix(self.sys_params.mu, self.sys_params.a, self.sys_params.e, t)
        Psi_vel = np.asarray(Psi)[3:6, :]
        # Thrust uses the pre-update rotation R_k, matching new_opts.py:640-645
        # (R_k = ca.DM(Rs[k]); v_next uses R_k @ thrust_body).
        rv = np.hstack([self.r, self.v])
        r_next = self.r + self.dt * self.v
        v_next = self.v + self.dt * (Psi_vel @ rv + (1.0 / self.m) * (self.R @ thrust_body))

        self.r, self.v, self.R, self.omega = r_next, v_next, R_next, omega_next
        self.t = t + self.dt
        return self.state_vector()
