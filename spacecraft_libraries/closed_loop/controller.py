"""Per-agent closed-loop state machine.

Modes: TRACKING -> DETUMBLE -> IDENTIFY -> REPLAN -> (TRACKING | FAILED) ; DONE.

The controller owns only LOCAL control + sensing logic and its own belief about
the fault set. The orchestrator (recovery_sim) drives the IDENTIFY consensus and
REPLAN coordination over the delayed comms bus, then calls ``adopt_plan`` on each
surviving agent.
"""
from __future__ import annotations

import numpy as np

from ..data_structures import StateVector, Trajectory
from ..new_opts import quat_to_rotmat
from ..evaluation.metrics import quaternion_aware_violation
from .cone import project_into_cone
from .config import RecoveryConfig

TRACKING = "TRACKING"
DETUMBLE = "DETUMBLE"
IDENTIFY = "IDENTIFY"
REPLAN = "REPLAN"
DONE = "DONE"
FAILED = "FAILED"


class AgentController:
    def __init__(self, agent_id: int, r_body: np.ndarray, nu: float, cfg: RecoveryConfig):
        self.agent_id = agent_id
        self.r_body = np.asarray(r_body, dtype=float).reshape(3)
        self.nu = float(nu)
        self.cfg = cfg

        self.mode = TRACKING
        self.tau = None            # (N,3) consensus torque (diagnostic)
        self.U_self = None         # (N,3) this agent's planned body-frame thrust
        self.ref: Trajectory | None = None
        self.k = 0                 # index into the current plan

        self._dev_count = 0
        self._rest_count = 0
        self._detumble_steps = 0

    # ------------------------------------------------------------------ plans
    def adopt_plan(self, tau, U_self, reference: Trajectory) -> None:
        self.tau = tau
        self.U_self = None if U_self is None else np.asarray(U_self, dtype=float).reshape(-1, 3)
        self.ref = reference
        self.k = 0
        self.mode = TRACKING
        self._dev_count = 0
        self._rest_count = 0
        self._detumble_steps = 0

    # --------------------------------------------------------------- sensing
    @staticmethod
    def sense(true_payload_state: StateVector) -> StateVector:
        """v1: rigid attachment => the agent recovers the exact payload state from
        its own sensed state and the known attachment offset. Returns ground truth.
        Hook point for sensor noise/bias/delay in a future version."""
        return true_payload_state

    def deviation(self, sensed: StateVector, ref: StateVector) -> float:
        c = self.cfg
        d_r = np.linalg.norm(np.asarray(sensed.r) - np.asarray(ref.r))
        d_v = np.linalg.norm(np.asarray(sensed.v) - np.asarray(ref.v))
        d_q = quaternion_aware_violation(np.asarray(sensed.eps), np.asarray(ref.eps))
        d_w = np.linalg.norm(np.asarray(sensed.omega) - np.asarray(ref.omega))
        return c.w_r * d_r + c.w_v * d_v + c.w_q * d_q + c.w_w * d_w

    # --------------------------------------------------------------- command
    def command(self, sensed: StateVector) -> np.ndarray:
        """Body-frame thrust this agent applies this step, per current mode."""
        if self.mode == TRACKING:
            if self.U_self is not None and self.k < self.U_self.shape[0]:
                return self.U_self[self.k]
            return np.zeros(3)
        if self.mode == DETUMBLE:
            return self._detumble_command(sensed)
        # IDENTIFY / REPLAN / DONE / FAILED: thrusters off.
        return np.zeros(3)

    def _detumble_command(self, sensed: StateVector) -> np.ndarray:
        """Local attachment-point damping, projected into the plume cone.

        U_des = -k_v (R^T v) - k_w (omega x r_body), where R^T v converts the
        inertial COM velocity into the body frame so it can be opposed by a
        body-frame thrust; (omega x r_body) is the body-frame velocity of this
        agent's attachment point due to rotation.
        """
        c = self.cfg
        R = quat_to_rotmat(sensed.eps)
        v_body = R.T @ np.asarray(sensed.v, dtype=float).reshape(3)
        omega = np.asarray(sensed.omega, dtype=float).reshape(3)
        u_des = -c.k_v * v_body - c.k_w * np.cross(omega, self.r_body)
        return project_into_cone(u_des, self.r_body, self.nu, c.u_max)

    # ----------------------------------------------------------- transitions
    def check_tracking_deviation(self, sensed: StateVector) -> bool:
        """Test deviation against the current reference state (no index advance).
        Returns True if a detumble trigger fires (deviation over tol for
        dev_hold consecutive steps), flipping the agent to DETUMBLE."""
        ref_state = self._ref_state()
        dev = self.deviation(sensed, ref_state) if ref_state is not None else 0.0
        if dev > self.cfg.dev_tol:
            self._dev_count += 1
        else:
            self._dev_count = 0
        if self._dev_count >= self.cfg.dev_hold:
            self.enter_detumble()
            return True
        return False

    def enter_detumble(self) -> None:
        self.mode = DETUMBLE
        self._detumble_steps = 0
        self._rest_count = 0
        self._dev_count = 0

    def advance_tracking(self) -> None:
        """Advance the plan index after a tracking step; mark DONE at plan end."""
        self.k += 1
        if self.U_self is not None and self.k >= self.U_self.shape[0]:
            self.mode = DONE

    def update_detumble(self, sensed: StateVector) -> bool:
        """Test for rest. Returns True when payload velocity & angular rate have
        held under tolerance for rest_hold steps (or the safety cap is hit)."""
        self._detumble_steps += 1
        v_mag = np.linalg.norm(np.asarray(sensed.v, dtype=float))
        w_mag = np.linalg.norm(np.asarray(sensed.omega, dtype=float))
        if v_mag < self.cfg.rest_v_tol and w_mag < self.cfg.rest_w_tol:
            self._rest_count += 1
        else:
            self._rest_count = 0
        if self._rest_count >= self.cfg.rest_hold:
            self.mode = IDENTIFY
            return True
        if self._detumble_steps >= self.cfg.max_detumble_steps:
            # Give up detumbling; proceed to identify with whatever rest we have.
            self.mode = IDENTIFY
            return True
        return False

    def _ref_state(self) -> StateVector | None:
        if self.ref is None:
            return None
        idx = min(self.k, len(self.ref.states) - 1)
        return self.ref.states[idx]
