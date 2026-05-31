from __future__ import annotations

import numpy as np
from ..new_opts import so3_exp, so3_log  # lazy import — avoids circular dependency


def terminal_violation(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(actual - target))


def quaternion_aware_violation(actual_q: np.ndarray, target_q: np.ndarray) -> float:
    return float(min(np.linalg.norm(actual_q - target_q), np.linalg.norm(actual_q + target_q)))


def lie_attitude_violation(phi_actual: np.ndarray, phi_target: np.ndarray) -> float:
    # calculates geodesic distance on SO(3) between two attitudes stored as rotation for violation measurement rather than previous method
    #vectors phi = log(R).
    # d(R_a, R_t) = || log(R_a^T @ R_t) || Returns a value in [0, pi] radians.
    # note: not sure if this is a good metric or not. check with results and ask sidh

    R_actual = so3_exp(phi_actual)
    R_target = so3_exp(phi_target)
    return float(np.linalg.norm(so3_log(R_actual.T @ R_target)))
