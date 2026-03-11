from __future__ import annotations

import numpy as np

from spacecraft_libraries.data_structures import BoundaryConditions


def terminal_constraint_violation(trajectory: np.ndarray, bc: BoundaryConditions) -> float:
    final_state = trajectory[-1]
    target = bc.xf.as_array()
    pos_vel_violation = float(np.linalg.norm(final_state[:6] - target[:6]))
    q_final = final_state[6:10]
    q_target = target[6:10]
    q_violation = min(np.linalg.norm(q_final - q_target), np.linalg.norm(q_final + q_target))
    omega_violation = float(np.linalg.norm(final_state[10:13] - target[10:13]))
    return pos_vel_violation + float(q_violation) + omega_violation
