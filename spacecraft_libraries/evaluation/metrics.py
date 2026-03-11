from __future__ import annotations

import numpy as np


def terminal_violation(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(actual - target))


def quaternion_aware_violation(actual_q: np.ndarray, target_q: np.ndarray) -> float:
    return float(min(np.linalg.norm(actual_q - target_q), np.linalg.norm(actual_q + target_q)))
