import numpy as np


def terminal_violation(traj, bc) -> float:
    s = traj.states[-1]
    eps_err = min(np.linalg.norm(s.eps - bc.xf.eps), np.linalg.norm(s.eps + bc.xf.eps))
    return float(np.linalg.norm(s.r - bc.xf.r) + np.linalg.norm(s.v - bc.xf.v) + eps_err + np.linalg.norm(s.omega - bc.xf.omega))


def control_cost(ctrl) -> float:
    return float(np.sum(np.square(ctrl.U)))
