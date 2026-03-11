import numpy as np
from .data_structures import BoundaryConditions, StateVector, SystemParams


def scenario_two(num_steps: int = 20) -> tuple[SystemParams, BoundaryConditions, float]:
    rs = [np.array([0.5, 1, 1.5]), np.array([0, 1.5, 2]), np.array([-0.5, 1, -1.5])]
    sys_params = SystemParams(
        mu=3.986e14,
        a=8_000e3,
        e=0.2,
        nu=np.pi / 4,
        I=1000 * np.diag([1, 2, 3]),
        m=2000,
        rs=rs,
        N=num_steps,
    )
    bc = BoundaryConditions(
        x0=StateVector(r=np.array([1, 10, -5]), v=np.array([0, 0.5, 0]), eps=np.array([0.707, 0, 0.707, 0]), omega=np.array([0, 0.1, 0])),
        xf=StateVector(r=np.array([0, 0, 0]), v=np.array([0, 0, 0]), eps=np.array([0, 0, 0, 1]), omega=np.array([0, 0, 0])),
        tf=60,
    )
    epsilon = 1e-12
    return sys_params, bc, epsilon
