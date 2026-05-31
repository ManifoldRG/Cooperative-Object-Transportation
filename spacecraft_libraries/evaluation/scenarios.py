from __future__ import annotations

import numpy as np

from spacecraft_libraries.data_structures import BoundaryConditions, StateVector, SystemParams


def default_scenario() -> tuple[SystemParams, BoundaryConditions]:
    rs = [np.array([0.5, 1.0, 1.5]), np.array([0.0, 0.5, 2.0]), np.array([-0.5, 1.0, -1.5])]
    sys_params = SystemParams(
        mu=3.98e14,
        a=8e6,
        e=0.2,
        nu=np.pi / 4,
        I=1000 * np.diag([1.0, 2.0, 3.0]),
        m=100.0,
        rs=rs,
        N=40,
    )
    x0 = StateVector(r=np.zeros(3), v=np.zeros(3), eps=np.array([0.0, 0.0, 0.0, 1.0]), omega=np.zeros(3))
    xf = StateVector(
        r=np.array([5.0, 5.0, 5.0]),
        v=np.zeros(3),
        eps=np.array([0.5, 0.5, 0.5, 0.5]),
        omega=np.zeros(3),
    )
    bc = BoundaryConditions(x0=x0, xf=xf, tf=50.0)
    return sys_params, bc
