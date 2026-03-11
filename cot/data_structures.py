from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SystemParams:
    mu: float
    a: float
    e: float
    nu: float
    I: np.ndarray
    m: float
    rs: list[np.ndarray]
    N: int


@dataclass
class StateVector:
    r: Optional[np.ndarray]
    v: Optional[np.ndarray]
    eps: Optional[np.ndarray]
    omega: Optional[np.ndarray]


@dataclass
class Trajectory:
    states: list[StateVector]
    times: np.ndarray


@dataclass
class ControlHistory:
    tau: np.ndarray
    U: np.ndarray
    force: np.ndarray
    dt: float


@dataclass
class BoundaryConditions:
    x0: StateVector
    xf: StateVector
    tf: float
