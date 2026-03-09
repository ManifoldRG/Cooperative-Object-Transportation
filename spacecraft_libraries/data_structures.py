from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SystemParams:
    mu: float #gravitational parameter
    a: float #semi-major axis
    e: float #eccentricity
    nu: float #initial anomaly angle
    I: np.ndarray #inertia matrix
    m: float #mass in kg
    rs: np.ndarray
    N: int #dicretisation steps


@dataclass
class StateVector:
    r: Optional[np.ndarray] #positionvector
    v: Optional[np.ndarray] #velocityvector
    eps: Optional[np.ndarray] #quaternion
    omega: Optional[np.ndarray] #angularvelocity


    def as_array(self):
        return np.hstack([self.r, self.v, self.eps, self.omega])

    @classmethod
    def from_array(cls, x):
        r = x[0:3]
        v = x[3:6]
        eps = x[6:10]
        omega = x[10:13]
        return cls(r, v, eps, omega)

@dataclass
class Trajectory:
    states: list[StateVector]
    times: np.ndarray

    def as_array(self) -> np.ndarray:
        return np.vstack([s.as_array() for s in self.states])

@dataclass
class ControlHistory:
    tau : np.ndarray #for inputs from GA
    U: Optional[list[np.ndarray]]
    force: Optional[np.ndarray]
    dt: Optional[float]

@dataclass
class BoundaryConditions:
    x0: StateVector
    xf:StateVector
    tf:float


