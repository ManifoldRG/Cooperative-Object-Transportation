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
class StateVectorLie:
    r: Optional[np.ndarray]  #  position
    v: Optional[np.ndarray]  # velocity
    phi: Optional[np.ndarray]  # rotation vector = log(R)
    omega: Optional[np.ndarray]  #  angular velocity

    def as_array(self):
        # Layout: [r(3), v(3), phi(3), omega(3)] = 12 elements
        return np.hstack([self.r, self.v, self.phi, self.omega])

    @classmethod
    def from_array(cls, x):
        return cls(r=x[0:3], v=x[3:6], phi=x[6:9], omega=x[9:12])

@dataclass
class Trajectory:
    states: list[StateVectorLie]
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
    x0: StateVectorLie
    xf:StateVectorLie
    tf:float


