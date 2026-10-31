"""Multi-payload scenarios on a shared chief orbit and time grid.

All payloads share (mu, a, e), the cone half-angle nu, tf and N, so they see
the same Tschauner-Hempel matrices th_psi_matrix(mu, a, e, k*dt) and their
trajectories can be compared step by step for collisions. Per-payload
properties (mass, candidate attachment points, inertia, attitude goal) are
drawn exactly as in spacecraft_libraries.evaluation.comparison.
random_scenario_generator; only the start/goal positions differ, because that
generator starts every payload at the origin.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spacecraft_libraries.data_structures import (
    BoundaryConditions, StateVectorLie, SystemParams)
from spacecraft_libraries.evaluation.comparison import sample_inertia_tensor
from spacecraft_libraries.new_opts import so3_log

MU = 3.98e14
R_EARTH = 6.371e6


@dataclass
class Payload:
    m: float
    I: np.ndarray            # (3,3) body-frame inertia
    points: np.ndarray       # (P,3) open attachment points, body frame
    x0: StateVectorLie
    xf: StateVectorLie

    @property
    def extent(self) -> float:
        return float(np.max(np.linalg.norm(self.points, axis=1)))


@dataclass
class MultiPayloadScenario:
    mu: float
    a: float
    e: float
    nu: float                # thrust-cone half-angle [rad]
    tf: float
    N: int
    epsilon: float
    delta: float             # minimum center-to-center separation [m]
    n_agents: int
    payloads: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def M(self) -> int:
        return len(self.payloads)

    @property
    def dt(self) -> float:
        return self.tf / self.N

    def sys_params_for(self, j: int, point_idx=()):
        """SystemParams / BoundaryConditions of payload j with agents at the
        given candidate points (body frame), as used by all COT code."""
        P = self.payloads[j]
        rs = [np.asarray(P.points[i], float) for i in sorted(point_idx)]
        sp = SystemParams(mu=self.mu, a=self.a, e=self.e, nu=self.nu,
                          I=np.asarray(P.I, float), m=float(P.m), rs=rs, N=self.N)
        bc = BoundaryConditions(x0=P.x0, xf=P.xf, tf=self.tf)
        return sp, bc


def _random_phi(np_rng):
    """Random attitude as a rotation vector (Gaussian unit quaternion), as in
    random_scenario_generator."""
    q = np_rng.standard_normal(4)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ])
    return so3_log(R)


def _random_payload_body(rng, np_rng, n_points):
    m = rng.uniform(1, 500)
    pts = []
    for _ in range(n_points):
        mag = rng.uniform(5, 50)
        d = np_rng.standard_normal(3)
        pts.append(d / np.linalg.norm(d) * mag)
    pts = np.asarray(pts)
    L = float(np.max(np.linalg.norm(pts, axis=1)))
    I = sample_inertia_tensor(m, L, rng=rng)
    return m, I, pts


def _state(r, phi=None):
    return StateVectorLie(r=np.asarray(r, float), v=np.zeros(3),
                          phi=np.zeros(3) if phi is None else np.asarray(phi, float),
                          omega=np.zeros(3))


def generate(layout: str = "crossing", M: int = 3, n_points: int = 6,
             n_agents: int | None = None, seed: int = 0,
             nu_deg: float = 60.0, ring_radius: float = 500.0,
             tf: float | None = None, N: int | None = None,
             delta_margin: float = 10.0, max_tries: int = 1000) -> MultiPayloadScenario:
    """Draw a multi-payload scenario.

    layout="crossing": starts evenly spaced on a ring (random plane through the
      origin), goals diametrically across with small jitter, so straight paths
      cross near the center and the repair step is exercised.
    layout="random": starts and goals uniform in a ball of radius ring_radius,
      rejection-sampled so starts (and goals) are pairwise >= delta + margin.
    tf, N default to the paper's generator: tf ~ U(300, 900) s, N = max(20, tf/10).
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    a = rng.uniform(1.1, 1.3) * R_EARTH
    e = rng.uniform(0.01, 0.3)
    tf = rng.uniform(300, 900) if tf is None else float(tf)
    N = max(20, int(tf / 10)) if N is None else int(N)
    epsilon = rng.uniform(1e-6, 1e-4)

    bodies = [_random_payload_body(rng, np_rng, n_points) for _ in range(M)]
    extent = max(float(np.max(np.linalg.norm(b[2], axis=1))) for b in bodies)
    delta = 2 * extent + delta_margin
    sep = delta + delta_margin

    if layout == "crossing":
        # random orthonormal plane
        u = np_rng.standard_normal(3)
        u /= np.linalg.norm(u)
        w = np.cross(u, np_rng.standard_normal(3))
        w /= np.linalg.norm(w)
        phase = rng.uniform(0, 2 * np.pi)
        starts, goals = [], []
        for j in range(M):
            th = phase + 2 * np.pi * j / M
            jit = rng.uniform(-0.25, 0.25) * np.pi / M
            starts.append(ring_radius * (np.cos(th) * u + np.sin(th) * w))
            goals.append(-ring_radius * (np.cos(th + jit) * u + np.sin(th + jit) * w))
        if M > 1 and 2 * ring_radius * np.sin(np.pi / M) < sep:
            raise ValueError("ring too small for the payload extents: increase ring_radius")
    elif layout == "random":
        def draw_set():
            for _ in range(max_tries):
                pts = []
                while len(pts) < M:
                    p = np_rng.uniform(-1, 1, 3) * ring_radius
                    if np.linalg.norm(p) <= ring_radius:
                        pts.append(p)
                if all(np.linalg.norm(pts[i] - pts[k]) >= sep
                       for i in range(M) for k in range(i + 1, M)):
                    return pts
            raise RuntimeError("could not place payloads: enlarge ring_radius")
        starts, goals = draw_set(), draw_set()
    elif layout == "random_joint":
        # dart-throwing over starts AND goals jointly: all 2M endpoints
        # pairwise >= sep, so start-goal cross clearances hold too (needed by
        # spatially disjoint corridor methods, where endpoints pin the tubes).
        pts = []
        for _ in range(2 * M):
            for _ in range(max_tries):
                p = np_rng.uniform(-1, 1, 3) * ring_radius
                if np.linalg.norm(p) > ring_radius:
                    continue
                if all(np.linalg.norm(p - q) >= sep for q in pts):
                    pts.append(p)
                    break
            else:
                raise RuntimeError("could not place payloads: enlarge ring_radius")
        starts, goals = pts[:M], pts[M:]
    elif layout == "random_seq":
        # sequential (dart-throwing) placement: same uniform-in-ball target
        # distribution thinned by the separation, but each point is accepted
        # or redrawn on its own, so dense sets that defeat the all-pairs
        # rejection draw above still place reliably.
        def draw_set():
            pts = []
            for _ in range(M):
                for _ in range(max_tries):
                    p = np_rng.uniform(-1, 1, 3) * ring_radius
                    if np.linalg.norm(p) > ring_radius:
                        continue
                    if all(np.linalg.norm(p - q) >= sep for q in pts):
                        pts.append(p)
                        break
                else:
                    raise RuntimeError(
                        "could not place payloads: enlarge ring_radius")
            return pts
        starts, goals = draw_set(), draw_set()
    else:
        raise ValueError(f"unknown layout {layout!r}")

    payloads = [Payload(m=m, I=I, points=pts, x0=_state(starts[j]),
                        xf=_state(goals[j], _random_phi(np_rng)))
                for j, (m, I, pts) in enumerate(bodies)]
    return MultiPayloadScenario(
        mu=MU, a=a, e=e, nu=np.deg2rad(nu_deg), tf=tf, N=N, epsilon=epsilon,
        delta=delta, n_agents=n_agents if n_agents is not None else 3 * M,
        payloads=payloads,
        meta=dict(layout=layout, seed=seed, nu_deg=nu_deg, n_points=n_points,
                  ring_radius=ring_radius))


def to_dict(sc: MultiPayloadScenario) -> dict:
    sv = lambda s: dict(r=s.r.tolist(), v=s.v.tolist(), phi=s.phi.tolist(),
                        omega=s.omega.tolist())
    return dict(mu=sc.mu, a=sc.a, e=sc.e, nu=sc.nu, tf=sc.tf, N=sc.N,
                epsilon=sc.epsilon, delta=sc.delta, n_agents=sc.n_agents,
                meta=sc.meta,
                payloads=[dict(m=P.m, I_diag=np.diag(P.I).tolist(),
                               points=P.points.tolist(), x0=sv(P.x0), xf=sv(P.xf))
                          for P in sc.payloads])


def from_dict(d: dict) -> MultiPayloadScenario:
    st = lambda s: StateVectorLie(**{k: np.asarray(v, float) for k, v in s.items()})
    payloads = [Payload(m=p["m"], I=np.diag(p["I_diag"]), points=np.asarray(p["points"]),
                        x0=st(p["x0"]), xf=st(p["xf"])) for p in d["payloads"]]
    return MultiPayloadScenario(
        mu=d["mu"], a=d["a"], e=d["e"], nu=d["nu"], tf=d["tf"], N=d["N"],
        epsilon=d["epsilon"], delta=d["delta"], n_agents=d["n_agents"],
        payloads=payloads, meta=d.get("meta", {}))


def save(sc: MultiPayloadScenario, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_dict(sc), indent=1))


def load(path) -> MultiPayloadScenario:
    return from_dict(json.loads(Path(path).read_text()))
