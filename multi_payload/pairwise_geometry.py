"""Endpoint-preserving native orbital state maps and pairwise segment checks."""
from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

from .translation import cost_value


class _TimeBudget(Exception):
    pass


def separation(refs, delta, eps=None):
    """Minimum on each linear segment, with conservative corridor-radius room.

    eps defaults to zero. Otherwise use the larger endpoint radius sum on each
    segment, so every interpolated radius is covered. This is a certificate for
    the discrete rollout's piecewise-linear paths, not continuous orbital ODEs.
    """
    positions = np.asarray([ref["r"] for ref in refs])
    count, nodes, _ = positions.shape
    i, j = np.triu_indices(count, 1)
    if len(i) == 0:
        return dict(ok=True, min_margin=float("inf"), times=np.array([]),
                    pairs=np.empty((0, 2), dtype=int), margins=np.array([]))
    radii = np.zeros((count, nodes)) if eps is None else np.asarray(eps)
    a = positions[i, :-1] - positions[j, :-1]
    b = positions[i, 1:] - positions[j, 1:] - a
    alpha = np.clip(-np.sum(a * b, axis=-1) /
                    np.maximum(np.sum(b * b, axis=-1), 1e-24), 0.0, 1.0)
    distance = np.linalg.norm(a + alpha[..., None] * b, axis=-1)
    radius = radii[i] + radii[j]
    requested = np.asarray(delta, dtype=float)
    if requested.ndim == 2:
        if requested.shape != (count, count):
            raise ValueError("pair clearance matrix must have shape (M,M)")
        requested = requested[i, j, None]
    elif requested.ndim == 1:
        if requested.shape != (len(i),):
            raise ValueError("pair clearance vector must follow upper-triangle pair order")
        requested = requested[:, None]
    need = requested + np.maximum(radius[:, :-1], radius[:, 1:])
    margin = distance - need
    segments = np.argmin(margin, axis=1)
    worst = margin[np.arange(len(i)), segments]
    times = segments + alpha[np.arange(len(i)), segments]
    return dict(ok=bool(np.min(worst) >= -1e-6), min_margin=float(np.min(worst)),
                times=times, pairs=np.column_stack([i, j]), margins=worst)


class _ReferenceProblem:
    """Shared Euler state maps and normalized net acceleration controls."""

    def __init__(self, sc, refs, psis, eps, objective):
        self.sc = sc
        self.length = max(100.0, float(sc.delta))
        self.objective = objective
        self.eps = np.asarray(eps)
        self.i, self.j = np.triu_indices(sc.M, 1)
        self.mass = np.array([p.m for p in sc.payloads])
        self.force_scale = self.mass * self.length / sc.tf ** 2
        self.maps = np.zeros((sc.N + 1, 6, 3 * sc.N))
        phi = np.zeros((sc.N + 1, 6, 6))
        phi[0] = np.eye(6)
        for k, psi in enumerate(psis):
            a = np.eye(6)
            a[:3, 3:] = sc.dt / sc.tf * np.eye(3)
            a[3:, :3] = sc.dt * sc.tf * psi[:, :3]
            a[3:, 3:] += sc.dt * psi[:, 3:]
            b = np.r_[np.zeros((3, 3)), sc.dt / sc.tf * np.eye(3)]
            self.maps[k + 1] = a @ self.maps[k]
            self.maps[k + 1, :, 3 * k:3 * k + 3] += b
            phi[k + 1] = a @ phi[k]
        x0 = np.array([np.r_[p.x0.r / self.length,
                              p.x0.v * sc.tf / self.length] for p in sc.payloads])
        xf = np.array([np.r_[p.xf.r / self.length,
                              p.xf.v * sc.tf / self.length] for p in sc.payloads])
        self.free = np.einsum("kab,jb->jka", phi, x0)
        self.base = np.array([r["F"] for r in refs]).reshape(sc.M, 3 * sc.N)
        self.base /= self.force_scale[:, None]
        e = self.maps[-1]
        self.base += (xf - self.free[:, -1] - self.base @ e.T) @ np.linalg.pinv(e).T
        self.z = null_space(e)
        self.scale = max(0.1, float(np.sqrt(np.mean(self.base ** 2))))
        self.size = self.z.shape[1]
        self.j0 = max(sum(r["cost"] for r in refs), 1e-12)

    def controls(self, y):
        return self.base + self.scale * np.asarray(y).reshape(self.sc.M, self.size) @ self.z.T

    def references(self, y):
        u = self.controls(y)
        state = self.free + np.einsum("kad,jd->jka", self.maps, u)
        forces = u.reshape(self.sc.M, self.sc.N, 3) * self.force_scale[:, None, None]
        return [dict(ok=True, r=state[j, :, :3] * self.length,
                     v=state[j, :, 3:] * self.length / self.sc.tf,
                     F=forces[j], cost=cost_value(forces[j], self.objective),
                     fuel=cost_value(forces[j], "fuel"), objective=self.objective,
                     waypoints={}) for j in range(self.sc.M)]
