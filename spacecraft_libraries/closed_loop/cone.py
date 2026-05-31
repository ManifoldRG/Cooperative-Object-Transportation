"""Plume-impingement cone projection for the PD detumble mode.

Mirrors the planner's pointing constraint (new_opts.py:624-630):
    U_i . r_i >= cos(nu) * ||U_i|| * ||r_i||
i.e. each agent's body-frame thrust must lie within a cone of half-angle ``nu``
around its body-frame attachment vector. The detumble controller computes a
desired damping thrust that may point anywhere; this projects it onto the cone
(plume-friendly) and caps its magnitude.
"""
from __future__ import annotations

import numpy as np


def project_into_cone(d: np.ndarray, axis: np.ndarray, half_angle: float,
                      u_max: float | None = None) -> np.ndarray:
    """Project desired thrust ``d`` onto the cone of half-angle ``half_angle``
    around ``axis``, then cap magnitude at ``u_max``.

    Returns the zero vector if ``d`` lies in the polar cone (more than
    ``pi/2 + half_angle`` from the axis), where no legal thrust has a positive
    projection.
    """
    d = np.asarray(d, dtype=float).reshape(3)
    axis = np.asarray(axis, dtype=float).reshape(3)

    d_norm = np.linalg.norm(d)
    a_norm = np.linalg.norm(axis)
    if d_norm < 1e-12 or a_norm < 1e-12:
        return np.zeros(3)

    ahat = axis / a_norm
    p = float(np.dot(d, ahat))            # component along axis
    d_perp = d - p * ahat
    q = float(np.linalg.norm(d_perp))     # perpendicular magnitude

    cos_nu = np.cos(half_angle)
    sin_nu = np.sin(half_angle)

    # Already inside the cone: keep as-is.
    if p >= cos_nu * d_norm:
        out = d
    else:
        # Outside the cone. Closest point on the cone surface has scale
        # s = p*cos(nu) + q*sin(nu); if s <= 0 the desired vector is in the
        # polar cone and the best legal thrust is zero.
        s = p * cos_nu + q * sin_nu
        if s <= 0.0:
            return np.zeros(3)
        if q < 1e-12:
            # d is (anti)parallel to axis but outside cone => p<0 handled above.
            perp_dir = np.zeros(3)
        else:
            perp_dir = d_perp / q
        direction = cos_nu * ahat + sin_nu * perp_dir
        out = s * direction

    if u_max is not None:
        out_norm = np.linalg.norm(out)
        if out_norm > u_max:
            out = out * (u_max / out_norm)
    return out
