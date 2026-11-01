"""Leader inflation PBD plus ball-constrained translational SOCP references.

The inflation kernel follows studies/benchmark_methods.py:relax_iso. Native
pipeline adaptation uses sc.delta, enforces every capped ball, and checks
complete linear segments after reconstruction before accepting a reference.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .pairwise_geometry import separation
from .translation import solve_reference


def corridor_radii(sc, refs, cap=150.0):
    """Disjoint balls with room certified along the linear reference segments.

    A node's room uses the minimum pair distance over both adjacent segments.
    Half of that surplus is assigned to each payload. Consequently the linearly
    interpolated balls are disjoint on every segment, up to solver tolerances.
    """
    positions = np.asarray([ref["r"] for ref in refs])
    eps = np.full((sc.M, sc.N + 1), float(cap))
    for j in range(sc.M):
        for l in range(j + 1, sc.M):
            a = positions[j, :-1] - positions[l, :-1]
            b = positions[j, 1:] - positions[l, 1:] - a
            s = np.clip(-np.sum(a * b, axis=1) /
                        np.maximum(np.sum(b * b, axis=1), 1e-24), 0.0, 1.0)
            room = np.linalg.norm(a + s[:, None] * b, axis=1) - sc.delta
            if np.min(room) < -1e-6:
                raise ValueError("reference paths still collide")
            node_room = np.r_[room[0], np.minimum(room[:-1], room[1:]), room[-1]]
            radius = 0.5 * np.maximum(node_room, 0.0)
            eps[j] = np.minimum(eps[j], radius)
            eps[l] = np.minimum(eps[l], radius)
    eps[:, 0] = eps[:, -1] = 0.0
    return [e.copy() for e in eps]


def relax_iso(rbar, pairs, tgt, N, M, max_sweeps=5000, alpha=0.25, sigma=1.5,
              tol=0.1, Winv2=None, profile="gauss", deadline=None):
    """Vectorized inflation PBD; isotropic pushes, or fuel-metric W^-2 g
    pushes when Winv2 (per-payload axis weights) is given.

    profile selects how a push spreads along the trajectory:
      "gauss" (default): per-step exact-gap projections + Gaussian smoothing
        of the displacement field (the original two-phase scheme);
      "tent": per pair and sweep, ONE whole-trajectory push peaked at the
        worst violated step, linearly decaying to zero at the pinned ends
        (the elastic-string / point-load Green's function of min int(u')^2);
      "beam": the same with a smoothstep profile, zero slope at the ends
        and the peak (beam-like, closer to the min int(u'')^2 fuel metric);
      "bump<W>" (e.g. "bump6"): compactly supported hybrid — a smoothstep
        bump of half-width W steps around the worst violated step, smooth
        (zero value AND slope) at the window borders, so the push keeps
        tent's no-self-fighting speed while deviation stays local.

    Step k=1 is frozen (no pushes, no smoothing displacement, excluded from
    the residual): v(0)=0 pins r_1 = r_0 for every payload, so step-1
    separation equals the start separation, which placement guarantees. An
    inflated step-1 target would be both unattainable downstream (the tube
    ball moves away from where the payload must be) and unnecessary."""
    x = [r.copy() for r in rbar]

    if profile != "gauss":
        ks = np.arange(N + 1, dtype=float)
        B = np.zeros((N + 1, N + 1))
        W = float(profile[4:] or 6) if profile.startswith("bump") else None
        for k in range(2, N):
            if W is not None:
                u = np.clip(1.0 - np.abs(ks - k) / W, 0.0, None)
                w = u * u * (3.0 - 2.0 * u)
            else:
                w = np.minimum((ks - 1) / (k - 1), (N - ks) / (N - k))
                w = np.clip(w, 0.0, None)
                if profile == "beam":
                    w = w * w * (3.0 - 2.0 * w)
            w[0] = w[1] = w[N] = 0.0
            B[k] = w

        def sweep_field():
            worst = 0.0
            for (j, l) in pairs:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("inflation budget exhausted")
                d = x[j][2:N] - x[l][2:N]
                dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                need = tgt[(j, l)] - dist
                kw = int(np.argmax(need))
                worst_need = float(need[kw])
                if worst_need <= 0:
                    continue
                worst = max(worst, worst_need)
                g = d[kw] / dist[kw]
                F = np.outer(B[kw + 2], 0.5 * worst_need * g)
                x[j] += F
                x[l] -= F
            return worst

        def sweep_point():
            worst = 0.0
            for (j, l) in pairs:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("inflation budget exhausted")
                d = x[j][2:N] - x[l][2:N]
                dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                need = tgt[(j, l)] - dist
                m = need > 0
                if m.any():
                    worst = max(worst, float(need[m].max()))
                    g = d[m] / dist[m][:, None]
                    x[j][2:N][m] += 0.5 * need[m][:, None] * g
                    x[l][2:N][m] -= 0.5 * need[m][:, None] * g
            return worst

        sweeps, worst = 0, np.inf
        while sweeps < max_sweeps and worst >= tol:
            worst = sweep_field()
            sweeps += 1
        if profile == "anneal" and worst < tol:
            # retraction: shrink the displacement field toward the nominal,
            # re-project real violations, and keep it SMOOTH while doing so
            # (shrink + point projections alone collapse to jagged spikes)
            for _ in range(120):
                for j in range(M):
                    x[j] = rbar[j] + 0.93 * (x[j] - rbar[j])
                sweep_point()
                for j in range(M):
                    u = x[j] - rbar[j]
                    us = gaussian_filter1d(u, sigma=sigma, axis=0)
                    u = (1 - alpha) * u + alpha * us
                    u[0] = 0.0
                    u[1] = 0.0
                    u[N] = 0.0
                    x[j] = rbar[j] + u
                sweeps += 1
            worst = np.inf
            while sweeps < max_sweeps and worst >= tol:
                worst = sweep_point()
                sweeps += 1
        return (x if worst < tol else None), sweeps, float(worst)

    def sweep(smooth):
        worst = 0.0
        for (j, l) in pairs:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("inflation budget exhausted")
            d = x[j][2:N] - x[l][2:N]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            need = tgt[(j, l)] - dist
            m = need > 0
            if m.any():
                worst = max(worst, float(need[m].max()))
                g = d[m] / dist[m][:, None]
                if Winv2 is None:
                    x[j][2:N][m] += 0.5 * need[m][:, None] * g
                    x[l][2:N][m] -= 0.5 * need[m][:, None] * g
                else:
                    pj = g * Winv2[j]
                    pl = g * Winv2[l]
                    x[j][2:N][m] += (0.5 * need[m] / np.sum(g * pj, axis=1)
                                     )[:, None] * pj
                    x[l][2:N][m] -= (0.5 * need[m] / np.sum(g * pl, axis=1)
                                     )[:, None] * pl
        if smooth:
            for j in range(M):
                u = x[j] - rbar[j]
                us = gaussian_filter1d(u, sigma=sigma, axis=0)
                u = (1 - alpha) * u + alpha * us
                u[0] = 0.0
                u[1] = 0.0
                u[N] = 0.0
                x[j] = rbar[j] + u
        return worst

    sweeps, worst = 0, np.inf
    while sweeps < 800 and worst >= 1.0:
        worst = sweep(True)
        sweeps += 1
    while sweeps < max_sweeps and worst >= tol:
        worst = sweep(False)
        sweeps += 1
    return (x if worst < tol else None), sweeps, float(worst)


def solve_reference_set(sc, refs, psis, objective="fuel", budget_s=60.0,
                        profile="gauss", headroom=60.0, corridor_cap=150.0,
                        max_sweeps=5000, attempts=2):
    """Inflate nominal centers, solve inside balls, and verify reconstruction.

    A failed check returns no accepted reference; this is not an infeasibility
    proof. The time budget is checked between solves and inflation pair sweeps;
    one running conic solve can overrun it.
    """
    if profile not in ("gauss", "tent", "beam"):
        raise ValueError("inflation profile must be gauss, tent, or beam")
    if headroom < 0 or corridor_cap < 0 or attempts < 1:
        raise ValueError("invalid inflation headroom, cap, or attempts")
    start = time.perf_counter()
    deadline = start + budget_s
    pairs = list(itertools.combinations(range(sc.M), 2))
    eps = [np.zeros(sc.N + 1) for _ in refs]
    info = dict(method="inflation_ball", joint_optimization=False,
                uses_ball_corridors=True, ok=False, status="no_safe_reference", profile=profile)
    for attempt in range(attempts):
        try:
            targets = {p: sc.delta + headroom * 2 ** attempt for p in pairs}
            centers, sweeps, residual = relax_iso(
                [r["r"] for r in refs], pairs, targets, sc.N, sc.M,
                max_sweeps=max_sweeps, profile=profile, deadline=deadline)
        except TimeoutError:
            info["status"] = "time_budget"
            break
        info.update(attempt=attempt, sweeps=sweeps, residual=residual)
        if centers is None:
            continue
        try:
            # Segment-based radii also protect the time between grid nodes.
            eps = corridor_radii(sc, [dict(r=x) for x in centers], corridor_cap)
        except ValueError:
            info["status"] = "inflated_centers_have_segment_conflict"
            continue
        result = []
        for j in range(sc.M):
            eps[j][0] = eps[j][-1] = 0.0
            if time.perf_counter() >= deadline:
                info["status"] = "time_budget"
                break
            ref = solve_reference(sc, j, psis=psis, objective=objective,
                                  corridor=dict(rbar=centers[j], eps=eps[j]))
            if not ref["ok"]:
                info["status"] = "ball_reference_solve_failed"
                break
            ref["corridor_rbar"] = centers[j]
            result.append(ref)
        if len(result) == sc.M:
            check = separation(result, sc.delta)
            info.update(min_margin=check["min_margin"])
            if check["ok"]:
                info.update(ok=True, status="safe_reference",
                            seconds=time.perf_counter() - start)
                return result, eps, info
    info["seconds"] = time.perf_counter() - start
    return refs, eps, info
