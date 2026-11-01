"""Direct pairwise AL-L-BFGS translational references, with native Euler dynamics.

All payload controls are optimized jointly. There is one inequality and AL
multiplier per payload pair and enforced time. Complete linear-segment closest
approaches generate new times. Balls are constructed only after this solve.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .pairwise_geometry import _ReferenceProblem, _TimeBudget, separation
from .translation import psi_list, solve_all


class _PairwiseProblem(_ReferenceProblem):
    """Reuse state/endpoint maps; no collective kernel is evaluated here."""

    def pairwise(self, y, times, multipliers, mu):
        u = self.controls(y)
        k = np.minimum(np.floor(times).astype(int), self.sc.N - 1)
        a = times - k
        maps = (1 - a[:, None, None]) * self.maps[k] + a[:, None, None] * self.maps[k + 1]
        free = (1 - a[None, :, None]) * self.free[:, k] + a[None, :, None] * self.free[:, k + 1]
        positions = (free[:, :, :3] + np.einsum("tad,jd->jta", maps[:, :3], u)) * self.length
        relative = positions[self.i] - positions[self.j]
        distance = np.linalg.norm(relative, axis=-1)
        # Match separation()'s conservative requested-radius room on each segment.
        radii = self.eps[self.i] + self.eps[self.j]
        pair_deltas = getattr(self, "pair_deltas", None)
        requested = self.sc.delta if pair_deltas is None else pair_deltas[self.i, self.j, None]
        required = requested + getattr(self, "clearance", 0.0) + np.maximum(radii[:, k], radii[:, k + 1])
        g = 1 - distance / required
        weight = np.maximum(0, multipliers + mu * g)
        vector = -weight[..., None] * relative / np.maximum(distance[..., None], 1e-12) / required[..., None]
        gp = np.zeros_like(positions)
        np.add.at(gp, self.i, vector)
        np.add.at(gp, self.j, -vector)
        mapping = maps[:, :3] @ self.z * self.length * self.scale
        penalty_gradient = np.einsum("jta,taz->jz", gp, mapping)
        forces = u.reshape(self.sc.M, self.sc.N, 3) * self.force_scale[:, None, None]
        if self.objective == "fuel":
            norm = np.sqrt(np.sum(forces * forces, axis=-1) + 1e-24)
            cost = float(np.sum(norm))
            gf = forces / norm[..., None]
        else:
            cost = float(np.sum(forces * forces))
            gf = 2 * forces
        gu = (gf * self.force_scale[:, None, None]).reshape(self.sc.M, 3 * self.sc.N)
        gradient = self.scale * gu @ self.z / self.j0
        return cost / self.j0, gradient, g, penalty_gradient


def solve_reference_set(sc, refs=None, eps=None, psis=None, objective="fuel",
                        budget_s=60.0, stages=12, inner_iterations=90,
                        tolerance=1e-7, seed=0, clearance=0.0, pair_deltas=None):
    """Return (reference dictionaries, info); info['ok'] requires safe references.

    pair_deltas optionally supplies the shared physical (M,M) clearance matrix.
    clearance adds metres to the optimization target only. Candidate acceptance
    and min_margin use the original physical sc.delta; planning residuals are
    reported separately. Fixed nodes must also accommodate the requested buffer.

    The collision-free check is analytic for the linear segments of the paper's
    discrete rollout. Final agent controls must still pass verify.verify. Fixed
    Euler nodes 0/1/N are checked geometrically and are not optimization variables.
    """
    start = time.perf_counter()
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("pairwise clearance must be a finite nonnegative distance in metres")
    physical_delta = sc.delta if pair_deltas is None else np.asarray(pair_deltas, dtype=float)
    if pair_deltas is not None:
        if (physical_delta.shape != (sc.M, sc.M) or not np.isfinite(physical_delta).all()
                or not np.allclose(physical_delta, physical_delta.T)):
            raise ValueError("pair_deltas must be a finite symmetric (M,M) matrix")
        if np.any(physical_delta[np.triu_indices(sc.M, 1)] <= 0):
            raise ValueError("off-diagonal pair clearances must be positive")
    planning_delta = physical_delta + clearance
    metadata = dict(method="full_pairwise_lbfgs",
                    constraint="(required + planning buffer - distance)/(required + planning buffer) <= 0",
                    planning_clearance_m=float(clearance),
                    physical_required_m=float(sc.delta) if pair_deltas is None else None,
                    planning_required_m=float(planning_delta) if pair_deltas is None else None,
                    clearance_mode="common" if pair_deltas is None else "pair_specific")
    if objective not in ("fuel", "energy"):
        raise ValueError("unknown objective")
    if budget_s <= 0 or stages < 1 or inner_iterations < 1:
        raise ValueError("pairwise budgets and iteration limits must be positive")
    if tolerance < 0:
        raise ValueError("pairwise tolerance must be nonnegative")
    psis = psi_list(sc) if psis is None else psis
    refs = solve_all(sc, psis=psis, objective=objective) if refs is None else refs
    if not all(r["ok"] for r in refs):
        return refs, dict(**metadata, ok=False, status="nominal_failed", runtime=time.perf_counter() - start)
    eps = [np.zeros(sc.N + 1) for _ in range(sc.M)] if eps is None else eps
    problem = _PairwiseProblem(sc, refs, psis, eps, objective)
    problem.clearance = float(clearance)
    problem.pair_deltas = None if pair_deltas is None else physical_delta
    y = np.zeros(sc.M * problem.size)
    original = problem.references(y)
    check = separation(original, physical_delta, eps)
    planning_check = separation(original, planning_delta, eps) if clearance else check
    if check["ok"] and planning_check["ok"]:
        return original, dict(**metadata, ok=True, status="nominal_safe", runtime=time.perf_counter() - start,
                              iterations=0, min_margin=check["min_margin"], history=[])
    first = separation([dict(r=r["r"][:2]) for r in original], physical_delta,
                       [np.asarray(e)[:2] for e in eps])
    endpoints = np.array([r["r"][-1] for r in original])
    final_distance = np.linalg.norm(endpoints[problem.i] - endpoints[problem.j], axis=1)
    final_radii = np.asarray(eps)[:, -1]
    final_required = sc.delta if pair_deltas is None else physical_delta[problem.i, problem.j]
    final_need = final_required + final_radii[problem.i] + final_radii[problem.j]
    fixed_failure = not first["ok"] or np.any(final_distance < final_need - 1e-6)
    if fixed_failure or problem.size == 0 or sc.N < 3:
        return original, dict(**metadata, ok=False, status="immutable_reference_conflict",
                              runtime=time.perf_counter() - start, history=[])
    first_planning = separation([dict(r=r["r"][:2]) for r in original], planning_delta,
                                [np.asarray(e)[:2] for e in eps]) if clearance else first
    if not first_planning["ok"] or np.any(final_distance < final_need + clearance - 1e-6):
        return original, dict(**metadata, ok=False, status="immutable_planning_clearance_conflict",
                              runtime=time.perf_counter() - start, history=[],
                              min_margin=check["min_margin"])
    # A tiny seeded perturbation breaks exact symmetric crossing degeneracy.
    y = np.random.default_rng(seed).normal(0.0, 1e-3, y.size)
    times = np.arange(2, sc.N, dtype=float)
    deadline = start + budget_s
    lam = np.zeros((len(problem.i), len(times)))
    mu = 10.0
    history, last, best = [], {}, None
    status = "iteration_limit"

    def evaluate(y):
        if time.perf_counter() >= deadline:
            raise _TimeBudget()
        cost, grad, g, penalty_gradient = problem.pairwise(y, times, lam, mu)
        weight = np.maximum(0.0, lam + mu * g)
        loss = cost + np.sum(weight * weight - lam * lam) / (2.0 * mu)
        gradient = grad + penalty_gradient
        last.update(y=y.copy(), cost=cost, g=g)
        return float(loss), gradient.ravel()

    def record(y):
        if not np.array_equal(last.get("y"), y):
            evaluate(y)
        history.append(dict(iteration=len(history), seconds=time.perf_counter() - start,
                            cost=last["cost"] * problem.j0,
                            max_pairwise_violation=float(np.max(last["g"]))))

    def inspect(y):
        nonlocal best
        candidate = problem.references(y)
        physical = separation(candidate, physical_delta, eps)
        check = separation(candidate, planning_delta, eps) if clearance else physical
        cost = sum(r["cost"] for r in candidate)
        if physical["ok"] and (best is None or cost < best[0]):
            best = (cost, candidate, check)
        return candidate, check

    try:
        for _ in range(stages):
            res = minimize(evaluate, y, jac=True, method="L-BFGS-B", callback=record,
                           options=dict(maxiter=inner_iterations, maxls=30, maxcor=12,
                                        ftol=1e-11, gtol=1e-5))
            y = res.x
            evaluate(y)
            candidate, check = inspect(y)
            if check["ok"] and last["g"].max() <= tolerance and res.success:
                status = "local_success_safe"
                break
            extra = check["times"][check["margins"] < -1e-6]
            # Immutable-node conflicts cannot be changed by translational controls.
            extra = extra[(extra > 1.0 + 1e-8) & (extra < sc.N - 1e-8)]
            merged = np.unique(np.r_[times, extra])
            multipliers = np.maximum(0.0, lam + mu * last["g"])
            lam = np.zeros((len(problem.i), len(merged)))
            lam[:, np.searchsorted(merged, times)] = multipliers
            times = merged
            mu = min(mu * 4.0, 1e10)
    except _TimeBudget:
        status = "time_budget"
    if last:
        inspect(last["y"])
    selected = best[1] if best is not None else problem.references(last.get("y", y))
    check = separation(selected, physical_delta, eps)
    planning_check = separation(selected, planning_delta, eps) if clearance else check
    return selected, dict(**metadata, ok=bool(best is not None), status=status,
                          runtime=time.perf_counter() - start, iterations=len(history),
                          training_times=len(times), pair_time_constraints=len(problem.i) * len(times),
                          min_margin=check["min_margin"],
                          min_planning_margin_m=planning_check["min_margin"],
                          planning_clearance_satisfied=planning_check["ok"],
                          cost=sum(r["cost"] for r in selected), history=history)
