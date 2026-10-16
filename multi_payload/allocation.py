"""Allocation: surrogate fuel gamma_j, residual e_j, greedy and MISOCP rules.

With the references fixed, the fuel with which the agents at the occupied
points P_j of payload j follow them is the SOCP
    gamma_j = min sum_{k,p} ||u_pk||
              s.t. sum_p u_pk = R_k^T F_k,  sum_p p x u_pk = tau_k,
                   cos(nu) ||u_pk|| <= -p_hat . u_pk        (exact inward cone)
(+inf if infeasible). The surrogate J_hat = sum_j gamma_j upper-bounds the
oracle value of the allocation. When some payload cannot produce its required
wrenches, allocations are compared by the normalized residual
    e_j = min sum_k ||w_k - sum_p G(p) u_pk|| / ||w_k||,
which is zero exactly when gamma_j is finite.

Rule: minimize J_hat if any allocation makes every gamma_j finite, otherwise
minimize max_j e_j (ties by sum_j e_j). Allocation never fails; only the
oracle reports infeasibility.
"""
from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

RES_TOL = 1e-6


def _cross_rows(p, u):
    """p x u for constant p and cvxpy 3-vector u."""
    return cp.hstack([p[1] * u[2] - p[2] * u[1],
                      p[2] * u[0] - p[0] * u[2],
                      p[0] * u[1] - p[1] * u[0]])


def _step_scale(points, W, nu):
    """Per-step thrust scale: force plus the thrust a single agent at the
    shortest lever arm would need for the torque."""
    lever = float(np.min(np.linalg.norm(points, axis=1)))
    s = max(np.sin(nu), 1e-3)
    return np.linalg.norm(W[:, :3], axis=1) + np.linalg.norm(W[:, 3:], axis=1) / (lever * s)


def _bigM(points, W, nu):
    """Bound on any single agent's thrust (only switches agents off)."""
    return 100.0 * max(float(np.max(_step_scale(points, W, nu))), 1e-6)


def _rho(points, W, nu):
    """Residual penalty: dominates any fuel difference (exact-penalty regime)
    without the huge coefficients that break the conic solvers."""
    return 100.0 * max(float(np.sum(_step_scale(points, W, nu))), 1e-6)


def _weights(W):
    n = np.linalg.norm(W, axis=1)
    return np.where(n > 1e-12, 1.0 / np.maximum(n, 1e-12), 0.0)


def _wrench_constraints(points, W, nu, U, S):
    """Constraints sum_p G(p) u_pk + S_k = w_k and exact cones. U[(p,k)] 3-vectors."""
    P, N = len(points), len(W)
    cosn = float(np.cos(nu))
    cons = []
    for k in range(N):
        f = sum(U[p * N + k] for p in range(P))
        t = sum(_cross_rows(points[p], U[p * N + k]) for p in range(P))
        cons.append(cp.hstack([f, t]) + S[k] == W[k])
    for p in range(P):
        ph = points[p] / np.linalg.norm(points[p])
        for k in range(N):
            cons.append(cosn * cp.norm(U[p * N + k]) <= -(ph @ U[p * N + k]))
    return cons


class PayloadAllocator:
    """gamma_j and e_j of one payload for any subset of its points, solved
    with a single parametric SOCP (occupancy mask as a cvxpy Parameter)."""

    def __init__(self, points, W, nu, solver="CLARABEL"):
        self.points = np.asarray(points, float)
        self.W = np.asarray(W, float)
        self.nu = nu
        self.solver = solver
        P, N = len(self.points), len(self.W)
        self.P, self.N = P, N
        self.mask = cp.Parameter(P, nonneg=True)
        self.U = cp.Variable((P * N, 3))
        S = cp.Variable((N, 6))
        wts = _weights(self.W)
        big = _bigM(self.points, self.W, nu)
        cons = _wrench_constraints(self.points, self.W, nu, self.U, S)
        for p in range(P):
            for k in range(N):
                cons.append(cp.norm(self.U[p * N + k]) <= big * self.mask[p])
        self.fuel = cp.sum(cp.norm(self.U, 2, axis=1))
        self.res = cp.sum(cp.multiply(wts, cp.norm(S, 2, axis=1)))
        self.prob = cp.Problem(cp.Minimize(self.fuel + _rho(self.points, self.W, nu) * self.res),
                               cons)
        self._cache = {}

    def evaluate(self, occupied):
        """(fuel, e, U (|occ|,N,3) body frame). fuel is gamma_j when e == 0."""
        key = frozenset(occupied)
        if key in self._cache:
            return self._cache[key]
        if not occupied:
            out = (0.0, float(np.count_nonzero(_weights(self.W))), np.zeros((0, self.N, 3)))
            self._cache[key] = out
            return out
        self.mask.value = np.array([1.0 if p in occupied else 0.0 for p in range(self.P)])
        for solver in dict.fromkeys((self.solver, "MOSEK", "SCS")):
            if solver not in cp.installed_solvers():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.prob.solve(solver=solver)
                except cp.error.SolverError:
                    continue
            if self.prob.status in ("optimal", "optimal_inaccurate") and self.res.value is not None:
                break
        else:
            raise RuntimeError(f"allocation SOCP failed for points {sorted(occupied)}")
        e = float(self.res.value)
        e = 0.0 if e < RES_TOL else e
        Uall = np.asarray(self.U.value).reshape(self.P, self.N, 3)
        out = (float(self.fuel.value), e, Uall[sorted(occupied)])
        self._cache[key] = out
        return out


def _summary(allocators, occupied, method, log=None, extra=None):
    res = [a.evaluate(o) for a, o in zip(allocators, occupied)]
    es = [r[1] for r in res]
    feasible = max(es) == 0.0
    out = dict(method=method, occupied=[sorted(o) for o in occupied],
               case="fuel" if feasible else "residual",
               gamma=[r[0] if r[1] == 0.0 else float("inf") for r in res],
               residual=es, J_hat=float(sum(r[0] for r in res)) if feasible else float("inf"),
               U=[r[2] for r in res], log=log or [])
    if extra:
        out.update(extra)
    return out


def greedy(allocators, n_agents):
    """Sequential greedy: each agent takes the open point minimizing the
    lexicographic key (max_j e_j, sum_j e_j, sum_j fuel_j)."""
    M = len(allocators)
    occ = [set() for _ in range(M)]
    state = [a.evaluate(set()) for a in allocators]
    log = []
    for n in range(n_agents):
        best = None
        for j, a in enumerate(allocators):
            for p in range(a.P):
                if p in occ[j]:
                    continue
                cand = a.evaluate(occ[j] | {p})
                es = [cand[1] if i == j else state[i][1] for i in range(M)]
                fs = [cand[0] if i == j else state[i][0] for i in range(M)]
                key = (max(es), sum(es), sum(fs))
                if best is None or key < best[0]:
                    best = (key, j, p, cand)
        if best is None:
            break  # every point occupied
        key, j, p, cand = best
        occ[j].add(p)
        state[j] = cand
        log.append(dict(agent=n + 1, payload=j, point=p, max_e=key[0], sum_e=key[1],
                        fuel=key[2]))
    return _summary(allocators, occ, "greedy", log)


def misocp(allocators, n_agents, solver="MOSEK", time_limit=300.0):
    """Exact allocation by mixed-integer SOCP (certified by branch and bound).

    Stage 1: minimize max_j e_j. If the optimum is 0, stage 2 minimizes total
    fuel with zero residual; otherwise stage 2 minimizes sum_j e_j subject to
    max_j e_j at its optimum.
    """
    M = len(allocators)
    z, U, S, fuel, res, base = [], [], [], [], [], []
    for a in allocators:
        zj = cp.Variable(a.P, boolean=True)
        Uj = cp.Variable((a.P * a.N, 3))
        Sj = cp.Variable((a.N, 6))
        big = _bigM(a.points, a.W, a.nu)
        cj = _wrench_constraints(a.points, a.W, a.nu, Uj, Sj)
        for p in range(a.P):
            for k in range(a.N):
                cj.append(cp.norm(Uj[p * a.N + k]) <= big * zj[p])
        z.append(zj)
        U.append(Uj)
        S.append(Sj)
        base += cj
        fuel.append(cp.sum(cp.norm(Uj, 2, axis=1)))
        res.append(cp.sum(cp.multiply(_weights(a.W), cp.norm(Sj, 2, axis=1))))
    total_pts = sum(a.P for a in allocators)
    count = [sum(cp.sum(zj) for zj in z) == min(n_agents, total_pts)]
    opts = {}
    if solver == "MOSEK":
        opts = dict(mosek_params={"MSK_DPAR_MIO_MAX_TIME": float(time_limit)})

    t = cp.Variable()
    p1 = cp.Problem(cp.Minimize(t), base + count + [r <= t for r in res])
    p1.solve(solver=solver, **opts)
    t_star = float(t.value)
    if t_star <= RES_TOL:
        p2 = cp.Problem(cp.Minimize(sum(fuel)), base + count + [Sj == 0 for Sj in S])
        case = "fuel"
    else:
        p2 = cp.Problem(cp.Minimize(sum(res)),
                        base + count + [r <= t_star * (1 + 1e-6) + RES_TOL for r in res])
        case = "residual"
    p2.solve(solver=solver, **opts)
    occupied = [set(int(i) for i in np.nonzero(np.round(zj.value) > 0.5)[0]) for zj in z]
    # report every quantity through the same parametric allocators as greedy
    return _summary(allocators, occupied, "misocp",
                    extra=dict(stage1_max_residual=t_star, stage2_status=p2.status,
                               stage2_value=float(p2.value), mip_case=case))


def build_allocators(sc, wrenches, solver="CLARABEL"):
    return [PayloadAllocator(sc.payloads[j].points, wrenches[j], sc.nu, solver)
            for j in range(sc.M)]
