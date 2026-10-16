"""Docking phase: free-flying agents rendezvous with their allocated points.

Timeline. Docking happens on t in [-T_dock, 0] and ends exactly at the
transport start state x0 of every payload, so the transport solution (t in
[0, tf]) is unchanged. During docking the payloads are uncontrolled: their
translational state is propagated BACKWARD from x0 through the same discrete
Tschauner-Hempel dynamics, and their attitude stays at phi0 (omega0 = 0).

Assignment. The allocation fixes which points are occupied; agents are
identical but start at different places, so agents are matched to occupied
points by the Hungarian method on the open-loop docking fuel.

Controller. Shrinking-horizon model predictive control in the style of Park,
Di Cairano and Kolmanovsky, "Model Predictive Control for Spacecraft
Rendezvous and Docking with a Rotating/Tumbling Platform and for Debris
Avoidance", ACC 2011: during the final approach the agent stays inside a
line-of-sight cone from the docking port (vertex moved r_tol inside the
payload) and in front of the tangent half-plane at the port, its approach
speed is limited, and it docks softly (velocity matched to the port). Here
the cost is fuel sum ||u||, the dynamics are the paper's discrete TH model
(agents as point masses, thrust in the Hill frame, ||u|| <= u_max), and
every constraint except the dynamics is softened with a large penalty so the
closed loop stays feasible under actuation noise.
"""
from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
from scipy.optimize import linear_sum_assignment

from spacecraft_libraries.new_opts import so3_exp, th_psi_matrix


def agent_starts(sc, n_agents=None, seed=0, radius=None, clearance=None, min_sep=30.0,
                 max_tries=20000):
    """Random agent start positions in a ball around the scene, away from the
    payloads' docking-phase start positions and from each other."""
    rng = np.random.default_rng(seed)
    n = sc.n_agents if n_agents is None else n_agents
    radius = sc.meta.get("ring_radius", 500.0) * 1.2 if radius is None else radius
    ext = max(P.extent for P in sc.payloads)
    clearance = 2.5 * ext if clearance is None else clearance
    centers = [P.x0.r for P in sc.payloads]
    pts = []
    for _ in range(max_tries):
        p = rng.uniform(-1, 1, 3) * radius
        if np.linalg.norm(p) > radius:
            continue
        if any(np.linalg.norm(p - c) < clearance for c in centers):
            continue
        if any(np.linalg.norm(p - q) < min_sep for q in pts):
            continue
        pts.append(p)
        if len(pts) == n:
            return np.asarray(pts)
    raise RuntimeError("could not place agents: enlarge radius")


def _step_matrix(sc, t, dt):
    Psi = np.asarray(th_psi_matrix(sc.mu, sc.a, sc.e, t))
    A = np.eye(6)
    A[:3, 3:] += dt * np.eye(3)
    A[3:, :] += dt * Psi[3:6, :]
    return A


def payload_drift_backward(sc, j, T_dock, N_dock):
    """Uncontrolled payload states at t_k = -T_dock + k*dt, k = 0..N_dock,
    ending at the transport start state (x0) at k = N_dock."""
    dt = T_dock / N_dock
    x = np.concatenate([sc.payloads[j].x0.r, sc.payloads[j].x0.v])
    xs = [x]
    for k in range(N_dock - 1, -1, -1):
        A = _step_matrix(sc, -T_dock + k * dt, dt)
        x = np.linalg.solve(A, x)
        xs.append(x)
    xs = np.asarray(xs[::-1])
    return xs[:, :3], xs[:, 3:]


class DockingProblem:
    """Shared docking data for one scenario."""

    def __init__(self, sc, starts, T_dock=400.0, N_dock=40, m_agent=20.0, u_max=2.0,
                 los_half_angle_deg=30.0, los_fraction=0.3, r_tol=2.0,
                 approach_rate=1.0 / 30.0, approach_v0=0.05, penalty=1e3):
        self.sc, self.starts = sc, np.asarray(starts, float)
        self.T, self.N = float(T_dock), int(N_dock)
        self.dt = self.T / self.N
        self.m, self.u_max = m_agent, u_max
        self.cos_los = np.cos(np.deg2rad(los_half_angle_deg))
        self.k_los = int(np.floor((1 - los_fraction) * self.N))
        self.r_tol, self.rate, self.v0, self.W = r_tol, approach_rate, approach_v0, penalty
        self.times = -self.T + self.dt * np.arange(self.N + 1)
        self.Psis = [np.asarray(th_psi_matrix(sc.mu, sc.a, sc.e, t))[3:6, :]
                     for t in self.times[:-1]]
        self.drift = [payload_drift_backward(sc, j, self.T, self.N) for j in range(sc.M)]
        self.R0 = [so3_exp(np.asarray(P.x0.phi, float)) for P in sc.payloads]

    def port(self, j, p):
        """Docking-port trajectory (positions, velocities) and outward normal."""
        off = self.R0[j] @ self.sc.payloads[j].points[p]
        r, v = self.drift[j]
        return r + off, v, off / np.linalg.norm(off)

    def solve(self, j, p, r0, v0, k0=0):
        """Fuel-optimal plan from step k0 (state r0, v0) to the port.
        Returns dict(ok, u (n,3), r, v, fuel, terminal_error)."""
        n = self.N - k0
        pr, pv, nh = self.port(j, p)
        r = cp.Variable((n + 1, 3))
        v = cp.Variable((n + 1, 3))
        u = cp.Variable((n, 3))
        cons = [r[0] == r0, v[0] == v0, cp.norm(u, 2, axis=1) <= self.u_max]
        soft = 0
        for i in range(n):
            Psi = self.Psis[k0 + i]
            cons += [r[i + 1] == r[i] + self.dt * v[i],
                     v[i + 1] == v[i] + self.dt * (Psi[:, :3] @ r[i] + Psi[:, 3:] @ v[i]
                                                   + u[i] / self.m)]
        for i in range(1, n + 1):
            k = k0 + i
            if k < self.k_los:
                continue
            e = r[i] - pr[k]
            ev = e + self.r_tol * nh
            # LOS cone (vertex r_tol inside the payload), tangent half-plane,
            # approach-speed limit proportional to the along-axis distance
            soft += cp.pos(self.cos_los * cp.norm(ev) - nh @ ev)
            soft += cp.pos(-(nh @ e))
            soft += cp.pos(cp.norm(v[i] - pv[k]) - self.v0 - self.rate * (nh @ ev))
        soft += cp.norm(r[n] - pr[self.N]) + cp.norm(v[n] - pv[self.N])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(u, 2, axis=1)) + self.W * soft), cons)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                prob.solve(solver="CLARABEL")
            except cp.error.SolverError:
                prob.solve(solver="SCS")
        if u.value is None:
            return dict(ok=False)
        uv = np.asarray(u.value)
        return dict(ok=True, u=uv, r=np.asarray(r.value), v=np.asarray(v.value),
                    fuel=float(np.linalg.norm(uv, axis=1).sum()),
                    terminal_error=float(np.linalg.norm(r.value[n] - pr[self.N])
                                         + np.linalg.norm(v.value[n] - pv[self.N])))

    def step(self, k, r, v, u):
        Psi = self.Psis[k]
        return (r + self.dt * v,
                v + self.dt * (Psi[:, :3] @ r + Psi[:, 3:] @ v + u / self.m))


def assign(dp, occupied):
    """Match agents to occupied points (Hungarian on open-loop docking fuel)."""
    targets = [(j, p) for j, occ in enumerate(occupied) for p in occ]
    n = len(dp.starts)
    if len(targets) != n:
        raise ValueError(f"{n} agents but {len(targets)} occupied points")
    cost = np.zeros((n, n))
    for a in range(n):
        for t, (j, p) in enumerate(targets):
            sol = dp.solve(j, p, dp.starts[a], np.zeros(3))
            cost[a, t] = sol["fuel"] + dp.W * sol["terminal_error"] if sol["ok"] else 1e12
    rows, cols = linear_sum_assignment(cost)
    return [targets[c] for c in cols[np.argsort(rows)]], cost


def simulate(dp, assignment, noise=0.02, seed=0, capture_tol=(0.5, 0.05)):
    """Closed-loop shrinking-horizon MPC for every agent. noise: relative
    thrust-magnitude error and direction error [rad] (std). Returns dict with
    per-agent r (N+1,3), u (N,3), fuel, capture errors, and safety metrics."""
    rng = np.random.default_rng(seed)
    agents = []
    for a, (j, p) in enumerate(assignment):
        r, v = dp.starts[a].copy(), np.zeros(3)
        rs, us = [r.copy()], []
        for k in range(dp.N):
            plan = dp.solve(j, p, r, v, k0=k)
            u_cmd = plan["u"][0] if plan["ok"] else np.zeros(3)
            u = u_cmd * (1 + noise * rng.standard_normal())
            if noise > 0 and np.linalg.norm(u) > 0:
                axis = np.cross(u, rng.standard_normal(3))
                axis /= np.linalg.norm(axis) + 1e-12
                u = so3_exp(axis * noise * rng.standard_normal()) @ u
            r, v = dp.step(k, r, v, u)
            rs.append(r.copy())
            us.append(u)
        pr, pv, _ = dp.port(j, p)
        pos_err = float(np.linalg.norm(r - pr[-1]))
        vel_err = float(np.linalg.norm(v - pv[-1]))
        agents.append(dict(payload=j, point=p, r=np.asarray(rs), u=np.asarray(us),
                           fuel=float(np.linalg.norm(us, axis=1).sum()),
                           pos_err=pos_err, vel_err=vel_err,
                           captured=bool(pos_err <= capture_tol[0] and vel_err <= capture_tol[1])))
    # safety: agent-agent distance, and agent distance to payloads other than its target
    dmin_aa = np.inf
    for a in range(len(agents)):
        for b in range(a + 1, len(agents)):
            dmin_aa = min(dmin_aa, float(np.min(np.linalg.norm(agents[a]["r"] - agents[b]["r"], axis=1))))
    dmin_ap = np.inf
    for ag in agents:
        for j in range(dp.sc.M):
            if j == ag["payload"]:
                continue
            d = np.linalg.norm(ag["r"] - dp.drift[j][0], axis=1) - dp.sc.payloads[j].extent
            dmin_ap = min(dmin_ap, float(np.min(d)))
    return dict(agents=agents, times=dp.times, payload_r=[d[0] for d in dp.drift],
                min_agent_agent=dmin_aa, min_agent_other_payload=dmin_ap,
                all_captured=all(a["captured"] for a in agents),
                fuel=float(sum(a["fuel"] for a in agents)))
