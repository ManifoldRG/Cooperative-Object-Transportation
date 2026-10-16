"""Steps 4-6: ball corridors, independent per-payload 6-DoF solves, relaxation.

Step 4. Payload j is confined to ||r_k - rbar_k|| <= eps_k. If the references
are conflict-free (Step 2 with these eps), trajectories inside the corridors
satisfy the collision constraint, so payloads are solved independently.

Step 5. Each payload is solved by the COT envelope-gradient method (C-GD by
default, D-GD optional) with its elastic corridor, warm-started from the
attitude reference used for allocation.

Step 6. If the slack is nonzero, the corridor is widened by the slack (plus a
margin) up to the room its neighbors leave; if a corridor cannot grow enough,
the references are repaired again with the larger radii (pushing the
waypoints apart) and the affected payloads are re-solved. A payload that is
infeasible even without a corridor makes the problem infeasible.
"""
from __future__ import annotations

import numpy as np

from .conflicts import repair
from .gd_corridor import solve_centralized_gd, solve_decentralized_gd


def room(sc, refs, eps, j):
    """Largest eps_j,k allowed by every neighbor (per step), >= 0."""
    cap = np.full(sc.N + 1, np.inf)
    for l in range(sc.M):
        if l == j:
            continue
        d = np.linalg.norm(refs[j]["r"] - refs[l]["r"], axis=1)
        vm = 0.5 * sc.dt * np.linalg.norm(refs[j]["v"] - refs[l]["v"], axis=1)
        cap = np.minimum(cap, d - sc.delta - eps[l] - vm)
    cap = np.maximum(cap, 0.0)
    cap[0] = cap[-1] = 0.0
    return cap


def solve_payload(sc, j, occupied, ref, tau_ref, eps_j, rho, method="cgd",
                  budget_s=60.0, seed=0, max_restarts=None, corridor=True,
                  objective="fuel"):
    sp, bc = sc.sys_params_for(j, occupied)
    cor = dict(rbar=ref["r"], eps=eps_j, rho=rho) if corridor else None
    common = dict(corridor=cor, max_runtime_s=budget_s, initial_tau=tau_ref,
                  objective=objective)
    if method == "dgd":
        return solve_decentralized_gd(sp, bc, sc.epsilon, base_seed=seed,
                                      max_restarts_per_island=max_restarts, **common)
    return solve_centralized_gd(sp, bc, sc.epsilon, seed=seed,
                                restarts=max_restarts != 1, max_restarts=max_restarts,
                                **common)


def run_oracle(sc, refs, waypoints, occupied, tau_refs, eps, rho=1e3,
               method="cgd", budget_s=60.0, seed=0, max_restarts=None,
               slack_tol=1e-3, expand_margin=1.0, max_expand=3, max_outer=3,
               psis=None, verbose=False, objective="fuel"):
    """Returns dict(status, payloads=[...], refs, waypoints, eps, log)."""
    say = print if verbose else (lambda *a, **k: None)
    eps = [np.asarray(e, float).copy() for e in eps]
    results = [None] * sc.M
    todo = set(range(sc.M))
    log = []
    status = "ok"
    for outer in range(max_outer + 1):
        need_room = {}
        for j in sorted(todo):
            for it in range(max_expand + 1):
                res = solve_payload(sc, j, occupied[j], refs[j], tau_refs[j], eps[j], rho,
                                    method, budget_s, seed, max_restarts, objective=objective)
                s = res["slack"]
                log.append(dict(outer=outer, payload=j, attempt=it, fuel=res["fuel"],
                                cost=res["objective_cost"],
                                max_slack=float(s.max()), max_eps=float(eps[j].max())))
                say(f"  payload {j}: attempt {it}, {objective} {res['objective_cost']:.4f}, "
                    f"max slack {s.max():.2e} m, max eps {eps[j].max():.2f} m")
                results[j] = res
                if s.max() <= slack_tol:
                    break
                cap = room(sc, refs, eps, j)
                want = eps[j] + np.where(s > slack_tol, s + expand_margin, 0.0)
                new = np.minimum(want, cap)
                if np.all(new <= eps[j] + 1e-9):      # no room left to grow
                    need_room[j] = want
                    break
                eps[j] = new
            else:
                need_room[j] = eps[j] + results[j]["slack"] + expand_margin
        if not need_room:
            break
        if outer == max_outer:
            status = "slack"
            break
        # Step 6 fallback: is the payload feasible at all without a corridor?
        for j in list(need_room):
            try:
                free = solve_payload(sc, j, occupied[j], refs[j], tau_refs[j], eps[j], rho,
                                     method, budget_s, seed, max_restarts, corridor=False,
                                     objective=objective)
                feasible = np.isfinite(free["cost"])
            except RuntimeError:
                feasible = False
            if not feasible:
                return dict(status="infeasible", payloads=results, refs=refs,
                            waypoints=waypoints, eps=eps, log=log)
        # back to Step 3: repair with the larger radii to make room
        for j, want in need_room.items():
            eps[j] = np.maximum(eps[j], want)
            eps[j][0] = eps[j][-1] = 0.0
        before = [r["r"].copy() for r in refs]
        refs, waypoints, rlog, ok = repair(sc, refs, waypoints, eps, psis=psis,
                                           objective=objective)
        log.append(dict(outer=outer, repair=rlog, converged=ok))
        say(f"  re-repair with larger corridors: {len(rlog)} waypoint(s), converged={ok}")
        todo = {j for j in range(sc.M)
                if j in need_room or not np.allclose(before[j], refs[j]["r"])}
    return dict(status=status, payloads=results, refs=refs, waypoints=waypoints,
                eps=eps, log=log)
