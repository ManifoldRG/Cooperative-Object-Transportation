"""Diminishing-returns check on the TRUE oracle value J*(A), single payload.

For one payload from the T-AES scenario generator, with P candidate attachment
points (the generator's agent positions), every subset A is scored by solving
the baseline direct NLP (solve_centralized_nlp_th: Tschauner-Hempel + body-frame
thrust + inward cones + fuel objective) with that subset of agents attached.
Collisions are not involved: with one payload J* is the single-payload oracle,
and with collisions inactive the multi-payload J* is a sum of these.

The NLP is non-convex, so each subset gets several starts and keeps the best
rollout-verified one:
  * zero thrust,
  * warm start from the best solution of A minus one point (that agent idle),
    which makes the previous value reachable and so targets monotonicity,
  * random thrust guesses.
Every returned control is re-simulated through the discrete dynamics; a solve
counts only if the terminal error is below ROLLOUT_TOL. Subsets with no verified
solve get J = inf. The spread between converged starts estimates solver noise.

Usage: py oracle_check.py --seed 0 --P 5 --steps 12 --angle 60
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

COT_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COT_REPO))
sys.path.insert(0, str(Path(__file__).parent))

ROLLOUT_TOL = 1e-2   # terminal error (m + m/s + rad + rad/s) for a verified solve
RHO = 1e3            # residual weight for ranking unverified subsets
N_STARTS = 3         # starts per subset: zero, warm (if any), random
RAND_SCALE = 0.05    # std of random thrust guesses


def popcount(x):
    return bin(x).count("1")


def make_problem(seed, P, steps, angle_deg):
    from spacecraft_libraries.evaluation.comparison import random_scenario_generator
    sp, bc, eps = random_scenario_generator(fixed_agents_num=P, seed=seed,
                                            thrust_angle=np.deg2rad(angle_deg))
    sp.N = steps
    return sp, bc, eps


def rollout_error(sp, bc, U):
    """Terminal error of controls U (agents, N, 3) through the discrete dynamics."""
    from scipy.spatial.transform import Rotation
    from spacecraft_libraries.new_opts import th_psi_matrix, state_attitude_to_phi
    N, dt = sp.N, bc.tf / sp.N
    I = np.asarray(sp.I, float)
    I_inv = np.linalg.inv(I)
    rs = [np.asarray(x, float) for x in sp.rs]
    r = np.asarray(bc.x0.r, float).copy()
    v = np.asarray(bc.x0.v, float).copy()
    R = Rotation.from_rotvec(state_attitude_to_phi(bc.x0)).as_matrix()
    w = np.asarray(bc.x0.omega, float).copy()
    for k in range(N):
        Uk = U[:, k, :]
        torque = sum(np.cross(rs[i], Uk[i]) for i in range(len(rs)))
        Psi = np.asarray(th_psi_matrix(sp.mu, sp.a, sp.e, k * dt))
        r_new = r + dt * v
        v_new = v + dt * (Psi[3:6, :] @ np.concatenate([r, v]) + R @ Uk.sum(axis=0) / sp.m)
        R = R @ Rotation.from_rotvec(dt * w).as_matrix()
        w = w + dt * (I_inv @ (torque - np.cross(w, I @ w)))
        r, v = r_new, v_new
        if not (np.isfinite(r).all() and np.isfinite(v).all() and np.isfinite(w).all()):
            return np.inf
    try:
        Rf = Rotation.from_rotvec(state_attitude_to_phi(bc.xf)).as_matrix()
        att = np.linalg.norm(Rotation.from_matrix(R.T @ Rf).as_rotvec())
    except Exception:
        return np.inf
    return float(np.linalg.norm(r - bc.xf.r) + np.linalg.norm(v - bc.xf.v)
                 + att + np.linalg.norm(w - bc.xf.omega))


def solve_subset(task):
    """Worker: all starts for one subset. Returns (A, best record, per-start fuels)."""
    from spacecraft_libraries.solvers.centralized_nlp_th import solve_centralized_nlp_th
    seed, P, steps, angle, A, warm, budget = task
    sp, bc, eps = make_problem(seed, P, steps, angle)
    idx = [i for i in range(P) if A >> i & 1]
    sp.rs = [sp.rs[i] for i in idx]
    rng = np.random.default_rng(A * 7919 + seed)
    starts = [("zero", None)]
    if warm is not None:
        starts.append(("warm", warm))
    # random starts fill up to N_STARTS (one extra when there is no warm start)
    starts += [(f"rand{r}", rng.normal(scale=RAND_SCALE, size=(len(idx), steps, 3)))
               for r in range(N_STARTS - len(starts))]
    best, fuels = None, {}
    for name, guess in starts:
        t0 = time.perf_counter()
        try:
            res = solve_centralized_nlp_th(sp, bc, eps, max_runtime_s=budget, U_guess=guess)
        except Exception:
            continue
        U = np.asarray(res["control"], float)
        fuel = float(np.linalg.norm(U, axis=2).sum())
        err = rollout_error(sp, bc, U)
        ok = err < ROLLOUT_TOL
        fuels[name] = fuel if ok else None
        rec = dict(fuel=fuel, err=err, ok=ok, start=name, status=res["ipopt_status"],
                   wall=time.perf_counter() - t0, U=U)
        key = (not ok, fuel if ok else err)
        if best is None or key < (not best["ok"], best["fuel"] if best["ok"] else best["err"]):
            best = rec
    return A, best, fuels


def pad_warm(best_by_subset, A, P):
    """Best verified solution among A minus one point, with the removed agent idle."""
    cand = None
    for e in range(P):
        if not A >> e & 1:
            continue
        B = A & ~(1 << e)
        rec = best_by_subset.get(B)
        if rec is None or not rec["ok"]:
            continue
        if cand is None or rec["fuel"] < cand[1]["fuel"]:
            cand = (e, rec)
    if cand is None:
        return None
    e, rec = cand
    idx_B = [i for i in range(P) if (A & ~(1 << e)) >> i & 1]
    idx_A = [i for i in range(P) if A >> i & 1]
    U = np.zeros((len(idx_A), rec["U"].shape[1], 3))
    for row, i in enumerate(idx_B):
        U[idx_A.index(i)] = rec["U"][row]
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--angle", type=float, default=60.0, help="cone half-angle, deg")
    ap.add_argument("--min-size", type=int, default=1)
    ap.add_argument("--budget", type=float, default=60.0, help="IPOPT cpu limit per start, s")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    tag = f"oracle_seed{args.seed}_P{args.P}_T{args.steps}_ang{args.angle:g}"

    sp, bc, _ = make_problem(args.seed, args.P, args.steps, args.angle)
    print(f"{tag}: m={sp.m:.1f} kg, tf={bc.tf:.0f} s, |r_f|={np.linalg.norm(bc.xf.r):.0f} m, "
          f"|phi_f|={np.linalg.norm(bc.xf.phi):.2f} rad, |rho|="
          f"{np.round([np.linalg.norm(r) for r in sp.rs], 1).tolist()}", flush=True)

    P = args.P
    n = 1 << P
    best = {}
    f = np.full(n, np.inf)
    fpen = np.full(n, np.inf)
    starts_log = {}
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for size in range(args.min_size, P + 1):
            subsets = [A for A in range(n) if popcount(A) == size]
            tasks = [(args.seed, P, args.steps, args.angle, A, pad_warm(best, A, P), args.budget)
                     for A in subsets]
            for A, rec, fuels in pool.map(solve_subset, tasks):
                if rec is None:
                    continue
                best[A] = rec
                f[A] = rec["fuel"] if rec["ok"] else np.inf
                fpen[A] = rec["fuel"] + RHO * (0.0 if rec["ok"] else rec["err"])
                starts_log[A] = fuels
            ok = np.isfinite(f[[A for A in subsets]]).sum()
            print(f"  size {size:2d}: {ok}/{len(subsets)} verified "
                  f"[{time.time() - t_all:.0f}s]", flush=True)

    f[0] = np.inf
    fpen[0] = np.inf

    # Solver-noise estimate: relative spread among verified starts of each subset.
    spreads, warm_wins = [], 0
    for A, fuels in starts_log.items():
        vals = [v for v in fuels.values() if v is not None]
        if len(vals) >= 2:
            spreads.append((max(vals) - min(vals)) / min(vals))
        if best[A]["ok"] and best[A]["start"] == "warm":
            warm_wins += 1
    spreads = np.array(spreads)

    # Monotone envelope: idle agents are allowed (U = 0 satisfies the cone), so a
    # verified solution for any B subset of A is feasible for A, and
    # J*(A) <= min_{B subset A} J_nlp(B). This removes local-minimum noise that
    # shows up as monotonicity violations and is a tighter estimate of J*.
    f_env = f.copy()
    for A in sorted(range(1, n), key=popcount):
        for e in range(P):
            if A >> e & 1:
                f_env[A] = min(f_env[A], f_env[A & ~(1 << e)])
    fpen_env = np.where(np.isfinite(f_env), f_env, fpen)

    from authority_check import analyze, greedy_report
    res = analyze(f_env, P)
    res["greedy"] = greedy_report(f_env, fpen_env, P, res["opt_by_N"])
    raw = analyze(f, P)
    res["raw_nlp"] = dict(monotonicity_violations=raw["monotonicity_violations"],
                          diminishing_returns_closed=raw["diminishing_returns_closed"],
                          submodularity_ratio_closed=raw["submodularity_ratio_closed"])
    res["envelope_improved_subsets"] = int((f_env < f - 1e-9).sum())
    res["solver_noise"] = dict(
        subsets_with_2plus_verified=int(len(spreads)),
        frac_spread_gt_1pct=float((spreads > 0.01).mean()) if len(spreads) else None,
        median_spread=float(np.median(spreads)) if len(spreads) else None,
        max_spread=float(spreads.max()) if len(spreads) else None,
        best_from_warm_start=int(warm_wins))
    res["config"] = dict(vars(args), out=str(args.out), rollout_tol=ROLLOUT_TOL,
                         seconds=time.time() - t_all)
    np.save(args.out / f"{tag}.npy", np.stack([f, fpen, f_env]))
    with open(args.out / f"{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=str)

    c = res["diminishing_returns_closed"]
    g = res["greedy"]
    print(f"{tag} done in {time.time() - t_all:.0f}s", flush=True)
    print(f"  smallest feasible allocation: {res['min_closed_size']} agents; "
          f"monotonicity violations: {res['monotonicity_violations']}", flush=True)
    print("  [monotone envelope of NLP values]", flush=True)
    print(f"  diminishing returns (feasible A): {c['significant']} significant / "
          f"{c['triples']} triples, worst excess {100 * c['worst_excess_rel']:.1f}% of J, "
          f"worst ratio {c['worst_ratio']:.3f}", flush=True)
    print(f"  submodularity ratio: {res['submodularity_ratio_closed']:.3f}", flush=True)
    rc = res["raw_nlp"]["diminishing_returns_closed"]
    print(f"  [raw NLP, before envelope] monotonicity violations: "
          f"{res['raw_nlp']['monotonicity_violations']}, DR significant: "
          f"{rc['significant']}/{rc['triples']}; envelope improved "
          f"{res['envelope_improved_subsets']} subsets", flush=True)
    print(f"  solver noise: {res['solver_noise']}", flush=True)
    for N in sorted(g):
        print(f"  N={N:2d}: opt={g[N]['opt']:.4f}  fwd/opt={g[N]['forward_ratio']:.3f}  "
              f"rev/opt={g[N]['reverse_ratio']:.3f}", flush=True)
    os._exit(0)  # skip CasADi's interpreter-teardown segfault


if __name__ == "__main__":
    main()
