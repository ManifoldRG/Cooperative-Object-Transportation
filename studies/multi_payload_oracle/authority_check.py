"""Numerical check of diminishing returns for the control-authority surrogate.

For one payload, every subset A of the open attachment points is scored by
    f(A) = sum_k gamma(w_k; A),
the minimum fuel with which the agents at A produce each required body-frame
wrench w_k (eq:gauge in the notes), with f(A) = inf when some w_k is infeasible.
Enumerating all 2^P subsets then gives:
  * monotonicity violations,
  * supermodularity (diminishing-returns) violations, split by whether the
    smaller set is already able to produce every required wrench ("closed"),
  * the submodularity ratio of the fuel reduction over closed sets,
  * forward greedy (penalized ranking, eq:greedy) and reverse greedy versus the
    exhaustive optimum at every agent count N.
A three-payload case compares three sequential allocation rules against the
exact separable optima (fuel, and max-min authority ratio):
  * fuel:       each agent takes the point that most reduces the surrogate fuel,
  * maxmin:     each agent maximizes min_j lambda_j (ties to the first candidate),
  * bottleneck: each agent joins the payload with the smallest lambda_j and takes
                the point maximizing it, ranked by penalized residual while it is 0,
where lambda_j = max{t : t*w_k producible for all k} under per-agent thrust u_max.
"""

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import cvxpy as cp
import numpy as np

INF = np.inf
FEAS_TOL = 1e-6   # wrench residual above which a subset is declared infeasible
VAL_TOL = 1e-6    # tolerance on value comparisons
SIG_REL = 1e-3    # relative size of a diminishing-returns violation counted as significant


def skew(p):
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])


def box_points(h, frac=0.5):
    """Two off-center open points on each face of a box with half-extents h."""
    pts = []
    for ax in range(3):
        o1, o2 = [a for a in range(3) if a != ax]
        for sgn in (1, -1):
            for t in (1, -1):
                p = np.zeros(3)
                p[ax] = sgn * h[ax]
                p[o1] = t * frac * h[o1]
                p[o2] = t * frac * h[o2]
                pts.append(p)
    return np.array(pts)


def rest_to_rest_wrenches(rng, mass, inertia, accel=1.0, alpha=1.0):
    """Accelerate/decelerate wrench pair for a translation along d and a rotation about e."""
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    e = rng.normal(size=3); e /= np.linalg.norm(e)
    w = np.concatenate([mass * accel * d, inertia @ (alpha * e)])
    return np.stack([w, -w])


class GaugeSolver:
    """Penalized form of eq:gauge, built once and re-solved with an occupancy mask."""

    def __init__(self, pts, nu_deg, W, rho=1e3, big=1e3):
        self.P, self.K = len(pts), len(W)
        self.rho = rho
        self.inaccurate = 0
        self.mask = cp.Parameter(self.P, nonneg=True)
        U = [cp.Variable((self.P, 3)) for _ in range(self.K)]
        S = [cp.Variable(6) for _ in range(self.K)]
        G = [np.vstack([np.eye(3), skew(p)]) for p in pts]
        cosn = np.cos(np.deg2rad(nu_deg))
        cons, fuel = [], 0
        for k in range(self.K):
            for i, p in enumerate(pts):
                u = U[k][i]
                cons += [cosn * np.linalg.norm(p) * cp.norm(u) <= -p @ u,
                         cp.norm(u) <= big * self.mask[i]]
                fuel += cp.norm(u)
            cons += [sum(G[i] @ U[k][i] for i in range(self.P)) + S[k] == W[k]]
        self.fuel_expr = fuel
        self.res_expr = sum(cp.norm1(s) for s in S)
        self.prob = cp.Problem(cp.Minimize(fuel + rho * self.res_expr), cons)

    def solve(self, subset_bits):
        self.mask.value = np.array([(subset_bits >> i) & 1 for i in range(self.P)], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prob.solve(solver=cp.CLARABEL)
        self.inaccurate += self.prob.status != cp.OPTIMAL
        fuel, res = float(self.fuel_expr.value), float(self.res_expr.value)
        exact = fuel if res < FEAS_TOL else INF
        return exact, fuel + self.rho * res


class RatioSolver:
    """Authority ratio lambda(A) = max{t : t*w_k producible for all k} under thrust limit u_max."""

    def __init__(self, pts, nu_deg, W, u_max):
        self.P = len(pts)
        self.inaccurate = 0
        self.mask = cp.Parameter(self.P, nonneg=True)
        t = cp.Variable(nonneg=True)
        G = [np.vstack([np.eye(3), skew(p)]) for p in pts]
        cosn = np.cos(np.deg2rad(nu_deg))
        cons = []
        for w in W:
            U = cp.Variable((self.P, 3))
            for i, p in enumerate(pts):
                cons += [cosn * np.linalg.norm(p) * cp.norm(U[i]) <= -p @ U[i],
                         cp.norm(U[i]) <= u_max * self.mask[i]]
            cons += [sum(G[i] @ U[i] for i in range(self.P)) == t * w]
        self.t = t
        self.prob = cp.Problem(cp.Maximize(t), cons)

    def solve(self, subset_bits):
        self.mask.value = np.array([(subset_bits >> i) & 1 for i in range(self.P)], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prob.solve(solver=cp.CLARABEL)
        self.inaccurate += self.prob.status != cp.OPTIMAL
        lam = float(self.t.value)
        return lam if lam > FEAS_TOL else 0.0


def enumerate_all(solver):
    n = 1 << solver.P
    f, fpen = np.empty(n), np.empty(n)
    for A in range(n):
        f[A], fpen[A] = solver.solve(A)
    return f, fpen


def enumerate_ratio(solver):
    return np.array([solver.solve(A) for A in range(1 << solver.P)])


def popcount(x):
    return bin(x).count("1")


def analyze(f, P):
    n = 1 << P
    sizes = np.array([popcount(A) for A in range(n)])
    out = {}

    # Monotonicity: f(A + e) <= f(A).
    mono = 0
    for A in range(n):
        for e in range(P):
            if not A >> e & 1 and f[A | 1 << e] > f[A] + VAL_TOL:
                mono += 1
    out["monotonicity_violations"] = mono

    # Diminishing returns: delta(e|A) >= delta(e|A+b), delta(e|A) = f(A) - f(A+e).
    # "significant" violations exceed SIG_REL * f(A), screening out solver noise.
    closed = dict(triples=0, violations=0, significant=0, worst_ratio=1.0,
                  worst_excess_rel=0.0)
    open_ = dict(triples=0, violations=0)
    for A in range(n):
        for e, b in itertools.permutations(range(P), 2):
            if A >> e & 1 or A >> b & 1:
                continue
            Ae, Ab, Abe = A | 1 << e, A | 1 << b, A | 1 << e | 1 << b
            if np.isfinite(f[A]):
                d_small, d_big = f[A] - f[Ae], f[Ab] - f[Abe]
                closed["triples"] += 1
                excess = d_big - d_small
                if excess > VAL_TOL:
                    closed["violations"] += 1
                    closed["worst_excess_rel"] = max(closed["worst_excess_rel"], excess / f[A])
                    if excess > SIG_REL * f[A]:
                        closed["significant"] += 1
                        closed["worst_ratio"] = min(closed["worst_ratio"], max(d_small, 0.0) / d_big)
            else:
                # Below closure, a gain is "infinite" exactly when adding e makes A feasible.
                open_["triples"] += 1
                gain_small = np.isinf(f[A]) and np.isfinite(f[Ae])
                gain_big_inf = np.isinf(f[Ab]) and np.isfinite(f[Abe])
                if gain_big_inf and not gain_small:
                    open_["violations"] += 1
    out["diminishing_returns_closed"] = closed
    out["diminishing_returns_below_closure"] = open_

    # Submodularity ratio of the reduction over closed L (Das & Kempe):
    # gamma = min_{L closed, S disjoint} sum_e delta(e|L) / delta(S|L).
    gamma = 1.0
    for L in range(n):
        if not np.isfinite(f[L]):
            continue
        comp = [e for e in range(P) if not L >> e & 1]
        single = {e: f[L] - f[L | 1 << e] for e in comp}
        for r in range(2, len(comp) + 1):
            for Sset in itertools.combinations(comp, r):
                Sbits = sum(1 << e for e in Sset)
                dS = f[L] - f[L | Sbits]
                if dS > VAL_TOL:
                    gamma = min(gamma, sum(single[e] for e in Sset) / dS)
    out["submodularity_ratio_closed"] = gamma

    # Exhaustive optimum per cardinality.
    out["min_closed_size"] = int(sizes[np.isfinite(f)].min()) if np.isfinite(f).any() else None
    out["opt_by_N"] = {int(N): float(f[sizes == N].min()) for N in range(P + 1)}
    return out


def forward_greedy(f, fpen, P, N):
    A = 0
    for _ in range(N):
        cands = [A | 1 << e for e in range(P) if not A >> e & 1]
        A = min(cands, key=lambda B: (f[B], fpen[B]))
    return A


def reverse_greedy(f, P, N):
    A = (1 << P) - 1
    for _ in range(P - N):
        cands = [A & ~(1 << e) for e in range(P) if A >> e & 1]
        A = min(cands, key=lambda B: f[B])
    return A


def greedy_report(f, fpen, P, opt):
    rows = {}
    for N in range(1, P + 1):
        o = opt[N]
        if not np.isfinite(o):
            continue
        fg = f[forward_greedy(f, fpen, P, N)]
        rg = f[reverse_greedy(f, P, N)]
        rows[N] = dict(opt=o, forward=float(fg), reverse=float(rg),
                       forward_ratio=float(fg / o), reverse_ratio=float(rg / o))
    return rows


def split_dp(best, combine, worst):
    """Exact optimum over agent splits n_1 + ... + n_M = N of separable per-payload tables."""
    P = len(best[0]) - 1
    acc = list(best[0])
    for b in best[1:]:
        new = [worst] * (len(acc) + P)
        for N, a in enumerate(acc):
            for n in range(P + 1):
                v = combine(a, b[n])
                if (worst == INF and v < new[N + n]) or (worst == -INF and v > new[N + n]):
                    new[N + n] = v
        acc = new
    return acc


def run_rule(rule, tabs, P, N):
    """Allocate N agents one at a time; tabs[j] = (f, fpen, lam) over subsets of payload j."""
    M = len(tabs)
    A = [0] * M
    for _ in range(N):
        moves = [(j, e) for j in range(M) for e in range(P) if not A[j] >> e & 1]

        def after(j, e):
            B = list(A)
            B[j] |= 1 << e
            return B

        if rule == "fuel":
            def key(m):
                B = after(*m)
                return (sum(t[0][b] for t, b in zip(tabs, B)), sum(t[1][b] for t, b in zip(tabs, B)))
            j, e = min(moves, key=key)
        elif rule == "maxmin":
            # Plain max-min of the ratio; ties go to the first candidate (lowest j, then e).
            j, e = max(moves, key=lambda m: min(t[2][b] for t, b in zip(tabs, after(*m))))
        elif rule == "bottleneck":
            # Payload with the smallest ratio, then the point maximizing its ratio;
            # while the ratio stays 0, rank points by the penalized residual.
            open_j = sorted({j for j, _ in moves}, key=lambda j: tabs[j][2][A[j]])
            j = open_j[0]
            f, fpen, lam = tabs[j]
            e = min((e for jj, e in moves if jj == j),
                    key=lambda e: (-lam[A[j] | 1 << e], fpen[A[j] | 1 << e]))
        A[j] |= 1 << e
    fuel = sum(t[0][a] for t, a in zip(tabs, A))
    lam_min = min(t[2][a] for t, a in zip(tabs, A))
    return fuel, lam_min, [popcount(a) for a in A]


def multi_payload(tabs, P):
    sizes = np.array([popcount(A) for A in range(1 << P)])
    best_f = [[t[0][sizes == n].min() for n in range(P + 1)] for t in tabs]
    best_l = [[t[2][sizes == n].max() for n in range(P + 1)] for t in tabs]
    opt_f = split_dp(best_f, lambda a, b: a + b, INF)
    opt_l = split_dp(best_l, min, -INF)
    rows = {}
    for N in range(1, len(tabs) * P + 1):
        row = dict(opt_fuel=float(opt_f[N]), opt_lam=float(opt_l[N]))
        for rule in ("fuel", "maxmin", "bottleneck"):
            fuel, lam, split = run_rule(rule, tabs, P, N)
            row[rule] = dict(fuel=float(fuel), lam=float(lam), split=split)
        rows[N] = row
    return rows


def first_N(rows, key, pred):
    for N in sorted(rows):
        if pred(rows[N] if key is None else rows[N][key]):
            return N
    return None


def summarize_multi(rows):
    out = {}
    out["opt"] = dict(N_all_capable=first_N(rows, None, lambda r: r["opt_lam"] > 0),
                      N_all_ratio_ge_1=first_N(rows, None, lambda r: r["opt_lam"] >= 1),
                      N_fuel_finite=first_N(rows, None, lambda r: np.isfinite(r["opt_fuel"])))
    for rule in ("fuel", "maxmin", "bottleneck"):
        ratios = [r[rule]["fuel"] / r["opt_fuel"] for r in rows.values()
                  if np.isfinite(r["opt_fuel"]) and np.isfinite(r[rule]["fuel"])]
        lam_gap = [r[rule]["lam"] / r["opt_lam"] for r in rows.values() if r["opt_lam"] > 0]
        out[rule] = dict(
            N_all_capable=first_N(rows, rule, lambda r: r["lam"] > 0),
            N_all_ratio_ge_1=first_N(rows, rule, lambda r: r["lam"] >= 1),
            N_fuel_finite=first_N(rows, rule, lambda r: np.isfinite(r["fuel"])),
            worst_fuel_over_opt=max(ratios) if ratios else None,
            mean_fuel_over_opt=float(np.mean(ratios)) if ratios else None,
            worst_lam_over_opt=min(lam_gap) if lam_gap else None,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nus", type=float, nargs="+", default=[20.0, 35.0, 50.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    ap.add_argument("--umax", type=float, default=0.5, help="per-agent thrust limit for the ratio")
    ap.add_argument("--skip-single", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    h = np.array([1.0, 0.75, 0.5])
    pts = box_points(h)
    P = len(pts)
    mass = 1.0
    inertia = mass / 3 * np.diag([h[1]**2 + h[2]**2, h[0]**2 + h[2]**2, h[0]**2 + h[1]**2])

    summary = []
    for nu in ([] if args.skip_single else args.nus):
        for seed in args.seeds:
            t0 = time.time()
            W = rest_to_rest_wrenches(np.random.default_rng(seed), mass, inertia)
            solver = GaugeSolver(pts, nu, W)
            f, fpen = enumerate_all(solver)
            res = analyze(f, P)
            res["inaccurate_solves"] = solver.inaccurate
            res["greedy"] = greedy_report(f, fpen, P, res["opt_by_N"])
            res.update(nu=nu, seed=seed, seconds=time.time() - t0)
            summary.append(res)
            np.save(args.out / f"f_nu{nu:g}_seed{seed}.npy", np.stack([f, fpen]))
            c = res["diminishing_returns_closed"]
            print(f"nu={nu:g} seed={seed}: closure at N={res['min_closed_size']}, "
                  f"mono viol={res['monotonicity_violations']}, "
                  f"DR viol closed={c['significant']} significant, {c['violations']} total"
                  f"/{c['triples']} (worst excess {100 * c['worst_excess_rel']:.1f}% of f, "
                  f"worst ratio {c['worst_ratio']:.3f}), "
                  f"below closure={res['diminishing_returns_below_closure']['violations']}"
                  f"/{res['diminishing_returns_below_closure']['triples']}, "
                  f"gamma={res['submodularity_ratio_closed']:.3f}, "
                  f"worst fwd/rev ratio="
                  f"{max(r['forward_ratio'] for r in res['greedy'].values()):.3f}/"
                  f"{max(r['reverse_ratio'] for r in res['greedy'].values()):.3f} "
                  f"inaccurate solves={res['inaccurate_solves']} "
                  f"[{res['seconds']:.0f}s]", flush=True)

    # Three payloads: same geometry, different masses and maneuvers, shared agent pool.
    multi = []
    for nu in args.nus:
        tabs = []
        for j, (m_scale, seed) in enumerate([(1.0, 10), (2.0, 11), (0.5, 12)]):
            W = rest_to_rest_wrenches(np.random.default_rng(seed), mass * m_scale, inertia * m_scale)
            f, fpen = enumerate_all(GaugeSolver(pts, nu, W))
            lam = enumerate_ratio(RatioSolver(pts, nu, W, args.umax))
            np.save(args.out / f"multi_nu{nu:g}_payload{j}.npy", np.stack([f, fpen, lam]))
            tabs.append((f, fpen, lam))
            print(f"  payload {j} (mass x{m_scale:g}): ratio with all {P} points = "
                  f"{lam[-1]:.2f}", flush=True)
        rows = multi_payload(tabs, P)
        s = summarize_multi(rows)
        multi.append(dict(nu=nu, umax=args.umax, rows=rows, summary=s))
        print(f"three payloads nu={nu:g} umax={args.umax:g}:", flush=True)
        for name, d in s.items():
            print(f"  {name:10s} {d}", flush=True)

    with open(args.out / "summary.json", "w") as fh:
        json.dump(dict(single=summary, multi_payload=multi), fh, indent=1, default=str)


if __name__ == "__main__":
    main()
