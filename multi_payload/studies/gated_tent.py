"""Gated tent: mass-weighted tent deconfliction with a monotone merit function.

Jowett's push (benchmark_inflation_vs_groupnlp.relax_tent_experimental, with
masses) for pair (j, l) at its worst step k* is the exact projection onto the
constraint set C = {x : |x_j(k*) - x_l(k*)| >= tau} in the metric

    E(D) = sum_j m_j sum_k |D_j(k+1) - D_j(k)|^2,   D = 0 at steps 0, 1, N,

because the tent B_{k*}(k) = G(k, k*) / G(k*, k*) is the Green's function of
the pinned discrete string, G(k, k') = (k-1)(N-k')/(N-1) for k <= k', and the
split m_l/(m_j+m_l) is the inverse-mass weighting. The E-distance of x to C is
v / sqrt((1/m_j + 1/m_l) G(k*, k*)), v = max(0, tau - d), so the merit function

    Phi(x) = sum_{pairs, free k} max(0, tau - d_k)^2 / ((1/m_j + 1/m_l) G(k, k))

is the sum of squared E-distances to all constraint sets. Variants:

  raw    -- Jowett's Gauss-Seidel sweeps, unchanged (Phi only logged);
  gated  -- the same sweeps, accepted only under SUFFICIENT decrease
                Phi(x+) <= Phi(x) - sigma * E(x+ - x);
            otherwise the sweep is redone with every push scaled by 1/2, 1/4,
            ... (max_halvings times); if that also fails, one gradient step
            on Phi in the metric E (the Jacobi sum of all projections, which
            is -grad_E Phi / 2) with Armijo backtracking; if that fails too,
            the point is stationary for Phi and the run stops and says so;
  jacobi -- gradient steps only (parallel / averaged projections).

Coincidence guard: when two payloads coincide at a step the push direction is
undefined (Phi has a kink there). Every unit vector gives an exact projection
onto the ball exterior, so a deterministic one is chosen (the nominal relative
direction at that step, else a fixed axis); the gradient step uses the same
choice, so the iteration is well defined everywhere.

Certificate: on convergence every free step clears tau - tol. Between steps
the relative path is a straight segment, and for a segment whose end points
are >= rho apart with relative step length ell, the closest approach is
>= sqrt(rho^2 - ell^2/4) (step-length lemma). certify() reports that bound and
the exact segment closest approach against the physical clearances; it is the
segment-safety certificate that seg_tubes() then preserves.
"""
from __future__ import annotations

import time

import numpy as np

_COINCIDE = 1e-6                                   # m: below this, direction undefined
_AXIS = np.array([0.5773502691896258, 0.5773502691896258, 0.5773502691896258])


def _setup(rbar, pairs, tgt, masses):
    X = np.array(rbar, float)
    M, N = X.shape[0], X.shape[1] - 1
    ks = np.arange(N + 1, dtype=float)
    B = np.zeros((N + 1, N + 1))
    for k in range(2, N):
        w = np.clip(np.minimum((ks - 1) / (k - 1), (N - ks) / (N - k)), 0.0, None)
        w[0] = w[1] = w[N] = 0.0
        B[k] = w
    Gkk = np.zeros(N + 1)
    Gkk[2:N] = (ks[2:N] - 1) * (N - ks[2:N]) / (N - 1)
    m = np.ones(M) if masses is None else np.asarray(masses, float)
    if m.shape != (M,) or not np.all(np.isfinite(m)) or np.any(m <= 0):
        raise ValueError("masses must contain one positive finite mass per payload")
    PI = np.array([p[0] for p in pairs], int)
    PJ = np.array([p[1] for p in pairs], int)
    TAU = np.empty((len(pairs), N - 2))
    for i, p in enumerate(pairs):
        t = np.atleast_1d(np.asarray(tgt[p], float))
        TAU[i] = t[0] if t.size == 1 else (t if t.size == N - 2 else t[2:N])
    W = 1.0 / ((1.0 / m[PI] + 1.0 / m[PJ])[:, None] * Gkk[None, 2:N])
    # fallback directions for coincident points: nominal relative direction
    d0 = X[PI, 2:N] - X[PJ, 2:N]
    n0 = np.linalg.norm(d0, axis=2, keepdims=True)
    FB = np.where(n0 > _COINCIDE, d0 / np.maximum(n0, _COINCIDE), _AXIS)
    return X, M, N, B, m, PI, PJ, TAU, W, FB


def _phi(X, PI, PJ, TAU, W, N):
    d = np.linalg.norm(X[PI, 2:N] - X[PJ, 2:N], axis=2)
    v = np.maximum(TAU - d, 0.0)
    return float(np.sum(W * v * v)), float(v.max(initial=0.0))


def _energy(D, m):
    """E(D) = sum_j m_j sum_k |D_j(k+1) - D_j(k)|^2."""
    return float(np.sum(m[:, None, None] * np.diff(D, axis=1) ** 2))


def _sweep(X, pairs, TAU, B, m, N, omega, FB):
    """One Jowett sweep (in place), every push scaled by omega."""
    worst = 0.0
    for i, (j, l) in enumerate(pairs):
        d = X[j, 2:N] - X[l, 2:N]
        dist = np.linalg.norm(d, axis=1)
        need = TAU[i] - dist
        kw = int(np.argmax(need))
        wn = float(need[kw])
        if wn <= 0:
            continue
        worst = max(worst, wn)
        g = d[kw] / dist[kw] if dist[kw] > _COINCIDE else FB[i, kw]
        push = np.outer(B[kw + 2], omega * wn * g)
        tot = m[j] + m[l]
        X[j] += (m[l] / tot) * push
        X[l] -= (m[j] / tot) * push
    return worst


def _grad_dir(X, M, N, B, m, PI, PJ, TAU, FB):
    """Sum of all projections (Jacobi) = -grad_E Phi / 2."""
    d = X[PI, 2:N] - X[PJ, 2:N]
    dist = np.linalg.norm(d, axis=2)
    v = np.maximum(TAU - dist, 0.0)
    g = np.where((dist > _COINCIDE)[..., None], d / np.maximum(dist, _COINCIDE)[..., None], FB)
    lj = (m[PJ] / (m[PI] + m[PJ]))[:, None, None]
    ll = (m[PI] / (m[PI] + m[PJ]))[:, None, None]
    C = np.zeros((M, N + 1, 3))
    np.add.at(C, (PI[:, None], np.arange(2, N)[None, :]), lj * v[..., None] * g)
    np.add.at(C, (PJ[:, None], np.arange(2, N)[None, :]), -ll * v[..., None] * g)
    return np.einsum("kq,mkc->mqc", B, C)


def _armijo(X, D, phi, m, PI, PJ, TAU, W, N, gamma0=1.0, c=1e-4):
    """Phi(X + gamma D) <= Phi - 2 c gamma E(D) (D = -grad/2, so the
    directional derivative is -2 E(D)). Returns (Xn, phi, worst, gamma, #halvings)
    or None if no step >= 1e-8 works (stationary)."""
    dE = _energy(D, m)
    if dE < 1e-12:
        return None
    gamma, h = gamma0, 0
    while gamma >= 1e-8:
        Xn = X + gamma * D
        new, nw = _phi(Xn, PI, PJ, TAU, W, N)
        if new <= phi - 2.0 * c * gamma * dE:
            return Xn, new, nw, gamma, h
        gamma *= 0.5
        h += 1
    return None


def certify(X, pairs, deltas, tol=1e-6):
    """Segment-safety certificate for piecewise-linear trajectories X (M, N+1, 3)
    against physical clearances deltas[j][l]. Returns
      seg_margin    -- exact min over pairs and segments of closest approach - delta,
      lemma_margin  -- min of sqrt(rho^2 - ell^2/4) - delta (step-length lemma,
                       rho = smaller end-point distance, ell = relative step),
      seg_bad       -- #segments below delta (exact), seg_certified (bool)."""
    X = np.asarray(X, float)
    seg, lem, bad = np.inf, np.inf, 0
    for (j, l) in pairs:
        d = X[j] - X[l]
        d0, e = d[:-1], np.diff(d, axis=0)
        ee = np.sum(e * e, axis=1)
        t = np.where(ee > 1e-12, np.clip(-np.sum(d0 * e, axis=1) / np.maximum(ee, 1e-12),
                                         0.0, 1.0), 0.0)
        dl = float(deltas[j][l])
        mind = np.linalg.norm(d0 + t[:, None] * e, axis=1) - dl
        nd = np.linalg.norm(d, axis=1)
        rho = np.minimum(nd[:-1], nd[1:])
        bound = np.sqrt(np.maximum(rho * rho - ee / 4.0, 0.0)) - dl
        seg = min(seg, float(mind.min()))
        lem = min(lem, float(bound.min()))
        bad += int(np.sum(mind < -tol))
    return dict(seg_margin=seg, lemma_margin=lem, seg_bad=bad, seg_certified=bad == 0)


def deconflict(rbar, pairs, tgt, masses=None, variant="gated", tol=0.1,
               max_sweeps=5000, budget_s=120.0, max_halvings=8, sigma=1e-4,
               deltas=None):
    """Deconflict time-indexed waypoints rbar (M, N+1, 3); steps 0, 1, N pinned.
    tgt[(j, l)]: node target tau (scalar, or per free step 2..N-1).
    deltas: physical clearances; if given, info carries the segment certificate.

    Returns (X or None, info). X is returned only on convergence (worst node
    violation < tol). info: status ('converged' | 'stationary ...' |
    'max sweeps' | 'time budget'), sweeps, phi0/phi_final, worst, t,
    phi_rises (raw), rejected_sweeps, halvings, grad_steps (gated/jacobi),
    phi_monotone, coincident (#coincident-point pushes guarded at the start)."""
    if variant not in ("raw", "gated", "jacobi"):
        raise ValueError(f"unknown variant {variant!r}")
    X, M, N, B, m, PI, PJ, TAU, W, FB = _setup(rbar, pairs, tgt, masses)
    t0 = time.perf_counter()
    phi, worst = _phi(X, PI, PJ, TAU, W, N)
    coincident = int(np.sum(np.linalg.norm(X[PI, 2:N] - X[PJ, 2:N], axis=2) <= _COINCIDE))
    hist = [phi]
    rises = rejected = halvings = grad_steps = 0
    gamma = 1.0
    status, it = "max sweeps", 0
    for it in range(max_sweeps):
        if worst < tol:
            status = "converged"
            break
        if time.perf_counter() - t0 > budget_s:
            status = "time budget"
            break
        if variant == "raw":
            _sweep(X, pairs, TAU, B, m, N, 1.0, FB)
            new, worst = _phi(X, PI, PJ, TAU, W, N)
            rises += new > phi + 1e-9
            phi = new
        elif variant == "gated":
            X0 = X.copy()
            omega, ok, h = 1.0, False, 0
            for h in range(max_halvings + 1):
                _sweep(X, pairs, TAU, B, m, N, omega, FB)
                new, nw = _phi(X, PI, PJ, TAU, W, N)
                if new <= phi - sigma * _energy(X - X0, m):
                    ok = True
                    break
                X[:] = X0
                omega *= 0.5
                halvings += 1
            if h > 0:
                rejected += 1
            if not ok:                              # gradient fallback
                step = _armijo(X, _grad_dir(X, M, N, B, m, PI, PJ, TAU, FB), phi,
                               m, PI, PJ, TAU, W, N, c=sigma)
                if step is None:
                    status = "stationary (no sweep or gradient decrease)"
                    break
                X, new, nw, _, hb = step
                halvings += hb
                grad_steps += 1
            phi, worst = new, nw
        else:                                       # jacobi
            gamma = min(1.0, 2.0 * gamma)
            step = _armijo(X, _grad_dir(X, M, N, B, m, PI, PJ, TAU, FB), phi,
                           m, PI, PJ, TAU, W, N, gamma0=gamma, c=sigma)
            if step is None:
                status = "stationary (no gradient decrease)"
                break
            X, phi, worst, gamma, hb = step
            halvings += hb
            grad_steps += 1
        hist.append(phi)
    else:
        it = max_sweeps
        if worst < tol:
            status = "converged"
    info = dict(variant=variant, sweeps=it, status=status, phi0=hist[0],
                phi_final=phi, worst=worst, t=round(time.perf_counter() - t0, 3),
                phi_rises=rises, rejected_sweeps=rejected, halvings=halvings,
                grad_steps=grad_steps, coincident=coincident, sigma=sigma,
                phi_monotone=bool(np.all(np.diff(hist) <= 1e-9)))
    if deltas is not None:
        info.update(certify(X, pairs, deltas))
    return (X if status == "converged" else None), info
