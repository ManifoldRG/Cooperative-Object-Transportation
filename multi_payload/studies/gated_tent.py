"""Mass-weighted gated tent with node and optional segment deconfliction.

The node merit is the weighted squared clearance deficit in the pinned-string
energy metric E(D) = sum_j m_j sum_k |D_j(k+1)-D_j(k)|^2.

With physical deltas supplied, gated/jacobi modes also penalize exact minimum
separation on every linear segment using FIXED midpoint-Green weights. A segment
push uses the Green representer of barycentric evaluation at closest approach,
normalized by its actual energy norm; this reduces to a node tent at a movable
endpoint. After each trial the closest time and combined merit are recomputed.

Only sufficient-decrease steps are accepted. A gradient fallback differentiates
both node and segment penalties. Exact crossings use a deterministic sideways
proposal and still require actual merit decrease. Success requires node targets
and physical segment clearance within tolerance plus an exact segment certificate.
No unconditional convergence theorem is asserted for these nonconvex constraints.

Raw defaults to the original node-only sweep. segment_aware=False explicitly
selects the old node-only merit for ablations; certify() then only reports safety.
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


def _phi(X, PI, PJ, TAU, W, N, SG=None):
    d = np.linalg.norm(X[PI, 2:N] - X[PJ, 2:N], axis=2)
    v = np.maximum(TAU - d, 0.0)
    node_phi = float(np.sum(W * v * v))
    return node_phi + (0.0 if SG is None else _segment_state(X, PI, PJ, SG)[0]), float(v.max(initial=0.0))


def _energy(D, m):
    """E(D) = sum_j m_j sum_k |D_j(k+1) - D_j(k)|^2."""
    return float(np.sum(m[:, None, None] * np.diff(D, axis=1) ** 2))


def _sweep(X, pairs, TAU, B, m, N, omega, FB, SG=None):
    """Sequential node pushes followed by the worst segment push per pair.

    Returns the number of segment push proposals, including rejected trials.
    """
    segment_pushes = 0
    for i, (j, l) in enumerate(pairs):
        d = X[j, 2:N] - X[l, 2:N]
        dist = np.linalg.norm(d, axis=1)
        need = TAU[i] - dist
        kw = int(np.argmax(need))
        wn = float(need[kw])
        if wn > 0.0:
            g = d[kw] / dist[kw] if dist[kw] > 0.0 else FB[i, kw]
            push = np.outer(B[kw + 2], omega * wn * g)
            total = m[j] + m[l]
            X[j] += m[l] / total * push
            X[l] -= m[j] / total * push
        if SG is not None:
            segment_pushes += _segment_push(X, j, l, i, m, SG, omega)
    return segment_pushes


def _grad_dir(X, M, N, B, m, PI, PJ, TAU, FB, SG=None):
    """Minus half the combined energy-metric gradient away from kinks."""
    d = X[PI, 2:N] - X[PJ, 2:N]
    dist = np.linalg.norm(d, axis=2)
    v = np.maximum(TAU - dist, 0.0)
    g = np.where((dist > 0.0)[..., None], d / np.maximum(dist, np.finfo(float).tiny)[..., None], FB)
    lj = (m[PJ] / (m[PI] + m[PJ]))[:, None, None]
    ll = (m[PI] / (m[PI] + m[PJ]))[:, None, None]
    C = np.zeros((M, N + 1, 3))
    np.add.at(C, (PI[:, None], np.arange(2, N)[None, :]), lj * v[..., None] * g)
    np.add.at(C, (PJ[:, None], np.arange(2, N)[None, :]), -ll * v[..., None] * g)
    D = np.einsum("kq,mkc->mqc", B, C)
    return D if SG is None else D + _segment_grad_dir(X, m, PI, PJ, SG)


def _armijo(X, D, phi, m, PI, PJ, TAU, W, N, gamma0=1.0, c=1e-4, SG=None):
    """Phi(X + gamma D) <= Phi - 2 c gamma E(D) (D = -grad/2, so the
    directional derivative is -2 E(D)). Returns (Xn, phi, worst, gamma, #halvings)
    or None if the finite numerical search fails (not a stationarity proof)."""
    dE = _energy(D, m)
    if not np.isfinite(dE) or dE <= 0.0:
        return None
    gamma, h = gamma0, 0
    while gamma >= 1e-8:
        Xn = X + gamma * D
        new, nw = _phi(Xn, PI, PJ, TAU, W, N, SG)
        if new <= phi - 2.0 * c * gamma * dE:
            return Xn, new, nw, gamma, h
        gamma *= 0.5
        h += 1
    return None



def _segment_geometry(d):
    """Exact closest approach on each relative piecewise-linear segment."""
    a, e = d[..., :-1, :], np.diff(d, axis=-2)
    ee = np.sum(e * e, axis=-1)
    alpha = np.divide(-np.sum(a * e, axis=-1), ee,
                      out=np.zeros_like(ee), where=ee > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = a + alpha[..., None] * e
    dist = np.linalg.norm(closest, axis=-1)
    return dist, alpha, closest, e


def _segment_normals(closest, e):
    """Radial normals, or a deterministic sideways direction at a crossing.

    The sideways guard avoids a longitudinal push that merely moves the
    collision time. At zero-length coincident segments it uses the fixed axis.
    """
    dist = np.linalg.norm(closest, axis=-1)
    length = np.linalg.norm(e, axis=-1)
    tangent = np.divide(e, length[..., None], out=np.zeros_like(e),
                        where=length[..., None] > 0.0)
    axis = np.eye(3)[np.argmin(np.abs(tangent), axis=-1)]
    sideways = np.cross(tangent, axis)
    sn = np.linalg.norm(sideways, axis=-1)
    sideways = np.divide(sideways, sn[..., None], out=np.zeros_like(sideways),
                         where=sn[..., None] > 0.0)
    sideways = np.where((length > 0.0)[..., None], sideways, _AXIS)
    threshold = 32.0 * np.finfo(float).eps * np.maximum(1.0, length)
    return np.where((dist > threshold)[..., None],
                    closest / np.maximum(dist[..., None], np.finfo(float).tiny),
                    sideways)


def _segment_setup(X, PI, PJ, m, B, deltas, buffer):
    """Fixed merit weights; barycentric Green profiles for segment pushes.

    The merit weight uses the MIDPOINT Green value, fixed throughout the run.
    Using a weight that changes with closest time would require extra gradient
    terms. A correction at alpha instead uses its actual Green norm q(alpha).
    """
    N = X.shape[1] - 1
    k = np.arange(N + 1, dtype=float)
    gkk = np.zeros(N + 1)
    gkk[2:N] = (k[2:N] - 1) * (N - k[2:N]) / (N - 1)
    G = gkk[:, None] * B
    qmid = 0.25 * (np.diag(G)[:-1] + 2 * np.diag(G, 1) + np.diag(G)[1:])
    dl = np.asarray(deltas, float)[PI, PJ]
    if not np.all(np.isfinite(dl)) or np.any(dl < 0):
        raise ValueError("physical pair clearances must be finite and nonnegative")
    target = np.broadcast_to((dl + buffer)[:, None], (len(PI), N)).copy()
    # Fixed endpoints cannot acquire a buffer beyond their existing clearance.
    for s in range(N):
        for endpoint in (s, s + 1):
            if endpoint in (0, 1, N):
                pinned_d = np.linalg.norm(X[PI, endpoint] - X[PJ, endpoint], axis=1)
                target[:, s] = np.minimum(target[:, s], np.maximum(dl, pinned_d))
    # Segment 0 joins two pinned nodes: require physical safety, not a buffer
    # which no correction can supply at an interior closest approach.
    target[:, 0] = dl
    invmass = 1.0 / m[PI] + 1.0 / m[PJ]
    weight = 1.0 / (invmass[:, None] * np.where(qmid > 0.0, qmid, 1.0))
    return dict(G=G, target=target, weight=weight, delta=dl)


def _segment_state(X, PI, PJ, SG):
    dist, alpha, closest, e = _segment_geometry(X[PI] - X[PJ])
    violation = np.maximum(SG['target'] - dist, 0.0)
    phi = float(np.sum(SG['weight'] * violation ** 2))
    return phi, float(violation.max(initial=0.0)), dist, alpha, closest, e


def _segment_push(X, j, l, i, m, SG, omega):
    dist, alpha, closest, e = _segment_geometry(X[j] - X[l])
    need = SG['target'][i] - dist
    s = int(np.argmax(need))
    if need[s] <= 0.0:
        return 0
    a, b = 1.0 - alpha[s], alpha[s]
    G = SG['G']
    q = a * a * G[s, s] + 2 * a * b * G[s, s + 1] + b * b * G[s + 1, s + 1]
    if q <= 0.0:  # Entire constraint pinned; diagnosed before iteration.
        return 0
    profile = (a * G[s] + b * G[s + 1]) / q
    normal = _segment_normals(closest[s], e[s])
    push = np.outer(profile, omega * need[s] * normal)
    total = m[j] + m[l]
    X[j] += m[l] / total * push
    X[l] -= m[j] / total * push
    return 1


def _segment_grad_dir(X, m, PI, PJ, SG):
    """Minus half the E-gradient of the fixed-weight segment merit.

    Danskin's derivative applies at a unique nonzero closest approach. At an
    exact crossing, the direction is a guarded proposal; the gate still checks
    actual merit decrease. No smooth convergence claim is made at that kink.
    """
    _, _, dist, alpha, closest, e = _segment_state(X, PI, PJ, SG)
    v = np.maximum(SG['target'] - dist, 0.0)
    force = (SG['weight'] * v)[..., None] * _segment_normals(closest, e)
    C = np.zeros_like(X)
    ss = np.arange(X.shape[1] - 1)[None, :]
    for indices, sign in ((PI, 1.0), (PJ, -1.0)):
        f = sign * force / m[indices, None, None]
        np.add.at(C, (indices[:, None], ss), (1.0 - alpha)[..., None] * f)
        np.add.at(C, (indices[:, None], ss + 1), alpha[..., None] * f)
    return np.einsum('kq,mkc->mqc', SG['G'], C)


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
               deltas=None, segment_aware=None, segment_buffer=0.1,
               segment_tol=1e-6, window=5):
    """Deconflict nodes and, when enabled, complete linear segments.

    segment_aware defaults to True for gated/jacobi when deltas is supplied.
    Set False to reproduce the node-only merit/sweeps. Raw stays node-only by
    default. Steps 0, 1, N are pinned. segment_buffer is in metres, capped at
    pinned endpoint clearance. The gate uses fixed node and segment weights.
    Gated sweeps satisfy Phi(next) <= max(last window accepted Phi) - sigma*E.
    window=5 permits temporary score increases; window=1 restores the monotone
    gate. Rejected trials never enter the history. The gradient fallback keeps
    its stronger monotone Armijo test. Neither rule guarantees collision-free
    convergence for arbitrary inputs.

    Returns X only if node tolerance and physical segment clearance pass. The
    buffer steers proposals beyond clearance, but is not an additional success
    requirement. A finite
    search failure is reported as stalled, not as a stationarity certificate.
    This implementation does not assert global collision-free convergence.
    """
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError('window must be a positive integer')
    if variant not in ("raw", "gated", "jacobi"):
        raise ValueError(f"unknown variant {variant!r}")
    if segment_aware is None:
        segment_aware = deltas is not None and variant != 'raw'
    if segment_aware and deltas is None:
        raise ValueError('segment-aware deconfliction requires physical deltas')
    if not 0 < sigma < 1 or not np.isfinite(sigma):
        raise ValueError('sigma must be finite and between zero and one')
    if not np.isfinite(segment_buffer) or segment_buffer < 0:
        raise ValueError('segment_buffer must be finite and nonnegative')
    if not np.isfinite(segment_tol) or not 0 < segment_tol <= 1e-6:
        raise ValueError('segment_tol must be positive and at most 1e-6 metres')
    if not np.isfinite(tol) or tol <= 0 or max_sweeps < 0 or max_halvings < 0:
        raise ValueError('require positive node tol and nonnegative iteration limits')
    if not np.isfinite(budget_s) or budget_s <= 0:
        raise ValueError('budget_s must be positive and finite')
    rbar = np.asarray(rbar, float)
    if rbar.ndim != 3 or rbar.shape[1] < 4 or rbar.shape[2] != 3 or not np.all(np.isfinite(rbar)):
        raise ValueError('rbar must be finite with shape (M, N+1, 3), N >= 3')
    pairs = list(pairs)
    X, M, N, B, m, PI, PJ, TAU, W, FB = _setup(rbar, pairs, tgt, masses)
    if not np.all(np.isfinite(TAU)) or np.any(TAU < 0):
        raise ValueError('node targets must be finite and nonnegative')
    SG = _segment_setup(X, PI, PJ, m, B, deltas, segment_buffer) if segment_aware else None
    t0 = time.perf_counter()
    phi, worst = _phi(X, PI, PJ, TAU, W, N, SG)
    coincident = int(np.sum(np.linalg.norm(X[PI, 2:N] - X[PJ, 2:N], axis=2) <= _COINCIDE))
    hist = [phi]
    envelopes = [phi]
    rises = rejected = halvings = grad_steps = segment_pushes = 0
    gamma = 1.0
    status, sweeps = 'max sweeps', 0
    pinned_bad = False
    if SG is not None:
        distances = _segment_state(X, PI, PJ, SG)[2]
        pinned = np.linalg.norm(X[PI][:, [0, 1, N]] - X[PJ][:, [0, 1, N]], axis=2)
        pinned_bad = (np.any(distances[:, 0] < SG['delta'] - segment_tol)
                      or np.any(pinned < SG['delta'][:, None] - segment_tol))

    def done():
        return worst < tol and (SG is None or np.all(
            _segment_state(X, PI, PJ, SG)[2] >= SG['delta'][:, None] - segment_tol))

    if pinned_bad:
        status = 'infeasible pinned clearance'
    else:
        while sweeps < max_sweeps:
            if done():
                status = 'converged'
                break
            if time.perf_counter() - t0 > budget_s:
                status = 'time budget'
                break
            reference = max(hist[-window:])
            previous_phi = phi
            if variant == 'raw':
                segment_pushes += _sweep(X, pairs, TAU, B, m, N, 1.0, FB, SG)
                new, worst = _phi(X, PI, PJ, TAU, W, N, SG)
                phi = new
            elif variant == 'gated':
                X0 = X.copy()
                omega, ok, rejected_full = 1.0, False, False
                for h in range(max_halvings + 1):
                    segment_pushes += _sweep(X, pairs, TAU, B, m, N, omega, FB, SG)
                    new, nw = _phi(X, PI, PJ, TAU, W, N, SG)
                    step_energy = _energy(X - X0, m)
                    if step_energy > 0.0 and new <= reference - sigma * step_energy:
                        ok = True
                        break
                    if h == 0:
                        rejected_full = True
                    X[:] = X0
                    if h < max_halvings:
                        omega *= 0.5
                        halvings += 1
                rejected += int(rejected_full)
                if not ok:
                    D = _grad_dir(X, M, N, B, m, PI, PJ, TAU, FB, SG)
                    step = _armijo(X, D, phi, m, PI, PJ, TAU, W, N, c=sigma, SG=SG)
                    if step is None:
                        status = 'stalled (no accepted sweep or gradient step)'
                        break
                    X, new, nw, _, hb = step
                    halvings += hb
                    grad_steps += 1
                phi, worst = new, nw
            else:
                gamma = min(1.0, 2.0 * gamma)
                D = _grad_dir(X, M, N, B, m, PI, PJ, TAU, FB, SG)
                step = _armijo(X, D, phi, m, PI, PJ, TAU, W, N,
                               gamma0=gamma, c=sigma, SG=SG)
                if step is None:
                    status = 'stalled (no accepted gradient step)'
                    break
                X, phi, worst, gamma, hb = step
                halvings += hb
                grad_steps += 1
            sweeps += 1
            rises += int(phi > previous_phi + 1e-9)
            hist.append(phi)
            envelopes.append(max(hist[-window:]))
        if status == 'max sweeps' and done():
            status = 'converged'
    cert = certify(X, pairs, deltas) if deltas is not None else {}
    if status == 'converged' and SG is not None and not cert['seg_certified']:
        status = 'segment certification failed'
    node_phi = _phi(X, PI, PJ, TAU, W, N)[0]
    seg_phi, seg_target_worst, seg_worst = 0.0, None, None
    if SG is not None:
        seg_phi, seg_target_worst, dist, _, _, _ = _segment_state(X, PI, PJ, SG)
        seg_worst = float(np.maximum(SG['delta'][:, None] - dist, 0.0).max(initial=0.0))
    info = dict(variant=variant, sweeps=sweeps, status=status, phi0=hist[0],
                phi_final=phi, worst=worst, t=round(time.perf_counter() - t0, 3),
                phi_rises=rises, rejected_sweeps=rejected, halvings=halvings,
                grad_steps=grad_steps, coincident=coincident, sigma=sigma,
                phi_monotone=bool(np.all(np.diff(hist) <= 1e-9)),
                window=int(window) if variant == 'gated' else None,
                gate_policy=('monotone' if window == 1 else 'window-max') if variant == 'gated' else None,
                envelope_final=envelopes[-1] if variant == 'gated' else None,
                envelope_monotone=bool(np.all(np.diff(envelopes) <= 1e-9)) if variant == 'gated' else None,
                segment_aware=bool(segment_aware), merit_version='node+segment-v1' if SG is not None else 'node-v1',
                node_phi_final=node_phi, segment_phi_final=seg_phi,
                segment_worst=seg_worst, segment_target_worst=seg_target_worst,
                segment_buffer=segment_buffer if SG is not None else None,
                segment_push_trials=segment_pushes, **cert)
    return (X if status == 'converged' else None), info
