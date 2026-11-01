"""Cached small conflict-block AL-L-BFGS repairs, followed by ball SOCPs.

Inactive payloads stay fixed inside a local solve, but every pair touching the
active block is constrained. Full-fleet segment checks rebuild the conflict
schedule after each block. Fixed-state sensitivities are cached during a block;
only current touching-pair encounter times enrich the permanent half-step grid.
This is a heuristic block method, not a joint optimum.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from .group_tube_pairwise import _PairwiseProblem
from .pairwise_geometry import separation, _TimeBudget
from .group_tube_corridors import corridor_radii
from .translation import solve_reference, cost_value


def choose_block(check, maximum):
    """Grow from the worst conflict; cap block size even for dense components."""
    bad = np.flatnonzero(check['margins'] < -1e-6)
    if not len(bad):
        return []
    edges = check['pairs'][bad]
    weights = -check['margins'][bad]
    block = list(edges[int(np.argmax(weights))])
    while len(block) < maximum:
        scores = {}
        for (i, j), weight in zip(edges, weights):
            if (i in block) != (j in block):
                node = int(j if i in block else i)
                scores[node] = scores.get(node, 0.) + float(weight)
        if not scores:
            break
        block.append(max(scores, key=scores.get))
    return sorted(map(int, block))


def block_value(problem, y, block, local, times, lam, mu, fuel_weight=1.):
    """Only selected controls vary; pair arrays must include all touching pairs."""
    trial = y.copy()
    trial[block] = np.asarray(local).reshape(len(block), problem.size)
    cost, grad, g, penalty = problem.pairwise(trial.ravel(), times, lam, mu)
    weight = np.maximum(0., lam + mu * g)
    loss = fuel_weight * cost + np.sum(weight * weight - lam * lam) / (2 * mu)
    return float(loss), (fuel_weight * grad + penalty)[block].ravel()



class _CachedBlock:
    """Cache fixed payload positions and linear sensitivity for one local solve."""
    def __init__(self, problem, y, block, times, multipliers, mu, fuel_weight=.05):
        p = self.p = problem
        self.block = np.asarray(block,dtype=int)
        self.origin = y[self.block].copy()
        self.lam, self.mu, self.fuel_weight = multipliers,mu,fuel_weight
        k=np.minimum(np.floor(times).astype(int),p.sc.N-1)
        a=times-k
        maps=(1-a[:,None,None])*p.maps[k]+a[:,None,None]*p.maps[k+1]
        free=(1-a[None,:,None])*p.free[:,k]+a[None,:,None]*p.free[:,k+1]
        u=p.controls(y.ravel())
        pos=(free[:,:,:3]+np.einsum('tad,jd->jta',maps[:,:3],u))*p.length
        self.relative=pos[p.i]-pos[p.j]
        self.mapping=maps[:,:3]@p.z*p.length*p.scale
        radii=p.eps[p.i]+p.eps[p.j]
        requested = (p.pair_deltas[p.i,p.j,None] if getattr(p,"pair_deltas",None) is not None else p.sc.delta)
        self.required=requested+p.clearance+np.maximum(radii[:,k],radii[:,k+1])
        lookup=np.full(p.sc.M,-1,dtype=int);lookup[self.block]=np.arange(len(block))
        self.li,self.lj=lookup[p.i],lookup[p.j]
        self.mi,self.mj=self.li>=0,self.lj>=0
        self.base=p.base[self.block]
        self.force_scale=p.force_scale[self.block]
        forces=u.reshape(p.sc.M,p.sc.N,3)*p.force_scale[:,None,None]
        norms=np.sqrt(np.sum(forces*forces,axis=-1)+1e-24)
        self.outside_cost=float(norms.sum()-norms[self.block].sum())
        self.g=None

    def __call__(self,local):
        p=self.p
        local=np.asarray(local).reshape(len(self.block),p.size)
        shift=np.einsum('gz,taz->gta',local-self.origin,self.mapping)
        relative=self.relative.copy()
        relative[self.mi]+=shift[self.li[self.mi]]
        relative[self.mj]-=shift[self.lj[self.mj]]
        distance=np.linalg.norm(relative,axis=-1)
        g=1-distance/self.required
        weight=np.maximum(0,self.lam+self.mu*g)
        vector=-weight[...,None]*relative/np.maximum(distance[...,None],1e-12)/self.required[...,None]
        gp=np.zeros_like(shift)
        np.add.at(gp,self.li[self.mi],vector[self.mi])
        np.add.at(gp,self.lj[self.mj],-vector[self.mj])
        gradient=np.einsum('gta,taz->gz',gp,self.mapping)
        u=self.base+p.scale*local@p.z.T
        forces=u.reshape(len(self.block),p.sc.N,3)*self.force_scale[:,None,None]
        norm=np.sqrt(np.sum(forces*forces,axis=-1)+1e-24)
        fuel_grad=(forces/norm[...,None]*self.force_scale[:,None,None]).reshape(len(self.block),-1)
        gradient+=self.fuel_weight*p.scale*fuel_grad@p.z/p.j0
        cost=(norm.sum()+self.outside_cost)/p.j0
        self.g=g
        loss=self.fuel_weight*cost+np.sum(weight*weight-self.lam*self.lam)/(2*self.mu)
        return float(loss),gradient.ravel()


def solve_reference_set(sc, refs, psis, budget_s=60., group_size=4,
                        inner_iterations=30, sweeps=16, clearance=.1,
                        corridor_cap=150., seed=0, pair_deltas=None, repair_only=False, fuel_weight=.05):
    """Spend at most 75% of budget on repair; retain safe centers on tube failure.

    fuel_weight balances normalized fuel against the AL collision penalty;
    it does not change physical clearances or acceptance checks. Default .05.
    pair_deltas optionally supplies the benchmark clearance matrix. With
    repair_only=True, use the full budget for repair and skip adapted ball solves.

    The returned trajectories always satisfy the native dynamics/endpoints by
    construction. Acceptance additionally requires a full-fleet segment check.
    Tube solves are independent after the coupled small-block repair stage.
    """
    if group_size < 2 or inner_iterations < 1 or sweeps < 1 or clearance < 0 or budget_s <= 0:
        raise ValueError('invalid conflict-block options')
    if not np.isfinite(fuel_weight) or fuel_weight < 0:
        raise ValueError('fuel_weight must be finite and nonnegative')
    start = time.perf_counter()
    deadline = start + budget_s
    repair_deadline = deadline if repair_only else start + .75 * budget_s
    physical_delta = sc.delta if pair_deltas is None else np.asarray(pair_deltas,dtype=float)
    if pair_deltas is not None:
        if physical_delta.shape != (sc.M,sc.M) or not np.isfinite(physical_delta).all() or not np.allclose(physical_delta,physical_delta.T):
            raise ValueError("pair_deltas must be a finite symmetric (M,M) matrix")
        if np.any(physical_delta[np.triu_indices(sc.M,1)] <= 0):
            raise ValueError("off-diagonal clearances must be positive")
        if not repair_only:
            raise ValueError("pair_deltas requires repair_only; use a matching external tube solver")
    p = _PairwiseProblem(sc, refs, psis, [np.zeros(sc.N+1) for _ in refs], 'fuel')
    p.clearance = clearance
    p.pair_deltas = None if pair_deltas is None else physical_delta
    all_i, all_j = p.i.copy(), p.j.copy()
    y = np.zeros((sc.M, p.size))
    baseline = p.references(y.ravel())
    history, best, status = [], None, 'repair_iteration_limit'
    check = separation(baseline, physical_delta + clearance)
    if check['ok']:
        best = baseline
    else:
        fixed = separation([dict(r=r['r'][:2]) for r in baseline], physical_delta + clearance)
        endpoints = np.array([r['r'][-1] for r in baseline])
        end_need = sc.delta if pair_deltas is None else physical_delta[all_i,all_j]
        if not fixed['ok'] or np.any(np.linalg.norm(endpoints[all_i]-endpoints[all_j],axis=1) < end_need+clearance-1e-6):
            return baseline, [], dict(ok=False, status='immutable_clearance_conflict', method='conflict_lbfgs_ball')
        y += np.random.default_rng(seed).normal(0, 1e-3, y.shape)
    times = np.arange(1., sc.N + .01, .5)
    lam = np.zeros((len(all_i),len(times)))
    mu = 10.
    local_count = 0
    try:
        for sweep in range(sweeps):
            if best is not None:
                break
            # Several overlapping blocks per sweep; schedule follows current conflicts.
            for _ in range(max(1, sc.M)):
                if time.perf_counter() >= repair_deadline:
                    raise _TimeBudget()
                current = p.references(y.ravel())
                check = separation(current, physical_delta + clearance)
                # The larger target is a numerical buffer, not an extra physical requirement.
                if check['min_margin'] + clearance >= min(clearance, .001):
                    best, status = current, 'safe_block_repair'
                    break
                block = choose_block(check, group_size)
                touched = np.isin(all_i,block) | np.isin(all_j,block)
                # Keep permanent grid multipliers; enrich only this block with current
                # closest-approach times. Old unrelated times do not accumulate.
                local_times = np.unique(np.r_[times,check['times'][touched & (check['margins'] < 0)]])
                grid_columns = np.searchsorted(local_times,times)
                local_lam = np.zeros((int(touched.sum()),len(local_times)))
                local_lam[:,grid_columns] = lam[touched]
                p.i, p.j = all_i[touched], all_j[touched]
                cached = _CachedBlock(p,y,block,local_times,local_lam,mu,fuel_weight)
                last = {'x':y[block].ravel().copy()}
                def evaluate(local):
                    if time.perf_counter() >= repair_deadline:
                        raise _TimeBudget()
                    value = cached(local)
                    return value
                def accepted(local):
                    last['x'] = local.copy()
                try:
                    res = minimize(evaluate,y[block].ravel(),jac=True,method='L-BFGS-B',callback=accepted,
                                   options=dict(maxiter=inner_iterations,maxls=20,maxcor=8,ftol=1e-10,gtol=1e-5))
                    y[block] = res.x.reshape(len(block),p.size)
                except _TimeBudget:
                    y[block] = last['x'].reshape(len(block),p.size)
                    raise
                finally:
                    p.i,p.j = all_i,all_j
                # Update multipliers of affected pairs after this block.
                cached(y[block].ravel())
                lam[touched] = np.maximum(0,lam[touched]+mu*cached.g[:,grid_columns])
                local_count += 1
                after = separation(p.references(y.ravel()),physical_delta+clearance)
                history.append(dict(block=block,sweep=sweep,sample_times=len(local_times),mu=mu,seconds=time.perf_counter()-start,
                                    conflicts=int(np.sum(after['margins'] < -1e-6)),min_margin=after['min_margin']))
            mu = min(mu*4,1e10)
    except _TimeBudget:
        status = 'repair_time_budget'
    centers = p.references(y.ravel()) if best is None else best
    if separation(centers,physical_delta)['ok']:
        best = centers
    repair_seconds = time.perf_counter()-start
    info = dict(method='conflict_lbfgs_ball',ok=best is not None,status=status,
                joint_optimization=False,uses_ball_corridors=not repair_only,group_size=group_size,
                local_solves=local_count,iterations=local_count,repair_seconds=repair_seconds,fuel_weight=float(fuel_weight),
                history=history)
    if best is None:
        info.update(seconds=time.perf_counter()-start,tube_seconds=0.)
        return centers,[],info
    if repair_only:
        info.update(status='safe_block_repair',uses_ball_corridors=False,
                    seconds=time.perf_counter()-start,tube_seconds=0.,
                    center_fuel=sum(cost_value(r['F'],'fuel') for r in best))
        return best,[],info
    eps = corridor_radii(sc,best,corridor_cap)
    tubes=[]
    tube_start=time.perf_counter()
    for j in range(sc.M):
        if time.perf_counter() >= deadline:
            break
        ref=solve_reference(sc,j,psis=psis,objective='fuel',corridor=dict(rbar=best[j]['r'],eps=eps[j]))
        if not ref['ok']:
            break
        ref['corridor_rbar']=best[j]['r']
        tubes.append(ref)
    if len(tubes)==sc.M and separation(tubes,sc.delta)['ok'] and sum(cost_value(r['F'],'fuel') for r in tubes) <= sum(cost_value(r['F'],'fuel') for r in best)+1e-6:
        selected,status=tubes,'safe_block_tubes'
    else:
        selected,status=best,'safe_block_centers_fallback'
    info.update(status=status,ok=True,tube_seconds=time.perf_counter()-tube_start,
                seconds=time.perf_counter()-start,center_fuel=sum(cost_value(r['F'],'fuel') for r in best))
    return selected,eps,info
