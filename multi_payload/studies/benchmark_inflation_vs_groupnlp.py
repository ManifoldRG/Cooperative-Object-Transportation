"""Original inflation+tubes versus group L-BFGS, with and without original tubes.

All methods share scenario, nominal trajectories and pair-specific clearances.
The leader's original inflation/tube functions are called directly. Group repair
keeps its existing internal algorithm (including its stricter segment checks);
the benchmark's reported safety criterion for all three outputs is node separation.

Every check also reports the exact between-node (segment) clearance of the
piecewise-linear trajectories (seg_safe_*, seg_margin_*); --segment-check makes
segment clearance part of the safety criterion. --gated-tent replaces the tent
sweeps with the gated tent (studies/gated_tent.py: same pushes, accepted only
under sufficient decrease of the merit function, gradient fallback; converges
to a stationary point). --seg-tubes replaces the original node tubes with
segment-sized tubes (expand_method.seg_tubes), which are safe between nodes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))

import benchmark_tubes as leader
import benchmark_scale100 as planes
from multi_payload import scenario, translation, group_tube_repair, group_tube_pairwise


SEGMENT_SAFETY=False  # --segment-check: 'safe' also requires segment clearance


def segment_margins(positions, deltas):
    """Exact closest approach - delta of every pair's relative segment between
    consecutive nodes (positions are piecewise linear in time). Shape (pairs, nodes-1)."""
    count=len(positions)
    i,j=np.triu_indices(count,1)
    d=positions[i]-positions[j]
    d0,e=d[:,:-1],np.diff(d,axis=1)
    ee=np.sum(e*e,axis=-1)
    t=np.where(ee>1e-12,np.clip(-np.sum(d0*e,axis=-1)/np.maximum(ee,1e-12),0.,1.),0.)
    return np.linalg.norm(d0+t[...,None]*e,axis=-1)-deltas[i,j,None],i,j


def node_check(positions, deltas, tolerance=1e-6):
    """Original min distance over SAVED NODES only; no interpolation.
    Also reports the exact segment (between-node) clearance; it enters 'safe'
    only with --segment-check (SEGMENT_SAFETY)."""
    positions=np.asarray(positions,dtype=float)
    deltas=np.asarray(deltas,dtype=float)
    if positions.ndim!=3 or positions.shape[-1]!=3 or len(positions)<2:
        raise ValueError('positions must have shape (M,nodes,3), M>=2')
    count,nodes,_=positions.shape
    if deltas.shape!=(count,count):
        raise ValueError('deltas must have shape (M,M)')
    if not np.isfinite(positions).all() or not np.isfinite(deltas).all():
        return dict(safe=False,min_margin=None,worst_pair=None,worst_node=None,
                    node_safe=False,seg_safe=False,seg_margin=None,seg_bad=None)
    i,j=np.triu_indices(count,1)
    margins=np.linalg.norm(positions[i]-positions[j],axis=-1)-deltas[i,j,None]
    pair,node=np.unravel_index(np.argmin(margins),margins.shape)
    margin=float(margins[pair,node])
    seg,_,_=segment_margins(positions,deltas)
    node_safe=bool(margin>=-tolerance)
    seg_safe=bool(seg.min()>=-tolerance)
    return dict(safe=node_safe and (seg_safe or not SEGMENT_SAFETY),min_margin=margin,
                worst_pair=[int(i[pair]),int(j[pair])],worst_node=int(node),
                node_safe=node_safe,seg_safe=seg_safe,seg_margin=float(seg.min()),
                seg_bad=int(np.sum(seg<-tolerance)))


def run_full_lbfgs_instance(sc,psis,refs,deltas,*,seed=0,budget=80.,
                            clearance=0.,maxiter=90,stages=12,lower=None):
    """Optimize all payload controls jointly with pairwise AL-L-BFGS-B.

    No group partition, inflation or tube refinement. Report node safety like
    other baselines, and retain the solver's internal segment result separately.
    """
    lower=float(sum(translation.cost_value(ref['F'],'fuel') for ref in refs)) if lower is None else float(lower)
    row=dict(J_full_lbfgs=None,J_full_lbfgs_safe=None,safe_full_lbfgs=False,
             full_lbfgs_status='no_result',full_lbfgs_budget=float(budget),
             full_lbfgs_clearance=float(clearance),full_lbfgs_maxiter=maxiter,
             full_lbfgs_stages=stages)
    start=time.perf_counter()
    try:
        selected,info=group_tube_pairwise.solve_reference_set(
            sc,refs=copy.deepcopy(refs),psis=psis,objective='fuel',seed=seed,
            budget_s=budget,clearance=clearance,inner_iterations=maxiter,
            stages=stages,pair_deltas=deltas)
        solve_seconds=time.perf_counter()-start
        check=node_check([ref['r'] for ref in selected],deltas)
        fuel=float(sum(translation.cost_value(ref['F'],'fuel') for ref in selected))
        row.update(J_full_lbfgs=fuel,J_full_lbfgs_safe=fuel if check['safe'] else None,
                   safe_full_lbfgs=check['safe'],full_lbfgs_status=info['status'],
                   min_margin_full_lbfgs=check['min_margin'],
                   worst_pair_full_lbfgs=check['worst_pair'],worst_node_full_lbfgs=check['worst_node'],
                   seg_safe_full_lbfgs=check['seg_safe'],seg_margin_full_lbfgs=check['seg_margin'],seg_bad_full_lbfgs=check['seg_bad'],
                   full_lbfgs_internal_segment_safe=bool(info['ok']),
                   full_lbfgs_iters=info.get('iterations',0),
                   full_lbfgs_training_times=info.get('training_times'),
                   full_lbfgs_pair_time_constraints=info.get('pair_time_constraints'),
                   full_lbfgs_gap_pct=100*(fuel/lower-1) if lower>0 else None,
                   t_full_lbfgs_solve=solve_seconds)
    except Exception as exc:
        row.update(full_lbfgs_status='error',error_full_lbfgs=str(exc))
    row['t_full_lbfgs']=time.perf_counter()-start
    return row


def run_alternating_instance(sc,psis,refs,deltas,*,budget=80.,iters=20,lower=None):
    """Run the original joint iterative-plane algorithm and verify saved nodes."""
    if not np.isfinite(budget) or budget<=0 or iters<1:
        raise ValueError('alternating budget must be positive and iterations >= 1')
    lower=float(sum(translation.cost_value(ref['F'],'fuel') for ref in refs)) if lower is None else float(lower)
    pairs=list(zip(*np.triu_indices(sc.M,1)))
    row=dict(J_alternation=None,J_alternation_safe=None,safe_alternation=False,
             alternation_status='no_result',alt_iters=0,alt_capped=False,
             alternating_budget=float(budget),alternating_iterations=iters,
             alternating_budget_scope='checked after each joint SOCP; individual solves can exceed budget')
    start=time.perf_counter()
    try:
        result=planes.alternate_capped(sc,psis,pairs,refs,deltas,iters=iters,
                                       budget=budget,return_details=True)
        row.update(alt_iters=result['iterations'],alt_capped=result['capped'],
                   alternation_status=result['status'])
        if result['fuel'] is not None and result['positions'] is not None:
            check=node_check(result['positions'],deltas)
            fuel=float(result['fuel'])
            row.update(J_alternation=fuel,J_alternation_safe=fuel if check['safe'] else None,
                       safe_alternation=check['safe'],min_margin_alternation=check['min_margin'],
                       worst_pair_alternation=check['worst_pair'],worst_node_alternation=check['worst_node'],
                   seg_safe_alternation=check['seg_safe'],seg_margin_alternation=check['seg_margin'],seg_bad_alternation=check['seg_bad'],
                       alternation_gap_pct=100*(fuel/lower-1) if lower>0 else None)
    except Exception as exc:
        row.update(alternation_status='error',error_alternation=str(exc))
    row['t_alternation']=time.perf_counter()-start
    return row


def relax_tent_experimental(rbar, pairs, tgt, N, M, *, masses=None,
                            local_tent_width=None, max_sweeps=5000, tol=0.1):
    """Original tent sweeps with optional mass weighting and local support.

    Targets can be scalars or arrays over movable nodes 2..N-1, including
    variable headroom. local_tent_width is an internal half-width in nodes;
    the benchmark converts the CLI duration in seconds before calling here.
    Nodes 0, 1 and N stay pinned. No smoothing or fuel optimization is added.
    Keep experimental changes here rather than in shared benchmark_methods.
    """
    if masses is not None:
        masses = np.asarray(masses, dtype=float)
        if masses.shape != (M,) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError('masses must contain one positive finite mass per payload')
    if local_tent_width is not None and (not np.isfinite(local_tent_width) or local_tent_width <= 0):
        raise ValueError('local_tent_width must be positive and finite')
    x = [r.copy() for r in rbar]
    ks = np.arange(N + 1, dtype=float)
    B = np.zeros((N + 1, N + 1))
    for k in range(2, N):
        w = np.minimum((ks - 1) / (k - 1), (N - ks) / (N - k))
        w = np.clip(w, 0.0, None)
        if local_tent_width is not None:
            local = np.clip(1.0 - np.abs(ks - k) / local_tent_width, 0.0, None)
            w = np.minimum(w, local)
        w[0] = w[1] = w[N] = 0.0
        B[k] = w
    sweeps, worst = 0, np.inf
    while sweeps < max_sweeps and worst >= tol:
        worst = 0.0
        for j, l in pairs:
            d = x[j][2:N] - x[l][2:N]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            need = tgt[(j, l)] - dist
            kw = int(np.argmax(need))
            worst_need = float(need[kw])
            if worst_need <= 0:
                continue
            worst = max(worst, worst_need)
            g = d[kw] / dist[kw]
            if masses is None:
                F = np.outer(B[kw + 2], 0.5 * worst_need * g)
                x[j] += F
                x[l] -= F
            else:
                total = masses[j] + masses[l]
                push = np.outer(B[kw + 2], worst_need * g)
                x[j] += (masses[l] / total) * push
                x[l] -= (masses[j] / total) * push
        sweeps += 1
    return (x if worst < tol else None), sweeps, float(worst)


def headroom_schedule(N,dt,peak,*,variable=False,ramp_seconds=50.):
    """Extra spacing at each node; physical pair clearances are unchanged.

    Variable headroom rises linearly from pinned node 1 and falls to zero at N.
    Overlapping ramps are allowed; in that case the requested peak may not be reached.
    """
    if N<3 or not np.isfinite(dt) or dt<=0:
        raise ValueError('Require N>=3 and a positive finite time step')
    if not np.isfinite(peak) or peak<0:
        raise ValueError('headroom must be finite and nonnegative')
    if not np.isfinite(ramp_seconds) or ramp_seconds<=0:
        raise ValueError('headroom_ramp_seconds must be positive and finite')
    extra=np.full(N+1,float(peak))
    if variable:
        k=np.arange(N+1,dtype=float)
        extra*=np.clip(np.minimum((k-1)*dt,(N-k)*dt)/ramp_seconds,0.,1.)
    return extra


def run_inflation_instance(sc,psis,refs,deltas,*,profile='gauss',inflation_sigma=1.5,lower=None,
                           headroom=60.,variable_headroom=False,headroom_ramp_seconds=50.,mass_based_push=False,local_tent_width_seconds=None,
                           gated_tent=False,seg_tubes=False,gated_budget=120.):
    """Run inflation and original tubes on a prepared shared instance.

    Return inflation metrics only; do not generate scenarios or run group repair.
    local_tent_width_seconds is the time half-width; convert to fractional nodes
    only at the relaxation call so the spread is independent of time resolution.
    """
    if not np.isfinite(inflation_sigma) or inflation_sigma<=0:
        raise ValueError('inflation_sigma must be positive and finite')
    if (mass_based_push or local_tent_width_seconds is not None or gated_tent) and profile != 'tent':
        raise ValueError('Mass-based push, local tent width and gated tent require --profile tent or --tent-only')
    if gated_tent and local_tent_width_seconds is not None:
        raise ValueError('Gated tent uses the global tent (its convergence proof needs it); drop --local-tent-width')
    if local_tent_width_seconds is not None and (not np.isfinite(local_tent_width_seconds) or local_tent_width_seconds <= 0):
        raise ValueError('local_tent_width_seconds must be positive and finite (seconds per side)')
    N,M=sc.N,sc.M
    headroom_schedule(N,sc.dt,headroom,variable=variable_headroom,ramp_seconds=headroom_ramp_seconds)
    pairs=list(zip(*np.triu_indices(M,1)))
    lower=float(sum(translation.cost_value(ref['F'],'fuel') for ref in refs)) if lower is None else float(lower)
    row=dict(inflation_sigma=float(inflation_sigma),
             inflation_sigma_units='nodes',inflation_sigma_seconds=float(inflation_sigma*sc.dt),
             headroom_mode='variable' if variable_headroom else 'constant',
             headroom_initial_m=float(headroom),
             headroom_ramp_seconds=float(headroom_ramp_seconds) if variable_headroom else 0.,
             mass_based_push=bool(mass_based_push),gated_tent=bool(gated_tent),
             tube_type='segment' if seg_tubes else 'node',
             local_tent_width_units='seconds',
             local_tent_width_nodes=float(local_tent_width_seconds/sc.dt) if local_tent_width_seconds is not None else None,
             local_tent_half_width_seconds=float(local_tent_width_seconds) if local_tent_width_seconds is not None else None)
    rbar=np.array([ref['r'] for ref in refs])
    # Original inflation and original tubes. Do not use adapted segment corridors.
    inflation_start=time.perf_counter()
    relaxation_seconds,tube_seconds=0.,0.
    attempts=[]
    row.update(J_tubes=None,J_tubes_safe=None,safe_tubes=False,tubes_status='no_result')
    try:
        for peak_headroom in (headroom,2*headroom):
            t0=time.perf_counter()
            extra=headroom_schedule(N,sc.dt,peak_headroom,variable=variable_headroom,
                                    ramp_seconds=headroom_ramp_seconds)
            targets={(i,j):deltas[i,j]+extra[2:N] for i,j in pairs} if variable_headroom else {
                (i,j):float(deltas[i,j]+peak_headroom) for i,j in pairs}
            gate=None
            if gated_tent:
                import gated_tent as gt
                X,gate=gt.deconflict(rbar,pairs,targets,
                                     masses=[p.m for p in sc.payloads] if mass_based_push else None,
                                     variant='gated',budget_s=gated_budget,deltas=deltas)
                centers=None if X is None else [x for x in X]
                sweeps,residual=gate['sweeps'],gate['worst']
            elif profile == 'tent' and (mass_based_push or local_tent_width_seconds is not None):
                centers,sweeps,residual=relax_tent_experimental(
                    rbar,pairs,targets,N,M,
                    masses=[p.m for p in sc.payloads] if mass_based_push else None,
                    local_tent_width=local_tent_width_seconds/sc.dt if local_tent_width_seconds is not None else None)
            else:
                centers,sweeps,residual=leader.relax_iso(
                    rbar,pairs,targets,N,M,profile=profile,sigma=inflation_sigma)
            tr=time.perf_counter()-t0;relaxation_seconds+=tr
            detail=dict(headroom=float(peak_headroom),t_relax=tr,sweeps=sweeps,residual=float(residual),
                        headroom_max_applied_m=float(extra[2:N].max()),
                        headroom_first_movable_m=float(extra[2]),
                        headroom_last_movable_m=float(extra[N-1]))
            if gate is not None:
                detail.update(gate_status=gate['status'],gate_rejected_sweeps=gate['rejected_sweeps'],
                              gate_halvings=gate['halvings'],gate_grad_steps=gate['grad_steps'],
                              gate_phi_monotone=gate['phi_monotone'],
                              gate_seg_certified=gate.get('seg_certified'),
                              gate_seg_margin=gate.get('seg_margin'),gate_lemma_margin=gate.get('lemma_margin'))
            attempts.append(detail)
            if centers is None:
                row['tubes_status']='relax_failed'
                if gate is not None:
                    row['gate_status']=gate['status']
                break
            t0=time.perf_counter()
            if seg_tubes:
                import expand_method as em
                fuel,positions=em.seg_tubes(sc,psis,[np.asarray(c,float) for c in centers],
                                            pairs,deltas,N,M)
                each=[time.perf_counter()-t0]
            else:
                fuel,positions,each=leader.tubes_detail(sc,psis,pairs,deltas,centers,N,M)
            tt=time.perf_counter()-t0;tube_seconds+=tt;detail['t_tubes']=tt
            if fuel is None:
                if seg_tubes:
                    detail['tube_failure']=str(positions)
                else:
                    detail['failed_payload']=int(positions)
                row['tubes_status']='tube_solve_failed'
                continue
            check=node_check(positions,deltas)
            row.update(J_tubes=float(fuel),J_tubes_safe=float(fuel) if check['safe'] else None,
                       safe_tubes=check['safe'],min_margin_tubes=check['min_margin'],
                       worst_pair_tubes=check['worst_pair'],worst_node_tubes=check['worst_node'],
                   seg_safe_tubes=check['seg_safe'],seg_margin_tubes=check['seg_margin'],seg_bad_tubes=check['seg_bad'],
                       headroom=float(peak_headroom),relax_sweeps=sweeps,t_tube_max=max(each),
                       tubes_gap_pct=100*(float(fuel)/lower-1),
                       tubes_status=('safe' if SEGMENT_SAFETY else 'node_safe') if check['safe']
                       else ('unsafe' if SEGMENT_SAFETY else 'node_unsafe'))
            # Same retry rule as leader: retry infeasible solves, not post-check violations.
            break
    except Exception as exc:
        row.update(tubes_status='error',error_tubes=str(exc))
    row.update(t_relax=relaxation_seconds,t_tubes=tube_seconds,
               t_inflation_tubes_total=time.perf_counter()-inflation_start,
               inflation_attempts=attempts)

    return row


def run_groupnlp_instance(sc,psis,refs,deltas,*,seed=0,group_size=4,
                          group_budget=80.,refine_group=True,group_fuel_weight=.05,group_inner_iterations=30,
                          lower=None):
    """Run group repair and optional original tubes on a prepared instance.

    Return group-only and group+tubes metrics; do not perform inflation.
    """
    if not isinstance(group_inner_iterations,(int,np.integer)) or group_inner_iterations<1:
        raise ValueError('group_inner_iterations must be a positive integer')
    if not np.isfinite(group_fuel_weight) or group_fuel_weight < 0:
        raise ValueError('group_fuel_weight must be finite and nonnegative')
    N,M=sc.N,sc.M
    pairs=list(zip(*np.triu_indices(M,1)))
    lower=float(sum(translation.cost_value(ref['F'],'fuel') for ref in refs)) if lower is None else float(lower)
    row={}
    # Existing group repair, retained separately from its optional tube refinement.
    group_start=time.perf_counter()
    row.update(J_group=None,J_group_safe=None,safe_group=False,group_status='no_result',
               group_inner_iterations=int(group_inner_iterations))
    try:
        selected,_,info=group_tube_repair.solve_reference_set(sc,copy.deepcopy(refs),psis,
                       budget_s=group_budget,group_size=group_size,seed=seed,
                       pair_deltas=deltas,repair_only=True,fuel_weight=group_fuel_weight,
                       inner_iterations=group_inner_iterations)
        solve_seconds=time.perf_counter()-group_start
        check=node_check([ref['r'] for ref in selected],deltas)
        fuel=float(sum(translation.cost_value(ref['F'],'fuel') for ref in selected))
        row.update(J_group=fuel,J_group_safe=fuel if check['safe'] else None,
                   safe_group=check['safe'],min_margin_group=check['min_margin'],
                   worst_pair_group=check['worst_pair'],worst_node_group=check['worst_node'],
                   seg_safe_group=check['seg_safe'],seg_margin_group=check['seg_margin'],seg_bad_group=check['seg_bad'],
                   t_group=solve_seconds,group_gap_pct=100*(fuel/lower-1),
                   group_status=info['status'],group_internal_segment_safe=bool(info['ok']),
                   group_local_solves=info.get('local_solves',0))
    except Exception as exc:
        row.update(group_status='error',error_group=str(exc),t_group=time.perf_counter()-group_start)
    row['t_group_total']=time.perf_counter()-group_start
    row['t_group_repair']=row['t_group']

    # Refine the SAME group solution; no extra group repair or geometric inflation.
    row.update(J_group_tubes=None,J_group_tubes_safe=None,safe_group_tubes=False,
               t_group_tubes=None,t_group_tubes_total=None,
               group_tubes_status='disabled' if not refine_group else 'skipped_group_not_node_safe')
    if refine_group and row['safe_group']:
        refinement_start=time.perf_counter()
        try:
            refined_fuel,refined_positions,each=leader.tubes_detail(
                sc,psis,pairs,deltas,[ref['r'] for ref in selected],N,M)
            row['t_group_tubes']=time.perf_counter()-refinement_start
            if refined_fuel is None:
                row.update(group_tubes_status='tube_solve_failed',group_tubes_failed_payload=int(refined_positions))
            else:
                refined_check=node_check(refined_positions,deltas)
                refined_fuel=float(refined_fuel)
                row.update(J_group_tubes=refined_fuel,
                           J_group_tubes_safe=refined_fuel if refined_check['safe'] else None,
                           safe_group_tubes=refined_check['safe'],
                           min_margin_group_tubes=refined_check['min_margin'],
                           worst_pair_group_tubes=refined_check['worst_pair'],
                           worst_node_group_tubes=refined_check['worst_node'],
                   seg_safe_group_tubes=refined_check['seg_safe'],seg_margin_group_tubes=refined_check['seg_margin'],seg_bad_group_tubes=refined_check['seg_bad'],
                           group_tubes_gap_pct=100*(refined_fuel/lower-1),
                           group_tubes_fuel_reduction_pct=100*(1-refined_fuel/row['J_group']) if row['J_group']>0 else None,
                           t_group_tube_max=max(each),
                           group_tubes_status='node_safe' if refined_check['safe'] else 'node_unsafe')
        except Exception as exc:
            row.update(group_tubes_status='error',error_group_tubes=str(exc),
                       t_group_tubes=time.perf_counter()-refinement_start)
        row['t_group_tubes_total']=row['t_group_total']+time.perf_counter()-refinement_start
    row['group_tubes_budget_scope']='original tube SOCPs are additional to group repair budget'
    return row


def run_sigma_sweep(M,seed,N,tf,*,layout='random_seq',profile='gauss',ring_radius=None,
                 inflation_sigmas=(1.5,),
                 group_size=4,group_budget=80.,refine_group=True,group_fuel_weight=.05,group_inner_iterations=30,
                 alternating=True,alternating_budget=80.,alternating_iters=20,
                 full_lbfgs=True,full_lbfgs_budget=80.,full_lbfgs_clearance=0.,
                 full_lbfgs_maxiter=90,full_lbfgs_stages=12,tent_only=False,
                 headroom=60.,variable_headroom=False,headroom_ramp_seconds=50.,mass_based_push=False,local_tent_width_seconds=None,
                 gated_tent=False,seg_tubes=False,gated_budget=120.):
    headroom_schedule(N,tf/N,headroom,variable=variable_headroom,ramp_seconds=headroom_ramp_seconds)
    if tent_only:
        profile='tent'
        inflation_sigmas=(1.5,)  # Tent has no Gaussian sigma sweep.
        alternating=False
        full_lbfgs=False
        refine_group=False
    if (mass_based_push or local_tent_width_seconds is not None) and profile != 'tent':
        raise ValueError('Mass-based push and local tent width require --profile tent or --tent-only')
    if local_tent_width_seconds is not None and (not np.isfinite(local_tent_width_seconds) or local_tent_width_seconds <= 0):
        raise ValueError('local_tent_width_seconds must be positive and finite (seconds per side)')
    if not isinstance(group_inner_iterations,(int,np.integer)) or group_inner_iterations<1:
        raise ValueError('group_inner_iterations must be a positive integer')
    if not np.isfinite(group_fuel_weight) or group_fuel_weight < 0:
        raise ValueError("group_fuel_weight must be finite and nonnegative")
    inflation_sigmas=tuple(dict.fromkeys(float(sigma) for sigma in inflation_sigmas))
    if not inflation_sigmas or any(not np.isfinite(sigma) or sigma<=0 for sigma in inflation_sigmas):
        raise ValueError('inflation_sigmas must contain positive finite values')
    ring=500.*max(1.,M/20.)**(1./3.) if ring_radius is None else float(ring_radius)
    if not np.isfinite(ring) or ring<=0:
        raise ValueError('ring radius must be finite and positive')
    setup_start=time.perf_counter()
    # Match the dedicated leader benchmark's bounded random-layout retries.
    for attempt in range(8 if layout=='random' and ring_radius is None else 1):
        try:
            sc=scenario.generate(layout,M=M,n_points=8,n_agents=3*M,seed=seed,
                                 nu_deg=60,N=N,tf=tf,ring_radius=ring,max_tries=5000)
            break
        except RuntimeError as exc:
            if 'could not place payloads' not in str(exc) or layout!='random' or ring_radius is not None:
                raise
            if attempt==7:
                raise
            ring*=1.15
    placement_seconds=time.perf_counter()-setup_start
    psis=translation.psi_list(sc)
    t0=time.perf_counter()
    refs=translation.solve_all(sc,psis=psis)
    nominal_seconds=time.perf_counter()-t0
    if not all(ref['ok'] for ref in refs):
        return [dict(M=M,seed=seed,inflation_sigma=sigma,status='nominal_failed',
                     verification='nodes_only') for sigma in inflation_sigmas]
    lower=float(sum(translation.cost_value(ref['F'],'fuel') for ref in refs))
    extent=np.array([float(np.max(np.linalg.norm(P.points,axis=1))) for P in sc.payloads])
    margin=sc.delta-2*max(extent)
    deltas=extent[:,None]+extent[None,:]+margin
    pairs=list(zip(*np.triu_indices(M,1)))
    rbar=np.array([ref['r'] for ref in refs])
    initial=node_check(rbar,deltas)
    conflicts=sum(np.min(np.linalg.norm(rbar[i]-rbar[j],axis=1))<deltas[i,j] for i,j in pairs)
    row=dict(M=M,seed=seed,N=N,tf=tf,layout=layout,profile=profile,ring_radius=ring,
             radius_source='density_scaling' if ring_radius is None else 'explicit',
             verification='nodes_only',verification_tolerance_m=1e-6,
             clearance='pair-specific extents plus shared margin',LB=lower,
             t_placement=placement_seconds,t_nominals=nominal_seconds,
             n_pairs=len(pairs),n_conflicts=int(conflicts),baseline_node_safe=initial['safe'],
             group_size=group_size,group_budget=group_budget,group_fuel_weight=float(group_fuel_weight),
             group_internal_verification='unchanged: full linear segments',
             objective='native sum of force norms; no dt multiplier')

    inflation_rows=[]
    for sigma in inflation_sigmas:
        label=('gated tent' if gated_tent else 'tent')+' inflation+tubes' if tent_only else f'inflation_sigma={sigma:g}: inflation+tubes'
        print(f'M={M} seed={seed} {label} (node check)',
              file=sys.stderr,flush=True)
        inflation_rows.append(run_inflation_instance(
            sc,psis,refs,deltas,profile=profile,inflation_sigma=sigma,lower=lower,
            headroom=headroom,variable_headroom=variable_headroom,
            headroom_ramp_seconds=headroom_ramp_seconds,
            mass_based_push=mass_based_push,local_tent_width_seconds=local_tent_width_seconds,
            gated_tent=gated_tent,seg_tubes=seg_tubes,gated_budget=gated_budget))
    if not tent_only:
        row.update(run_groupnlp_instance(sc,psis,refs,deltas,seed=seed,
                   group_size=group_size,group_budget=group_budget,refine_group=refine_group,
                   group_fuel_weight=group_fuel_weight,group_inner_iterations=group_inner_iterations,lower=lower))
    else:
        row.update(group_internal_verification='disabled',J_group=None,J_group_safe=None,safe_group=False,
                   group_status='disabled',t_group=None,t_group_total=None,t_group_repair=None,
                   J_group_tubes=None,J_group_tubes_safe=None,safe_group_tubes=False,
                   group_tubes_status='disabled',t_group_tubes=None,t_group_tubes_total=None)
    if alternating:
        row.update(run_alternating_instance(sc,psis,refs,deltas,budget=alternating_budget,
                                             iters=alternating_iters,lower=lower))
    else:
        row.update(J_alternation=None,J_alternation_safe=None,safe_alternation=False,
                   alternation_status='disabled',t_alternation=None,alt_iters=0,alt_capped=False)
    if full_lbfgs:
        row.update(run_full_lbfgs_instance(sc,psis,refs,deltas,seed=seed,budget=full_lbfgs_budget,
                   clearance=full_lbfgs_clearance,maxiter=full_lbfgs_maxiter,stages=full_lbfgs_stages,lower=lower))
    else:
        row.update(J_full_lbfgs=None,J_full_lbfgs_safe=None,safe_full_lbfgs=False,
                   full_lbfgs_status='disabled',t_full_lbfgs=None)
    rows=[]
    for inflation_row in inflation_rows:
        result=dict(row)
        result.update(inflation_row)
        if tent_only:
            result['tent_only']=True
            for key in ('inflation_sigma','inflation_sigma_units','inflation_sigma_seconds'):
                result.pop(key,None)
        result['status']='partial' if any(key in result for key in ('error_tubes','error_group','error_group_tubes','error_alternation','error_full_lbfgs')) else 'completed'
        if len(inflation_sigmas)>1:
            result['sigma_sweep_shared_baselines']=True
        rows.append(result)
    return rows


def run_instance(M,seed,N,tf,*,inflation_sigma=1.5,**kwargs):
    """Backward-compatible single-sigma benchmark returning one result dict."""
    return run_sigma_sweep(M,seed,N,tf,inflation_sigmas=(inflation_sigma,),**kwargs)[0]


def parse_sigma_values(value):
    """Accept a number or comma-separated list as a CLI argument."""
    try:
        values=[float(part) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('sigma must be a number or comma-separated numbers') from exc
    if any(not np.isfinite(sigma) or sigma<=0 for sigma in values):
        raise argparse.ArgumentTypeError('sigma values must be positive and finite')
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Ms','--M',type=int,nargs='+',default=[20,40,70,100])
    parser.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5,6])
    parser.add_argument('--N',type=int,default=30)
    parser.add_argument('--tf',type=float,default=500.)
    parser.add_argument('--layout',choices=('random','random_seq','crossing'),default='random_seq')
    parser.add_argument('--ring','--ring-radius',dest='ring_radius',type=float)
    parser.add_argument('--profile',choices=('gauss','tent','beam'),default='gauss')
    parser.add_argument('--tent-only',action='store_true',
                        help='Run only tent inflation plus original tubes; skip group repair/refinement, alternating and full L-BFGS. Ignore the Gaussian sigma sweep.')
    parser.add_argument('--inflation_sigma','--inflation-sigma','--iflation_sigma',
                        dest='inflation_sigma',type=parse_sigma_values,nargs='+',default=[[1.5]],
                        help='One or more Gaussian smoothing sigmas in nodes, not seconds: 3 4 5 6 or 3,4,5,6 (default: 1.5). Applies to gauss. Each sigma gets a result row; other baselines run once per M/seed.')
    parser.add_argument('--mass-based-push',action='store_true',
                        help='Split tent pushes inversely by payload mass (lighter moves farther); requires tent profile')
    parser.add_argument('--local-tent-width-seconds','--local-tent-width',dest='local_tent_width_seconds',type=float,default=None,
                        help='Tent half-width in SECONDS on each side of the conflict peak (e.g. 133.33333333333334); default: global tent. Both flag spellings use seconds. Requires tent profile')
    parser.add_argument('--gated-tent',action='store_true',
                        help='Gated tent (studies/gated_tent.py): tent sweeps accepted only under sufficient decrease of the merit function, with halving and gradient fallback; provably converges to a stationary point. Implies --profile tent; combine with --mass-based-push for mass weighting')
    parser.add_argument('--gated-budget',type=float,default=120.,help='Gated-tent time budget per relaxation in seconds (default: 120)')
    parser.add_argument('--seg-tubes',action='store_true',
                        help='Use segment-sized tubes (radius = half the smallest segment surplus over delta; safe between nodes) instead of the original node tubes')
    parser.add_argument('--segment-check',action='store_true',
                        help='Count a result as safe only if every between-node segment also clears delta (segment fields are always reported)')
    parser.add_argument('--headroom',type=float,default=60.,
                        help='Initial extra inflation spacing in metres (default: 60); doubled on tube-infeasibility retry')
    parser.add_argument('--variable-headroom',action='store_true',
                        help='Gradually increase extra spacing after pinned node 1 and decrease it before arrival; physical clearance is unchanged')
    parser.add_argument('--headroom-ramp-seconds',type=float,default=50.,
                        help='Duration of each variable-headroom ramp in seconds (default: 50); used with --variable-headroom')
    parser.add_argument('--group-size',type=int,default=4)
    parser.add_argument('--group-inner-iterations','--group_inner_iterations',type=int,default=30,
                        help='Maximum L-BFGS iterations per local group solve (default: 30); independent of full joint L-BFGS limits')
    parser.add_argument('--group-budget',type=float,default=80.,help='Group repair seconds; subsequent original tube solves are additional')
    parser.add_argument('--group-fuel-weight','--group_fuel_weight',dest='group_fuel_weight',type=float,default=.05,
                        help='Nonnegative group repair fuel weight versus collision penalties; tube objective unchanged (default: 0.05)')
    parser.add_argument('--alternating-budget','--alt-budget',type=float,default=80.,
                        help='Iterative-plane seconds, checked after each SOCP (default: 80)')
    parser.add_argument('--alternating-iters',type=int,default=20,help='Maximum iterative-plane SOCP iterations (default: 20)')
    parser.add_argument('--no-alternating',action='store_true',help='Skip the iterative-plane baseline')
    parser.add_argument('--full-lbfgs-budget',type=float,default=80.,help='Full joint pairwise AL-L-BFGS budget in seconds (default: 80)')
    parser.add_argument('--full-lbfgs-clearance',type=float,default=0.,help='Extra planning clearance in metres; physical checks unchanged (default: 0)')
    parser.add_argument('--full-lbfgs-maxiter',type=int,default=90,help='Inner L-BFGS iterations per AL stage (default: 90)')
    parser.add_argument('--full-lbfgs-stages',type=int,default=12,help='Maximum AL stages (default: 12)')
    parser.add_argument('--no-full-lbfgs',action='store_true',help='Skip the full joint pairwise AL-L-BFGS baseline')
    parser.add_argument('--no-group-tubes'  ,action='store_true',help='Skip the added group-to-original-tubes refinement')
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('results')/'benchmark_tube_vs_groupnlp.json')
    args=parser.parse_args()
    if args.tent_only:
        args.profile='tent'
        args.inflation_sigma=[[1.5]]
        args.no_alternating=True
        args.no_full_lbfgs=True
        args.no_group_tubes=True
    if args.gated_tent:
        args.profile='tent'
        if args.local_tent_width_seconds is not None:
            parser.error('--gated-tent uses the global tent; drop --local-tent-width')
    if args.segment_check:
        global SEGMENT_SAFETY
        SEGMENT_SAFETY=True
    if (args.mass_based_push or args.local_tent_width_seconds is not None) and args.profile != 'tent':
        parser.error('--mass-based-push and --local-tent-width require --profile tent or --tent-only')
    if args.local_tent_width_seconds is not None and (not np.isfinite(args.local_tent_width_seconds) or args.local_tent_width_seconds <= 0):
        parser.error('--local-tent-width-seconds must be positive and finite (seconds per side)')
    if min(args.Ms)<2 or args.N<3 or args.group_size<2 or not np.isfinite(args.tf) or args.tf<=0 or not np.isfinite(args.group_budget) or args.group_budget<=0:
        parser.error('Require M>=2, N>=3, group-size>=2 and positive finite tf/budget')
    if args.group_inner_iterations<1:
        parser.error('--group-inner-iterations must be a positive integer')
    if not np.isfinite(args.group_fuel_weight) or args.group_fuel_weight < 0:
        parser.error("--group-fuel-weight must be finite and nonnegative")
    if not np.isfinite(args.alternating_budget) or args.alternating_budget<=0 or args.alternating_iters<1:
        parser.error('Require positive finite --alternating-budget and --alternating-iters >= 1')
    if not np.isfinite(args.full_lbfgs_budget) or args.full_lbfgs_budget<=0 or not np.isfinite(args.full_lbfgs_clearance) or args.full_lbfgs_clearance<0 or args.full_lbfgs_maxiter<1 or args.full_lbfgs_stages<1:
        parser.error('Require positive full L-BFGS budget/iteration limits and finite nonnegative clearance')
    try:
        headroom_schedule(args.N,args.tf/args.N,args.headroom,variable=args.variable_headroom,
                          ramp_seconds=args.headroom_ramp_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    args.inflation_sigma=list(dict.fromkeys(sigma for values in args.inflation_sigma for sigma in values))
    if args.ring_radius is not None and (not np.isfinite(args.ring_radius) or args.ring_radius<=0):
        parser.error('--ring must be positive and finite')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for M in args.Ms:
        for seed in args.seeds:
            methods='tent inflation+tubes only' if args.tent_only else 'inflation+tubes versus group, with/without tubes, plus iterative planes and full joint L-BFGS'
            print(f'M={M} seed={seed}: {methods} (node check)',file=sys.stderr,flush=True)
            try:
                instance_rows=run_sigma_sweep(M,seed,args.N,args.tf,layout=args.layout,profile=args.profile,
                                 ring_radius=args.ring_radius,inflation_sigmas=args.inflation_sigma,group_size=args.group_size,group_budget=args.group_budget,
                                 refine_group=not args.no_group_tubes,group_fuel_weight=args.group_fuel_weight,
                                 group_inner_iterations=args.group_inner_iterations,
                                 alternating=not args.no_alternating,alternating_budget=args.alternating_budget,
                                 alternating_iters=args.alternating_iters,
                                 full_lbfgs=not args.no_full_lbfgs,full_lbfgs_budget=args.full_lbfgs_budget,
                                 full_lbfgs_clearance=args.full_lbfgs_clearance,full_lbfgs_maxiter=args.full_lbfgs_maxiter,
                                 full_lbfgs_stages=args.full_lbfgs_stages,tent_only=args.tent_only,
                                 headroom=args.headroom,variable_headroom=args.variable_headroom,
                                 headroom_ramp_seconds=args.headroom_ramp_seconds,
                                 mass_based_push=args.mass_based_push,local_tent_width_seconds=args.local_tent_width_seconds,
                                 gated_tent=args.gated_tent,seg_tubes=args.seg_tubes,gated_budget=args.gated_budget)
            except Exception as exc:
                instance_rows=[dict(M=M,seed=seed,inflation_sigma=sigma,verification='nodes_only',
                                    status='setup_error',error=str(exc)) for sigma in args.inflation_sigma]
            for row in instance_rows:
                rows.append(row)
                print(json.dumps(row,allow_nan=False),flush=True)
            args.out.write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Wrote {args.out}',flush=True)
    sys.stdout.flush()
    os._exit(0)  # Native project workaround for CasADi process teardown.


if __name__=='__main__':
    main()

