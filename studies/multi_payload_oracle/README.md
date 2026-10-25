# Multi-payload oracle studies

Numerical studies behind the multi-payload transport notes (paper repo:
*Safe, Fuel-Efficient Transportation of Multiple Rigid Payloads with a
Spacecraft Swarm*). This folder holds the experiments that selected the
oracle's deconfliction method; the reusable model code (scenarios, dynamics,
allocation, per-payload solvers) lives in `multi_payload/` at the repo root,
and these scripts import it by relative path, so run them from anywhere.

Outputs go to `results/` (gitignored — regenerate with the commands below).

## The method these studies settled on

1. **Nominals.** One convex translational solve per payload
   (`multi_payload.translation`), ignoring collisions. Their total fuel
   `LB = sum(l_j)` is a lower bound on the optimum, independent of the
   allocation, so every later result carries a certified suboptimality gap.
2. **Inflation relaxation** (`relax_iso` in `benchmark_methods.py`). Every
   trajectory point carries a protective sphere; same-step pairs closer than
   the per-pair clearance `delta_jl + h` (headroom `h = 60 m`) are pushed
   apart by exact-gap symmetric projections, the displacement field is
   Gaussian-smoothed along time (endpoints and step 1 pinned: `v0 = 0` makes
   `r_1 = r_0` immovable), and termination is the exact Euclidean clearance
   check. Pure numpy, no solver.
3. **Tube hand-off** (`tubes_solve` / `benchmark_tubes.py`). Each payload
   gets per-step ball corridors of radius = half its worst pairwise surplus;
   corridors are disjoint by construction, so the re-plans are M independent
   convex solves — parallel and communication-free.

Push profiles for step 2 (`relax_iso(..., profile=...)`), all measured on
10 seeds x M = 10..100 at constant congestion density:

| profile | certified gap | relax time (M=90) | use |
|---|---|---|---|
| `gauss` (default) | flat 11-13% | up to ~43 s | the method |
| `tent` | flat 13-16.5% | < 1 s | fast mode / onboard replanning |
| `beam`, `bump<W>`, `anneal` | worse | — | negative results, kept for the record |

## Main entry points

- `benchmark_tubes.py` — the headline Monte Carlo: nominals -> relaxation ->
  tubes, with LB, certified gap, exact separation check, per-stage times.
  `py benchmark_tubes.py --Ms 10 20 ... 100 --seeds 0 ... 9 --layout random_seq
  --profile gauss --out benchmark_tubes_fair.json` (~25 min for 100 instances;
  `--profile tent` for the fast mode). `--layout random_seq` is dart-throwing
  placement at constant congestion density (~1 conflicting pair per payload);
  plain `random` is the legacy all-pairs rejection draw.
- `plot_results.py` — regenerates the two paper charts from `results/`.
- `benchmark_methods.py` — alternation vs inflation+tubes vs inflation+planes
  (iso and anisotropic), the method-comparison study at M = 2..20.
- `benchmark_scale100.py` — the large-M comparison with a wall-clock-budgeted
  alternation baseline and a pruned one-shot planes solve.
- `benchmark_pipeline.py` — full certified pipeline per instance: closure-greedy
  allocation -> deconfliction -> attitude references -> per-payload agent-level
  cone-constrained solves -> gap vs LB.
- `scp_oracle.py` — full-pose (6-DoF) SCP oracle with joint convex polish;
  the strongest solutions at small M, not part of the notes' method.

## Exploration record (kept for reference, all measured to a verdict)

- `inflate_oracle.py` — original (non-vectorized) inflation relaxation with
  fuel-metric anisotropic pushes; superseded by `relax_iso`.
- `corridor_oracle.py` — spatially disjoint corridors via grid search + prism
  merging + keep-in MILP: works, communication-free, but +28% fuel. Also holds
  `alternate_pp` (the iterated separating-planes baseline) and `axis_prices`.
- `occupancy_misocp.py`, `occupancy_probe.py`, `occupancy_cubes.py` —
  cell-occupancy binaries (MISOCP): sound encoding, intractable or
  geometry-rejecting in practice (cubes variant built, never run).
- `warp_oracle.py` — global space-warp deconfliction: terminates by
  construction but freezes the schedule; up to 5x fuel.
- `route2_check.py`, `route2_multi.py` — separating-plane alternation studies
  and the (vacuous) McCormick relaxation check.
- `sandwich_check.py` — closure-greedy allocation + bounds sandwich; exports
  `BASIS` (the 12 basis wrenches for the closure residual).
- `measure_price.py`, `authority_check.py`, `oracle_check.py` — lower-bound
  tightening diagnostics (all closed: LB = sum(l_j) is the structural ceiling).

## Dependencies

numpy, scipy, cvxpy (CLARABEL ships with it), matplotlib for plots.
MOSEK is optional: only the MISOCP studies and some allocation fallbacks use it.
