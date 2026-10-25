# Multi-payload cooperative transport

Extension of the paper code to **M rigid payloads moved by one swarm of N identical
agents**. Agents choose which open attachment points to occupy (the *Allocation
Problem*), then every payload is moved to its goal without coming within a safety
distance δ of another payload, minimizing total fuel (the *Oracle Problem*).

Everything uses the paper's physics unchanged: Tschauner–Hempel relative orbital
dynamics on a shared chief orbit, SO(3) attitude, body-frame thrust with inward
cones of half-angle ν, fuel Σ‖u‖, and independent rollout verification.

The package (this directory) is the model layer: scenarios, dynamics, references,
allocation, corridors, per-payload solvers, verification, docking, animation.
`studies/` holds the investigation that selected the deconfliction method, its
negative results, and the Monte Carlo benchmarks — see **The studies** below.

## Method

**Oracle Problem (allocation given)**

1. **Translational references** (`translation.py`). Translation depends on the agents
   only through the net force at the center of mass, so each payload gets a convex
   fuel-optimal reference. These don't depend on the allocation or the attitude, and
   their total fuel `Σℓ_j` is a lower bound on the optimum — the certificate every
   result is measured against.
2. **Inflation relaxation** (`studies/benchmark_methods.py:relax_iso`, the current
   reference generator). Every trajectory point carries a protective sphere sized to
   its payload, so sphere disjointness is exactly the per-pair separation
   δ_jl = ext_j + ext_l + margin. All references are deconflicted *simultaneously*:
   same-step pairs closer than δ_jl + h (headroom h = 60 m) are pushed apart by
   exact-gap symmetric projections, the displacement field is Gaussian-smoothed along
   the time index (endpoints pinned, and step 1 pinned because v₀ = 0 makes r₁ = r₀
   immovable), and termination is the exact Euclidean clearance check. Pure numpy,
   no solver. Push profiles: `gauss` (default, best fuel), `tent` (one whole-path
   pull per conflict, ~90× faster for ~3 points more gap — the onboard/replanning
   mode). *The package's original waypoint repair (`conflicts.py`) is kept as a
   comparison baseline (`oracles.py` method `"waypoint"`); the relaxation supersedes
   it — it deconflicts all payloads at once and cannot oscillate.*
3. **Ball corridors** (`oracle.py`; `studies/benchmark_tubes.py:tubes_solve`).
   Each payload is confined to per-step balls around its cleared reference with
   radius = half its worst pairwise surplus, so corridors are **disjoint by
   construction**: a payload that stays in its corridor is safe whatever the others
   fly. Everything downstream is decoupled, parallel, and communication-free.
4. **Single-payload 6-DoF solves** (`gd_corridor.py`, `socp_corridor.py`). Modified
   copies of the paper's COT solver: the corridor sits in the convex inner SOCP and
   the envelope gradient is unchanged. C-GD by default, D-GD with `--step5 dgd`.
5. **Corridor relaxation** (`oracle.py`). Elastic slack; nonzero slack widens the
   corridor within its neighbors' room, then falls back to re-running the relaxation
   with more headroom.

If every payload's agents have wrench closure, radius-0 corridors are always
feasible and step 5 never triggers.

**Allocation Problem** (`attitude.py`, `allocation.py`)

The references above plus an allocation-independent attitude reference (the paper's
shooting warm start, projected) fix the wrench each payload needs at every step. The
surrogate cost of an allocation is the minimum fuel to produce those wrenches, an
upper bound on the oracle value. Rule: minimize it if some allocation makes every
payload feasible, otherwise minimize the worst normalized wrench residual. Solved by
**sequential greedy** or exactly by a **mixed-integer SOCP** (`--allocation misocp`,
needs MOSEK). The trajectory-free variant used in the notes places agents greedily
on the wrench-closure residual over the 12 basis wrenches alone
(`studies/sandwich_check.py:BASIS` + `allocation.greedy`).

**Certificate:** lower bound ≤ J* ≤ J_final ≤ surrogate, all reported.

**Objective.** `--objective fuel` (Σ‖u‖, the paper's, default) or `--objective energy`
(Σ‖u‖², continuous thrust). It is used consistently by every stage, and the
certificate is reported in its units. For energy, the lower bound is
Σ_j ℓ_j / A_j (Cauchy–Schwarz over the A_j agents of payload j).

**Docking** (`docking.py`)

Agents start free-flying and dock before transport begins. Docking runs on
t ∈ [−T_dock, 0] and ends exactly at each payload's transport start state; during
it the payloads drift uncontrolled (propagated backward through the same dynamics).
Agents are matched to the allocated points by the Hungarian method on docking fuel,
then each runs shrinking-horizon fuel-optimal MPC in the style of Park, Di Cairano
and Kolmanovsky (ACC 2011): a line-of-sight cone from the docking port, a tangent
half-plane, an approach-speed limit and soft docking, with a thrust limit.
Constraints are softened so the closed loop stays feasible under thrust execution
noise. Agents do not avoid each other during docking; the minimum agent–agent
distance is reported.

**Animation** (`animate.py`). PyVista render of the whole mission (allocation,
docking, transport), driven by the computed trajectories, camera to scale.

## The studies (`studies/`)

The investigation that selected the inflation relaxation, with every alternative
measured to a verdict. Outputs go to `studies/results/` (gitignored — regenerate
with the commands below; `plot_results.py` remakes the headline charts).

**Headline Monte Carlo** — `benchmark_tubes.py`: nominals → relaxation → disjoint
tubes, reporting the lower bound, the certified gap, exact pairwise separation, and
per-stage times. On random orbital instances at constant congestion density
(~1 conflicting pair per payload, dart-throwing placement `--layout random_seq`),
M = 10…100 payloads × 10 seeds, 100/100 collision-free:

| push profile | certified gap | relaxation time at M=90 |
|---|---|---|
| `gauss` (the method) | flat 11–13 % | up to ~43 s |
| `tent` (fast mode) | flat 13–16.5 % | < 1 s |

```bash
py multi_payload/studies/benchmark_tubes.py --Ms 10 20 30 40 50 60 70 80 90 100 \
   --seeds 0 1 2 3 4 5 6 7 8 9 --layout random_seq --profile gauss \
   --out benchmark_tubes_fair.json          # ~25 min; tent: --profile tent
py multi_payload/studies/plot_results.py    # regenerates both charts
```

**Other entry points**

- `benchmark_methods.py` — alternation vs inflation+tubes vs inflation+planes,
  isotropic and anisotropic (fuel-metric) pushes, M = 2–20. Holds `relax_iso`.
- `benchmark_scale100.py` — large-M comparison with a wall-clock-budgeted
  alternation baseline and a pruned one-shot separating-planes solve.
- `benchmark_pipeline.py` — the full certified pipeline per instance:
  closure-greedy allocation → deconfliction → attitude references → per-payload
  agent-level cone-constrained solves → gap vs the lower bound.
- `scp_oracle.py` — full-pose (6-DoF) SCP oracle with joint convex polish; the
  strongest known solutions at small M (certified gaps 21–30 % at M = 5).

**Exploration record** (all measured, kept so nobody re-treads them)

- `inflate_oracle.py` — original inflation relaxation with fuel-metric anisotropic
  pushes; anisotropy is a wash on random layouts; superseded by `relax_iso`.
- Push profiles `beam`, `bump<W>`, `anneal` in `relax_iso` — negative results:
  beam/bump over- or mis-spread deviation (washboard tiling); anneal (tent then
  shrink+smooth) cannot re-litigate tent's side decisions. Measured in
  `benchmark_tubes_{beam,bump6,anneal}.json`.
- `corridor_oracle.py` — spatially disjoint corridors via grid search, prism
  merging and a keep-in MILP: works, communication-free even at planning time, but
  +28 % fuel; spatial decoupling costs ~2× what time-sharing costs. Also holds
  `alternate_pp` (the iterated separating-planes baseline) and `axis_prices`.
- `occupancy_misocp.py` / `occupancy_probe.py` / `occupancy_cubes.py` —
  cell-occupancy binaries (MISOCP): the encoding is sound (touched-cell
  disjointness ⇒ ball disjointness for any closed partition) but intractable or
  geometry-rejecting in practice; the cubes synthesis was built and never run.
- `warp_oracle.py` — one global space-warp: terminates by construction but freezes
  the time schedule; up to 5× fuel.
- `route2_check.py` / `route2_multi.py` — separating-plane alternation studies and
  the (provably vacuous) McCormick relaxation of the bilinear plane constraint.
- `sandwich_check.py` — closure-greedy allocation and the bounds sandwich.
- `measure_price.py`, `authority_check.py`, `oracle_check.py` — lower-bound
  tightening diagnostics, all closed: Σℓ_j is the structural ceiling; no
  attitude-free bound can capture the cone inefficiency.

**Dependencies:** numpy, scipy, cvxpy (CLARABEL ships with it), matplotlib for the
plots. MOSEK is optional — only the MISOCP studies and the exact allocation use it.

## Comparing oracle methods

`oracles.py` defines the interface every Oracle-Problem method implements: given a
scenario and a fixed allocation (`OracleProblem`), return each payload's body-frame
agent thrusts and torque trajectory (`OracleResult`). The waypoint-repair method is
registered as `"waypoint"`. A new method implements `solve(prob)` and calls
`register(name, cls)`.

`harness.py` computes one allocation per scenario (shared by all methods) and scores
every method with the same independent verification. `benchmark/` runs grids of
scenarios × methods, one subprocess per task, resume-safe with a per-task timeout:

```bash
py multi_payload/benchmark/driver.py --out multi_payload/results/bench --M 3 4 5 6 \
   --layouts crossing random --nus 15 60 --seeds 0 1 --oracles waypoint --workers 4
py multi_payload/benchmark/analyze.py multi_payload/results/bench/results.csv --by oracle M
```

### Stress test of the waypoint-repair oracle (2026-10-16)

`results/bench_stress` (3 agents per payload) and `results/bench_scarce` (2 per
payload): M = 3–6 payloads, crossing and random layouts, ν = 15° and 60°, 2 seeds,
N = 30, C-GD with 30 s per payload, greedy allocation. 48 tasks in total.

- **Repair convergence: 48/48 converged** in 0–2 iterations, no oscillation.
- **ν = 60°, 3 agents/payload: 16/16 succeed:** collision-free (minimum separation
  ≥ 1.10 δ), terminal error ≈ 0, certified gap J_final / lower bound median 1.44.
- **Fallbacks used in real runs:** 35 corridor expansions; every feasible run that
  needed them succeeded.
- **Infeasible cases are reported, not crashed:** ν = 15° with 3 agents (12/16), and
  most 2-agent cases — the allocation cannot produce the required wrenches, so the
  Oracle's sufficiency assumption fails. These verdicts are local.

## Run

```bash
py multi_payload/scripts/run_demo.py --layout crossing --M 3 --points 6 --agents 12 --nu 60
py multi_payload/scripts/run_demo.py --layout random --M 4 --allocation misocp
py multi_payload/scripts/run_mission.py --layout crossing --M 3 --agents 12   # + docking + mission.mp4
py multi_payload/scripts/run_mission.py --out <dir> --render-only             # re-render from cache
pytest tests/test_multi_payload.py
```

`run_demo.py --help` lists all options. Each run writes `scenario.json`,
`summary.json` (allocation, repairs, fuel certificate, separation, per-payload
terminal errors) and three figures to `--out`.

Scenarios (`scenario.py`) share the chief orbit, ν, t_f and N. Per-payload mass,
candidate points and inertia follow the paper's generator. Layouts: `crossing`
places starts on a ring with goals across it; `random` draws starts and goals in a
ball with an all-pairs rejection check; `random_seq` is the dart-throwing variant
that still places reliably at high payload counts (used by the constant-density
Monte Carlos).
