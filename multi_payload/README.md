# Multi-payload cooperative transport

Extension of the paper code to **M rigid payloads moved by one swarm of N identical
agents**. Agents choose which open attachment points to occupy (the *Allocation
Problem*), then every payload is moved to its goal without coming within a safety
distance Î´ of another payload, minimizing total fuel (the *Oracle Problem*).

Everything uses the paper's physics unchanged: Tschaunerâ€“Hempel relative orbital
dynamics on a shared chief orbit, SO(3) attitude, body-frame thrust with inward
cones of half-angle Î½, fuel Î£â€–uâ€–, and independent rollout verification.


> **Note (2026-10):** waypoint repair (steps 2-3 below) has been superseded as the reference generator by the *inflation relaxation* (simultaneous sphere-inflation deconfliction); that method, its fast *tent* variant, and all scaling benchmarks live in `studies/multi_payload_oracle/` (see its README). This package remains the model layer they build on.

## Method

**Oracle Problem (allocation given)**

1. **Translational references** (`translation.py`). Translation depends on the agents
   only through the net force at the center of mass, so each payload gets a convex
   fuel-optimal reference. These don't depend on the allocation or the attitude, and
   their total fuel is a lower bound on the optimum.
2. **Conflict detection** (`conflicts.py`). Pairs closer than
   Î´ + Îµ_j + Îµ_l + (Î”t/2)â€–Î”vâ€– at any step.
3. **Waypoint repair** (`conflicts.py`). The worst conflict is pushed apart. Candidates
   are 5 push directions (the separation axis and four 45Â° tilts) Ã— 3 splits (move j,
   move l, half each), each pushed exactly far enough; the cheapest by exact re-solve
   of step 1 is applied, re-solving only the affected payloads. Recurring conflicts get
   a geometrically growing push margin (anti-oscillation); convergence, iterations and
   recurrences are reported.
4. **Ball corridors** (`oracle.py`). â€–r_k âˆ’ rÌ„_kâ€– â‰¤ Îµ_k decouples the payloads.
5. **Single-payload 6-DoF solves** (`gd_corridor.py`, `socp_corridor.py`). Modified
   copies of the paper's COT solver: the corridor sits in the convex inner SOCP and the
   envelope gradient is unchanged. C-GD by default, D-GD with `--step5 dgd`.
6. **Corridor relaxation** (`oracle.py`). Elastic slack; nonzero slack widens the
   corridor within its neighbors' room, then falls back to re-repairing the references.

If every payload's agents have wrench closure, radius-0 corridors are always
feasible and step 6 never triggers.

**Allocation Problem** (`attitude.py`, `allocation.py`)

The references above plus an allocation-independent attitude reference (the paper's
shooting warm start, projected) fix the wrench each payload needs at every step. The
surrogate cost of an allocation is the minimum fuel to produce those wrenches, an
upper bound on the oracle value. Rule: minimize it if some allocation makes every
payload feasible, otherwise minimize the worst normalized wrench residual. Solved by
**sequential greedy** or exactly by a **mixed-integer SOCP** (`--allocation misocp`,
needs MOSEK).

**Certificate:** lower bound â‰¤ J* â‰¤ J_final â‰¤ surrogate, all reported.

**Objective.** `--objective fuel` (Î£â€–uâ€–, the paper's, default) or `--objective energy`
(Î£â€–uâ€–Â², continuous thrust). It is used consistently by every stage (references,
repair scoring, allocation, corridor SOCP, COT solver, docking MPC), and the
certificate is reported in its units. For energy, the lower bound is
Î£_j â„“_j / A_j (Cauchyâ€“Schwarz over the A_j agents of payload j).

**Docking** (`docking.py`)

Agents start free-flying and dock before transport begins. Docking runs on
t âˆˆ [âˆ’T_dock, 0] and ends exactly at each payload's transport start state; during
it the payloads drift uncontrolled (propagated backward through the same dynamics).
Agents are matched to the allocated points by the Hungarian method on docking fuel,
then each runs shrinking-horizon fuel-optimal MPC in the style of Park, Di Cairano
and Kolmanovsky (ACC 2011): a line-of-sight cone from the docking port (vertex
moved slightly inside the payload), a tangent half-plane, an approach-speed limit
and soft docking (velocity matched to the port), with a thrust limit. Constraints
are softened with a large penalty so the closed loop stays feasible under thrust
execution noise. Agents do not avoid each other during docking; the minimum
agentâ€“agent distance is reported.

**Animation** (`animate.py`). PyVista render of the whole mission (allocation,
docking, transport) in the style of the eight-agent demo, driven by the computed
trajectories, with a camera that smoothly zooms to frame the active objects.
Geometry is to scale.

## Comparing oracle methods

`oracles.py` defines the interface every Oracle-Problem method implements: given a
scenario and a fixed allocation (`OracleProblem`), return each payload's body-frame
agent thrusts and torque trajectory (`OracleResult`). The waypoint-repair method is
registered as `"waypoint"`. A new method (e.g. the fused GCS + gate-scheduling
planner) implements `solve(prob)` and calls `register("gcs", GCSOracle)`.

`harness.py` computes one allocation per scenario (shared by all methods) and scores
every method with the same independent verification. `benchmark/` runs grids of
scenarios Ã— methods, one subprocess per task, resume-safe with a per-task timeout:

```bash
py multi_payload/benchmark/driver.py --out multi_payload/results/bench --M 3 4 5 6 \
   --layouts crossing random --nus 15 60 --seeds 0 1 --oracles waypoint gcs --workers 4
py multi_payload/benchmark/analyze.py multi_payload/results/bench/results.csv --by oracle M
```

The summary reports per method: success (collision-free and terminal state reached),
repair convergence, iterations and recurrences, corridor expansions and re-repairs,
minimum separation / Î´, terminal error, certified gap J_final / lower bound,
certificate, and runtime.

### Stress test of the waypoint-repair oracle (2026-10-16)

`results/bench_stress` (3 agents per payload) and `results/bench_scarce` (2 per
payload): M = 3â€“6 payloads, crossing and random layouts, Î½ = 15Â° and 60Â°, 2 seeds,
N = 30, C-GD with 30 s per payload, greedy allocation. 48 tasks in total.

- **Repair convergence (the known risk): 48/48 converged** in 0â€“2 iterations, with
  no recurring conflicts and no oscillation.
- **Î½ = 60Â°, 3 agents/payload: 16/16 succeed:** collision-free (minimum separation
  â‰¥ 1.10 Î´), terminal error â‰ˆ 0, certified gap J_final / lower bound median 1.44.
- **Fallbacks used in real runs:** 35 corridor expansions; every run that needed them
  and was feasible succeeded.
- **Infeasible cases are reported, not crashed:** Î½ = 15Â° with 3 agents (12/16), and
  most 2-agent cases. Here the allocation cannot produce the required wrenches
  (allocation residual > 0), so the Oracle's sufficiency assumption fails. These
  verdicts are local, since the COT torque search is non-convex.
- The re-repair fallback (Step 6 back to Step 3) was not needed in any real run;
  it is covered by a unit test with a stubbed solver.

## Run

```bash
py multi_payload/scripts/run_demo.py --layout crossing --M 3 --points 6 --agents 12 --nu 60
py multi_payload/scripts/run_demo.py --layout random --M 4 --allocation misocp
py multi_payload/scripts/run_mission.py --layout crossing --M 3 --agents 12   # + docking + mission.mp4
py multi_payload/scripts/run_mission.py --out <dir> --render-only             # re-render from cache
pytest tests/test_multi_payload.py
```

`run_demo.py --help` lists all options (time steps, horizon, budget, corridor radius,
D-GD). Each run writes `scenario.json`, `summary.json` (allocation, repairs, fuel
certificate, separation, per-payload terminal errors) and three figures to `--out`.

Scenarios (`scenario.py`) share the chief orbit, Î½, t_f and N. Per-payload mass,
candidate points and inertia follow the paper's generator. `crossing` places starts
on a ring with goals across it, so paths cross; `random` draws starts and goals in a
ball.

