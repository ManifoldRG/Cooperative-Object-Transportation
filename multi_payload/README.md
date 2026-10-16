# Multi-payload cooperative transport

Extension of the paper code to **M rigid payloads moved by one swarm of N identical
agents**. Agents choose which open attachment points to occupy (the *Allocation
Problem*), then every payload is moved to its goal without coming within a safety
distance δ of another payload, minimizing total fuel (the *Oracle Problem*).

Everything uses the paper's physics unchanged: Tschauner–Hempel relative orbital
dynamics on a shared chief orbit, SO(3) attitude, body-frame thrust with inward
cones of half-angle ν, fuel Σ‖u‖, and independent rollout verification.

## Method

**Oracle Problem (allocation given)**

1. **Translational references** (`translation.py`). Translation depends on the agents
   only through the net force at the center of mass, so each payload gets a convex
   fuel-optimal reference. These don't depend on the allocation or the attitude, and
   their total fuel is a lower bound on the optimum.
2. **Conflict detection** (`conflicts.py`). Pairs closer than
   δ + ε_j + ε_l + (Δt/2)‖Δv‖ at any step.
3. **Waypoint repair** (`conflicts.py`). The worst conflict is pushed apart (move j,
   move l, or split), each option scored by re-solving step 1 with the displaced
   position as a waypoint; only the affected payloads are re-solved.
4. **Ball corridors** (`oracle.py`). ‖r_k − r̄_k‖ ≤ ε_k decouples the payloads.
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

**Certificate:** lower bound ≤ J* ≤ J_final ≤ surrogate, all reported.

**Docking** (`docking.py`)

Agents start free-flying and dock before transport begins. Docking runs on
t ∈ [−T_dock, 0] and ends exactly at each payload's transport start state; during
it the payloads drift uncontrolled (propagated backward through the same dynamics).
Agents are matched to the allocated points by the Hungarian method on docking fuel,
then each runs shrinking-horizon fuel-optimal MPC in the style of Park, Di Cairano
and Kolmanovsky (ACC 2011): a line-of-sight cone from the docking port (vertex
moved slightly inside the payload), a tangent half-plane, an approach-speed limit
and soft docking (velocity matched to the port), with a thrust limit. Constraints
are softened with a large penalty so the closed loop stays feasible under thrust
execution noise. Agents do not avoid each other during docking; the minimum
agent–agent distance is reported.

**Animation** (`animate.py`). PyVista render of the whole mission (allocation,
docking, transport) in the style of the eight-agent demo, driven by the computed
trajectories, with a camera that smoothly zooms to frame the active objects.
Geometry is to scale.

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

Scenarios (`scenario.py`) share the chief orbit, ν, t_f and N. Per-payload mass,
candidate points and inertia follow the paper's generator. `crossing` places starts
on a ring with goals across it, so paths cross; `random` draws starts and goals in a
ball.
