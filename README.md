# Cooperative Object Transportation

> **Project Type:** Research

This repository contains the code for the paper

> **Decentralized Bi-Level Optimization for Cooperative Orbital Transportation of a Rigid Payload**
> Sidhdharth D. Sikka, Revathy Ravuri, Hei Yin Jowett Chan, Tim Echtermeyer, Shaoshuai Mou.
> Submitted to *IEEE Transactions on Aerospace and Electronic Systems*, 2026 (under review).

The `main` branch matches the paper. It extends the earlier centralized method of
Sikka et al., *On Orbit Object Transportation With Spacecraft Swarms*, AIAA SciTech 2025.

## What it does

A swarm of $N$ spacecraft is rigidly attached to a payload and must move it from an
initial to a final position and attitude in a relative orbital (Hill) frame while
minimizing total fuel, $\sum_{i,k}\|\boldsymbol{u}_{i,k}\|$. Each agent can only push
inward, within a thrust cone of half-angle $\nu$ about its attachment direction.

The problem is non-convex because the thrusts couple to the payload's rotation. It is
split into two levels:

- **Inner problem (convex).** For a fixed body-frame torque trajectory, the attitude
  is rolled out and the minimum-fuel thrust allocation is an exact second-order cone
  program (`spacecraft_libraries/solvers/socp_inner.py`, solved with CLARABEL).
- **Outer problem.** Search over torque trajectories. The paper's method uses the
  exact envelope-theorem gradient of the inner optimal value (from the SOCP duals and
  an adjoint pass through the attitude dynamics) inside projected gradient descent
  with random restarts.

Decentralization: each agent runs its own restart stream from its own seeds, and a
max-consensus round over a line-of-sight graph shares the swarm-wide best trajectory.
A closed-loop mode detects in-flight actuation and communication faults, detumbles the
payload, removes failed agents and replans with the survivors.

## Methods

| Paper name | Code | Function |
|---|---|---|
| D-GD (ours) | `decentralized_gd` | `solvers.gradient_descent.solve_decentralized_gd` |
| C-GD (ours) | `centralized_gd` | `solvers.gradient_descent.solve_centralized_gd` |
| Direct NLP | `centralized_nlp_th` | `solvers.centralized_nlp_th.solve_centralized_nlp_th` |
| GA (prior method) | `centralized_ga` | `solvers.centralized_ga.solve_centralized_ga` |
| C-MPPI / D-MPPI | `centralized_mppi` / `decentralized_mppi` | `solvers.centralized_mppi`, `solvers.decentralized_mppi` |

`solvers/greedy_sampler.py` (GS) and `solvers/decentralized_island_ga.py` are kept for
comparison but are not part of the paper's results. `solvers/centralized_nlp.py`
(`og_opts.full_nlp`) uses different physics and is superseded by `centralized_nlp_th`.

## Installation

Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

CLARABEL is installed with cvxpy. IPOPT ships with CasADi.

## Quick start

Generate one scenario with a 60° thrust cone, then solve it with D-GD:

```bash
python scripts/csf/generate_scenarios.py --n 1 --thrust-angles-deg 60 --seed 1 --output results/quick/scenarios.json
python scripts/run_gd_local.py --scenarios results/quick/scenarios.json --output-dir results/quick/gd --time-limit 60 --decentralized
```

Each scenario writes one CSV row with the fuel cost (`cost`), the terminal violation
and the runtime. Drop `--decentralized` to run C-GD.

Run the tests (about one minute):

```bash
pytest tests
```

## Reproducing the paper

| Experiment | Entry point |
|---|---|
| Baseline comparison (all methods × cone angles × time budgets), local | `scripts/run_baseline_local.py` |
| Baseline comparison on the CSF cluster (SLURM) | `scripts/csf/baseline_comparison.py`, `scripts/csf/baseline_submit.sh` |
| Fuel-objective campaign with independent rollout verification | `scripts/baseline_fuel/driver.py` (see its docstring for `WORKERS`, `BUDGETS`, `METHODS`) |
| Tables and figures from a campaign CSV | `python scripts/baseline_fuel/analyze.py <results.csv> [outdir]` |
| Scalability in the number of agents, open and closed loop (D-GD) | `scripts/csf/run_mc_scalability_dgd_csf.py`, launched by `scripts/csf/submit_scalability_openloop.sh` and `submit_scalability_closed_loop.sh` |
| Closed-loop fault recovery | `scripts/run_recovery_sim.py` (e.g. `--scenario 1 --fault 2:25.0:both`) |
| Merging per-task CSVs from cluster runs | `scripts/csf/merge_tasks.py` |

Every campaign script re-simulates the returned controls through the discrete
dynamics and records the terminal violation, so reported fuel always belongs to a
verified trajectory. The scenarios used in the paper are in
`CSF Runs/results_20260823/baseline_large_20_scenarios_th.csv`.

## Repository layout

```
spacecraft_libraries/
  solvers/        all solvers in the table above, the SOCP inner problem and its gradient
  closed_loop/    fault injection, detection, detumble and replanning simulation
  graph/          line-of-sight communication graph and max-consensus
  evaluation/     scenario generator and comparison harness
  new_opts.py     dynamics, attitude projection and inner-problem utilities
scripts/          experiment drivers (local and CSF/SLURM)
CSF Runs/         campaign results and scenario sets
tests/            closed-loop and MPPI tests
```

## Citation

```bibtex
@inproceedings{sikka_orbit_2025,
  title     = {On Orbit Object Transportation With Spacecraft Swarms},
  author    = {Sikka, Sidhdharth D. and Lu, Zehui and Rai, Ayush and DeLaurentis, Daniel and Mou, Shaoshuai},
  booktitle = {AIAA SciTech 2025 Forum},
  year      = {2025}
}
```

A citation for the decentralized bi-level paper will be added once it is published.

## License

MIT. See [LICENSE](LICENSE).
