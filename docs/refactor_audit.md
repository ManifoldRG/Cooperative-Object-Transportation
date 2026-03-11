# Repository audit (pre-refactor)

## Notebooks inspected

- `Search_Based_OCOOT.ipynb`
  - Reusable logic found: orbital helpers, quaternion/SO(3) dynamics helpers, centralized NLP (`full_nlp`, `opt_given_tau_ipopt`), GA helpers (`pop_gen`, `fitness_func`, `genetic_algorithm_optimization`), forward rollout, plotting helpers.
  - Overlap: substantial duplication with `spacecraft_libraries/orbital_helpers.py`, `spacecraft_libraries/dynamics.py`, `spacecraft_libraries/og_opts.py`, `spacecraft_libraries/genetic_code.py`, `spacecraft_libraries/plotters.py`.
- `PIM_COT.ipynb`
  - Reusable logic found: `SpaceAgent`, `GraphSpawner` (parallel island migration + consensus), Manim animation scene class.
  - Overlap: island-GA pieces called into `spacecraft_libraries/genetic_code.py` and `spacecraft_libraries/new_opts.py`.

## Existing Python modules (pre-refactor)

- `spacecraft_libraries/data_structures.py`: canonical dataclasses for system/scenario/trajectory.
- `spacecraft_libraries/dynamics.py`: rotational + translational helpers and rollout.
- `spacecraft_libraries/orbital_helpers.py`: orbital parameter helpers.
- `spacecraft_libraries/og_opts.py`: centralized baseline NLP and older optimization loops.
- `spacecraft_libraries/new_opts.py`: newer SO(3)-based projection + optimization path.
- `spacecraft_libraries/genetic_code.py`: GA population generation and fitness evaluation.
- `spacecraft_libraries/plotters.py`: plotting functions duplicated from notebook.
- `spacecraft_libraries/agent_class.py`: placeholder file, superseded by notebook classes.
- `spacecraft_libraries/optimisers.py`: cleaned variants overlapping with `og_opts.py`/`new_opts.py`.

## Consolidation decisions

- Promoted notebook-only island-GA classes into package code under `spacecraft_libraries/graph`.
- Added explicit solver modules for the three requested paths under `spacecraft_libraries/solvers`.
- Added evaluation/comparison harness under `spacecraft_libraries/evaluation` and `scripts/compare_methods.py`.
- Reduced notebooks to non-essential demo cells; core logic now lives in `.py` modules.
- Kept `og_opts.py` and `new_opts.py` as backend implementations because solver internals still rely on them.
