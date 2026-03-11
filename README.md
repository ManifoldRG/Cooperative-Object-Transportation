# Cooperative Object Transportation

Clean Python package for cooperative payload transportation with three solver paths:

- centralized NLP baseline
- centralized genetic algorithm
- decentralized parallel island genetic algorithm

## Package layout

- `cot/`: canonical library code
- `cot/solvers/`: solver implementations
- `cot/graph/`: island-agent and communication manager
- `cot/evaluation/`: metrics and method comparison
- `scripts/compare_methods.py`: runnable comparison entry point
- `tests/test_compare_methods.py`: test harness

## Run comparison

```bash
python scripts/compare_methods.py
```

## Run tests

```bash
pytest -q
```
