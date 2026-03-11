# Cooperative Object Transportation
> **Project Type:** Research

##  About • Goal • Vision
Implementation of Parallel Island Model Genetic Algorithm for $N$ Spacecraft agents to cooperatively manipulate a Rigid Payload from a starting position to a final state in a decentralised manner while minimising total costs

**Goal**: Implement the algorithm and decentralised control process 

**Vision**: Provide foundation for decentralised cooperative object manipulation in space.

## Project Roadmap
```
1. Implement the agent and payload dynamics
1. Animate the motion of the agent and the payload for better visualisation
2. Implement inner control optimisation code
2. Test with Deterministic cases and known scenarios
3. Implement Parallel Island Model
3. Write blog post about the implementation
4. Implement the communication layer and consensus
5. Compare with prior centralised approach and NLP benchmarks
6. Perform parametric sensitivity studies
7. Provide Analytical guarantee of Genetic Diversity
```

How to Contribute
Contributor workflow

```markdown
1. Fork → `feat/` branch  
2. Pass tests: `pytest tests/`  
3. Update [CONTRIBUTING.md](CONTRIBUTING.md)
📜 License & Attribution
(Mandatory section)
Citation:

bibtex
@software{Manifold_UrbanGeo_2025,
  author = {Manifold Research Group},
  title = {{ProjectName}: Geospatial analysis toolkit},
  url = {https://github.com/ManifoldRG/project},
  version = {0.1.0},
  year = {2025}
}
License: MIT
Acknowledgements: NSF Award #203445 • City of Seattle Open Data Portal


## Refactored package entry points

- Comparison runner: `python scripts/compare_methods.py`
- Core solver modules:
  - `spacecraft_libraries/solvers/centralized_nlp.py`
  - `spacecraft_libraries/solvers/centralized_ga.py`
  - `spacecraft_libraries/solvers/decentralized_island_ga.py`
- Decentralized island model graph logic:
  - `spacecraft_libraries/graph/agent.py`
  - `spacecraft_libraries/graph/graph_manager.py`
