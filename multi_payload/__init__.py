"""Multi-payload cooperative transport: allocation + waypoint/corridor oracle.

Implements the proposed approach of the multi-payload notes on top of the
single-payload paper physics (Tschauner-Hempel translation, SO(3) attitude,
body-frame thrust, inward thrust cones, fuel objective):

  scenario      shared-orbit multi-payload scenarios (crossing / random layouts)
  translation   Step 1: center-of-mass force references (convex), with waypoints
  conflicts     Steps 2-3: conflict detection and waypoint repair
  attitude      allocation-independent attitude references, required wrenches
  allocation    surrogate fuel gamma_j, residual e_j, greedy and MISOCP rules
  socp_corridor exact inner SOCP with a ball corridor (copy of solvers/socp_inner)
  gd_corridor   envelope-gradient COT solver with the corridor (copy of
                solvers/gradient_descent)
  oracle        Steps 4-6: corridors, per-payload 6-DoF solves, relaxation
  verify        independent rollout, pairwise separation, fuel certificate
  pipeline      end-to-end run
"""
