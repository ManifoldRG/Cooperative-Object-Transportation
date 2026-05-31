"""CLI driver for the closed-loop fault-recovery simulation.

Examples:
  py scripts/run_recovery_sim.py --scenario 1 --fault 2:25.0:both
  py scripts/run_recovery_sim.py --scenario 1 --no-fault
  py scripts/run_recovery_sim.py --scenario 1 --fault 2:25:both --fault 0:40:comms
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spacecraft_libraries.closed_loop import RecoveryConfig, run_recovery_episode
from spacecraft_libraries.closed_loop.faults import FaultEvent
from spacecraft_libraries.evaluation import get_scenario


def parse_fault(spec: str) -> FaultEvent:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"fault must be 'agent_id:trigger_time:fault_type', got {spec!r}")
    return FaultEvent(agent_id=int(parts[0]), trigger_time=float(parts[1]), fault_type=parts[2])


def parse_args():
    p = argparse.ArgumentParser(description="Run a closed-loop fault-recovery episode.")
    p.add_argument("--scenario", type=int, default=1, choices=(1, 2, 3))
    p.add_argument("--fault", type=parse_fault, action="append", default=[],
                   help="Repeatable. Format 'agent_id:trigger_time:fault_type', "
                        "fault_type in {actuation,comms,both}.")
    p.add_argument("--no-fault", action="store_true", help="Run the healthy baseline.")
    p.add_argument("--mppi-iterations", type=int, default=10)
    p.add_argument("--mppi-samples", type=int, default=5)
    p.add_argument("--comms-delay-steps", type=int, default=2)
    p.add_argument("--max-recovery-cycles", type=int, default=3)
    p.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sys_params, bc, epsilon = get_scenario(args.scenario)
    faults = [] if args.no_fault else args.fault
    cfg = RecoveryConfig(
        mppi_iterations=args.mppi_iterations,
        mppi_samples=args.mppi_samples,
        comms_delay_steps=args.comms_delay_steps,
        max_recovery_cycles=args.max_recovery_cycles,
    )

    print(f"[scenario {args.scenario}] agents={len(sys_params.rs)} N={sys_params.N} tf={bc.tf}")
    print(f"[faults] {[ (f.agent_id, f.trigger_time, f.fault_type) for f in faults ] or 'none'}")
    t0 = time.perf_counter()
    result = run_recovery_episode(sys_params, bc, epsilon, fault_events=faults, cfg=cfg,
                                  verbose=not args.quiet)
    wall = time.perf_counter() - t0

    print("\n=== Episode summary ===")
    print(f"status            : {result['status']}")
    print(f"terminal_violation: {result['terminal_violation']:.4e}")
    print(f"fuel (sum||u||^2) : {result['fuel']:.4e}")
    print(f"recovery_cycles   : {result['recovery_cycles']}")
    print(f"removed_agents    : {result['removed_agents']}")
    print(f"final_active      : {result['final_active_agents']}")
    print(f"sim_time / steps  : {result['sim_time']:.2f}s / {result['steps']}")
    print(f"wall              : {wall:.1f}s")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in result.items() if k not in ("final_state",)}
        serializable["final_state"] = {
            "r": result["final_state"].r.tolist(),
            "v": result["final_state"].v.tolist(),
            "phi": result["final_state"].phi.tolist(),
            "omega": result["final_state"].omega.tolist(),
        }
        with args.output.open("w") as f:
            json.dump(serializable, f, indent=2, default=float)
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()