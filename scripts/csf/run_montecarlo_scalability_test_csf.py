"""CLI driver for the closed-loop fault-recovery simulation.

Examples:
py run_montecarlo_scalability_test_csf.py --task-id 1 --output-dir here_test --fixed-agents-num 25 --no-fault 
py run_montecarlo_scalability_test_csf.py --task-id 2 --output-dir here_test --fixed-agents-num 25 --at-least-n-survivors 23 
"""
from pathlib import Path
import argparse
import json
import sys
import time
import random
import csv
import numpy as np
from dataclasses import asdict, is_dataclass
from pprint import pprint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spacecraft_libraries import plotters as plt
from spacecraft_libraries.dynamics import SystemParams
from spacecraft_libraries.evaluation.comparison import random_scenario_generator, comms_delay_generator
from spacecraft_libraries.solvers.decentralized_mppi import _build_line_of_sight_graph
from spacecraft_libraries.closed_loop import RecoveryConfig, run_recovery_episode
from spacecraft_libraries.closed_loop.faults import FaultEvent
from spacecraft_libraries.data_structures import BoundaryConditions
from spacecraft_libraries.evaluation import get_scenario, random_dropout_fault_generator


def parse_fault(spec: str) -> FaultEvent:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"fault must be 'agent_id:trigger_time:fault_type', got {spec!r}")
    return FaultEvent(agent_id=int(parts[0]), trigger_time=float(parts[1]), fault_type=parts[2])


def parse_args():
    p = argparse.ArgumentParser(description="Run a closed-loop fault-recovery episode.")
    p.add_argument("--scenario", type=int, default=4, choices=(1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 19, 20))
    p.add_argument("--fault", type=parse_fault, action="append", default=[],
                   help="Repeatable. Format 'agent_id:trigger_time:fault_type', "
                        "fault_type in {actuation,comms,both}.")
    p.add_argument("--no-fault", action="store_true", help="Run the healthy baseline.")
    p.add_argument("--at-least-n-survivors", type=int, default=4)
    p.add_argument("--fault-model", type=str, choices=("random", "localized"), help="fault_model in {random,localized}", default="random")
    p.add_argument("--fault-type", type=str, choices=("actuation", "comms", "both"), help="fault_type in {actuation,comms,both}", default="actuation")
    p.add_argument("--mppi-iterations", type=int, default=20)
    p.add_argument("--mppi-samples", type=int, default=10)
    p.add_argument("--mppi-sigma", type=float, default=0.8)
    p.add_argument("--comms-delay-steps", type=int, default=100)
    p.add_argument("--random-extra-comms-delay-steps", type=int, default=0)
    p.add_argument("--max-recovery-cycles", type=int, default=5)
    p.add_argument("--graph-degree", type=int, default=3)
    p.add_argument("--quiet", action="store_false")
    p.add_argument("--fixed-agents-num", type=int, default=4)
    p.add_argument("--task-id",    type=int,  required=True)
    p.add_argument("--seed",   type=int,  default=42)
    p.add_argument("--thrust_angle",   type=float,  default=np.pi / 2.0)
    p.add_argument("--output-dir", type=Path, default=Path("results/tasks"))
    
    return p.parse_args()

def format_fault_event_result(fault_events: list[FaultEvent], result: dict) -> str:
    if not fault_events:
        fault_str = "none"
    else:
        fault_str = ";".join(
            f"agent={f.agent_id};t={f.trigger_time:.3f};type={f.fault_type}"
            for f in fault_events
        )

    removed_agents = result.get("removed_agents", [])
    final_active = result.get("final_active_agents", [])
    status = result.get("status", None)

    return (
        f"faults={fault_str};"
        f"removed_agents={removed_agents};"
        f"final_active={final_active};"
        f"status={status}"
    )

def format_scenario(sys_params, bc, epsilon):
    """Serialize scenario parameters, accepting both lists and NumPy arrays."""

    def to_list(value):
        return np.asarray(value, dtype=float).tolist()

    return (
        f"N={sys_params.N};"
        f"n_agents={len(sys_params.rs)};"
        f"a={sys_params.a};"
        f"e={sys_params.e};"
        f"m={sys_params.m};"
        f"tf={bc.tf};"
        f"epsilon={epsilon};"
        f"I={json.dumps(to_list(sys_params.I))};"
        f"rs={json.dumps(to_list(sys_params.rs))};"
        f"xf_r={json.dumps(to_list(bc.xf.r))};"
        f"xf_phi={json.dumps(to_list(bc.xf.phi))}"
    )

def to_printable(obj):
    """Convert dataclasses / numpy arrays / custom objects into readable dicts."""
    if is_dataclass(obj):
        return to_printable(asdict(obj))

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (list, tuple)):
        return [to_printable(x) for x in obj]

    if isinstance(obj, dict):
        return {k: to_printable(v) for k, v in obj.items()}

    # For custom classes like FaultEvent if they are not dataclasses
    if hasattr(obj, "__dict__"):
        return {
            k: to_printable(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    return obj

def print_all_scenarios_faults_commsmap(all_scenarios: list[tuple[SystemParams, BoundaryConditions, float]], 
                                        all_fault_events: list[list[list[FaultEvent]]],
                                        all_commdelay_maps: list[dict[int,int]],
                                        cfg: RecoveryConfig ):
    for run_id, ((sys_params, bc, epsilon), fevent, commdelay_map) in enumerate(
        zip(all_scenarios, all_fault_events, all_commdelay_maps),
        start=0,
    ):
        print("\n" + "=" * 90)
        print(f"[run {run_id}] FULL SCENARIO")
        print("=" * 90)
        print("\n[sys_params]")
        pprint(to_printable(sys_params), width=140, sort_dicts=False)
        print("\n[boundary_conditions]")
        pprint(to_printable(bc), width=140, sort_dicts=False)
        print("\n[epsilon]")
        pprint(to_printable(epsilon), width=140, sort_dicts=False)
        print("\n[recovery_config]")
        pprint(to_printable(cfg), width=140, sort_dicts=False)
        print(
            f"[faults] "
            f"{[(f.agent_id, f.trigger_time, f.fault_type) for f in fevent] or 'none'}"
        )
        print(
            f"[delay] "
            f"{[(aid, delay) for aid, delay in commdelay_map.items()]}"
        )
        print()
def main():

    # for writing
    fieldnames = [
        "run_id", "time_limit_s", "method",
        "cost", "terminal_violation", "runtime_s", "simtime_s",
        "n_agents", "a", "e", "m", "tf", "epsilon_tol",
        "agent_commdelay", "faults", "scenario",
    ]

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = RecoveryConfig(
        mppi_iterations=args.mppi_iterations,
        mppi_samples=args.mppi_samples,
        max_recovery_cycles=args.max_recovery_cycles,
        mppi_sigma=args.mppi_sigma,
        graph_degree=args.graph_degree
    )

    # get all fault events from random generator
    num_of_events = 1
    all_scenarios = [ random_scenario_generator(fixed_agent_num=args.fixed_agents_num,
                                                seed=args.seed,
                                                thrust_angle=args.thrust_angle) for i in range(num_of_events)]
    if args.no_fault:
        all_fault_events = [ [] for i in range(num_of_events)]
    else:
        all_fault_events = [random_dropout_fault_generator( sys_params,
                                                          bc.tf,
                                                          num_of_events,
                                                          args.fault_model,
                                                          args.fault_type,
                                                          at_least_n_survivors=args.at_least_n_survivors,
                                                          affected_radius=1.0,
                                                          trigger_time=0 )[0] for (sys_params,bc,_) in all_scenarios]
    all_commdelay_maps = [comms_delay_generator(sys_params, "fixed", args.comms_delay_steps, args.random_extra_comms_delay_steps) for (sys_params,_,_) in all_scenarios]

    # print out scenarios, faults and communicaiton maps before running
    print_all_scenarios_faults_commsmap(all_scenarios, all_fault_events, all_commdelay_maps, cfg)

    # start writing file
    out = args.output_dir / f"task_{args.task_id:04d}_Nagents_{args.fixed_agents_num}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # start running all fault events
        for run_id, ((sys_params, bc, epsilon), fevent, commdelay_map) in enumerate( zip(all_scenarios, all_fault_events, all_commdelay_maps), start=0):

            cfg.agents_comms_delay_step_map = commdelay_map
            print(f"[scenario {args.scenario}] " f"agents={len(sys_params.rs)} N={sys_params.N} tf={bc.tf:.2f}" )
            print(f"[faults] {[ (f.agent_id, f.trigger_time, f.fault_type) for f in fevent ] or 'none'}")
            print(f"[delay] { [(aid, delay) for aid, delay in commdelay_map.items()]}")
            t0 = time.perf_counter()
            result = run_recovery_episode(sys_params, bc, epsilon, fault_events=fevent, cfg=cfg, verbose=True )
            wall = time.perf_counter() - t0

            row = { "run_id": run_id,
                    "time_limit_s": getattr(args, "time_limit", None),
                    "method": "MPPI-Recovery",
                    "cost": result["fuel"],
                    "terminal_violation": result["terminal_violation"],
                    "runtime_s": wall,
                    "simtime_s": result["sim_time"],
                    "n_agents": len(sys_params.rs),
                    "a": sys_params.a,
                    "e": sys_params.e,
                    "m": sys_params.m,
                    "tf": bc.tf,
                    "epsilon_tol": epsilon,
                    "agent_commdelay": ";".join(f"agent={aid}:delay={delay}" for aid, delay in cfg.agents_comms_delay_step_map.items()),
                    "faults": format_fault_event_result(fevent, result),
                    "scenario": format_scenario(sys_params, bc, epsilon),
            }
            writer.writerow(row)
            f.flush()

            print("\n=== Episode summary ===")
            print(f"status            : {result['status']}")
            print(f"terminal_violation: {result['terminal_violation']:.4e}")
            print(f"fuel (sum||u||^2) : {result['fuel']:.4e}")
            print(f"recovery_cycles   : {result['recovery_cycles']}")
            print(f"removed_agents    : {result['removed_agents']}")
            print(f"final_active      : {result['final_active_agents']}")
            print(f"sim_time / steps  : {result['sim_time']:.2f}s / {result['steps']}")
            print(f"wall              : {wall:.1f}s")

if __name__ == "__main__":
    main()