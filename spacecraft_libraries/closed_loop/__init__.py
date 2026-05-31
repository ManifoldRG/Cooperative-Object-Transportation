"""Closed-loop fault-recovery layer over the open-loop cooperative-transport planners.

Adds a forward payload simulator (matching the planner's TH + SO(3) dynamics), a
configurable fault schedule (actuation / comms dropout), a delayed comms bus, a
per-agent state machine (track -> detumble -> identify -> replan), and an
orchestrator that runs a full recovery episode.
"""
from .config import RecoveryConfig
from .faults import FaultEvent, FaultModel
from .comms import CommsBus
from .simulator import PayloadSimulator
from .cone import project_into_cone
from .recovery_sim import run_recovery_episode

__all__ = [
    "RecoveryConfig",
    "FaultEvent",
    "FaultModel",
    "CommsBus",
    "PayloadSimulator",
    "project_into_cone",
    "run_recovery_episode",
]
