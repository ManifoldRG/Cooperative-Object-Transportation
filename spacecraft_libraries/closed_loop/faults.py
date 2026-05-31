"""Configurable fault schedule for the recovery simulation.

A fault event takes an agent down at a trigger time. ``fault_type``:
  - "actuation": thrust produces zero wrench (dead thruster), comms still alive.
  - "comms":     outbound messages dropped (silent), thruster still works.
  - "both":      dead thruster AND silent.
"""
from __future__ import annotations

from dataclasses import dataclass

FAULT_TYPES = ("actuation", "comms", "both")


@dataclass
class FaultEvent:
    agent_id: int
    trigger_time: float
    fault_type: str = "both"

    def __post_init__(self):
        if self.fault_type not in FAULT_TYPES:
            raise ValueError(f"fault_type must be one of {FAULT_TYPES}, got {self.fault_type!r}")


class FaultModel:
    """Tracks per-agent (actuation_alive, comms_alive) as sim time advances."""

    def __init__(self, num_agents: int, events: list[FaultEvent] | None = None):
        self.num_agents = num_agents
        self.events = list(events or [])
        self._actuation_alive = [True] * num_agents
        self._comms_alive = [True] * num_agents
        self._applied: set[int] = set()

    def update(self, t: float) -> list[int]:
        """Apply any events whose trigger_time has passed. Returns the list of
        agent_ids whose health changed on this call (for logging)."""
        newly_faulted = []
        for idx, ev in enumerate(self.events):
            if idx in self._applied or t < ev.trigger_time:
                continue
            if ev.fault_type in ("actuation", "both"):
                self._actuation_alive[ev.agent_id] = False
            if ev.fault_type in ("comms", "both"):
                self._comms_alive[ev.agent_id] = False
            self._applied.add(idx)
            newly_faulted.append(ev.agent_id)
        return newly_faulted

    def actuation_alive(self, agent_id: int) -> bool:
        return self._actuation_alive[agent_id]

    def comms_alive(self, agent_id: int) -> bool:
        return self._comms_alive[agent_id]

    def actuation_mask(self) -> list[bool]:
        return list(self._actuation_alive)

    @property
    def truly_faulted(self) -> set[int]:
        """Ground-truth set of agents with any fault (for scoring the ID round)."""
        return {
            i for i in range(self.num_agents)
            if not self._actuation_alive[i] or not self._comms_alive[i]
        }
