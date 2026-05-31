"""Delayed inter-agent communication bus.

Messages sent at step ``k`` over an edge with latency ``delay_steps`` are
delivered at step ``k + delay_steps``. Messages from comms-dead senders are
dropped (modeling an unresponsive agent). Used for the deviation-trigger
broadcast, the fault-identification round, and the post-replan winner consensus.

The bus is step-synchronous and in-process; "delay" is modeled by buffering each
message with a delivery step. Only line-of-sight neighbors (graph edges) exchange
messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class Message:
    sender: int
    kind: str
    payload: Any
    deliver_step: int


class CommsBus:
    def __init__(self, graph: nx.Graph, delay_steps: int = 1):
        self.graph = graph
        self.delay_steps = int(delay_steps)
        self._inflight: list[Message] = []

    def broadcast(self, sender: int, kind: str, payload: Any, step: int,
                  comms_alive: bool = True) -> None:
        """Queue a message from ``sender`` to all its graph neighbors.

        Dropped entirely if the sender's comms are dead (unresponsive agent).
        """
        if not comms_alive:
            return
        deliver = step + self.delay_steps
        self._inflight.append(Message(sender, kind, payload, deliver))

    def deliver(self, step: int) -> dict[int, list[Message]]:
        """Return {receiver_id: [messages]} for all messages due at or before
        ``step``, and drop them from the in-flight buffer."""
        due = [m for m in self._inflight if m.deliver_step <= step]
        self._inflight = [m for m in self._inflight if m.deliver_step > step]

        inbox: dict[int, list[Message]] = {n: [] for n in self.graph.nodes}
        for m in due:
            for nbr in self.graph.neighbors(m.sender):
                inbox[nbr].append(m)
        return inbox

    def pending(self) -> int:
        return len(self._inflight)
