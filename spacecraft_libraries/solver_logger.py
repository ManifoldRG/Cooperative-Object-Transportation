"""
solver_logger.py
----------------
Thread-safe logging for two events:
  1. NLP solver failures  (status != "Solve_Succeeded" / "feasible")
  2. Every call to opt_given_tau_ipopt_new, including runtime

Log files
---------
solver_failures_<timestamp>.log  – one entry per NLP failure
opt_given_tau_<timestamp>.log    – one entry per opt_given_tau call

Both files sit under logs/solver_logs/ next to this module and are appended
to on subsequent runs so data accumulates across experiments.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

_RUN_TIMESTAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_LOG_DIR = Path(__file__).parent / "logs" / "solver_logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_FAILURE_LOG = _LOG_DIR / f"solver_failures_{_RUN_TIMESTAMP}.log"
_TIMING_LOG  = _LOG_DIR / f"opt_given_tau_{_RUN_TIMESTAMP}.log"

_lock = threading.Lock()
_ctx  = threading.local()


def set_scenario_context(
    run_id: int | None = None,
    method: str | None = None,
    n_agents: int | None = None,
    time_limit_s: float | None = None,
    a: float | None = None,
    e: float | None = None,
    m: float | None = None,
    tf: float | None = None,
    epsilon: float | None = None,
) -> None:
    """Tag the current thread with scenario metadata for all subsequent log lines."""
    _ctx.run_id       = run_id
    _ctx.method       = method
    _ctx.n_agents     = n_agents
    _ctx.time_limit_s = time_limit_s
    _ctx.a            = a
    _ctx.e            = e
    _ctx.m            = m
    _ctx.tf           = tf
    _ctx.epsilon      = epsilon


def clear_scenario_context() -> None:
    for attr in ("run_id", "method", "n_agents", "time_limit_s", "a", "e", "m", "tf", "epsilon"):
        _ctx.__dict__.pop(attr, None)


def _scenario_ctx_str() -> str:
    parts = []
    for attr in ("run_id", "method", "n_agents", "time_limit_s", "a", "e", "m", "tf", "epsilon"):
        val = getattr(_ctx, attr, None)
        if val is not None:
            parts.append(f"{attr}={val}")
    return (" | " + " ".join(parts)) if parts else ""


def _make_logger(name: str, filepath: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filepath, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _failure_logger() -> logging.Logger:
    return _make_logger("solver_failures", _FAILURE_LOG)


def _timing_logger() -> logging.Logger:
    return _make_logger("opt_given_tau", _TIMING_LOG)


def log_nlp_failure(
    function_name: str,
    solver_status: str,
    context: Optional[dict] = None,
) -> None:
    ctx_str = (" | " + " ".join(f"{k}={v}" for k, v in context.items())) if context else ""
    msg = f"FAILURE | fn={function_name} | status={solver_status}{ctx_str}{_scenario_ctx_str()}"
    with _lock:
        _failure_logger().warning(msg)


def log_opt_given_tau_call(
    function_name: str,
    runtime_s: float,
    solver_status: str,
    cost: Optional[float] = None,
    context: Optional[dict] = None,
) -> None:
    cost_str = f"{cost:.6g}" if cost is not None else "N/A"
    ctx_str = (" | " + " ".join(f"{k}={v}" for k, v in context.items())) if context else ""
    msg = (
        f"CALL | fn={function_name} | runtime_s={runtime_s:.4f} "
        f"| status={solver_status} | cost={cost_str}{ctx_str}{_scenario_ctx_str()}"
    )
    with _lock:
        _timing_logger().info(msg)