"""The interface every diverse planner is driven through.

A diverse planner is asked one *diversity task* -- a parsed UP problem (plus its
PDDL files), ``k``, ``q`` and a diversity model -- and returns a list of UP plans
over that problem. Everything else (limits, validation, behaviour evaluation,
result files) is the runner's job, so an adapter stays a few dozen lines.

Adapters should keep :attr:`PlanningContext.partial` up to date while they
search (or set :attr:`PlanningContext.snapshot`), so that when the task's own
time or memory limit fires the plans found so far are still recorded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Outcomes an adapter reports. The runner maps them to result statuses.
COMPLETE = 'COMPLETE'          # k plans returned
EXHAUSTED = 'EXHAUSTED'        # fewer than k plans exist within the quality bound
UNSOLVABLE = 'UNSOLVABLE'      # no plan at all
TIMEOUT = 'TIMEOUT'            # the planner's own deadline passed (partial plans kept)
MEMOUT = 'MEMOUT'
UNSUPPORTED = 'UNSUPPORTED'    # the planner cannot handle this task or model
ERROR = 'ERROR'


@dataclass
class PlanningContext:
    problem: Any                               # the parsed UP problem
    domain_file: str
    problem_file: str
    k: int
    q: float
    features: List[Tuple[str, Any]]            # the diversity model, (key, config) pairs
    deadline: Optional[float]                  # time.monotonic() value, or None
    workdir: str
    track: str = 'classical'
    partial: List[Any] = field(default_factory=list)
    snapshot: Optional[Callable[[], Sequence[Any]]] = None

    def remaining(self, margin: float = 0.0) -> Optional[float]:
        """Seconds left before the deadline, less `margin`; ``None`` = no limit."""
        if self.deadline is None:
            return None
        return max(1.0, self.deadline - time.monotonic() - margin)

    def plans_so_far(self) -> List[Any]:
        if self.snapshot is not None:
            try:
                return list(self.snapshot())
            except Exception:                                  # noqa: BLE001
                pass
        return list(self.partial)


@dataclass
class DiverseRun:
    plans: List[Any]
    outcome: str
    logs: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    # The planner's own description of each plan's behaviour, if it has one.
    native_behaviours: Optional[List[str]] = None


class DiversePlannerAdapter:
    """Base class. Subclasses set ``name`` and implement :meth:`plan`."""

    name = ''
    #: the diversity-model features this planner can be asked to diversify
    #: over; ``None`` means any (or: it ignores the model, like FI).
    supported_features: Optional[Sequence[str]] = None

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})

    def unsupported_reason(self, ctx: PlanningContext) -> Optional[str]:
        """Why this planner cannot run this task, or ``None``."""
        if self.supported_features is not None:
            missing = [k for k, _ in ctx.features if k not in self.supported_features]
            if missing:
                return f'{self.name} does not support feature(s) {missing}'
        return None

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        raise NotImplementedError

    @staticmethod
    def outcome_for(count: int, k: int, exhausted_ok: bool = True) -> str:
        if count >= k:
            return COMPLETE
        if count == 0:
            return UNSOLVABLE
        return EXHAUSTED if exhausted_ok else TIMEOUT
