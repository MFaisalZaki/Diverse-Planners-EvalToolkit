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

import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

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
    features: list[tuple[str, Any]]            # the diversity model, (key, config) pairs
    deadline: float | None                     # time.monotonic() value, or None
    workdir: str
    track: str = 'classical'
    partial: list[Any] = field(default_factory=list)
    snapshot: Callable[[], Sequence[Any]] | None = None

    def remaining(self, margin: float = 0.0) -> float | None:
        """Seconds left before the deadline, less `margin`; ``None`` = no limit."""
        if self.deadline is None:
            return None
        return max(1.0, self.deadline - time.monotonic() - margin)

    def has_feature(self, key: str) -> bool:
        return any(k == key for k, _ in self.features)

    def plans_so_far(self) -> list[Any]:
        if self.snapshot is not None:
            with contextlib.suppress(Exception):
                return list(self.snapshot())
        return list(self.partial)


@dataclass
class DiverseRun:
    plans: list[Any]
    outcome: str
    logs: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    # The planner's own description of each plan's behaviour, if it has one.
    native_behaviours: list[str] | None = None


class DiversePlannerAdapter:
    """Base class. Subclasses set ``name`` and implement :meth:`plan`."""

    name = ''
    #: the diversity-model features this planner can be asked to diversify
    #: over; ``None`` means any (or: it ignores the model, like FI).
    supported_features: Sequence[str] | None = None

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})

    def unsupported_reason(self, ctx: PlanningContext) -> str | None:
        """Why this planner cannot run this task, or ``None``."""
        if self.supported_features is not None:
            missing = [k for k, _ in ctx.features if k not in self.supported_features]
            if missing:
                return f'{self.name} does not support feature(s) {missing}'
        return None

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        raise NotImplementedError

    @staticmethod
    def outcome_for(count: int, k: int, timed_out: bool = False) -> str:
        """The outcome of a run that returned `count` plans: ``COMPLETE`` with
        `k` of them, ``TIMEOUT`` when the budget ran out first, otherwise the
        planner stopped by itself -- ``EXHAUSTED`` with some plans, ``UNSOLVABLE``
        with none."""
        if count >= k:
            return COMPLETE
        if timed_out:
            return TIMEOUT
        return EXHAUSTED if count else UNSOLVABLE
