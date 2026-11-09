"""Any UP engine that enumerates plans through the ``AnytimePlanner`` interface
(e.g. ``symk``), taking the first ``k`` distinct plans it returns.

``planner-params``:

``engine``     the UP engine name (required)
``params``     engine parameters; the strings ``"{k}"`` and ``"{q}"`` anywhere in a
               value are replaced by this task's k and q, and ``"{timeout}"`` by the
               remaining seconds
``modules``    modules to import first (plugins that self-register on import)

The engine is not told about the diversity model; as with FI, the plans are
scored by the toolkit's evaluator.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import DiversePlannerAdapter, DiverseRun, PlanningContext


class UPAnytimeAdapter(DiversePlannerAdapter):
    name = 'up-anytime'
    supported_features = None

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        import unified_planning.shortcuts as ups

        engine = self.params.get('engine')
        if not engine:
            raise ValueError('up-anytime needs "engine" in planner-params')
        for module in self.params.get('modules', []):
            importlib.import_module(module)
        remaining = ctx.remaining(margin=5.0)
        params = _substitute(self.params.get('params') or {}, ctx.k, ctx.q, remaining)

        plans: list[Any] = []
        ctx.partial = plans                     # recovered if the runner's limit fires
        seen: set[str] = set()
        timed_out = False
        with ups.AnytimePlanner(name=engine, params=params) as planner:
            for result in planner.get_solutions(ctx.problem, timeout=remaining):
                if result.plan is not None and str(result.plan) not in seen:
                    seen.add(str(result.plan))
                    plans.append(result.plan)
                timed_out = timed_out or result.status.name == 'TIMEOUT'
                if len(plans) >= ctx.k:
                    break
        return DiverseRun(plans=plans[:ctx.k], outcome=self.outcome_for(len(plans), ctx.k, timed_out),
                          info={'engine': engine})


def _substitute(value, k: int, q: float, timeout) -> Any:
    """Replace ``{k}``, ``{q}`` and ``{timeout}`` throughout `value`; a string
    that is exactly one placeholder becomes the typed value."""
    if isinstance(value, dict):
        return {key: _substitute(v, k, q, timeout) for key, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, k, q, timeout) for v in value]
    if not isinstance(value, str):
        return value
    placeholders = {'{k}': k, '{q}': q, '{timeout}': int(timeout or 0)}
    if value in placeholders:
        return placeholders[value]
    for placeholder, replacement in placeholders.items():
        value = value.replace(placeholder, str(replacement))
    return value
