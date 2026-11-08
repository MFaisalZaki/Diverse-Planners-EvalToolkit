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
from typing import Any, Dict, List

from .base import COMPLETE, EXHAUSTED, TIMEOUT, UNSOLVABLE, DiversePlannerAdapter, DiverseRun, PlanningContext


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

        plans: List[Any] = []
        seen = set()
        ctx.partial = plans
        timed_out = False
        with ups.AnytimePlanner(name=engine, params=params) as planner:
            for result in planner.get_solutions(ctx.problem, timeout=remaining):
                if result.plan is not None:
                    key = str(result.plan)
                    if key not in seen:
                        seen.add(key)
                        plans.append(result.plan)
                if result.status.name == 'TIMEOUT':
                    timed_out = True
                if len(plans) >= ctx.k:
                    break
        if len(plans) >= ctx.k:
            outcome = COMPLETE
        elif timed_out:
            outcome = TIMEOUT
        else:
            outcome = EXHAUSTED if plans else UNSOLVABLE
        return DiverseRun(plans=plans[:ctx.k], outcome=outcome, info={'engine': engine})


def _substitute(value, k: int, q: float, timeout) -> Any:
    if isinstance(value, dict):
        return {key: _substitute(v, k, q, timeout) for key, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, k, q, timeout) for v in value]
    if isinstance(value, str):
        if value == '{k}':
            return k
        if value == '{q}':
            return q
        if value == '{timeout}':
            return int(timeout or 0)
        return value.replace('{k}', str(k)).replace('{q}', str(q)).replace('{timeout}', str(int(timeout or 0)))
    return value
