"""Behaviour planning with SMT: ``ForbiddenBehaviorSMTPlanner`` from
pyBehaviourPlanningSMT (FBI-SMT in the paper).

The planner takes the diversity model as its behaviour-space dimensions
(``go``, ``ru``, ``fn``, ``cb``, ...). Its quality bound lives *inside* the
``cb`` dimension, so a ``q`` other than 1.0 needs a ``cb`` dimension; when the
model has none, one is added for the planner (and the addition logged). The
toolkit's evaluator still scores the plans under the experiment's own model,
so this never changes what is measured -- only what the planner is told.

``planner-params``:

``implicit-cost-bound``   add ``cb`` when the model lacks it (default: only when q != 1.0;
                          ``true`` always, ``false`` never -- then q != 1.0 is unsupported)
``solver-memory-mb``      z3's own memory cap per check (default: none beyond the task limit)
``horizon_length``, ``horizon_planning_mode``, ``use_pypmt``
                          forwarded to the planner (see its README)
"""

from __future__ import annotations

import time
from typing import Any

from .base import DiversePlannerAdapter, DiverseRun, PlanningContext

_FORWARDED = ('horizon_length', 'horizon_planning_mode', 'use_pypmt')
# Seconds before the deadline at which a run that is still going counts as timed out.
_DEADLINE_SLACK = 6.0


class BehaviourPlanningSMTAdapter(DiversePlannerAdapter):
    name = 'bp-smt'
    supported_features = ('go', 'ru', 'fn', 'cb', 'uv')

    def unsupported_reason(self, ctx: PlanningContext):
        reason = super().unsupported_reason(ctx)
        if reason:
            return reason
        if ctx.q != 1.0 and not ctx.has_feature('cb') and self.params.get('implicit-cost-bound') is False:
            return 'q != 1.0 needs a cb dimension and implicit-cost-bound is false'
        return None

    def dimensions(self, ctx: PlanningContext, logs: list[str]) -> list[list[Any]]:
        """The behaviour-space dimensions: the model's features, with ``cb``
        carrying the quality bound when the model has one or the task needs one."""
        dims = [[key, config or {}] for key, config in ctx.features if key != 'cb']
        has_cb = ctx.has_feature('cb')
        implicit = self.params.get('implicit-cost-bound')
        if has_cb or implicit is True or (implicit is None and ctx.q != 1.0):
            dims.append(['cb', {'quality-bound': ctx.q}])
            if not has_cb:
                logs.append(f'added a cb dimension to carry the quality bound q={ctx.q}')
        return dims

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        from behaviour_planning_smt.fbi.planner import ForbiddenBehaviorSMTPlanner

        logs: list[str] = []
        dims = self.dimensions(ctx, logs)
        kwargs = {k: self.params[k] for k in _FORWARDED if k in self.params}

        build_start = time.monotonic()
        planner = ForbiddenBehaviorSMTPlanner(ctx.problem, dims, **kwargs)
        space = planner.behaviour_space
        info: dict[str, Any] = {
            'dimensions': [d[0] for d in dims],
            'optimal-length': getattr(space, 'optimal_plan_length', None),
            'build-seconds': time.monotonic() - build_start,
        }
        if self.params.get('solver-memory-mb'):
            planner.solver_memorylimit = int(self.params['solver-memory-mb'])
        space.check = _budgeted_check(space.check, ctx)
        ctx.snapshot = lambda: list(planner.plan_set)

        plans = list(planner.plan(ctx.k))[:ctx.k]
        out_of_time = ctx.deadline is not None and time.monotonic() >= ctx.deadline - _DEADLINE_SLACK
        logs += [str(line) for line in getattr(space, 'log', [])]
        return DiverseRun(plans=plans, outcome=self.outcome_for(len(plans), ctx.k, out_of_time), logs=logs,
                          info=info, native_behaviours=[getattr(p, 'behaviour_str', None) for p in plans])


def _budgeted_check(original_check, ctx: PlanningContext):
    """Wrap the space's ``check`` so every z3 call gets what is left of the
    task's budget, not a fixed 300 s."""
    def check(assumptions=(), timeout=None, memorylimit=None):
        remaining = ctx.remaining(margin=5.0)
        budget_ms = None if remaining is None else max(1000, int(remaining * 1000))
        return original_check(list(assumptions), budget_ms, memorylimit)
    return check
