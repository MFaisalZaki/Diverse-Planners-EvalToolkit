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
from typing import Any, Dict, List

from .base import COMPLETE, EXHAUSTED, TIMEOUT, UNSOLVABLE, DiversePlannerAdapter, DiverseRun, PlanningContext

_FORWARDED = ('horizon_length', 'horizon_planning_mode', 'use_pypmt')


class BehaviourPlanningSMTAdapter(DiversePlannerAdapter):
    name = 'bp-smt'
    supported_features = ('go', 'ru', 'fn', 'cb', 'uv')

    def unsupported_reason(self, ctx: PlanningContext):
        reason = super().unsupported_reason(ctx)
        if reason:
            return reason
        has_cb = any(k == 'cb' for k, _ in ctx.features)
        if ctx.q != 1.0 and not has_cb and self.params.get('implicit-cost-bound') is False:
            return 'q != 1.0 needs a cb dimension and implicit-cost-bound is false'
        return None

    def dimensions(self, ctx: PlanningContext, logs: List[str]) -> List[List[Any]]:
        dims: List[List[Any]] = []
        for key, config in ctx.features:
            if key == 'cb':
                continue
            dims.append([key, config if config not in (None, '') else {}])
        implicit = self.params.get('implicit-cost-bound')
        has_cb = any(k == 'cb' for k, _ in ctx.features)
        if has_cb or implicit is True or (implicit is None and ctx.q != 1.0):
            dims.append(['cb', {'quality-bound': ctx.q}])
            if not has_cb:
                logs.append(f'added a cb dimension to carry the quality bound q={ctx.q}')
        return dims

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        from behaviour_planning_smt.fbi.planner import ForbiddenBehaviorSMTPlanner

        logs: List[str] = []
        dims = self.dimensions(ctx, logs)
        kwargs: Dict[str, Any] = {k: self.params[k] for k in _FORWARDED if k in self.params}

        build_start = time.monotonic()
        planner = ForbiddenBehaviorSMTPlanner(ctx.problem, dims, **kwargs)
        space = planner.behaviour_space
        info: Dict[str, Any] = {
            'dimensions': [d[0] for d in dims],
            'optimal-length': getattr(space, 'optimal_plan_length', None),
            'build-seconds': time.monotonic() - build_start,
        }
        if self.params.get('solver-memory-mb'):
            planner.solver_memorylimit = int(self.params['solver-memory-mb'])
        # Every z3 check gets what is left of the budget, not a fixed 300 s.
        original_check = space.check

        def check(assumptions=(), timeout=None, memorylimit=None):
            remaining = ctx.remaining(margin=5.0)
            budget = None if remaining is None else max(1000, int(remaining * 1000))
            return original_check(list(assumptions), budget, memorylimit)

        space.check = check
        ctx.snapshot = lambda: list(planner.plan_set)

        plans = list(planner.plan(ctx.k))
        out_of_time = ctx.deadline is not None and time.monotonic() >= ctx.deadline - 6.0
        if len(plans) >= ctx.k:
            outcome = COMPLETE
        elif out_of_time:
            outcome = TIMEOUT
        else:
            outcome = EXHAUSTED if plans else UNSOLVABLE
        logs += [str(line) for line in getattr(space, 'log', [])]
        return DiverseRun(plans=plans[:ctx.k], outcome=outcome, logs=logs, info=info,
                          native_behaviours=[getattr(p, 'behaviour_str', None) for p in plans[:ctx.k]])
