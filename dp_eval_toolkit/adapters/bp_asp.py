"""Behaviour planning with ASP: ``BCoveragePlanner`` / ``BMaxSumPlanner`` from
pyBehaviourPlanningASP.

The quality bound is the space's own ``quality_bound``; the diversity model is
passed as its feature specs (``go``, ``ru``). The planner gets the remaining
budget as its own timeout, so on a timeout it returns the plans it holds and
those are recorded.

``planner-params``:

``indicator``        ``B-Coverage`` (default) or ``B-MaxSum``
``max-horizon``      iterative-deepening cap for the shortest plan (default 1000)
``compilationlist``  forwarded to the space (ASPPlanners' compilation pipeline)
``validate``         let the planner validate every plan itself (default false;
                     the runner validates every returned plan anyway)
``time-margin``      seconds kept back from the budget (default 10)
"""

from __future__ import annotations

from typing import Any

from .base import (COMPLETE, EXHAUSTED, TIMEOUT, UNSOLVABLE, UNSUPPORTED, DiversePlannerAdapter, DiverseRun,
                   PlanningContext)


class BehaviourPlanningASPAdapter(DiversePlannerAdapter):
    name = 'bp-asp'
    supported_features = ('go', 'ru')

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        from behaviour_planning_asp import PLANNERS, BehaviourSpace, UnsupportedTaskError
        from behaviour_planning_asp.planners import base as asp_base

        indicator = str(self.params.get('indicator', 'B-Coverage'))
        if indicator not in PLANNERS:
            raise ValueError(f'unknown indicator {indicator!r}; known: {sorted(PLANNERS)}')
        specs = [(key, config if config not in ({}, '') else None) for key, config in ctx.features]
        try:
            space = BehaviourSpace(ctx.problem, specs, quality_bound=ctx.q,
                                   compilationlist=self.params.get('compilationlist'),
                                   validate=bool(self.params.get('validate', False)))
        except UnsupportedTaskError as error:
            return DiverseRun(plans=[], outcome=UNSUPPORTED, logs=[str(error)])

        planner = PLANNERS[indicator](space)
        timeout = ctx.remaining(margin=float(self.params.get('time-margin', 10)))
        result = planner.plan(ctx.k, timeout=timeout, max_horizon=int(self.params.get('max-horizon', 1000)))
        outcomes = {asp_base.COMPLETE: COMPLETE, asp_base.EXHAUSTED: EXHAUSTED,
                    asp_base.UNSOLVABLE: UNSOLVABLE, asp_base.TIMED_OUT: TIMEOUT}
        info: dict[str, Any] = {
            'indicator': indicator,
            'optimal-length': result.optimal_length,
            'cost-bound': result.horizon,
            'native-b-coverage': result.b_coverage,
            'native-b-maxsum': float(result.b_maxsum(space.model)),
        }
        return DiverseRun(plans=[p.plan for p in result.plans],
                          outcome=outcomes.get(result.status, result.status.upper()),
                          logs=list(result.logs), info=info,
                          native_behaviours=[space.model.describe(p.behaviour) for p in result.plans])
