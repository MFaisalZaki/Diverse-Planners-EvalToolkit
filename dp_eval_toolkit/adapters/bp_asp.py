"""Behaviour planning with ASP: ``BCoveragePlanner`` / ``BMaxSumPlanner`` from
pyBehaviourPlanningASP.

The quality bound is the space's own ``quality_bound``; the diversity model is
passed as its feature specs (``go``, ``ru``). The planner gets the remaining
budget as its own timeout, so on a timeout it returns the plans it holds and
those are recorded.

``planner-params``:

``indicator``        ``B-Coverage`` (default) or ``B-MaxSum``
``naive``            plan over an empty behaviour space instead of the task's
                     model (the library's top-k version): every plan has the one
                     empty behaviour, so the planner returns the first k distinct
                     plans within the cost bound (default false)
``select``           pick k plans from the planner's pool by ``B-Coverage``,
``pool-size``        ``B-MaxSum``, ``B-MaxMin`` or ``B-Novelty`` under the task's
                     model, and how many plans to ask for (a count or ``"3x"``);
                     see :class:`~.base.DiversePlannerAdapter`. This is how the
                     naive baseline gets its B-Coverage and B-MaxSum versions.
``max-horizon``      iterative-deepening cap for the shortest plan (default 1000)
``compilationlist``  forwarded to the space (ASPPlanners' compilation pipeline)
``validate``         let the planner validate every plan itself (default false;
                     the runner validates every returned plan anyway)
``time-margin``      seconds kept back from the budget (default 10)
"""

from __future__ import annotations

from typing import Any

from .base import UNSUPPORTED, DiversePlannerAdapter, DiverseRun, PlanningContext


class BehaviourPlanningASPAdapter(DiversePlannerAdapter):
    name = 'bp-asp'
    supported_features = ('go', 'ru')

    @property
    def naive(self) -> bool:
        return bool(self.params.get('naive', False))

    def unsupported_reason(self, ctx: PlanningContext) -> str | None:
        # The naive baseline never looks at the model, so any model will do.
        return None if self.naive else super().unsupported_reason(ctx)

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        from behaviour_planning_asp import PLANNERS, BehaviourSpace, UnsupportedTaskError
        from behaviour_planning_asp.planners import base as asp_base

        indicator = str(self.params.get('indicator', 'B-Coverage'))
        if indicator not in PLANNERS:
            raise ValueError(f'unknown indicator {indicator!r}; known: {sorted(PLANNERS)}')
        try:
            space = BehaviourSpace(ctx.problem, self._specs(ctx), quality_bound=ctx.q,
                                   compilationlist=self.params.get('compilationlist'),
                                   validate=bool(self.params.get('validate', False)))
        except UnsupportedTaskError as error:
            return DiverseRun(plans=[], outcome=UNSUPPORTED, logs=[str(error)])

        planner = PLANNERS[indicator](space)
        timeout = ctx.remaining(margin=float(self.params.get('time-margin', 10)))
        result = planner.plan(self.pool_size(ctx.k), timeout=timeout,
                              max_horizon=int(self.params.get('max-horizon', 1000)))
        logs = list(result.logs)
        pool = [p.plan for p in result.plans]
        plans = self.select(ctx, pool, logs)
        info: dict[str, Any] = {
            'indicator': indicator,
            'naive': self.naive,
            'optimal-length': result.optimal_length,
            'cost-bound': result.horizon,
            # The planner's own count of distinct behaviours; the evaluator's is the yardstick.
            'native-b-coverage': len({getattr(p, 'behaviour_attr', None) for p in plans}),
            **self.selection_info(pool, ctx.k),
        }
        return DiverseRun(plans=plans, outcome=self.outcome_for(len(pool), ctx.k, result.status == asp_base.TIMED_OUT),
                          logs=logs, info=info,
                          native_behaviours=[getattr(p, 'behaviour_str', None) for p in plans])

    def _specs(self, ctx: PlanningContext) -> list[tuple[str, Any]]:
        """The feature specs the space is built from: none when naive, else the task's model."""
        if self.naive:
            return []
        return [(key, config if config not in ({}, '') else None) for key, config in ctx.features]
