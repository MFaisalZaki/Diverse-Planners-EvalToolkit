"""A stand-in diverse planner for testing the runner end to end: it returns
the plans it was configured with, parsed against the task."""

from __future__ import annotations

from dp_eval_toolkit.adapters.base import DiversePlannerAdapter, DiverseRun, PlanningContext


class FakeAdapter(DiversePlannerAdapter):
    name = 'fake'
    supported_features = None

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        from unified_planning.io import PDDLReader

        reader = PDDLReader()
        plans = [reader.parse_plan_string(ctx.problem, text) for text in self.params['plans']]
        ctx.partial = plans
        return DiverseRun(plans=plans[:ctx.k], outcome=self.outcome_for(len(plans), ctx.k),
                          info={'optimal-length': min(len(p.actions) for p in plans)},
                          native_behaviours=['fake'] * min(len(plans), ctx.k))
