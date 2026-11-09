"""Diverse-planner adapters and their registry.

A planner configuration's ``"adapter"`` is either one of the names below or
``"package.module:ClassName"`` naming a :class:`DiversePlannerAdapter`
subclass of your own -- which is how a further planner joins a sweep without
touching the toolkit.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import (COMPLETE, ERROR, EXHAUSTED, MEMOUT, TIMEOUT, UNSOLVABLE, UNSUPPORTED, DiversePlannerAdapter,
                   DiverseRun, PlanningContext)

BUILTIN_ADAPTERS: dict[str, str] = {
    'fi': 'dp_eval_toolkit.adapters.fi:ForbidIterativeAdapter',
    'bp-smt': 'dp_eval_toolkit.adapters.bp_smt:BehaviourPlanningSMTAdapter',
    'bp-asp': 'dp_eval_toolkit.adapters.bp_asp:BehaviourPlanningASPAdapter',
    'up-anytime': 'dp_eval_toolkit.adapters.up_anytime:UPAnytimeAdapter',
}


def load_adapter(name: str, params: dict[str, Any] | None = None) -> DiversePlannerAdapter:
    spec = BUILTIN_ADAPTERS.get(name, name)
    if ':' not in spec:
        raise ValueError(f'unknown adapter {name!r}: use one of {sorted(BUILTIN_ADAPTERS)} '
                         f'or "package.module:ClassName"')
    module_name, class_name = spec.split(':', 1)
    cls: type[DiversePlannerAdapter] = getattr(importlib.import_module(module_name), class_name)
    return cls(params)


__all__ = ['BUILTIN_ADAPTERS', 'COMPLETE', 'ERROR', 'EXHAUSTED', 'MEMOUT', 'TIMEOUT', 'UNSOLVABLE', 'UNSUPPORTED',
           'DiversePlannerAdapter', 'DiverseRun', 'PlanningContext', 'load_adapter']
