"""Planner-agnostic behaviour evaluation of plan sets.

Every planner's plans are scored *here*, under the task's diversity model, by
simulating each plan on the original problem. That is what makes the diversity
numbers comparable across planners: FI knows nothing about behaviours at all,
and the two behaviour planners compute theirs from their own encodings, so
neither planner's self-reported behaviours can be the yardstick. (The planners'
own behaviour strings are still recorded in the result for cross-checking.)

The vocabulary follows *Behaviour Spaces for Diversity Planning*: a feature is a
dimension with an extracting function and a dissimilarity in [0, 1]; a
diversity model is a list of features; a plan's behaviour is its tuple of
feature values; and the indicators are

* **B-Coverage**: the number of distinct behaviours in the plan set;
* **B-MaxSum**: the sum, over unordered pairs of distinct behaviours, of the
  model's dissimilarity (the average of the features' dissimilarities).

On top of those, the classic plan-space diversity metrics are reported so the
behaviour indicators can be read against them: *stability* (mean pairwise
Jaccard distance of the plans' action sets), *uniqueness* (fraction of plan
pairs that are not permutations of each other) and the number of distinct plans.

Features
--------
``go``  goal ordering: the rank at which each goal atom is first achieved;
        dissimilarity = fraction of goals whose rank differs.
``ru``  resource usage: number of listed resource objects the plan's actions
        take as arguments; dissimilarity = |a - b| / (n - 1), clamped to 1.
``cb``  cost bound: the plan's cost (unit cost: its length); dissimilarity =
        0 when equal, 1 otherwise.
``fn``  numeric function values: the box each listed fluent ends in, from a
        functions file ``(:function NAME MIN MAX STEP)``; dissimilarity =
        fraction of fluents whose box differs.

``go`` and ``ru`` match the definitions of pyBehaviourPlanningASP exactly, so
on its plans the toolkit's behaviours agree with the planner's own. New
features register with :func:`register_feature`.
"""

from __future__ import annotations

import re
from fractions import Fraction
from itertools import combinations
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

Behaviour = Tuple[Hashable, ...]


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------

class Feature:
    """Extracting function + dissimilarity for one dimension."""

    key = ''

    def __init__(self, problem, config: Any = None):
        self.problem = problem
        self.config = config

    def extract(self, trace: 'PlanTrace') -> Hashable:
        raise NotImplementedError

    def dissimilarity(self, a: Hashable, b: Hashable) -> Fraction:
        raise NotImplementedError

    def describe(self, value: Hashable) -> str:
        return str(value)


FEATURES: Dict[str, Callable[..., Feature]] = {}


def register_feature(cls):
    """Class decorator: make a feature available under ``cls.key``."""
    FEATURES[cls.key] = cls
    return cls


@register_feature
class GoalOrdering(Feature):
    key = 'go'

    def __init__(self, problem, config: Any = None):
        super().__init__(problem, config)
        self.goals = goal_atoms(problem)
        if not self.goals:
            raise ValueError('goal ordering needs at least one goal')
        self.labels = [str(g) for g in self.goals]

    def extract(self, trace: 'PlanTrace') -> Tuple[int, ...]:
        first: List[Optional[int]] = [None] * len(self.goals)
        for t, state in enumerate(trace.states):
            for i, goal in enumerate(self.goals):
                if first[i] is None and trace.holds(goal, state):
                    first[i] = t
        if any(f is None for f in first):
            raise ValueError('the plan never achieves some goal')
        # rank = number of goals first achieved strictly earlier (0 at step 0):
        # the ASP definition, goal_rank(G, R) with R = #count{G2 seen at t-1}.
        return tuple(0 if f == 0 else sum(1 for g in first if g < f) for f in first)

    def dissimilarity(self, a, b) -> Fraction:
        return Fraction(sum(1 for x, y in zip(a, b) if x != y), len(self.goals))

    def describe(self, value) -> str:
        ordered = sorted(zip(value, self.labels))
        out = ordered[0][1]
        for (prev, _), (rank, label) in zip(ordered, ordered[1:]):
            out += (' = ' if prev == rank else ' < ') + label
        return out


_RESOURCE_RE = re.compile(r'\(\s*:resource\s+([^\s()]+)')
_FUNCTION_RE = re.compile(r'\(\s*:function\s+([^\s()]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)')


@register_feature
class ResourceUsage(Feature):
    key = 'ru'

    def __init__(self, problem, config: Any = None):
        super().__init__(problem, config)
        names = read_resource_names(config)
        objects = {o.name.lower(): o.name for o in problem.all_objects}
        # Names are matched case-insensitively and with - and _ interchangeable:
        # the resource data predates the PDDL writers that mangle one into the other.
        normal = {_norm(k): v for k, v in objects.items()}
        self.resources = sorted({normal[_norm(n)] for n in names if _norm(n) in normal})
        self.dropped = sorted(n for n in names if _norm(n) not in normal)
        if not self.resources:
            raise ValueError(f'none of the listed resources {names} is an object of the task')

    def extract(self, trace: 'PlanTrace') -> int:
        wanted = set(self.resources)
        used = set()
        for action in trace.plan.actions:
            for param in action.actual_parameters:
                if param.is_object_exp() and param.object().name in wanted:
                    used.add(param.object().name)
        return len(used)

    @property
    def scale(self) -> int:
        return max(1, len(self.resources) - 1)

    def dissimilarity(self, a, b) -> Fraction:
        return min(Fraction(1), Fraction(abs(a - b), self.scale))


@register_feature
class CostBound(Feature):
    key = 'cb'

    def extract(self, trace: 'PlanTrace') -> int:
        return trace.cost

    def dissimilarity(self, a, b) -> Fraction:
        return Fraction(0 if a == b else 1)


@register_feature
class FunctionValues(Feature):
    key = 'fn'

    def __init__(self, problem, config: Any = None):
        super().__init__(problem, config)
        text = _read_text(config)
        self.boxes = []
        fluents = {f.name.lower(): f for f in problem.fluents}
        for name, low, high, step in _FUNCTION_RE.findall(text):
            # "energy_rover0" names fluent energy applied to rover0, or a 0-ary fluent.
            self.boxes.append((name, float(low), float(high), float(step) or 1.0))
        if not self.boxes:
            raise ValueError('no (:function NAME MIN MAX STEP) entries')
        self._fluents = fluents

    def _expression(self, name: str):
        em = self.problem.environment.expression_manager
        if name.lower() in self._fluents:
            return em.FluentExp(self._fluents[name.lower()])
        for fname, fluent in self._fluents.items():
            if name.lower().startswith(fname + '_') and fluent.arity > 0:
                args = name[len(fname) + 1:].split('_')
                if len(args) == fluent.arity and all(self.problem.has_object(a) for a in args):
                    return em.FluentExp(fluent, [em.ObjectExp(self.problem.object(a)) for a in args])
        raise ValueError(f'{name} is not a numeric fluent of the task')

    def extract(self, trace: 'PlanTrace') -> Tuple[int, ...]:
        final = trace.states[-1]
        boxes = []
        for name, low, high, step in self.boxes:
            value = float(trace.value(self._expression(name), final))
            value = min(max(value, min(low, high)), max(low, high))
            boxes.append(int((value - min(low, high)) // step))
        return tuple(boxes)

    def dissimilarity(self, a, b) -> Fraction:
        return Fraction(sum(1 for x, y in zip(a, b) if x != y), len(self.boxes))


# ----------------------------------------------------------------------
# Diversity model
# ----------------------------------------------------------------------

class DiversityModel:
    def __init__(self, problem, specs: Sequence[Tuple[str, Any]]):
        self.features: List[Feature] = []
        for key, config in specs:
            if key not in FEATURES:
                raise ValueError(f'unknown feature {key!r}; known: {sorted(FEATURES)}')
            self.features.append(FEATURES[key](problem, config))
        if not self.features:
            raise ValueError('a diversity model needs at least one feature')
        self.keys = [f.key for f in self.features]

    def behaviour(self, trace: 'PlanTrace') -> Behaviour:
        return tuple(f.extract(trace) for f in self.features)

    def dissimilarity(self, a: Behaviour, b: Behaviour) -> Fraction:
        total = sum((f.dissimilarity(x, y) for f, x, y in zip(self.features, a, b)), Fraction(0))
        return total / len(self.features)

    def describe(self, behaviour: Behaviour) -> str:
        return '<' + ', '.join(f'{f.key}={f.describe(v)}' for f, v in zip(self.features, behaviour)) + '>'


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------

class PlanTrace:
    """A plan with the states it visits on the original problem."""

    def __init__(self, problem, plan, simulator, evaluator):
        self.plan = plan
        self._evaluator = evaluator
        state = simulator.get_initial_state()
        self.states = [state]
        for action in plan.actions:
            nxt = simulator.apply(state, action)
            if nxt is None:
                raise ValueError(f'action {action} is not applicable')
            state = nxt
            self.states.append(state)
        self.cost = len(plan.actions)

    def value(self, expression, state):
        return self._evaluator.evaluate(expression, state)

    def holds(self, expression, state) -> bool:
        return bool(self.value(expression, state).bool_constant_value())


def goal_atoms(problem) -> List[Any]:
    """The conjuncts of the problem's goals (each a boolean expression)."""
    atoms: List[Any] = []

    def flatten(expression):
        if expression.is_and():
            for arg in expression.args:
                flatten(arg)
        elif not expression.is_true():
            atoms.append(expression)

    for goal in problem.goals:
        flatten(goal)
    unique, seen = [], set()
    for atom in atoms:
        if atom not in seen:
            seen.add(atom)
            unique.append(atom)
    return sorted(unique, key=str)


def make_tracer(problem):
    """A function plan -> :class:`PlanTrace` over `problem`."""
    from unified_planning.shortcuts import SequentialSimulator
    from unified_planning.model.walkers import StateEvaluator

    simulator = SequentialSimulator(problem=problem)
    evaluator = StateEvaluator(problem)
    return lambda plan: PlanTrace(problem, plan, simulator, evaluator)


# ----------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------

def b_coverage(behaviours: Sequence[Behaviour]) -> int:
    return len(set(behaviours))


def b_maxsum(model: DiversityModel, behaviours: Sequence[Behaviour]) -> Fraction:
    distinct = list(dict.fromkeys(behaviours))
    return sum((model.dissimilarity(a, b) for a, b in combinations(distinct, 2)), Fraction(0))


def _action_key(action) -> str:
    return f'{action.action.name}({",".join(str(p) for p in action.actual_parameters)})'


def plan_space_metrics(plans: Sequence[Any]) -> Dict[str, Any]:
    """Stability, uniqueness and the number of distinct plans."""
    sequences = [tuple(_action_key(a) for a in p.actions) for p in plans]
    sets = [frozenset(s) for s in sequences]
    multisets = [tuple(sorted(s)) for s in sequences]
    pairs = list(combinations(range(len(plans)), 2))
    if not pairs:
        return {'distinct-plans': len(set(sequences)), 'stability': None, 'uniqueness': None}
    jaccard = []
    for i, j in pairs:
        union = sets[i] | sets[j]
        jaccard.append(1.0 - (len(sets[i] & sets[j]) / len(union)) if union else 0.0)
    unique = sum(1 for i, j in pairs if multisets[i] != multisets[j])
    return {
        'distinct-plans': len(set(sequences)),
        'stability': sum(jaccard) / len(jaccard),
        'uniqueness': unique / len(pairs),
    }


def evaluate_plan_set(problem, plans: Sequence[Any],
                      specs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """Behaviour of every plan and the indicators of the set.

    A plan that cannot be simulated (or whose behaviour cannot be extracted) is
    reported under ``errors`` and left out of the indicators, so one bad plan
    does not hide the diversity of the rest.
    """
    model = DiversityModel(problem, specs)
    tracer = make_tracer(problem)
    behaviours: List[Behaviour] = []
    per_plan: List[Optional[str]] = []
    errors: List[str] = []
    costs: List[int] = []
    for index, plan in enumerate(plans):
        try:
            trace = tracer(plan)
            behaviour = model.behaviour(trace)
        except Exception as error:                            # noqa: BLE001 -- reported
            errors.append(f'plan {index}: {type(error).__name__}: {error}')
            per_plan.append(None)
            continue
        behaviours.append(behaviour)
        per_plan.append(model.describe(behaviour))
        costs.append(trace.cost)

    maxsum = b_maxsum(model, behaviours)
    n = len(behaviours)
    pairs = n * (n - 1) // 2
    notes = []
    for feature in model.features:
        dropped = getattr(feature, 'dropped', None)
        if dropped:
            notes.append(f'{feature.key}: listed names that are not objects of the task: {dropped}')
    return {
        'features': model.keys,
        'behaviours': per_plan,
        'b-coverage': b_coverage(behaviours),
        'b-maxsum': float(maxsum),
        'b-maxsum-fraction': f'{maxsum.numerator}/{maxsum.denominator}',
        # B-MaxSum over the number of plan pairs: the mean pairwise behaviour
        # dissimilarity, comparable across different k.
        'b-maxsum-normalised': float(maxsum) / pairs if pairs else None,
        'evaluated-plans': n,
        'costs': costs,
        **plan_space_metrics([p for p, b in zip(plans, per_plan) if b is not None]),
        'errors': errors,
        'notes': notes,
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _norm(name: str) -> str:
    return name.lower().replace('-', '_')


def _read_text(config: Any) -> str:
    if isinstance(config, dict):
        config = config.get('path') or config.get('resources') or config.get('functions')
    if isinstance(config, (list, tuple)):
        return '\n'.join(f'(:resource {name})' for name in config)
    if not config:
        raise ValueError('this feature needs a data file')
    import os
    if os.path.isfile(str(config)):
        with open(str(config)) as handle:
            return handle.read()
    return str(config)


def read_resource_names(config: Any) -> List[str]:
    names = _RESOURCE_RE.findall(_read_text(config))
    if not names:
        raise ValueError('no (:resource NAME ...) entries')
    return list(dict.fromkeys(names))
