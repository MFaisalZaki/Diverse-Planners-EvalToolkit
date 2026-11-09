"""Planner-agnostic behaviour evaluation of plan sets.

Every planner's plans are scored *here*, under the task's diversity model, by
`BehaviourDiversityCounter <https://github.com/MFaisalZaki/BehaviourDiversityCounter>`_,
which replays each plan on the original problem. That is what makes the
diversity numbers comparable across planners: FI knows nothing about
behaviours at all, and the two behaviour planners compute theirs from their own
encodings, so neither planner's self-reported behaviours can be the yardstick.
(The planners' own behaviour strings are still recorded in the result for
cross-checking.)

The vocabulary follows *Behaviour Spaces for Diversity Planning*: a feature is a
dimension with an extracting function and a dissimilarity in [0, 1]; a
diversity model is a list of features; a plan's behaviour is its tuple of
feature values; and the indicators, each over the *distinct* behaviours of the
plan set, are

* **B-Coverage**: the number of distinct behaviours;
* **B-MaxSum**: the sum of the model's dissimilarity (the mean of the features'
  dissimilarities) over the unordered pairs of behaviours;
* **B-MaxMin**: the smallest pairwise dissimilarity;
* **B-Novelty**: the mean nearest-neighbour dissimilarity.

On top of those, the classic plan-space diversity metrics are reported so the
behaviour indicators can be read against them: *stability* (mean pairwise
Jaccard distance of the plans' action sets), *uniqueness* (fraction of plan
pairs that are not permutations of each other) and the number of distinct plans.

Features are the library's dimensions, named as the planners name them:

``go``      goal ordering                          (the counter's ``go``)
``ru``      how many listed resources a plan uses  (the counter's ``rc``; the
            definition of pyBehaviourPlanningASP/SMT and of the paper experiments)
``ru-set``  which listed resources a plan uses     (the counter's ``ru``)
``cb``      cost; ``cbin`` cost bin; ``fn`` final numeric fluent values, binned;
``uv``      utility value; ``stability`` the plan's action set

``ru``, ``ru-set`` and ``fn`` take their declarations -- ``(:resource NAME MIN MAX
DELTA)`` or ``(:function NAME MIN MAX DELTA)`` lines -- from the file the
feature's config names (see :mod:`dp_eval_toolkit.resources`); resource names
are matched to the task's objects case-insensitively, with ``-`` and ``_``
interchangeable, and names that match no object are dropped and noted.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from itertools import combinations
from typing import Any

from behaviour_diversity_counter import BehaviourDiversityCounter, plan_cost

_RESOURCE_RE = re.compile(r'\(\s*:resource\s+([^\s()]+)([^)]*)\)')
# min max delta for a resource declared by name alone.
_DEFAULT_RESOURCE_FIELDS = '0 1 1'
# Where a dict config keeps its declarations (a path, lines, or object names).
_DECLARATION_KEYS = ('path', 'resources', 'functions', 'objects')
# Toolkit feature key -> counter dimension key, where the two differ.
COUNTER_DIMENSIONS = {'ru': 'rc', 'ru-set': 'ru'}
RESOURCE_DIMENSIONS = ('rc', 'ru')


# ----------------------------------------------------------------------
# Diversity model -> counter dimensions
# ----------------------------------------------------------------------

def counter_dimensions(problem, specs: Sequence[tuple[str, Any]]) -> tuple[list[tuple[str, Any]], list[str]]:
    """The ``(key, addinfo)`` pairs :class:`BehaviourDiversityCounter` takes for
    the toolkit's ``(key, config)`` features, plus notes about dropped names."""
    dimensions: list[tuple[str, Any]] = []
    notes: list[str] = []
    for key, config in specs:
        dimension = COUNTER_DIMENSIONS.get(key, key)
        if dimension in RESOURCE_DIMENSIONS:
            declarations, dropped = resource_declarations(problem, config)
            if dropped:
                notes.append(f'{key}: listed names that are not objects of the task: {dropped}')
            if not declarations:
                raise ValueError(f'{key}: none of the listed resources is an object of the task')
            dimensions.append((dimension, {'resources': declarations}))
        elif dimension == 'fn':
            dimensions.append((dimension, {'functions': read_declarations(config)}))
        else:
            dimensions.append((dimension, dict(config) if isinstance(config, dict) and config else None))
    return dimensions, notes


def resource_declarations(problem, config: Any) -> tuple[list[str], list[str]]:
    """``(:resource OBJECT MIN MAX DELTA)`` lines naming the task's own objects,
    and the declared names that matched none of them."""
    objects = {_norm(o.name): o.name for o in problem.all_objects}
    declarations: list[str] = []
    dropped: set[str] = set()
    for name, fields in _RESOURCE_RE.findall('\n'.join(read_declarations(config))):
        matched = objects.get(_norm(name))
        if matched is None:
            dropped.add(name)
            continue
        declarations.append(f'(:resource {matched} {fields.strip() or _DEFAULT_RESOURCE_FIELDS})')
    return list(dict.fromkeys(declarations)), sorted(dropped)


def read_declarations(config: Any) -> list[str]:
    """The declaration lines a feature was configured with: a file's lines, a
    list of names as ``(:resource NAME ...)`` lines, or the string itself."""
    if isinstance(config, dict):
        config = next((config[key] for key in _DECLARATION_KEYS if config.get(key)), None)
    if isinstance(config, (list, tuple)):
        return [f'(:resource {name} {_DEFAULT_RESOURCE_FIELDS})' for name in config]
    if not config:
        raise ValueError('this feature needs a data file')
    text = str(config)
    if os.path.isfile(text):
        with open(text) as handle:
            text = handle.read()
    return [line for line in text.splitlines() if line.strip()]


def _norm(name: str) -> str:
    return name.lower().replace('-', '_')


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------

# Indicator names as planner configurations spell them -> the counter's.
INDICATORS = {'B-Coverage': 'bcoverage', 'B-MaxSum': 'bmaxsum',
              'B-MaxMin': 'bmaxmin', 'B-Novelty': 'bnovelty'}


def select_plans(problem, plans: Sequence[Any], specs: Sequence[tuple[str, Any]], k: int,
                 indicator: str) -> list[Any]:
    """`k` plans out of `plans`, chosen greedily for `indicator` (``B-Coverage``,
    ``B-MaxSum``, ``B-MaxMin`` or ``B-Novelty``) under the diversity model `specs`.

    This is the post-hoc selection a planner that knows nothing about behaviours
    (FI) gets: generate a pool, keep the `k` most diverse. With `k` or fewer
    plans there is nothing to choose and the pool comes back as it is.
    """
    if indicator not in INDICATORS:
        raise ValueError(f'unknown indicator {indicator!r}; known: {sorted(INDICATORS)}')
    if len(plans) <= k:
        return list(plans)
    dimensions, _notes = counter_dimensions(problem, specs)
    counter = BehaviourDiversityCounter(problem, dimensions)
    return list(counter.extract(list(plans), k, indicator=INDICATORS[indicator]))


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate_plan_set(problem, plans: Sequence[Any],
                      specs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Behaviour of every plan and the indicators of the set.

    A plan that cannot be simulated (or whose behaviour cannot be extracted) is
    reported under ``errors`` and left out of the indicators, so one bad plan
    does not hide the diversity of the rest.
    """
    dimensions, notes = counter_dimensions(problem, specs)
    counter = BehaviourDiversityCounter(problem, dimensions)
    behaviours: list[str | None] = []
    errors: list[str] = []
    evaluated: list[Any] = []
    for index, plan in enumerate(plans):
        try:
            counter.behaviours([plan])
        except Exception as error:
            errors.append(f'plan {index}: {type(error).__name__}: {error}')
            behaviours.append(None)
            continue
        behaviours.append(plan.behaviour)
        evaluated.append(plan)

    n = len(evaluated)
    pairs = n * (n - 1) // 2
    maxsum = counter.b_maxsum(evaluated)
    return {
        'features': [key for key, _ in specs],
        'behaviours': behaviours,
        'b-coverage': counter.b_coverage(evaluated),
        'b-maxsum': maxsum,
        # B-MaxSum over the number of plan pairs: the mean pairwise behaviour
        # dissimilarity, comparable across different k.
        'b-maxsum-normalised': maxsum / pairs if pairs else None,
        'b-maxmin': counter.b_maxmin(evaluated),
        'b-novelty': counter.b_novelty(evaluated),
        'evaluated-plans': n,
        'costs': [_cost(problem, plan) for plan in evaluated],
        **plan_space_metrics(evaluated),
        'errors': errors,
        'notes': notes,
    }


def _cost(problem, plan) -> int | float:
    """The plan's cost: the one the counter recorded while simulating it, else a replay."""
    cost = getattr(plan, 'cost', None)
    if cost is None:
        cost = plan_cost(problem, plan)
    if hasattr(cost, 'constant_value'):
        cost = cost.constant_value()
    return int(cost) if float(cost).is_integer() else float(cost)


# ----------------------------------------------------------------------
# Plan-space metrics
# ----------------------------------------------------------------------

def _action_key(action) -> str:
    return f'{action.action.name}({",".join(str(p) for p in action.actual_parameters)})'


def plan_space_metrics(plans: Sequence[Any]) -> dict[str, Any]:
    """Stability, uniqueness and the number of distinct plans."""
    sequences = [tuple(_action_key(a) for a in p.actions) for p in plans]
    distinct = len(set(sequences))
    pairs = list(combinations(sequences, 2))
    if not pairs:
        return {'distinct-plans': distinct, 'stability': None, 'uniqueness': None}
    jaccard = [_jaccard_distance(frozenset(a), frozenset(b)) for a, b in pairs]
    not_permutations = sum(1 for a, b in pairs if sorted(a) != sorted(b))
    return {
        'distinct-plans': distinct,
        'stability': sum(jaccard) / len(pairs),
        'uniqueness': not_permutations / len(pairs),
    }


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    union = a | b
    return 1.0 - len(a & b) / len(union) if union else 0.0
