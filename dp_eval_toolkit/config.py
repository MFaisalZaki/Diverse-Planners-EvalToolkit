"""Experiment configuration: limits, slurm knobs, the diversity grid, planners.

An experiment directory holds

    <exp-dir>/exp-details.json      limits + slurm + diversity grid + task selection
    <exp-dir>/planners/*.json       one diverse-planner configuration per file

A *diversity task* is the tuple (planning problem, k, q, diversity model): the
number of plans to return, the quality bound (cost bound as a factor of the
optimal cost; 1.0 is optimal plans only) and the features the plans should be
diverse over. ``exp-details.json`` spells out the grid; the generator crosses it
with the benchmark instances and with every planner configuration, one slurm
array index per (planner, instance, k, q, model).

Every ``.json`` under ``planners/`` is benchmarked; to leave a planner out,
delete its file.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_EXP_DETAILS: dict[str, Any] = {
    'name': 'default',
    'cfgs': {
        'timelimit': '00:30:00',
        'memorylimit': '8GB',
        # Head-room the scheduler gets on top of the task's own limits, so the
        # runner hits its limit first and records TIMEOUT/MEMOUT (with whatever
        # plans it had) rather than the job vanishing into a slurm cancellation.
        'slurm-time-headroom': '00:05:00',
        'slurm-memory-headroom': '1GB',
        'validate-plans': True,
        'slurm': {
            'cpus-per-task': 1,
            'partition': None,
            'qos': None,
            'account': None,
            'max-parallel-jobs': 50,
            'max-array-size': 1000,
            'extra-directives': [],
        },
    },
    'diversity': {
        'k-values': [5, 10],
        'quality-bounds': [1.0],
        # name -> list of [feature-key, config] pairs. See behaviours.py for
        # the feature keys the toolkit can evaluate.
        'models': {
            'go': [['go', {}]],
        },
    },
    'tasks': {
        'tracks': ['classical'],
        'max-instances-per-domain': 0,
        'selection': 'even',
        'include-domains': [],
        'exclude-domains': [],
        'ipc-years': [],
        'selected-tasks': [],
        # "(ipc-year, domain, n)" references in the paper experiments' convention:
        # a list, a file of them (one per line, # comments), or "builtin" for the
        # paper's classical instances shipped under dp_eval_toolkit/data.
        'selected-instances': [],
    },
}

BUILTIN_INSTANCES_FILE = os.path.join(os.path.dirname(__file__), 'data', 'classical-instances.txt')

# A planner configuration may name the environment its commands run in.
LAUNCHER_KEYS = ('venv-dir', 'conda-env', 'apptainer-image')


# ----------------------------------------------------------------------
# Diversity models
# ----------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """One feature of a diversity model as written in ``exp-details.json``.

    ``config`` is whatever followed the key: ``{}`` for ``go``, a path or a
    ``{"resources": ..., "optional": ...}`` object for ``ru``. ``optional``
    features are dropped (and the drop recorded) for an instance that has no
    data for them, instead of taking the instance out of the experiment.
    """

    key: str
    config: Any = None

    @property
    def optional(self) -> bool:
        return isinstance(self.config, dict) and bool(self.config.get('optional'))

    @classmethod
    def parse(cls, raw) -> FeatureSpec:
        if isinstance(raw, str):
            return cls(raw, {})
        if isinstance(raw, dict):
            raw = dict(raw)
            key = raw.pop('key', None) or raw.pop('name', None)
            if not key:
                raise ValueError(f'feature {raw!r} has no "key"')
            return cls(str(key), raw)
        if isinstance(raw, (list, tuple)) and raw:
            return cls(str(raw[0]), raw[1] if len(raw) > 1 else {})
        raise ValueError(f'cannot read feature specification {raw!r}')

    def to_json(self) -> list[Any]:
        return [self.key, self.config]


@dataclass
class DiversityModel:
    name: str
    features: list[FeatureSpec]

    @classmethod
    def parse(cls, name: str, raw) -> DiversityModel:
        if isinstance(raw, dict):
            raw = raw.get('features', [])
        features = [FeatureSpec.parse(f) for f in raw or []]
        if not features:
            raise ValueError(f'diversity model {name!r} has no features')
        keys = [f.key for f in features]
        if len(set(keys)) != len(keys):
            raise ValueError(f'diversity model {name!r} repeats a feature: {keys}')
        return cls(name=name, features=features)


# ----------------------------------------------------------------------
# Planner configurations
# ----------------------------------------------------------------------

@dataclass
class PlannerConfig:
    """One diverse-planner configuration.

    ``adapter`` names how the planner is driven (see
    :mod:`dp_eval_toolkit.adapters`): a built-in name (``fi``, ``bp-smt``,
    ``bp-asp``, ``up-anytime``) or ``"package.module:Class"`` for an adapter of
    your own. ``planner-params`` reach the adapter verbatim.
    """

    tag: str
    adapter: str
    params: dict[str, Any] = field(default_factory=dict)
    tracks: list[str] | None = None
    models: list[str] | None = None        # restrict to some diversity models
    path: str | None = None
    # Where this planner's commands run, overriding generate's --venv-dir etc.:
    # planners with conflicting dependency pins can live in separate environments.
    launcher: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> PlannerConfig:
        with open(path) as handle:
            raw = json.load(handle)
        tag = raw.get('planner-tag') or os.path.splitext(os.path.basename(path))[0]
        adapter = raw.get('adapter')
        if not adapter:
            raise ValueError(f'{path}: "adapter" is required (e.g. "fi", "bp-smt", "bp-asp")')
        launcher = {key: os.path.expanduser(str(raw[key])) for key in LAUNCHER_KEYS if raw.get(key)}
        if len(launcher) > 1:
            raise ValueError(f'{path}: give at most one of {", ".join(LAUNCHER_KEYS)}')
        return cls(tag=tag, adapter=adapter, params=dict(raw.get('planner-params') or {}),
                   tracks=raw.get('tracks'), models=raw.get('models'),
                   path=os.path.abspath(path), launcher=launcher)

    def runs_track(self, track: str) -> bool:
        return not self.tracks or track in self.tracks

    def runs_model(self, model: str) -> bool:
        return not self.models or model in self.models


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------

@dataclass
class Experiment:
    name: str
    details: dict[str, Any]
    planners: list[PlannerConfig]
    path: str

    # -- limits ----------------------------------------------------------
    @property
    def time_limit_seconds(self) -> int:
        return parse_time(self._cfg('timelimit'))

    @property
    def memory_limit_mb(self) -> int:
        return parse_memory(self._cfg('memorylimit'))

    @property
    def slurm_time(self) -> str:
        head = parse_time(self._cfg('slurm-time-headroom'))
        return format_time(self.time_limit_seconds + head)

    @property
    def slurm_memory(self) -> str:
        head = parse_memory(self._cfg('slurm-memory-headroom'))
        return f'{self.memory_limit_mb + head}M'

    @property
    def slurm(self) -> dict[str, Any]:
        return self._merged_section('cfgs', 'slurm')

    @property
    def validate_plans(self) -> bool:
        return bool(self._cfg('validate-plans'))

    # -- the diversity grid ---------------------------------------------
    @property
    def diversity(self) -> dict[str, Any]:
        return self._merged_section('diversity')

    @property
    def k_values(self) -> list[int]:
        values = [int(k) for k in _as_list(self.diversity.get('k-values'))]
        if not values or any(k < 1 for k in values):
            raise ValueError(f'k-values must be positive integers, got {values}')
        return sorted(set(values))

    @property
    def quality_bounds(self) -> list[float]:
        values = [float(q) for q in _as_list(self.diversity.get('quality-bounds'))]
        if not values or any(q < 1.0 for q in values):
            raise ValueError(f'quality-bounds must be >= 1.0, got {values}')
        return sorted(set(values))

    @property
    def models(self) -> list[DiversityModel]:
        raw = self.diversity.get('models') or {}
        if not isinstance(raw, dict) or not raw:
            raise ValueError('diversity.models must map a model name to its features')
        return [DiversityModel.parse(name, body) for name, body in raw.items()]

    def grid(self) -> list[tuple[DiversityModel, int, float]]:
        """Every (model, k, q) combination, in a stable order."""
        return [(m, k, q) for m in self.models for k in self.k_values for q in self.quality_bounds]

    # -- tasks -------------------------------------------------------------
    @property
    def task_selection(self) -> dict[str, Any]:
        return self._merged_section('tasks')

    @property
    def selected_instances(self) -> list[str]:
        """The ``selected-instances`` references, read from their file if need be."""
        return read_instance_refs(self.task_selection.get('selected-instances'), self.resolve_path)

    def _cfg(self, key: str):
        """A ``cfgs`` entry, falling back to the default."""
        return (self.details.get('cfgs') or {}).get(key, DEFAULT_EXP_DETAILS['cfgs'][key])

    def _merged_section(self, *keys: str) -> dict[str, Any]:
        """A section of the details on top of its defaults; `keys` is the
        path to it (``'tasks'``, or ``'cfgs', 'slurm'``)."""
        defaults = _dig(DEFAULT_EXP_DETAILS, keys)
        given = _dig(self.details, keys)
        return {**defaults, **given}

    def resolve_path(self, path: str) -> str:
        """A path from the configuration: absolute, ``~``-expanded, relative to
        the experiment directory otherwise."""
        path = os.path.expanduser(str(path))
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(self.path, path))

    @classmethod
    def load(cls, exp_dir: str) -> Experiment:
        exp_dir = os.path.abspath(os.path.expanduser(exp_dir))
        details_file = exp_dir if os.path.isfile(exp_dir) else os.path.join(exp_dir, 'exp-details.json')
        exp_dir = os.path.dirname(details_file)
        if not os.path.isfile(details_file):
            raise FileNotFoundError(f'experiment details not found: {details_file}')
        with open(details_file) as handle:
            details = json.load(handle)
        experiment = cls(name=details.get('name', os.path.basename(exp_dir)), details=details,
                         planners=_load_planners(os.path.join(exp_dir, 'planners')), path=exp_dir)
        experiment.validate()
        return experiment

    def validate(self) -> None:
        """Fail at load time, not on a compute node: these are read by every stage."""
        self.grid()  # raises on a bad k-values, quality-bounds or models entry
        known_models = {m.name for m in self.models}
        for planner in self.planners:
            unknown = set(planner.models or []) - known_models
            if unknown:
                raise ValueError(f'{planner.path}: names unknown diversity model(s) {sorted(unknown)}')


def _dig(mapping: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """``mapping[keys[0]][keys[1]]...``, with ``{}`` wherever a level is missing."""
    for key in keys:
        mapping = (mapping or {}).get(key) or {}
    return mapping


def read_instance_refs(value, resolve_path=os.path.abspath) -> list[str]:
    """``selected-instances`` as a list of reference strings: from an inline list,
    a file (one reference per line, ``#`` comments) or ``"builtin"``."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [ref if isinstance(ref, str) else '(' + ', '.join(str(p) for p in ref) + ')' for ref in value]
    path = BUILTIN_INSTANCES_FILE if value == 'builtin' else resolve_path(str(value))
    with open(path) as handle:
        lines = (line.split('#', 1)[0].strip() for line in handle)
        return [line for line in lines if line]


def _load_planners(planners_dir: str) -> list[PlannerConfig]:
    """Every ``*.json`` under `planners_dir`, with distinct tags."""
    if not os.path.isdir(planners_dir):
        raise FileNotFoundError(f'planner configurations not found: {planners_dir}')
    planners = [PlannerConfig.from_file(os.path.join(planners_dir, name))
                for name in sorted(os.listdir(planners_dir)) if name.endswith('.json')]
    if not planners:
        raise ValueError(f'no planner configuration (*.json) in {planners_dir}')
    tags = [p.tag for p in planners]
    duplicates = {t for t in tags if tags.count(t) > 1}
    if duplicates:
        raise ValueError(f'duplicate planner tags would overwrite each other: {sorted(duplicates)}')
    return planners


# ----------------------------------------------------------------------
# Limit parsing (same conventions as pyPMTEvalToolkit)
# ----------------------------------------------------------------------

_MEMORY_RE = re.compile(r'^\s*([0-9]*\.?[0-9]+)\s*([kmgt]?b?)\s*$', re.IGNORECASE)
_MEMORY_UNITS_MB = {'': 1, 'b': 1 / (1024 ** 2), 'k': 1 / 1024, 'kb': 1 / 1024,
                    'm': 1, 'mb': 1, 'g': 1024, 'gb': 1024, 't': 1024 ** 2, 'tb': 1024 ** 2}
_TIME_RE = re.compile(r'^([0-9]*\.?[0-9]+)\s*([smhd]?)$')
_TIME_UNITS_S = {'': 1, 's': 1, 'm': 60, 'h': 3600, 'd': 86400}
# Seconds per field of a clock reading, last field first: SS, MM, HH, D.
_CLOCK_FIELDS_S = (1, 60, 3600, 86400)


def parse_memory(value) -> int:
    """``"8GB"`` / ``"8192"`` / ``8192`` -> MiB. A bare number is MiB."""
    if isinstance(value, (int, float)):
        return int(value)
    match = _MEMORY_RE.match(str(value))
    if not match:
        raise ValueError(f'cannot parse memory limit: {value!r} (try "8GB" or "8192MB")')
    amount, unit = float(match.group(1)), match.group(2).lower()
    return max(1, round(amount * _MEMORY_UNITS_MB[unit]))


def parse_time(value) -> int:
    """``"00:30:00"`` / ``"30m"`` / ``"1800"`` / ``1800`` -> seconds."""
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if ':' in text:
        fields = [float(p) for p in text.split(':')]
        if not 2 <= len(fields) <= len(_CLOCK_FIELDS_S):
            raise ValueError(f'cannot parse time limit: {value!r}')
        return int(sum(amount * unit for amount, unit in zip(reversed(fields), _CLOCK_FIELDS_S)))
    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f'cannot parse time limit: {value!r} (try "00:30:00" or "30m")')
    return int(float(match.group(1)) * _TIME_UNITS_S[match.group(2)])


def format_time(seconds: int) -> str:
    """Seconds -> the ``HH:MM:SS`` (or ``D-HH:MM:SS``) slurm wants."""
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    clock = f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{days}-{clock}' if days else clock


def format_q(q: float) -> str:
    """``1.0`` -> ``"1.0"``, ``1.25`` -> ``"1.25"``: stable across the stages."""
    return repr(float(q))


def run_label(model: str, k: int, q: float) -> str:
    """The directory a (model, k, q) combination's results live in."""
    return f'{model}__k{int(k)}__q{format_q(q)}'


def _as_list(value) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]
