"""Per-instance data for diversity-model features (stdlib only).

Some features need data that is not in the PDDL: ``ru`` (resource usage; also
``ru-set`` and ``rc``) needs the list of objects that count as resources for
*this* instance. The generator
resolves that data once, per (instance, model), writes it under the sandbox and
hands the solve command a plain file path -- so a compute node never has to
know where the data came from.

``ru`` accepts, as its config:

* ``"builtin"`` (or ``{"resources": "builtin"}``): the resource data shipped
  with the toolkit (``dp_eval_toolkit/data/ru-info``), taken from the
  Behaviour Planning SMT paper experiments;
* a directory (``{"resources": "path/to/dir"}``), searched for the same JSON
  files *or* for plain resources files laid out as
  ``<dir>/<domain>/<instance>.txt``, ``<dir>/<domain>/resources.txt`` or
  ``<dir>/<domain>.txt``;
* a single resources file (``{"resources": "file.txt"}``), used for every
  instance;
* a list of object names (``{"objects": ["rover0", "rover1"]}``).

A resources file holds ``(:resource NAME MIN MAX DELTA)`` lines, the declaration
format of BehaviourDiversityCounter.
"""

from __future__ import annotations

import json
import os
from functools import cache
from typing import Any

from .tasks import Task, natural_key, trailing_number

BUILTIN_RU_DIR = os.path.join(os.path.dirname(__file__), 'data', 'ru-info')
# Features that take a per-instance list of resource objects.
RESOURCE_FEATURES = ('ru', 'ru-set', 'rc')

# (domain, year, path) of one resources JSON file.
_JsonEntry = tuple[str, str | None, str]


class FeatureDataMissing(Exception):
    """The feature needs per-instance data that is not available."""


def resolve_feature(key: str, config: Any, task: Task, out_dir: str,
                    base_dir: str) -> str | None:
    """The ``KEY[=CONFIG]`` string a solve command gets for this feature.

    Returns ``None`` for a feature that needs no data. Raises
    :class:`FeatureDataMissing` when it needs data this instance does not have.
    """
    if key in RESOURCE_FEATURES:
        return resolve_resources(config, task, out_dir, base_dir)
    if key == 'fn':
        # Functions file for numeric tasks; same lookup rules as ru, no builtin.
        return resolve_resources(config, task, out_dir, base_dir, builtin=False, kind='functions')
    # Anything else: a plain value config is passed through as text.
    if isinstance(config, str) and config:
        return config
    if isinstance(config, dict) and config.get('value') is not None:
        return str(config['value'])
    return None


def resolve_resources(config: Any, task: Task, out_dir: str, base_dir: str,
                      builtin: bool = True, kind: str = 'resources') -> str:
    """The path of the `kind` file for `task`, written under `out_dir` if needed."""
    if isinstance(config, str):
        config = {'resources': config}
    config = dict(config or {})
    if config.get('objects'):
        text = '\n'.join(f'(:resource {name} 0 1 1)' for name in config['objects'])
        return _dump(text, task, out_dir, kind)

    source = _source_path(config, kind, builtin, base_dir)
    if os.path.isfile(source):
        return source
    if not os.path.isdir(source):
        raise FeatureDataMissing(f'{kind} source not found: {source}')

    text = _from_json_dir(source, task)
    if text is not None:
        return _dump(text, task, out_dir, kind)
    path = _from_text_dir(source, task)
    if path is None:
        raise FeatureDataMissing(f'no {kind} for {task.domain} instance {task.instance} in {source}')
    return path


def _source_path(config: dict[str, Any], kind: str, builtin: bool, base_dir: str) -> str:
    """Where the configuration says the data lives: the builtin directory, or a
    path resolved against `base_dir`."""
    source = config.get('resources') or config.get(kind) or ('builtin' if builtin else None)
    if not source:
        raise FeatureDataMissing(f'no "{kind}" given')
    if source == 'builtin':
        return BUILTIN_RU_DIR
    source = os.path.expanduser(str(source))
    if os.path.isabs(source):
        return source
    return os.path.normpath(os.path.join(base_dir, source))


def _dump(text: str, task: Task, out_dir: str, kind: str) -> str:
    """Write `text` to the task's data file, unless it already holds exactly that."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{task.slug}.{kind}.txt')
    if os.path.exists(path):
        with open(path) as handle:
            if handle.read() == text:
                return path
    with open(path, 'w') as handle:
        handle.write(text.rstrip() + '\n')
    return path


# ----------------------------------------------------------------------
# The paper's JSON layout: {"info": {"domain", "year"}, "instances": {"1": "..."}}
# ----------------------------------------------------------------------

@cache
def _json_index(directory: str) -> tuple[_JsonEntry, ...]:
    """(domain, year, path) for every resources JSON under `directory`."""
    entries: list[_JsonEntry] = []
    for root, _dirs, files in os.walk(directory):
        for name in sorted(files):
            if not name.endswith('.json'):
                continue
            path = os.path.join(root, name)
            try:
                with open(path) as handle:
                    info = (json.load(handle) or {}).get('info') or {}
            except (OSError, ValueError, AttributeError):
                continue
            if info.get('domain'):
                year = info.get('year')
                entries.append((str(info['domain']).lower(), str(year) if year else None, path))
    return tuple(entries)


@cache
def _json_instances(path: str) -> dict[str, str]:
    with open(path) as handle:
        return {str(k): v for k, v in (json.load(handle).get('instances') or {}).items()}


def _domain_matches(entry_domain: str, entry_year: str | None, task: Task) -> bool:
    domain = task.domain.lower()
    candidates = {entry_domain}
    if entry_year:
        candidates |= {f'{entry_domain}-{entry_year}', f'{entry_domain}{entry_year}',
                       f'{entry_domain}-{entry_year[-2:]}'}
    return domain in candidates or domain.split('-')[0] == entry_domain


def _from_json_dir(directory: str, task: Task) -> str | None:
    """The task's resources text from the JSON files under `directory`, or
    ``None`` when no file -- or more than one -- could be the right one."""
    matches = [entry for entry in _json_index(directory) if _domain_matches(entry[0], entry[1], task)]
    chosen = _single_match(matches, task)
    number = _instance_number(task.instance)
    if chosen is None or number is None:
        return None
    return _json_instances(chosen[2]).get(number)


def _single_match(matches: list[_JsonEntry], task: Task) -> _JsonEntry | None:
    """Prefer the file of the task's IPC year; the domain name may carry the
    year (``rovers-2006``) when the task itself has none. Several candidates
    with nothing to tell them apart are ambiguous, not a guess."""
    year = task.ipc or next((y for _d, y, _p in matches if y and y in task.domain), None)
    if year:
        same_year = [m for m in matches if m[1] == str(year)]
        if same_year:
            matches = same_year
    return matches[0] if len(matches) == 1 else None


def _instance_number(instance: str) -> str | None:
    number = trailing_number(instance)
    return str(int(number)) if number is not None else None


def _from_text_dir(directory: str, task: Task) -> str | None:
    candidates = [
        os.path.join(directory, task.domain, f'{task.instance}.txt'),
        os.path.join(directory, task.domain, 'resources.txt'),
        os.path.join(directory, f'{task.domain}.txt'),
    ]
    return next((c for c in candidates if os.path.isfile(c)), None)


def builtin_domains() -> list[str]:
    """The domains the shipped resource data covers, e.g. ``rovers-2006``."""
    return sorted((f'{d}-{y}' if y else d for d, y, _p in _json_index(BUILTIN_RU_DIR)), key=natural_key)
