"""Per-instance data for diversity-model features (stdlib only).

Some features need data that is not in the PDDL: ``ru`` (resource usage) needs
the list of objects that count as resources for *this* instance. The generator
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

A resources file holds ``(:resource NAME ...)`` lines; only ``NAME`` is read.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .tasks import Task, natural_key

BUILTIN_RU_DIR = os.path.join(os.path.dirname(__file__), 'data', 'ru-info')

_TRAILING_NUM_RE = re.compile(r'(\d+)(?!.*\d)')


class FeatureDataMissing(Exception):
    """The feature needs per-instance data that is not available."""


def resolve_feature(key: str, config: Any, task: Task, out_dir: str,
                    base_dir: str) -> Optional[str]:
    """The ``KEY[=CONFIG]`` string a solve command gets for this feature.

    Returns ``None`` for a feature that needs no data. Raises
    :class:`FeatureDataMissing` when it needs data this instance does not have.
    """
    if key == 'ru':
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
    if isinstance(config, str):
        config = {'resources': config}
    config = dict(config or {})
    if config.get('objects'):
        text = '\n'.join(f'(:resource {name})' for name in config['objects'])
        return _dump(text, task, out_dir, kind)

    source = config.get('resources') or config.get(kind) or ('builtin' if builtin else None)
    if not source:
        raise FeatureDataMissing(f'no "{kind}" given')
    if source == 'builtin':
        source = BUILTIN_RU_DIR
    else:
        source = os.path.expanduser(str(source))
        if not os.path.isabs(source):
            source = os.path.normpath(os.path.join(base_dir, source))

    if os.path.isfile(source):
        return source
    if not os.path.isdir(source):
        raise FeatureDataMissing(f'{kind} source not found: {source}')

    text = _from_json_dir(source, task)
    if text is None:
        path = _from_text_dir(source, task)
        if path:
            return path
        raise FeatureDataMissing(f'no {kind} for {task.domain} instance {task.instance} in {source}')
    return _dump(text, task, out_dir, kind)


def _dump(text: str, task: Task, out_dir: str, kind: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{task.slug}.{kind}.txt')
    if not os.path.exists(path) or open(path).read() != text:
        with open(path, 'w') as handle:
            handle.write(text.rstrip() + '\n')
    return path


# ----------------------------------------------------------------------
# The paper's JSON layout: {"info": {"domain", "year"}, "instances": {"1": "..."}}
# ----------------------------------------------------------------------

@lru_cache(maxsize=None)
def _json_index(directory: str) -> Tuple[Tuple[str, Optional[str], str], ...]:
    """(domain, year, path) for every resources JSON under `directory`."""
    entries: List[Tuple[str, Optional[str], str]] = []
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


@lru_cache(maxsize=None)
def _json_instances(path: str) -> Dict[str, str]:
    with open(path) as handle:
        return {str(k): v for k, v in (json.load(handle).get('instances') or {}).items()}


def _domain_matches(entry_domain: str, entry_year: Optional[str], task: Task) -> bool:
    domain = task.domain.lower()
    candidates = {entry_domain}
    if entry_year:
        candidates |= {f'{entry_domain}-{entry_year}', f'{entry_domain}{entry_year}',
                       f'{entry_domain}-{entry_year[-2:]}'}
    return domain in candidates or domain.split('-')[0] == entry_domain


def _from_json_dir(directory: str, task: Task) -> Optional[str]:
    matches = [(d, y, p) for d, y, p in _json_index(directory) if _domain_matches(d, y, task)]
    if not matches:
        return None
    # Prefer the file of the task's IPC year; the domain name may carry the year
    # (``rovers-2006``) when the task itself has none.
    year = task.ipc or next((y for _d, y, _p in matches if y and y in task.domain), None)
    if year:
        same_year = [m for m in matches if m[1] == str(year)]
        if same_year:
            matches = same_year
        elif len(matches) > 1:
            return None
    elif len(matches) > 1:
        return None
    number = _instance_number(task.instance)
    if number is None:
        return None
    return _json_instances(matches[0][2]).get(number)


def _instance_number(instance: str) -> Optional[str]:
    match = _TRAILING_NUM_RE.search(instance)
    return str(int(match.group(1))) if match else None


def _from_text_dir(directory: str, task: Task) -> Optional[str]:
    candidates = [
        os.path.join(directory, task.domain, f'{task.instance}.txt'),
        os.path.join(directory, task.domain, 'resources.txt'),
        os.path.join(directory, f'{task.domain}.txt'),
    ]
    return next((c for c in candidates if os.path.isfile(c)), None)


def builtin_domains() -> List[str]:
    """The domains the shipped resource data covers, e.g. ``rovers-2006``."""
    return sorted((f'{d}-{y}' if y else d for d, y, _p in _json_index(BUILTIN_RU_DIR)), key=natural_key)
