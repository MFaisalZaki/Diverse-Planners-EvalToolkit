"""Planning tasks: the record every stage passes around, and how a benchmark's
tasks are filtered down to an experiment's.

Finding tasks in a benchmark repository is :mod:`dp_eval_toolkit.discovery`'s
job; this module only knows what a task is and which ones to keep.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from fnmatch import fnmatch

TRACKS = ('classical', 'numeric', 'temporal')

_TRAILING_NUM_RE = re.compile(r'(\d+)(?!.*\d)')
_NATURAL_RE = re.compile(r'(\d+)')
_UNSAFE_RE = re.compile(r'[^A-Za-z0-9._-]+')


@dataclass(frozen=True)
class Task:
    """One (domain file, problem file) pair, with everything needed to run and
    to name it."""

    suite: str            # label of the repository it came from
    domain: str           # domain name
    instance: str         # instance name (problem file stem)
    domain_file: str
    problem_file: str
    track: str            # 'classical' | 'numeric' | 'temporal'
    ipc: str | None = None
    # The api.py convention of the behaviour-planning paper experiments: the
    # domain ``name`` the api.py gives, and the 1-based position of the instance
    # once the api.py's problems are sorted by path. None outside api.py layouts.
    api_name: str | None = None
    api_index: int | None = None

    @property
    def task_id(self) -> str:
        return f'{self.suite}:{self.domain}:{self.instance}'

    @property
    def slug(self) -> str:
        """Filesystem-safe form of :attr:`task_id`, used to name result files."""
        return slugify(self.task_id)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['task_id'] = self.task_id
        return d


# ----------------------------------------------------------------------
# Filtering and selection
# ----------------------------------------------------------------------

def filter_tasks(tasks: Iterable[Task],
                 tracks: Sequence[str] | None = None,
                 include: Sequence[str] | None = None,
                 exclude: Sequence[str] | None = None,
                 ipc_years: Sequence[str] | None = None,
                 selected: Sequence[Sequence[str]] | None = None) -> list[Task]:
    """Keep the tasks matching every given filter.

    `include`/`exclude` are glob patterns matched against both the bare domain
    name and ``suite:domain``, so ``--domains 'ipc-2002:*-time-*'`` and
    ``--domains counters`` both work.

    `selected` is the explicit ``[ipc-year, domain, instance]`` list of the
    older toolkit's ``selected-tasks``; when given, nothing outside it survives.
    """
    predicates: list[Callable[[Task], bool]] = []
    if tracks:
        wanted_tracks = set(tracks)
        predicates.append(lambda t: t.track in wanted_tracks)
    if ipc_years:
        wanted_years = {str(y) for y in ipc_years}
        predicates.append(lambda t: str(t.ipc) in wanted_years)
    if selected:
        chosen = {tuple(str(part) for part in entry) for entry in selected}
        predicates.append(lambda t: (str(t.ipc), t.domain, t.instance) in chosen)
    if include:
        predicates.append(lambda t: _matches_any(t, include))
    if exclude:
        predicates.append(lambda t: not _matches_any(t, exclude))
    return [task for task in tasks if all(keep(task) for keep in predicates)]


InstanceRef = tuple[str, str, int]      # (ipc year or 'None', api.py domain name, 1-based instance number)


def parse_instance_ref(raw) -> InstanceRef:
    """``"(2000, schedule, 9)"`` or ``[2000, "schedule", 9]`` -> ``('2000', 'schedule', 9)``.

    The written form of the paper experiments' ``classical_instances`` list; a
    missing IPC year is spelled ``None`` there and matched as such.
    """
    if isinstance(raw, str):
        raw = raw.strip().strip('()[]').split(',')
    parts = [str(part).strip() for part in raw]
    if len(parts) != 3:
        raise ValueError(f'instance reference {raw!r} is not (ipc-year, domain, number)')
    year, domain, number = parts
    return str(year).strip('\'" '), str(domain).strip('\'" '), int(number)


def select_instances(tasks: Iterable[Task], refs: Iterable) -> list[Task]:
    """The tasks the paper experiments' instance list names, one per reference.

    A reference ``(year, domain, n)`` is matched against a task's IPC year, its
    api.py ``name`` and its 1-based position in the api.py's problems sorted by
    path. classical-domains holds several directories per (year, name) -- the
    ``opt`` and ``sat`` tracks of an IPC, ``strips`` beside ``full``/``adl``
    encodings -- and the paper's generator took whichever its directory walk met
    first. Here the choice is fixed: ``adl`` directories are skipped, then the
    ``strips`` variant is preferred, then ``opt`` over ``sat``, then the
    alphabetically first directory.
    """
    wanted = {parse_instance_ref(ref) for ref in refs}
    candidates: dict[InstanceRef, list[Task]] = {}
    for task in tasks:
        if task.api_index is None or 'adl' in task.domain.lower():
            continue
        key = (str(task.ipc), str(task.api_name), task.api_index)
        if key in wanted:
            candidates.setdefault(key, []).append(task)
    return sort_tasks(min(options, key=_variant_preference) for options in candidates.values())


def _variant_preference(task: Task) -> tuple[bool, bool, str]:
    domain = task.domain.lower()
    return ('strips' not in domain, 'opt' not in domain, domain)


def _matches_any(task: Task, patterns: Sequence[str]) -> bool:
    names = (task.domain, f'{task.suite}:{task.domain}')
    return any(fnmatch(name, pattern) for pattern in patterns for name in names)


def limit_per_domain(tasks: Sequence[Task], limit: int | None,
                     strategy: str = 'even') -> list[Task]:
    """Cap each domain at `limit` instances.

    ``even`` samples across the instance range, which keeps the easy *and* the
    hard end of a domain in the experiment; ``first`` takes the N smallest by
    natural order, which is what you want for a smoke run.
    """
    if not limit or limit <= 0:
        return list(tasks)

    by_domain: dict[str, list[Task]] = {}
    for task in tasks:
        by_domain.setdefault(f'{task.suite}:{task.domain}', []).append(task)

    selected: list[Task] = []
    for domain_tasks in by_domain.values():
        ordered = sorted(domain_tasks, key=lambda t: natural_key(t.instance))
        selected.extend(_sample(ordered, limit, strategy))
    return sort_tasks(selected)


def _sample(ordered: Sequence[Task], limit: int, strategy: str) -> list[Task]:
    if len(ordered) <= limit:
        return list(ordered)
    if strategy == 'first':
        return list(ordered[:limit])
    step = (len(ordered) - 1) / float(limit - 1) if limit > 1 else 0.0
    picked = {round(i * step) for i in range(limit)}
    return [ordered[i] for i in sorted(picked)]


def sort_tasks(tasks: Iterable[Task]) -> list[Task]:
    return sorted(tasks, key=lambda t: (t.suite, t.track, t.domain, natural_key(t.instance)))


def summarize(tasks: Sequence[Task]) -> dict[str, dict[str, int]]:
    """Task counts per track and per suite, for the ``discover`` report."""
    domains_per_track: dict[str, set[str]] = {}
    for task in tasks:
        domains_per_track.setdefault(task.track, set()).add(f'{task.suite}:{task.domain}')
    return {
        'instances_per_track': dict(Counter(task.track for task in tasks)),
        'instances_per_suite': dict(Counter(task.suite for task in tasks)),
        'domains_per_track': {track: len(domains) for track, domains in domains_per_track.items()},
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def natural_key(text: str) -> list:
    """Sort key that orders ``instance-2`` before ``instance-10``."""
    return [int(part) if part.isdigit() else part.lower()
            for part in _NATURAL_RE.split(str(text))]


def slugify(text: str) -> str:
    """Filesystem-safe form of `text`: anything outside ``[A-Za-z0-9._-]`` becomes ``_``."""
    return _UNSAFE_RE.sub('_', str(text))


def trailing_number(text: str) -> str | None:
    """The last run of digits in `text`, as written (``'p007'`` -> ``'007'``)."""
    match = _TRAILING_NUM_RE.search(text)
    return match.group(1) if match else None
