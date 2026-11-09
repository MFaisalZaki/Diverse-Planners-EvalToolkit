"""Discovery of planning tasks in a benchmark repository.

Three repository layouts are recognised, so the same command works on all of
the benchmark sets we care about:

``api.py``   a domain directory with an ``api.py`` that defines a ``domains``
             list of ``{'name', 'ipc', 'problems': [(domain, problem), ...]}``
             dicts, with the paths relative to the *parent* of that directory.
             This is `AI-Planning/classical-domains
             <https://github.com/AI-Planning/classical-domains>`_ and
             `pyPMT/numeric-domains <https://github.com/pyPMT/numeric-domains>`_.

``instances``  a domain directory holding ``instances/*.pddl`` plus either a
             single ``domain.pddl`` or a ``domains/domain-<n>.pddl`` per
             instance (airport and friends). This is
             `potassco/pddl-instances <https://github.com/potassco/pddl-instances>`_.

``flat``     a directory with a ``domain.pddl`` and its problems as siblings —
             the shape of most hand-made task sets.

The track a task belongs to is decided by *reading the domain file*, not by
which repository it came from: a repo named "classical-domains" still holds
domains with ``:functions``, and the temporal instances sit in the middle of
the IPC tree next to their non-temporal siblings. Sniffing keeps the three
tracks meaningful whatever you point the tool at.
"""

from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Iterable, Sequence

from .tasks import Task, natural_key, sort_tasks, trailing_number

# Directories that never hold tasks; pruned from the walk.
_PRUNED = {'.git', '.github', '__pycache__', '.idea', '.vscode', 'generator',
           'generators', 'venv', '.venv', 'node_modules'}

_COMMENT_RE = re.compile(r';[^\n]*')
_IPC_YEAR_RE = re.compile(r'ipc[-_]?(\d{4})', re.IGNORECASE)

# How much of a domain file is read to classify it.
_CLASSIFY_BYTES = 2 * 1024 * 1024
_TEMPORAL_MARKERS = (':durative-action', ':process', ':event')
# Action costs are classical planning: a functions block declaring only
# total-cost does not make a domain numeric.
_TOTAL_COST_RE = re.compile(r'\(\s*total-cost\s*\)(\s*-\s*number)?')


# ----------------------------------------------------------------------
# Track classification
# ----------------------------------------------------------------------

def classify_domain(domain_file: str) -> str:
    """Read `domain_file` and say which track it belongs to.

    Durative actions (or PDDL+ processes/events) make it temporal; a
    ``:numeric-fluents`` requirement or a ``(:functions ...)`` block declaring
    anything beyond ``total-cost`` makes it numeric; anything else -- action
    costs included -- is classical. Comments are stripped first so a
    commented-out ``:durative-action`` does not mislabel a whole domain.
    """
    try:
        with open(domain_file, errors='replace') as handle:
            text = handle.read(_CLASSIFY_BYTES)
    except OSError:
        return 'classical'
    text = _COMMENT_RE.sub(' ', text).lower()
    if any(marker in text for marker in _TEMPORAL_MARKERS):
        return 'temporal'
    if ':numeric-fluents' in text or _declares_numeric_fluents(text):
        return 'numeric'
    return 'classical'


def _declares_numeric_fluents(text: str) -> bool:
    """Does the (lower-cased, comment-free) domain text declare a function other than ``total-cost``?"""
    start = text.find('(:functions')
    if start < 0:
        return False
    depth = 0
    for position in range(start, len(text)):
        depth += text[position] == '('
        depth -= text[position] == ')'
        if depth == 0:
            block = text[start + len('(:functions'):position]
            return '(' in _TOTAL_COST_RE.sub('', block)
    return True   # unbalanced block: assume the worst


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

def discover(root: str, suite: str | None = None,
             skipped: list[tuple[str, str]] | None = None) -> list[Task]:
    """Find every task under `root`. `suite` defaults to its directory name.

    `skipped` collects ``(directory, reason)`` for directories that look like a
    domain but yielded nothing, so a benchmark set that silently contributes
    fewer instances than it holds shows up in the report instead of in nobody's
    coverage denominator.
    """
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise NotADirectoryError(f'benchmark directory not found: {root}')
    suite = suite or os.path.basename(root.rstrip(os.sep))

    tasks: list[Task] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _PRUNED)
        found, looks_like_domain = _tasks_in_directory(dirpath, dirnames, filenames, suite)
        if looks_like_domain and not found and skipped is not None:
            skipped.append((dirpath, _skip_reason(dirpath, dirnames, filenames)))
        tasks.extend(found)

    tasks = _unique_pairs(tasks)
    if skipped:
        skipped[:] = _without_productive_parents(skipped, tasks)
    return sort_tasks(tasks)


def _tasks_in_directory(dirpath: str, dirnames: Sequence[str], filenames: Sequence[str],
                        suite: str) -> tuple[list[Task], bool]:
    """The tasks one directory holds, and whether it looked like a domain at all.

    The three layouts are tried in order. An api.py that resolves to nothing
    does not veto the directory: a few domains in the wild point their api.py
    at paths that were renamed or never shipped, and the instances are sitting
    right there on disk.
    """
    looks_like_domain = False
    if 'api.py' in filenames:
        looks_like_domain = True
        found = _from_api(dirpath, suite)
        if found:
            return found, True
    if 'instances' in dirnames:
        looks_like_domain = True
        found = _from_instances_dir(dirpath, suite)
        if found:
            return found, True
    shared = _shared_domain_file(dirpath)
    # A domain file with no problems beside it is not a flat domain directory --
    # most often it is an `instances/` directory whose one file the parent
    # already took as a problem.
    if shared and any(_is_problem_file(f, shared) for f in filenames):
        return _from_flat_dir(dirpath, filenames, suite, shared), True
    return [], looks_like_domain


def _is_problem_file(name: str, domain_file: str) -> bool:
    return name.endswith('.pddl') and name != os.path.basename(domain_file)


def _unique_pairs(tasks: Iterable[Task]) -> list[Task]:
    """Drop tasks whose (domain, problem) files were already seen under other paths."""
    seen: set[tuple[str, str]] = set()
    unique: list[Task] = []
    for task in tasks:
        key = (os.path.realpath(task.domain_file), os.path.realpath(task.problem_file))
        if key not in seen:
            seen.add(key)
            unique.append(task)
    return unique


def _without_productive_parents(skipped: Sequence[tuple[str, str]],
                                tasks: Sequence[Task]) -> list[tuple[str, str]]:
    """A directory whose *children* turned out to hold the tasks (one
    sub-directory per instance) is not a skip; drop those entries."""
    productive = {os.path.dirname(t.problem_file) for t in tasks}
    return [(directory, reason) for directory, reason in skipped
            if not any(p.startswith(directory + os.sep) for p in productive)]


def _skip_reason(dirpath: str, dirnames: Sequence[str], filenames: Sequence[str]) -> str:
    if 'api.py' in filenames:
        return 'api.py lists no existing (domain, problem) pair'
    if 'instances' in dirnames:
        return 'instances/ present but no domain file could be identified'
    return 'no problem files beside domain.pddl'


# -- api.py layout -------------------------------------------------------

def _from_api(dirpath: str, suite: str) -> list[Task]:
    """Load ``<dirpath>/api.py`` and read its ``domains`` list.

    Paths inside it are relative to the parent of the domain directory
    (``agricola-opt18/domain.pddl`` living in ``classical/agricola-opt18``).
    """
    tasks: list[Task] = []
    for position, entry in enumerate(_api_domain_entries(os.path.join(dirpath, 'api.py'))):
        name = str(entry.get('name') or os.path.basename(dirpath))
        ipc = _clean_ipc(entry.get('ipc')) or _ipc_from_path(dirpath)
        domain = _qualified_domain_name(name, dirpath)
        for domain_file, problem_file, index in _api_problem_pairs(entry, dirpath):
            tasks.append(Task(suite=suite, domain=domain, instance=_stem(problem_file),
                              domain_file=domain_file, problem_file=problem_file,
                              track=classify_domain(domain_file), ipc=ipc,
                              # The paper experiments read only the first entry of an api.py.
                              api_name=name, api_index=index if position == 0 else None))
    return tasks


def _api_domain_entries(api: str) -> list[dict]:
    """The ``domains`` list of an api.py; empty when it cannot be loaded.

    A broken api.py should cost us that one domain, not the whole walk.
    """
    module_name = f'dpeval_api_{abs(hash(api))}'
    spec = importlib.util.spec_from_file_location(module_name, api)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        entries = list(getattr(module, 'domains', []))
    except Exception:
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _api_problem_pairs(entry: dict, dirpath: str) -> list[tuple[str, str, int]]:
    """The existing ``(domain_file, problem_file, index)`` triples an api.py
    entry lists, where ``index`` numbers the entry's problems from 1 in path
    order -- the paper experiments' instance number, counted before any missing
    file is dropped so that it stays comparable.

    A domain's api.py must reference its own directory. Some in the wild are
    copy-pasted from a sibling and resolve to *that* domain's files -- taking
    them would file a hundred instances under the wrong domain and hide the
    ones actually sitting here. Ignoring them lets the directory-based
    discovery find the real instances.
    """
    base = os.path.dirname(dirpath)
    inside = os.path.realpath(dirpath) + os.sep
    problems = [p for p in entry.get('problems') or [] if isinstance(p, (list, tuple)) and len(p) >= 2]
    pairs: list[tuple[str, str, int]] = []
    for index, problem in enumerate(sorted(problems, key=lambda p: str(p[1])), start=1):
        domain_file = os.path.join(base, str(problem[0]))
        problem_file = os.path.join(base, str(problem[1]))
        if not (os.path.isfile(domain_file) and os.path.isfile(problem_file)):
            continue
        if os.path.realpath(problem_file).startswith(inside):
            pairs.append((domain_file, problem_file, index))
    return pairs


# -- instances/ layout ---------------------------------------------------

def _from_instances_dir(dirpath: str, suite: str) -> list[Task]:
    """IPC layout: ``instances/*.pddl`` beside one ``domain.pddl`` or a
    ``domains/`` directory with one domain file per instance."""
    problems = _pddl_files(os.path.join(dirpath, 'instances'))
    if not problems:
        return []

    shared = _shared_domain_file(dirpath)
    per_instance = _per_instance_domain_files(os.path.join(dirpath, 'domains'))
    domain_name = os.path.basename(dirpath)
    ipc = _ipc_from_path(dirpath)

    tasks: list[Task] = []
    for problem_file in problems:
        domain_file = _domain_file_for(problem_file, per_instance, shared)
        if domain_file is None:
            continue
        tasks.append(Task(suite=suite, domain=domain_name, instance=_stem(problem_file),
                          domain_file=domain_file, problem_file=problem_file,
                          track=classify_domain(domain_file), ipc=ipc))
    return tasks


def _pddl_files(directory: str) -> list[str]:
    """The ``.pddl`` files of `directory` in natural order; empty if it is not one."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted((os.path.join(directory, f) for f in names if f.endswith('.pddl')), key=natural_key)


def _per_instance_domain_files(domains_dir: str) -> dict[str, str]:
    """``instance number -> domain file`` for a ``domains/domain-<n>.pddl`` layout."""
    per_instance: dict[str, str] = {}
    for path in _pddl_files(domains_dir):
        number = trailing_number(_stem(path))
        if number is not None:
            per_instance[number] = path
    return per_instance


def _domain_file_for(problem_file: str, per_instance: dict[str, str], shared: str | None) -> str | None:
    """The domain file an instance uses: its numbered one, the shared one, or
    the single per-instance file when there is only one."""
    number = trailing_number(_stem(problem_file))
    domain_file = per_instance.get(number or '', shared)
    if domain_file is None and len(per_instance) == 1:
        domain_file = next(iter(per_instance.values()))
    if domain_file is None or not os.path.isfile(domain_file):
        return None
    return domain_file


def _shared_domain_file(dirpath: str) -> str | None:
    """The one domain file all instances of `dirpath` share, if there is one.

    Usually ``domain.pddl``; some sets name it after the variant instead
    (``domain_snp.pddl``), and a domain directory holding exactly one PDDL file
    beside its ``instances/`` leaves no ambiguity either way.
    """
    default = os.path.join(dirpath, 'domain.pddl')
    if os.path.isfile(default):
        return default
    candidates = sorted(os.path.basename(p) for p in _pddl_files(dirpath) if os.path.isfile(p))
    # `korf1_domain.pddl` beside `korf1_problem.pddl`: one instance per
    # directory, the domain file named after it.
    for pick in ([f for f in candidates if f.lower().startswith('domain')],
                 [f for f in candidates if 'domain' in f.lower()],
                 candidates):
        if len(pick) == 1:
            return os.path.join(dirpath, pick[0])
    return None


# -- flat layout ---------------------------------------------------------

def _from_flat_dir(dirpath: str, filenames: Sequence[str], suite: str,
                   domain_file: str | None = None) -> list[Task]:
    """A domain file with its problems as siblings."""
    domain_file = domain_file or os.path.join(dirpath, 'domain.pddl')
    if not os.path.isfile(domain_file):
        return []
    track = classify_domain(domain_file)
    ipc = _ipc_from_path(dirpath)
    domain, instance_prefix = _flat_names(dirpath)
    problems = sorted((f for f in filenames if _is_problem_file(f, domain_file)), key=natural_key)
    return [Task(suite=suite, domain=domain,
                 instance=_flat_instance_name(name, instance_prefix, len(problems)),
                 domain_file=domain_file, problem_file=os.path.join(dirpath, name),
                 track=track, ipc=ipc)
            for name in problems]


def _flat_instance_name(filename: str, instance_prefix: str | None, problem_count: int) -> str:
    """One directory per instance (``instances/korf1/korf1_problem.pddl``):
    the directory names the instance, not the file inside it."""
    if not instance_prefix:
        return _stem(filename)
    if problem_count == 1:
        return instance_prefix
    return f'{instance_prefix}-{_stem(filename)}'


def _flat_names(dirpath: str) -> tuple[str, str | None]:
    """``(domain name, instance prefix)`` for a flat directory.

    A directory sitting directly under ``instances/`` is one instance of the
    domain above it, not a domain of its own -- otherwise a set like 15-puzzle
    turns into a hundred one-instance "domains" called ``korf1``..``korf100``.
    """
    basename = os.path.basename(dirpath.rstrip(os.sep))
    parent = os.path.dirname(dirpath.rstrip(os.sep))
    if os.path.basename(parent) == 'instances':
        return os.path.basename(os.path.dirname(parent)), basename
    return basename, None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _ipc_from_path(path: str) -> str | None:
    match = _IPC_YEAR_RE.search(path)
    return match.group(1) if match else None


def _clean_ipc(value) -> str | None:
    if value in (None, '', 'None'):
        return None
    return str(value)


def _qualified_domain_name(name: str, dirpath: str) -> str:
    """Prefer the directory name when it is more specific than ``api.py``'s.

    ``classical-domains`` gives every IPC variant of a domain the same ``name``
    (``agricola`` for both ``agricola-opt18`` and ``agricola-sat18``), which
    would collapse distinct benchmark sets into one row of the coverage table.
    The directory name distinguishes them.
    """
    basename = os.path.basename(dirpath.rstrip(os.sep))
    if basename and name.lower() in basename.lower() and basename != name:
        return basename
    return name or basename
