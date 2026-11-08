"""Run ONE diversity task -- (planner, instance, k, q, diversity model) -- and
dump a JSON result. This is what every slurm array index executes.

It is the only stage that imports ``unified_planning`` and the planners. The
structure follows pyPMTEvalToolkit's runner:

* the task's own time and memory limits are armed *inside* the process, a
  little under the scheduler's, so running out of either comes back as a
  ``TIMEOUT``/``MEMOUT`` row -- with the plans found up to that point -- rather
  than as a missing file; slurm's limits stay as the backstop;
* every run works in a private scratch directory;
* a ``.running`` marker is left while it runs, so a job killed outright is
  counted as ``KILLED`` instead of vanishing from the denominators.

After planning, every returned plan is validated against the original problem
and the plan set is scored by :mod:`dp_eval_toolkit.behaviours` under the
task's diversity model -- the same evaluator for every planner.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import re
import resource
import signal
import socket
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Result statuses.
SOLVED = 'SOLVED'              # k plans
EXHAUSTED = 'EXHAUSTED'        # fewer than k plans, and the planner says that is all within q
UNSOLVABLE = 'UNSOLVABLE'
TIMEOUT = 'TIMEOUT'
MEMOUT = 'MEMOUT'
UNSUPPORTED = 'UNSUPPORTED'
ERROR = 'ERROR'
KILLED = 'KILLED'

_OUTCOME_STATUS = {
    'COMPLETE': SOLVED, 'EXHAUSTED': EXHAUSTED, 'UNSOLVABLE': UNSOLVABLE, 'TIMEOUT': TIMEOUT,
    'MEMOUT': MEMOUT, 'UNSUPPORTED': UNSUPPORTED, 'ERROR': ERROR,
}


class _LimitReached(Exception):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def parse_feature(text: str) -> Tuple[str, Any]:
    """``go`` -> ('go', {}); ``ru=/path/res.txt`` -> ('ru', '/path/res.txt');
    ``key={"a": 1}`` -> JSON-decoded config."""
    key, _, config = text.partition('=')
    if not config:
        return key, {}
    if config[:1] in '{[':
        try:
            return key, json.loads(config)
        except ValueError:
            pass
    return key, config


def solve(args) -> int:
    started_wall = time.time()
    started = time.monotonic()
    for attribute in ('planner_cfg', 'domain', 'problem', 'results_dir', 'errors_dir', 'run_dir'):
        value = getattr(args, attribute, None)
        if value:
            setattr(args, attribute, os.path.abspath(os.path.expanduser(value)))

    with open(args.planner_cfg) as handle:
        planner_cfg = json.load(handle)
    tag = planner_cfg.get('planner-tag') or os.path.splitext(os.path.basename(args.planner_cfg))[0]
    adapter_name = planner_cfg.get('adapter')
    params = dict(planner_cfg.get('planner-params') or {})
    features = [parse_feature(f) for f in (args.feature or [])]

    slug = _slug(args.task_id)
    os.makedirs(args.results_dir, exist_ok=True)
    result_file = os.path.join(args.results_dir, f'{slug}.json')
    marker_file = os.path.join(args.results_dir, f'{slug}.running')

    result: Dict[str, Any] = {
        'task': {
            'task-id': args.task_id, 'suite': args.suite, 'domain': args.domain_name,
            'instance': args.instance, 'track': args.track, 'ipc': args.ipc,
            'domain-file': args.domain, 'problem-file': args.problem,
        },
        'diversity': {
            'k': args.k, 'q': args.q, 'model': args.model,
            'features': [[k, c] for k, c in features],
        },
        'planner': {'tag': tag, 'adapter': adapter_name, 'params': _jsonable(params),
                    'config-file': args.planner_cfg},
        'limits': {'time-seconds': args.time_limit, 'memory-mb': args.memory_limit},
        'run': {
            'host': socket.gethostname(), 'pid': os.getpid(), 'started': started_wall,
            'slurm-job-id': os.environ.get('SLURM_JOB_ID'),
            'slurm-array-job-id': os.environ.get('SLURM_ARRAY_JOB_ID'),
            'slurm-array-task-id': os.environ.get('SLURM_ARRAY_TASK_ID'),
            'python': platform.python_version(),
        },
        'status': ERROR,
        'adapter-outcome': None,
        'timings': {},
        'plans': [],
        'metrics': {},
        'diversity-scores': None,
        'native-behaviours': None,
        'info': {},
        'logs': [],
    }

    _write_marker(marker_file, result)
    _install_limits(args.time_limit, args.memory_limit)
    workdir = _enter_workdir(args.run_dir, tag, slug)
    result['run']['work-dir'] = workdir
    deadline = started + args.time_limit if args.time_limit and args.time_limit > 0 else None

    problem = None
    plans: List[Any] = []
    ctx = None
    try:
        import unified_planning.shortcuts as ups
        from unified_planning.io import PDDLReader

        from .adapters import load_adapter
        from .adapters.base import PlanningContext

        environment = ups.get_environment()
        environment.credits_stream = None
        environment.error_used_name = False

        adapter = load_adapter(adapter_name, params)
        parse_start = time.monotonic()
        problem = PDDLReader().parse_problem(args.domain, args.problem)
        result['timings']['parse-seconds'] = time.monotonic() - parse_start
        result['metrics']['problem-kind'] = sorted(problem.kind.features)

        ctx = PlanningContext(problem=problem, domain_file=args.domain, problem_file=args.problem,
                              k=args.k, q=args.q, features=features, deadline=deadline,
                              workdir=workdir or os.getcwd(), track=args.track)
        reason = adapter.unsupported_reason(ctx)
        if reason:
            result['status'] = UNSUPPORTED
            result['logs'].append(reason)
        else:
            solve_start = time.monotonic()
            run = adapter.plan(ctx)
            signal.alarm(0)
            result['timings']['solve-seconds'] = time.monotonic() - solve_start
            result['adapter-outcome'] = run.outcome
            result['status'] = _OUTCOME_STATUS.get(run.outcome, ERROR)
            result['logs'] += [str(line) for line in run.logs]
            result['info'] = _jsonable(run.info)
            result['native-behaviours'] = run.native_behaviours
            plans = list(run.plans)

    except _LimitReached as limit:
        result['status'] = limit.status
        result['logs'].append(limit.detail)
        plans = ctx.plans_so_far() if ctx is not None else []
    except MemoryError:
        result['status'] = MEMOUT
        result['logs'].append(f'MemoryError under a {args.memory_limit}MB limit.')
        plans = ctx.plans_so_far() if ctx is not None else []
    except Exception as error:                                  # noqa: BLE001 -- reported
        result['status'] = ERROR
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
        result['logs'].append(f'{type(error).__name__}: {error}')
        _dump_traceback(args, tag, slug, error)
        plans = ctx.plans_so_far() if ctx is not None else []
    finally:
        signal.alarm(0)

    # The planning clock stops here: validation and behaviour evaluation are the
    # harness's work, not the planner's, and are not charged to it.
    result['timings']['planning-seconds'] = time.monotonic() - started
    result['metrics']['peak-memory-mb'] = _peak_memory_mb()
    gc.collect()

    if problem is not None and plans and result['status'] != KILLED:
        _post_process(args, problem, plans, features, result)

    _leave_workdir(workdir, keep=getattr(args, 'keep_run_dir', False))
    return _finish(result, result_file, marker_file, started)


def _post_process(args, problem, plans: List[Any], features, result: Dict[str, Any]) -> None:
    """Validate, write out and score the plans, under a time limit of its own."""
    eval_limit = int(getattr(args, 'eval_time_limit', 0) or 0)
    _arm(eval_limit, TIMEOUT, 'Plan validation / behaviour evaluation hit its own time limit.')
    eval_start = time.monotonic()
    try:
        result['plans'] = [_plan_lines(problem, p) for p in plans]
        result['metrics']['plan-count'] = len(plans)
        result['metrics']['plan-lengths'] = [len(getattr(p, 'actions', [])) for p in plans]
        if getattr(args, 'validate', True):
            verdicts = [_validate(problem, p, result) for p in plans]
            result['metrics']['valid-plans'] = sum(1 for v in verdicts if v is True)
            result['metrics']['invalid-plans'] = sum(1 for v in verdicts if v is False)
            result['metrics']['unvalidated-plans'] = sum(1 for v in verdicts if v is None)
            result['metrics']['validated'] = (False if any(v is False for v in verdicts) else
                                              None if any(v is None for v in verdicts) else True)
        if features:
            from .behaviours import evaluate_plan_set
            result['diversity-scores'] = evaluate_plan_set(problem, plans, features)
    except _LimitReached as limit:
        result['logs'].append(limit.detail)
        result['metrics']['evaluation-incomplete'] = True
    except Exception as error:                                  # noqa: BLE001
        result['logs'].append(f'evaluation failed: {type(error).__name__}: {error}')
        result['metrics']['evaluation-incomplete'] = True
        _dump_traceback(args, result['planner']['tag'], _slug(args.task_id) + '.eval', error)
    finally:
        signal.alarm(0)
        result['timings']['evaluation-seconds'] = time.monotonic() - eval_start


# ----------------------------------------------------------------------
# Limits
# ----------------------------------------------------------------------

def _install_limits(time_limit: Optional[int], memory_limit: Optional[int]) -> None:
    for signame, status, detail in (
        ('SIGALRM', TIMEOUT, 'Hit the task time limit.'),
        ('SIGTERM', KILLED, 'Received SIGTERM (scheduler cancellation).'),
        ('SIGXCPU', TIMEOUT, 'Received SIGXCPU (CPU limit).'),
        ('SIGUSR1', KILLED, 'Received SIGUSR1 (slurm --signal warning).'),
    ):
        signum = getattr(signal, signame, None)
        if signum is None:
            continue
        try:
            signal.signal(signum, _make_handler(status, detail))
        except (ValueError, OSError):
            pass
    if time_limit and time_limit > 0:
        signal.alarm(int(time_limit))
    if memory_limit and memory_limit > 0:
        limit_bytes = int(memory_limit) * 1024 * 1024
        for limit_name in ('RLIMIT_AS', 'RLIMIT_DATA'):
            rlimit = getattr(resource, limit_name, None)
            if rlimit is None:
                continue
            try:
                _soft, hard = resource.getrlimit(rlimit)
                capped = limit_bytes if hard in (resource.RLIM_INFINITY, -1) else min(limit_bytes, hard)
                resource.setrlimit(rlimit, (capped, hard))
                break
            except (ValueError, OSError):
                continue


def _arm(seconds: int, status: str, detail: str) -> None:
    try:
        signal.signal(signal.SIGALRM, _make_handler(status, detail))
    except (ValueError, OSError):
        return
    if seconds > 0:
        signal.alarm(seconds)


def _make_handler(status: str, detail: str):
    def handler(signum, frame):                                  # noqa: ARG001
        raise _LimitReached(status, detail)
    return handler


def _peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == 'darwin' else 1024.0
    return round(max(peak, children) / divisor, 2)


# ----------------------------------------------------------------------
# Working directory
# ----------------------------------------------------------------------

def _enter_workdir(run_dir: Optional[str], tag: str, slug: str) -> Optional[str]:
    if not run_dir:
        return None
    try:
        workdir = os.path.join(run_dir, f'{_slug(tag)}__{slug}.{os.getpid()}')
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)
        return workdir
    except OSError:
        return None


def _leave_workdir(workdir: Optional[str], keep: bool = False) -> None:
    if not workdir:
        return
    import shutil
    try:
        os.chdir(os.path.dirname(workdir) or '/')
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
    except OSError:
        pass


# ----------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------

def _plan_lines(problem, plan) -> List[str]:
    try:
        from unified_planning.io import PDDLWriter
        text = PDDLWriter(problem).get_plan(plan)
    except Exception:                                            # noqa: BLE001
        text = str(plan)
    return [line for line in text.splitlines() if line.strip()]


def _rebind_plan(problem, plan, result: Dict[str, Any]):
    """Point the plan's action instances at *this* problem's actions (by name and
    arity). Planners solve compiled copies of the problem and lift their plans
    back; UP identifies an action by its whole definition, so without this the
    validator rejects such a plan without checking a single precondition."""
    from unified_planning.plans import ActionInstance

    unmatched: List[str] = []

    def rebind(instance):
        action = instance.action
        original = problem.action(action.name) if problem.has_action(action.name) else None
        if original is None or len(original.parameters) != len(instance.actual_parameters):
            unmatched.append(action.name)
            return instance
        if original is action:
            return instance
        return ActionInstance(original, instance.actual_parameters)

    rebound = plan.replace_action_instances(rebind)
    if unmatched:
        result['logs'].append('Validation unavailable: the plan names actions this problem does '
                              f'not have: {", ".join(sorted(set(unmatched)))}')
        return None
    return rebound


def _validate(problem, plan, result: Dict[str, Any]) -> Optional[bool]:
    try:
        import unified_planning.shortcuts as ups
        plan = _rebind_plan(problem, plan, result)
        if plan is None:
            return None
        with ups.PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
            outcome = validator.validate(problem, plan)
        status = getattr(getattr(outcome, 'status', None), 'name', str(outcome))
        if status != 'VALID':
            messages = [getattr(m, 'message', str(m)) for m in getattr(outcome, 'log_messages', None) or []]
            result['logs'].append(f'Validation failed: {"; ".join(messages) or status}')
        return status == 'VALID'
    except Exception as error:                                   # noqa: BLE001
        result['logs'].append(f'Validation unavailable: {type(error).__name__}: {error}')
        return None


# ----------------------------------------------------------------------
# Result files
# ----------------------------------------------------------------------

def _finish(result: Dict[str, Any], result_file: str, marker_file: str, started: float) -> int:
    signal.alarm(0)
    result['timings']['total-seconds'] = time.monotonic() - started
    _write_json(result_file, result)
    if os.path.exists(marker_file):
        try:
            os.remove(marker_file)
        except OSError:
            pass
    scores = result.get('diversity-scores') or {}
    print(f"{result['status']:<12} {result['planner']['tag']} {result['task']['task-id']} "
          f"k={result['diversity']['k']} q={result['diversity']['q']} model={result['diversity']['model']} "
          f"plans={len(result['plans'])} b-cov={scores.get('b-coverage', '-')} "
          f"[{result['timings'].get('planning-seconds', 0):.1f}s] -> {result_file}")
    return 0


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, path)


def _write_marker(path: str, result: Dict[str, Any]) -> None:
    try:
        _write_json(path, {k: result[k] for k in ('task', 'diversity', 'planner', 'run', 'limits')})
    except OSError:
        pass


def _dump_traceback(args, tag: str, slug: str, error: Exception) -> None:
    errors_dir = getattr(args, 'errors_dir', None)
    if not errors_dir:
        return
    try:
        os.makedirs(errors_dir, exist_ok=True)
        label = f'{_slug(tag)}__{_slug(str(args.model))}__k{args.k}__q{args.q}__{slug}'
        with open(os.path.join(errors_dir, f'{label}.log'), 'w') as handle:
            handle.write(f'{args.task_id}\n{type(error).__name__}: {error}\n\n')
            handle.write(traceback.format_exc())
    except OSError:
        pass


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _slug(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(text))
