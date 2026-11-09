"""Forbid-Iterative (FI) diverse planner, driven through its command line.

This is the baseline of the Behaviour Planning paper experiments
(``paper_experiments/run-benchmark.py`` in pyBehaviourPlanningSMT): the
``forbiditerative`` package's top-quality planner, asked for ``k`` plans within
quality bound ``q``. FI does not take a diversity model: it is asked for ``k``
plans and, as its top-quality search usually finds more than that within the
bound, ``select`` picks the ``k`` most diverse of them under the task's model
with BehaviourDiversityCounter -- the ``fi-bc`` baseline of the paper. Without
``select`` the first ``k`` plans are kept. Either way the plans are scored by the
toolkit's behaviour evaluator like everyone else's.

``planner-params``:

``select``             pick k plans from FI's pool by ``B-Coverage``, ``B-MaxSum``,
                       ``B-MaxMin`` or ``B-Novelty`` (default: none, keep the first k)
``pool-size``          how many plans to ask FI for: a count, or a multiple of k such
                       as ``"3x"`` (default: k, as in the paper -- then ``select`` only
                       matters when FI returns more plans than asked); both as in
                       :class:`~.base.DiversePlannerAdapter`
``planner``            FI planner configuration (default ``extended_unordered_topq``)
``symmetries``         pass ``--symmetries`` (default true)
``extra-args``         list of further command-line arguments, verbatim
``python``             interpreter to run FI with (default: this one)
``time-margin``        seconds kept back from the budget for reading plans (default 10)
``local-dir``          where FI keeps its intermediate files (default /tmp/planner_runs;
                       a node-local scratch such as $TMPDIR is a good choice on a cluster)
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from .base import DiversePlannerAdapter, DiverseRun, PlanningContext

# forbiditerative's --use-local-folder works in /tmp/planner_runs/<uuid> and only
# copies the plans back to the cwd when it finishes cleanly: on a timeout the plans
# stay behind in /tmp, with nothing saying which run they belong to, and are never
# cleaned up. (Without the flag it works inside its own install directory, which
# concurrent runs would share.) So FI is launched through this shim, which keeps
# FI's kind of folder -- a short path: Fast Downward cannot open plan files under
# some long paths -- but records which one it is in this run's directory. The
# adapter reads the plans from there, whether FI finished or not, and deletes it.
_LAUNCHER = """
import os, runpy, sys, tempfile
from forbiditerative import copy_plans
base = os.environ.get('DPEVAL_FI_LOCAL_BASE') or os.path.join('/tmp', 'planner_runs')
os.makedirs(base, exist_ok=True)
local = tempfile.mkdtemp(prefix='fi', dir=base)
with open('fi-local.path', 'w') as handle:
    handle.write(local)
def create_local_folder(create_found_plans_dir=True):
    if create_found_plans_dir:
        found = os.path.join(local, copy_plans.get_found_plans_dir())
        os.makedirs(found, exist_ok=True)
        return found
    return local
copy_plans.create_local_folder = create_local_folder
sys.argv = ['forbiditerative.plan'] + sys.argv[1:]
runpy.run_module('forbiditerative.plan', run_name='__main__', alter_sys=True)
"""

_LOCAL_PATH_FILE = 'fi-local.path'


class ForbidIterativeAdapter(DiversePlannerAdapter):
    name = 'fi'
    supported_features = None          # FI ignores the diversity model

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        margin = float(self.params.get('time-margin', 10))
        budget = ctx.remaining(margin)
        cmd = self._command(ctx, budget)
        logs = [' '.join(cmd)]
        ctx.snapshot = lambda: _read_plans(ctx, logs=None)

        started = time.monotonic()
        exit_code, timed_out = _run(cmd, self._environment(), ctx.workdir, budget, margin, logs)
        pool = _read_plans(ctx, logs)
        _discard_local_folder(ctx)
        fi_seconds = time.monotonic() - started
        if budget is not None and fi_seconds >= budget - 1:
            timed_out = True
        plans = self.select(ctx, pool, logs)
        info = {'fi-seconds': fi_seconds, 'fi-exit-code': exit_code, **self.selection_info(pool, ctx.k),
                'selection-seconds': time.monotonic() - started - fi_seconds}
        outcome = self.outcome_for(len(pool), ctx.k, timed_out)
        return DiverseRun(plans=plans, outcome=outcome, logs=logs, info=info)

    def _command(self, ctx: PlanningContext, budget: float | None) -> list[str]:
        cmd = [str(self.params.get('python') or sys.executable), '-c', _LAUNCHER,
               '--planner', str(self.params.get('planner', 'extended_unordered_topq')),
               '--domain', ctx.domain_file, '--problem', ctx.problem_file,
               '--number-of-plans', str(self.pool_size(ctx.k)), '--quality-bound', str(ctx.q)]
        if self.params.get('symmetries', True):
            cmd.append('--symmetries')
        cmd += ['--use-local-folder', '--clean-local-folder', '--suppress-planners-output']
        if budget is not None:
            cmd += ['--overall-time-limit', str(int(budget))]
        cmd += [str(a) for a in self.params.get('extra-args', [])]
        return cmd

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.params.get('local-dir'):
            env['DPEVAL_FI_LOCAL_BASE'] = os.path.expanduser(str(self.params['local-dir']))
        return env


def _run(cmd: list[str], env: dict[str, str], cwd: str, budget: float | None, margin: float,
         logs: list[str]) -> tuple[int | None, bool]:
    """Run FI to completion or until the budget (plus half the margin) is spent.
    Returns its exit code and whether it had to be killed."""
    process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=None if budget is None else budget + margin / 2)
        if process.returncode != 0:
            logs.append(f'forbiditerative exited with code {process.returncode}')
            logs += _tail(output)
    except subprocess.TimeoutExpired:
        timed_out = True
        logs.append('forbiditerative overran its time limit and was killed')
    finally:
        # Also reached when the runner's own alarm fires mid-wait: never leave
        # the planner running behind the harness's back.
        _kill_group(process)
    return process.returncode, timed_out


def _read_plans(ctx: PlanningContext, logs: list[str] | None) -> list[Any]:
    """FI's plan files parsed as UP plans; parse failures are logged when `logs` is given."""
    from unified_planning.io import PDDLReader

    reader = PDDLReader()
    plans = []
    for text in _plan_texts(ctx):
        try:
            plans.append(reader.parse_plan_string(ctx.problem, text))
        except Exception as error:
            if logs is not None:
                logs.append(f'could not parse an FI plan: {type(error).__name__}: {error}')
    return plans


def _plan_texts(ctx: PlanningContext) -> list[str]:
    """A clean finish leaves the plans in found_plans/done; a timeout leaves
    them as sas_plan.N in FI's local folder. Take the first place that has any."""
    local = _local_folder(ctx) or os.path.join(ctx.workdir, 'no-local-folder')
    for directory in (os.path.join(ctx.workdir, 'found_plans', 'done'),
                      os.path.join(ctx.workdir, 'found_plans'),
                      os.path.join(local, 'found_plans', 'done'),
                      os.path.join(local, 'found_plans'),
                      local):
        texts = _sas_plans_in(directory)
        if texts:
            return texts
    return []


def _sas_plans_in(directory: str) -> list[str]:
    """The distinct ``sas_plan*`` files of `directory`, comments stripped, in plan order."""
    if not os.path.isdir(directory):
        return []
    texts: list[str] = []
    for name in sorted(os.listdir(directory), key=_plan_number):
        path = os.path.join(directory, name)
        if not name.startswith('sas_plan') or not os.path.isfile(path):
            continue
        with open(path) as handle:
            text = '\n'.join(line for line in handle.read().splitlines()
                             if line.strip() and not line.lstrip().startswith(';'))
        if text and text not in texts:
            texts.append(text)
    return texts


def _local_folder(ctx: PlanningContext) -> str | None:
    try:
        with open(os.path.join(ctx.workdir, _LOCAL_PATH_FILE)) as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _discard_local_folder(ctx: PlanningContext) -> None:
    local = _local_folder(ctx)
    if local:
        ctx.snapshot = None
        shutil.rmtree(local, ignore_errors=True)


def _plan_number(name: str) -> tuple[int, str]:
    digits = ''.join(c for c in name if c.isdigit())
    return (int(digits) if digits else -1, name)


def _kill_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


def _tail(output, lines: int = 20) -> list[str]:
    if not output:
        return []
    text = output.decode(errors='replace') if isinstance(output, bytes) else str(output)
    return text.splitlines()[-lines:]
