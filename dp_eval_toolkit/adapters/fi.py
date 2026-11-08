"""Forbid-Iterative (FI) diverse planner, driven through its command line.

This is the baseline of the Behaviour Planning paper experiments
(``paper_experiments/run-benchmark.py`` in pyBehaviourPlanningSMT): the
``forbiditerative`` package's top-quality planner, asked for ``k`` plans within
quality bound ``q``. FI does not take a diversity model -- its plans are scored
by the toolkit's behaviour evaluator like everyone else's.

``planner-params``:

``planner``            FI planner configuration (default ``extended_unordered_topq``)
``symmetries``         pass ``--symmetries`` (default true)
``extra-args``         list of further command-line arguments, verbatim
``python``             interpreter to run FI with (default: this one)
``time-margin``        seconds kept back from the budget for reading plans (default 10)
``local-dir``          where FI keeps its intermediate files (default /tmp/planner_runs;
                       a node-local scratch such as $TMPDIR is a good choice on a cluster)
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, List

from .base import COMPLETE, EXHAUSTED, TIMEOUT, UNSOLVABLE, DiversePlannerAdapter, DiverseRun, PlanningContext

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


class ForbidIterativeAdapter(DiversePlannerAdapter):
    name = 'fi'
    supported_features = None          # FI ignores the diversity model

    def plan(self, ctx: PlanningContext) -> DiverseRun:
        margin = float(self.params.get('time-margin', 10))
        budget = ctx.remaining(margin)
        cmd = [str(self.params.get('python') or sys.executable), '-c', _LAUNCHER,
               '--planner', str(self.params.get('planner', 'extended_unordered_topq')),
               '--domain', ctx.domain_file, '--problem', ctx.problem_file,
               '--number-of-plans', str(ctx.k), '--quality-bound', str(ctx.q)]
        if self.params.get('symmetries', True):
            cmd.append('--symmetries')
        cmd += ['--use-local-folder', '--clean-local-folder', '--suppress-planners-output']
        if budget is not None:
            cmd += ['--overall-time-limit', str(int(budget))]
        cmd += [str(a) for a in self.params.get('extra-args', [])]

        env = os.environ.copy()
        if self.params.get('local-dir'):
            env['DPEVAL_FI_LOCAL_BASE'] = os.path.expanduser(str(self.params['local-dir']))
        logs = [' '.join(cmd)]
        ctx.snapshot = lambda: self._read_plans(ctx, logs=None)

        started = time.monotonic()
        process = subprocess.Popen(cmd, cwd=ctx.workdir, env=env, stdout=subprocess.PIPE,
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

        plans = self._read_plans(ctx, logs)[:ctx.k]
        local = _local_folder(ctx)
        if local:
            ctx.snapshot = None
            shutil.rmtree(local, ignore_errors=True)
        elapsed = time.monotonic() - started
        if budget is not None and elapsed >= budget - 1:
            timed_out = True
        if len(plans) >= ctx.k:
            outcome = COMPLETE
        elif timed_out:
            outcome = TIMEOUT
        else:
            # FI stops by itself once no further plan fits the quality bound.
            outcome = EXHAUSTED if plans else UNSOLVABLE
        return DiverseRun(plans=plans, outcome=outcome, logs=logs,
                          info={'fi-seconds': elapsed, 'fi-exit-code': process.returncode})

    def _read_plans(self, ctx: PlanningContext, logs) -> List[Any]:
        from unified_planning.io import PDDLReader

        texts: List[str] = []
        # A clean finish leaves the plans in found_plans/done; a timeout leaves
        # them as sas_plan.N in FI's local folder. Take the first that has any.
        local = _local_folder(ctx) or os.path.join(ctx.workdir, 'no-local-folder')
        for directory in (os.path.join(ctx.workdir, 'found_plans', 'done'),
                          os.path.join(ctx.workdir, 'found_plans'),
                          os.path.join(local, 'found_plans', 'done'),
                          os.path.join(local, 'found_plans'),
                          local):
            if not os.path.isdir(directory):
                continue
            for name in sorted(os.listdir(directory), key=_plan_number):
                path = os.path.join(directory, name)
                if not os.path.isfile(path) or not name.startswith('sas_plan'):
                    continue
                with open(path) as handle:
                    text = '\n'.join(line for line in handle.read().splitlines()
                                     if line.strip() and not line.lstrip().startswith(';'))
                if text and text not in texts:
                    texts.append(text)
            if texts:
                break
        reader = PDDLReader()
        plans = []
        for text in texts:
            try:
                plans.append(reader.parse_plan_string(ctx.problem, text))
            except Exception as error:                         # noqa: BLE001
                if logs is not None:
                    logs.append(f'could not parse an FI plan: {type(error).__name__}: {error}')
        return plans


def _local_folder(ctx: PlanningContext):
    try:
        with open(os.path.join(ctx.workdir, 'fi-local.path')) as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _plan_number(name: str):
    digits = ''.join(c for c in name if c.isdigit())
    return (int(digits) if digits else -1, name)


def _kill_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _tail(output, lines: int = 20) -> List[str]:
    if not output:
        return []
    text = output.decode(errors='replace') if isinstance(output, bytes) else str(output)
    return text.splitlines()[-lines:]
