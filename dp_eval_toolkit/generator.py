"""Command generation: planners x instances x (model, k, q) -> one solve command
each, wrapped in slurm job arrays.

Same packaging as pyPMTEvalToolkit: every command is one line of
``cmds/<planner>.txt``, and one job array per planner (split into chunks of at
most ``max-array-size``) runs line ``SLURM_ARRAY_TASK_ID`` of that file, with
the experiment's ``--time``, ``--mem``, ``--partition`` and ``--qos``. A
``run_local.sh`` runs the same commands without a scheduler.

The unit of work is a *diversity task*: one (planner, instance, k, q, model).
"""

from __future__ import annotations

import json
import os
import shlex
import stat
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import DiversityModel, Experiment, PlannerConfig, format_q, run_label
from .resources import FeatureDataMissing, resolve_feature
from .tasks import Task, discover, filter_tasks, limit_per_domain, summarize

CLI = 'dpevalcli'


class Sandbox:
    """The directory layout every stage agrees on."""

    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.tasks_file = os.path.join(self.root, 'tasks.json')
        self.cmds_dir = os.path.join(self.root, 'cmds')
        self.slurm_dir = os.path.join(self.root, 'slurm')
        self.slurm_logs_dir = os.path.join(self.slurm_dir, 'logs')
        self.results_dir = os.path.join(self.root, 'results')
        self.errors_dir = os.path.join(self.root, 'errors')
        self.runs_dir = os.path.join(self.root, 'runs')
        self.feature_data_dir = os.path.join(self.root, 'feature-data')

    def create(self) -> None:
        for path in (self.root, self.cmds_dir, self.slurm_dir, self.slurm_logs_dir,
                     self.results_dir, self.errors_dir, self.runs_dir, self.feature_data_dir):
            os.makedirs(path, exist_ok=True)

    def results_for(self, tag: str, model: str, k: int, q: float) -> str:
        return os.path.join(self.results_dir, tag, run_label(model, k, q))

    def result_file(self, tag: str, model: str, k: int, q: float, slug: str) -> str:
        return os.path.join(self.results_for(tag, model, k, q), f'{slug}.json')


def generate(args) -> int:
    experiment = Experiment.load(args.exp_dir)
    sandbox = Sandbox(args.sandbox_dir)
    sandbox.create()

    tasks = collect_tasks(args.tasks_dir, experiment, args)
    if not tasks:
        print('No tasks matched the experiment selection; nothing to generate.')
        return 1

    resolved, skipped = resolve_models(experiment, tasks, sandbox)
    default_prefix = _launcher_prefix(args)
    validate = experiment.validate_plans and not getattr(args, 'no_validate', False)

    expected: List[Dict[str, Any]] = []
    per_planner: Dict[str, List[str]] = {}
    skipped_done = 0
    for planner in experiment.planners:
        commands: List[str] = []
        prefix = _planner_prefix(planner, experiment) or default_prefix
        for model, k, q in experiment.grid():
            if not planner.runs_model(model.name):
                continue
            for task in tasks:
                if not planner.runs_track(task.track):
                    continue
                features = resolved.get((model.name, task.task_id))
                if features is None:
                    continue
                expected.append({'planner': planner.tag, 'model': model.name, 'k': k, 'q': q,
                                 **task.to_dict()})
                if args.skip_existing and os.path.exists(
                        sandbox.result_file(planner.tag, model.name, k, q, task.slug)):
                    skipped_done += 1
                    continue
                commands.append(prefix + solve_command(planner, task, model.name, features, k, q,
                                                       sandbox, experiment, validate))
        per_planner[planner.tag] = commands

    # `expected` is the denominator `analyze` uses: written before
    # --skip-existing narrows the work, so a resumed sweep keeps its full size.
    with open(sandbox.tasks_file, 'w') as handle:
        json.dump({
            'experiment': experiment.name,
            'grid': {'k-values': experiment.k_values, 'quality-bounds': experiment.quality_bounds,
                     'models': {m.name: [f.to_json() for f in m.features] for m in experiment.models}},
            'limits': {'time-seconds': experiment.time_limit_seconds,
                       'memory-mb': experiment.memory_limit_mb},
            'summary': summarize(tasks),
            'expected': expected,
            'not-applicable': skipped,
            'tasks': [t.to_dict() for t in tasks],
        }, handle, indent=1)

    total = sum(len(c) for c in per_planner.values())
    if total == 0:
        print('Every diversity task already has a result; nothing to generate.')
        return 0

    written = _write_command_files(sandbox, per_planner)
    _write_slurm_arrays(sandbox, experiment, per_planner)
    _write_local_runner(sandbox, args.local_jobs)
    _report(sandbox, experiment, tasks, per_planner, total, skipped_done, skipped, written)
    return 0


# ----------------------------------------------------------------------
# Tasks and models
# ----------------------------------------------------------------------

def collect_tasks(tasks_dirs: Sequence[str], experiment: Optional[Experiment], args) -> List[Task]:
    selection = experiment.task_selection if experiment else {}
    suite_tracks: Dict[str, List[str]] = {}
    for entry in getattr(args, 'suite_tracks', None) or []:
        label, _, tracks = entry.partition('=')
        suite_tracks[label.strip()] = [t.strip() for t in tracks.split(',') if t.strip()]

    tasks: List[Task] = []
    skipped: List[Tuple[str, str]] = []
    for entry in tasks_dirs:
        label, _, path = entry.partition('=')
        if not path:
            label, path = None, entry
        suite = label or os.path.basename(os.path.abspath(os.path.expanduser(path)).rstrip(os.sep))
        found = discover(path, suite=suite, skipped=skipped)
        allowed = suite_tracks.get(suite)
        if allowed:
            found = [t for t in found if t.track in allowed]
        tasks.extend(found)
    for directory, reason in skipped:
        print(f'note: no tasks from {directory} ({reason})', file=sys.stderr)

    tracks = getattr(args, 'tracks', None) or selection.get('tracks')
    include = getattr(args, 'domains', None) or selection.get('include-domains')
    exclude = getattr(args, 'exclude_domains', None) or selection.get('exclude-domains')
    ipc_years = getattr(args, 'ipc_years', None) or selection.get('ipc-years')
    tasks = filter_tasks(tasks, tracks=tracks, include=include, exclude=exclude,
                         ipc_years=ipc_years, selected=selection.get('selected-tasks'))
    limit = getattr(args, 'max_instances_per_domain', None)
    if limit is None:
        limit = selection.get('max-instances-per-domain')
    strategy = getattr(args, 'selection', None) or selection.get('selection', 'even')
    return limit_per_domain(tasks, limit, strategy)


def resolve_models(experiment: Experiment, tasks: Sequence[Task], sandbox: Sandbox
                   ) -> Tuple[Dict[Tuple[str, str], List[str]], List[Dict[str, Any]]]:
    """The ``--feature`` arguments of every (model, instance).

    An instance missing the data a *required* feature needs is left out of that
    model's runs (and listed under ``not-applicable`` in tasks.json); an
    *optional* feature is dropped for that instance instead.
    """
    resolved: Dict[Tuple[str, str], List[str]] = {}
    skipped: List[Dict[str, Any]] = []
    for model in experiment.models:
        for task in tasks:
            arguments = _feature_arguments(model, task, sandbox, experiment.path)
            if isinstance(arguments, str):
                skipped.append({'model': model.name, 'task_id': task.task_id, 'reason': arguments})
            else:
                resolved[(model.name, task.task_id)] = arguments
    return resolved, skipped


def _feature_arguments(model: DiversityModel, task: Task, sandbox: Sandbox, base_dir: str):
    arguments: List[str] = []
    for feature in model.features:
        try:
            value = resolve_feature(feature.key, feature.config, task, sandbox.feature_data_dir, base_dir)
        except FeatureDataMissing as error:
            if feature.optional:
                continue
            return f'{feature.key}: {error}'
        arguments.append(feature.key if value is None else f'{feature.key}={value}')
    if not arguments:
        return 'every feature of the model was optional and had no data'
    return arguments


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def solve_command(planner: PlannerConfig, task: Task, model: str, features: Sequence[str],
                  k: int, q: float, sandbox: Sandbox, experiment: Experiment,
                  validate: bool = True) -> str:
    parts = [
        CLI, 'solve',
        '--planner-cfg', planner.path or '',
        '--task-id', task.task_id,
        '--suite', task.suite,
        '--domain-name', task.domain,
        '--instance', task.instance,
        '--track', task.track,
        '--domain', task.domain_file,
        '--problem', task.problem_file,
        '--k', str(k),
        '--q', format_q(q),
        '--model', model,
    ]
    for feature in features:
        parts += ['--feature', feature]
    parts += [
        '--results-dir', sandbox.results_for(planner.tag, model, k, q),
        '--errors-dir', sandbox.errors_dir,
        '--run-dir', sandbox.runs_dir,
        '--time-limit', str(experiment.time_limit_seconds),
        '--memory-limit', str(experiment.memory_limit_mb),
    ]
    if task.ipc:
        parts += ['--ipc', str(task.ipc)]
    if not validate:
        parts.append('--no-validate')
    return ' '.join(shlex.quote(p) for p in parts)


def _launcher_prefix(args) -> str:
    if getattr(args, 'apptainer_image', None):
        image = shlex.quote(os.path.abspath(args.apptainer_image))
        return f'apptainer run --cleanenv {image} '
    if getattr(args, 'venv_dir', None):
        activate = shlex.quote(os.path.join(os.path.abspath(args.venv_dir), 'bin', 'activate'))
        return f'source {activate} && '
    if getattr(args, 'conda_env', None):
        return f'conda activate {shlex.quote(args.conda_env)} && '
    return ''


def _planner_prefix(planner: PlannerConfig, experiment: Experiment) -> str:
    """A planner configuration's own environment, if it names one."""
    launcher = planner.launcher
    if launcher.get('apptainer-image'):
        return f'apptainer run --cleanenv {shlex.quote(experiment.resolve_path(launcher["apptainer-image"]))} '
    if launcher.get('venv-dir'):
        activate = os.path.join(experiment.resolve_path(launcher['venv-dir']), 'bin', 'activate')
        return f'source {shlex.quote(activate)} && '
    if launcher.get('conda-env'):
        return f'conda activate {shlex.quote(launcher["conda-env"])} && '
    return ''


def _write_command_files(sandbox: Sandbox, per_planner: Dict[str, List[str]]) -> Dict[str, str]:
    written = {}
    all_lines: List[str] = []
    for tag, commands in per_planner.items():
        if not commands:
            continue
        path = os.path.join(sandbox.cmds_dir, f'{tag}.txt')
        with open(path, 'w') as handle:
            handle.write('\n'.join(commands) + '\n')
        written[tag] = path
        all_lines.extend(commands)
    combined = os.path.join(sandbox.cmds_dir, 'generated_cmds.sh')
    with open(combined, 'w') as handle:
        handle.write('#!/bin/bash\n' + '\n'.join(all_lines) + '\n')
    _make_executable(combined)
    written['__all__'] = combined
    return written


# ----------------------------------------------------------------------
# Slurm
# ----------------------------------------------------------------------

def slurm_directives(experiment: Experiment, job_name: str, sandbox: Sandbox) -> List[str]:
    slurm = experiment.slurm
    lines = [
        f'#SBATCH --job-name={job_name}',
        f'#SBATCH --output={sandbox.slurm_logs_dir}/%x_%A_%a.out',
        f'#SBATCH --error={sandbox.slurm_logs_dir}/%x_%A_%a.err',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1',
        f'#SBATCH --cpus-per-task={slurm.get("cpus-per-task", 1)}',
        f'#SBATCH --mem={experiment.slurm_memory}',
        f'#SBATCH --time={experiment.slurm_time}',
    ]
    for key, flag in (('partition', '--partition'), ('qos', '--qos'), ('account', '--account')):
        value = slurm.get(key)
        if value:
            lines.append(f'#SBATCH {flag}={value}')
    for directive in slurm.get('extra-directives') or []:
        directive = str(directive).strip()
        lines.append(directive if directive.startswith('#SBATCH') else f'#SBATCH {directive}')
    return lines


def _write_slurm_arrays(sandbox: Sandbox, experiment: Experiment,
                        per_planner: Dict[str, List[str]]) -> List[str]:
    slurm = experiment.slurm
    chunk_size = int(slurm.get('max-array-size') or 1000)
    throttle = int(slurm.get('max-parallel-jobs') or 0)
    scripts: List[str] = []
    for tag, commands in per_planner.items():
        if not commands:
            continue
        cmd_file = os.path.join(sandbox.cmds_dir, f'{tag}.txt')
        chunks = [(i, min(i + chunk_size, len(commands))) for i in range(0, len(commands), chunk_size)]
        for number, (start, end) in enumerate(chunks, start=1):
            suffix = f'.{number}' if len(chunks) > 1 else ''
            job_name = f'dpeval-{tag}{suffix}'
            array = f'0-{end - start - 1}' + (f'%{throttle}' if throttle > 0 else '')
            body = '\n'.join([
                '#!/bin/bash',
                *slurm_directives(experiment, job_name, sandbox),
                f'#SBATCH --array={array}',
                '',
                '# Each array index runs one diversity task: its own line of the command file.',
                'set -uo pipefail',
                f'CMD_FILE={shlex.quote(cmd_file)}',
                f'OFFSET={start}',
                'LINE=$((OFFSET + SLURM_ARRAY_TASK_ID + 1))',
                'CMD=$(sed -n "${LINE}p" "$CMD_FILE")',
                'if [ -z "$CMD" ]; then',
                '    echo "no command at line $LINE of $CMD_FILE" >&2',
                '    exit 1',
                'fi',
                'echo "[$(date -Is)] host=$(hostname) line=$LINE"',
                'echo "$CMD"',
                'eval "$CMD"',
                'status=$?',
                'echo "[$(date -Is)] exit=$status"',
                '# The runner records timeouts and crashes itself; never fail the array.',
                'exit 0',
                '',
            ])
            path = os.path.join(sandbox.slurm_dir, f'{job_name}.sbatch')
            with open(path, 'w') as handle:
                handle.write(body)
            _make_executable(path)
            scripts.append(path)

    submit = os.path.join(sandbox.slurm_dir, 'submit_all.sh')
    with open(submit, 'w') as handle:
        handle.write('#!/bin/bash\n# Submit every generated job array.\nset -euo pipefail\n')
        for path in scripts:
            handle.write(f'sbatch {shlex.quote(path)}\n')
    _make_executable(submit)
    return scripts


def _write_local_runner(sandbox: Sandbox, jobs: int) -> None:
    path = os.path.join(sandbox.root, 'run_local.sh')
    combined = os.path.join(sandbox.cmds_dir, 'generated_cmds.sh')
    with open(path, 'w') as handle:
        handle.write('\n'.join([
            '#!/bin/bash',
            '# Run the whole sweep locally, N tasks at a time. Usage: ./run_local.sh [jobs]',
            'set -uo pipefail',
            f'JOBS=${{1:-{max(1, jobs)}}}',
            f'CMDS={shlex.quote(combined)}',
            'if command -v parallel >/dev/null 2>&1; then',
            '    grep -v "^#" "$CMDS" | grep -v "^[[:space:]]*$" | parallel -j "$JOBS" --halt never',
            '    exit 0',
            'fi',
            'while IFS= read -r cmd; do',
            '    case "$cmd" in ""|"#"*) continue ;; esac',
            '    while [ "$(jobs -rp | wc -l | tr -d " ")" -ge "$JOBS" ]; do sleep 0.5; done',
            '    bash -c "$cmd" &',
            'done < "$CMDS"',
            'wait',
            '',
        ]))
    _make_executable(path)


def _report(sandbox, experiment, tasks, per_planner, total, skipped_done, skipped, written) -> None:
    counts = summarize(tasks)
    print(f'Experiment      : {experiment.name} ({experiment.path})')
    print(f'Sandbox         : {sandbox.root}')
    print(f'Time limit      : {experiment.time_limit_seconds}s per task (slurm --time={experiment.slurm_time})')
    print(f'Memory limit    : {experiment.memory_limit_mb}MB per task (slurm --mem={experiment.slurm_memory})')
    slurm = experiment.slurm
    print(f'Slurm           : partition={slurm.get("partition") or "-"} qos={slurm.get("qos") or "-"} '
          f'account={slurm.get("account") or "-"}')
    print(f'Instances       : {len(tasks)}')
    for track, count in sorted(counts['instances_per_track'].items()):
        print(f'  {track:<10} {count:>6} instances across {counts["domains_per_track"].get(track, 0)} domains')
    print(f'k values        : {experiment.k_values}')
    print(f'quality bounds  : {experiment.quality_bounds}')
    print(f'models          : {", ".join(m.name for m in experiment.models)}')
    if skipped:
        print(f'Not applicable  : {len(skipped)} (model, instance) pairs lack feature data '
              f'(listed in tasks.json)')
    print(f'Planners        : {len(experiment.planners)}')
    for planner in experiment.planners:
        print(f'  {planner.tag:<24} {len(per_planner.get(planner.tag, [])):>7} commands (adapter: {planner.adapter})')
    if skipped_done:
        print(f'Skipped         : {skipped_done} tasks that already have results')
    print(f'Total commands  : {total}')
    print()
    print(f'Commands        : {written.get("__all__")}')
    print(f'Submit with     : bash {os.path.join(sandbox.slurm_dir, "submit_all.sh")}')
    print(f'Or run locally  : bash {os.path.join(sandbox.root, "run_local.sh")} 8')


def _make_executable(path: str) -> None:
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
