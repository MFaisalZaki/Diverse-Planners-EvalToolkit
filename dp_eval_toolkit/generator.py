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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import DiversityModel, Experiment, PlannerConfig, format_q, read_instance_refs, run_label
from .resources import FeatureDataMissing, resolve_feature
from .discovery import discover
from .tasks import Task, filter_tasks, limit_per_domain, select_instances, summarize

CLI = 'dpevalcli'

# (model name, task id) -> the --feature arguments of that diversity task.
FeatureArguments = dict[tuple[str, str], list[str]]


class Sandbox:
    """The directory layout every stage agrees on."""

    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.tasks_file = os.path.join(self.root, 'tasks.json')
        self.cmds_dir = os.path.join(self.root, 'cmds')
        self.all_cmds_file = os.path.join(self.cmds_dir, 'generated_cmds.sh')
        self.slurm_dir = os.path.join(self.root, 'slurm')
        self.slurm_logs_dir = os.path.join(self.slurm_dir, 'logs')
        self.submit_file = os.path.join(self.slurm_dir, 'submit_all.sh')
        self.local_runner = os.path.join(self.root, 'run_local.sh')
        self.results_dir = os.path.join(self.root, 'results')
        self.errors_dir = os.path.join(self.root, 'errors')
        self.runs_dir = os.path.join(self.root, 'runs')
        self.feature_data_dir = os.path.join(self.root, 'feature-data')

    def create(self) -> None:
        for path in (self.root, self.cmds_dir, self.slurm_dir, self.slurm_logs_dir,
                     self.results_dir, self.errors_dir, self.runs_dir, self.feature_data_dir):
            os.makedirs(path, exist_ok=True)

    def commands_file(self, tag: str) -> str:
        return os.path.join(self.cmds_dir, f'{tag}.txt')

    def results_for(self, tag: str, model: str, k: int, q: float) -> str:
        return os.path.join(self.results_dir, tag, run_label(model, k, q))

    def result_file(self, tag: str, model: str, k: int, q: float, slug: str) -> str:
        return os.path.join(self.results_for(tag, model, k, q), f'{slug}.json')


@dataclass
class CommandPlan:
    """What ``generate`` worked out: every diversity task the experiment asks
    for, and the commands still to run for each planner."""

    expected: list[dict[str, Any]] = field(default_factory=list)
    commands: dict[str, list[str]] = field(default_factory=dict)
    already_done: int = 0

    @property
    def total(self) -> int:
        return sum(len(c) for c in self.commands.values())


def generate(args) -> int:
    experiment = Experiment.load(args.exp_dir)
    sandbox = Sandbox(args.sandbox_dir)
    sandbox.create()

    tasks = collect_tasks(args.tasks_dir, experiment, args)
    if not tasks:
        print('No tasks matched the experiment selection; nothing to generate.')
        return 1

    resolved, not_applicable = resolve_models(experiment, tasks, sandbox)
    plan = _plan_commands(experiment, tasks, resolved, sandbox, args)
    # `expected` is the denominator `analyze` uses: written before
    # --skip-existing narrows the work, so a resumed sweep keeps its full size.
    _write_tasks_file(sandbox, experiment, tasks, plan.expected, not_applicable)
    if plan.total == 0:
        print('Every diversity task already has a result; nothing to generate.')
        return 0

    _write_command_files(sandbox, plan.commands)
    _write_slurm_arrays(sandbox, experiment, plan.commands)
    _write_local_runner(sandbox, args.local_jobs)
    _report(sandbox, experiment, tasks, plan, not_applicable)
    return 0


# ----------------------------------------------------------------------
# Tasks and models
# ----------------------------------------------------------------------

def collect_tasks(tasks_dirs: Sequence[str], experiment: Experiment | None, args) -> list[Task]:
    """Discover, filter and cap the tasks of every ``--tasks-dir``; command-line
    options override the experiment's ``tasks`` section."""
    selection = experiment.task_selection if experiment else {}
    tasks = _discover_suites(tasks_dirs, _suite_tracks(args))

    def option(attribute: str, key: str):
        value = getattr(args, attribute, None)
        return value if value is not None else selection.get(key)

    tasks = filter_tasks(tasks,
                         tracks=option('tracks', 'tracks') or None,
                         include=option('domains', 'include-domains') or None,
                         exclude=option('exclude_domains', 'exclude-domains') or None,
                         ipc_years=option('ipc_years', 'ipc-years') or None,
                         selected=selection.get('selected-tasks'))
    instances = _selected_instances(experiment, args)
    if instances:
        tasks = select_instances(tasks, instances)
    return limit_per_domain(tasks, option('max_instances_per_domain', 'max-instances-per-domain'),
                            option('selection', 'selection') or 'even')


def _selected_instances(experiment: Experiment | None, args) -> list[str]:
    """``--selected-instances FILE|builtin`` on the command line, else the experiment's."""
    given = getattr(args, 'selected_instances', None)
    if given:
        return read_instance_refs(given)
    return experiment.selected_instances if experiment else []


def _suite_tracks(args) -> dict[str, list[str]]:
    """``--suite-tracks LABEL=TRACK,TRACK`` -> suite label -> tracks allowed from it."""
    suite_tracks: dict[str, list[str]] = {}
    for entry in getattr(args, 'suite_tracks', None) or []:
        label, _, tracks = entry.partition('=')
        suite_tracks[label.strip()] = [t.strip() for t in tracks.split(',') if t.strip()]
    return suite_tracks


def _discover_suites(tasks_dirs: Sequence[str], suite_tracks: dict[str, list[str]]) -> list[Task]:
    """Every task under the ``[LABEL=]PATH`` entries, reporting directories that yielded none."""
    tasks: list[Task] = []
    skipped: list[tuple[str, str]] = []
    for entry in tasks_dirs:
        label, _, path = entry.partition('=')
        if not path:
            label, path = '', entry
        suite = label or os.path.basename(os.path.abspath(os.path.expanduser(path)).rstrip(os.sep))
        found = discover(path, suite=suite, skipped=skipped)
        allowed = suite_tracks.get(suite)
        if allowed:
            found = [t for t in found if t.track in allowed]
        tasks.extend(found)
    for directory, reason in skipped:
        print(f'note: no tasks from {directory} ({reason})', file=sys.stderr)
    return tasks


def resolve_models(experiment: Experiment, tasks: Sequence[Task],
                   sandbox: Sandbox) -> tuple[FeatureArguments, list[dict[str, Any]]]:
    """The ``--feature`` arguments of every (model, instance).

    An instance missing the data a *required* feature needs is left out of that
    model's runs (and listed under ``not-applicable`` in tasks.json); an
    *optional* feature is dropped for that instance instead.
    """
    resolved: FeatureArguments = {}
    not_applicable: list[dict[str, Any]] = []
    for model in experiment.models:
        for task in tasks:
            try:
                arguments = _feature_arguments(model, task, sandbox, experiment.path)
                resolved[(model.name, task.task_id)] = arguments
            except FeatureDataMissing as error:
                not_applicable.append({'model': model.name, 'task_id': task.task_id, 'reason': str(error)})
    return resolved, not_applicable


def _feature_arguments(model: DiversityModel, task: Task, sandbox: Sandbox, base_dir: str) -> list[str]:
    arguments: list[str] = []
    for feature in model.features:
        try:
            value = resolve_feature(feature.key, feature.config, task, sandbox.feature_data_dir, base_dir)
        except FeatureDataMissing as error:
            if feature.optional:
                continue
            raise FeatureDataMissing(f'{feature.key}: {error}') from error
        arguments.append(feature.key if value is None else f'{feature.key}={value}')
    if not arguments:
        raise FeatureDataMissing('every feature of the model was optional and had no data')
    return arguments


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def _plan_commands(experiment: Experiment, tasks: Sequence[Task], resolved: FeatureArguments,
                   sandbox: Sandbox, args) -> CommandPlan:
    plan = CommandPlan()
    default_prefix = _launcher_prefix(args)
    validate = experiment.validate_plans and not getattr(args, 'no_validate', False)
    skip_existing = getattr(args, 'skip_existing', False)
    for planner in experiment.planners:
        prefix = _planner_prefix(planner, experiment) or default_prefix
        commands: list[str] = []
        for model, k, q, task in _diversity_tasks(experiment, planner, tasks):
            features = resolved.get((model.name, task.task_id))
            if features is None:
                continue
            plan.expected.append({'planner': planner.tag, 'model': model.name, 'k': k, 'q': q,
                                  **task.to_dict()})
            result_file = sandbox.result_file(planner.tag, model.name, k, q, task.slug)
            if skip_existing and os.path.exists(result_file):
                plan.already_done += 1
                continue
            commands.append(prefix + solve_command(planner, task, model.name, features, k, q,
                                                   sandbox, experiment, validate))
        plan.commands[planner.tag] = commands
    return plan


def _diversity_tasks(experiment: Experiment, planner: PlannerConfig,
                     tasks: Sequence[Task]) -> Iterator[tuple[DiversityModel, int, float, Task]]:
    """Every (model, k, q, task) this planner takes part in."""
    for model, k, q in experiment.grid():
        if not planner.runs_model(model.name):
            continue
        for task in tasks:
            if planner.runs_track(task.track):
                yield model, k, q, task


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


def environment_prefix(apptainer_image: str | None = None, venv_dir: str | None = None,
                       conda_env: str | None = None) -> str:
    """The shell prefix that puts a command into the given environment."""
    if apptainer_image:
        return f'apptainer run --cleanenv {shlex.quote(apptainer_image)} '
    if venv_dir:
        return f'source {shlex.quote(os.path.join(venv_dir, "bin", "activate"))} && '
    if conda_env:
        return f'conda activate {shlex.quote(conda_env)} && '
    return ''


def _launcher_prefix(args) -> str:
    """The environment ``generate`` was given on the command line (``--venv-dir`` etc.)."""
    apptainer_image = getattr(args, 'apptainer_image', None)
    venv_dir = getattr(args, 'venv_dir', None)
    return environment_prefix(apptainer_image=os.path.abspath(apptainer_image) if apptainer_image else None,
                              venv_dir=os.path.abspath(venv_dir) if venv_dir else None,
                              conda_env=getattr(args, 'conda_env', None))


def _planner_prefix(planner: PlannerConfig, experiment: Experiment) -> str:
    """A planner configuration's own environment, if it names one; paths are
    relative to the experiment directory."""
    launcher = planner.launcher
    apptainer_image = launcher.get('apptainer-image')
    venv_dir = launcher.get('venv-dir')
    resolve = experiment.resolve_path
    return environment_prefix(apptainer_image=resolve(apptainer_image) if apptainer_image else None,
                              venv_dir=resolve(venv_dir) if venv_dir else None,
                              conda_env=launcher.get('conda-env'))


def _write_tasks_file(sandbox: Sandbox, experiment: Experiment, tasks: Sequence[Task],
                      expected: list[dict[str, Any]], not_applicable: list[dict[str, Any]]) -> None:
    with open(sandbox.tasks_file, 'w') as handle:
        json.dump({
            'experiment': experiment.name,
            'grid': {'k-values': experiment.k_values, 'quality-bounds': experiment.quality_bounds,
                     'models': {m.name: [f.to_json() for f in m.features] for m in experiment.models}},
            'limits': {'time-seconds': experiment.time_limit_seconds,
                       'memory-mb': experiment.memory_limit_mb},
            'summary': summarize(tasks),
            'expected': expected,
            'not-applicable': not_applicable,
            'tasks': [t.to_dict() for t in tasks],
        }, handle, indent=1)


def _write_command_files(sandbox: Sandbox, per_planner: dict[str, list[str]]) -> None:
    """One command file per planner, plus every command in one script."""
    all_lines: list[str] = []
    for tag, commands in per_planner.items():
        if commands:
            _write_text(sandbox.commands_file(tag), '\n'.join(commands) + '\n')
            all_lines.extend(commands)
    _write_script(sandbox.all_cmds_file, ['#!/bin/bash', *all_lines])


# ----------------------------------------------------------------------
# Slurm
# ----------------------------------------------------------------------

def slurm_directives(experiment: Experiment, job_name: str, sandbox: Sandbox) -> list[str]:
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
    for key in ('partition', 'qos', 'account'):
        if slurm.get(key):
            lines.append(f'#SBATCH --{key}={slurm[key]}')
    for directive in slurm.get('extra-directives') or []:
        text = str(directive).strip()
        lines.append(text if text.startswith('#SBATCH') else f'#SBATCH {text}')
    return lines


def _write_slurm_arrays(sandbox: Sandbox, experiment: Experiment,
                        per_planner: dict[str, list[str]]) -> list[str]:
    """One job array per planner (chunked to ``max-array-size``) and a script submitting them all."""
    slurm = experiment.slurm
    chunk_size = int(slurm.get('max-array-size') or 1000)
    throttle = int(slurm.get('max-parallel-jobs') or 0)
    scripts: list[str] = []
    for tag, commands in per_planner.items():
        chunks = list(_chunks(len(commands), chunk_size))
        for number, (start, end) in enumerate(chunks, start=1):
            job_name = f'dpeval-{tag}.{number}' if len(chunks) > 1 else f'dpeval-{tag}'
            path = os.path.join(sandbox.slurm_dir, f'{job_name}.sbatch')
            _write_script(path, _sbatch_lines(experiment, sandbox, job_name, sandbox.commands_file(tag),
                                              start, end, throttle))
            scripts.append(path)
    _write_script(sandbox.submit_file, ['#!/bin/bash', '# Submit every generated job array.',
                                        'set -euo pipefail', *(f'sbatch {shlex.quote(path)}' for path in scripts)])
    return scripts


def _chunks(count: int, size: int) -> Iterator[tuple[int, int]]:
    """``(start, end)`` index ranges splitting `count` items into pieces of at most `size`."""
    for start in range(0, count, size):
        yield start, min(start + size, count)


def _sbatch_lines(experiment: Experiment, sandbox: Sandbox, job_name: str, cmd_file: str,
                  start: int, end: int, throttle: int) -> list[str]:
    array = f'0-{end - start - 1}' + (f'%{throttle}' if throttle > 0 else '')
    return [
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
    ]


def _write_local_runner(sandbox: Sandbox, jobs: int) -> None:
    _write_script(sandbox.local_runner, [
        '#!/bin/bash',
        '# Run the whole sweep locally, N tasks at a time. Usage: ./run_local.sh [jobs]',
        'set -uo pipefail',
        f'JOBS=${{1:-{max(1, jobs)}}}',
        f'CMDS={shlex.quote(sandbox.all_cmds_file)}',
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
    ])


def _report(sandbox: Sandbox, experiment: Experiment, tasks: Sequence[Task], plan: CommandPlan,
            not_applicable: Sequence[dict[str, Any]]) -> None:
    counts = summarize(tasks)
    slurm = experiment.slurm
    print(f'Experiment      : {experiment.name} ({experiment.path})')
    print(f'Sandbox         : {sandbox.root}')
    print(f'Time limit      : {experiment.time_limit_seconds}s per task '
          f'(slurm --time={experiment.slurm_time})')
    print(f'Memory limit    : {experiment.memory_limit_mb}MB per task '
          f'(slurm --mem={experiment.slurm_memory})')
    print(f'Slurm           : partition={slurm.get("partition") or "-"} qos={slurm.get("qos") or "-"} '
          f'account={slurm.get("account") or "-"}')
    print(f'Instances       : {len(tasks)}')
    for track, count in sorted(counts['instances_per_track'].items()):
        domains = counts['domains_per_track'].get(track, 0)
        print(f'  {track:<10} {count:>6} instances across {domains} domains')
    print(f'k values        : {experiment.k_values}')
    print(f'quality bounds  : {experiment.quality_bounds}')
    print(f'models          : {", ".join(m.name for m in experiment.models)}')
    if not_applicable:
        print(f'Not applicable  : {len(not_applicable)} (model, instance) pairs lack feature data '
              f'(listed in tasks.json)')
    print(f'Planners        : {len(experiment.planners)}')
    for planner in experiment.planners:
        commands = len(plan.commands.get(planner.tag, []))
        print(f'  {planner.tag:<24} {commands:>7} commands (adapter: {planner.adapter})')
    if plan.already_done:
        print(f'Skipped         : {plan.already_done} tasks that already have results')
    print(f'Total commands  : {plan.total}')
    print()
    print(f'Commands        : {sandbox.all_cmds_file}')
    print(f'Submit with     : bash {sandbox.submit_file}')
    print(f'Or run locally  : bash {sandbox.local_runner} 8')


def _write_text(path: str, text: str) -> None:
    with open(path, 'w') as handle:
        handle.write(text)


def _write_script(path: str, lines: Sequence[str]) -> None:
    """Write `lines` as an executable shell script."""
    _write_text(path, '\n'.join(lines) + '\n')
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
