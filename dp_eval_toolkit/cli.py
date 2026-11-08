"""Command line entry point: ``dpevalcli <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .tasks import TRACKS


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'func', None):
        parser.print_help()
        return 1
    return args.func(args) or 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='dpevalcli',
        description='Benchmark harness for diverse planners: (problem, k, q, diversity model) '
                    'tasks, one slurm array index each.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'typical flow:\n'
            '  dpevalcli discover  --tasks-dir classical-domains\n'
            '  dpevalcli generate  --exp-dir exp --sandbox-dir sandbox --tasks-dir classical-domains --venv-dir venv\n'
            '  bash sandbox/slurm/submit_all.sh\n'
            '  dpevalcli analyze   --sandbox-dir sandbox\n'
            '  dpevalcli report    --sandbox-dir sandbox\n'
        ),
    )
    parser.add_argument('--version', action='version', version=f'dpevalcli {__version__}')
    sub = parser.add_subparsers(dest='command')

    check = sub.add_parser('check', help='check that every planner of an experiment can be imported here')
    check.add_argument('--exp-dir', required=True)
    check.set_defaults(func=_check)

    discover = sub.add_parser('discover', help='list the tasks found in a benchmark repository')
    _add_task_arguments(discover)
    discover.add_argument('--json', dest='as_json', action='store_true')
    discover.set_defaults(func=_discover)

    generate = sub.add_parser('generate', help='one solve command per diversity task, plus slurm arrays')
    generate.add_argument('--exp-dir', required=True, help='experiment directory (exp-details.json + planners/)')
    generate.add_argument('--sandbox-dir', required=True, help='where commands, results and logs go')
    _add_task_arguments(generate)
    launcher = generate.add_mutually_exclusive_group()
    launcher.add_argument('--venv-dir', help='virtualenv each command activates')
    launcher.add_argument('--conda-env', help='conda environment each command activates')
    launcher.add_argument('--apptainer-image', help='run each command inside this image')
    generate.add_argument('--skip-existing', action='store_true', help='skip tasks that already have a result')
    generate.add_argument('--no-validate', action='store_true', help='do not validate returned plans')
    generate.add_argument('--local-jobs', type=int, default=4, help='default parallelism of run_local.sh')
    generate.set_defaults(func=_generate)

    solve = sub.add_parser('solve', help='run ONE diversity task (called by slurm)')
    solve.add_argument('--planner-cfg', required=True)
    solve.add_argument('--domain', required=True)
    solve.add_argument('--problem', required=True)
    solve.add_argument('--k', type=int, required=True, help='number of plans')
    solve.add_argument('--q', type=float, default=1.0, help='quality bound (1.0 = optimal plans only)')
    solve.add_argument('--model', default='model', help='name of the diversity model')
    solve.add_argument('--feature', action='append', default=[], metavar='KEY[=CONFIG]',
                       help="a feature of the diversity model, e.g. 'go' or 'ru=resources.txt'; repeatable")
    solve.add_argument('--results-dir', required=True)
    solve.add_argument('--task-id', default=None, help='defaults to <suite>:<domain>:<instance>')
    solve.add_argument('--suite', default='')
    solve.add_argument('--domain-name', default='')
    solve.add_argument('--instance', default='')
    solve.add_argument('--track', default='classical')
    solve.add_argument('--ipc', default=None)
    solve.add_argument('--errors-dir', default=None)
    solve.add_argument('--run-dir', default=None, help="parent of this run's private working directory")
    solve.add_argument('--keep-run-dir', action='store_true')
    solve.add_argument('--no-validate', dest='validate', action='store_false')
    solve.add_argument('--time-limit', type=int, default=1800, help='seconds; 0 disables the runner-side limit')
    solve.add_argument('--memory-limit', type=int, default=8192, help='MB; 0 disables the runner-side limit')
    solve.add_argument('--eval-time-limit', type=int, default=240,
                       help='seconds for validation + behaviour evaluation after planning (0: none)')
    solve.set_defaults(func=_solve, validate=True)

    for name, helptext, func in (('analyze', 'aggregate results into CSVs and a text summary', _analyze),
                                 ('report', 'paper-ready LaTeX tables and figures', _report)):
        p = sub.add_parser(name, help=helptext)
        p.add_argument('--sandbox-dir', required=True)
        p.add_argument('--results-dir', default=None, help='default: <sandbox>/results')
        p.add_argument('--output-dir', default=None)
        p.add_argument('--strict-k', action='store_true',
                       help='count only runs that returned k plans as solved (not EXHAUSTED ones)')
        if name == 'analyze':
            p.add_argument('--per-domain', action='store_true', help='add a per-domain table to the summary')
        else:
            p.add_argument('--formats', default='pdf,png', help='figure formats (default: pdf,png)')
            p.add_argument('--no-plots', action='store_true', help='tables only (no matplotlib needed)')
        p.set_defaults(func=func)
    return parser


def _add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--tasks-dir', action='append', required=True, metavar='[LABEL=]PATH',
                        help='benchmark repository to scan; repeatable')
    parser.add_argument('--suite-tracks', action='append', default=None, metavar='LABEL=TRACKS')
    parser.add_argument('--tracks', nargs='+', choices=list(TRACKS), default=None)
    parser.add_argument('--domains', nargs='+', default=None, help='glob patterns of domains to include')
    parser.add_argument('--exclude-domains', nargs='+', default=None)
    parser.add_argument('--ipc-years', nargs='+', default=None)
    parser.add_argument('--max-instances-per-domain', type=int, default=None)
    parser.add_argument('--selection', choices=('even', 'first'), default=None)


def _check(args) -> int:
    from .adapters import load_adapter
    from .config import Experiment

    experiment = Experiment.load(args.exp_dir)
    print(f'Experiment {experiment.name}: k={experiment.k_values} q={experiment.quality_bounds} '
          f'models={[m.name for m in experiment.models]}')
    probes = {'fi': ['forbiditerative'], 'bp-smt': ['behaviour_planning_smt.fbi.planner'],
              'bp-asp': ['behaviour_planning_asp'], 'up-anytime': ['unified_planning']}
    failed = 0
    import importlib
    for planner in experiment.planners:
        try:
            load_adapter(planner.adapter, planner.params)
            for module in probes.get(planner.adapter, []):
                importlib.import_module(module)
            print(f'  {planner.tag:<26} {planner.adapter:<12} ok')
        except Exception as error:                                   # noqa: BLE001
            failed += 1
            print(f'  {planner.tag:<26} {planner.adapter:<12} NOT AVAILABLE: {type(error).__name__}: {error}')
    return 1 if failed else 0


def _discover(args) -> int:
    from .generator import collect_tasks
    from .tasks import summarize

    tasks = collect_tasks(args.tasks_dir, None, args)
    if args.as_json:
        json.dump([t.to_dict() for t in tasks], sys.stdout, indent=2)
        print()
        return 0
    counts = summarize(tasks)
    print(f'{len(tasks)} tasks')
    for track, count in sorted(counts['instances_per_track'].items()):
        print(f'  {track:<10} {count:>6} instances, {counts["domains_per_track"].get(track, 0)} domains')
    return 0


def _generate(args) -> int:
    from .generator import generate
    return generate(args)


def _solve(args) -> int:
    from .runner import solve
    if not args.task_id:
        args.task_id = ':'.join(p for p in (args.suite, args.domain_name, args.instance) if p) \
            or os.path.splitext(os.path.basename(args.problem))[0]
    return solve(args)


def _analyze(args) -> int:
    from .analyzer import analyze
    return analyze(args)


def _report(args) -> int:
    from .report import report
    return report(args)


if __name__ == '__main__':
    sys.exit(main())
