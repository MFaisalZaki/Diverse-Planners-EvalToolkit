"""Tests for the stages that need no planner: config, resources, generate,
analyze, report -- plus the behaviour evaluator when unified-planning is
installed."""

from __future__ import annotations

import json
import os
import shutil
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from dp_eval_toolkit import analyzer, generator, resources  # noqa: E402
from dp_eval_toolkit.config import Experiment, FeatureSpec, parse_memory, parse_time, run_label  # noqa: E402
from dp_eval_toolkit.runner import parse_feature  # noqa: E402
from dp_eval_toolkit.tasks import Task  # noqa: E402

import synthetic  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOMAIN = """(define (domain delivery)
  (:requirements :strips :typing)
  (:types truck location parcel)
  (:predicates (truck-at ?t - truck ?l - location) (parcel-at ?p - parcel ?l - location)
               (in ?p - parcel ?t - truck) (road ?a ?b - location))
  (:action drive :parameters (?t - truck ?a ?b - location)
    :precondition (and (truck-at ?t ?a) (road ?a ?b))
    :effect (and (not (truck-at ?t ?a)) (truck-at ?t ?b)))
  (:action load :parameters (?p - parcel ?t - truck ?l - location)
    :precondition (and (truck-at ?t ?l) (parcel-at ?p ?l))
    :effect (and (not (parcel-at ?p ?l)) (in ?p ?t)))
  (:action unload :parameters (?p - parcel ?t - truck ?l - location)
    :precondition (and (truck-at ?t ?l) (in ?p ?t))
    :effect (and (not (in ?p ?t)) (parcel-at ?p ?l))))
"""

PROBLEM = """(define (problem two) (:domain delivery)
  (:objects t1 t2 - truck src dest - location p1 p2 - parcel)
  (:init (truck-at t1 src) (truck-at t2 src) (parcel-at p1 src) (parcel-at p2 src)
         (road src dest) (road dest src))
  (:goal (and (parcel-at p1 dest) (parcel-at p2 dest))))
"""


def _bench(tmp_path):
    bench = tmp_path / 'bench' / 'delivery'
    bench.mkdir(parents=True)
    (bench / 'domain.pddl').write_text(DOMAIN)
    (bench / 'p01.pddl').write_text(PROBLEM)
    (bench / 'p02.pddl').write_text(PROBLEM.replace('two', 'two-b'))
    res = tmp_path / 'res' / 'delivery'
    res.mkdir(parents=True)
    (res / 'resources.txt').write_text('(:resource t1 0 1 1)\n(:resource t2 0 1 1)\n')
    return tmp_path / 'bench'


def _experiment(tmp_path, models=None, extra_planner=None):
    exp = tmp_path / 'exp'
    (exp / 'planners').mkdir(parents=True)
    details = {
        'cfgs': {'timelimit': '10m', 'memorylimit': '4GB',
                 'slurm': {'partition': 'long', 'qos': 'high', 'max-array-size': 5}},
        'diversity': {'k-values': [10, 5], 'quality-bounds': [1.0, 1.2],
                      'models': models or {'go': [['go', {}]],
                                           'go-ru': [['go', {}], ['ru', {'resources': str(tmp_path / 'res')}]]}},
        'tasks': {'tracks': ['classical']},
    }
    (exp / 'exp-details.json').write_text(json.dumps(details))
    (exp / 'planners' / 'a.json').write_text(json.dumps({'planner-tag': 'A', 'adapter': 'bp-asp'}))
    (exp / 'planners' / 'b.json').write_text(json.dumps(
        extra_planner or {'planner-tag': 'B', 'adapter': 'fi', 'models': ['go']}))
    return exp


# ----------------------------------------------------------------------

def test_limits_and_labels():
    assert parse_time('00:30:00') == 1800 and parse_time('30m') == 1800 and parse_time(90) == 90
    assert parse_memory('8GB') == 8192 and parse_memory('512MB') == 512
    assert run_label('go-ru', 5, 1.0) == 'go-ru__k5__q1.0'
    assert parse_feature('go') == ('go', {})
    assert parse_feature('ru=/a/b.txt') == ('ru', '/a/b.txt')
    assert parse_feature('x={"a": 1}') == ('x', {'a': 1})
    assert FeatureSpec.parse(['ru', {'optional': True}]).optional


def test_experiment_grid(tmp_path):
    experiment = Experiment.load(str(_experiment(tmp_path)))
    assert experiment.k_values == [5, 10]
    assert experiment.quality_bounds == [1.0, 1.2]
    assert len(experiment.grid()) == 8
    assert experiment.slurm['qos'] == 'high'


def test_bad_quality_bound(tmp_path):
    exp = _experiment(tmp_path)
    details = json.loads((exp / 'exp-details.json').read_text())
    details['diversity']['quality-bounds'] = [0.5]
    (exp / 'exp-details.json').write_text(json.dumps(details))
    with pytest.raises(ValueError):
        Experiment.load(str(exp))


def test_shipped_configurations_load():
    for name in os.listdir(os.path.join(ROOT, 'exp-configurations')):
        experiment = Experiment.load(os.path.join(ROOT, 'exp-configurations', name))
        assert experiment.planners


def test_builtin_resources(tmp_path):
    task = Task('classical-domains', 'rovers', 'p07', '/d.pddl', '/p07.pddl', 'classical', '2006')
    path = resources.resolve_feature('ru', 'builtin', task, str(tmp_path), str(tmp_path))
    text = open(path).read()
    assert text.count(':resource') == 3 and 'rover2' in text
    # no year: two rovers files exist, so the instance is ambiguous
    task = Task('s', 'rovers', 'p07', '/d.pddl', '/p07.pddl', 'classical', None)
    with pytest.raises(resources.FeatureDataMissing):
        resources.resolve_feature('ru', 'builtin', task, str(tmp_path), str(tmp_path))
    assert 'rovers-2006' in resources.builtin_domains()


def test_generate(tmp_path):
    bench = _bench(tmp_path)
    exp = _experiment(tmp_path)
    sandbox = tmp_path / 'sandbox'
    args = types.SimpleNamespace(exp_dir=str(exp), sandbox_dir=str(sandbox), tasks_dir=[f'mini={bench}'],
                                 venv_dir='/opt/venv', conda_env=None, apptainer_image=None,
                                 skip_existing=False, no_validate=False, local_jobs=2, suite_tracks=None,
                                 tracks=None, domains=None, exclude_domains=None, ipc_years=None,
                                 max_instances_per_domain=None, selection=None)
    assert generator.generate(args) == 0
    tasks = json.loads((sandbox / 'tasks.json').read_text())
    # A: 2 models x 2 k x 2 q x 2 instances; B: model go only
    assert sum(1 for e in tasks['expected'] if e['planner'] == 'A') == 16
    assert sum(1 for e in tasks['expected'] if e['planner'] == 'B') == 8
    lines = (sandbox / 'cmds' / 'A.txt').read_text().splitlines()
    assert len(lines) == 16
    assert all(line.startswith('source /opt/venv/bin/activate && dpevalcli solve') for line in lines)
    assert any('--feature ru=' in line and '--q 1.2' in line and '--k 10' in line for line in lines)
    # 16 commands with arrays of at most 5 -> 4 chunks
    sbatch = sorted(p for p in os.listdir(sandbox / 'slurm') if p.startswith('dpeval-A'))
    assert len(sbatch) == 4
    body = (sandbox / 'slurm' / sbatch[0]).read_text()
    assert '#SBATCH --partition=long' in body and '#SBATCH --qos=high' in body
    assert '#SBATCH --time=00:15:00' in body and '#SBATCH --mem=5120M' in body

    # resume: a result file for one task drops exactly that command
    first = lines[0]
    results_dir = first.split('--results-dir ')[1].split()[0]
    os.makedirs(results_dir, exist_ok=True)
    slug = first.split('--task-id ')[1].split()[0].replace(':', '_')
    open(os.path.join(results_dir, f'{slug}.json'), 'w').write('{}')
    args.skip_existing = True
    assert generator.generate(args) == 0
    assert len((sandbox / 'cmds' / 'A.txt').read_text().splitlines()) == 15
    assert len(json.loads((sandbox / 'tasks.json').read_text())['expected']) == 24


def test_generate_required_feature_missing(tmp_path):
    bench = _bench(tmp_path)
    shutil.rmtree(tmp_path / 'res')
    (tmp_path / 'res').mkdir()
    exp = _experiment(tmp_path, extra_planner={'planner-tag': 'B', 'adapter': 'fi', 'venv-dir': '/opt/fi'})
    sandbox = tmp_path / 'sandbox'
    args = types.SimpleNamespace(exp_dir=str(exp), sandbox_dir=str(sandbox), tasks_dir=[str(bench)],
                                 venv_dir=None, conda_env=None, apptainer_image=None, skip_existing=False,
                                 no_validate=False, local_jobs=2, suite_tracks=None, tracks=None, domains=None,
                                 exclude_domains=None, ipc_years=None, max_instances_per_domain=None,
                                 selection=None)
    assert generator.generate(args) == 0
    tasks = json.loads((sandbox / 'tasks.json').read_text())
    assert {e['model'] for e in tasks['expected']} == {'go'}
    assert len(tasks['not-applicable']) == 2
    # the planner's own environment overrides the (absent) global one
    assert (sandbox / 'cmds' / 'B.txt').read_text().startswith('source /opt/fi/bin/activate && ')


def test_analyzer_on_synthetic(tmp_path):
    truth = synthetic.build(str(tmp_path))
    rows = analyzer.load_rows(str(tmp_path))
    assert len(rows) == len(truth['statuses'])
    for row in rows:
        assert row['status'] == truth['statuses'][(row['planner'], row['k'], row['q'], row['task_id'])]

    coverage = {(c['planner'], c['k'], c['q']): c for c in analyzer.coverage_table(rows)}
    fi = coverage[('FI', 5, 1.2)]
    expected_solved = sum(1 for (p, k, q, t), s in truth['statuses'].items()
                          if (p, k, q) == ('FI', 5, 1.2) and s in ('SOLVED', 'EXHAUSTED')) - 1  # one invalid
    assert fi['solved'] == expected_solved and fi['invalid-plan-sets'] == 1
    assert coverage[('BP-ASP', 5, 1.0)]['missing'] == 1
    assert coverage[('BP-SMT', 5, 1.0)]['killed'] == 1

    group = ('go-ru', 5, 1.0)
    common = analyzer.common_solved(rows, group)
    for planner in synthetic.PLANNERS:
        solved = {r['task_id'] for r in analyzer.in_group(rows, group) if r['planner'] == planner and r['solved']}
        assert set(common) <= solved
    diversity = [d for d in analyzer.diversity_table(rows) if (d['model'], d['k'], d['q']) == group]
    assert {d['common-solved'] for d in diversity} == {len(common)}

    strict = analyzer.load_rows(str(tmp_path), strict_k=True)
    assert sum(r['solved'] for r in strict) < sum(r['solved'] for r in rows)


def test_analyze_and_report_cli(tmp_path):
    synthetic.build(str(tmp_path))
    from dp_eval_toolkit.cli import main
    assert main(['analyze', '--sandbox-dir', str(tmp_path), '--per-domain']) == 0
    for name in ('results.csv', 'coverage.csv', 'diversity-common-solved.csv', 'summary.txt'):
        assert (tmp_path / 'analysis' / name).exists()
    assert main(['report', '--sandbox-dir', str(tmp_path), '--no-plots']) == 0
    assert r'\begin{tabular}' in (tmp_path / 'report' / 'coverage.tex').read_text()


def test_report_figures(tmp_path):
    pytest.importorskip('matplotlib')
    synthetic.build(str(tmp_path), instances=8)
    from dp_eval_toolkit.cli import main
    assert main(['report', '--sandbox-dir', str(tmp_path), '--formats', 'png']) == 0
    plots = os.listdir(tmp_path / 'report' / 'plots')
    for name in ('survival-grid-go-ru.png', 'coverage-vs-k-go-ru.png', 'outcomes-go-ru.png',
                 'bcoverage-distribution-go-ru.png', 'bcoverage-vs-k-go-ru.png'):
        assert name in plots
    assert len(os.listdir(tmp_path / 'report' / 'plots' / 'survival')) == 6


# ----------------------------------------------------------------------
# Behaviour evaluator (needs unified-planning)

def test_behaviour_evaluator(tmp_path):
    pytest.importorskip('unified_planning')
    from unified_planning.io import PDDLReader
    from dp_eval_toolkit.behaviours import evaluate_plan_set

    bench = _bench(tmp_path)
    reader = PDDLReader()
    problem = reader.parse_problem(str(bench / 'delivery' / 'domain.pddl'), str(bench / 'delivery' / 'p01.pddl'))
    plans_text = [
        # p1 first, one truck
        '(load p1 t1 src)\n(drive t1 src dest)\n(unload p1 t1 dest)\n(drive t1 dest src)\n'
        '(load p2 t1 src)\n(drive t1 src dest)\n(unload p2 t1 dest)',
        # p2 first, one truck
        '(load p2 t1 src)\n(drive t1 src dest)\n(unload p2 t1 dest)\n(drive t1 dest src)\n'
        '(load p1 t1 src)\n(drive t1 src dest)\n(unload p1 t1 dest)',
        # both at once, two trucks
        '(load p1 t1 src)\n(load p2 t2 src)\n(drive t1 src dest)\n(drive t2 src dest)\n'
        '(unload p1 t1 dest)\n(unload p2 t2 dest)',
        # same behaviour as the first
        '(load p1 t2 src)\n(drive t2 src dest)\n(unload p1 t2 dest)\n(drive t2 dest src)\n'
        '(load p2 t2 src)\n(drive t2 src dest)\n(unload p2 t2 dest)',
    ]
    plans = [reader.parse_plan_string(problem, text) for text in plans_text]
    scores = evaluate_plan_set(problem, plans, [('go', {}), ('ru', str(tmp_path / 'res' / 'delivery' / 'resources.txt'))])
    assert scores['errors'] == []
    assert scores['b-coverage'] == 3
    # Behaviours (goal ranks, resources used): A = p1 then p2 with one truck, B = p2 then
    # p1 with one truck, C = p1 then p2 with two trucks; the fourth plan repeats A.
    # psi_M = mean of the two features: A-B = (1 + 0) / 2, A-C = (0 + 1) / 2, B-C = (1 + 1) / 2.
    assert scores['b-maxsum'] == pytest.approx(2.0)
    assert scores['distinct-plans'] == 4
    assert scores['costs'] == [7, 7, 6, 7]
