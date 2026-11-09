"""Tests for the stages that need no planner: config, resources, generate,
analyze, report -- plus the behaviour evaluator when unified-planning is
installed."""

from __future__ import annotations

import json
import os
import shutil
import types

import pytest
import synthetic

from dp_eval_toolkit import analyzer, generator, resources
from dp_eval_toolkit.config import Experiment, FeatureSpec, parse_memory, parse_time, read_instance_refs, run_label
from dp_eval_toolkit.runner import parse_feature
from dp_eval_toolkit.discovery import classify_domain, discover
from dp_eval_toolkit.tasks import Task, parse_instance_ref, select_instances

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


def _api_domain(root, directory, name, ipc, problems):
    """A classical-domains style domain directory: api.py + the files it names."""
    dom = root / directory
    dom.mkdir(parents=True)
    (dom / 'domain.pddl').write_text(DOMAIN)
    for problem in problems:
        (dom / problem).write_text(PROBLEM)
    pairs = [(f'{directory}/domain.pddl', f'{directory}/{problem}') for problem in problems]
    (dom / 'api.py').write_text(f'domains = [{{"name": {name!r}, "ipc": {ipc!r}, "problems": {pairs!r}}}]\n')


def test_select_instances(tmp_path):
    root = tmp_path / 'classical'
    # Two IPC-2008 tracks of one domain, an ADL encoding, and a domain without a year.
    _api_domain(root, 'openstacks-opt08-strips', 'openstacks', '2008', ['p01.pddl', 'p02.pddl', 'p10.pddl'])
    _api_domain(root, 'openstacks-sat08-strips', 'openstacks', '2008', ['p01.pddl', 'p02.pddl', 'p10.pddl'])
    _api_domain(root, 'openstacks-opt08-adl', 'openstacks', '2008', ['p01.pddl', 'p02.pddl', 'p10.pddl'])
    _api_domain(root, 'miconic', 'miconic', None, ['s1-0.pddl', 's10-0.pddl', 's2-0.pddl'])
    tasks = discover(str(root), suite='cd')
    assert len(tasks) == 12
    # Numbering follows the api.py problems sorted by path: p01 < p02 < p10, s1-0 < s10-0 < s2-0.
    by_id = {t.task_id: t for t in tasks}
    assert by_id['cd:miconic:s10-0'].api_index == 2 and by_id['cd:miconic:s2-0'].api_index == 3

    chosen = select_instances(tasks, ['(2008, openstacks, 3)', '(None, miconic, 2)', '(2008, openstacks, 99)'])
    assert [(t.domain, t.instance) for t in chosen] == [('miconic', 's10-0'), ('openstacks-opt08-strips', 'p10')]
    assert parse_instance_ref([2008, 'openstacks', '3']) == ('2008', 'openstacks', 3)
    with pytest.raises(ValueError):
        parse_instance_ref('(2008, openstacks)')

    # The shipped list of the paper experiments, and a file with comments.
    builtin = read_instance_refs('builtin')
    assert len(builtin) == 1076 and '(2006, rovers, 14)' in builtin
    listing = tmp_path / 'some.txt'
    listing.write_text('# header\n(2008, openstacks, 3)  # trailing\n\n(None, miconic, 1)\n')
    assert read_instance_refs(str(listing)) == ['(2008, openstacks, 3)', '(None, miconic, 1)']
    assert read_instance_refs([[2008, 'openstacks', 3]]) == ['(2008, openstacks, 3)']


def test_classify_domain(tmp_path):
    def track(text):
        path = tmp_path / 'domain.pddl'
        path.write_text(text)
        return classify_domain(str(path))

    assert track(DOMAIN) == 'classical'
    costs = DOMAIN.replace('(:predicates', '(:functions (total-cost) - number)\n  (:predicates')
    assert track(costs) == 'classical'
    fuel = DOMAIN.replace('(:predicates', '(:functions (total-cost) (fuel ?t - truck))\n  (:predicates')
    assert track(fuel) == 'numeric'
    assert track(DOMAIN.replace(':typing', ':typing :numeric-fluents')) == 'numeric'
    assert track(DOMAIN.replace('(:action drive', ';; (:durative-action old\n  (:action drive')) == 'classical'
    assert track(DOMAIN.replace('(:action drive', '(:durative-action drive')) == 'temporal'


def test_shipped_configurations_load():
    for name in os.listdir(os.path.join(ROOT, 'exp-configurations')):
        experiment = Experiment.load(os.path.join(ROOT, 'exp-configurations', name))
        assert experiment.planners


def test_builtin_resources(tmp_path):
    task = Task('classical-domains', 'rovers', 'p07', '/d.pddl', '/p07.pddl', 'classical', '2006')
    path = resources.resolve_feature('ru', 'builtin', task, str(tmp_path), str(tmp_path))
    with open(path) as handle:
        text = handle.read()
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
    with open(os.path.join(results_dir, f'{slug}.json'), 'w') as handle:
        handle.write('{}')
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
# Runner and behaviour evaluator (need unified-planning)

PLANS = [
    # p1 first, one truck
    '(load p1 t1 src)\n(drive t1 src dest)\n(unload p1 t1 dest)\n(drive t1 dest src)\n'
    '(load p2 t1 src)\n(drive t1 src dest)\n(unload p2 t1 dest)',
    # p2 first, one truck
    '(load p2 t1 src)\n(drive t1 src dest)\n(unload p2 t1 dest)\n(drive t1 dest src)\n'
    '(load p1 t1 src)\n(drive t1 src dest)\n(unload p1 t1 dest)',
    # both at once, two trucks
    '(load p1 t1 src)\n(load p2 t2 src)\n(drive t1 src dest)\n(drive t2 src dest)\n'
    '(unload p1 t1 dest)\n(unload p2 t2 dest)',
    # p1 first again, the other truck
    '(load p1 t2 src)\n(drive t2 src dest)\n(unload p1 t2 dest)\n(drive t2 dest src)\n'
    '(load p2 t2 src)\n(drive t2 src dest)\n(unload p2 t2 dest)',
]


def test_solve_with_custom_adapter(tmp_path):
    pytest.importorskip('unified_planning')
    from dp_eval_toolkit.cli import main

    bench = _bench(tmp_path)
    planner_cfg = tmp_path / 'fake.json'
    planner_cfg.write_text(json.dumps({'planner-tag': 'FAKE', 'adapter': 'fake_adapter:FakeAdapter',
                                       'planner-params': {'plans': PLANS[:3]}}))
    results_dir = tmp_path / 'results'
    assert main(['solve', '--planner-cfg', str(planner_cfg),
                 '--domain', str(bench / 'delivery' / 'domain.pddl'), '--problem', str(bench / 'delivery' / 'p01.pddl'),
                 '--task-id', 'mini:delivery:p01', '--k', '5', '--q', '1.2', '--model', 'go-ru',
                 '--feature', 'go', '--feature', f'ru={tmp_path / "res" / "delivery" / "resources.txt"}',
                 '--results-dir', str(results_dir), '--errors-dir', str(tmp_path / 'errors'),
                 '--run-dir', str(tmp_path / 'runs'), '--time-limit', '0', '--memory-limit', '0']) == 0
    result = json.loads((results_dir / 'mini_delivery_p01.json').read_text())
    assert result['status'] == 'EXHAUSTED' and result['adapter-outcome'] == 'EXHAUSTED'
    assert result['metrics']['plan-count'] == 3 and result['metrics']['validated'] is True
    assert result['metrics']['valid-plans'] == 3 and result['metrics']['invalid-plans'] == 0
    assert result['diversity-scores']['b-coverage'] == 3
    assert result['diversity-scores']['costs'] == [7, 7, 6]
    assert result['info'] == {'optimal-length': 6} and result['native-behaviours'] == ['fake'] * 3
    assert len(result['plans']) == 3 and result['plans'][2][0].startswith('(load p1 t1 src)')
    assert not list(results_dir.glob('*.running')) and not list((tmp_path / 'runs').glob('FAKE__*'))
    assert result['timings']['planning-seconds'] >= 0 and 'evaluation-seconds' in result['timings']

def test_behaviour_evaluator(tmp_path):
    pytest.importorskip('unified_planning')
    from unified_planning.io import PDDLReader
    from dp_eval_toolkit.behaviours import evaluate_plan_set

    bench = _bench(tmp_path)
    reader = PDDLReader()
    problem = reader.parse_problem(str(bench / 'delivery' / 'domain.pddl'), str(bench / 'delivery' / 'p01.pddl'))
    plans = [reader.parse_plan_string(problem, text) for text in PLANS]
    resources_file = str(tmp_path / 'res' / 'delivery' / 'resources.txt')

    # Behaviours (goal order, number of trucks used -- the planners' `ru`, the
    # counter's `rc`): A = p1 then p2 with one truck, B = p2 then p1 with one truck,
    # C = p1 then p2 with both trucks; D repeats A with the other truck. psi_M is the
    # mean of go (Hamming / 2 goals) and ru (|a - b| / (a + b)):
    # A-B = (1 + 0)/2, A-C = (0 + 1/3)/2, B-C = (1 + 1/3)/2.
    scores = evaluate_plan_set(problem, plans, [('go', {}), ('ru', resources_file)])
    assert scores['errors'] == [] and scores['notes'] == []
    assert scores['b-coverage'] == 3
    assert scores['b-maxsum'] == pytest.approx(4 / 3)
    assert scores['distinct-plans'] == 4
    assert scores['costs'] == [7, 7, 6, 7]
    assert all(b.startswith('go:') and ' $$ rc:' in b for b in scores['behaviours'])

    # Naming the trucks instead of counting them tells D apart from A: four
    # behaviours, A-B = (1 + 0)/2, A-C = (0 + 1/2)/2, A-D = (0 + 1)/2, B-C = (1 + 1/2)/2,
    # B-D = (1 + 1)/2, C-D = (0 + 1/2)/2 (Jaccard over the used sets).
    scores = evaluate_plan_set(problem, plans, [('go', {}), ('ru-set', resources_file)])
    assert scores['b-coverage'] == 4
    assert scores['b-maxsum'] == pytest.approx(3.25)
    assert scores['b-maxmin'] == pytest.approx(0.25)

    # Post-hoc selection, as the FI configurations use it: two plans by B-Coverage
    # cover two distinct behaviours; a pool of k or fewer comes back untouched.
    from dp_eval_toolkit.behaviours import select_plans
    picked = select_plans(problem, plans, [('go', {}), ('ru', resources_file)], 2, 'B-Coverage')
    assert len(picked) == 2 and len({p.behaviour for p in picked}) == 2
    assert select_plans(problem, plans, [('go', {})], 4, 'B-MaxSum') == plans
    with pytest.raises(ValueError):
        select_plans(problem, plans, [('go', {})], 2, 'B-Mean')

    # A plan that cannot be replayed is reported, not counted.
    broken = reader.parse_plan_string(problem, '(unload p1 t1 dest)')
    scores = evaluate_plan_set(problem, [*plans, broken], [('go', {})])
    assert len(scores['errors']) == 1 and scores['evaluated-plans'] == 4


def test_bp_asp_naive_ignores_the_model():
    from dp_eval_toolkit.adapters import load_adapter
    from dp_eval_toolkit.adapters.base import PlanningContext

    ctx = PlanningContext(problem=None, domain_file='', problem_file='', k=2, q=1.0,
                          features=[('go', {}), ('cb', {})], deadline=None, workdir='')
    assert load_adapter('bp-asp', {'indicator': 'B-Coverage'}).unsupported_reason(ctx) == \
        "bp-asp does not support feature(s) ['cb']"
    assert load_adapter('bp-asp', {'naive': True}).unsupported_reason(ctx) is None


def test_bp_asp_naive_solve(tmp_path):
    pytest.importorskip('behaviour_planning_asp')
    from dp_eval_toolkit.cli import main

    bench = _bench(tmp_path)
    planner_cfg = tmp_path / 'naive.json'
    planner_cfg.write_text(json.dumps({'planner-tag': 'NAIVE', 'adapter': 'bp-asp',
                                       'planner-params': {'naive': True, 'select': 'B-MaxSum', 'pool-size': '2x'}}))
    results_dir = tmp_path / 'results'
    assert main(['solve', '--planner-cfg', str(planner_cfg),
                 '--domain', str(bench / 'delivery' / 'domain.pddl'), '--problem', str(bench / 'delivery' / 'p01.pddl'),
                 '--task-id', 'mini:delivery:p01', '--k', '5', '--q', '1.2', '--model', 'go-ru',
                 '--feature', 'go', '--feature', f'ru={tmp_path / "res" / "delivery" / "resources.txt"}',
                 '--results-dir', str(results_dir), '--errors-dir', str(tmp_path / 'errors'),
                 '--run-dir', str(tmp_path / 'runs'), '--time-limit', '120', '--memory-limit', '0']) == 0
    result = json.loads((results_dir / 'mini_delivery_p01.json').read_text())
    assert result['status'] == 'SOLVED' and result['adapter-outcome'] == 'COMPLETE'
    assert result['metrics']['plan-count'] == 5 and result['metrics']['valid-plans'] == 5
    assert len({tuple(p) for p in result['plans']}) == 5, 'the naive planner returns distinct plans'
    # An empty behaviour space: the planner sees one (empty) behaviour; the evaluator scores the set.
    assert result['info'] == {'indicator': 'B-Coverage', 'naive': True, 'optimal-length': 5, 'cost-bound': 6,
                              'native-b-coverage': 1, 'pool-plans': 10, 'pool-asked': 10, 'selection': 'B-MaxSum'}
    assert 'selected 5 of 10 plans by B-MaxSum' in result['logs']
    assert result['native-behaviours'] == ['<>'] * 5
    assert 1 <= result['diversity-scores']['b-coverage'] <= 5
    assert all(cost <= 6 for cost in result['diversity-scores']['costs'])
