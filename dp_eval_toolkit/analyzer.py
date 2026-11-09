"""Aggregate result files into tables (stdlib only).

Groups are (diversity model, k, q). Within a group:

* **coverage / outcomes over all instances** -- per planner (and per track):
  attempted, solved, coverage, timeouts (and how many of them still returned
  some plans), memouts, errors, ... ``attempted`` is the set of diversity tasks
  ``generate`` asked for (``tasks.json``), so a task that never produced a
  result is counted as ``MISSING`` instead of shrinking the denominator;
* **diversity over the commonly solved instances** -- the instances *every*
  planner of the group solved, so that a planner solving only easy instances
  cannot look more (or less) diverse for it: mean B-Coverage, B-MaxSum, the
  normalised B-MaxSum, stability, uniqueness, plus how often each planner had
  the strictly best B-Coverage on an instance.

"Solved" means the planner terminated with a complete answer -- ``k`` plans
(``SOLVED``) or every plan the quality bound admits when there are fewer
(``EXHAUSTED``) -- and no returned plan failed validation. ``--strict-k``
counts ``SOLVED`` only.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from .status import ERROR, EXHAUSTED, KILLED, MEMOUT, MISSING, SOLVED, TIMEOUT, UNSOLVABLE, UNSUPPORTED

ROW_COLUMNS = [
    'planner', 'model', 'k', 'q', 'task_id', 'suite', 'domain', 'instance', 'track', 'ipc',
    'status', 'solved', 'planning_seconds', 'total_seconds', 'peak_memory_mb',
    'plan_count', 'valid_plans', 'invalid_plans', 'distinct_plans',
    'b_coverage', 'b_maxsum', 'b_maxsum_normalised', 'b_maxmin', 'b_novelty', 'stability', 'uniqueness',
    'min_cost', 'max_cost', 'mean_cost', 'native_b_coverage', 'optimal_length', 'eval_errors',
]

# Diversity indicators averaged over the commonly solved instances.
METRICS = ['b_coverage', 'b_maxsum', 'b_maxsum_normalised', 'b_maxmin', 'b_novelty',
           'stability', 'uniqueness', 'distinct_plans', 'plan_count', 'mean_cost']
# ... and the ones that also get a median and a sum.
DETAILED_METRICS = ('b_coverage', 'b_maxsum')

VIOLATION_COLUMNS = ['planner', 'model', 'k', 'q', 'task_id', 'max-cost', 'reference', 'bound']

Group = tuple[str, int, float]
Row = dict[str, Any]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_rows(sandbox: str, results_dir: str | None = None, strict_k: bool = False) -> list[Row]:
    """One row per expected diversity task (plus any result not in tasks.json)."""
    sandbox = os.path.abspath(os.path.expanduser(sandbox))
    results, markers = _read_payloads(results_dir or os.path.join(sandbox, 'results'))

    rows: dict[tuple, Row] = {}
    for payload in results:
        row = _row(payload)
        rows[_key(row)] = row
    # A marker without a result: the job was killed before it could report.
    for marker in markers:
        row = _row({**marker, 'status': KILLED})
        rows.setdefault(_key(row), row)
    for entry in _expected_tasks(sandbox):
        row = _expected_row(entry)
        rows.setdefault(_key(row), row)

    for row in rows.values():
        row['solved'] = is_solved(row, strict_k)
    return sorted(rows.values(), key=lambda r: (r['model'], r['k'], r['q'], r['planner'], r['task_id']))


def _read_payloads(results_dir: str) -> tuple[list[dict], list[dict]]:
    """Every readable ``.json`` result and ``.running`` marker under `results_dir`."""
    results: list[dict] = []
    markers: list[dict] = []
    for root, _dirs, files in os.walk(results_dir):
        for name in files:
            if name.endswith('.json'):
                target = results
            elif name.endswith('.running'):
                target = markers
            else:
                continue
            try:
                with open(os.path.join(root, name)) as handle:
                    target.append(json.load(handle))
            except (OSError, ValueError):
                continue
    return results, markers


def _expected_tasks(sandbox: str) -> list[dict]:
    tasks_file = os.path.join(sandbox, 'tasks.json')
    if not os.path.exists(tasks_file):
        return []
    with open(tasks_file) as handle:
        return json.load(handle).get('expected', [])


def _expected_row(entry: dict) -> Row:
    """The row of a task ``generate`` asked for but no result file reports."""
    row = dict.fromkeys(ROW_COLUMNS)
    row.update({'planner': entry['planner'], 'model': entry['model'], 'k': int(entry['k']),
                'q': float(entry['q']), 'task_id': entry['task_id'], 'status': MISSING})
    for column in ('suite', 'domain', 'instance', 'track', 'ipc'):
        row[column] = entry.get(column)
    return row


def is_solved(row: Row, strict_k: bool = False) -> bool:
    complete = (SOLVED,) if strict_k else (SOLVED, EXHAUSTED)
    has_plans = (row.get('plan_count') or 0) > 0
    all_valid = not (row.get('invalid_plans') or 0)
    return row.get('status') in complete and has_plans and all_valid


def _key(row: Row) -> tuple:
    return (row['planner'], row['model'], int(row['k']), float(row['q']), row['task_id'])


def _row(payload: dict[str, Any]) -> Row:
    """Flatten one result file into a table row."""
    task = payload.get('task') or {}
    div = payload.get('diversity') or {}
    timings = payload.get('timings') or {}
    metrics = payload.get('metrics') or {}
    scores = payload.get('diversity-scores') or {}
    info = payload.get('info') or {}
    costs = scores.get('costs') or metrics.get('plan-lengths') or []
    return {
        'planner': (payload.get('planner') or {}).get('tag'),
        'model': div.get('model'), 'k': int(div.get('k') or 0), 'q': float(div.get('q') or 1.0),
        'task_id': task.get('task-id'), 'suite': task.get('suite'), 'domain': task.get('domain'),
        'instance': task.get('instance'), 'track': task.get('track'), 'ipc': task.get('ipc'),
        'status': payload.get('status'),
        'planning_seconds': timings.get('planning-seconds'),
        'total_seconds': timings.get('total-seconds'),
        'peak_memory_mb': metrics.get('peak-memory-mb'),
        'plan_count': metrics.get('plan-count', len(payload.get('plans') or [])),
        'valid_plans': metrics.get('valid-plans'),
        'invalid_plans': metrics.get('invalid-plans'),
        'distinct_plans': scores.get('distinct-plans'),
        'b_coverage': scores.get('b-coverage'),
        'b_maxsum': scores.get('b-maxsum'),
        'b_maxsum_normalised': scores.get('b-maxsum-normalised'),
        'b_maxmin': scores.get('b-maxmin'),
        'b_novelty': scores.get('b-novelty'),
        'stability': scores.get('stability'),
        'uniqueness': scores.get('uniqueness'),
        'min_cost': min(costs) if costs else None,
        'max_cost': max(costs) if costs else None,
        'mean_cost': _mean(costs),
        'native_b_coverage': info.get('native-b-coverage'),
        'optimal_length': info.get('optimal-length'),
        'eval_errors': len(scores.get('errors') or []),
    }


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def groups(rows: Sequence[Row]) -> list[Group]:
    return sorted({(r['model'], r['k'], r['q']) for r in rows})


def planners_of(rows: Sequence[Row]) -> list[str]:
    return sorted({r['planner'] for r in rows})


def in_group(rows, group: Group) -> list[Row]:
    model, k, q = group
    return [r for r in rows if r['model'] == model and r['k'] == k and r['q'] == q]


def _bucket(rows: Sequence[Row], *columns: str) -> list[tuple[tuple, list[Row]]]:
    """Rows grouped by the given columns, in sorted key order."""
    buckets: dict[tuple, list[Row]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[c] for c in columns)].append(row)
    return sorted(buckets.items())


def coverage_table(rows: Sequence[Row], by_track: bool = False) -> list[Row]:
    """Outcomes per (group, planner[, track]) over every attempted instance."""
    columns = ('model', 'k', 'q', 'planner') + (('track',) if by_track else ())
    return [{**dict(zip(columns, key)), **_outcome_summary(members)}
            for key, members in _bucket(rows, *columns)]


def _outcome_summary(members: Sequence[Row]) -> Row:
    status = Counter(r['status'] for r in members)
    solved = sum(1 for r in members if r['solved'])
    timeouts = [r for r in members if r['status'] == TIMEOUT]
    times = [r['planning_seconds'] for r in members if r['solved'] and r['planning_seconds'] is not None]
    return {
        'attempted': len(members), 'solved': solved,
        'coverage': solved / len(members) if members else 0.0,
        'complete-k': status[SOLVED], 'exhausted': status[EXHAUSTED],
        'timeouts': status[TIMEOUT],
        'timeouts-with-plans': sum(1 for r in timeouts if (r['plan_count'] or 0) > 0),
        'mean-plans-at-timeout': _mean([r['plan_count'] or 0 for r in timeouts]),
        'memouts': status[MEMOUT], 'unsolvable': status[UNSOLVABLE],
        'unsupported': status[UNSUPPORTED], 'errors': status[ERROR],
        'killed': status[KILLED], 'missing': status[MISSING],
        'invalid-plan-sets': sum(1 for r in members if (r['invalid_plans'] or 0) > 0),
        'mean-time-solved': _mean(times), 'median-time-solved': _median(times),
    }


def common_solved(rows: Sequence[Row], group: Group) -> list[str]:
    """Task ids solved by every planner that took part in the group."""
    members = in_group(rows, group)
    planners = planners_of(members)
    if not planners:
        return []
    solved_by = [{r['task_id'] for r in members if r['planner'] == p and r['solved']} for p in planners]
    return sorted(set.intersection(*solved_by))


def diversity_table(rows: Sequence[Row]) -> list[Row]:
    """Diversity indicators per (group, planner) over the commonly solved instances."""
    table = []
    for group in groups(rows):
        members = in_group(rows, group)
        common = set(common_solved(rows, group))
        by_task = _rows_by_task(members, common)
        wins, ties = _bcoverage_leaders(by_task)
        for planner in planners_of(members):
            selected = [by_task[t][planner] for t in sorted(by_task) if planner in by_task[t]]
            entry = {'model': group[0], 'k': group[1], 'q': group[2], 'planner': planner,
                     'common-solved': len(common), **_metric_summary(selected)}
            entry['b-coverage-ratio'] = _mean([(r['b_coverage'] or 0) / group[1] for r in selected])
            entry['best-b-coverage'] = wins[planner]
            entry['tied-b-coverage'] = ties[planner]
            table.append(entry)
    return table


def _rows_by_task(members: Sequence[Row], common: set[str]) -> dict[str, dict[str, Row]]:
    """``task_id -> planner -> row`` over the commonly solved tasks."""
    by_task: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in members:
        if r['task_id'] in common:
            by_task[r['task_id']][r['planner']] = r
    return by_task


def _bcoverage_leaders(by_task: dict[str, dict[str, Row]]) -> tuple[Counter, Counter]:
    """Per planner: on how many tasks it alone had the best B-Coverage (wins),
    and on how many it shared the best value (ties)."""
    wins: Counter = Counter()
    ties: Counter = Counter()
    for per_planner in by_task.values():
        values = {p: (r['b_coverage'] or 0) for p, r in per_planner.items()}
        best = max(values.values())
        leaders = [p for p, v in values.items() if v == best]
        (wins if len(leaders) == 1 else ties).update(leaders)
    return wins, ties


def _metric_summary(selected: Sequence[Row]) -> Row:
    summary: Row = {}
    for metric in METRICS:
        values = [r[metric] for r in selected if r[metric] is not None]
        summary[f'mean-{metric}'] = _mean(values)
        if metric in DETAILED_METRICS:
            summary[f'median-{metric}'] = _median(values)
            summary[f'sum-{metric}'] = sum(values) if values else None
    return summary


def per_domain_table(rows: Sequence[Row]) -> list[Row]:
    columns = ('model', 'k', 'q', 'track', 'domain', 'planner')
    return [{**dict(zip(columns, key)),
             'attempted': len(members),
             'solved': sum(1 for r in members if r['solved']),
             'timeouts': sum(1 for r in members if r['status'] == TIMEOUT),
             'mean-b-coverage-solved': _mean([r['b_coverage'] for r in members
                                              if r['solved'] and r['b_coverage'] is not None])}
            for key, members in _bucket(rows, *columns)]


def quality_bound_check(rows: Sequence[Row]) -> list[Row]:
    """Plan sets whose costliest plan exceeds q x the best known shortest plan.

    The best known shortest plan of an instance is the cheapest plan any
    planner returned for it under any (model, k, q) -- or a planner's reported
    optimal length, when smaller. A violation means the planner returned a plan
    outside the quality bound it was asked for (or the reference is not
    optimal, which the ``reference`` column lets you judge).
    """
    best: dict[str, float] = {}
    for r in rows:
        for value in (r.get('min_cost'), r.get('optimal_length')):
            if value:
                best[r['task_id']] = min(best.get(r['task_id'], math.inf), float(value))
    violations = []
    for r in rows:
        if not r['solved'] or r['max_cost'] is None or r['task_id'] not in best:
            continue
        bound = math.floor(r['q'] * best[r['task_id']] + 1e-9)
        if r['max_cost'] > bound:
            violations.append({'planner': r['planner'], 'model': r['model'], 'k': r['k'], 'q': r['q'],
                               'task_id': r['task_id'], 'max-cost': r['max_cost'],
                               'reference': best[r['task_id']], 'bound': bound})
    return violations


def native_agreement(rows: Sequence[Row]) -> list[Row]:
    """Where a planner reports its own B-Coverage: how often it matches ours."""
    compared: Counter = Counter()
    agree: Counter = Counter()
    for r in rows:
        if r['native_b_coverage'] is None or r['b_coverage'] is None:
            continue
        key = (r['planner'], r['model'])
        compared[key] += 1
        agree[key] += int(r['native_b_coverage'] == r['b_coverage'])
    return [{'planner': p, 'model': m, 'compared': compared[(p, m)], 'agree': agree[(p, m)]}
            for p, m in sorted(compared)]


# ----------------------------------------------------------------------
# CLI stage
# ----------------------------------------------------------------------

def analyze(args) -> int:
    strict_k = bool(getattr(args, 'strict_k', False))
    rows = load_rows(args.sandbox_dir, args.results_dir, strict_k=strict_k)
    if not rows:
        print(f'No results under {args.sandbox_dir}.', file=sys.stderr)
        return 1
    out_dir = args.output_dir or os.path.join(os.path.abspath(args.sandbox_dir), 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    coverage = coverage_table(rows)
    coverage_track = coverage_table(rows, by_track=True)
    diversity = diversity_table(rows)
    per_domain = per_domain_table(rows)
    violations = quality_bound_check(rows)
    agreement = native_agreement(rows)

    write_csv(os.path.join(out_dir, 'results.csv'), rows, ROW_COLUMNS)
    write_csv(os.path.join(out_dir, 'coverage.csv'), coverage)
    write_csv(os.path.join(out_dir, 'coverage-per-track.csv'), coverage_track)
    write_csv(os.path.join(out_dir, 'diversity-common-solved.csv'), diversity)
    write_csv(os.path.join(out_dir, 'per-domain.csv'), per_domain)
    write_csv(os.path.join(out_dir, 'quality-bound-violations.csv'), violations, VIOLATION_COLUMNS)
    common = {f'{m}|k={k}|q={q}': common_solved(rows, (m, k, q)) for m, k, q in groups(rows)}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as handle:
        json.dump({'coverage': coverage, 'coverage-per-track': coverage_track, 'diversity': diversity,
                   'common-solved': common, 'quality-bound-violations': violations,
                   'native-agreement': agreement, 'strict-k': strict_k},
                  handle, indent=1, default=str)
    text = render_summary(rows, coverage, diversity, violations, agreement,
                          per_domain if getattr(args, 'per_domain', False) else None)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as handle:
        handle.write(text)
    print(text)
    print(f'\nWritten to {out_dir}')
    _list_errors(args.sandbox_dir)
    return 0


def render_summary(rows, coverage, diversity, violations, agreement, per_domain=None) -> str:
    lines: list[str] = []
    for group in groups(rows):
        lines += _coverage_section(group, coverage)
        lines += _diversity_section([d for d in diversity if (d['model'], d['k'], d['q']) == group])
        lines.append('')
    if per_domain:
        lines += _per_domain_section(per_domain)
    if violations:
        lines.append(f'{len(violations)} solved plan set(s) exceed q x best-known shortest plan '
                     f'(see quality-bound-violations.csv)')
    for a in agreement:
        lines.append(f'native vs evaluated B-Coverage, {a["planner"]} / {a["model"]}: '
                     f'{a["agree"]}/{a["compared"]} agree')
    return '\n'.join(lines) + '\n'


def _coverage_section(group: Group, coverage: Sequence[Row]) -> list[str]:
    model, k, q = group
    lines = ['=' * 100, f'model={model}  k={k}  q={q}', '=' * 100,
             f'{"planner":<22}{"solved":>15}{"complete":>10}{"exhaust":>9}{"timeout":>9}'
             f'{"(w/plans)":>10}{"memout":>8}{"error":>7}{"unsup":>7}{"missing":>8}']
    for c in coverage:
        if (c['model'], c['k'], c['q']) == group:
            lines.append(f'{c["planner"]:<22}{c["solved"]:>5}/{c["attempted"]:<4}{100 * c["coverage"]:>4.0f}%'
                         f'{c["complete-k"]:>10}{c["exhausted"]:>9}{c["timeouts"]:>9}'
                         f'{c["timeouts-with-plans"]:>10}{c["memouts"]:>8}{c["errors"]:>7}'
                         f'{c["unsupported"]:>7}{c["missing"]:>8}')
    return lines


def _diversity_section(group_div: Sequence[Row]) -> list[str]:
    if not group_div:
        return []
    lines = ['', f'diversity over the {group_div[0]["common-solved"]} commonly solved instances:',
             f'{"planner":<22}{"B-Cov":>8}{"B-Cov/k":>9}{"B-MaxSum":>10}{"B-MS/pair":>10}'
             f'{"stab.":>7}{"uniq.":>7}{"#plans":>8}{"best":>6}{"tied":>6}']
    for d in group_div:
        lines.append(f'{d["planner"]:<22}{_fmt(d["mean-b_coverage"]):>8}{_fmt(d["b-coverage-ratio"]):>9}'
                     f'{_fmt(d["mean-b_maxsum"]):>10}{_fmt(d["mean-b_maxsum_normalised"]):>10}'
                     f'{_fmt(d["mean-stability"]):>7}{_fmt(d["mean-uniqueness"]):>7}'
                     f'{_fmt(d["mean-plan_count"]):>8}{d["best-b-coverage"]:>6}{d["tied-b-coverage"]:>6}')
    return lines


def _per_domain_section(per_domain: Sequence[Row]) -> list[str]:
    lines = ['per-domain coverage',
             f'{"model":<12}{"k":>4}{"q":>6}  {"domain":<34}{"planner":<22}{"solved":>10}{"TO":>5}']
    for d in per_domain:
        lines.append(f'{d["model"][:11]:<12}{d["k"]:>4}{d["q"]:>6}  {str(d["domain"])[:33]:<34}'
                     f'{d["planner"][:21]:<22}{d["solved"]:>4}/{d["attempted"]:<5}{d["timeouts"]:>5}')
    lines.append('')
    return lines


def _list_errors(sandbox: str) -> None:
    errors_dir = os.path.join(os.path.abspath(sandbox), 'errors')
    if os.path.isdir(errors_dir):
        logs = sorted(os.listdir(errors_dir))
        if logs:
            print(f'{len(logs)} traceback(s) in {errors_dir}')


def write_csv(path: str, rows: Iterable[Row], columns: Sequence[str] | None = None) -> None:
    rows = list(rows)
    columns = list(columns or (rows[0].keys() if rows else []))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _csv(row.get(c)) for c in columns})


def _csv(value):
    if isinstance(value, float):
        return round(value, 4)
    return value


def _mean(values: Sequence[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _fmt(value) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.2f}'
    return str(value)
