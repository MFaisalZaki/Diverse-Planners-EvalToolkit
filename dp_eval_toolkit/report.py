"""Paper-ready tables (text + LaTeX) and figures.

Tables (stdlib only):

* ``coverage.tex``        planners x (k, q) coverage, one table per diversity model
* ``timeouts.tex``        the same grid for timeouts (with how many kept plans)
* ``diversity.tex``       mean B-Coverage / B-MaxSum over the commonly solved
                          instances, per (k, q)
* ``report.json``         every number above

Figures (matplotlib, ``--no-plots`` skips them; drawn by :mod:`dp_eval_toolkit.figures`),
one set per diversity model:

* ``survival/<model>__k<k>__q<q>``   instances solved (y) against time on a log x
                                     axis -- a step curve per planner
* ``survival-grid-<model>``          the same as small multiples, rows k, columns q
* ``coverage-vs-k-<model>``          coverage per planner as k grows, one panel per q
* ``bcoverage-vs-k-<model>``         mean B-Coverage / k over the commonly solved
                                     instances as k grows, one panel per q
* ``outcomes-<model>``               stacked outcome bars per (k, q) and planner
* ``bcoverage-distribution-<model>`` per-instance B-Coverage / k on the commonly
                                     solved instances, a box per planner and (k, q)

Planners keep one colour, marker and line style everywhere (fixed order, never
cycled), so a figure prints in grey and still reads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from .analyzer import coverage_table, diversity_table, groups, load_rows, planners_of, quality_bound_check
from .figures import plots
from .tasks import slugify

def report(args) -> int:
    sandbox = os.path.abspath(args.sandbox_dir)
    rows = load_rows(sandbox, args.results_dir, strict_k=getattr(args, 'strict_k', False))
    if not rows:
        print(f'No results under {args.sandbox_dir}.')
        return 1
    out_dir = args.output_dir or os.path.join(sandbox, 'report')
    os.makedirs(out_dir, exist_ok=True)
    _write_tables(out_dir, rows)
    print(f'Tables written to {out_dir}')

    if getattr(args, 'no_plots', False):
        return 0
    formats = [f.strip() for f in (args.formats or 'pdf,png').split(',') if f.strip()]
    plots_dir = os.path.join(out_dir, 'plots')
    written = plots(rows, planners_of(rows), plots_dir, formats, _time_limit(sandbox))
    if written is None:
        print('matplotlib is not installed; skipped the figures (pip install matplotlib).')
    else:
        print(f'{len(written)} figure(s) written to {plots_dir}')
    return 0


def _write_tables(out_dir: str, rows) -> None:
    """The three LaTeX tables and report.json."""
    planners = planners_of(rows)
    coverage = coverage_table(rows)
    diversity = diversity_table(rows)
    models = sorted({model for model, _k, _q in groups(rows)})
    _write(os.path.join(out_dir, 'coverage.tex'),
           '\n'.join(_grid_latex(coverage, planners, m, 'coverage', 'Coverage') for m in models))
    _write(os.path.join(out_dir, 'timeouts.tex'),
           '\n'.join(_grid_latex(coverage, planners, m, 'timeouts', 'Timeouts') for m in models))
    _write(os.path.join(out_dir, 'diversity.tex'),
           '\n'.join(_diversity_latex(diversity, planners, m) for m in models))
    with open(os.path.join(out_dir, 'report.json'), 'w') as handle:
        json.dump({'planners': planners, 'coverage': coverage, 'diversity': diversity,
                   'quality-bound-violations': quality_bound_check(rows)}, handle, indent=1, default=str)


def _time_limit(sandbox: str) -> float:
    """The per-task time limit ``generate`` recorded in tasks.json, or 0."""
    try:
        with open(os.path.join(sandbox, 'tasks.json')) as handle:
            return float(json.load(handle).get('limits', {}).get('time-seconds') or 0)
    except (OSError, ValueError, TypeError):
        return 0.0


# ----------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------

def _tex(text: str) -> str:
    return str(text).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def _grid_columns(entries) -> list[tuple]:
    return sorted({(e['k'], e['q']) for e in entries})


def _latex_table(caption: str, label: str, column_spec: str, header_rows: Sequence[str],
                 body: Sequence[str]) -> str:
    return '\n'.join([
        r'\begin{table}[t]', r'\centering', r'\small',
        rf'\caption{{{caption}}}',
        rf'\label{{{label}}}',
        rf'\begin{{tabular}}{{{column_spec}}}', r'\toprule',
        *(row + r' \\' for row in header_rows), r'\midrule', *body, r'\bottomrule', r'\end{tabular}',
        r'\end{table}', ''])


def _grid_latex(coverage, planners, model, field, caption) -> str:
    """Planners x (q, k) table of coverage ratios or timeout counts."""
    entries = [c for c in coverage if c['model'] == model]
    columns = _grid_columns(entries)
    qs = sorted({q for _k, q in columns})
    ks = sorted({k for k, _q in columns})
    by_cell = {(e['planner'], e['k'], e['q']): e for e in entries}
    body = []
    for planner in planners:
        cells = [_grid_cell(by_cell.get((planner, k, q)), field) for q in qs for k in ks]
        body.append(' & '.join([_tex(planner), *cells]) + r' \\')
    note = ' Parentheses: timeouts that still returned plans.' if field == 'timeouts' else ''
    return _latex_table(
        caption=rf'{caption} per number of plans $k$ and quality bound $q$ '
                rf'(diversity model \texttt{{{_tex(model)}}}).{note}',
        label=f'tab:{field}-{slugify(model)}',
        column_spec='l' + 'r' * (len(qs) * len(ks)),
        header_rows=_grid_headers(ks, qs),
        body=body)


def _grid_headers(ks, qs) -> list[str]:
    """Two header rows: one cell per q spanning its k columns, then the k values."""
    top = ' & '.join(['', *(rf'\multicolumn{{{len(ks)}}}{{c}}{{$q={q}$}}' for q in qs)])
    bottom = ' & '.join(['Planner', *(f'$k={k}$' for _q in qs for k in ks)])
    return [top, bottom]


def _grid_cell(entry: dict[str, Any] | None, field: str) -> str:
    if entry is None:
        return '--'
    if field == 'coverage':
        return f'{entry["solved"]}/{entry["attempted"]}'
    return f'{entry["timeouts"]} ({entry["timeouts-with-plans"]})'


def _diversity_latex(diversity, planners, model) -> str:
    """Per (k, q): one row per planner with its mean diversity indicators over
    the commonly solved instances; the best B-Coverage in bold."""
    entries = [d for d in diversity if d['model'] == model]
    body: list[str] = []
    for k, q in _grid_columns(entries):
        group = [d for d in entries if d['k'] == k and d['q'] == q]
        if group:
            body += _diversity_rows(group, planners, k, q)
            body.append(r'\midrule')
    if body:
        body.pop()
    return _latex_table(
        caption=rf'Behaviour diversity over the commonly solved instances (diversity model '
                rf'\texttt{{{_tex(model)}}}): means per instance; ``best'' counts instances where the planner '
                r'alone had the highest B-Coverage.',
        label=f'tab:diversity-{slugify(model)}',
        column_spec='rrrlrrrrr',
        header_rows=[r'$k$ & $q$ & \#common & Planner & B-Cov & B-MaxSum & B-MaxSum/pair & Stability & best'],
        body=body)


def _diversity_rows(group: Sequence[dict[str, Any]], planners: Sequence[str], k: int, q: float) -> list[str]:
    by_planner = {d['planner']: d for d in group}
    best = max((d['mean-b_coverage'] or 0) for d in group)
    rows = []
    for planner in planners:
        d = by_planner.get(planner)
        if d is None:
            continue
        bcov = _num(d['mean-b_coverage'])
        if d['mean-b_coverage'] is not None and d['mean-b_coverage'] == best:
            bcov = rf'\textbf{{{bcov}}}'
        # Only the first row of a (k, q) block carries the k, q and #common cells.
        lead = f'{k} & {q} & {d["common-solved"]}' if not rows else ' & & '
        rows.append(' & '.join([lead, _tex(planner), bcov, _num(d['mean-b_maxsum']),
                                _num(d['mean-b_maxsum_normalised']), _num(d['mean-stability']),
                                str(d['best-b-coverage'])]) + r' \\')
    return rows


def _num(value) -> str:
    return '--' if value is None else f'{value:.2f}'


def _write(path: str, text: str) -> None:
    with open(path, 'w') as handle:
        handle.write(text)
