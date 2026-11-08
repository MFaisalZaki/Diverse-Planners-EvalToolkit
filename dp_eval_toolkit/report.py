"""Paper-ready tables (text + LaTeX) and figures.

Tables (stdlib only):

* ``coverage.tex``        planners x (k, q) coverage, one table per diversity model
* ``timeouts.tex``        the same grid for timeouts (with how many kept plans)
* ``diversity.tex``       mean B-Coverage / B-MaxSum over the commonly solved
                          instances, per (k, q)
* ``report.json``         every number above

Figures (matplotlib, ``--no-plots`` skips them), one set per diversity model:

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
from typing import Any, Dict, List, Optional, Sequence

from .analyzer import (EXHAUSTED, KILLED, MEMOUT, MISSING, SOLVED, STATUS_ORDER, TIMEOUT, UNSOLVABLE,
                       UNSUPPORTED, ERROR, common_solved, coverage_table, diversity_table, groups,
                       in_group, load_rows, planners_of, quality_bound_check)

# Categorical slots, fixed order (validated default palette, light mode).
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SERIES_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
SERIES_DASHES = [(None, None), (5, 2), (1.5, 1.5), (7, 2, 1.5, 2), (3, 1, 1, 1), (8, 3), (2, 2, 6, 2), (4, 4)]
INK, INK_MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
# Outcomes: solved dark, failure modes greys + hatches (status colours stay out of series).
STATUS_STYLE = {
    SOLVED: ('#3b3b3a', ''), EXHAUSTED: ('#6f6e6a', ''), TIMEOUT: ('#bdbcb6', '////'),
    MEMOUT: ('#d6d5cf', 'xxxx'), UNSOLVABLE: ('#ffffff', '....'), UNSUPPORTED: ('#ffffff', '\\\\\\\\'),
    ERROR: ('#9a9994', '++'), KILLED: ('#ffffff', 'oo'), MISSING: ('#ffffff', ''),
    'INVALID': ('#ffffff', '----'),
}
OUTCOME_ORDER = STATUS_ORDER[:2] + ['INVALID'] + STATUS_ORDER[2:]
MIN_TIME = 0.1
_SANDBOX: Optional[str] = None


def _time_limit(rows) -> float:
    try:
        with open(os.path.join(_SANDBOX or '', 'tasks.json')) as handle:
            return float(json.load(handle).get('limits', {}).get('time-seconds') or 0)
    except (OSError, ValueError, TypeError):
        return 0.0


def report(args) -> int:
    global _SANDBOX
    _SANDBOX = os.path.abspath(args.sandbox_dir)
    rows = load_rows(args.sandbox_dir, args.results_dir, strict_k=getattr(args, 'strict_k', False))
    if not rows:
        print(f'No results under {args.sandbox_dir}.')
        return 1
    out_dir = args.output_dir or os.path.join(os.path.abspath(args.sandbox_dir), 'report')
    os.makedirs(out_dir, exist_ok=True)

    planners = planners_of(rows)
    coverage = coverage_table(rows)
    diversity = diversity_table(rows)
    models = sorted({g[0] for g in groups(rows)})

    tex = []
    for model in models:
        tex.append(_grid_latex(coverage, planners, model, 'coverage', 'Coverage'))
        tex.append(_grid_latex(coverage, planners, model, 'timeouts', 'Timeouts'))
        tex.append(_diversity_latex(diversity, planners, model))
    _write(os.path.join(out_dir, 'coverage.tex'), '\n'.join(t for i, t in enumerate(tex) if i % 3 == 0))
    _write(os.path.join(out_dir, 'timeouts.tex'), '\n'.join(t for i, t in enumerate(tex) if i % 3 == 1))
    _write(os.path.join(out_dir, 'diversity.tex'), '\n'.join(t for i, t in enumerate(tex) if i % 3 == 2))
    with open(os.path.join(out_dir, 'report.json'), 'w') as handle:
        json.dump({'planners': planners, 'coverage': coverage, 'diversity': diversity,
                   'quality-bound-violations': quality_bound_check(rows)}, handle, indent=1, default=str)

    print(f'Tables written to {out_dir}')
    if getattr(args, 'no_plots', False):
        return 0
    formats = [f.strip() for f in (args.formats or 'pdf,png').split(',') if f.strip()]
    written = plots(rows, planners, os.path.join(out_dir, 'plots'), formats)
    if written is None:
        print('matplotlib is not installed; skipped the figures (pip install matplotlib).')
    else:
        print(f'{len(written)} figure(s) written to {os.path.join(out_dir, "plots")}')
    return 0


# ----------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------

def _tex(text: str) -> str:
    return str(text).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def _grid_columns(entries) -> List[tuple]:
    return sorted({(e['k'], e['q']) for e in entries})


def _grid_latex(coverage, planners, model, field, caption) -> str:
    entries = [c for c in coverage if c['model'] == model]
    columns = _grid_columns(entries)
    qs = sorted({q for _k, q in columns})
    ks = sorted({k for k, _q in columns})
    header1 = ' & '.join([''] + [rf'\multicolumn{{{len(ks)}}}{{c}}{{$q={q}$}}' for q in qs])
    header2 = ' & '.join(['Planner'] + [f'$k={k}$' for _q in qs for k in ks])
    body = []
    for planner in planners:
        cells = [_tex(planner)]
        for q in qs:
            for k in ks:
                e = next((c for c in entries if c['planner'] == planner and c['k'] == k and c['q'] == q), None)
                if e is None:
                    cells.append('--')
                elif field == 'coverage':
                    cells.append(f'{e["solved"]}/{e["attempted"]}')
                else:
                    cells.append(f'{e["timeouts"]} ({e["timeouts-with-plans"]})')
        body.append(' & '.join(cells) + r' \\')
    note = ' Parentheses: timeouts that still returned plans.' if field == 'timeouts' else ''
    return '\n'.join([
        r'\begin{table}[t]', r'\centering', r'\small',
        rf'\caption{{{caption} per number of plans $k$ and quality bound $q$ '
        rf'(diversity model \texttt{{{_tex(model)}}}).{note}}}',
        rf'\label{{tab:{field}-{_slug(model)}}}',
        r'\begin{tabular}{l' + 'r' * (len(qs) * len(ks)) + '}', r'\toprule',
        header1 + r' \\', header2 + r' \\', r'\midrule', *body, r'\bottomrule', r'\end{tabular}',
        r'\end{table}', ''])


def _diversity_latex(diversity, planners, model) -> str:
    entries = [d for d in diversity if d['model'] == model]
    body = []
    for k, q in _grid_columns(entries):
        group = [d for d in entries if d['k'] == k and d['q'] == q]
        if not group:
            continue
        best = max((d['mean-b_coverage'] or 0) for d in group)
        for i, planner in enumerate(planners):
            d = next((x for x in group if x['planner'] == planner), None)
            if d is None:
                continue
            bcov = _num(d['mean-b_coverage'])
            if d['mean-b_coverage'] is not None and d['mean-b_coverage'] == best:
                bcov = rf'\textbf{{{bcov}}}'
            lead = f'{k} & {q} & {d["common-solved"]}' if i == 0 else ' & & '
            body.append(' & '.join([lead, _tex(planner), bcov, _num(d['mean-b_maxsum']),
                                    _num(d['mean-b_maxsum_normalised']), _num(d['mean-stability']),
                                    str(d['best-b-coverage'])]) + r' \\')
        body.append(r'\midrule')
    if body:
        body.pop()
    return '\n'.join([
        r'\begin{table}[t]', r'\centering', r'\small',
        rf'\caption{{Behaviour diversity over the commonly solved instances (diversity model '
        rf'\texttt{{{_tex(model)}}}): means per instance; ``best'' counts instances where the planner '
        r'alone had the highest B-Coverage.}',
        rf'\label{{tab:diversity-{_slug(model)}}}',
        r'\begin{tabular}{rrrlrrrrr}', r'\toprule',
        r'$k$ & $q$ & \#common & Planner & B-Cov & B-MaxSum & B-MaxSum/pair & Stability & best \\',
        r'\midrule', *body, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def _num(value) -> str:
    return '--' if value is None else f'{value:.2f}'


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def plots(rows, planners, plots_dir: str, formats: Sequence[str]) -> Optional[List[str]]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    _style(matplotlib)
    os.makedirs(os.path.join(plots_dir, 'survival'), exist_ok=True)
    styles = {p: i for i, p in enumerate(planners)}
    written: List[str] = []
    # The x axis ends at the per-task time limit (from tasks.json), so a curve that
    # stops short of the right edge reads as "never solved the rest in time".
    limit = max([_time_limit(rows)] + [(r['planning_seconds'] or 0) for r in rows]) or 1.0
    for model in sorted({g[0] for g in groups(rows)}):
        model_groups = [g for g in groups(rows) if g[0] == model]
        for group in model_groups:
            fig, ax = plt.subplots(figsize=(4.8, 3.4))
            _draw_survival(ax, in_group(rows, group), planners, styles, limit)
            ax.set_title(f'{model}, k={group[1]}, q={group[2]}', fontsize=9, color=INK)
            _legend(ax, planners)
            written += _save(fig, os.path.join(plots_dir, 'survival'), _slug(f'{model}__k{group[1]}__q{group[2]}'),
                             formats, plt)
        written += _survival_grid(plt, rows, planners, styles, model, model_groups, limit, plots_dir, formats)
        written += _versus_k(plt, rows, planners, styles, model, plots_dir, formats, 'coverage')
        written += _versus_k(plt, rows, planners, styles, model, plots_dir, formats, 'bcoverage')
        written += _outcome_bars(plt, rows, planners, model, model_groups, plots_dir, formats)
        written += _bcoverage_boxes(plt, rows, planners, styles, model, model_groups, plots_dir, formats)
    return written


def _style(matplotlib) -> None:
    matplotlib.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8, 'axes.edgecolor': INK_MUTED,
        'axes.labelcolor': INK, 'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'legend.frameon': False, 'lines.linewidth': 2.0,
    })


def _series(index: int) -> Dict[str, Any]:
    i = index % len(SERIES_COLORS)
    dashes = SERIES_DASHES[i]
    style = {'color': SERIES_COLORS[i], 'marker': SERIES_MARKERS[i], 'markersize': 5,
             'markeredgecolor': SURFACE, 'markeredgewidth': 0.8}
    if dashes[0] is not None:
        style['dashes'] = dashes
    return style


def _draw_survival(ax, rows, planners, styles, limit) -> None:
    """Instances solved (y) within time t (x, log scale): a step per solved instance."""
    attempted = 0
    for planner in planners:
        mine = [r for r in rows if r['planner'] == planner]
        if not mine:
            continue
        attempted = max(attempted, len(mine))
        times = sorted(max(MIN_TIME, r['planning_seconds'] or MIN_TIME) for r in mine if r['solved'])
        xs = [MIN_TIME] + times + [limit]
        ys = [0] + list(range(1, len(times) + 1)) + [len(times)]
        style = _series(styles[planner])
        ax.step(xs, ys, where='post', label=planner, color=style['color'], linewidth=2,
                dashes=style.get('dashes', (None, None)))
        if times:
            ax.plot(times[-1:], [len(times)], linestyle='none', **{k: v for k, v in style.items() if k != 'dashes'})
    ax.set_xscale('log')
    ax.set_xlim(MIN_TIME, limit * 1.2)
    ax.set_ylim(0, max(1, attempted) * 1.04)
    ax.set_xlabel('time (s, log scale)')
    ax.set_ylabel('instances solved')
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='minor', visible=False)


def _legend(ax, planners, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len(planners) >= 2 and handles:
        ax.legend(handles, labels, loc=kwargs.get('loc', 'lower right'), fontsize=7)


def _survival_grid(plt, rows, planners, styles, model, model_groups, limit, plots_dir, formats) -> List[str]:
    ks = sorted({g[1] for g in model_groups})
    qs = sorted({g[2] for g in model_groups})
    fig, axes = plt.subplots(len(ks), len(qs), figsize=(2.9 * len(qs) + 0.6, 2.3 * len(ks) + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    for i, k in enumerate(ks):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            _draw_survival(ax, in_group(rows, (model, k, q)), planners, styles, limit)
            ax.set_title(f'k={k}, q={q}', fontsize=8, color=INK)
            if i < len(ks) - 1:
                ax.set_xlabel('')
            if j > 0:
                ax.set_ylabel('')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(f'Instances solved over time — model {model}', fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.06 if len(planners) >= 2 else 0, 1, 1))
    if len(planners) >= 2:
        fig.legend(handles, labels, loc='lower center', ncol=min(len(planners), 4), fontsize=8)
    return _save(fig, plots_dir, f'survival-grid-{_slug(model)}', formats, plt, layout=False)


def _versus_k(plt, rows, planners, styles, model, plots_dir, formats, what) -> List[str]:
    model_rows = [r for r in rows if r['model'] == model]
    ks = sorted({r['k'] for r in model_rows})
    qs = sorted({r['q'] for r in model_rows})
    if len(ks) < 2:
        return []
    fig, axes = plt.subplots(1, len(qs), figsize=(3.2 * len(qs) + 0.4, 2.9), sharey=True, squeeze=False)
    coverage = {(c['planner'], c['k'], c['q']): c for c in coverage_table(model_rows)}
    diversity = {(d['planner'], d['k'], d['q']): d for d in diversity_table(model_rows)}
    for j, q in enumerate(qs):
        ax = axes[0][j]
        for planner in planners:
            ys = []
            for k in ks:
                if what == 'coverage':
                    entry = coverage.get((planner, k, q))
                    ys.append(100 * entry['coverage'] if entry else None)
                else:
                    entry = diversity.get((planner, k, q))
                    ys.append(entry['b-coverage-ratio'] if entry and entry['b-coverage-ratio'] is not None else None)
            points = [(k, y) for k, y in zip(ks, ys) if y is not None]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], label=planner, **_series(styles[planner]))
        ax.set_xscale('log')
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel('number of plans k')
        ax.set_title(f'q={q}', fontsize=8, color=INK)
        if what == 'coverage':
            ax.set_ylim(0, 100)
            if j == 0:
                ax.set_ylabel('coverage (%)')
        else:
            ax.set_ylim(0, 1.05)
            if j == 0:
                ax.set_ylabel('B-Coverage / k (common solved)')
    _legend(axes[0][-1], planners, loc='best')
    name = 'coverage-vs-k' if what == 'coverage' else 'bcoverage-vs-k'
    return _save(fig, plots_dir, f'{name}-{_slug(model)}', formats, plt)


def _outcome_bars(plt, rows, planners, model, model_groups, plots_dir, formats) -> List[str]:
    labels, stacks = [], []
    for group in model_groups:
        members = in_group(rows, group)
        for planner in planners:
            mine = [r for r in members if r['planner'] == planner]
            if not mine:
                continue
            labels.append(f'k={group[1]} q={group[2]}  {planner}')
            counts = {s: 0 for s in OUTCOME_ORDER}
            for r in mine:
                status = r['status'] if r['status'] in counts else ERROR
                if status in (SOLVED, EXHAUSTED) and not r['solved']:
                    status = 'INVALID'      # a plan failed validation (or no plan)
                counts[status] += 1
            stacks.append(counts)
    if not stacks:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(labels) + 1.2))
    ys = list(range(len(labels)))[::-1]
    lefts = [0] * len(labels)
    for status in OUTCOME_ORDER:
        widths = [s[status] for s in stacks]
        if not any(widths):
            continue
        color, hatch = STATUS_STYLE[status]
        ax.barh(ys, widths, left=lefts, height=0.72, color=color, hatch=hatch, edgecolor=INK_MUTED,
                linewidth=0.5, label=status.lower())
        lefts = [l + w for l, w in zip(lefts, widths)]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('diversity tasks')
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12 - 0.6 / max(1, len(labels))), ncol=5, fontsize=7)
    ax.set_title(f'Outcomes — model {model}', fontsize=9, color=INK)
    return _save(fig, plots_dir, f'outcomes-{_slug(model)}', formats, plt)


def _bcoverage_boxes(plt, rows, planners, styles, model, model_groups, plots_dir, formats) -> List[str]:
    data, positions, colors, markers, ticks, tick_labels = [], [], [], [], [], []
    x = 0.0
    for group in model_groups:
        common = set(common_solved(rows, group))
        if not common:
            continue
        start = x
        for planner in planners:
            values = [(r['b_coverage'] or 0) / group[1] for r in in_group(rows, group)
                      if r['planner'] == planner and r['task_id'] in common]
            if values:
                data.append(values)
                positions.append(x)
                colors.append(_series(styles[planner])['color'])
                markers.append(_series(styles[planner]))
                x += 1
        ticks.append((start + x - 1) / 2)
        tick_labels.append(f'k={group[1]}\nq={group[2]}\n(n={len(common)})')
        x += 0.8
    if not data:
        return []
    fig, ax = plt.subplots(figsize=(max(4.0, 0.42 * len(data) + 1.5), 3.0))
    boxes = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=True,
                       medianprops={'color': INK, 'linewidth': 1.4},
                       flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': INK_MUTED,
                                   'markeredgecolor': 'none'})
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(INK_MUTED)
    # The mean, in the planner's colour and marker: keeps a planner visible even
    # when every instance has the same value and its box collapses to a line.
    for values, position, marker_style in zip(data, positions, markers):
        ax.plot([position], [sum(values) / len(values)], linestyle='none', zorder=5,
                **{key: v for key, v in marker_style.items() if key not in ('dashes', 'markersize')}, markersize=7)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('B-Coverage / k')
    ax.grid(axis='x', visible=False)
    from matplotlib.lines import Line2D
    if len(planners) >= 2:
        handles = [Line2D([], [], linestyle='none', markersize=7,
                          **{key: v for key, v in _series(styles[p]).items() if key not in ('dashes', 'markersize')})
                   for p in planners]
        ax.legend(handles, [f'{p} (mean)' for p in planners], loc='lower center', bbox_to_anchor=(0.5, 1.0),
                  fontsize=7, ncol=min(4, len(planners)))
    ax.set_title(f'Per-instance B-Coverage on commonly solved instances — model {model}', fontsize=9,
                 color=INK, pad=22 if len(planners) >= 2 else 6)
    return _save(fig, plots_dir, f'bcoverage-distribution-{_slug(model)}', formats, plt)


def _save(fig, directory: str, name: str, formats: Sequence[str], plt, layout: bool = True) -> List[str]:
    paths = []
    if layout:
        fig.tight_layout()
    for fmt in formats:
        path = os.path.join(directory, f'{name}.{fmt}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def _write(path: str, text: str) -> None:
    with open(path, 'w') as handle:
        handle.write(text)


def _slug(text: str) -> str:
    import re
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(text))
