"""The report's figures (matplotlib), one set per diversity model.

Planners keep one colour, marker and line style everywhere (fixed order, never
cycled), so a figure prints in grey and still reads. :class:`Canvas` carries
what every figure needs; each figure function draws one kind of plot.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .analyzer import common_solved, coverage_table, diversity_table, groups, in_group
from .status import ERROR, EXHAUSTED, KILLED, MEMOUT, MISSING, SOLVED, STATUS_ORDER, TIMEOUT, UNSOLVABLE, UNSUPPORTED
from .tasks import slugify

# Categorical slots, fixed order (validated default palette, light mode).
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SERIES_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
SERIES_DASHES = [(None, None), (5, 2), (1.5, 1.5), (7, 2, 1.5, 2), (3, 1, 1, 1), (8, 3), (2, 2, 6, 2), (4, 4)]
INK, INK_MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
# Outcomes: solved dark, failure modes greys + hatches (status colours stay out of series).
INVALID = 'INVALID'            # a plan failed validation (or no plan came back)
STATUS_STYLE = {
    SOLVED: ('#3b3b3a', ''), EXHAUSTED: ('#6f6e6a', ''), TIMEOUT: ('#bdbcb6', '////'),
    MEMOUT: ('#d6d5cf', 'xxxx'), UNSOLVABLE: ('#ffffff', '....'), UNSUPPORTED: ('#ffffff', '\\\\\\\\'),
    ERROR: ('#9a9994', '++'), KILLED: ('#ffffff', 'oo'), MISSING: ('#ffffff', ''),
    INVALID: ('#ffffff', '----'),
}
OUTCOME_ORDER = [*STATUS_ORDER[:2], INVALID, *STATUS_ORDER[2:]]
MIN_TIME = 0.1



@dataclass(frozen=True)
class VersusK:
    """One "quantity against k" figure: which table it reads and what it plots."""

    name: str
    table: Callable[[Sequence[dict]], list[dict]]      # rows -> entries per (planner, k, q)
    y_of: Callable[[dict], float | None]               # the plotted value of one entry
    y_label: str
    y_range: tuple[float, float]


COVERAGE_VS_K = VersusK('coverage-vs-k', coverage_table, lambda e: 100 * e['coverage'], 'coverage (%)', (0, 100))
BCOVERAGE_VS_K = VersusK('bcoverage-vs-k', diversity_table, lambda e: e['b-coverage-ratio'],
                         'B-Coverage / k (common solved)', (0, 1.05))


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def plots(rows, planners, plots_dir: str, formats: Sequence[str], time_limit: float = 0.0) -> list[str] | None:
    """Every figure, or ``None`` when matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    _style(matplotlib)
    canvas = Canvas(plt, rows, planners, plots_dir, formats, time_limit)
    written: list[str] = []
    for model in sorted({g[0] for g in groups(rows)}):
        model_groups = [g for g in groups(rows) if g[0] == model]
        for group in model_groups:
            written += _survival(canvas, group)
        written += _survival_grid(canvas, model, model_groups)
        written += _versus_k(canvas, model, COVERAGE_VS_K)
        written += _versus_k(canvas, model, BCOVERAGE_VS_K)
        written += _outcome_bars(canvas, model, model_groups)
        written += _bcoverage_boxes(canvas, model, model_groups)
    return written


@dataclass
class Canvas:
    """What every figure needs: the rows, the planners' fixed styles, where
    and in which formats to save."""

    plt: Any
    rows: list[dict]
    planners: list[str]
    plots_dir: str
    formats: Sequence[str]
    time_limit: float = 0.0

    def __post_init__(self):
        os.makedirs(os.path.join(self.plots_dir, 'survival'), exist_ok=True)
        self._slot = {planner: i for i, planner in enumerate(self.planners)}
        # The x axis ends at the per-task time limit (from tasks.json), so a curve that
        # stops short of the right edge reads as "never solved the rest in time".
        self.time_axis_end = max([self.time_limit] + [(r['planning_seconds'] or 0) for r in self.rows]) or 1.0

    @property
    def several_planners(self) -> bool:
        return len(self.planners) >= 2

    def style(self, planner: str) -> dict[str, Any]:
        """Line + marker style of a planner: one colour, marker and dash pattern everywhere."""
        i = self._slot[planner] % len(SERIES_COLORS)
        style = {'color': SERIES_COLORS[i], 'marker': SERIES_MARKERS[i], 'markersize': 5,
                 'markeredgecolor': SURFACE, 'markeredgewidth': 0.8}
        if SERIES_DASHES[i][0] is not None:
            style['dashes'] = SERIES_DASHES[i]
        return style

    def marker(self, planner: str, markersize: int = 5) -> dict[str, Any]:
        """The marker part of a planner's style, for points drawn without a line."""
        return {**{k: v for k, v in self.style(planner).items() if k != 'dashes'}, 'markersize': markersize}

    def rows_of(self, group) -> list[dict]:
        return in_group(self.rows, group)

    def legend(self, ax, loc: str = 'lower right') -> None:
        handles, labels = ax.get_legend_handles_labels()
        if self.several_planners and handles:
            ax.legend(handles, labels, loc=loc, fontsize=7)

    def save(self, fig, name: str, subdir: str = '', layout: bool = True) -> list[str]:
        """Write `fig` in every format and close it; returns the paths."""
        if layout:
            fig.tight_layout()
        directory = os.path.join(self.plots_dir, subdir)
        paths = []
        for fmt in self.formats:
            path = os.path.join(directory, f'{name}.{fmt}')
            fig.savefig(path, dpi=200, bbox_inches='tight')
            paths.append(path)
        self.plt.close(fig)
        return paths


def _style(matplotlib) -> None:
    matplotlib.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8, 'axes.edgecolor': INK_MUTED,
        'axes.labelcolor': INK, 'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'legend.frameon': False, 'lines.linewidth': 2.0,
    })


# -- survival ------------------------------------------------------------

def _survival(canvas: Canvas, group) -> list[str]:
    model, k, q = group
    fig, ax = canvas.plt.subplots(figsize=(4.8, 3.4))
    _draw_survival(canvas, ax, canvas.rows_of(group))
    ax.set_title(f'{model}, k={k}, q={q}', fontsize=9, color=INK)
    canvas.legend(ax)
    return canvas.save(fig, slugify(f'{model}__k{k}__q{q}'), subdir='survival')


def _draw_survival(canvas: Canvas, ax, rows) -> None:
    """Instances solved (y) within time t (x, log scale): a step per solved instance."""
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    end = canvas.time_axis_end
    attempted = 0
    for planner in canvas.planners:
        mine = [r for r in rows if r['planner'] == planner]
        if not mine:
            continue
        attempted = max(attempted, len(mine))
        times = sorted(max(MIN_TIME, r['planning_seconds'] or MIN_TIME) for r in mine if r['solved'])
        xs = [MIN_TIME, *times, end]
        ys = [0, *range(1, len(times) + 1), len(times)]
        style = canvas.style(planner)
        ax.step(xs, ys, where='post', label=planner, color=style['color'], linewidth=2,
                dashes=style.get('dashes', (None, None)))
        if times:
            ax.plot(times[-1:], [len(times)], linestyle='none', **canvas.marker(planner))
    ax.set_xscale('log')
    ax.set_xlim(MIN_TIME, end * 1.2)
    ax.set_ylim(0, max(1, attempted) * 1.04)
    ax.set_xlabel('time (s, log scale)')
    ax.set_ylabel('instances solved')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='minor', visible=False)


def _survival_grid(canvas: Canvas, model, model_groups) -> list[str]:
    """The survival curves of one model as small multiples: rows k, columns q."""
    ks = sorted({g[1] for g in model_groups})
    qs = sorted({g[2] for g in model_groups})
    fig, axes = canvas.plt.subplots(len(ks), len(qs), figsize=(2.9 * len(qs) + 0.6, 2.3 * len(ks) + 0.6),
                                    sharex=True, sharey=True, squeeze=False)
    for i, k in enumerate(ks):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            _draw_survival(canvas, ax, canvas.rows_of((model, k, q)))
            ax.set_title(f'k={k}, q={q}', fontsize=8, color=INK)
            if i < len(ks) - 1:
                ax.set_xlabel('')
            if j > 0:
                ax.set_ylabel('')
    fig.suptitle(f'Instances solved over time — model {model}', fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.06 if canvas.several_planners else 0, 1, 1))
    if canvas.several_planners:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=min(len(canvas.planners), 4), fontsize=8)
    return canvas.save(fig, f'survival-grid-{slugify(model)}', layout=False)


# -- against k -----------------------------------------------------------

def _versus_k(canvas: Canvas, model, spec: VersusK) -> list[str]:
    """One quantity per planner as k grows; one panel per q."""
    model_rows = [r for r in canvas.rows if r['model'] == model]
    ks = sorted({r['k'] for r in model_rows})
    qs = sorted({r['q'] for r in model_rows})
    if len(ks) < 2:
        return []
    entries = {(e['planner'], e['k'], e['q']): e for e in spec.table(model_rows)}
    fig, axes = canvas.plt.subplots(1, len(qs), figsize=(3.2 * len(qs) + 0.4, 2.9), sharey=True, squeeze=False)
    for j, q in enumerate(qs):
        ax = axes[0][j]
        for planner in canvas.planners:
            points = _points_against_k(entries, planner, ks, q, spec.y_of)
            if points:
                ax.plot([k for k, _ in points], [y for _, y in points], label=planner, **canvas.style(planner))
        _k_axis(ax, ks)
        ax.set_title(f'q={q}', fontsize=8, color=INK)
        ax.set_ylim(*spec.y_range)
        if j == 0:
            ax.set_ylabel(spec.y_label)
    canvas.legend(axes[0][-1], loc='best')
    return canvas.save(fig, f'{spec.name}-{slugify(model)}')


def _points_against_k(entries, planner, ks, q, y_of) -> list[tuple[int, float]]:
    """``(k, y)`` for every k at which the planner has an entry with a value, at this q."""
    points = []
    for k in ks:
        entry = entries.get((planner, k, q))
        y = y_of(entry) if entry is not None else None
        if y is not None:
            points.append((k, y))
    return points


def _k_axis(ax, ks) -> None:
    ax.set_xscale('log')
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel('number of plans k')


# -- outcomes ------------------------------------------------------------

def _outcome_bars(canvas: Canvas, model, model_groups) -> list[str]:
    """One stacked bar of outcome counts per (k, q) and planner."""
    from matplotlib.ticker import MaxNLocator

    labels, stacks = [], []
    for group in model_groups:
        members = canvas.rows_of(group)
        for planner in canvas.planners:
            mine = [r for r in members if r['planner'] == planner]
            if mine:
                labels.append(f'k={group[1]} q={group[2]}  {planner}')
                stacks.append(_outcome_counts(mine))
    if not stacks:
        return []
    fig, ax = canvas.plt.subplots(figsize=(6.4, 0.28 * len(labels) + 1.2))
    ys = list(range(len(labels)))[::-1]
    lefts = [0] * len(labels)
    for status in OUTCOME_ORDER:
        widths = [s[status] for s in stacks]
        if not any(widths):
            continue
        color, hatch = STATUS_STYLE[status]
        ax.barh(ys, widths, left=lefts, height=0.72, color=color, hatch=hatch, edgecolor=INK_MUTED,
                linewidth=0.5, label=status.lower())
        lefts = [left + width for left, width in zip(lefts, widths)]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('diversity tasks')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12 - 0.6 / max(1, len(labels))), ncol=5, fontsize=7)
    ax.set_title(f'Outcomes — model {model}', fontsize=9, color=INK)
    return canvas.save(fig, f'outcomes-{slugify(model)}')


def _outcome_counts(rows) -> dict[str, int]:
    """Rows per outcome; a SOLVED/EXHAUSTED row that does not count as solved
    (a plan failed validation, or no plan came back) is INVALID."""
    counts = dict.fromkeys(OUTCOME_ORDER, 0)
    for r in rows:
        status = r['status'] if r['status'] in counts else ERROR
        if status in (SOLVED, EXHAUSTED) and not r['solved']:
            status = INVALID
        counts[status] += 1
    return counts


# -- B-Coverage distribution ---------------------------------------------

@dataclass
class Boxes:
    """The boxes of :func:`_bcoverage_boxes`: one per (k, q) and planner."""

    values: list[list[float]]
    positions: list[float]
    planners: list[str]
    ticks: list[float]              # one tick per (k, q) block
    tick_labels: list[str]


def _bcoverage_boxes(canvas: Canvas, model, model_groups) -> list[str]:
    """Per-instance B-Coverage / k on the commonly solved instances: a box per
    planner within each (k, q), the mean marked in the planner's style."""
    from matplotlib.lines import Line2D

    boxes = _bcoverage_box_data(canvas, model_groups)
    if not boxes.values:
        return []
    fig, ax = canvas.plt.subplots(figsize=(max(4.0, 0.42 * len(boxes.values) + 1.5), 3.0))
    drawn = ax.boxplot(boxes.values, positions=boxes.positions, widths=0.7, patch_artist=True, showfliers=True,
                       medianprops={'color': INK, 'linewidth': 1.4},
                       flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': INK_MUTED,
                                   'markeredgecolor': 'none'})
    for patch, planner in zip(drawn['boxes'], boxes.planners):
        patch.set_facecolor(canvas.style(planner)['color'])
        patch.set_edgecolor(INK_MUTED)
    # The mean, in the planner's colour and marker: keeps a planner visible even
    # when every instance has the same value and its box collapses to a line.
    for values, position, planner in zip(boxes.values, boxes.positions, boxes.planners):
        ax.plot([position], [sum(values) / len(values)], linestyle='none', zorder=5,
                **canvas.marker(planner, markersize=7))
    ax.set_xticks(boxes.ticks)
    ax.set_xticklabels(boxes.tick_labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('B-Coverage / k')
    ax.grid(axis='x', visible=False)
    if canvas.several_planners:
        handles = [Line2D([], [], linestyle='none', **canvas.marker(p, markersize=7)) for p in canvas.planners]
        ax.legend(handles, [f'{p} (mean)' for p in canvas.planners], loc='lower center', bbox_to_anchor=(0.5, 1.0),
                  fontsize=7, ncol=min(4, len(canvas.planners)))
    ax.set_title(f'Per-instance B-Coverage on commonly solved instances — model {model}', fontsize=9,
                 color=INK, pad=22 if canvas.several_planners else 6)
    return canvas.save(fig, f'bcoverage-distribution-{slugify(model)}')


def _bcoverage_box_data(canvas: Canvas, model_groups) -> Boxes:
    boxes = Boxes([], [], [], [], [])
    x = 0.0
    for group in model_groups:
        common = set(common_solved(canvas.rows, group))
        if not common:
            continue
        start = x
        members = canvas.rows_of(group)
        for planner in canvas.planners:
            values = [(r['b_coverage'] or 0) / group[1] for r in members
                      if r['planner'] == planner and r['task_id'] in common]
            if values:
                boxes.values.append(values)
                boxes.positions.append(x)
                boxes.planners.append(planner)
                x += 1
        boxes.ticks.append((start + x - 1) / 2)
        boxes.tick_labels.append(f'k={group[1]}\nq={group[2]}\n(n={len(common)})')
        x += 0.8
    return boxes
