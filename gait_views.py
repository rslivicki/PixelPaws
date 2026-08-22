"""
Gait & Limb graph views  (gait_views.py)
========================================
Headless-of-Tk-state port of the old Gait & Limb "graph window"
(gait_limb_tab._open_graphs and its builders). The new Gait tab drives this
module through:

    host  = ViewHost(summary_df, bins_df, intermediates, cfg, injured_paw,
                     enable_stats, ...)
    reg   = build_registry(host)          # {category: [entry, ...]}
    render_entry(host, frame, entry)      # build one graph into a frame

Widgets are still tkinter/ttk + embedded matplotlib figures - what is banned
here is reading Tk *variables* or the application object; every input comes
through the ViewHost. All metric math, display names, y-labels, descriptions
and tooltip texts are kept verbatim from gait_limb_tab.py.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)
    from scipy import stats as _sp_stats
    _PLOT_OK = True
except ImportError:
    _PLOT_OK = False

try:
    from ui_utils import (ToolTip as _ToolTip,
                          _bind_tight_layout_on_resize, _draw_canvas_fit,
                          FONT_FAMILY)
except Exception:  # pragma: no cover - minimal fallbacks for isolated tests
    FONT_FAMILY = 'Segoe UI'

    class _ToolTip:  # noqa: D401 - no-op stand-in
        def __init__(self, widget, text):
            pass

    def _bind_tight_layout_on_resize(canvas, fig, rect=None):
        pass

    def _draw_canvas_fit(canvas, fig):
        canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level graph helpers (verbatim from gait_limb_tab)
# ─────────────────────────────────────────────────────────────────────────────

_SIG_STYLES = {
    'asterisk': ('*', '**', '***'),
    'hash':     ('#', '##', '###'),
    'dagger':   ('†', '††', '†††'),
    'letters':  ('a', 'b', 'c'),
}


def _p_label(p: float, style: str = 'asterisk') -> str:
    syms = _SIG_STYLES.get(style, _SIG_STYLES['asterisk'])
    if p < 0.001:
        return syms[2]
    if p < 0.01:
        return syms[1]
    if p < 0.05:
        return syms[0]
    return ''


def _draw_bracket(ax, x1: int, x2: int, y: float, label: str):
    ax.plot([x1, x1, x2, x2], [y * 0.97, y, y, y * 0.97],
            color='black', linewidth=1)
    ax.text((x1 + x2) / 2, y * 1.02, label,
            ha='center', va='bottom', fontsize=12, fontweight='bold')


ROLES = ('HL', 'HR', 'FL', 'FR')

_PAW_COLORS_ROLE = {
    'HL': '#2ca02c', 'HR': '#d62728',
    'FL': '#bcbd22', 'FR': '#ff7f0e',
}

_PAW_LABELS = {'HL': 'Hind Left', 'HR': 'Hind Right',
               'FL': 'Fore Left', 'FR': 'Fore Right'}


# ── Pure contour statics ─────────────────────────────────────────────────────
# TODO: dedupe with gait_core - imported from there when available so both
# modules share one implementation; local copies (verbatim from
# gait_limb_tab) are the fallback while gait_core is being extracted.

def _resample_contour(pts, n_points=64):
    """Resample a contour to a fixed number of evenly spaced points.

    pts: (M, 2) array of contour coordinates.
    Returns: (n_points, 2) array.
    """
    if len(pts) < 3:
        return None
    # Close the contour
    closed = np.vstack([pts, pts[0:1]])
    # Cumulative arc length
    diffs = np.diff(closed, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len <= 0:
        return None
    # Evenly spaced parameter values (exclude endpoint to avoid duplicate)
    t_new = np.linspace(0, total_len, n_points, endpoint=False)
    # Interpolate x and y
    x_new = np.interp(t_new, cum_len, closed[:, 0])
    y_new = np.interp(t_new, cum_len, closed[:, 1])
    return np.column_stack([x_new, y_new])


def _normalize_contour(pts, area):
    """Center contour at origin and normalize by sqrt(area)."""
    if pts is None or area <= 0:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    scale = np.sqrt(area)
    if scale > 0:
        centered = centered / scale
    return centered


def _shape_metrics(pts):
    """Compute aspect_ratio and circularity from (N,2) contour points."""
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    dim_max, dim_min = max(w, h), min(w, h)
    ar = dim_max / dim_min if dim_min > 0 else 999.0
    # Perimeter (sum of segment lengths)
    diffs = np.diff(np.vstack([pts, pts[0:1]]), axis=0)
    perimeter = np.sum(np.sqrt((diffs ** 2).sum(axis=1)))
    # Area (shoelace)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    return ar, circ


def _shape_metrics_batch(stacked):
    """Vectorized aspect_ratio + circularity for (N, 64, 2) contour array."""
    mins = stacked.min(axis=1)
    maxs = stacked.max(axis=1)
    wh = maxs - mins
    dim_max = wh.max(axis=1)
    dim_min = wh.min(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ar = np.where(dim_min > 0, dim_max / dim_min, 999.0)
    closed = np.concatenate([stacked, stacked[:, 0:1, :]], axis=1)
    diffs = np.diff(closed, axis=1)
    perimeter = np.sqrt((diffs ** 2).sum(axis=2)).sum(axis=1)
    x, y = stacked[:, :, 0], stacked[:, :, 1]
    area = 0.5 * np.abs(
        (x * np.roll(y, -1, axis=1)).sum(axis=1) -
        (y * np.roll(x, -1, axis=1)).sum(axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        circ = np.where(perimeter > 0,
                        (4 * np.pi * area) / (perimeter ** 2), 0.0)
    return ar, circ


try:  # prefer the single shared implementation once gait_core lands
    from gait_core import (_resample_contour, _normalize_contour,   # noqa: F811
                           _shape_metrics, _shape_metrics_batch)
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ViewHost
# ─────────────────────────────────────────────────────────────────────────────

def _default_pawlike_thresholds():
    # Same defaults as gait_limb_tab.__init__ (:269)
    return {'solidity': 1.00, 'aspect_ratio': 1.6, 'circularity': 0.10}


@dataclass
class ViewHost:
    """Everything the graph views need, decoupled from the Tk application.

    cfg is the graph_cfg dict (contract unchanged from the old settings
    dialog): colors {treatment: hex}, order [treatments], time_window (min or
    None), error_type, error_display, rebin_minutes, opacities/marker_sizes/
    marker_shapes/marker_fills/marker_edge_colors/line_widths/line_styles
    (per-treatment dicts), marker_size, marker_shape, show_individual,
    show_stats, sig_style, graph_sets. Additionally the stats knobs that the
    old code read from Tk variables are read from cfg with the old defaults:
    stats_paradigm ('parametric'), stats_alpha (0.05), stats_test ('auto'),
    timecourse_posthoc (False).
    """
    summary_df: pd.DataFrame
    bins_df: Optional[pd.DataFrame]
    intermediates: dict
    cfg: dict
    injured_paw: str                       # 'HL' | 'HR'
    enable_stats: Callable[[], bool]       # () -> bool, live read
    log: Callable[[str], None] = lambda m: None
    tip: Callable = lambda widget, text: None
    pawlike_thresholds: dict = field(default_factory=_default_pawlike_thresholds)
    on_pawlike_change: Callable[[], None] = lambda: None


def _sig_style(host) -> str:
    return (host.cfg or {}).get('sig_style', 'asterisk')


def _stats_paradigm(host) -> str:
    return (host.cfg or {}).get('stats_paradigm', 'parametric')


def _stats_alpha(host) -> float:
    return float((host.cfg or {}).get('stats_alpha', 0.05))


def _stats_test(host) -> str:
    return (host.cfg or {}).get('stats_test', 'auto')


def _timecourse_posthoc(host) -> bool:
    return bool((host.cfg or {}).get('timecourse_posthoc', False))


# ─────────────────────────────────────────────────────────────────────────────
# Shared computation helpers (ported verbatim; take host where Tk state was)
# ─────────────────────────────────────────────────────────────────────────────

def _treatment_groups(df: pd.DataFrame, metric: str) -> dict:
    """Return {treatment_label: np.array_of_values}."""
    if ('treatment' in df.columns
            and df['treatment'].ne('').any()
            and df['treatment'].notna().any()):
        groups = {}
        for tr, sub in df.groupby('treatment'):
            vals = sub[metric].dropna().values
            if len(vals) > 0:
                groups[str(tr)] = vals
        if groups:
            return groups
    vals = df[metric].dropna().values
    return {'All sessions': vals} if len(vals) > 0 else {}


def _treatment_labels(df: pd.DataFrame, graph_cfg) -> tuple:
    """(has_treats, ordered treatment label list) - the repeated block from
    the grouped/contour builders (sorted fallback, cfg order wins)."""
    has_treats = ('treatment' in df.columns
                  and df['treatment'].ne('').any()
                  and df['treatment'].notna().any())
    if has_treats:
        all_treats = [str(t) for t in df['treatment'].dropna().unique()
                      if str(t).strip()]
        if graph_cfg and graph_cfg.get('order'):
            treatment_labels = [t for t in graph_cfg['order'] if t in all_treats]
            treatment_labels += [t for t in all_treats if t not in treatment_labels]
        else:
            treatment_labels = sorted(all_treats)
    else:
        treatment_labels = ['All sessions']
    return has_treats, treatment_labels


def _calc_error(values, error_type):
    """Compute error bar half-width for a 1-D array of values.

    error_type: 'SEM', 'SD', or '95CI'.
    Returns a single float.
    """
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return 0.0
    sd = np.nanstd(v, ddof=1)
    if error_type == 'SD':
        return sd
    sem = sd / np.sqrt(n)
    if error_type == '95CI':
        from scipy.stats import t as _t_dist
        return _t_dist.ppf(0.975, df=n - 1) * sem
    return sem  # default SEM


def _error_label(error_type):
    """Return a display string for the error type (e.g. 'mean ± SEM')."""
    labels = {'SEM': 'mean ± SEM', 'SD': 'mean ± SD',
              '95CI': 'mean ± 95% CI'}
    return labels.get(error_type, 'mean ± SEM')


def _style_ax(ax, title='', xlabel='', ylabel=''):
    """Apply publication-quality styling to an axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')


def _add_stat_annotation(host, ax, groups: dict, y_top: float):
    """Add significance bracket / ANOVA note."""
    keys = list(groups.keys())
    vals = [groups[k] for k in keys]
    if len(vals) < 2:
        return
    paradigm = _stats_paradigm(host)
    use_nonparam = paradigm == 'nonparametric'
    if paradigm == 'auto':
        for v in vals:
            if len(v) >= 3:
                try:
                    _, sw_p = _sp_stats.shapiro(v)
                    if sw_p < 0.05:
                        use_nonparam = True
                        break
                except Exception as _sw_err:
                    print(f"Warning: Shapiro-Wilk test failed: {_sw_err}")
    try:
        if len(vals) == 2:
            if use_nonparam:
                _, p = _sp_stats.mannwhitneyu(vals[0], vals[1], alternative='two-sided')
            else:
                _, p = _sp_stats.ttest_ind(vals[0], vals[1], equal_var=False)
            label = _p_label(p, _sig_style(host))
            if label:
                _draw_bracket(ax, 0, 1, y_top * 1.05, label)
        else:
            if use_nonparam:
                _, p = _sp_stats.kruskal(*vals)
                test_name = 'Kruskal-Wallis'
            else:
                _, p = _sp_stats.f_oneway(*vals)
                test_name = 'ANOVA'
            label = _p_label(p, _sig_style(host))
            if label:
                ax.text(0.5, 0.97,
                        f"{test_name}: {label}  (p = {p:.3f})",
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=11, color='darkred')
    except Exception as _stats_err:
        print(f"Warning: statistical test failed in limb use graph: {_stats_err}")


def _embed_figure(frame, fig, ax=None):
    """Embed a figure in a frame with toolbar + the axis-limit override row."""
    canvas = FigureCanvasTkAgg(fig, master=frame)

    # Pack bottom widgets first (side='bottom') so canvas can expand
    if ax is not None:
        axis_row = ttk.Frame(frame)
        axis_row.pack(side='bottom', fill='x', padx=4, pady=(0, 2))
        ttk.Label(axis_row, text="Y:", font=(FONT_FAMILY, 9)).pack(side='left')
        y_min_e = ttk.Entry(axis_row, width=6)
        y_min_e.pack(side='left', padx=2)
        ttk.Label(axis_row, text="to", font=(FONT_FAMILY, 9)).pack(side='left')
        y_max_e = ttk.Entry(axis_row, width=6)
        y_max_e.pack(side='left', padx=2)
        ttk.Label(axis_row, text="   X:", font=(FONT_FAMILY, 9)).pack(side='left')
        x_min_e = ttk.Entry(axis_row, width=6)
        x_min_e.pack(side='left', padx=2)
        ttk.Label(axis_row, text="to", font=(FONT_FAMILY, 9)).pack(side='left')
        x_max_e = ttk.Entry(axis_row, width=6)
        x_max_e.pack(side='left', padx=2)

        # Store original limits for reset
        _orig_ylim = ax.get_ylim()
        _orig_xlim = ax.get_xlim()

        def _apply_range():
            try:
                yl = list(ax.get_ylim())
                if y_min_e.get().strip():
                    yl[0] = float(y_min_e.get())
                if y_max_e.get().strip():
                    yl[1] = float(y_max_e.get())
                ax.set_ylim(*yl)
            except ValueError:
                pass
            try:
                xl = list(ax.get_xlim())
                if x_min_e.get().strip():
                    xl[0] = float(x_min_e.get())
                if x_max_e.get().strip():
                    xl[1] = float(x_max_e.get())
                ax.set_xlim(*xl)
            except ValueError:
                pass
            canvas.draw_idle()

        def _reset_range():
            y_min_e.delete(0, 'end')
            y_max_e.delete(0, 'end')
            x_min_e.delete(0, 'end')
            x_max_e.delete(0, 'end')
            ax.set_ylim(*_orig_ylim)
            ax.set_xlim(*_orig_xlim)
            canvas.draw_idle()

        ttk.Button(axis_row, text="Apply", command=_apply_range,
                   width=6).pack(side='left', padx=(6, 2))
        ttk.Button(axis_row, text="Reset", command=_reset_range,
                   width=6).pack(side='left', padx=2)

    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    cw = canvas.get_tk_widget()
    # Tag the widget so _close_figures_recursive can actually find the
    # figure (the old code's walk relied on this attribute but the Tk
    # backend never set it, silently leaking figures).
    cw.figure = fig
    cw.pack(fill='both', expand=True)
    _bind_tight_layout_on_resize(canvas, fig)
    _draw_canvas_fit(canvas, fig)


def _close_figures_recursive(widget):
    """Close any matplotlib figures attached to widget or its children
    (ported from _open_graphs' close handler)."""
    fig_obj = getattr(widget, 'figure', None)
    if fig_obj is not None:
        try:
            plt.close(fig_obj)
        except Exception:
            pass
    for child in getattr(widget, 'winfo_children', lambda: [])():
        _close_figures_recursive(child)


def _rebin_timecourse(xs, means, errs, rebin_min):
    """Aggregate timecourse data into larger bins."""
    if not xs or rebin_min <= 0:
        return xs, means, errs
    new_xs, new_means, new_errs = [], [], []
    i = 0
    while i < len(xs):
        bin_start = xs[i]
        bin_end = bin_start + rebin_min
        group_m, group_e = [], []
        while i < len(xs) and xs[i] < bin_end:
            group_m.append(means[i])
            group_e.append(errs[i])
            i += 1
        if group_m:
            new_xs.append(bin_start)
            new_means.append(float(np.mean(group_m)))
            # Propagate error: average of errors (simple approach)
            new_errs.append(float(np.mean(group_e)))
    return new_xs, new_means, new_errs


# ─────────────────────────────────────────────────────────────────────────────
# Injured-paw display flip (single, NON-mutating implementation)
# ─────────────────────────────────────────────────────────────────────────────

def injured_contra(host) -> tuple:
    """('HL'|'HR' injured, contralateral) from host, defaulting to HL."""
    inj = (host.injured_paw or 'HL').upper()
    if inj not in ('HL', 'HR'):
        inj = 'HL'
    return inj, ('HR' if inj == 'HL' else 'HL')


def injured_display_frame(host, src: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return a frame whose *col* reads injured/contralateral.

    Stored ratio columns are literal HL/HR; when the injured paw is HR the
    displayed value is inverted (1/x) so every ratio graph reads
    injured/contralateral. Unlike the old _hl_col/_disp_col helpers this
    NEVER mutates *src* and never introduces extra ``*_injflip`` columns -
    it returns *src* untouched for HL, or a copy with *col* replaced for HR
    (so CSV exports carry the displayed value under the original name).
    """
    inj, _ = injured_contra(host)
    if src is None or col not in src.columns or inj == 'HL':
        return src
    out = src.copy()
    v = pd.to_numeric(out[col], errors='coerce')
    out[col] = np.where(v > 0, 1.0 / v, np.nan)
    return out


def _entry_data(host, entry):
    """Resolve (df, column) for an entry, applying the display-only injured
    flip on a copy when the entry is a hind-ratio metric."""
    df = entry.get('data')
    col = entry.get('column')
    if df is None or col is None:
        return df, col
    if entry.get('flip'):
        df = injured_display_frame(host, df, col)
    return df, col

# ─────────────────────────────────────────────────────────────────────────────
# Figure builders (build into an existing frame)
# ─────────────────────────────────────────────────────────────────────────────

def build_bar_graph(host, frame, df, metric, tab_name,
                    reference=None, y_label=''):
    """Create a bar chart figure and embed it in the given frame."""
    graph_cfg = host.cfg
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    groups = _treatment_groups(df, metric)
    if not groups:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
    else:
        if graph_cfg and graph_cfg.get('order'):
            ordered_keys = [t for t in graph_cfg['order'] if t in groups]
            ordered_keys += [t for t in groups if t not in ordered_keys]
        else:
            ordered_keys = list(groups.keys())
        treatments = ordered_keys
        vals_list = [groups[t] for t in treatments]
        _err_type = graph_cfg.get('error_type', 'SEM') if graph_cfg else 'SEM'
        means = [np.nanmean(v) for v in vals_list]
        errs = [_calc_error(v, _err_type) for v in vals_list]
        x_pos = np.arange(len(treatments))
        rng = np.random.default_rng(42)
        for xi, (t, vals) in enumerate(zip(treatments, vals_list)):
            if graph_cfg and graph_cfg.get('colors') and t in graph_cfg['colors']:
                raw = graph_cfg['colors'][t]
            else:
                raw = 'steelblue'
            if raw == 'white_black':
                bar_color, edge_color, txt_color = 'white', 'black', 'black'
            else:
                bar_color, edge_color, txt_color = raw, 'black', 'black'
            ax.bar(xi, means[xi], yerr=errs[xi], capsize=4,
                   color=bar_color, alpha=0.85, edgecolor=edge_color,
                   linewidth=0.9)
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(xi + jitter, vals, color=txt_color, s=30,
                       zorder=5, alpha=0.8)
        if reference is not None:
            ax.axhline(reference, color='crimson', linestyle='--',
                       linewidth=1.2, alpha=0.7,
                       label=f'Reference = {reference}')
            ax.legend(fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(treatments, fontsize=11)
        _style_ax(ax, title=tab_name, ylabel=y_label or metric)
        y_top = max(means) if means else 0
        if host.enable_stats():
            _add_stat_annotation(
                host, ax, {t: groups[t] for t in treatments}, y_top)
    _add_export_buttons(host, frame, fig, groups, metric, tab_name,
                        data_type='summary')
    _embed_figure(frame, fig, ax=ax)


def build_box_graph(host, frame, df, metric, tab_name,
                    reference=None, y_label=''):
    """Create a box plot figure and embed it in the given frame."""
    graph_cfg = host.cfg
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    groups = _treatment_groups(df, metric)
    if not groups:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
    else:
        if graph_cfg and graph_cfg.get('order'):
            ordered_keys = [t for t in graph_cfg['order'] if t in groups]
            ordered_keys += [t for t in groups if t not in ordered_keys]
        else:
            ordered_keys = list(groups.keys())
        treatments = ordered_keys
        data = [groups[t] for t in treatments]
        bp_dict = ax.boxplot(data, labels=treatments, patch_artist=True,
                             medianprops=dict(color='black', linewidth=2))
        fallback_colors = plt.cm.Set2.colors
        for i, (patch, t) in enumerate(zip(bp_dict['boxes'], treatments)):
            if (graph_cfg and graph_cfg.get('colors')
                    and t in graph_cfg['colors']):
                raw = graph_cfg['colors'][t]
                fc = 'white' if raw == 'white_black' else raw
                patch.set_facecolor(fc)
            else:
                patch.set_facecolor(
                    fallback_colors[i % len(fallback_colors)])
            patch.set_alpha(0.7)
        rng = np.random.default_rng(42)
        for xi, vals in enumerate(data, 1):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(xi + jitter, vals, color='black', s=28,
                       zorder=5, alpha=0.8)
        if reference is not None:
            ax.axhline(reference, color='crimson', linestyle='--',
                       linewidth=1.2, alpha=0.7)
        _style_ax(ax, title=tab_name, ylabel=y_label or metric)
        y_top = max(np.nanmax(v) for v in data if len(v) > 0)
        if host.enable_stats():
            _add_stat_annotation(host, ax, groups, y_top)
    _add_export_buttons(host, frame, fig, groups, metric, tab_name,
                        data_type='summary')
    _embed_figure(frame, fig, ax=ax)


def build_violin_graph(host, frame, df, metric, tab_name,
                       reference=None, y_label=''):
    """Create a violin plot figure and embed it in the given frame."""
    graph_cfg = host.cfg
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    groups = _treatment_groups(df, metric)
    if not groups:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
    else:
        if graph_cfg and graph_cfg.get('order'):
            ordered_keys = [t for t in graph_cfg['order'] if t in groups]
            ordered_keys += [t for t in groups if t not in ordered_keys]
        else:
            ordered_keys = list(groups.keys())
        treatments = ordered_keys
        data = [groups[t] for t in treatments]
        parts = ax.violinplot(data, positions=range(len(treatments)),
                              showmeans=True, showmedians=True,
                              showextrema=False)
        for i, (pc, t) in enumerate(zip(parts['bodies'], treatments)):
            if (graph_cfg and graph_cfg.get('colors')
                    and t in graph_cfg['colors']):
                raw = graph_cfg['colors'][t]
                fc = 'white' if raw == 'white_black' else raw
            else:
                fc = plt.cm.Set2.colors[i % len(plt.cm.Set2.colors)]
            pc.set_facecolor(fc)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        rng = np.random.default_rng(42)
        for xi, vals in enumerate(data):
            jitter = rng.uniform(-0.1, 0.1, len(vals))
            ax.scatter(xi + jitter, vals, color='black', s=28,
                       zorder=5, alpha=0.8)
        if reference is not None:
            ax.axhline(reference, color='crimson', linestyle='--',
                       linewidth=1.2, alpha=0.7)
        ax.set_xticks(range(len(treatments)))
        ax.set_xticklabels(treatments, fontsize=11)
        _style_ax(ax, title=tab_name, ylabel=y_label or metric)
        y_top = max(np.nanmax(v) for v in data if len(v) > 0)
        if host.enable_stats():
            _add_stat_annotation(
                host, ax, {t: groups[t] for t in treatments}, y_top)
    _add_export_buttons(host, frame, fig, groups, metric, tab_name,
                        data_type='summary')
    _embed_figure(frame, fig, ax=ax)


def build_timecourse_graph(host, frame, bins_df, metric, tab_name,
                           reference=None, y_label=None):
    """Create a timecourse figure and embed it in the given frame."""
    graph_cfg = host.cfg
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    if reference is None:
        if 'SI' in metric or 'symmetry' in metric.lower():
            reference = 0.0
        elif 'WBI' in metric or 'SBI' in metric:
            reference = 50.0
    if reference is not None:
        ax.axhline(reference, color='crimson', linestyle='--',
                   linewidth=1.1, alpha=0.6)

    _err_type = graph_cfg.get('error_type', 'SEM') if graph_cfg else 'SEM'

    # Filter bins to the user-specified time window so lines don't
    # extend beyond the requested range.
    if graph_cfg and graph_cfg.get('time_window') is not None:
        _tw_sec = graph_cfg['time_window'] * 60.0
        bins_df = bins_df[bins_df['bin_start_s'] <= _tw_sec]

    if ('treatment' in bins_df.columns
            and bins_df['treatment'].ne('').any()):
        all_treats = bins_df['treatment'].dropna().unique().tolist()
        if graph_cfg and graph_cfg.get('order'):
            ordered = [t for t in graph_cfg['order'] if t in all_treats]
            ordered += [t for t in all_treats if t not in ordered]
        else:
            ordered = all_treats

        for treatment in ordered:
            grp = bins_df[bins_df['treatment'] == treatment]
            if grp.empty:
                continue
            tg = grp.groupby('bin_start_s')[metric]
            tmean = tg.mean()
            if _err_type == 'SD':
                terr = tg.std(ddof=1).fillna(0)
            elif _err_type == '95CI':
                terr = tg.apply(
                    lambda x: _calc_error(x.values, '95CI')).fillna(0)
            else:
                terr = tg.sem().fillna(0)
            t_min = tmean.index.values / 60.0

            raw_color = None
            if graph_cfg and graph_cfg.get('colors'):
                raw_color = graph_cfg['colors'].get(str(treatment))
            line_color = ('black' if raw_color == 'white_black'
                          else (raw_color if raw_color else None))
            _tkey = str(treatment)
            _lstyle = (graph_cfg.get('line_styles', {}).get(_tkey, '-')
                       if graph_cfg else '-')
            _mshape = (graph_cfg.get('marker_shapes', {}).get(_tkey, 'o')
                       if graph_cfg else 'o')
            _msize = (graph_cfg.get('marker_sizes', {}).get(_tkey, 5)
                      if graph_cfg else 5)
            _lw = (graph_cfg.get('line_widths', {}).get(_tkey, 1.8)
                   if graph_cfg else 1.8)
            _alpha = (graph_cfg.get('opacities', {}).get(_tkey, 1.0)
                      if graph_cfg else 1.0)
            if not _mshape or _mshape == 'none':
                _mshape_g = (graph_cfg.get('marker_shape', 'none')
                             if graph_cfg else 'none')
                if _mshape_g and _mshape_g != 'none':
                    _mshape = _mshape_g
                    _msize = (graph_cfg.get('marker_size', 5)
                              if graph_cfg else 5)
            _mfill = (graph_cfg.get('marker_fills', {}).get(_tkey, 'full')
                      if graph_cfg else 'full')
            plot_kw = dict(label=_tkey, linewidth=_lw,
                           linestyle=_lstyle, alpha=_alpha)
            if _mshape and _mshape != 'none':
                plot_kw['marker'] = _mshape
                plot_kw['markersize'] = _msize
                plot_kw['fillstyle'] = _mfill
                if _mfill == 'none':
                    plot_kw['markeredgewidth'] = 1.2
                elif _mfill in ('left', 'right', 'top', 'bottom'):
                    plot_kw['markerfacecoloralt'] = 'white'
            if line_color:
                plot_kw['color'] = line_color
            if raw_color == 'white_black':
                plot_kw['markerfacecolor'] = 'white'
                plot_kw['markeredgecolor'] = 'black'
            _edge_sel = graph_cfg.get('marker_edge_colors', {}).get(_tkey, 'auto') if graph_cfg else 'auto'
            if _edge_sel == 'black':
                plot_kw['markeredgecolor'] = 'black'
            elif _edge_sel == 'white':
                plot_kw['markeredgecolor'] = 'white'
            elif _edge_sel == 'match':
                plot_kw['markeredgecolor'] = line_color or plot_kw.get('color', 'black')
            rebin = graph_cfg.get('rebin_minutes') if graph_cfg else None
            if rebin and rebin > 0:
                t_min, _means, _errs = _rebin_timecourse(
                    list(t_min), list(tmean.values),
                    list(terr.values), rebin)
                t_min = np.array(t_min)
                tmean_v = np.array(_means)
                terr_v = np.array(_errs)
            else:
                tmean_v = tmean.values
                terr_v = terr.values
            _err_disp = (graph_cfg.get('error_display', 'circles_caps')
                         if graph_cfg else 'circles_caps')
            # When circles_caps, draw line without markers (markers drawn with errorbar)
            if _err_disp == 'circles_caps':
                _line_kw = {k: v for k, v in plot_kw.items()
                            if k not in ('marker', 'markersize', 'fillstyle',
                                         'markeredgewidth', 'markerfacecoloralt',
                                         'markerfacecolor', 'markeredgecolor')}
                ax.plot(t_min, tmean_v, **_line_kw)
            else:
                ax.plot(t_min, tmean_v, **plot_kw)
            _ec = line_color or plot_kw.get('color')
            if _err_disp == 'caps':
                ax.errorbar(t_min, tmean_v, yerr=terr_v, capsize=4,
                            capthick=1.5, fmt='none', alpha=0.6,
                            ecolor=_ec)
            elif _err_disp == 'circles_caps':
                ax.errorbar(t_min, tmean_v, yerr=terr_v, capsize=4,
                            capthick=1.5, fmt='none', alpha=0.6 * _alpha,
                            ecolor=_ec)
                _msz = _msize if (_mshape and _mshape != 'none') else 5
                _mkw = dict(linestyle='none', color=_ec,
                            marker='o', markersize=_msz, alpha=_alpha)
                _mkw['fillstyle'] = _mfill
                if _mfill == 'none':
                    _mkw['markeredgewidth'] = 1.2
                elif _mfill in ('left', 'right', 'top', 'bottom'):
                    _mkw['markerfacecoloralt'] = 'white'
                if raw_color == 'white_black':
                    _mkw['markerfacecolor'] = 'white'
                    _mkw['markeredgecolor'] = 'black'
                if _edge_sel == 'black':
                    _mkw['markeredgecolor'] = 'black'
                elif _edge_sel == 'white':
                    _mkw['markeredgecolor'] = 'white'
                elif _edge_sel == 'match':
                    _mkw['markeredgecolor'] = _ec or 'black'
                ax.plot(t_min, tmean_v, **_mkw)
            else:
                ax.fill_between(
                    t_min, tmean_v - terr_v, tmean_v + terr_v,
                    alpha=0.18,
                    **(dict(color=line_color) if line_color else {}))
            if graph_cfg and graph_cfg.get('show_individual'):
                for subj in grp['subject'].unique():
                    subj_data = grp[grp['subject'] == subj]
                    sg = subj_data.groupby('bin_start_s')[metric].mean()
                    s_min = sg.index.values / 60.0
                    ax.plot(s_min, sg.values, alpha=0.3, linewidth=0.8,
                            **(dict(color=line_color)
                               if line_color else {}))
        ax.legend(fontsize=10)

        if host.enable_stats() and len(ordered) >= 2:
            from matplotlib.transforms import blended_transform_factory
            import warnings as _warnings
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            tw = graph_cfg.get('time_window') if graph_cfg else None
            alpha = _stats_alpha(host)

            # Step 1: Two-way ANOVA gatekeeper
            _do_posthoc = False
            _anova_text = ''
            try:
                import statsmodels.api as _sm
                from statsmodels.formula.api import ols as _ols
                _adf = bins_df[['subject', 'treatment', 'bin_start_s', metric]].dropna().copy()
                _adf['treatment'] = _adf['treatment'].astype('category')
                _adf['bin_start_s'] = _adf['bin_start_s'].astype('category')
                with _warnings.catch_warnings():
                    _warnings.filterwarnings('ignore', module='statsmodels')
                    _model = _ols(f'{metric} ~ C(treatment) + C(bin_start_s) + C(treatment):C(bin_start_s)',
                                  data=_adf).fit()
                    _at = _sm.stats.anova_lm(_model, typ=2)
                _treat_p = _at.loc['C(treatment)', 'PR(>F)']
                _inter_p = _at.loc['C(treatment):C(bin_start_s)', 'PR(>F)']
                if _inter_p < alpha or _treat_p < alpha:
                    _do_posthoc = True
                def _pfmt(p):
                    if p < 0.001: return 'p<0.001***'
                    if p < 0.01:  return f'p={p:.3f}**'
                    if p < alpha: return f'p={p:.3f}*'
                    return f'p={p:.3f} ns'
                _anova_text = f"Two-way ANOVA: Treatment {_pfmt(_treat_p)}, Interaction {_pfmt(_inter_p)}"
            except Exception:
                _do_posthoc = True

            # Step 2: Per-bin post-hoc (only if ANOVA is significant)
            if _do_posthoc:
                _rebin_min = graph_cfg.get('rebin_minutes') if graph_cfg else None
                if _rebin_min and _rebin_min > 0:
                    _rebin_s = _rebin_min * 60.0
                    _all_bins = sorted(bins_df['bin_start_s'].unique())
                    _rebin_windows = {}
                    for bs in _all_bins:
                        win_start = int(bs // _rebin_s) * _rebin_s
                        _rebin_windows.setdefault(win_start, []).append(bs)
                    _bin_iter = [
                        (win_start / 60.0,
                         bins_df[bins_df['bin_start_s'].isin(win_bins)])
                        for win_start, win_bins in sorted(_rebin_windows.items())
                    ]
                else:
                    _bin_iter = [
                        (bin_t / 60.0, bgrp)
                        for bin_t, bgrp in bins_df.groupby('bin_start_s')
                    ]

                paradigm = _stats_paradigm(host)
                for t_min_val, bin_data in _bin_iter:
                    if tw is not None and t_min_val > tw:
                        continue
                    gvals = []
                    for t in ordered:
                        if _rebin_min and _rebin_min > 0:
                            t_data = bin_data[bin_data['treatment'] == t]
                            vals = t_data.groupby('subject')[metric].mean().dropna().values
                        else:
                            vals = bin_data[bin_data['treatment'] == t][metric].dropna().values
                        if len(vals) >= 2:
                            gvals.append(vals)
                    if len(gvals) < 2:
                        continue

                    use_nonparam = paradigm == 'nonparametric'
                    if paradigm == 'auto':
                        for g in gvals:
                            if len(g) >= 3:
                                try:
                                    _, sw_p = _sp_stats.shapiro(g)
                                    if sw_p < 0.05:
                                        use_nonparam = True
                                        break
                                except Exception:
                                    pass

                    try:
                        if len(gvals) == 2:
                            if use_nonparam:
                                _, p = _sp_stats.mannwhitneyu(gvals[0], gvals[1], alternative='two-sided')
                            else:
                                _, p = _sp_stats.ttest_ind(gvals[0], gvals[1], equal_var=False)
                        else:
                            if use_nonparam:
                                _, p = _sp_stats.kruskal(*gvals)
                            else:
                                _, p = _sp_stats.f_oneway(*gvals)
                        lbl = _p_label(p, _sig_style(host))
                        if lbl:
                            ax.text(t_min_val, 0.98, lbl, transform=trans,
                                    ha='center', va='top', fontsize=9, color='black')
                    except Exception:
                        pass

            # Show ANOVA annotation
            if _anova_text:
                ax.text(0.02, 0.98, _anova_text, transform=ax.transAxes,
                        va='top', ha='left', fontsize=7, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    else:
        tg = bins_df.groupby('bin_start_s')[metric]
        tmean = tg.mean()
        t_min = tmean.index.values / 60.0
        rebin = graph_cfg.get('rebin_minutes') if graph_cfg else None
        if rebin and rebin > 0:
            t_min, _means, _ = _rebin_timecourse(
                list(t_min), list(tmean.values),
                [0] * len(t_min), rebin)
            t_min = np.array(t_min)
            tmean_v = np.array(_means)
        else:
            tmean_v = tmean.values
        _st_plot_kw = dict(linewidth=1.8, color='steelblue')
        _ls_d = graph_cfg.get('line_styles', {}) if graph_cfg else {}
        if _ls_d:
            _ft = next(iter(_ls_d))
            _st_plot_kw['linestyle'] = _ls_d.get(_ft, '-')
            _st_plot_kw['linewidth'] = (
                graph_cfg.get('line_widths', {}).get(_ft, 1.8))
            _st_plot_kw['alpha'] = (
                graph_cfg.get('opacities', {}).get(_ft, 1.0))
            _mk = graph_cfg.get('marker_shapes', {}).get(_ft, 'o')
            if _mk and _mk != 'none':
                _st_plot_kw['marker'] = _mk
                _st_plot_kw['markersize'] = (
                    graph_cfg.get('marker_sizes', {}).get(_ft, 5))
                _mf = graph_cfg.get('marker_fills', {}).get(_ft, 'full')
                _st_plot_kw['fillstyle'] = _mf
                if _mf == 'none':
                    _st_plot_kw['markeredgewidth'] = 1.2
                elif _mf in ('left', 'right', 'top', 'bottom'):
                    _st_plot_kw['markerfacecoloralt'] = 'white'
            _me = graph_cfg.get('marker_edge_colors', {}).get(_ft, 'auto') if graph_cfg else 'auto'
            if _me == 'black':
                _st_plot_kw['markeredgecolor'] = 'black'
            elif _me == 'white':
                _st_plot_kw['markeredgecolor'] = 'white'
            elif _me == 'match':
                _st_plot_kw['markeredgecolor'] = _st_plot_kw.get('color', 'steelblue')
        else:
            _mshape = (graph_cfg.get('marker_shape', 'o')
                       if graph_cfg else 'o')
            if _mshape and _mshape != 'none':
                _st_plot_kw['marker'] = _mshape
                _st_plot_kw['markersize'] = (
                    graph_cfg.get('marker_size', 5))
        ax.plot(t_min, tmean_v, **_st_plot_kw)

    if graph_cfg and graph_cfg.get('time_window') is not None:
        ax.set_xlim(-0.5, graph_cfg['time_window'] + 0.5)
    else:
        ax.margins(x=0.02)

    _style_ax(ax, title=tab_name,
              xlabel='Time (min)',
              ylabel=y_label or metric.replace('_', ' '))
    # Annotate error type on the plot
    ax.annotate(_error_label(_err_type),
                xy=(1, 1), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=8,
                color='gray', style='italic')

    _add_export_buttons(host, frame, fig, None, metric, tab_name,
                        data_type='timecourse', bins_df=bins_df)
    _embed_figure(frame, fig, ax=ax)


# ─────────────────────────────────────────────────────────────────────────────
# Export (per-graph CSV / PNG) - headless core + button bar
# ─────────────────────────────────────────────────────────────────────────────

def _export_dataframe(groups, metric, graph_cfg, data_type='summary',
                      bins_df=None):
    """Build the DataFrame the old 'Export Data' button wrote (or None)."""
    if data_type == 'timecourse' and bins_df is not None:
        cols = [c for c in ['treatment', 'bin_start_s', metric]
                if c in bins_df.columns]
        out = bins_df[cols].copy()
        if 'bin_start_s' in out.columns:
            out.insert(
                out.columns.get_loc('bin_start_s') + 1,
                'bin_start_min', out['bin_start_s'] / 60.0)
        return out
    if groups:
        gc = graph_cfg or {}
        ordered_keys = (
            [t for t in gc.get('order', []) if t in groups]
            + [t for t in groups if t not in gc.get('order', [])])
        rows = [{'treatment': t, metric: float(v)}
                for t in (ordered_keys or list(groups.keys()))
                for v in groups[t]]
        return pd.DataFrame(rows)
    return None


def export_entry_csv(host, entry, path):
    """Headless CSV export for a registry entry (same rows as the old
    per-graph 'Export Data' button). Returns the path, or None when the
    entry has no tabular data behind it. Never mutates host frames."""
    df, col = _entry_data(host, entry)
    if df is None or col is None:
        return None
    if entry.get('graph_type') == 'timecourse':
        out = _export_dataframe(None, col, host.cfg,
                                data_type='timecourse', bins_df=df)
    else:
        out = _export_dataframe(_treatment_groups(df, col), col, host.cfg,
                                data_type='summary')
    if out is None:
        return None
    out.to_csv(path, index=False)
    return path


def _add_export_buttons(host, frame, fig, groups, metric, tab_name,
                        data_type='summary', bins_df=None):
    """Add Export Graph / Export Data buttons to a frame."""
    graph_cfg = host.cfg
    btn_bar = ttk.Frame(frame)
    btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

    def _exp_graph(f=fig, n=tab_name):
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'),
                       ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
            initialfile=n.replace(' ', '_') + '.png',
            parent=frame.winfo_toplevel())
        if path:
            f.savefig(path, dpi=300, bbox_inches='tight')

    def _exp_data():
        out = _export_dataframe(groups, metric, graph_cfg,
                                data_type=data_type, bins_df=bins_df)
        if out is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv', filetypes=[('CSV', '*.csv')],
            initialfile=tab_name.replace(' ', '_') + '_data.csv',
            parent=frame.winfo_toplevel())
        if path:
            out.to_csv(path, index=False)

    ttk.Button(btn_bar, text="Export Graph",
               command=_exp_graph).pack(side='left', padx=4)
    ttk.Button(btn_bar, text="Export Data",
               command=_exp_data).pack(side='left', padx=2)

# ─────────────────────────────────────────────────────────────────────────────
# Custom builders: Contact % grouped bar, Support patterns
# ─────────────────────────────────────────────────────────────────────────────

def add_paw_contact_bar_tab(host, frame, df):
    """Grouped bar chart: per-paw contact%, one bar group per treatment."""
    graph_cfg = host.cfg
    _PAW_COLORS = {
        'contact_pct_HL': '#2ca02c',
        'contact_pct_HR': '#d62728',
        'contact_pct_FL': '#bcbd22',
        'contact_pct_FR': '#ff7f0e',
    }
    paw_roles = ['HL', 'HR', 'FL', 'FR']
    paw_cols = [f'contact_pct_{r}' for r in paw_roles]
    active = [(r, c) for r, c in zip(paw_roles, paw_cols)
              if c in df.columns and df[c].notna().any()]
    if not active:
        ttk.Label(frame, text='No per-paw contact data available.',
                  font=(FONT_FAMILY, 10, 'italic')).pack(padx=10, pady=10)
        return

    has_treats, treatment_labels = _treatment_labels(df, graph_cfg)
    n_treats = len(treatment_labels)

    _err_type = graph_cfg.get('error_type', 'SEM') if graph_cfg else 'SEM'

    if n_treats > 1:
        colors = list(plt.cm.Set2(np.linspace(0, 0.8, n_treats)))
    else:
        colors = [_PAW_COLORS.get(active[0][1], '#1f77b4')]

    fig, ax = plt.subplots(
        figsize=(max(5, len(active) * 1.4 + 1), 4), constrained_layout=True)
    x = np.arange(len(active))
    bar_w = 0.7 / n_treats
    rng = np.random.default_rng(42)

    for ti, treat in enumerate(treatment_labels):
        xpos = x - 0.35 + (ti + 0.5) * bar_w
        for pi, (role, col) in enumerate(active):
            if has_treats and n_treats > 1:
                subset = df[df['treatment'] == treat][col].dropna()
            else:
                subset = df[col].dropna()
            mean_ = subset.mean() if len(subset) else 0
            err_ = _calc_error(subset.values, _err_type)
            # Color: from graph_cfg if available, else fallback
            if graph_cfg and graph_cfg.get('colors') and treat in graph_cfg['colors']:
                raw = graph_cfg['colors'][treat]
                color = 'white' if raw == 'white_black' else raw
                edge = 'black' if raw == 'white_black' else 'black'
            elif n_treats > 1:
                color = colors[ti]
                edge = 'black'
            else:
                color = _PAW_COLORS.get(col, colors[0])
                edge = 'black'
            ax.bar(xpos[pi], mean_, bar_w * 0.9,
                   color=color, edgecolor=edge, yerr=err_, capsize=4,
                   label=treat if pi == 0 else None)
            jx = xpos[pi] + rng.uniform(-bar_w * 0.3, bar_w * 0.3, len(subset))
            ax.scatter(jx, subset.values, color=color, s=22, alpha=0.6,
                       zorder=3, edgecolors='black', linewidths=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([r for r, _ in active], fontsize=11)
    _style_ax(ax, title='Per-paw contact', ylabel='Contact %')
    if n_treats > 1:
        ax.legend(fontsize=10)

    # Per-paw significance annotation
    if host.enable_stats():
        if n_treats == 2:
            for pi, (role, col) in enumerate(active):
                v0 = (df[df['treatment'] == treatment_labels[0]][col].dropna().values
                      if has_treats else df[col].dropna().values)
                v1 = (df[df['treatment'] == treatment_labels[1]][col].dropna().values
                      if has_treats else np.array([]))
                if len(v0) > 0 and len(v1) > 0:
                    try:
                        _, p = _sp_stats.ttest_ind(v0, v1, equal_var=False)
                        lbl = _p_label(p, _sig_style(host))
                        if lbl:
                            x0 = x[pi] - 0.35 + 0.5 * bar_w
                            x1 = x[pi] - 0.35 + 1.5 * bar_w
                            y_ann = ax.get_ylim()[1] * 0.96
                            ax.plot([x0, x0, x1, x1],
                                    [y_ann * 0.97, y_ann, y_ann, y_ann * 0.97],
                                    color='black', linewidth=1)
                            ax.text((x0 + x1) / 2, y_ann * 1.01, lbl,
                                    ha='center', va='bottom',
                                    fontsize=11, fontweight='bold')
                    except Exception as _stats_err:
                        print(f"Warning: pairwise stats failed: {_stats_err}")
        elif n_treats > 2:
            for pi, (role, col) in enumerate(active):
                grp_vals = [df[df['treatment'] == t][col].dropna().values
                            for t in treatment_labels]
                grp_vals = [v for v in grp_vals if len(v) > 0]
                if len(grp_vals) > 1:
                    try:
                        _, p = _sp_stats.f_oneway(*grp_vals)
                        lbl = _p_label(p, _sig_style(host))
                        if lbl:
                            ax.text(x[pi], ax.get_ylim()[1] * 0.96, lbl,
                                    ha='center', va='bottom',
                                    fontsize=11, fontweight='bold')
                    except Exception as _stats_err:
                        print(f"Warning: ANOVA stats failed: {_stats_err}")

    # -- export button bar --
    btn_bar = ttk.Frame(frame)
    btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

    def _exp_graph(f=fig):
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
            initialfile='Contact_pct.png',
            parent=frame.winfo_toplevel())
        if path:
            f.savefig(path, dpi=300, bbox_inches='tight')

    def _exp_data(d=df, act=active, ht=has_treats):
        paw_cols_active = [col for _, col in act]
        export_cols = (['treatment'] + paw_cols_active if ht else paw_cols_active)
        out = d[[c for c in export_cols if c in d.columns]].copy()
        path = filedialog.asksaveasfilename(
            defaultextension='.csv', filetypes=[('CSV', '*.csv')],
            initialfile='Contact_pct_data.csv',
            parent=frame.winfo_toplevel())
        if path:
            out.to_csv(path, index=False)

    ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)
    ttk.Button(btn_bar, text="Export Data", command=_exp_data).pack(side='left', padx=2)
    _embed_figure(frame, fig, ax=ax)


def add_support_pattern_tab(host, frame, df):
    """Stacked bar chart: support pattern distribution per treatment."""
    graph_cfg = host.cfg
    support_cols = [f'support_{n}paw_pct' for n in range(5)]
    active = [c for c in support_cols if c in df.columns and df[c].notna().any()]
    if not active:
        ttk.Label(frame, text='No support-pattern data available.',
                  font=(FONT_FAMILY, 10, 'italic')).pack(padx=10, pady=10)
        return

    has_treats, treatment_labels = _treatment_labels(df, graph_cfg)

    fig, ax = plt.subplots(figsize=(max(5, len(treatment_labels) * 1.5 + 1), 4),
                           constrained_layout=True)
    x = np.arange(len(treatment_labels))
    _support_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    _support_labels = ['0 paw', '1 paw', '2 paw', '3 paw', '4 paw']

    bottom = np.zeros(len(treatment_labels))
    for i, col in enumerate(support_cols):
        if col not in df.columns:
            continue
        means = []
        for treat in treatment_labels:
            if has_treats and len(treatment_labels) > 1:
                vals = df[df['treatment'] == treat][col].dropna().values
            else:
                vals = df[col].dropna().values
            means.append(np.nanmean(vals) if len(vals) > 0 else 0)
        means = np.array(means)
        ax.bar(x, means, bottom=bottom, color=_support_colors[i],
               label=_support_labels[i], edgecolor='black', linewidth=0.5)
        bottom += means

    ax.set_xticks(x)
    ax.set_xticklabels(treatment_labels, fontsize=11)
    _style_ax(ax, title='Support Patterns (locomotion)',
              ylabel='% of locomotion frames')
    ax.legend(fontsize=9, loc='upper right')

    btn_bar = ttk.Frame(frame)
    btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

    def _exp_graph(f=fig):
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
            initialfile='Support_patterns.png',
            parent=frame.winfo_toplevel())
        if path:
            f.savefig(path, dpi=300, bbox_inches='tight')

    ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)

    def _exp_data():
        path = filedialog.asksaveasfilename(
            defaultextension='.csv', filetypes=[('CSV', '*.csv')],
            initialfile='Support_patterns.csv',
            parent=frame.winfo_toplevel())
        if not path:
            return
        keep = ['session', 'subject', 'treatment'] + \
               [c for c in support_cols if c in df.columns]
        df[[c for c in keep if c in df.columns]].to_csv(path, index=False)

    ttk.Button(btn_bar, text="Export Data", command=_exp_data).pack(side='left', padx=2)
    _embed_figure(frame, fig, ax=ax)


# ─────────────────────────────────────────────────────────────────────────────
# Contour builders (Shape / Print / Filter Preview)
# ─────────────────────────────────────────────────────────────────────────────

def _no_contour_label(frame, role):
    """Graceful degradation when intermediates carry no contour shapes."""
    paw_label = _PAW_LABELS.get(role, role)
    ttk.Label(frame,
              text=(f'No stored paw-contour shapes for {paw_label}. '
                    'Re-run the analysis with paw-contour detection enabled '
                    '(contour intermediates are kept in memory per session).'),
              font=(FONT_FAMILY, 10, 'italic'),
              wraplength=520).pack(padx=12, pady=12)


def add_contour_shape_tab(host, frame, sessions_df, intermediates, role,
                          tab_name='Shape', filter_paw=False):
    """Mean contour outline for a paw, per treatment group."""
    graph_cfg = host.cfg
    has_treats, treatment_labels = _treatment_labels(sessions_df, graph_cfg)

    # Collect shapes (and solidities for filtering) per treatment
    treat_shapes = {t: [] for t in treatment_labels}
    treat_sols = {t: [] for t in treatment_labels}

    for _, row in sessions_df.iterrows():
        sess_name = row.get('session', '')
        treat = str(row.get('treatment', '')) if has_treats else 'All sessions'
        if treat not in treat_shapes:
            continue
        interm = intermediates.get(sess_name, {})
        pcd = interm.get('paw_contour_data', {})
        role_data = pcd.get(role, {})
        shapes = role_data.get('contour_shapes')
        if shapes:
            treat_shapes[treat].extend(shapes)
            sols = role_data.get('contour_solidities', [])
            if len(sols) < len(shapes):
                sols = list(sols) + [1.0] * (len(shapes) - len(sols))
            treat_sols[treat].extend(sols[:len(shapes)])

    # Apply paw-like filter if requested
    treat_total = {}  # original counts (before filtering)
    if filter_paw:
        sol_thresh = host.pawlike_thresholds.get('solidity', 1.00)
        ar_thresh = host.pawlike_thresholds.get('aspect_ratio', 1.6)
        circ_thresh = host.pawlike_thresholds.get('circularity', 0.10)
        for treat in treatment_labels:
            shapes = treat_shapes[treat]
            if not shapes:
                continue
            treat_total[treat] = len(shapes)
            stacked_t = np.array(shapes)
            ar_t, circ_t = _shape_metrics_batch(stacked_t)
            sols_t = np.array(treat_sols[treat][:len(shapes)], dtype=float)
            if len(sols_t) < len(shapes):
                sols_t = np.concatenate([sols_t, np.ones(len(shapes) - len(sols_t))])
            mask = (sols_t <= sol_thresh) & (ar_t <= ar_thresh) & (circ_t >= circ_thresh)
            treat_shapes[treat] = [shapes[i] for i in np.where(mask)[0]]

    # Check we have any shapes
    if not any(treat_shapes.values()):
        _no_contour_label(frame, role)
        return

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.set_aspect('equal')

    paw_label = _PAW_LABELS.get(role, role)

    for treat in treatment_labels:
        shapes = treat_shapes[treat]
        if not shapes:
            continue
        # Stack all shapes: (N_shapes, 64, 2)
        stacked = np.array(shapes)
        mean_shape = stacked.mean(axis=0)
        sd_shape = stacked.std(axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean_shape)

        # Close the polygon for plotting
        mean_closed = np.vstack([mean_shape, mean_shape[0:1]])
        sd_closed = np.vstack([sd_shape, sd_shape[0:1]])

        if graph_cfg and graph_cfg.get('colors') and treat in graph_cfg['colors']:
            raw = graph_cfg['colors'][treat]
            color = 'black' if raw == 'white_black' else raw
        else:
            color = _PAW_COLORS_ROLE.get(role, '#1f77b4')

        if filter_paw and treat in treat_total:
            _shape_lbl = f'{treat} (n={len(stacked)}/{treat_total[treat]})'
        else:
            _shape_lbl = f'{treat} (n={len(stacked)})'
        ax.plot(mean_closed[:, 0], mean_closed[:, 1],
                color=color, linewidth=2.0, label=_shape_lbl)

        # SD envelope: offset contour points radially by ±1 SD
        radial_sd = np.sqrt(sd_closed[:, 0] ** 2 + sd_closed[:, 1] ** 2)
        centroid = mean_closed[:-1].mean(axis=0)
        directions = mean_closed - centroid
        norms = np.sqrt((directions ** 2).sum(axis=1, keepdims=True))
        norms[norms == 0] = 1
        unit_dirs = directions / norms

        outer = mean_closed + unit_dirs * radial_sd[:, np.newaxis]
        inner = mean_closed - unit_dirs * radial_sd[:, np.newaxis]

        # Draw SD band as a filled ring (outer → reversed inner)
        ring_x = np.concatenate([outer[:, 0], inner[::-1, 0]])
        ring_y = np.concatenate([outer[:, 1], inner[::-1, 1]])
        ax.fill(ring_x, ring_y, alpha=0.15, color=color)

    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
    _style_ax(ax, title=f'Mean Contour - {paw_label}',
              xlabel='Normalized X', ylabel='Normalized Y')
    if len(treatment_labels) > 1:
        ax.legend(fontsize=9)
    # Invert y-axis (image coordinates: y increases downward)
    ax.invert_yaxis()

    btn_bar = ttk.Frame(frame)
    btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

    def _exp_graph(f=fig, n=f'contour_shape_{role}'):
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
            initialfile=f'{n}.png',
            parent=frame.winfo_toplevel())
        if path:
            f.savefig(path, dpi=300, bbox_inches='tight')

    ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)

    def _exp_data(n=f'contour_shape_{role}'):
        path = filedialog.asksaveasfilename(
            defaultextension='.csv', filetypes=[('CSV', '*.csv')],
            initialfile=f'{n}.csv', parent=frame.winfo_toplevel())
        if not path:
            return
        rows = []
        for treat in treatment_labels:
            shapes = treat_shapes.get(treat) or []
            if not shapes:
                continue
            stacked = np.array(shapes)
            mean_shape = stacked.mean(axis=0)
            sd_shape = (stacked.std(axis=0, ddof=1) if len(stacked) > 1
                        else np.zeros_like(mean_shape))
            for vi in range(mean_shape.shape[0]):
                rows.append({'treatment': treat, 'n_shapes': len(stacked),
                             'vertex': vi,
                             'mean_x': mean_shape[vi, 0],
                             'mean_y': mean_shape[vi, 1],
                             'sd_x': sd_shape[vi, 0],
                             'sd_y': sd_shape[vi, 1]})
        pd.DataFrame(rows).to_csv(path, index=False)

    ttk.Button(btn_bar, text="Export Data", command=_exp_data).pack(side='left', padx=2)
    _embed_figure(frame, fig, ax=ax)


def add_contour_print_tab(host, frame, sessions_df, intermediates, role,
                          tab_name='Paw Print', n_prints=5,
                          group_by='treatment', filter_paw=False):
    """Representative individual paw-print contours.

    Parameters
    ----------
    group_by : str
        'treatment' - one subplot per treatment group (default).
        'subject'  - one subplot per subject, colored by treatment.
    n_prints : int
        Number of representative contours to draw. 1 gives a single
        clean print with no background cloud.
    filter_paw : bool
        If True, only show shapes passing the paw-like thresholds.
    """
    graph_cfg = host.cfg
    has_treats, treatment_labels = _treatment_labels(sessions_df, graph_cfg)

    # ---- Gather shapes + solidities keyed by group (treatment or subject) ----
    group_shapes = {}       # group_key -> list of (64,2) arrays
    group_solidities = {}   # group_key -> list of float (extraction-time solidity)
    group_treat = {}        # group_key -> treatment label (for subject coloring)

    for _, row in sessions_df.iterrows():
        sess_name = row.get('session', '')
        treat = str(row.get('treatment', '')) if has_treats else 'All sessions'
        interm = intermediates.get(sess_name, {})
        pcd = interm.get('paw_contour_data', {})
        role_data = pcd.get(role, {})
        shapes = role_data.get('contour_shapes')
        if not shapes:
            continue
        sols = role_data.get('contour_solidities', [])
        # Pad solidities with 1.0 if missing (legacy data without stored solidities)
        if len(sols) < len(shapes):
            sols = list(sols) + [1.0] * (len(shapes) - len(sols))

        if group_by == 'subject':
            subj = str(row.get('subject', sess_name))
            if not subj.strip():
                subj = sess_name
            group_shapes.setdefault(subj, []).extend(shapes)
            group_solidities.setdefault(subj, []).extend(sols[:len(shapes)])
            group_treat[subj] = treat
        else:
            if treat not in treatment_labels:
                continue
            group_shapes.setdefault(treat, []).extend(shapes)
            group_solidities.setdefault(treat, []).extend(sols[:len(shapes)])
            group_treat[treat] = treat

    if not any(group_shapes.values()):
        _no_contour_label(frame, role)
        return

    # Determine ordered group keys
    if group_by == 'subject':
        # Sort subjects by treatment order, then alphabetically within
        treat_rank = {t: i for i, t in enumerate(treatment_labels)}
        active_groups = sorted(
            [g for g in group_shapes if group_shapes[g]],
            key=lambda g: (treat_rank.get(group_treat.get(g, ''), 999), g))
    else:
        active_groups = [t for t in treatment_labels if group_shapes.get(t)]

    n_groups = len(active_groups)
    if n_groups == 0:
        _no_contour_label(frame, role)
        return

    # Layout: wrap to multiple rows when many subjects
    max_cols = min(n_groups, 6)
    n_rows = (n_groups + max_cols - 1) // max_cols
    n_cols = min(n_groups, max_cols)

    paw_label = _PAW_LABELS.get(role, role)

    # Pre-compute per-group data for drawing (and interactive navigation)
    group_data = []  # list of dicts with stacked, sorted_idx, color, label
    for grp in active_groups:
        shapes = group_shapes[grp]
        stacked = np.array(shapes)  # (N, 64, 2)
        treat_for_color = group_treat.get(grp, '')
        if graph_cfg and graph_cfg.get('colors') and treat_for_color in graph_cfg['colors']:
            raw = graph_cfg['colors'][treat_for_color]
            color = 'black' if raw == 'white_black' else raw
        else:
            color = _PAW_COLORS_ROLE.get(role, '#1f77b4')
        total_all = len(stacked)
        mean_shape = stacked.mean(axis=0)
        dists = ((stacked - mean_shape) ** 2).sum(axis=(1, 2))
        sorted_idx = np.argsort(dists)  # all indices, ranked by closeness

        # Apply paw-like filter using stored solidities + shape metrics
        if filter_paw:
            sol_thresh = host.pawlike_thresholds.get('solidity', 1.00)
            ar_thresh = host.pawlike_thresholds.get('aspect_ratio', 1.6)
            circ_thresh = host.pawlike_thresholds.get('circularity', 0.10)
            grp_sols = np.array(group_solidities.get(grp, [1.0] * total_all))
            ar_all, circ_all = _shape_metrics_batch(stacked)
            paw_mask = ((grp_sols <= sol_thresh)
                        & (ar_all <= ar_thresh)
                        & (circ_all >= circ_thresh))
            sorted_idx = np.array([i for i in sorted_idx if paw_mask[i]])
            if len(sorted_idx) == 0:
                sorted_idx = np.argsort(dists)[:1]  # fallback: keep closest

        group_data.append(dict(
            stacked=stacked, sorted_idx=sorted_idx,
            color=color, grp=grp, treat=treat_for_color,
            total=len(sorted_idx), total_all=total_all))

    # Maximum offset: limited by the smallest group
    max_offset = max(0, min(gd['total'] - n_prints for gd in group_data))

    # Mutable state for navigation
    state = dict(offset=0, canvas_widget=None, fig=None)

    def _draw(offset):
        """Draw (or redraw) all subplots at the given offset."""
        if state['fig'] is not None:
            plt.close(state['fig'])
        if state['canvas_widget'] is not None:
            state['canvas_widget'].destroy()

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 4.5 * n_rows),
                                 constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()

        for idx, gd in enumerate(group_data):
            ax = axes_flat[idx]
            stacked = gd['stacked']
            si = gd['sorted_idx']
            color = gd['color']
            k = min(n_prints, gd['total'] - offset)
            if k <= 0:
                k = 1
            sel = si[offset:offset + k]

            best_alpha = 0.85 if k == 1 else 0.7

            # Background representatives (indices 1..k-1)
            for ci in sel[1:]:
                pts = stacked[ci]
                closed = np.vstack([pts, pts[0:1]])
                ax.fill(closed[:, 0], closed[:, 1],
                        facecolor=color, alpha=0.15,
                        edgecolor=color, linewidth=0.8)

            # Primary contour on top
            best = stacked[sel[0]]
            best_closed = np.vstack([best, best[0:1]])
            ax.fill(best_closed[:, 0], best_closed[:, 1],
                    facecolor=color, alpha=best_alpha,
                    edgecolor=color, linewidth=0.8)

            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)

            rank_start = offset + 1  # 1-based for display
            if filter_paw and gd.get('total_all', gd['total']) != gd['total']:
                count_lbl = f'#{rank_start} of {gd["total"]} paw-like / {gd["total_all"]} total'
            else:
                count_lbl = f'#{rank_start} of {gd["total"]}'
            if group_by == 'subject':
                subtitle = f'{gd["grp"]} ({count_lbl})'
            elif n_groups > 1:
                subtitle = f'{gd["treat"]} ({count_lbl})'
            else:
                subtitle = f'{paw_label} ({count_lbl})'
            _style_ax(ax, title=subtitle,
                      xlabel='Normalized X', ylabel='Normalized Y')

        for j in range(n_groups, len(axes_flat)):
            axes_flat[j].set_visible(False)

        if group_by == 'subject':
            fig.suptitle(f'Paw Print - {paw_label} (by Subject)', fontsize=12)
        elif n_groups > 1:
            fig.suptitle(f'Paw Print - {paw_label}', fontsize=12)

        state['fig'] = fig
        canvas = FigureCanvasTkAgg(fig, master=frame)
        cw = canvas.get_tk_widget()
        cw.figure = fig
        cw.pack(fill='both', expand=True)
        _bind_tight_layout_on_resize(canvas, fig)
        _draw_canvas_fit(canvas, fig)
        state['canvas_widget'] = cw
        # Update button states
        if max_offset > 0:
            btn_prev.config(state='disabled' if offset == 0 else 'normal')
            btn_next.config(state='disabled' if offset >= max_offset else 'normal')

    # --- Button bar (packed at bottom before the figure) ---
    btn_bar = ttk.Frame(frame)
    btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

    suffix = f'_{group_by}' if group_by == 'subject' else ''

    def _exp_graph():
        if state['fig'] is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
            initialfile=f'contour_print_{role}{suffix}.png',
            parent=frame.winfo_toplevel())
        if path:
            state['fig'].savefig(path, dpi=300, bbox_inches='tight')

    ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)

    def _exp_data():
        path = filedialog.asksaveasfilename(
            defaultextension='.csv', filetypes=[('CSV', '*.csv')],
            initialfile=f'contour_print_{role}{suffix}.csv',
            parent=frame.winfo_toplevel())
        if not path:
            return
        rows = []
        off = state.get('offset', 0)
        for gd in group_data:
            stacked = gd['stacked']
            idx = gd['sorted_idx']
            picks = idx[off * n_prints:(off + 1) * n_prints]
            for rank, si in enumerate(picks):
                shp = stacked[si]
                for vi in range(shp.shape[0]):
                    rows.append({'group': gd.get('label', ''),
                                 'print_rank': rank, 'shape_index': int(si),
                                 'vertex': vi,
                                 'x': shp[vi, 0], 'y': shp[vi, 1]})
        pd.DataFrame(rows).to_csv(path, index=False)

    ttk.Button(btn_bar, text="Export Data", command=_exp_data).pack(side='left', padx=2)

    # Navigation buttons (prev / next contour selection)
    def _prev():
        if state['offset'] > 0:
            state['offset'] -= 1
            _draw(state['offset'])

    def _next():
        if state['offset'] < max_offset:
            state['offset'] += 1
            _draw(state['offset'])

    btn_prev = ttk.Button(btn_bar, text="◀ Prev", command=_prev)
    btn_next = ttk.Button(btn_bar, text="Next ▶", command=_next)
    if max_offset > 0:
        btn_prev.pack(side='left', padx=4)
        btn_next.pack(side='left', padx=4)

    # Initial draw
    _draw(0)

def add_contour_filter_preview_tab(host, frame, sessions_df, intermediates,
                                   role):
    """Interactive filter preview with sliders for solidity, aspect ratio,
    circularity. Adjusting + Apply updates host.pawlike_thresholds,
    recomputes the pawlike_* columns and fires host.on_pawlike_change so the
    tab can re-render dependent entries.

    Note: the old window's "Sync Left/Right thresholds" checkbox mirrored
    slider positions between two simultaneously-built L/R tabs. Entries are
    now rendered one at a time and the thresholds are (and always were) a
    single global set applied to all paws on Apply, so the mirror checkbox
    is obsolete and not ported.
    """
    GRID_COLS = 4
    GRID_ROWS = 5
    PAGE_SIZE = GRID_COLS * GRID_ROWS

    # ---- Gather all shapes + extraction-time solidities ----
    all_shapes = []
    all_sols = []
    for _, row in sessions_df.iterrows():
        sess_name = row.get('session', '')
        interm = intermediates.get(sess_name, {})
        pcd = interm.get('paw_contour_data', {})
        role_data = pcd.get(role, {})
        shapes = role_data.get('contour_shapes')
        if not shapes:
            continue
        sols = role_data.get('contour_solidities', [])
        # Pad with 1.0 for legacy data without stored solidities
        if len(sols) < len(shapes):
            sols = list(sols) + [1.0] * (len(shapes) - len(sols))
        all_shapes.extend(shapes)
        all_sols.extend(sols[:len(shapes)])

    if not all_shapes:
        _no_contour_label(frame, role)
        return

    stacked = np.array(all_shapes)  # (N, 64, 2)
    sol_vals = np.array(all_sols, dtype=float)
    total_all = len(stacked)

    # Pre-compute aspect ratio and circularity for all shapes (vectorized)
    ar_vals, circ_vals = _shape_metrics_batch(stacked)

    paw_label = _PAW_LABELS.get(role, role)
    role_color = _PAW_COLORS_ROLE.get(role, '#1f77b4')

    # -- Slider controls --
    ctrl_frame = ttk.LabelFrame(frame, text='Filter Thresholds', padding=6)
    ctrl_frame.pack(side='top', fill='x', padx=6, pady=(4, 2))

    sol_var = tk.DoubleVar(value=host.pawlike_thresholds.get('solidity', 1.00))
    ar_var = tk.DoubleVar(value=host.pawlike_thresholds.get('aspect_ratio', 1.6))
    circ_var = tk.DoubleVar(value=host.pawlike_thresholds.get('circularity', 0.10))

    def _make_slider(parent, label, var, from_, to, resolution, row, tooltip):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(0, 4))
        scale = ttk.Scale(parent, from_=from_, to=to, variable=var,
                          orient='horizontal', length=260,
                          command=lambda _: _on_slider_change())
        scale.grid(row=row, column=1, sticky='ew', padx=4)
        val_lbl = ttk.Label(parent, text=f'{var.get():.2f}', width=6)
        val_lbl.grid(row=row, column=2, padx=(0, 4))
        host.tip(scale, tooltip)
        return val_lbl

    ctrl_frame.columnconfigure(1, weight=1)
    sol_lbl = _make_slider(ctrl_frame, 'Solidity  ≤', sol_var, 0.50, 1.00, 0.01, 0,
                           'Shapes with solidity ≤ threshold pass (lower = stricter)')
    ar_lbl = _make_slider(ctrl_frame, 'Aspect Ratio  ≤', ar_var, 1.0, 10.0, 0.1, 1,
                          'Shapes with aspect ratio ≤ threshold pass (filters elongated shapes)')
    circ_lbl = _make_slider(ctrl_frame, 'Circularity  ≥', circ_var, 0.0, 1.0, 0.01, 2,
                            'Shapes with circularity ≥ threshold pass (filters near-circular blobs)')

    # Status label
    status_var = tk.StringVar(value='')
    ttk.Label(ctrl_frame, textvariable=status_var, font=('TkDefaultFont', 9, 'bold')).grid(
        row=3, column=0, columnspan=3, sticky='w', pady=(4, 0))

    # -- View toggle --
    view_frame = ttk.Frame(frame)
    view_frame.pack(side='top', fill='x', padx=6, pady=(2, 2))
    view_var = tk.StringVar(value='Included')
    ttk.Label(view_frame, text='View:').pack(side='left', padx=(0, 4))
    view_combo = ttk.Combobox(view_frame, textvariable=view_var,
                              values=['Included', 'Excluded'],
                              state='readonly', width=12)
    view_combo.pack(side='left')

    # -- Button bar (pack before canvas so it reserves space at bottom) --
    btn_bar = ttk.Frame(frame)
    btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 4))

    # -- Canvas area --
    canvas_frame = ttk.Frame(frame)
    canvas_frame.pack(side='top', fill='both', expand=True, padx=4, pady=2)

    # Mutable state
    state = dict(page=0, canvas_widget=None, fig=None,
                 included_idx=np.array([], dtype=int),
                 excluded_idx=np.array([], dtype=int))

    def _compute_mask():
        """Recompute filter mask from current slider values."""
        sol_t = sol_var.get()
        ar_t = ar_var.get()
        circ_t = circ_var.get()
        mask = (sol_vals <= sol_t) & (ar_vals <= ar_t) & (circ_vals >= circ_t)
        state['included_idx'] = np.where(mask)[0]
        state['excluded_idx'] = np.where(~mask)[0]
        n_inc = len(state['included_idx'])
        pct = int(100 * n_inc / total_all) if total_all > 0 else 0
        status_var.set(f'Keeping {n_inc} of {total_all} shapes ({pct}%)')
        # Update value labels
        sol_lbl.config(text=f'{sol_t:.2f}')
        ar_lbl.config(text=f'{ar_t:.2f}')
        circ_lbl.config(text=f'{circ_t:.2f}')

    def _draw_page(page):
        if state['fig'] is not None:
            plt.close(state['fig'])
        if state['canvas_widget'] is not None:
            state['canvas_widget'].destroy()
            state['canvas_widget'] = None

        viewing = view_var.get()
        idx_pool = state['included_idx'] if viewing == 'Included' else state['excluded_idx']
        total_pool = len(idx_pool)

        if total_pool == 0:
            # Show empty message
            lbl = ttk.Label(canvas_frame,
                            text=f'No {viewing.lower()} shapes with current thresholds.',
                            font=('TkDefaultFont', 10))
            lbl.pack(fill='both', expand=True)
            state['canvas_widget'] = lbl
            state['fig'] = None
            btn_prev.config(state='disabled')
            btn_next.config(state='disabled')
            return

        n_pages = max(1, (total_pool + PAGE_SIZE - 1) // PAGE_SIZE)
        page = max(0, min(page, n_pages - 1))
        state['page'] = page

        start = page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total_pool)
        page_indices = idx_pool[start:end]
        n_show = len(page_indices)
        n_r = (n_show + GRID_COLS - 1) // GRID_COLS
        n_c = min(n_show, GRID_COLS)

        fig, axes = plt.subplots(n_r, n_c,
                                 figsize=(2.5 * n_c, 2.5 * n_r),
                                 constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()

        for i, si in enumerate(page_indices):
            ax = axes_flat[i]
            pts = stacked[si]
            closed = np.vstack([pts, pts[0:1]])
            if viewing == 'Included':
                fc, ec = role_color, role_color
            else:
                fc, ec = '#999999', '#666666'
            ax.fill(closed[:, 0], closed[:, 1],
                    facecolor=fc, alpha=0.5,
                    edgecolor=ec, linewidth=0.8)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f'sol={sol_vals[si]:.2f}  ar={ar_vals[si]:.1f}  ci={circ_vals[si]:.2f}',
                         fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n_show, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f'{viewing}: {total_pool} shapes - {paw_label}  '
            f'(page {page + 1}/{n_pages})',
            fontsize=10, y=1.02)

        state['fig'] = fig
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        cw = canvas.get_tk_widget()
        cw.figure = fig
        cw.pack(fill='both', expand=True)
        _bind_tight_layout_on_resize(canvas, fig)
        _draw_canvas_fit(canvas, fig)
        state['canvas_widget'] = cw

        btn_prev.config(state='disabled' if page == 0 else 'normal')
        btn_next.config(state='disabled' if page >= n_pages - 1 else 'normal')

    def _on_slider_change():
        _compute_mask()
        state['page'] = 0
        _draw_page(0)

    def _on_view_change(_event=None):
        state['page'] = 0
        _draw_page(0)

    view_combo.bind('<<ComboboxSelected>>', _on_view_change)

    def _prev():
        if state['page'] > 0:
            _draw_page(state['page'] - 1)

    def _next():
        _draw_page(state['page'] + 1)

    btn_prev = ttk.Button(btn_bar, text="◀ Prev", command=_prev)
    btn_prev.pack(side='left', padx=4)
    btn_next = ttk.Button(btn_bar, text="Next ▶", command=_next)
    btn_next.pack(side='left', padx=4)

    def _exp_graph():
        if state['fig'] is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
            initialfile=f'contour_filter_preview_{role}.png',
            parent=frame.winfo_toplevel())
        if path:
            state['fig'].savefig(path, dpi=300, bbox_inches='tight')

    ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)

    def _exp_data():
        path = filedialog.asksaveasfilename(
            defaultextension='.csv', filetypes=[('CSV', '*.csv')],
            initialfile=f'contour_filter_metrics_{role}.csv',
            parent=frame.winfo_toplevel())
        if not path:
            return
        sol_t, ar_t = sol_var.get(), ar_var.get()
        circ_t = circ_var.get()
        inc = (sol_vals <= sol_t) & (ar_vals <= ar_t) & (circ_vals >= circ_t)
        pd.DataFrame({'shape_index': np.arange(len(sol_vals)),
                      'solidity': sol_vals, 'aspect_ratio': ar_vals,
                      'circularity': circ_vals,
                      'included': inc}).to_csv(path, index=False)

    ttk.Button(btn_bar, text="Export Data", command=_exp_data).pack(side='left', padx=2)

    def _apply():
        """Store current thresholds, recompute pawlike metrics, and notify
        the host so dependent (Filtered Contour) entries re-render."""
        host.pawlike_thresholds['solidity'] = sol_var.get()
        host.pawlike_thresholds['aspect_ratio'] = ar_var.get()
        host.pawlike_thresholds['circularity'] = circ_var.get()
        try:
            recompute_pawlike_metrics(host)
            host.on_pawlike_change()
        except Exception as exc:
            messagebox.showerror(
                'Rebuild Error',
                f'Failed to regenerate Filtered Contour tab:\n{exc}',
                parent=frame.winfo_toplevel())

    ttk.Button(btn_bar, text="Apply", command=_apply).pack(side='right', padx=4)

    # Initial draw (entries are built on demand, so draw immediately -
    # the old <Map> deferral existed because all paw tabs were built at once)
    _compute_mask()
    _draw_page(0)


# ─────────────────────────────────────────────────────────────────────────────
# Pawlike (Filtered Contour) recompute - mutates host.summary_df / bins_df
# by design: these columns are DERIVED from intermediates + thresholds.
# ─────────────────────────────────────────────────────────────────────────────

def recompute_pawlike_metrics(host):
    """Recompute pawlike columns in summary_df and bins_df using current
    thresholds (verbatim math from gait_limb_tab._recompute_pawlike_metrics)."""
    if host.summary_df is None:
        return
    PAWLIKE_SOL = host.pawlike_thresholds.get('solidity', 1.00)
    PAWLIKE_AR = host.pawlike_thresholds.get('aspect_ratio', 1.6)
    PAWLIKE_CIRC = host.pawlike_thresholds.get('circularity', 0.10)

    intermediates = host.intermediates or {}

    for idx, row in host.summary_df.iterrows():
        sess_name = row.get('session')
        if sess_name not in intermediates:
            continue
        inter = intermediates[sess_name]
        pcd = inter.get('paw_contour_data', {})
        if not pcd:
            continue
        fps = inter.get('fps', 60)
        n_frames = inter.get('n_frames', 0)
        contact_masks = inter.get('contact_masks', {})

        for role in list(pcd.keys()):
            areas = pcd[role]['areas']
            spreads = pcd[role]['spreads']
            intensities = pcd[role]['intensities']
            cm = contact_masks.get(role)
            if cm is not None:
                mask_arr = cm.values.astype(bool) if hasattr(cm, 'values') else np.asarray(cm, dtype=bool)
            else:
                mask_arr = np.ones(n_frames, dtype=bool)

            widths = pcd[role].get('widths')
            solidities = pcd[role].get('solidities')
            aspect_ratios = pcd[role].get('aspect_ratios')
            circularities = pcd[role].get('circularities')

            valid = mask_arr[:len(areas)] & (areas[:len(mask_arr)] > 0)
            if solidities is not None:
                sol_arr = solidities[:len(mask_arr)]
                valid_paw = valid & (sol_arr <= PAWLIKE_SOL)
                if aspect_ratios is not None:
                    valid_paw = valid_paw & (aspect_ratios[:len(mask_arr)] <= PAWLIKE_AR)
                if circularities is not None:
                    valid_paw = valid_paw & (circularities[:len(mask_arr)] >= PAWLIKE_CIRC)

                _pca = areas[:len(mask_arr)][valid_paw]
                host.summary_df.at[idx, f'pawlike_area_{role}'] = round(float(np.nanmean(_pca)), 2) if len(_pca) > 0 else float('nan')
                _pcs = spreads[:len(mask_arr)][valid_paw]
                host.summary_df.at[idx, f'pawlike_spread_{role}'] = round(float(np.nanmean(_pcs)), 2) if len(_pcs) > 0 else float('nan')
                _pci = intensities[:len(mask_arr)][valid_paw]
                host.summary_df.at[idx, f'pawlike_intensity_{role}'] = round(float(np.nanmean(_pci)), 2) if len(_pci) > 0 else float('nan')
                if widths is not None:
                    host.summary_df.at[idx, f'pawlike_width_{role}'] = round(float(np.nanmean(widths[:len(mask_arr)][valid_paw])), 2) if valid_paw.any() else float('nan')
                    host.summary_df.at[idx, f'pawlike_solidity_{role}'] = round(float(np.nanmean(sol_arr[valid_paw])), 4) if valid_paw.any() else float('nan')
                    host.summary_df.at[idx, f'pawlike_aspect_ratio_{role}'] = round(float(np.nanmean(aspect_ratios[:len(mask_arr)][valid_paw])), 4) if valid_paw.any() else float('nan')
                    host.summary_df.at[idx, f'pawlike_circularity_{role}'] = round(float(np.nanmean(circularities[:len(mask_arr)][valid_paw])), 4) if valid_paw.any() else float('nan')

        # Recompute ratios
        df = host.summary_df
        if 'pawlike_area_HL' in df.columns and 'pawlike_area_HR' in df.columns:
            aHL, aHR = df.at[idx, 'pawlike_area_HL'], df.at[idx, 'pawlike_area_HR']
            if not (np.isnan(aHL) or np.isnan(aHR)) and aHR > 0:
                df.at[idx, 'pawlike_area_ratio_hind'] = round(aHL / aHR, 4)
            else:
                df.at[idx, 'pawlike_area_ratio_hind'] = float('nan')
        if 'pawlike_intensity_HL' in df.columns and 'pawlike_intensity_HR' in df.columns:
            iHL, iHR = df.at[idx, 'pawlike_intensity_HL'], df.at[idx, 'pawlike_intensity_HR']
            if not (np.isnan(iHL) or np.isnan(iHR)) and iHR > 0:
                df.at[idx, 'pawlike_intensity_ratio_hind'] = round(iHL / iHR, 4)
            else:
                df.at[idx, 'pawlike_intensity_ratio_hind'] = float('nan')

    # Recompute bins
    if host.bins_df is not None and not host.bins_df.empty:
        for bidx, brow in host.bins_df.iterrows():
            sess_name = brow.get('session')
            if sess_name not in intermediates:
                continue
            inter = intermediates[sess_name]
            pcd = inter.get('paw_contour_data', {})
            if not pcd:
                continue
            fps = inter.get('fps', 60)
            n_frames = inter.get('n_frames', 0)
            contact_masks = inter.get('contact_masks', {})
            bin_start_s = brow.get('bin_start_s', 0)
            bin_end_s = brow.get('bin_end_s', 0)
            start_f = int(round(bin_start_s * fps))
            end_f = int(round(bin_end_s * fps))
            frame_slice = slice(start_f, end_f)

            for role in list(pcd.keys()):
                areas_full = pcd[role]['areas']
                spreads_full = pcd[role]['spreads']
                ints_full = pcd[role]['intensities']
                cm = contact_masks.get(role)
                if cm is not None:
                    mask_full = cm.values.astype(bool) if hasattr(cm, 'values') else np.asarray(cm, dtype=bool)
                else:
                    mask_full = np.ones(n_frames, dtype=bool)
                mask_arr = mask_full[frame_slice]
                areas_sl = areas_full[frame_slice]
                spreads_sl = spreads_full[frame_slice]
                ints_sl = ints_full[frame_slice]

                widths_full = pcd[role].get('widths')
                solidities_full = pcd[role].get('solidities')
                ar_full = pcd[role].get('aspect_ratios')
                circ_full = pcd[role].get('circularities')
                widths_sl = widths_full[frame_slice] if widths_full is not None else None
                solidities_sl = solidities_full[frame_slice] if solidities_full is not None else None
                ar_sl = ar_full[frame_slice] if ar_full is not None else None
                circ_sl = circ_full[frame_slice] if circ_full is not None else None

                valid = mask_arr[:len(areas_sl)] & (areas_sl[:len(mask_arr)] > 0)
                if solidities_sl is not None:
                    sol_arr = solidities_sl[:len(mask_arr)]
                    valid_paw = valid & (sol_arr <= PAWLIKE_SOL)
                    if ar_sl is not None:
                        valid_paw = valid_paw & (ar_sl[:len(mask_arr)] <= PAWLIKE_AR)
                    if circ_sl is not None:
                        valid_paw = valid_paw & (circ_sl[:len(mask_arr)] >= PAWLIKE_CIRC)

                    _pca = areas_sl[:len(mask_arr)][valid_paw]
                    host.bins_df.at[bidx, f'pawlike_area_{role}'] = round(float(np.nanmean(_pca)), 2) if len(_pca) > 0 else float('nan')
                    _pcs = spreads_sl[:len(mask_arr)][valid_paw]
                    host.bins_df.at[bidx, f'pawlike_spread_{role}'] = round(float(np.nanmean(_pcs)), 2) if len(_pcs) > 0 else float('nan')
                    _pci = ints_sl[:len(mask_arr)][valid_paw]
                    host.bins_df.at[bidx, f'pawlike_intensity_{role}'] = round(float(np.nanmean(_pci)), 2) if len(_pci) > 0 else float('nan')
                    if widths_sl is not None:
                        host.bins_df.at[bidx, f'pawlike_width_{role}'] = round(float(np.nanmean(widths_sl[:len(mask_arr)][valid_paw])), 2) if valid_paw.any() else float('nan')
                        host.bins_df.at[bidx, f'pawlike_solidity_{role}'] = round(float(np.nanmean(sol_arr[valid_paw])), 4) if valid_paw.any() else float('nan')
                        host.bins_df.at[bidx, f'pawlike_aspect_ratio_{role}'] = round(float(np.nanmean(ar_sl[:len(mask_arr)][valid_paw])), 4) if valid_paw.any() else float('nan')
                        host.bins_df.at[bidx, f'pawlike_circularity_{role}'] = round(float(np.nanmean(circ_sl[:len(mask_arr)][valid_paw])), 4) if valid_paw.any() else float('nan')

            # Recompute bin ratios
            bdf = host.bins_df
            if 'pawlike_area_HL' in bdf.columns and 'pawlike_area_HR' in bdf.columns:
                aHL, aHR = bdf.at[bidx, 'pawlike_area_HL'], bdf.at[bidx, 'pawlike_area_HR']
                if not (np.isnan(aHL) or np.isnan(aHR)) and aHR > 0:
                    bdf.at[bidx, 'pawlike_area_ratio_hind'] = round(aHL / aHR, 4)
                else:
                    bdf.at[bidx, 'pawlike_area_ratio_hind'] = float('nan')
            if 'pawlike_intensity_HL' in bdf.columns and 'pawlike_intensity_HR' in bdf.columns:
                iHL, iHR = bdf.at[bidx, 'pawlike_intensity_HL'], bdf.at[bidx, 'pawlike_intensity_HR']
                if not (np.isnan(iHL) or np.isnan(iHR)) and iHR > 0:
                    bdf.at[bidx, 'pawlike_intensity_ratio_hind'] = round(iHL / iHR, 4)
                else:
                    bdf.at[bidx, 'pawlike_intensity_ratio_hind'] = float('nan')

# ─────────────────────────────────────────────────────────────────────────────
# Statistics (Σ flip + Statistics category)
# ─────────────────────────────────────────────────────────────────────────────

def perform_wb_statistical_test(host, data_by_treatment, treatments):
    """Statistical test with parametric/non-parametric support and effect sizes."""
    from scipy import stats as _scipy_stats

    if not host.enable_stats():
        return None

    groups = [data_by_treatment.get(t, np.array([])) for t in treatments]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return None

    alpha = _stats_alpha(host)
    test_type = _stats_test(host)
    paradigm = _stats_paradigm(host)

    # Determine parametric vs non-parametric
    use_nonparam = False
    if test_type == 'nonparametric' or paradigm == 'nonparametric':
        use_nonparam = True
    elif paradigm == 'auto':
        # Shapiro-Wilk on each group
        for g in groups:
            if len(g) >= 3:
                try:
                    _, sw_p = _scipy_stats.shapiro(g)
                    if sw_p < 0.05:
                        use_nonparam = True
                        break
                except Exception as _sw_err:
                    print(f"Warning: Shapiro-Wilk normality test failed: {_sw_err}")

    # Auto-select test based on group count
    if test_type == 'auto' or test_type == 'nonparametric':
        test_type = 't-test' if len(groups) == 2 else 'ANOVA'

    results = {'alpha': alpha}

    if test_type == 't-test' and len(groups) == 2:
        if use_nonparam:
            stat, p_val = _scipy_stats.mannwhitneyu(groups[0], groups[1],
                                                    alternative='two-sided')
            results['test_type'] = 'Mann-Whitney U'
        else:
            stat, p_val = _scipy_stats.ttest_ind(groups[0], groups[1],
                                                 equal_var=False)
            results['test_type'] = "Welch's t-test"
        results['p_value'] = float(p_val)
        results['significant'] = bool(p_val < alpha)
        results['comparison'] = f"{treatments[0]} vs {treatments[1]}"
        # Cohen's d
        n1, n2 = len(groups[0]), len(groups[1])
        if n1 > 1 and n2 > 1:
            m1, m2 = np.mean(groups[0]), np.mean(groups[1])
            s1, s2 = np.std(groups[0], ddof=1), np.std(groups[1], ddof=1)
            pooled_s = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            if pooled_s > 0:
                results['effect_size'] = float(abs(m1 - m2) / pooled_s)
                results['effect_size_type'] = "Cohen's d"

    else:  # ANOVA / Kruskal-Wallis
        if use_nonparam:
            stat, p_val = _scipy_stats.kruskal(*groups)
            results['test_type'] = 'Kruskal-Wallis'
        else:
            stat, p_val = _scipy_stats.f_oneway(*groups)
            results['test_type'] = 'ANOVA'
        results['p_value'] = float(p_val)
        results['significant'] = bool(p_val < alpha)

        # Eta-squared
        if not use_nonparam:
            grand_mean = np.mean(np.concatenate(groups))
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = sum((x - grand_mean)**2 for g in groups for x in g)
            if ss_total > 0:
                results['effect_size'] = float(ss_between / ss_total)
                results['effect_size_type'] = 'eta-squared'

        if p_val < alpha:
            pairwise = {}
            valid_treats = [t for t in treatments
                            if len(data_by_treatment.get(t, [])) > 0]
            valid_groups = [data_by_treatment[t] for t in valid_treats]
            n_comparisons = len(valid_treats) * (len(valid_treats) - 1) // 2
            for i in range(len(valid_treats)):
                for j in range(i + 1, len(valid_treats)):
                    if use_nonparam:
                        _, pp = _scipy_stats.mannwhitneyu(
                            valid_groups[i], valid_groups[j],
                            alternative='two-sided')
                    else:
                        _, pp = _scipy_stats.ttest_ind(
                            valid_groups[i], valid_groups[j],
                            equal_var=False)
                    pp_adj = min(float(pp) * n_comparisons, 1.0)
                    key = f"{valid_treats[i]}_vs_{valid_treats[j]}"
                    pairwise[key] = {
                        'p_value':     pp_adj,
                        'significant': bool(pp_adj < alpha),
                    }
            results['pairwise'] = pairwise

    return results


def add_wb_stats_section(host, parent_frame, title, df, metric,
                         treatments, agg_method):
    """Create a descriptive-stats + test-result block for one WB metric.

    agg_method: 'value' - one row per subject (no aggregation needed)
                'mean'  - aggregate per subject across time bins
    Column names: treatment (lower), subject (lower).
    """
    section_frame = ttk.LabelFrame(parent_frame, text=title, padding=10)
    section_frame.pack(fill='x', padx=10, pady=10)

    # ── Aggregate per subject ────────────────────────────────────────
    if agg_method == 'mean':
        per_subject = (df.groupby(['subject', 'treatment'])[metric]
                       .mean().reset_index())
    else:
        # 'value': already one row per session/subject
        cols = [c for c in ('subject', 'treatment', metric) if c in df.columns]
        per_subject = df[cols].copy()

    # ── Descriptive statistics table ─────────────────────────────────
    ttk.Label(section_frame, text="Descriptive Statistics:",
              font=(FONT_FAMILY, 10, 'bold')).pack(anchor='w')

    desc_table = ttk.Frame(section_frame)
    desc_table.pack(fill='x', pady=5)

    headers = ['Treatment', 'N', 'Mean', 'SD', 'SEM', 'Min', 'Max']
    for i, hdr in enumerate(headers):
        ttk.Label(desc_table, text=hdr, font=(FONT_FAMILY, 9, 'bold'),
                  relief='solid', borderwidth=1, width=12).grid(
            row=0, column=i, sticky='ew', padx=1, pady=1)

    for row_idx, treat in enumerate(treatments, start=1):
        vals = per_subject[per_subject['treatment'] == treat][metric].dropna().values
        if len(vals) == 0:
            continue
        n = len(vals)
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if n > 1 else 0.0
        sem = sd / np.sqrt(n)
        mn = np.min(vals)
        mx = np.max(vals)
        for col_idx, cell in enumerate(
                [treat, str(n), f'{mean:.2f}', f'{sd:.2f}',
                 f'{sem:.2f}', f'{mn:.2f}', f'{mx:.2f}']):
            ttk.Label(desc_table, text=cell, relief='solid',
                      borderwidth=1, width=12).grid(
                row=row_idx, column=col_idx, sticky='ew', padx=1, pady=1)

    # ── Statistical test ─────────────────────────────────────────────
    ttk.Label(section_frame, text="Statistical Test:",
              font=(FONT_FAMILY, 10, 'bold')).pack(anchor='w', pady=(10, 5))

    data_by_treatment = {t: per_subject[per_subject['treatment'] == t][metric]
                            .dropna().values
                         for t in treatments}
    stats_res = perform_wb_statistical_test(host, data_by_treatment, treatments)

    if stats_res is None:
        ttk.Label(section_frame,
                  text="No statistical test performed (enable in Statistical Tests panel).",
                  foreground='gray', font=(FONT_FAMILY, 9, 'italic')).pack(anchor='w')
        return

    p_val = stats_res['p_value']
    alpha = stats_res['alpha']
    if p_val < 0.001:
        p_text, sig = 'p < 0.001', '***'
    elif p_val < 0.01:
        p_text, sig = f'p = {p_val:.4f}', '**'
    elif p_val < alpha:
        p_text, sig = f'p = {p_val:.4f}', '*'
    else:
        p_text, sig = f'p = {p_val:.4f}', 'ns'

    result_text = f"{stats_res['test_type']}: {p_text} {sig}"
    ttk.Label(section_frame, text=result_text, font=(FONT_FAMILY, 10),
              foreground='darkblue').pack(anchor='w')

    if 'effect_size' in stats_res:
        es_text = f"Effect size ({stats_res['effect_size_type']}): {stats_res['effect_size']:.3f}"
        ttk.Label(section_frame, text=es_text, font=(FONT_FAMILY, 10),
                  foreground='darkblue').pack(anchor='w')

    # ── Pairwise comparisons (ANOVA + significant) ───────────────────
    if not stats_res.get('pairwise') and len(treatments) > 2:
        ttk.Label(section_frame,
                  text="Pairwise comparisons omitted - omnibus not "
                       "significant (protected testing).",
                  font=(FONT_FAMILY, 9, 'italic'),
                  foreground='gray').pack(anchor='w', pady=(4, 0))
    if 'pairwise' in stats_res and stats_res['pairwise']:
        ttk.Label(section_frame, text="Pairwise Comparisons:",
                  font=(FONT_FAMILY, 10, 'bold')).pack(anchor='w', pady=(10, 5))
        pw_table = ttk.Frame(section_frame)
        pw_table.pack(fill='x', pady=5)

        for col_idx, (hdr, w) in enumerate(
                [('Comparison', 30), ('p-value', 15), ('Significance', 15)]):
            ttk.Label(pw_table, text=hdr, font=(FONT_FAMILY, 9, 'bold'),
                      relief='solid', borderwidth=1, width=w).grid(
                row=0, column=col_idx, sticky='ew', padx=1, pady=1)

        for row_idx, (comparison, result) in enumerate(
                stats_res['pairwise'].items(), start=1):
            comp_text = comparison.replace('_vs_', ' vs ')
            p = result['p_value']
            a = stats_res['alpha']
            if p < 0.001:
                p_display, sig, fg = 'p < 0.001', '***', 'darkgreen'
            elif p < 0.01:
                p_display, sig, fg = f'p = {p:.4f}', '**', 'green'
            elif p < a:
                p_display, sig, fg = f'p = {p:.4f}', '*', 'orange'
            else:
                p_display, sig, fg = f'p = {p:.4f}', 'ns', 'gray'

            ttk.Label(pw_table, text=comp_text, relief='solid',
                      borderwidth=1, width=30).grid(
                row=row_idx, column=0, sticky='ew', padx=1, pady=1)
            ttk.Label(pw_table, text=p_display, relief='solid',
                      borderwidth=1, width=15).grid(
                row=row_idx, column=1, sticky='ew', padx=1, pady=1)
            ttk.Label(pw_table, text=sig, relief='solid', borderwidth=1,
                      width=15, foreground=fg,
                      font=(FONT_FAMILY, 9, 'bold')).grid(
                row=row_idx, column=2, sticky='ew', padx=1, pady=1)


def add_wb_timecourse_stats_section(host, parent_frame, bins_df,
                                    metric, treatments):
    """Two-way ANOVA (treatment × time) + optional per-timepoint post-hoc.

    Time column in bins_df is bin_start_s (seconds).
    """
    from scipy import stats as _scipy_stats

    section_frame = ttk.LabelFrame(
        parent_frame,
        text=f"Time Course Statistics - {metric.replace('_', ' ')}",
        padding=10)
    section_frame.pack(fill='x', padx=10, pady=10)

    alpha = _stats_alpha(host)

    # ── Part A: Two-Way ANOVA ────────────────────────────────────────
    ttk.Label(section_frame,
              text="═══ Two-Way ANOVA (Treatment × Time) ═══",
              font=(FONT_FAMILY, 10, 'bold'), foreground='darkblue').pack(
        anchor='w', pady=5)

    try:
        import statsmodels.api as _sm
        from statsmodels.formula.api import ols as _ols

        anova_df = bins_df[['subject', 'treatment', 'bin_start_s', metric]].dropna().copy()
        anova_df['treatment'] = anova_df['treatment'].astype('category')
        anova_df['bin_start_s'] = anova_df['bin_start_s'].astype('category')

        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings('ignore', module='statsmodels')
            model = _ols(
                f'{metric} ~ C(treatment) + C(bin_start_s) + C(treatment):C(bin_start_s)',
                data=anova_df).fit()
            anova_table = _sm.stats.anova_lm(model, typ=2)

        main_effects_frame = ttk.Frame(section_frame)
        main_effects_frame.pack(fill='x', pady=5, padx=20)

        headers = ['Source', 'df', 'Sum Sq', 'F-value', 'p-value', 'Significance']
        for i, hdr in enumerate(headers):
            ttk.Label(main_effects_frame, text=hdr,
                      font=(FONT_FAMILY, 9, 'bold'), relief='solid',
                      borderwidth=1, width=13).grid(
                row=0, column=i, sticky='ew', padx=1, pady=1)

        sources = [
            ('Treatment',       'C(treatment)'),
            ('Time',            'C(bin_start_s)'),
            ('Time×Treatment',  'C(treatment):C(bin_start_s)'),
        ]
        treatment_p = None
        time_p = None
        interaction_p = None

        for row_idx, (source_name, source_key) in enumerate(sources, start=1):
            if source_key not in anova_table.index:
                continue
            row_data = anova_table.loc[source_key]
            df_val = int(row_data['df'])
            sum_sq = row_data['sum_sq']
            f_val = row_data['F']
            p_val = row_data['PR(>F)']

            if source_name == 'Treatment':
                treatment_p = p_val
            elif source_name == 'Time':
                time_p = p_val
            else:
                interaction_p = p_val

            if p_val < 0.001:
                p_text, sig, fg = 'p < 0.001', '***', 'darkgreen'
            elif p_val < 0.01:
                p_text, sig, fg = f'p = {p_val:.4f}', '**', 'green'
            elif p_val < alpha:
                p_text, sig, fg = f'p = {p_val:.4f}', '*', 'orange'
            else:
                p_text, sig, fg = f'p = {p_val:.4f}', 'ns', 'gray'

            for col_i, cell in enumerate(
                    [source_name, str(df_val), f'{sum_sq:.2f}',
                     f'{f_val:.3f}', p_text, sig]):
                kw = {}
                if col_i == 5:
                    kw = {'foreground': fg, 'font': ('Arial', 9, 'bold')}
                ttk.Label(main_effects_frame, text=cell,
                          relief='solid', borderwidth=1, width=13,
                          **kw).grid(row=row_idx, column=col_i,
                                     sticky='ew', padx=1, pady=1)

        # Interpretation text
        interp = "Interpretation: "
        if treatment_p is not None and treatment_p < alpha:
            interp += "Treatment groups differ overall. "
        if time_p is not None and time_p < alpha:
            interp += "Metric changes over time. "
        if interaction_p is not None:
            if interaction_p < alpha:
                interp += "Groups show different time patterns (interaction significant)."
            else:
                interp += "Groups show similar time patterns (no interaction)."
        ttk.Label(section_frame, text=interp,
                  font=(FONT_FAMILY, 9, 'italic'), foreground='darkblue',
                  wraplength=700).pack(anchor='w', padx=20, pady=5)

    except Exception as e:
        ttk.Label(section_frame,
                  text=f"Two-way ANOVA failed: {e}",
                  foreground='red', font=(FONT_FAMILY, 9, 'italic')).pack(anchor='w')

    ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)

    # ── Part B: Per-Timepoint Post-hoc ──────────────────────────────
    if not _timecourse_posthoc(host):
        ttk.Label(section_frame,
                  text='Enable "Show pairwise post-hoc at each timepoint" to see per-bin results.',
                  foreground='gray', font=(FONT_FAMILY, 9, 'italic')).pack(anchor='w')
        return

    ttk.Label(section_frame,
              text="═══ Post-hoc Tests (Per Timepoint) ═══",
              font=(FONT_FAMILY, 10, 'bold'), foreground='darkblue').pack(
        anchor='w', pady=5)
    ttk.Label(section_frame,
              text="Tests if treatments differ at each individual timepoint. "
                   "Only significant results shown.",
              font=(FONT_FAMILY, 9, 'italic'), foreground='gray').pack(
        anchor='w', pady=(0, 5))

    time_bins = sorted(bins_df['bin_start_s'].dropna().unique())
    table_frame = ttk.Frame(section_frame)
    table_frame.pack(fill='x', pady=5)

    col_specs = [('Time (s)', 15), ('Test', 18), ('Statistic', 15),
                 ('p-value', 15), ('Significance', 12)]
    for col_idx, (hdr, w) in enumerate(col_specs):
        ttk.Label(table_frame, text=hdr, font=(FONT_FAMILY, 9, 'bold'),
                  relief='solid', borderwidth=1, width=w).grid(
            row=0, column=col_idx, sticky='ew', padx=1, pady=1)

    row_idx = 1
    n_sig = 0

    # Count testable bins for Bonferroni correction (2-group case)
    n_tests = 0
    for bin_s in time_bins:
        bin_df = bins_df[bins_df['bin_start_s'] == bin_s]
        _gc = []
        for treat in treatments:
            vals = bin_df[bin_df['treatment'] == treat][metric].dropna().values
            if len(vals) >= 2:
                _gc.append(vals)
        if len(_gc) >= 2:
            n_tests += 1
    n_tests = max(n_tests, 1)

    for bin_s in time_bins:
        bin_df = bins_df[bins_df['bin_start_s'] == bin_s]
        groups = []
        for treat in treatments:
            vals = bin_df[bin_df['treatment'] == treat][metric].dropna().values
            if len(vals) >= 2:
                groups.append(vals)
        if len(groups) < 2:
            continue

        if len(groups) == 2:
            t_stat, p_val = _scipy_stats.ttest_ind(groups[0], groups[1], equal_var=False)
            if np.isnan(t_stat) or np.isnan(p_val):
                continue
            p_val = min(p_val * n_tests, 1.0)  # Bonferroni
            test_name = f"Welch's t (Bonf. ×{n_tests})"
            stat_display = f't={t_stat:.3f}'
        else:
            f_stat, anova_p = _scipy_stats.f_oneway(*groups)
            if np.isnan(anova_p) or anova_p >= alpha:
                continue
            try:
                from scipy.stats import tukey_hsd
                res_hsd = tukey_hsd(*groups)
                p_val = float(res_hsd.pvalue.min())
                test_name = f'Tukey HSD ({len(groups)} groups)'
                stat_display = f'q(min)={res_hsd.statistic.min():.3f}'
            except (ImportError, AttributeError):
                min_p = 1.0
                for gi in range(len(groups)):
                    for gj in range(gi + 1, len(groups)):
                        _, pp = _scipy_stats.ttest_ind(groups[gi], groups[gj], equal_var=False)
                        if not np.isnan(pp):
                            min_p = min(min_p, pp)
                p_val = min_p
                test_name = f'Bonferroni ({len(groups)} groups)'
                stat_display = f'p(min)={min_p:.4f}'

        if np.isnan(p_val) or p_val >= alpha:
            continue

        n_sig += 1
        if p_val < 0.001:
            p_display, sig, fg = 'p < 0.001', '***', 'darkgreen'
        elif p_val < 0.01:
            p_display, sig, fg = f'p = {p_val:.4f}', '**', 'green'
        else:
            p_display, sig, fg = f'p = {p_val:.4f}', '*', 'orange'

        cells = [(f'{bin_s:.1f}', 15), (test_name, 18),
                 (stat_display, 15), (p_display, 15), (sig, 12)]
        for col_idx, (cell, w) in enumerate(cells):
            kw = {}
            if col_idx == 4:
                kw = {'foreground': fg, 'font': ('Arial', 9, 'bold')}
            ttk.Label(table_frame, text=cell, relief='solid',
                      borderwidth=1, width=w, **kw).grid(
                row=row_idx, column=col_idx, sticky='ew', padx=1, pady=1)
        row_idx += 1

    if n_sig == 0:
        ttk.Label(section_frame,
                  text='No significant differences found at any timepoint.',
                  foreground='gray', font=(FONT_FAMILY, 9, 'italic')).pack(
            anchor='w', pady=5)
    else:
        ttk.Label(section_frame,
                  text=f'Found {n_sig} significant timepoint(s) out of {len(time_bins)} bins.',
                  foreground='darkblue', font=(FONT_FAMILY, 9, 'bold')).pack(
            anchor='w', pady=5)


def create_wb_statistics_tab(host, frame, summary_df, bins_df,
                             treatments, max_time_min):
    """Build the Statistics tables (category notebook) into *frame*.

    summary_df    - per-session summary DataFrame (lowercase columns)
    bins_df       - per-bin DataFrame (bin_start_s in seconds)
    treatments    - ordered list of treatment labels
    max_time_min  - maximum time in minutes (from bins_df)
    """
    max_time_int = max(1, int(max_time_min))

    # ── Control bar (shared across sub-tabs) ─────────────────────────
    ctrl_frame = ttk.Frame(frame)
    ctrl_frame.pack(fill='x', padx=10, pady=(8, 4))

    ttk.Label(ctrl_frame, text="Statistics time window:").pack(side='left')
    stats_time_var = tk.IntVar(value=max_time_int)
    ttk.Spinbox(ctrl_frame, from_=1, to=max_time_int,
                textvariable=stats_time_var, width=8).pack(side='left', padx=5)
    ttk.Label(ctrl_frame, text="min").pack(side='left')

    status_lbl = ttk.Label(ctrl_frame,
                           text=f"(showing 0-{max_time_int} min)",
                           font=(FONT_FAMILY, 9), foreground='gray')
    status_lbl.pack(side='left', padx=10)

    # ── Category notebook for sub-tabs ────────────────────────────────
    stats_nb = ttk.Notebook(frame)
    stats_nb.pack(fill='both', expand=True)

    # Metrics grouped by category. Fore metrics are included when the data
    # carries them (the old tab keyed this off its use_fore toggle; fore
    # columns only exist when fore analysis ran, so data-presence is the
    # same gate without Tk state).
    has_fore = any(
        c in summary_df.columns and summary_df[c].notna().any()
        for c in ('WBI_fore', 'SI_fore', 'SBI_fore'))
    hind_metrics = ['WBI_hind', 'SI_hind', 'SBI_hind']
    fore_metrics = (['WBI_fore', 'SI_fore', 'SBI_fore'] if has_fore else [])
    contact_metrics = [f'contact_pct_{r}' for r in ROLES]
    brightness_metrics = ['brightness_HL', 'brightness_HR',
                          'brightness_ratio_HL_HR']
    gait_metrics = [
        'stance_dur_HL', 'stance_dur_HR', 'swing_dur_HL', 'swing_dur_HR',
        'stride_dur_HL', 'stride_dur_HR', 'cadence_HL', 'cadence_HR',
        'duty_cycle_HL', 'duty_cycle_HR',
        'stride_len_HL', 'stride_len_HR',
        'step_len_hind', 'step_width_hind',
        'swing_speed_HL', 'swing_speed_HR',
        'stride_cv_HL', 'stride_cv_HR',
        'stance_SI_hind', 'stride_len_SI_hind',
    ]
    loco_metrics = ['total_distance', 'loco_total_distance',
                    'time_moving_s', 'time_moving_pct',
                    'body_speed_mean', 'body_speed_loco']
    phase_metrics = ['phase_HL_HR', 'phase_diagonal',
                     'phase_FL_FR', 'phase_HL_FL', 'phase_HR_FR', 'phase_HL_FR']
    coordination_metrics = ['regularity_index',
                            'print_position_L', 'print_position_R',
                            'support_0paw_pct', 'support_1paw_pct',
                            'support_2paw_pct', 'support_3paw_pct',
                            'support_4paw_pct', 'quad_stance_pct',
                            'analyzed_pct']
    contour_metrics = [f'paw_area_{r}' for r in ('HL', 'HR')]
    contour_metrics += [f'paw_spread_{r}' for r in ('HL', 'HR')]
    contour_metrics += ['paw_area_ratio_hind']

    # Define sub-tab categories
    _stats_categories = [
        ('Limb Use', hind_metrics + fore_metrics + contact_metrics + brightness_metrics),
        ('Gait',           gait_metrics),
        ('Movement',       loco_metrics + phase_metrics + coordination_metrics),
        ('Paw Contour',    contour_metrics),
    ]

    # Content holders per sub-tab (for rebuild on recalculate)
    _cat_holders = {}

    def _build_category_content(holder, cat_metrics, s_df, b_df):
        """Build scrollable statistics content for one category."""
        for w in holder.winfo_children():
            w.destroy()

        canvas = tk.Canvas(holder, bg='white')
        sb_inner = ttk.Scrollbar(holder, orient='vertical',
                                 command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=sb_inner.set)

        sec = 1
        for metric in cat_metrics:
            if s_df is None or metric not in s_df.columns:
                continue
            if s_df[metric].dropna().empty:
                continue
            add_wb_stats_section(
                host, scrollable_frame,
                f"{sec}. {metric.replace('_', ' ')} - Summary",
                s_df, metric, treatments, 'value')
            sec += 1

        if b_df is not None and not b_df.empty:
            for metric in cat_metrics:
                if metric not in b_df.columns:
                    continue
                if b_df[metric].dropna().empty:
                    continue
                add_wb_timecourse_stats_section(
                    host, scrollable_frame, b_df, metric, treatments)
                sec += 1

        canvas.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        sb_inner.pack(side='right', fill='y')

        export_frame = ttk.Frame(holder)
        export_frame.pack(side='bottom', fill='x', padx=10, pady=5)
        stats_data = {'summary_df': s_df, 'bins_df': b_df,
                      'treatments': treatments, 'metrics': cat_metrics}
        ttk.Button(export_frame, text="📊 Export Statistics CSV",
                   command=lambda sd=stats_data: export_wb_statistics(host, sd)
                   ).pack()

    # Create sub-tabs
    for cat_name, cat_metrics in _stats_categories:
        # Only add sub-tab if at least one metric has data
        has_data = any(
            m in summary_df.columns and summary_df[m].notna().any()
            for m in cat_metrics
        )
        if not has_data:
            continue
        cat_frame = ttk.Frame(stats_nb)
        stats_nb.add(cat_frame, text=cat_name)
        holder = ttk.Frame(cat_frame)
        holder.pack(fill='both', expand=True)
        _cat_holders[cat_name] = (holder, cat_metrics)
        _build_category_content(holder, cat_metrics, summary_df, bins_df)

    if not _cat_holders:
        ttk.Label(frame, text='No metrics with data to summarize.',
                  font=(FONT_FAMILY, 10, 'italic')).pack(padx=10, pady=10)
        return

    def on_recalculate():
        win_min = stats_time_var.get()
        win_sec = win_min * 60.0
        filtered_bins = (bins_df[bins_df['bin_start_s'] <= win_sec].copy()
                         if bins_df is not None and not bins_df.empty
                         else bins_df)
        status_lbl.config(text=f"(showing 0-{win_min} min)")
        for cat_name, (holder, cat_metrics) in _cat_holders.items():
            _build_category_content(holder, cat_metrics, summary_df, filtered_bins)

    ttk.Button(ctrl_frame, text="↺ Recalculate",
               command=on_recalculate).pack(side='left')


def wb_statistics_frame(host, stats_data) -> pd.DataFrame:
    """Headless core of the statistics CSV export (same rows as the old
    _export_wb_statistics wrote)."""
    s_df = stats_data.get('summary_df')
    treatments = stats_data.get('treatments', [])
    metrics = stats_data.get('metrics', [])

    rows = []
    blank = {'Section': '', 'Treatment': '', 'N': '', 'Mean': '', 'SD': '',
             'SEM': '', 'Min': '', 'Max': '', 'Test': '', 'p_value': '',
             'Significance': ''}

    for metric in metrics:
        if s_df is None or metric not in s_df.columns:
            continue
        rows.append({**blank, 'Section': f'Summary - {metric}'})

        for treat in treatments:
            vals = s_df[s_df['treatment'] == treat][metric].dropna().values
            if len(vals) == 0:
                continue
            n = len(vals)
            sd = np.std(vals, ddof=1) if n > 1 else 0.0
            sem = sd / np.sqrt(n)
            rows.append({
                'Section':      f'Summary - {metric}',
                'Treatment':    treat,
                'N':            n,
                'Mean':         f'{np.mean(vals):.4f}',
                'SD':           f'{sd:.4f}',
                'SEM':          f'{sem:.4f}',
                'Min':          f'{np.min(vals):.4f}',
                'Max':          f'{np.max(vals):.4f}',
                'Test':         '',
                'p_value':      '',
                'Significance': '',
            })

        # Statistical test row
        data_by_t = {t: s_df[s_df['treatment'] == t][metric].dropna().values
                     for t in treatments}
        res = perform_wb_statistical_test(host, data_by_t, treatments)
        if res:
            p = res['p_value']
            a = res['alpha']
            sig = ('***' if p < 0.001 else
                   '**'  if p < 0.01  else
                   '*'   if p < a     else 'ns')
            rows.append({**blank,
                         'Section':      f'Test - {metric}',
                         'Test':         res['test_type'],
                         'p_value':      f'{p:.4f}',
                         'Significance': sig})

        rows.append(blank.copy())

    return pd.DataFrame(rows)


def export_wb_statistics(host, stats_data, path=None, parent=None):
    """Export limb use statistics summary to CSV. When *path* is None a
    save dialog is shown (the old flow); pass a path for headless use."""
    if path is None:
        analysis_dir = (host.cfg or {}).get('analysis_dir', '')
        if analysis_dir:
            os.makedirs(analysis_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = filedialog.asksaveasfilename(
            title="Save Statistics CSV",
            initialdir=analysis_dir or None,
            initialfile=f'wb_statistics_{ts}.csv',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv')],
            parent=parent)
        if not path:
            return None
    wb_statistics_frame(host, stats_data).to_csv(path, index=False)
    host.log(f"Statistics saved: {os.path.basename(path)}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# Σ Stats flip for a single entry
# ─────────────────────────────────────────────────────────────────────────────

def _render_metric_stats(host, frame, entry):
    """Statistics table for one registered metric: per-treatment
    descriptives of the metric column (per-session values; binned data
    is averaged per session first) plus a group test."""
    from tkinter import scrolledtext as _stext
    txt = _stext.ScrolledText(frame, wrap='none', font=('Consolas', 9))
    txt.pack(fill='both', expand=True, padx=6, pady=4)

    def _put(msg):
        txt.insert('end', msg + "\n")

    df, col = _entry_data(host, entry)
    if df is None or col is None or col not in getattr(df, 'columns', []):
        _put("No tabular data behind this graph (custom rendering) "
             "- statistics unavailable here.")
        txt.configure(state='disabled')
        return
    d = df.copy()
    if 'treatment' not in d.columns:
        _put("No treatment column - run with a key file for group "
             "statistics.")
        _put("")
        _put(str(d[col].describe()))
        txt.configure(state='disabled')
        return
    # per-session value (bins average to one value per session)
    if 'session' in d.columns:
        per = (d.groupby(['treatment', 'session'])[col]
               .mean().reset_index())
    else:
        per = d[['treatment', col]].rename(columns={col: col}).copy()
        per['session'] = np.arange(len(per))
    groups = [t for t in per['treatment'].dropna().unique()]
    cfg = host.cfg or {}
    err_t = cfg.get('error_type', 'SEM')
    _put(f"{entry.get('display_name', col)}")
    _put("=" * max(len(str(entry.get('display_name', col))), 8))
    _put("")
    samples = []
    for g in groups:
        v = per.loc[per['treatment'] == g, col].to_numpy(float)
        v = v[np.isfinite(v)]
        samples.append(v)
        e = _calc_error(v, err_t)
        _put(f"  {str(g):<18} mean {np.mean(v):.4g}  ± {e:.4g} "
             f"({_error_label(err_t).split(' ')[-1]})   n={len(v)}  "
             f"median {np.median(v):.4g}" if len(v) else
             f"  {str(g):<18} no data")
    _put("")
    samples = [v for v in samples if len(v) >= 2]
    if len(samples) >= 2:
        try:
            from scipy import stats as _st
            if len(samples) == 2:
                tt = _st.ttest_ind(*samples, equal_var=False)
                mw = _st.mannwhitneyu(*samples, alternative='two-sided')
                _put(f"  Welch t-test:    t={tt[0]:.3g}  p={tt[1]:.4g}")
                _put(f"  Mann-Whitney U:  U={mw[0]:.3g}  p={mw[1]:.4g}")
            else:
                an = _st.f_oneway(*samples)
                kw = _st.kruskal(*samples)
                _put(f"  one-way ANOVA:   F={an[0]:.3g}  p={an[1]:.4g}")
                _put(f"  Kruskal-Wallis:  H={kw[0]:.3g}  p={kw[1]:.4g}")
        except Exception as e:
            _put(f"  test unavailable: {e}")
    else:
        _put("  not enough groups with n ≥ 2 for a test")
    # (fixed vs old code: binned frames carry bin_start_s, not bin_start_min)
    if 'session' in df.columns and 'bin_start_s' in df.columns:
        _put("")
        _put("  (binned data averaged to one value per session first)")
    txt.configure(state='disabled')


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def _make_entry(host, display_name, graph_type, data, column,
                reference=None, y_label='', description='',
                flip=False, create_fn=None):
    """Build a registry entry. Mirrors the old _register_metric data plus a
    deferred `build(frame)` callable (figures are only created on render)."""
    entry = {
        'display_name': display_name,
        'graph_type': graph_type,
        'data': data,
        'column': column,
        'reference': reference,
        'y_label': y_label,
        'description': description,
        'flip': flip,
        'create_fn': create_fn,
        'has_stats': data is not None and column is not None,
    }

    def _build(frame, _e=entry):
        if _e.get('create_fn'):
            _e['create_fn'](frame)
            return
        df, col = _entry_data(host, _e)
        if df is None or col is None or col not in getattr(df, 'columns', []):
            ttk.Label(frame, text=f'No data for {_e["display_name"]}.',
                      font=(FONT_FAMILY, 10, 'italic')).pack(padx=10, pady=10)
            return
        gtype = _e['graph_type']
        if gtype == 'bar':
            build_bar_graph(host, frame, df, col, _e['display_name'],
                            _e.get('reference'), _e.get('y_label', ''))
        elif gtype == 'box':
            build_box_graph(host, frame, df, col, _e['display_name'],
                            _e.get('reference'), _e.get('y_label', ''))
        elif gtype == 'violin':
            build_violin_graph(host, frame, df, col, _e['display_name'],
                               _e.get('reference'), _e.get('y_label', ''))
        elif gtype == 'timecourse':
            build_timecourse_graph(host, frame, df, col, _e['display_name'],
                                   _e.get('reference'), _e.get('y_label', ''))
        else:
            ttk.Label(frame, text=f'Unknown graph type: {gtype}',
                      font=(FONT_FAMILY, 10, 'italic')).pack(padx=10, pady=10)

    entry['build'] = _build
    return entry


def build_registry(host) -> 'OrderedDict[str, list]':
    """Build the full {category: [entry, ...]} registry from host data.

    Mirrors every reachable registration in gait_limb_tab._open_graphs
    (:7100-:7952), flattened from the old group→category→metric notebooks to
    one category level driven by two comboboxes. Column-presence gates,
    display names, references, y-labels, and descriptions are verbatim.
    """
    df = host.summary_df
    bdf = (host.bins_df
           if host.bins_df is not None and not host.bins_df.empty
           else None)
    graph_cfg = host.cfg or {}
    _sets = graph_cfg.get('graph_sets', {})
    show_wb = _sets.get('weight_bearing', True)
    show_gait = _sets.get('gait', True)
    show_movement = _sets.get('movement', True)
    show_contour = _sets.get('paw_contour', True)
    show_stats = _sets.get('statistics', True)

    inj, contra = injured_contra(host)

    registry = OrderedDict()

    def _cat(name):
        return registry.setdefault(name, [])

    def _reg(cat_entries, display_name, graph_type, data, column,
             reference=None, y_label='', description='',
             flip=False, create_fn=None):
        cat_entries.append(_make_entry(
            host, display_name, graph_type, data, column,
            reference=reference, y_label=y_label, description=description,
            flip=flip, create_fn=create_fn))

    # ── Paw Contact headline (injured/contralateral ratios first) ────────
    _hl_area = 'paw_area_ratio_hind'
    _hl_int = 'contact_intensity_ratio_hind'
    if show_contour and any(
            rk in df.columns and df[rk].notna().any()
            for rk in (_hl_area, _hl_int)):
        cat = _cat('Paw Contact')
        for rk, nice in ((_hl_area, 'Paw Area Ratio'),
                         (_hl_int, 'Intensity Ratio')):
            ylbl = f'{nice} {inj}/{contra}'
            desc = (f'{nice.lower().capitalize()}, injured/contralateral '
                    f'({inj}/{contra}). Below 1.0 = injured paw '
                    'bears less. Contour-area gate, all frames.')
            if rk in df.columns and df[rk].notna().any():
                _reg(cat, nice, 'bar', df, rk,
                     reference=1.0, y_label=ylbl, description=desc,
                     flip=True)
            if (bdf is not None and rk in bdf.columns
                    and bdf[rk].notna().any()):
                _reg(cat, f'{nice} - TC', 'timecourse', bdf, rk,
                     reference=1.0, y_label=ylbl,
                     description=f'{desc} Across time bins.', flip=True)

    # ── Limb Use - Hind ───────────────────────────────────────────────────
    if show_wb:
        cat = _cat('Limb Use - Hind')
        if 'WBI_hind' in df.columns:
            _reg(cat, 'WBI Hind', 'bar', df, 'WBI_hind',
                 reference=50.0, y_label='Weight Bearing Index - hind (%)',
                 description='HL / (HL+HR) × 100.  Reference 50 = symmetric.  >50 = more stance on left hind.')
        if 'SI_hind' in df.columns:
            _reg(cat, 'SI Hind (Box)', 'box', df, 'SI_hind',
                 reference=0.0, y_label='Symmetry Index - hind (%)',
                 description='(HL−HR) / (HL+HR) × 100.  Reference 0 = symmetric.  Positive = left bias.')
            _reg(cat, 'SI Hind (Violin)', 'violin', df, 'SI_hind',
                 reference=0.0, y_label='Symmetry Index - hind (%)',
                 description='(HL−HR) / (HL+HR) × 100.  Reference 0 = symmetric.  Positive = left bias.')
        if 'SBI_hind' in df.columns and df['SBI_hind'].notna().any():
            _reg(cat, 'SBI Hind', 'bar', df, 'SBI_hind',
                 reference=0.0, y_label='Symmetry Balance Index - hind (%)',
                 description='2 × |HL−HR| / (HL+HR) × 100.  Always ≥ 0.  0 = perfect symmetry.')
        if bdf is not None:
            for col, lbl, ref, ylbl, desc in [
                ('SI_hind', 'SI Hind - TC', None,
                 'Symmetry Index - hind (%)',
                 'SI hind across time bins (mean ± SEM).  Dashed line = 0 (symmetric).'),
                ('WBI_hind', 'WBI Hind - TC', 50.0,
                 'Weight Bearing Index - hind (%)',
                 'WBI hind across time bins. Reference 50 = symmetric.'),
                ('SBI_hind', 'SBI Hind - TC', 0.0,
                 'Symmetry Balance Index - hind (%)',
                 'SBI hind across time bins. 0 = perfect symmetry.'),
            ]:
                if col in bdf.columns and bdf[col].notna().any():
                    _reg(cat, lbl, 'timecourse', bdf, col,
                         reference=ref, y_label=ylbl, description=desc)
        if not cat:
            del registry['Limb Use - Hind']

    # ── Limb Use - Fore (only when any fore data) ─────────────────────────
    has_fore = (
        ('WBI_fore' in df.columns and df['WBI_fore'].notna().any()) or
        ('SI_fore' in df.columns and df['SI_fore'].notna().any()) or
        ('SBI_fore' in df.columns and df['SBI_fore'].notna().any())
    )
    if show_wb and has_fore:
        cat = _cat('Limb Use - Fore')
        if 'WBI_fore' in df.columns and df['WBI_fore'].notna().any():
            _reg(cat, 'WBI Fore', 'bar', df, 'WBI_fore',
                 reference=50.0, y_label='Weight Bearing Index - fore (%)',
                 description='FL / (FL+FR) × 100.  Reference 50 = symmetric.')
        if 'SI_fore' in df.columns and df['SI_fore'].notna().any():
            _reg(cat, 'SI Fore (Box)', 'box', df, 'SI_fore',
                 reference=0.0, y_label='Symmetry Index - fore (%)',
                 description='(FL−FR) / (FL+FR) × 100.  Reference 0.')
            _reg(cat, 'SI Fore (Violin)', 'violin', df, 'SI_fore',
                 reference=0.0, y_label='Symmetry Index - fore (%)')
        if 'SBI_fore' in df.columns and df['SBI_fore'].notna().any():
            _reg(cat, 'SBI Fore', 'bar', df, 'SBI_fore',
                 reference=0.0, y_label='Symmetry Balance Index - fore (%)',
                 description='2 × |FL−FR| / (FL+FR) × 100.  Always ≥ 0.')
        if bdf is not None:
            for col, lbl, ref, ylbl, desc in [
                ('WBI_fore', 'WBI Fore - TC', 50.0,
                 'Weight Bearing Index - fore (%)',
                 'WBI fore across time bins. Reference 50 = symmetric.'),
                ('SI_fore', 'SI Fore - TC', 0.0,
                 'Symmetry Index - fore (%)',
                 'SI fore across time bins. Reference 0 = symmetric.'),
                ('SBI_fore', 'SBI Fore - TC', 0.0,
                 'Symmetry Balance Index - fore (%)',
                 'SBI fore across time bins. 0 = perfect symmetry.'),
            ]:
                if col in bdf.columns and bdf[col].notna().any():
                    _reg(cat, lbl, 'timecourse', bdf, col,
                         reference=ref, y_label=ylbl, description=desc)
        if not cat:
            del registry['Limb Use - Fore']

    # ── Contact % / Brightness (old Limb Use → Contact % + Brightness) ────
    if show_wb:
        cat = _cat('Contact % / Brightness')
        _reg(cat, 'Contact % (All Paws)', 'custom', None, None,
             description='Percentage of frames each paw is in contact with the surface.',
             create_fn=lambda f, _df=df: add_paw_contact_bar_tab(host, f, _df))
        for role in ROLES:
            col = f'contact_pct_{role}'
            if (bdf is not None and col in bdf.columns
                    and 'bin_start_s' in bdf.columns):
                _reg(cat, f'Contact % {role}', 'timecourse',
                     bdf, col, reference=None,
                     y_label=f'Contact % - {role}',
                     description=f'{role} paw contact % across time bins.')
        if (bdf is not None and 'hind_fore_ratio' in bdf.columns
                and bdf['hind_fore_ratio'].notna().any()):
            _reg(cat, 'Hind/Fore Ratio - TC', 'timecourse',
                 bdf, 'hind_fore_ratio', reference=None,
                 y_label='Mean hind / mean fore contact %',
                 description='Ratio of mean hind to mean fore contact % across time bins.')
        if ('brightness_ratio_HL_HR' in df.columns
                and df['brightness_ratio_HL_HR'].notna().any()):
            _reg(cat, 'Brightness Ratio', 'bar', df,
                 'brightness_ratio_HL_HR', reference=1.0,
                 y_label='HL / HR brightness (contact frames)',
                 description='HL/HR brightness ratio. Reference 1.0 = equal.')
        if bdf is not None:
            for col, lbl, ref, ylbl, desc in [
                ('brightness_HL', 'Brightness HL - TC', None,
                 'Mean brightness - HL',
                 'Mean brightness in HL ROI during contact frames.'),
                ('brightness_HR', 'Brightness HR - TC', None,
                 'Mean brightness - HR',
                 'Mean brightness in HR ROI during contact frames.'),
                ('brightness_FL', 'Brightness FL - TC', None,
                 'Mean brightness - FL',
                 'Mean brightness in FL ROI during contact frames.'),
                ('brightness_FR', 'Brightness FR - TC', None,
                 'Mean brightness - FR',
                 'Mean brightness in FR ROI during contact frames.'),
                ('brightness_ratio_HL_HR', 'Brightness Ratio - TC', 1.0,
                 'HL / HR brightness',
                 'HL/HR brightness ratio across time bins. Reference 1.0 = equal.'),
            ]:
                if col in bdf.columns and bdf[col].notna().any():
                    _reg(cat, lbl, 'timecourse', bdf, col,
                         reference=ref, y_label=ylbl, description=desc)
        if not cat:
            del registry['Contact % / Brightness']

    # ── Gait ──────────────────────────────────────────────────────────────
    has_gait_timing = any(f'stance_dur_{r}' in df.columns and df[f'stance_dur_{r}'].notna().any()
                          for r in ROLES)
    has_gait_spatial = any(f'stride_len_{r}' in df.columns and df[f'stride_len_{r}'].notna().any()
                           for r in ROLES)
    has_gait_sym = ('stance_SI_hind' in df.columns and df['stance_SI_hind'].notna().any()) or \
                   ('stride_len_SI_hind' in df.columns and df['stride_len_SI_hind'].notna().any())

    if show_gait and has_gait_timing:
        cat = _cat('Gait - Timing')
        for role in ROLES:
            for col, lbl, ref, ylbl, desc in [
                (f'stance_dur_{role}', f'Stance Dur {role}', None,
                 f'Stance duration (s) - {role}',
                 f'Mean stance duration (s) for {role}.'),
                (f'swing_dur_{role}', f'Swing Dur {role}', None,
                 f'Swing duration (s) - {role}',
                 f'Mean swing duration (s) for {role}.'),
                (f'duty_cycle_{role}', f'Duty Cycle {role}', None,
                 f'Duty cycle (%) - {role}',
                 f'Duty cycle (%) for {role}. >50% = more stance than swing.'),
            ]:
                if col in df.columns and df[col].notna().any():
                    _reg(cat, lbl, 'bar', df, col,
                         reference=ref, y_label=ylbl, description=desc)
            for col, lbl, ylbl, desc in [
                (f'swing_speed_{role}', f'Swing Speed {role}',
                 f'Swing speed (px/s) - {role}',
                 f'Mean swing speed for {role}. Injured animals swing slower on affected side.'),
            ]:
                if col in df.columns and df[col].notna().any():
                    _reg(cat, lbl, 'bar', df, col,
                         reference=None, y_label=ylbl, description=desc)
        if bdf is not None:
            for role in ROLES:
                for col, lbl, ylbl, desc in [
                    (f'stance_dur_{role}', f'Stance {role} - TC',
                     f'Stance dur (s) - {role}',
                     f'Stance duration for {role} across time bins.'),
                    (f'duty_cycle_{role}', f'Duty {role} - TC',
                     f'Duty cycle (%) - {role}',
                     f'Duty cycle for {role} across time bins.'),
                    (f'cadence_{role}', f'Cadence {role} - TC',
                     f'Cadence (strides/min) - {role}',
                     f'Stride cadence for {role} across time bins.'),
                    (f'swing_speed_{role}', f'Swing Spd {role} - TC',
                     f'Swing speed (px/s) - {role}',
                     f'Swing speed for {role} across time bins.'),
                ]:
                    if col in bdf.columns and bdf[col].notna().any():
                        _reg(cat, lbl, 'timecourse', bdf, col,
                             reference=None, y_label=ylbl, description=desc)
        if not cat:
            del registry['Gait - Timing']

    if show_gait and has_gait_spatial:
        cat = _cat('Gait - Spatial')
        for role in ROLES:
            col = f'stride_len_{role}'
            if col in df.columns and df[col].notna().any():
                _reg(cat, f'Stride Len {role}', 'bar', df, col,
                     reference=None,
                     y_label=f'Stride length (px) - {role}',
                     description=f'Mean stride length (px) for {role}.')
        for pair in ['hind', 'fore']:
            for col, lbl, ylbl in [
                (f'step_len_{pair}', f'Step Len {pair}',
                 f'Step length (px) - {pair}'),
                (f'step_width_{pair}', f'Step Width {pair}',
                 f'Step width (px) - {pair}'),
            ]:
                if col in df.columns and df[col].notna().any():
                    _reg(cat, lbl, 'bar', df, col,
                         reference=None, y_label=ylbl, description=ylbl)
        if bdf is not None:
            for role in ROLES:
                col = f'stride_len_{role}'
                if col in bdf.columns and bdf[col].notna().any():
                    _reg(cat, f'Stride Len {role} - TC', 'timecourse',
                         bdf, col, reference=None,
                         y_label=f'Stride length (px) - {role}',
                         description=f'Stride length for {role} across time bins.')
        if not cat:
            del registry['Gait - Spatial']

    if show_gait and has_gait_sym:
        cat = _cat('Gait - Symmetry')
        for col, lbl, ref, ylbl in [
            ('stance_SI_hind', 'Stance SI Hind', 0.0,
             'Stance Symmetry Index (%)'),
            ('stride_len_SI_hind', 'Stride Len SI Hind', 0.0,
             'Stride Length Symmetry Index (%)'),
        ]:
            if col in df.columns and df[col].notna().any():
                _reg(cat, lbl, 'bar', df, col,
                     reference=ref, y_label=ylbl,
                     description=f'{ylbl}. Reference 0 = symmetric.')
                _reg(cat, f'{lbl} (Box)', 'box', df, col,
                     reference=ref, y_label=ylbl,
                     description=f'{ylbl} - box plot distribution.')
        for role in ROLES:
            col = f'stride_cv_{role}'
            if col in df.columns and df[col].notna().any():
                _reg(cat, f'Stride CV {role}', 'bar', df, col,
                     reference=None,
                     y_label=f'Stride CV (%) - {role}',
                     description=f'Stride-to-stride CV for {role}. Higher = less rhythmic.')
        for col, lbl, ylbl in [
            ('phase_HL_HR', 'Phase HL-HR',
             'HL-HR phase (0.5 = alternating)'),
            ('phase_diagonal', 'Phase Diagonal',
             'HR-FL diagonal phase'),
        ]:
            if col in df.columns and df[col].notna().any():
                _reg(cat, lbl, 'bar', df, col,
                     reference=0.5, y_label=ylbl,
                     description=f'{ylbl}. Reference 0.5 = alternating.')
        if bdf is not None:
            for col, lbl, ylbl, desc in [
                ('stance_SI_hind', 'Stance SI - TC',
                 'Stance SI (%)',
                 'Stance Symmetry Index across time bins.'),
                ('stride_len_SI_hind', 'Stride Len SI - TC',
                 'Stride Len SI (%)',
                 'Stride Length SI across time bins.'),
            ]:
                if col in bdf.columns and bdf[col].notna().any():
                    _reg(cat, lbl, 'timecourse', bdf, col,
                         reference=0.0, y_label=ylbl, description=desc)
        if not cat:
            del registry['Gait - Symmetry']

    # ── Movement ──────────────────────────────────────────────────────────
    has_loco = ('total_distance' in df.columns and df['total_distance'].notna().any())
    has_coord = any(c in df.columns and df[c].notna().any()
                    for c in ['regularity_index', 'print_position_L',
                              'support_0paw_pct', 'phase_FL_FR'])

    if show_movement and has_loco:
        cat = _cat('Movement')
        for col, lbl, ylbl, desc in [
            ('total_distance', 'Total Distance', 'Distance (px)',
             'Total body displacement across the session.'),
            ('loco_total_distance', 'Distance (Moving)',
             'Distance (px) - locomotion only',
             'Displacement during locomotion bouts only.'),
            ('time_moving_s', 'Time Moving', 'Time moving (s)',
             'Total time in locomotion (seconds).'),
            ('time_moving_pct', 'Time Moving %',
             'Time in locomotion (%)',
             'Percentage of session spent in locomotion.'),
            ('body_speed_mean', 'Body Speed Mean',
             'Body speed (px/s)',
             'Mean body centroid speed across all frames.'),
            ('body_speed_loco', 'Body Speed (Loco)',
             'Body speed (px/s) - locomotion',
             'Mean body speed during locomotion bouts only.'),
        ]:
            if col in df.columns and df[col].notna().any():
                _reg(cat, lbl, 'bar', df, col,
                     y_label=ylbl, description=desc)
        if bdf is not None:
            for col, lbl, ylbl, desc in [
                ('total_distance', 'Distance - TC',
                 'Distance (px)',
                 'Total body displacement per time bin.'),
                ('loco_total_distance', 'Distance (Moving) - TC',
                 'Distance (px) - locomotion',
                 'Displacement during locomotion per time bin.'),
                ('time_moving_s', 'Time Moving - TC',
                 'Time moving (s)',
                 'Time in locomotion per time bin.'),
                ('time_moving_pct', 'Time Moving % - TC',
                 'Time in locomotion (%)',
                 'Percentage of each time bin in locomotion.'),
                ('body_speed_mean', 'Body Speed - TC',
                 'Body speed (px/s)',
                 'Mean body speed per time bin.'),
                ('body_speed_loco', 'Body Speed Loco - TC',
                 'Body speed (px/s) - loco',
                 'Body speed during locomotion per time bin.'),
            ]:
                if col in bdf.columns and bdf[col].notna().any():
                    _reg(cat, lbl, 'timecourse', bdf, col,
                         y_label=ylbl, description=desc)
        if not cat:
            del registry['Movement']

    if show_movement and has_coord:
        cat = _cat('Coordination')
        if ('regularity_index' in df.columns
                and df['regularity_index'].notna().any()):
            _reg(cat, 'Regularity Index', 'bar', df,
                 'regularity_index', reference=100.0,
                 y_label='Regularity Index (%)',
                 description='% of steps following normal 4-paw pattern. 100% = perfect.')
        for side in ['L', 'R']:
            col = f'print_position_{side}'
            if col in df.columns and df[col].notna().any():
                _reg(cat, f'Print Position {side}', 'bar',
                     df, col, reference=None,
                     y_label=f'Print position (px) - {side}',
                     description=f'Hind-fore paw overlap distance ({side}). Smaller = better.')
        support_cols = [f'support_{n}paw_pct' for n in range(5)]
        if all(c in df.columns for c in support_cols[:3]):
            _reg(cat, 'Support Patterns', 'custom', None, None,
                 description='Distribution of paw support during locomotion.',
                 create_fn=lambda f, _df=df: add_support_pattern_tab(host, f, _df))
        for col, lbl, ylbl in [
            ('phase_HL_HR', 'Phase HL-HR', 'HL-HR phase'),
            ('phase_diagonal', 'Phase Diagonal',
             'HR-FL diagonal phase'),
            ('phase_FL_FR', 'Phase FL-FR', 'FL-FR phase'),
            ('phase_HL_FL', 'Phase HL-FL', 'HL-FL phase'),
            ('phase_HR_FR', 'Phase HR-FR', 'HR-FR phase'),
            ('phase_HL_FR', 'Phase HL-FR', 'HL-FR phase'),
        ]:
            if col in df.columns and df[col].notna().any():
                _reg(cat, lbl, 'bar', df, col,
                     reference=0.5, y_label=ylbl,
                     description=f'{ylbl}. 0.5 = alternating (normal).')
        if not cat:
            del registry['Coordination']

    # ── Paw Contour ───────────────────────────────────────────────────────
    _contour_metrics = [
        ('paw_area',          'Area',         'Paw area (px²) - {}'),
        ('paw_spread',        'Spread',       'Paw spread (px) - {}'),
        ('contact_intensity', 'Intensity',    'Contact intensity - {}'),
        ('paw_width',         'Width',        'Paw width (px) - {}'),
        ('paw_solidity',      'Solidity',     'Paw solidity - {}'),
        ('paw_aspect_ratio',  'Aspect Ratio', 'Aspect ratio - {}'),
        ('paw_circularity',   'Circularity',  'Circularity - {}'),
    ]
    has_contour = any(
        f'{mk}_{role}' in df.columns and df[f'{mk}_{role}'].notna().any()
        for mk, _, _ in _contour_metrics for role in ROLES
    ) or ('paw_area_ratio_hind' in df.columns and df['paw_area_ratio_hind'].notna().any()) or ('contact_intensity_ratio_hind' in df.columns and df['contact_intensity_ratio_hind'].notna().any())
    _contour_stance_metrics = [
        ('paw_area_stance',          'Area',         'Paw area (px²) - {}'),
        ('paw_spread_stance',        'Spread',       'Paw spread (px) - {}'),
        ('contact_intensity_stance', 'Intensity',    'Contact intensity - {}'),
        ('paw_width_stance',         'Width',        'Paw width (px) - {}'),
        ('paw_solidity_stance',      'Solidity',     'Paw solidity - {}'),
        ('paw_aspect_ratio_stance',  'Aspect Ratio', 'Aspect ratio - {}'),
        ('paw_circularity_stance',   'Circularity',  'Circularity - {}'),
    ]
    has_stance_contour = any(
        f'{mk}_{role}' in df.columns and df[f'{mk}_{role}'].notna().any()
        for mk, _, _ in _contour_stance_metrics for role in ROLES
    )
    _contour_pawlike_metrics = [
        ('pawlike_area',          'Area',         'Paw area (px²) - {}'),
        ('pawlike_spread',        'Spread',       'Paw spread (px) - {}'),
        ('pawlike_intensity',     'Intensity',    'Contact intensity - {}'),
        ('pawlike_width',         'Width',        'Paw width (px) - {}'),
        ('pawlike_solidity',      'Solidity',     'Paw solidity - {}'),
        ('pawlike_aspect_ratio',  'Aspect Ratio', 'Aspect ratio - {}'),
        ('pawlike_circularity',   'Circularity',  'Circularity - {}'),
    ]
    has_pawlike_contour = any(
        f'{mk}_{role}' in df.columns and df[f'{mk}_{role}'].notna().any()
        for mk, _, _ in _contour_pawlike_metrics for role in ROLES
    )

    # Rich per-metric descriptions (from the old window's description bar)
    _contour_descs = {
        'Area':         ('Mean paw contour area (px²) during contact frames. '
                         'Larger area indicates greater paw-surface contact, which may '
                         'reflect normal weight-bearing.'),
        'Spread':       ('Maximum dimension (px) of paw contour bounding box. '
                         'Larger spread indicates more toe-spreading or a flatter paw placement.'),
        'Intensity':    ('Mean pixel brightness within paw contour shape during contact. '
                         'Higher intensity indicates stronger paw-surface contact signal.'),
        'Width':        ('Minimum dimension (px) of paw contour bounding box. '
                         'Represents the narrower axis of the paw print.'),
        'Solidity':     ('Solidity of paw contour - ratio of contour area to convex hull area. '
                         'Values near 1.0 indicate a solid, compact paw print; lower values suggest '
                         'irregular or fragmented contact.'),
        'Aspect Ratio': ('Aspect ratio of paw contour bounding box (max/min dimension). '
                         'Higher values indicate an elongated paw print; values near 1.0 indicate '
                         'a round print.'),
        'Circularity':  ('Circularity of paw contour - 4π×area/perimeter². '
                         'Values near 1.0 indicate a circular shape; lower values indicate '
                         'irregular or elongated contours.'),
    }
    _shape_desc = ('Mean paw contour outline averaged across contact frames. '
                   'Normalized by contour area for size-independent shape comparison. '
                   '±1 SD envelope shown as shaded region.')
    _filter_preview_desc = (
        'Interactive preview of the paw-like contour filter. '
        'Adjust solidity, aspect ratio, and circularity thresholds '
        'to see which paw shapes are included or excluded.')

    def _add_contour_variant(cat_prefix, metrics_list, stance_suffix,
                             ratio_key, intensity_ratio_key, filter_paw=False):
        """Per-paw categories + a Ratios category for one contour variant
        (port of _build_contour_paw_tabs, flattened to categories)."""
        for role in ROLES:
            has_paw = any(
                f'{mk}_{role}' in df.columns and df[f'{mk}_{role}'].notna().any()
                for mk, _, _ in metrics_list
            )
            if not has_paw:
                continue
            paw_label = _PAW_LABELS.get(role, role)
            cat = _cat(f'{cat_prefix} - {paw_label}')

            # Shape (custom)
            _reg(cat, 'Shape', 'custom', None, None,
                 description=_shape_desc,
                 create_fn=lambda f, _r=role, _fp=filter_paw:
                     add_contour_shape_tab(
                         host, f, df, host.intermediates, _r,
                         tab_name='Shape', filter_paw=_fp))
            # Prints
            _reg(cat, 'Print (Single)', 'custom', None, None,
                 description='Single representative paw print.',
                 create_fn=lambda f, _r=role, _fp=filter_paw:
                     add_contour_print_tab(
                         host, f, df, host.intermediates, _r,
                         tab_name='Single', n_prints=1, filter_paw=_fp))
            _reg(cat, 'Prints (Multi 5)', 'custom', None, None,
                 description='Five representative paw prints.',
                 create_fn=lambda f, _r=role, _fp=filter_paw:
                     add_contour_print_tab(
                         host, f, df, host.intermediates, _r,
                         tab_name='Multi (5)', n_prints=5, filter_paw=_fp))
            # All contour metrics
            for metric_key, tab_prefix, ylbl_template in metrics_list:
                col = f'{metric_key}_{role}'
                ylbl = ylbl_template.format(paw_label)
                mdesc = _contour_descs.get(tab_prefix, ylbl)
                if col in df.columns and df[col].notna().any():
                    _reg(cat, tab_prefix, 'bar', df, col,
                         y_label=ylbl, description=mdesc)
                if (bdf is not None and col in bdf.columns
                        and bdf[col].notna().any()):
                    _reg(cat, f'{tab_prefix} - TC', 'timecourse', bdf, col,
                         y_label=ylbl,
                         description=mdesc + ' Shown across time bins.')
            if not cat:
                del registry[f'{cat_prefix} - {paw_label}']

        # Ratios
        has_ratios = False
        for rk in [ratio_key, intensity_ratio_key]:
            if rk in df.columns and df[rk].notna().any():
                has_ratios = True
            if (bdf is not None and rk in bdf.columns
                    and bdf[rk].notna().any()):
                has_ratios = True
        if has_ratios:
            cat = _cat(f'{cat_prefix} - Ratios')
            ylbl_sfx = ' (stance)' if stance_suffix else ''
            # Stored ratio columns are literal HL/HR. Relabel (and invert
            # when the injured paw is HR) at display time so every ratio
            # graph reads injured/contralateral.
            if ratio_key in df.columns and df[ratio_key].notna().any():
                _reg(cat, 'Area Ratio Hind', 'bar',
                     df, ratio_key, reference=1.0,
                     y_label=f'Paw area ratio {inj}/{contra}{ylbl_sfx}',
                     description=f'{inj}/{contra} (injured/contralateral) '
                                 'paw area ratio. 1.0 = equal.',
                     flip=True)
            if (intensity_ratio_key in df.columns
                    and df[intensity_ratio_key].notna().any()):
                _reg(cat, 'Intensity Ratio Hind', 'bar',
                     df, intensity_ratio_key, reference=1.0,
                     y_label=f'Intensity ratio {inj}/{contra}{ylbl_sfx}',
                     description=f'{inj}/{contra} (injured/contralateral) '
                                 'intensity ratio. 1.0 = equal.',
                     flip=True)
            if bdf is not None:
                if (ratio_key in bdf.columns
                        and bdf[ratio_key].notna().any()):
                    _reg(cat, 'Area Ratio - TC', 'timecourse',
                         bdf, ratio_key, reference=1.0,
                         y_label=f'Paw area ratio {inj}/{contra}{ylbl_sfx}',
                         description=f'{inj}/{contra} area ratio '
                                     'across time bins.',
                         flip=True)
                if (intensity_ratio_key in bdf.columns
                        and bdf[intensity_ratio_key].notna().any()):
                    _reg(cat, 'Intensity Ratio - TC', 'timecourse',
                         bdf, intensity_ratio_key, reference=1.0,
                         y_label=f'Intensity ratio {inj}/{contra}{ylbl_sfx}',
                         description=f'{inj}/{contra} intensity ratio '
                                     'across time bins.',
                         flip=True)
            if not cat:
                del registry[f'{cat_prefix} - Ratios']

    if show_contour and (has_contour or has_stance_contour or has_pawlike_contour):
        # All Frames (default / first variant)
        if has_contour:
            _add_contour_variant(
                'Paw Contour', _contour_metrics, False,
                'paw_area_ratio_hind', 'contact_intensity_ratio_hind')

        # Filter Preview
        if has_pawlike_contour:
            cat = _cat('Paw Contour - Filter Preview')
            for role in ROLES:
                has_paw = any(
                    f'{mk}_{role}' in df.columns and df[f'{mk}_{role}'].notna().any()
                    for mk, _, _ in _contour_pawlike_metrics
                )
                if has_paw:
                    _reg(cat, _PAW_LABELS.get(role, role), 'custom',
                         None, None,
                         description=_filter_preview_desc,
                         create_fn=lambda f, _r=role:
                             add_contour_filter_preview_tab(
                                 host, f, df, host.intermediates, _r))
            if not cat:
                del registry['Paw Contour - Filter Preview']

        # Full Stance (optional, off by default as in the old window)
        if has_stance_contour and _sets.get('full_stance', False):
            _add_contour_variant(
                'Paw Contour - Full Stance', _contour_stance_metrics, True,
                'paw_area_ratio_stance_hind',
                'contact_intensity_ratio_stance_hind')

        # Filtered Contour (renamed from Paw-like)
        if has_pawlike_contour:
            _add_contour_variant(
                'Paw Contour - Filtered', _contour_pawlike_metrics, False,
                'pawlike_area_ratio_hind', 'pawlike_intensity_ratio_hind',
                filter_paw=True)

    # ── Statistics ────────────────────────────────────────────────────────
    if show_stats and bdf is not None:
        treatments = []
        if 'treatment' in df.columns:
            treatments = [str(t) for t in df['treatment'].dropna().unique()
                          if str(t).strip()]
        if not treatments:
            treatments = ['All sessions']
        max_t = bdf['bin_start_s'].max() / 60.0
        cat = _cat('Statistics')
        _reg(cat, 'Statistics Tables', 'custom', None, None,
             description='Descriptive statistics, group tests, and '
                         'time-course ANOVA tables for every metric with data.',
             create_fn=lambda f, _t=treatments, _mt=max_t:
                 create_wb_statistics_tab(host, f, df, bdf, _t, _mt))

    # Drop any categories that ended up empty (all gates failed)
    for k in [k for k, v in registry.items() if not v]:
        del registry[k]

    return registry


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_entry(host, frame, entry, stats_mode=False):
    """Clear *frame*, close stale figures, and build the entry into it.

    stats_mode=True renders the Σ statistics view instead of the graph
    (entries without tabular data show the informative fallback message,
    matching the old Σ flip behavior for custom graphs).
    """
    for w in list(frame.winfo_children()):
        _close_figures_recursive(w)
        w.destroy()
    if stats_mode:
        _render_metric_stats(host, frame, entry)
        return
    entry['build'](frame)
