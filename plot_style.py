"""plot_style.py - shared, per-project group colors for the analysis tabs.

The Multi-Classifier, Sequencing, and Locomotion tabs all color by Treatment
group. Custom colors are stored in ``PixelPaws_project.json`` under
``group_colors`` (a ``{group: "#rrggbb"}`` dict), so a cohort keeps its
palette across sessions and across tabs. Groups without a stored color fall
back to the default cycle by position.

The color dialog is palette-first: named colorway strips (one click assigns
the whole cohort), a quick-pick chip grid per group, a full custom picker as
the escape hatch, and a live legend preview.
"""
from __future__ import annotations

import copy
import json
import os
import tkinter as tk
from tkinter import ttk, colorchooser

import numpy as np

# Every figure save defaults to publication resolution - this also covers
# the embedded matplotlib toolbar's save button, which otherwise writes at
# the on-screen figure dpi (~100).
try:
    import matplotlib
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['savefig.bbox'] = 'tight'
except Exception:  # pragma: no cover
    pass

DEFAULT_CYCLE = ["#8D99AE", "#CC79A7", "#3b528b", "#21918c",
                 "#f59f00", "#2f9e44"]

# Named colorways. First entry is the shipping default (the manuscript's).
PALETTES = [
    ("Manuscript", DEFAULT_CYCLE),
    ("Okabe-Ito", ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                   "#E69F00", "#56B4E9"]),
    ("Tol Bright", ["#4477AA", "#EE6677", "#228833", "#CCBB44",
                    "#66CCEE", "#AA3377"]),
    ("Viridis", ["#440154", "#3B528B", "#21918C", "#5EC962",
                 "#FDE725", "#90D743"]),
    ("Warm", ["#7F1D1D", "#C2410C", "#D97706", "#B45309",
              "#E11D48", "#9F1239"]),
    ("Cool", ["#1E3A8A", "#0E7490", "#0F766E", "#4338CA",
              "#0369A1", "#5B21B6"]),
    ("Grayscale", ["#1F1F1F", "#5A5A5A", "#8C8C8C", "#B3B3B3",
                   "#6E6E6E", "#3D3D3D"]),
]
_CONFIG = "PixelPaws_project.json"


def _config_path(project_folder):
    return os.path.join(project_folder or "", _CONFIG)


def _read_stored(project_folder):
    try:
        with open(_config_path(project_folder), "r") as f:
            data = json.load(f)
        gc = data.get("group_colors") or {}
        return {str(k): str(v) for k, v in gc.items()}
    except Exception:
        return {}


def get_colors(project_folder, groups):
    """{group: hex} for the given ordered group list - stored colors first,
    default cycle by position for the rest."""
    stored = _read_stored(project_folder)
    out = {}
    for i, g in enumerate(groups):
        out[g] = stored.get(str(g), DEFAULT_CYCLE[i % len(DEFAULT_CYCLE)])
    return out


def save_colors(project_folder, mapping):
    """Merge ``mapping`` into the project config's group_colors (preserving
    every other key). Silently no-ops without a project."""
    path = _config_path(project_folder)
    if not project_folder or not os.path.isdir(project_folder):
        return
    data = {}
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    gc = data.get("group_colors") or {}
    gc.update({str(k): str(v) for k, v in mapping.items()})
    data["group_colors"] = gc
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def reset_colors(project_folder, groups):
    """Remove stored overrides for the given groups."""
    path = _config_path(project_folder)
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
        gc = data.get("group_colors") or {}
        for g in groups:
            gc.pop(str(g), None)
        data["group_colors"] = gc
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Dialog
# --------------------------------------------------------------------------- #

def open_color_dialog(parent, project_folder, groups, on_apply=None):
    """Palette-first group-color editor.

    Layout: named colorway strips (click a strip → whole cohort recolored),
    per-group rows whose chips open a quick-pick grid, a live legend preview,
    and Apply/Reset/Cancel. Saved into the project config on Apply; calls
    ``on_apply()`` so the owning tab redraws.
    """
    if not groups:
        from tkinter import messagebox
        messagebox.showinfo(
            "No groups",
            "Load a key file first - colors are assigned per Treatment group.",
            parent=parent)
        return

    groups = list(groups)
    current = get_colors(project_folder, groups)

    win = tk.Toplevel(parent)
    win.title("Group colors")
    win.transient(parent.winfo_toplevel())
    win.resizable(False, False)
    frm = ttk.Frame(win, padding=12)
    frm.pack(fill="both", expand=True)

    # ---- colorway strips --------------------------------------------------
    ttk.Label(frm, text="Colorways", font=("Segoe UI", 9, "bold")).pack(
        anchor="w")
    ttk.Label(frm, text="Click a strip to recolor every group at once.",
              foreground="#777").pack(anchor="w", pady=(0, 4))
    strips = ttk.Frame(frm)
    strips.pack(fill="x", pady=(0, 8))

    CHIP, GAP = 22, 3

    def _apply_palette(cols):
        for i, g in enumerate(groups):
            current[g] = cols[i % len(cols)]
        _refresh_rows()
        _draw_preview()

    for pi, (name, cols) in enumerate(PALETTES):
        row = ttk.Frame(strips)
        row.grid(row=pi // 2, column=pi % 2, sticky="w", padx=(0, 16),
                 pady=2)
        n_show = max(len(groups), 3)
        cv = tk.Canvas(row, width=n_show * (CHIP + GAP), height=CHIP,
                       highlightthickness=1, highlightbackground="#bbbbbb",
                       cursor="hand2")
        cv.pack(side="left")
        for i in range(n_show):
            c = cols[i % len(cols)]
            cv.create_rectangle(i * (CHIP + GAP), 0,
                                i * (CHIP + GAP) + CHIP, CHIP,
                                fill=c, outline=c)
        cv.bind("<Button-1>", lambda e, c=cols: _apply_palette(c))
        lbl = ttk.Label(row, text=name, cursor="hand2")
        lbl.pack(side="left", padx=(6, 0))
        lbl.bind("<Button-1>", lambda e, c=cols: _apply_palette(c))

    ttk.Separator(frm).pack(fill="x", pady=4)

    # ---- per-group rows ---------------------------------------------------
    ttk.Label(frm, text="Groups", font=("Segoe UI", 9, "bold")).pack(
        anchor="w")
    ttk.Label(frm, text="Click a group's chip for quick picks or a custom "
                        "color.", foreground="#777").pack(anchor="w",
                                                          pady=(0, 4))
    rows_frame = ttk.Frame(frm)
    rows_frame.pack(fill="x")
    chip_canvases = {}

    _QUICK = []
    for _, cols in PALETTES:
        for c in cols:
            if c not in _QUICK:
                _QUICK.append(c)

    def _open_quick_pick(g):
        pop = tk.Toplevel(win)
        pop.title(f"Color - {g}")
        pop.transient(win)
        pop.resizable(False, False)
        pf = ttk.Frame(pop, padding=10)
        pf.pack()
        ttk.Label(pf, text=f"Quick picks for {g}:").grid(
            row=0, column=0, columnspan=7, sticky="w", pady=(0, 4))
        SW = 26
        for i, c in enumerate(_QUICK[:28]):
            cv = tk.Canvas(pf, width=SW, height=SW, highlightthickness=1,
                           highlightbackground="#999999", cursor="hand2")
            cv.grid(row=1 + i // 7, column=i % 7, padx=2, pady=2)
            cv.create_rectangle(0, 0, SW, SW, fill=c, outline=c)

            def _pick(_e=None, c=c):
                current[g] = c
                _refresh_rows()
                _draw_preview()
                pop.destroy()
            cv.bind("<Button-1>", _pick)

        def _custom():
            rgb, hexv = colorchooser.askcolor(color=current[g], parent=pop,
                                              title=f"Custom color for {g}")
            if hexv:
                current[g] = hexv
                _refresh_rows()
                _draw_preview()
            pop.destroy()

        ttk.Button(pf, text="Custom…", command=_custom).grid(
            row=6, column=0, columnspan=7, sticky="ew", pady=(6, 0))
        pop.grab_set()

    def _refresh_rows():
        for w in rows_frame.winfo_children():
            w.destroy()
        chip_canvases.clear()
        for i, g in enumerate(groups):
            ttk.Label(rows_frame, text=str(g), width=22).grid(
                row=i, column=0, sticky="w", pady=2)
            cv = tk.Canvas(rows_frame, width=46, height=20,
                           highlightthickness=1,
                           highlightbackground="#999999", cursor="hand2")
            cv.grid(row=i, column=1, sticky="e", padx=(10, 0), pady=2)
            cv.create_rectangle(0, 0, 46, 20, fill=current[g],
                                outline=current[g])
            cv.bind("<Button-1>", lambda e, g=g: _open_quick_pick(g))
            chip_canvases[g] = cv

    # ---- live preview -----------------------------------------------------
    ttk.Separator(frm).pack(fill="x", pady=6)
    ttk.Label(frm, text="Preview", font=("Segoe UI", 9, "bold")).pack(
        anchor="w")
    pv = tk.Canvas(frm, width=320, height=max(26 * len(groups), 40),
                   highlightthickness=0)
    pv.pack(anchor="w", pady=(2, 4))

    def _draw_preview():
        pv.delete("all")
        import math
        for i, g in enumerate(groups):
            y0 = 8 + i * 24
            c = current[g]
            pts = []
            for k in range(28):
                x = 6 + k * 3.2
                y = y0 + 6 - 5 * math.sin(k / 4.5 + i)
                pts += [x, y]
            pv.create_line(*pts, fill=c, width=2, smooth=True)
            pv.create_oval(98, y0 - 1, 112, y0 + 13, fill=c, outline=c)
            pv.create_text(122, y0 + 6, text=str(g), anchor="w",
                           font=("Segoe UI", 9))

    # ---- buttons ----------------------------------------------------------
    br = ttk.Frame(frm)
    br.pack(fill="x", pady=(8, 0))

    def _rotate():
        vals = [current[g] for g in groups]
        vals = vals[1:] + vals[:1]
        for g, c in zip(groups, vals):
            current[g] = c
        _refresh_rows()
        _draw_preview()

    def _reset():
        reset_colors(project_folder, groups)
        fresh = get_colors(project_folder, groups)
        current.update(fresh)
        _refresh_rows()
        _draw_preview()

    def _apply():
        save_colors(project_folder, {g: current[g] for g in groups})
        if on_apply:
            try:
                on_apply()
            except Exception:
                pass
        win.destroy()

    ttk.Button(br, text="Rotate", command=_rotate).pack(side="left")
    ttk.Button(br, text="Reset defaults", command=_reset).pack(
        side="left", padx=(6, 0))
    ttk.Button(br, text="Cancel", command=win.destroy).pack(side="right")
    ttk.Button(br, text="Apply", command=_apply).pack(side="right",
                                                      padx=(0, 6))

    _refresh_rows()
    _draw_preview()
    win.grab_set()


# --------------------------------------------------------------------------- #
# Plot options (error bars, lines, markers, frame) - shared across the tabs
# --------------------------------------------------------------------------- #

LINE_STYLE_NAMES = ["solid", "dashed", "dotted", "dashdot"]
_MPL_LS = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}

DEFAULT_OPTIONS = {
    # ---- core ----
    "error_type": "SEM",         # "SEM" | "SD" | "95CI"
    "error_display": "band",     # "band" | "caps" | "none"
    "line_width": 1.6,
    "line_styles": {},           # {group: name in LINE_STYLE_NAMES}
    "show_individual": False,    # spaghetti / per-animal points
    "show_markers": False,
    "marker_size": 4.0,
    "marker_alpha": 0.95,
    # ---- axis & frame ----
    "grid": False,
    "font_size": 9,
    "y_from_zero": False,
    # ---- bar graphs ----
    "bar_kind": "bar",           # "bar" | "box"
    "point_jitter": 0.0,         # 0-0.4, fraction of bar width
    "bar_edges": False,
    # ---- significance markers ----
    "sig_style": "asterisk",     # "asterisk" | "hash" | "dagger" | "p-value"
    # ---- heatmaps ----
    "heatmap_cmap": "viridis",
    # ---- labels & legend ----
    "show_titles": True,         # figure/panel titles
    "show_n": True,              # append (n=X) to group labels
    "legend_loc": "bottom",      # "bottom" | "right" | "none" (figure legends)
}

HEATMAP_CMAPS = ["viridis", "magma", "plasma", "inferno", "cividis",
                 "hot", "YlOrRd", "Blues", "coolwarm"]


def project_of(app):
    """Normalized project folder from the app handle (StringVar or str)."""
    proj = getattr(app, "current_project_folder", None)
    if hasattr(proj, "get"):
        proj = proj.get()
    return proj or ""


def get_options(project_folder):
    """DEFAULT_OPTIONS with the project's stored ``plot_options`` merged over
    them (known keys only; line_styles dict-merged). Never raises."""
    opts = copy.deepcopy(DEFAULT_OPTIONS)
    try:
        with open(_config_path(project_folder), "r") as f:
            stored = (json.load(f).get("plot_options") or {})
    except Exception:
        return opts
    for k, v in stored.items():
        if k not in DEFAULT_OPTIONS:
            continue
        if k == "line_styles" and isinstance(v, dict):
            opts["line_styles"].update({str(g): str(ls) for g, ls in v.items()
                                        if str(ls) in LINE_STYLE_NAMES})
        else:
            opts[k] = v
    return opts


def save_options(project_folder, opts):
    """Write the known option keys under ``plot_options`` (read-merge-write,
    preserving group_colors and every other key). No-ops without a project."""
    path = _config_path(project_folder)
    if not project_folder or not os.path.isdir(project_folder):
        return
    data = {}
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["plot_options"] = {k: opts[k] for k in DEFAULT_OPTIONS if k in opts}
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def reset_options(project_folder):
    """Remove the stored plot_options key."""
    path = _config_path(project_folder)
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
        data.pop("plot_options", None)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def calc_error(values, error_type):
    """Error-bar half-width for a 1-D array (port of the gait tab's
    ``_calc_error``). 'SEM' | 'SD' | '95CI'; n < 2 -> 0.0."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return 0.0
    sd = np.nanstd(v, ddof=1)
    if error_type == "SD":
        return float(sd)
    sem = sd / np.sqrt(n)
    if error_type == "95CI":
        from scipy.stats import t as _t_dist
        return float(_t_dist.ppf(0.975, df=n - 1) * sem)
    return float(sem)


def error_label(error_type):
    labels = {"SEM": "mean ± SEM", "SD": "mean ± SD",
              "95CI": "mean ± 95% CI"}
    return labels.get(error_type, "mean ± SEM")


def mean_and_error(rows, opts):
    """(mean, err) per column for rows shaped (n_subjects, n_bins)."""
    rows = np.asarray(rows, float)
    mean = np.nanmean(rows, axis=0)
    err = np.array([calc_error(rows[:, j], opts["error_type"])
                    for j in range(rows.shape[1])])
    return mean, err


def draw_series(ax, x, rows, color, opts, label=None, group=None):
    """Draw a group timecourse honoring the plot options: optional per-animal
    spaghetti, mean line (width/style/marker), and error as shaded band,
    capped bars, or nothing. Returns (mean, err)."""
    rows = np.asarray(rows, float)
    mean, err = mean_and_error(rows, opts)
    if opts.get("show_individual") and rows.shape[0] > 1:
        for r in rows:
            ax.plot(x, r, color=color, lw=0.8, alpha=0.3, zorder=1)
    ls = _MPL_LS.get(opts.get("line_styles", {}).get(str(group), "solid"), "-")
    line_kw = dict(color=color, lw=float(opts.get("line_width", 1.6)),
                   ls=ls, zorder=3, label=label)
    if opts.get("show_markers"):
        line_kw.update(marker="o", markersize=float(opts.get("marker_size", 4)))
    ax.plot(x, mean, **line_kw)
    disp = opts.get("error_display", "band")
    if disp == "band":
        ax.fill_between(x, mean - err, mean + err, color=color, alpha=0.2,
                        lw=0, zorder=2)
    elif disp == "caps":
        ax.errorbar(x, mean, yerr=err, fmt="none", ecolor=color, capsize=3,
                    capthick=1.2, elinewidth=0.9, alpha=0.7, zorder=2)
    return mean, err


def apply_frame_options(ax, opts, base=None, y0=True):
    """Grid / font sizes / y-from-zero. Call at the END of a draw (after the
    legend exists) so legend text is resized too."""
    base = int(base or opts.get("font_size", 9))
    ax.tick_params(labelsize=max(base - 1, 5))
    ax.xaxis.label.set_size(base)
    ax.yaxis.label.set_size(base)
    if ax.get_title():
        ax.title.set_size(base + 1)
    if opts.get("grid"):
        ax.grid(True, lw=0.5, alpha=0.35)
        ax.set_axisbelow(True)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(max(base - 1, 5))
    if y0 and opts.get("y_from_zero"):
        ax.set_ylim(bottom=0)


def fmt_mean_err(values, opts):
    """'mean ± err (n=X)' for a 1-D sample, honoring error_type."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "-"
    m = float(np.mean(v))
    e = calc_error(v, opts.get("error_type", "SEM"))
    return f"{m:.3g} ± {e:.3g} (n={len(v)})"


def stats_table(title, row_labels, rows_by_group, opts, test_fn=None,
                unit=""):
    """Monospace stats table: one row per label, one column per group
    (mean ± err (n)), plus a test column when test_fn is given.

    rows_by_group: {group: 2-D array (n_subjects, n_labels)}.
    test_fn(list_of_samples) -> (p, test_name) or None.
    """
    groups = list(rows_by_group)
    colw = max([14] + [len(str(g)) + 2 for g in groups])
    lw = max([12] + [len(str(l)) + 1 for l in row_labels])
    lines = [title, "=" * len(title), ""]
    hdr = " " * lw + "".join(f"{str(g):>{colw + 8}}" for g in groups)
    if test_fn:
        hdr += f"{'test':>16}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, lab in enumerate(row_labels):
        cells = []
        samples = []
        for g in groups:
            col = rows_by_group[g][:, i]
            col = col[np.isfinite(col)]
            samples.append(col)
            cells.append(f"{fmt_mean_err(col, opts):>{colw + 8}}")
        row = f"{str(lab):<{lw}}" + "".join(cells)
        if test_fn:
            res = test_fn([c for c in samples if len(c) >= 2])
            if res is not None:
                pv, name = res
                mark = sig_text(pv, 0.05, opts.get("sig_style", "asterisk"))
                if opts.get("sig_style") == "p-value":
                    mark = ""
                row += f"{name + ' p=' + format(pv, '.3g') + (' ' + mark if mark else ''):>16}"
            else:
                row += f"{'-':>16}"
        lines.append(row)
    if unit:
        lines += ["", f"values: {unit}; error: "
                      + error_label(opts.get("error_type", "SEM"))]
    return "\n".join(lines)


def group_label(group, n, opts):
    """Group legend/title label honoring the show_n option."""
    return f"{group} (n={n})" if opts.get("show_n", True) else str(group)


def fig_legend(fig, handles, labels, opts, fontsize=8):
    """Figure-level legend honoring legend_loc ("bottom"/"right"/"none").
    Uses matplotlib's constrained-layout-aware "outside" locations so the
    legend never overlaps panels."""
    loc = opts.get("legend_loc", "bottom")
    if loc == "none" or not handles:
        return None
    if loc == "right":
        return fig.legend(handles, labels, loc="outside center right",
                          frameon=False, fontsize=fontsize)
    return fig.legend(handles, labels, loc="outside lower center",
                      frameon=False, fontsize=fontsize,
                      ncol=min(max(len(labels), 1), 4))


def sig_text(p_value, alpha=0.05, style="asterisk"):
    """Significance annotation for a p-value, honoring the sig_style option.

    Symbol styles use GraphPad-like tiers (<1e-4, <1e-3, <0.01, <alpha);
    "p-value" style returns the number itself. Empty string when not
    significant (symbol styles) so callers can skip the annotation.
    """
    if style == "p-value":
        return f"p={p_value:.3g}"
    sym = {"asterisk": "*", "hash": "#", "dagger": "†"}.get(style, "*")
    if p_value < 1e-4:
        return sym * 4
    if p_value < 1e-3:
        return sym * 3
    if p_value < 1e-2:
        return sym * 2
    if p_value < alpha:
        return sym
    return ""


def open_options_dialog(parent, project_folder, groups, on_apply=None):
    """Combined Colors + Plot Options dialog (single entry point for all
    styling): colorway strips and per-group chips on top, then error/line/
    marker/significance options, per-group line styles, and two collapsed
    sections (Axis & frame incl. heatmap colormap, Bar graphs), with a live
    preview. Apply saves ``group_colors`` AND ``plot_options`` and calls
    ``on_apply()``. Works with an empty group list."""
    from ui_tooltip import Tip, collapsible

    groups = list(groups or [])
    opts = get_options(project_folder)
    current_colors = get_colors(project_folder, groups) if groups else {}

    win = tk.Toplevel(parent)
    win.title("Plot style")
    win.transient(parent.winfo_toplevel())
    win.resizable(False, True)
    outer_c = tk.Canvas(win, highlightthickness=0, width=430)
    vsb = ttk.Scrollbar(win, orient="vertical", command=outer_c.yview)
    outer_c.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    outer_c.pack(side="left", fill="both", expand=True)
    frm = ttk.Frame(outer_c, padding=12)
    outer_c.create_window((0, 0), window=frm, anchor="nw")
    frm.bind("<Configure>",
             lambda e: outer_c.configure(scrollregion=outer_c.bbox("all"),
                                         height=min(e.height + 10, 760)))

    # ================= Colors =================
    ttk.Label(frm, text="Colors", font=("Segoe UI", 9, "bold")).pack(
        anchor="w")
    if groups:
        ttk.Label(frm, text="Click a strip to recolor every group; click a "
                            "group's chip for quick picks.",
                  foreground="#777").pack(anchor="w", pady=(0, 3))
        strips = ttk.Frame(frm)
        strips.pack(fill="x", pady=(0, 4))
        CHIP, GAP = 18, 3

        def _apply_palette(cols):
            for i, g in enumerate(groups):
                current_colors[g] = cols[i % len(cols)]
            _refresh_chips()
            _draw_preview()

        for pi, (name, cols) in enumerate(PALETTES):
            row = ttk.Frame(strips)
            row.grid(row=pi // 2, column=pi % 2, sticky="w", padx=(0, 14),
                     pady=1)
            n_show = max(len(groups), 3)
            cv = tk.Canvas(row, width=n_show * (CHIP + GAP), height=CHIP,
                           highlightthickness=1,
                           highlightbackground="#bbbbbb", cursor="hand2")
            cv.pack(side="left")
            for i in range(n_show):
                c = cols[i % len(cols)]
                cv.create_rectangle(i * (CHIP + GAP), 0,
                                    i * (CHIP + GAP) + CHIP, CHIP,
                                    fill=c, outline=c)
            cv.bind("<Button-1>", lambda e, c=cols: _apply_palette(c))
            lbl = ttk.Label(row, text=name, cursor="hand2",
                            font=("Segoe UI", 8))
            lbl.pack(side="left", padx=(4, 0))
            lbl.bind("<Button-1>", lambda e, c=cols: _apply_palette(c))

        chips_frame = ttk.Frame(frm)
        chips_frame.pack(fill="x")

        _QUICK = []
        for _, cols in PALETTES:
            for c in cols:
                if c not in _QUICK:
                    _QUICK.append(c)

        def _quick_pick(g):
            pop = tk.Toplevel(win)
            pop.title(f"Color - {g}")
            pop.transient(win)
            pop.resizable(False, False)
            pf = ttk.Frame(pop, padding=10)
            pf.pack()
            SW = 24
            for i, c in enumerate(_QUICK[:28]):
                cv = tk.Canvas(pf, width=SW, height=SW, highlightthickness=1,
                               highlightbackground="#999999", cursor="hand2")
                cv.grid(row=i // 7, column=i % 7, padx=2, pady=2)
                cv.create_rectangle(0, 0, SW, SW, fill=c, outline=c)

                def _pick(_e=None, c=c):
                    current_colors[g] = c
                    _refresh_chips()
                    _draw_preview()
                    pop.destroy()
                cv.bind("<Button-1>", _pick)

            def _custom():
                _rgb, hexv = colorchooser.askcolor(
                    color=current_colors[g], parent=pop,
                    title=f"Custom color for {g}")
                if hexv:
                    current_colors[g] = hexv
                    _refresh_chips()
                    _draw_preview()
                pop.destroy()

            ttk.Button(pf, text="Custom...", command=_custom).grid(
                row=4, column=0, columnspan=7, sticky="ew", pady=(6, 0))
            pop.grab_set()

        def _refresh_chips():
            for w in chips_frame.winfo_children():
                w.destroy()
            for i, g in enumerate(groups):
                ttk.Label(chips_frame, text=str(g), width=20).grid(
                    row=i // 2, column=(i % 2) * 2, sticky="w", pady=1)
                cv = tk.Canvas(chips_frame, width=40, height=16,
                               highlightthickness=1,
                               highlightbackground="#999999",
                               cursor="hand2")
                cv.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w",
                        padx=(4, 14), pady=1)
                cv.create_rectangle(0, 0, 40, 16, fill=current_colors[g],
                                    outline=current_colors[g])
                cv.bind("<Button-1>", lambda e, g=g: _quick_pick(g))
    else:
        ttk.Label(frm, foreground="#777",
                  text="Load a key file to assign group colors.").pack(
            anchor="w", pady=(0, 3))

        def _refresh_chips():
            pass

    ttk.Separator(frm).pack(fill="x", pady=5)

    # ================= Error & lines =================
    v_err = tk.StringVar(value=opts["error_type"])
    v_disp = tk.StringVar(value=opts["error_display"])
    v_lw = tk.DoubleVar(value=opts["line_width"])
    v_ind = tk.BooleanVar(value=opts["show_individual"])
    v_mark = tk.BooleanVar(value=opts["show_markers"])
    v_ms = tk.DoubleVar(value=opts["marker_size"])
    v_ma = tk.DoubleVar(value=opts["marker_alpha"])
    v_sig = tk.StringVar(value=opts["sig_style"])
    v_grid = tk.BooleanVar(value=opts["grid"])
    v_font = tk.IntVar(value=opts["font_size"])
    v_y0 = tk.BooleanVar(value=opts["y_from_zero"])
    v_hm = tk.StringVar(value=opts["heatmap_cmap"])
    v_ttl = tk.BooleanVar(value=opts["show_titles"])
    v_n = tk.BooleanVar(value=opts["show_n"])
    v_leg = tk.StringVar(value=opts["legend_loc"])
    v_bar = tk.StringVar(value=opts["bar_kind"])
    v_jit = tk.DoubleVar(value=opts["point_jitter"])
    v_edge = tk.BooleanVar(value=opts["bar_edges"])
    v_ls = {g: tk.StringVar(value=opts["line_styles"].get(str(g), "solid"))
            for g in groups}

    def _row(parent_, label):
        r = ttk.Frame(parent_)
        r.pack(fill="x", pady=1)
        ttk.Label(r, text=label).pack(side="left")
        return r

    ttk.Label(frm, text="Error & lines", font=("Segoe UI", 9, "bold")).pack(
        anchor="w")
    r = _row(frm, "Error value:")
    for val, lab in (("SEM", "SEM"), ("SD", "SD"), ("95CI", "95% CI")):
        ttk.Radiobutton(r, text=lab, value=val, variable=v_err).pack(
            side="left", padx=(8, 0))
    r = _row(frm, "Error display:")
    for val, lab in (("band", "Shaded band"), ("caps", "Capped bars"),
                     ("none", "None")):
        ttk.Radiobutton(r, text=lab, value=val, variable=v_disp).pack(
            side="left", padx=(8, 0))
    r = _row(frm, "Line width:")
    ttk.Spinbox(r, textvariable=v_lw, from_=0.5, to=5.0, increment=0.1,
                width=6).pack(side="right")
    cb_ind = ttk.Checkbutton(frm, text="Individual animals "
                                       "(spaghetti / points)",
                             variable=v_ind)
    cb_ind.pack(anchor="w", pady=(2, 0))
    Tip(cb_ind, "Draw each animal's own thin trace behind the group mean "
                "(timecourses), and per-animal points on bar graphs.")
    r = _row(frm, "Markers on means:")
    ttk.Checkbutton(r, variable=v_mark).pack(side="left", padx=(8, 0))
    ttk.Label(r, text="size").pack(side="left", padx=(10, 2))
    ttk.Spinbox(r, textvariable=v_ms, from_=2, to=12, increment=0.5,
                width=5).pack(side="left")
    ttk.Label(r, text="alpha").pack(side="left", padx=(10, 2))
    ttk.Spinbox(r, textvariable=v_ma, from_=0.1, to=1.0, increment=0.05,
                width=5).pack(side="left")
    r = _row(frm, "Significance markers:")
    sigcb = ttk.Combobox(r, textvariable=v_sig, state="readonly", width=10,
                         values=["asterisk", "hash", "dagger", "p-value"])
    sigcb.pack(side="right")
    Tip(sigcb, "How significant comparisons are annotated on the graphs: "
               "tiered symbols (*, **, *** ...; # or † variants) or the "
               "numeric p-value.")

    ttk.Separator(frm).pack(fill="x", pady=4)
    ttk.Label(frm, text="Per-group line style",
              font=("Segoe UI", 9, "bold")).pack(anchor="w")
    if groups:
        for g in groups:
            r = _row(frm, str(g))
            ttk.Combobox(r, textvariable=v_ls[g], state="readonly",
                         width=10, values=LINE_STYLE_NAMES).pack(side="right")
    else:
        ttk.Label(frm, foreground="#777",
                  text="Load a key file to set per-group styles.").pack(
            anchor="w")

    axf = collapsible(frm, "Axis & frame", collapsed=True, fill="x",
                      pady=(6, 0))
    ttk.Checkbutton(axf, text="Grid lines", variable=v_grid).pack(anchor="w")
    r = _row(axf, "Font size:")
    ttk.Spinbox(r, textvariable=v_font, from_=6, to=16, width=5).pack(
        side="right")
    ttk.Checkbutton(axf, text="Start y-axis at 0", variable=v_y0).pack(
        anchor="w")
    ttk.Checkbutton(axf, text="Show titles", variable=v_ttl).pack(anchor="w")
    ttk.Checkbutton(axf, text="Show n in group labels",
                    variable=v_n).pack(anchor="w")
    r = _row(axf, "Legend:")
    lcb = ttk.Combobox(r, textvariable=v_leg, state="readonly", width=10,
                       values=["bottom", "right", "none"])
    lcb.pack(side="right")
    Tip(lcb, "Where multi-panel figures place their legend.")
    r = _row(axf, "Heatmap colormap:")
    hmcb = ttk.Combobox(r, textvariable=v_hm, state="readonly", width=10,
                        values=HEATMAP_CMAPS)
    hmcb.pack(side="right")
    Tip(hmcb, "Colormap for occupancy heatmaps (also switchable from the "
              "dropdown that appears next to the Graph picker on heatmap "
              "views).")

    barf = collapsible(frm, "Bar graphs", collapsed=True, fill="x",
                       pady=(4, 0))
    r = _row(barf, "Occupancy:")
    for val, lab in (("bar", "Bars"), ("box", "Box plots")):
        ttk.Radiobutton(r, text=lab, value=val, variable=v_bar).pack(
            side="left", padx=(8, 0))
    r = _row(barf, "Point jitter:")
    jsp = ttk.Spinbox(r, textvariable=v_jit, from_=0.0, to=0.4,
                      increment=0.05, width=5)
    jsp.pack(side="right")
    Tip(jsp, "Horizontal spread of per-animal points on bar graphs, as a "
             "fraction of the bar width. 0 = a straight column.")
    ttk.Checkbutton(barf, text="Black bar edges", variable=v_edge).pack(
        anchor="w")

    ttk.Separator(frm).pack(fill="x", pady=6)
    ttk.Label(frm, text="Preview", font=("Segoe UI", 9, "bold")).pack(
        anchor="w")
    pv = tk.Canvas(frm, width=340, height=92, highlightthickness=0)
    pv.pack(anchor="w", pady=(2, 4))

    _DASH = {"solid": None, "dashed": (6, 3), "dotted": (2, 3),
             "dashdot": (6, 3, 2, 3)}
    _pv_groups = groups[:2] if groups else ["Group A", "Group B"]
    import math

    def _draw_preview(*_a):
        pv.delete("all")
        for i, g in enumerate(_pv_groups):
            col = current_colors.get(g, DEFAULT_CYCLE[i])
            y0_ = 26 + i * 40
            pts, band_top, band_bot = [], [], []
            for k in range(30):
                x = 8 + k * 10.6
                y = y0_ - 9 * math.sin(k / 5.0 + i * 1.4)
                pts += [x, y]
                band_top += [x, y - 7]
                band_bot = [x, y + 7] + band_bot
            if v_ind.get():
                for off in (-5, 4):
                    sp = [c + (off if j % 2 else 0)
                          for j, c in enumerate(pts)]
                    pv.create_line(*sp, fill=col, width=1,
                                   stipple="gray50", smooth=True)
            if v_disp.get() == "band":
                pv.create_polygon(*(band_top + band_bot), fill=col,
                                  outline="", stipple="gray25")
            elif v_disp.get() == "caps":
                for k in range(0, 30, 6):
                    x = 8 + k * 10.6
                    y = y0_ - 9 * math.sin(k / 5.0 + i * 1.4)
                    pv.create_line(x, y - 7, x, y + 7, fill=col)
                    pv.create_line(x - 3, y - 7, x + 3, y - 7, fill=col)
                    pv.create_line(x - 3, y + 7, x + 3, y + 7, fill=col)
            ls = v_ls.get(g)
            name = ls.get() if ls is not None else "solid"
            try:
                lw = max(int(round(float(v_lw.get()))), 1)
            except Exception:
                lw = 2
            kw = dict(fill=col, width=lw, smooth=True)
            if _DASH.get(name):
                kw["dash"] = _DASH[name]
            pv.create_line(*pts, **kw)
            if v_mark.get():
                try:
                    ms = max(float(v_ms.get()), 2) / 2.0
                except Exception:
                    ms = 2
                for k in range(0, 30, 6):
                    x = 8 + k * 10.6
                    y = y0_ - 9 * math.sin(k / 5.0 + i * 1.4)
                    pv.create_oval(x - ms, y - ms, x + ms, y + ms,
                                   fill=col, outline=col)
        mark = sig_text(0.004, 0.05, v_sig.get())
        pv.create_text(330, 10, text=mark or "ns", anchor="e",
                       font=("Segoe UI", 10, "bold"), fill="#555555")

    for var in [v_err, v_disp, v_lw, v_ind, v_mark, v_ms, v_ma, v_sig,
                v_grid, v_font, v_y0, v_hm, v_bar, v_jit, v_edge,
                v_ttl, v_n, v_leg] + list(v_ls.values()):
        var.trace_add("write", _draw_preview)

    br = ttk.Frame(frm)
    br.pack(fill="x", pady=(6, 0))

    def _collect():
        return {
            "error_type": v_err.get(), "error_display": v_disp.get(),
            "line_width": float(v_lw.get()),
            "line_styles": {str(g): v_ls[g].get() for g in groups
                            if v_ls[g].get() != "solid"},
            "show_individual": bool(v_ind.get()),
            "show_markers": bool(v_mark.get()),
            "marker_size": float(v_ms.get()),
            "marker_alpha": float(v_ma.get()),
            "sig_style": v_sig.get(),
            "grid": bool(v_grid.get()), "font_size": int(v_font.get()),
            "y_from_zero": bool(v_y0.get()),
            "heatmap_cmap": v_hm.get(),
            "show_titles": bool(v_ttl.get()), "show_n": bool(v_n.get()),
            "legend_loc": v_leg.get(),
            "bar_kind": v_bar.get(), "point_jitter": float(v_jit.get()),
            "bar_edges": bool(v_edge.get()),
        }

    def _apply():
        save_options(project_folder, _collect())
        if groups:
            save_colors(project_folder,
                        {g: current_colors[g] for g in groups})
        if on_apply:
            try:
                on_apply()
            except Exception:
                pass
        win.destroy()

    def _reset():
        reset_options(project_folder)
        if groups:
            reset_colors(project_folder, groups)
            current_colors.update(get_colors(project_folder, groups))
            _refresh_chips()
        fresh = get_options(project_folder)
        v_err.set(fresh["error_type"]); v_disp.set(fresh["error_display"])
        v_lw.set(fresh["line_width"]); v_ind.set(fresh["show_individual"])
        v_mark.set(fresh["show_markers"]); v_ms.set(fresh["marker_size"])
        v_ma.set(fresh["marker_alpha"]); v_sig.set(fresh["sig_style"])
        v_grid.set(fresh["grid"]); v_font.set(fresh["font_size"])
        v_y0.set(fresh["y_from_zero"]); v_hm.set(fresh["heatmap_cmap"])
        v_bar.set(fresh["bar_kind"]); v_jit.set(fresh["point_jitter"])
        v_edge.set(fresh["bar_edges"])
        v_ttl.set(fresh["show_titles"]); v_n.set(fresh["show_n"])
        v_leg.set(fresh["legend_loc"])
        for g in groups:
            v_ls[g].set(fresh["line_styles"].get(str(g), "solid"))

    ttk.Button(br, text="Reset defaults", command=_reset).pack(side="left")
    ttk.Button(br, text="Cancel", command=win.destroy).pack(side="right")
    ttk.Button(br, text="Apply", command=_apply).pack(side="right",
                                                      padx=(0, 6))
    win._pp_apply = _apply          # handle for headless tests

    _refresh_chips()
    _draw_preview()
    win.grab_set()
