# -*- coding: utf-8 -*-
"""
Shared "Sessions" picker for the analysis tabs.

A compact button that reads "Sessions: All (6)" / "Sessions: 4 of 6" and
expands an inline session table right under it (no separate window): one row
per session with an include tick plus informative columns (Subject / Group /
Video / Cache — whatever the tab supplies). Clicking a row toggles it;
All / None act on the whole cohort. Everything is included by default, so
tabs behave exactly as before unless the user opts out of sessions.

Usage:
    flt = SessionFilter(parent, on_change=self._refresh)
    flt.pack(...)
    flt.set_sessions(names)                       # plain list
    flt.set_sessions(names, key_df=self._key_df)  # + Subject/Group columns
    flt.set_sessions(names, extra={n: {"Video": "✓", "Cache": "brt"}, ...})
    keep = flt.selected()          # list of ticked names
    if flt.allows(name): ...       # membership test (unknown names pass)
"""

import tkinter as tk
from tkinter import ttk

try:
    from ui_tooltip import Tip
except Exception:  # pragma: no cover
    Tip = None

# Canonical column order for whatever info the tabs supply.
_COL_ORDER = ("Subject", "Group", "Video", "Cache")


def _match_key_row(key_df, session):
    """Match a session name against key Subjects the way the analysis tabs
    do: the Subject must appear as a whole underscore-separated token of the
    session name (so 'mouse1' matches 'mouse1_veh' but not 'mouse10_veh')."""
    if key_df is None or "Subject" not in getattr(key_df, "columns", ()):
        return None
    toks = str(session).split("_")
    best = None
    for _, row in key_df.iterrows():
        subj = str(row.get("Subject", "")).strip()
        if subj and subj in toks:
            if best is None or len(subj) > len(str(best.get("Subject", ""))):
                best = row
    return best


class SessionFilter(ttk.Frame):
    """Session picker: compact button + gait-style table popup."""

    def __init__(self, parent, on_change=None, label="Sessions", **kw):
        super().__init__(parent, **kw)
        self._label = label
        self._on_change = on_change
        self._included = {}      # name -> bool (insertion-ordered)
        self._info = {}          # name -> {col: value}
        self._columns = []       # active info columns, canonical order
        self._popup = None
        self._btn = ttk.Button(self, command=self._open_popup)
        self._btn.pack(fill="x")
        self._update_text()
        if Tip is not None:
            Tip(self._btn,
                "Choose which sessions feed this tab's graphs and stats.\n"
                "Opens a session table; everything is included by default —\n"
                "click rows to exclude them.")

    # ── public API ─────────────────────────────────────────────────────────

    def set_sessions(self, names, key_df=None, extra=None):
        """Declare the available sessions. Tick states persist for names
        that remain; new names start ticked. `key_df` (Subject/Treatment)
        adds Subject and Group columns; `extra` = {name: {col: value}} adds
        arbitrary columns (e.g. Video, Cache)."""
        names = [str(n) for n in names]
        old = self._included
        self._included = {n: old.get(n, True) for n in names}

        self._info = {}
        cols = set()
        for n in names:
            row = {}
            if key_df is not None:
                m = _match_key_row(key_df, n)
                if m is not None:
                    row["Subject"] = str(m.get("Subject", ""))
                    row["Group"] = str(m.get("Treatment", ""))
                    cols.update(("Subject", "Group"))
            for k, v in ((extra or {}).get(n, {}) or {}).items():
                row[str(k)] = str(v)
                cols.add(str(k))
            self._info[n] = row
        self._columns = ([c for c in _COL_ORDER if c in cols]
                         + sorted(c for c in cols if c not in _COL_ORDER))
        if self._popup is not None and self._popup.winfo_exists():
            self._fill_tree()
        self._update_text()

    def selected(self):
        """Names currently ticked (in declared order)."""
        return [n for n, v in self._included.items() if v]

    def allows(self, name):
        """True when `name` is unknown (never filter surprises) or ticked."""
        v = self._included.get(str(name))
        return True if v is None else bool(v)

    def all_selected(self):
        return all(self._included.values())

    def select_all(self, value=True):
        for n in self._included:
            self._included[n] = bool(value)
        if self._popup is not None and self._popup.winfo_exists():
            self._fill_tree()
        self._changed()

    # ── inline table panel (expands in place under the button) ─────────────

    def _open_popup(self):
        """Toggle the inline session table under the button."""
        if self._popup is not None and self._popup.winfo_exists():
            self._close_popup()
            return
        frame = ttk.Frame(self, padding=(2, 3, 2, 2))
        frame.pack(fill="x")
        self._popup = frame

        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(0, 3))
        ttk.Button(bar, text="All", width=5,
                   command=lambda: self.select_all(True)).pack(side="left")
        ttk.Button(bar, text="None", width=6,
                   command=lambda: self.select_all(False)).pack(side="left",
                                                               padx=(4, 0))
        self._count_lbl = ttk.Label(bar, text="", foreground="#666666")
        self._count_lbl.pack(side="left", padx=8)

        tf = ttk.Frame(frame)
        tf.pack(fill="x")
        cols = ["include", "session"] + [c.lower() for c in self._columns]
        tree = ttk.Treeview(tf, columns=cols, show="headings",
                            height=min(max(len(self._included), 3), 10),
                            selectmode="none")
        tree.heading("include", text="✓")
        tree.column("include", width=30, anchor="center", stretch=False)
        tree.heading("session", text="Session")
        tree.column("session", width=140, stretch=True)
        for c in self._columns:
            tree.heading(c.lower(), text=c)
            tree.column(c.lower(), width=70 if c in ("Subject", "Group")
                        else 55, stretch=False)
        vsb = ttk.Scrollbar(tf, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="x", expand=True)
        vsb.pack(side="right", fill="y")
        tree.bind("<Button-1>", self._on_tree_click)
        self._tree = tree
        self._fill_tree()
        self._update_text()

    def _close_popup(self, *_):
        if self._popup is not None:
            try:
                self._popup.destroy()
            except Exception:
                pass
            self._popup = None
        self._update_text()

    def _fill_tree(self):
        tree = self._tree
        for item in tree.get_children():
            tree.delete(item)
        for n, inc in self._included.items():
            info = self._info.get(n, {})
            vals = [("✓" if inc else "—"), n] + \
                   [info.get(c, "") for c in self._columns]
            tree.insert("", "end", iid=n, values=vals,
                        tags=("on" if inc else "off",))
        tree.tag_configure("off", foreground="#999999")
        n_sel = len(self.selected())
        self._count_lbl.config(
            text=f"{n_sel} of {len(self._included)} included")

    def _on_tree_click(self, event):
        row = self._tree.identify_row(event.y)
        if not row:
            return
        self._included[row] = not self._included.get(row, True)
        self._fill_tree()
        self._changed()

    # ── internals ──────────────────────────────────────────────────────────

    def _changed(self):
        self._update_text()
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                pass

    def _update_text(self):
        total = len(self._included)
        n_sel = len(self.selected())
        if total == 0:
            txt = f"{self._label}: —"
        elif n_sel == total:
            txt = f"{self._label}: All ({total})"
        else:
            txt = f"{self._label}: {n_sel} of {total}"
        opened = self._popup is not None and self._popup.winfo_exists()
        self._btn.configure(text=txt + ("  ▴" if opened else "  ▾"))
