# -*- coding: utf-8 -*-
"""
Shared "Sessions" picker for the analysis tabs.

A compact button that reads "Sessions: All (6)" / "Sessions: 4 of 6" and
opens a small "Sessions" window with the session table (the rails are too
narrow to host it inline): one row per session with an include tick plus
informative columns (Subject / Group / Video / Cache - whatever the tab
supplies). Clicking a row toggles it; All / None act on the whole cohort.
Everything is included by default, so tabs behave exactly as before unless
the user opts out of sessions.

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
                "Opens a session table; everything is included by default -\n"
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

    # ── pop-out session table (small titled dialog near the button) ────────

    def _open_popup(self):
        """Toggle the session-table window."""
        if self._popup is not None and self._popup.winfo_exists():
            self._close_popup()
            return
        win = tk.Toplevel(self)
        self._popup = win
        win.title("Sessions")
        win.transient(self.winfo_toplevel())
        win.resizable(True, True)

        frame = ttk.Frame(win, padding=6)
        frame.pack(fill="both", expand=True)

        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Button(bar, text="All", width=5,
                   command=lambda: self.select_all(True)).pack(side="left")
        ttk.Button(bar, text="None", width=6,
                   command=lambda: self.select_all(False)).pack(side="left",
                                                               padx=(4, 0))
        self._count_lbl = ttk.Label(bar, text="", foreground="#666666")
        self._count_lbl.pack(side="left", padx=8)
        ttk.Button(bar, text="Close", width=6,
                   command=self._close_popup).pack(side="right")
        ttk.Label(frame, text="Click a row to include / exclude it.",
                  foreground="#888888").pack(anchor="w", pady=(0, 3))

        tf = ttk.Frame(frame)
        tf.pack(fill="both", expand=True)
        cols = ["include", "session"] + [c.lower() for c in self._columns]
        tree = ttk.Treeview(tf, columns=cols, show="headings",
                            height=min(max(len(self._included), 3), 16),
                            selectmode="none")
        tree.heading("include", text="✓")
        tree.column("include", width=34, anchor="center", stretch=False)
        tree.heading("session", text="Session")
        tree.column("session", width=180, stretch=True)
        for c in self._columns:
            tree.heading(c.lower(), text=c)
            tree.column(c.lower(), width=95 if c in ("Subject", "Group")
                        else 70, stretch=False)
        vsb = ttk.Scrollbar(tf, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        tree.bind("<Button-1>", self._on_tree_click)
        self._tree = tree
        self._fill_tree()

        # place near the button, kept on-screen
        win.update_idletasks()
        x = self._btn.winfo_rootx()
        y = self._btn.winfo_rooty() + self._btn.winfo_height() + 4
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"+{min(x, sw - w - 12)}+{min(y, sh - h - 48)}")

        win.bind("<Escape>", self._close_popup)
        win.protocol("WM_DELETE_WINDOW", self._close_popup)
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
            vals = [("✓" if inc else "✗"), n] + \
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
            txt = f"{self._label}: -"
        elif n_sel == total:
            txt = f"{self._label}: All ({total})"
        else:
            txt = f"{self._label}: {n_sel} of {total}"
        opened = self._popup is not None and self._popup.winfo_exists()
        self._btn.configure(text=txt + ("  ▴" if opened else "  ▾"))
