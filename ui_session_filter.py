# -*- coding: utf-8 -*-
"""
Shared "Sessions" dropdown filter for the analysis tabs.

A compact Menubutton that reads "Sessions: All (6)" / "Sessions: 4 of 6" and
opens a checkbutton menu (All / None / one entry per session). Default is
everything checked, so tabs behave exactly as before unless the user opts
out of specific sessions.

Usage:
    flt = SessionFilter(parent, on_change=self._refresh)
    flt.pack(...)
    flt.set_sessions(["mouse1_veh", "mouse2_veh", ...])   # idempotent
    keep = flt.selected()          # list of checked names
    if flt.allows(name): ...       # convenience membership test
"""

import tkinter as tk
from tkinter import ttk

try:
    from ui_tooltip import Tip
except Exception:  # pragma: no cover
    Tip = None


class SessionFilter(ttk.Menubutton):
    """Checkbutton-menu session picker; all sessions selected by default."""

    MAX_MENU = 40   # menus longer than this get scrollable columns

    def __init__(self, parent, on_change=None, label="Sessions", **kw):
        super().__init__(parent, direction="below", **kw)
        self._label = label
        self._on_change = on_change
        self._vars = {}          # name -> tk.BooleanVar (insertion-ordered)
        self._menu = tk.Menu(self, tearoff=False)
        self.configure(menu=self._menu)
        self._update_text()
        if Tip is not None:
            Tip(self, "Choose which sessions feed this tab's graphs and\n"
                      "stats. Everything is included by default; untick\n"
                      "sessions to exclude them.")

    # ── public API ─────────────────────────────────────────────────────────

    def set_sessions(self, names):
        """Declare the available sessions. Keeps existing tick states for
        names that persist; new names start ticked."""
        names = [str(n) for n in names]
        old = self._vars
        self._vars = {}
        for n in names:
            if n in old:
                self._vars[n] = old[n]
            else:
                self._vars[n] = tk.BooleanVar(master=self, value=True)
        self._rebuild_menu()
        self._update_text()

    def selected(self):
        """Names currently ticked (in declared order)."""
        return [n for n, v in self._vars.items() if v.get()]

    def allows(self, name):
        """True when `name` is unknown (never filter surprises) or ticked."""
        v = self._vars.get(str(name))
        return True if v is None else bool(v.get())

    def all_selected(self):
        return all(v.get() for v in self._vars.values())

    def select_all(self, value=True):
        for v in self._vars.values():
            v.set(bool(value))
        self._changed()

    # ── internals ──────────────────────────────────────────────────────────

    def _rebuild_menu(self):
        self._menu.delete(0, "end")
        self._menu.add_command(label="All",
                               command=lambda: self.select_all(True))
        self._menu.add_command(label="None",
                               command=lambda: self.select_all(False))
        self._menu.add_separator()
        for i, (n, v) in enumerate(self._vars.items()):
            self._menu.add_checkbutton(
                label=n, variable=v, command=self._changed,
                columnbreak=(i > 0 and i % self.MAX_MENU == 0))

    def _changed(self):
        self._update_text()
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                pass

    def _update_text(self):
        total = len(self._vars)
        n_sel = len(self.selected())
        if total == 0:
            txt = f"{self._label}: —"
        elif n_sel == total:
            txt = f"{self._label}: All ({total})"
        else:
            txt = f"{self._label}: {n_sel} of {total}"
        self.configure(text=txt + "  ▾")
