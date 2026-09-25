"""ui_tooltip.py - minimal hover tooltip shared by the analysis tabs.

Standalone (no PixelPaws_GUI import) so tab modules stay circular-import-free.
"""

import tkinter as tk


class Tip:
    """Attach a delayed hover tooltip to a widget.

    Usage: Tip(widget, "explanation").  The tip follows the widget it is
    bound to; multi-line text wraps at ``wraplength`` pixels.
    """

    DELAY_MS = 550

    def __init__(self, widget, text, wraplength=320):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self._after_id = None
        self._tipwin = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _evt=None):
        self._cancel()
        self._after_id = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tipwin is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tk.Label(tw, text=self.text, justify="left", background="#ffffe0",
                     foreground="#333333", relief="solid", borderwidth=1,
                     wraplength=self.wraplength, font=("Segoe UI", 9),
                     padx=6, pady=4).pack()
            self._tipwin = tw
        except Exception:
            self._tipwin = None

    def _hide(self, _evt=None):
        self._cancel()
        if self._tipwin is not None:
            try:
                self._tipwin.destroy()
            except Exception:
                pass
            self._tipwin = None


def collapsible(parent, title, collapsed=True, **pack_kw):
    """A section with an always-visible clickable header that shows/hides
    its body. Returns the body frame to build content into.

    Implemented as a header Label + body Frame (an empty ttk.Labelframe
    renders with its caption clipped under ttkbootstrap, which made the
    collapsed state invisible).
    """
    from tkinter import ttk
    wrap = ttk.Frame(parent)
    wrap.pack(**pack_kw)
    shown = [not collapsed]
    hdr = ttk.Label(wrap,
                    text=("▾  " if not collapsed else "▸  ") + title,
                    cursor="hand2", font=("Segoe UI", 9, "bold"))
    hdr.pack(fill="x", anchor="w", pady=(2, 1))
    ttk.Separator(wrap, orient="horizontal").pack(fill="x", pady=(0, 2))
    body = ttk.Frame(wrap)
    if not collapsed:
        body.pack(fill="x")

    def _toggle(_evt=None):
        if shown[0]:
            body.pack_forget()
            hdr.configure(text="▸  " + title)
        else:
            body.pack(fill="x")
            hdr.configure(text="▾  " + title)
        shown[0] = not shown[0]

    hdr.bind("<Button-1>", _toggle)
    Tip(hdr, "Click to expand or collapse this section.")
    return body
