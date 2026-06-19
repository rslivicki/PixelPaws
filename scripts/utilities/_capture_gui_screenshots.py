"""Headless-ish screenshot capture for the PixelPaws GUI tabs.

Bypasses the startup wizard, loads a known-good project, then iterates
over each visible nav item and writes a PNG per tab to
`E:/PixelPaws/_audit_screenshots/`.

Run from a real Windows desktop session (not from a service or remote
ssh without an attached display) — PIL.ImageGrab needs the window to
actually be on screen. The window is forced topmost and given ~1.5 s
to settle before each grab so live matplotlib canvases render.

Usage:  py _capture_gui_screenshots.py [project_folder]
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab

PROJECT = (sys.argv[1] if len(sys.argv) > 1
           else r'E:/RSVIDS/Blackbox/2512_Blackbox_Formalin_Oxy/Left_paws')
OUT_DIR = r'E:/PixelPaws/_audit_screenshots'
os.makedirs(OUT_DIR, exist_ok=True)

# Patch out the startup wizard BEFORE importing PixelPaws_GUI so it
# never gets a chance to deiconify-then-block.
sys.path.insert(0, r'E:/PixelPaws')
import PixelPaws_GUI as ppg

_orig_init = ppg.PixelPawsGUI.__init__

def _patched_init(self, root):
    # Don't show the wizard — we'll set the project folder directly.
    self._show_startup_wizard = lambda *a, **k: None
    _orig_init(self, root)
    # Make sure the main window is visible.
    try:
        self.root.deiconify()
    except Exception:
        pass

ppg.PixelPawsGUI.__init__ = _patched_init


def _settle(root, ms=1500):
    """Pump the Tk event loop for *ms* milliseconds so widgets render."""
    end = time.time() + ms / 1000.0
    while time.time() < end:
        root.update_idletasks()
        root.update()
        time.sleep(0.02)


def _grab_window(root) -> 'PIL.Image.Image':
    root.lift()
    root.attributes('-topmost', True)
    _settle(root, 350)
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    w = root.winfo_width()
    h = root.winfo_height()
    bbox = (x, y, x + w, y + h)
    img = ImageGrab.grab(bbox=bbox, all_screens=True)
    root.attributes('-topmost', False)
    return img


def _safe_filename(text: str) -> str:
    keep = [c if (c.isalnum() or c in (' ', '_', '-')) else '_' for c in text]
    s = ''.join(keep).strip().replace(' ', '_')
    # Trim leading nav emoji code points / underscores
    while s and not s[0].isalnum():
        s = s[1:]
    return s or 'tab'


def main():
    root = tk.Tk()
    # Big enough that all widgets render at full size.
    root.geometry('1900x1100+40+40')

    try:
        app = ppg.PixelPawsGUI(root)
    except Exception:
        print('GUI construction failed:')
        traceback.print_exc()
        return

    # Drive the project loader directly — same path the wizard would use.
    print(f'[setup] loading project: {PROJECT}')
    try:
        app.current_project_folder.set(PROJECT)
    except Exception:
        traceback.print_exc()
    _settle(root, 2500)  # let _on_project_folder_changed fire + scans queue

    nav = app.notebook
    # Visible tabs only — `hide_item` packs them away
    visible_keys = []
    for key, widgets in nav._nav_widgets.items():
        if widgets['row'].winfo_ismapped():
            visible_keys.append(key)
    print(f'[setup] visible tabs: {visible_keys}')

    captured = []
    for key in visible_keys:
        try:
            print(f'[grab ] {key}')
            nav.select(key)
            _settle(root, 1400)  # generous — let matplotlib canvases redraw
            img = _grab_window(root)
            fname = f'{len(captured):02d}_{_safe_filename(key)}.png'
            path = os.path.join(OUT_DIR, fname)
            img.save(path)
            captured.append((key, path))
            print(f'         wrote {path}')
        except Exception as e:
            print(f'  ! {key}: {e}')
            traceback.print_exc()

    # Tear down cleanly — destroy() avoids the after-cancel error spam.
    try:
        root.destroy()
    except Exception:
        pass

    print('\nDone. Captured tabs:')
    for k, p in captured:
        print(f'  {k:30s} -> {p}')


if __name__ == '__main__':
    main()
