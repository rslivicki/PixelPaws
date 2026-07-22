"""PixelPaws keypoint labeler — standalone Tkinter Canvas app (NO napari, NO matplotlib).

We own the data model ( self.coords = {image: {bodypart: (x,y) or None}} ), and the canvas only DRAWS
from it — so there is no hidden state to desync (the root cause of every napari failure). Saves directly
to CollectedData in the project's single-level-index format, dropping all-off-frame frames.

TOOLS (toolbar / hotkeys) — left-drag behavior depends on the active tool:
  • Edit  (E) [default] .. drag a dot to move it; pick a bodypart (1-9 / panel) + click empty to place one.
  • Pan   (H) ........... left-drag moves the image.   (also: HOLD SPACE + left-drag pans in any tool)
  • Zoom  (Z) ........... left-drag a box -> zoom into it; a plain click zooms in a step there.
IMAGE (for dark frames):  Brightness / Contrast / Gamma sliders · [Auto] = make a dark frame readable.
ALWAYS available in any tool:  mouse-wheel = zoom (on cursor) · right-drag = pan · +/- = zoom ·
  arrow keys / ◄▲▼► = pan · F = fit/reset view.
OTHER:  Delete/Backspace/[Delete pt] = mark a point off-frame · r = reset frame to prefill · Ctrl+Z = undo ·
  s = save · b/n = prev/next frame (auto-saves) · [Folder ✓] = done -> next folder · ? = help.
Works through every labeled-data folder that has a `_scratch_to_label.txt` marker.

PROJECT/SCORER awareness: at startup this reads an optional `_labeler_config.json`
sidecar (same directory as this script, or the path in env PIXELPAWS_LABELER_CONFIG) with
keys {ldir, scorer, bps, skeleton, colors, marker}. Any key absent falls back to the
defaults below, so a plain double-click with no sidecar behaves exactly as before. The
PixelPaws GUI writes this sidecar pointing at the active project + its scorer before
launching, so extractor output (CollectedData_<scorer>) and the labeler always agree.
"""
import os, glob, sys, shutil, time, json

# --- PyInstaller --windowed hardening -------------------------------------------------------
# A windowed (no-console) frozen build sets sys.stdout/sys.stderr to None. Native libraries in
# the HDF5 save path (numexpr / PyTables) write to them and can crash or deadlock; give them a
# real sink, and pin their thread pools to 1 so they never hang spinning up worker threads in a
# GUI process. No effect when run normally from source (stdout is a real stream).
if getattr(sys, "frozen", False):
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if sys.stdout is None or sys.stderr is None:
        _sink = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = _sink
        if sys.stderr is None:
            sys.stderr = _sink

import numpy as np, pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageOps, ImageEnhance

_DEF_LDIR = r"E:\Sync_from_lab\2511_RSGNKK_Blackbox\labeled-data"
_DEF_SCORER = "AlexZ"
_DEF_BPS = ['tailtip', 'tailbase', 'centroid', 'neck', 'snout', 'hlpaw', 'hrpaw', 'flpaw', 'frpaw']
_DEF_COLORS = {'tailtip': '#999999', 'tailbase': '#cccccc', 'centroid': '#33dd33', 'neck': '#22aa22',
               'snout': '#ee44ee', 'hlpaw': '#ffdd00', 'hrpaw': '#ff8800', 'flpaw': '#00dddd', 'frpaw': '#3399ff'}
_DEF_SKELETON = [('snout', 'neck'), ('neck', 'centroid'), ('centroid', 'tailbase'), ('tailbase', 'tailtip'),
                 ('neck', 'frpaw'), ('neck', 'flpaw'), ('centroid', 'hrpaw'), ('centroid', 'hlpaw')]
_PALETTE = ['#3399ff', '#ff8800', '#33dd33', '#ee44ee', '#ffdd00', '#00dddd', '#22aa22', '#cc5555', '#999999']


def _base_dir():
    # PyInstaller-frozen build: resolve the sidecar + a relative ldir next to the .exe so a
    # portable review package (exe + _labeler_config.json + labeled-data\) is self-contained.
    # Running from source (or launched by the GUI with an absolute ldir): unchanged.
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _load_config():
    p = os.environ.get("PIXELPAWS_LABELER_CONFIG",
                       os.path.join(_base_dir(), "_labeler_config.json"))
    try:
        return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
    except Exception as e:
        print(f"! could not read labeler config {p}: {e}", flush=True)
        return {}


_CFG = _load_config()
LDIR = _CFG.get("ldir", _DEF_LDIR)
if not os.path.isabs(LDIR):
    LDIR = os.path.join(_base_dir(), LDIR)
SCORER = _CFG.get("scorer", _DEF_SCORER)
BPS = list(_CFG.get("bps", _DEF_BPS))
COLS = pd.MultiIndex.from_product([[SCORER], BPS, ['x', 'y']], names=['scorer', 'bodyparts', 'coords'])
# colors: use sidecar/default map, then fill any missing bodypart from a rotating palette
COLORS = dict(_DEF_COLORS); COLORS.update(_CFG.get("colors", {}))
for _n, _bp in enumerate(BPS):
    COLORS.setdefault(_bp, _PALETTE[_n % len(_PALETTE)])
SKELETON = [tuple(e) for e in _CFG.get("skeleton", _DEF_SKELETON) if e[0] in BPS and e[1] in BPS]
MARKER = _CFG.get("marker", "_scratch_to_label.txt")
HIT = 14          # px (image space) to grab a keypoint
R = 6             # default dot radius (canvas px)
PAN = 90          # px per pan step (buttons / arrow keys)


def _img_key(i):
    """Bare image filename for a CollectedData index entry — handles both the flat single-level
    path string ("labeled-data\\<session>\\img###.png") and the 3-level DLC MultiIndex tuple
    ("labeled-data", "<session>", "img###.png")."""
    if isinstance(i, tuple):
        return i[-1]
    return str(i).replace("/", "\\").split("\\")[-1]


def list_folders():
    return sorted(d for d in os.listdir(LDIR)
                  if os.path.isfile(os.path.join(LDIR, d, MARKER))
                  and glob.glob(os.path.join(LDIR, d, "img*.png")))


class Labeler:
    def __init__(self, root):
        self.root = root
        root.title("PixelPaws keypoint labeler")
        self.folder = None
        self.imgs = []
        self.coords = {}
        self.prefill = {}
        self.i = 0
        self.active = 5            # default hlpaw
        self.sel = None            # selected keypoint (bodypart name) for drag/delete
        self.dragging = False
        self.zoom = 1.0
        self.ox = self.oy = 0.0    # image origin on canvas (px)
        self.base = None           # original PIL frame
        self.adj = None            # brightness/contrast/gamma-adjusted PIL frame (display only)
        self.tkimg = None
        self._panning = None
        self._fitted = False
        self.tool = "edit"         # "edit" | "pan" | "zoom"
        self._zbox = None
        self._zrect = None
        # image-adjust state (display only — never affects saved coordinates)
        self.bright = 0.0          # -100..100
        self.contrast = 1.0        # 0.3..3
        self.gamma = 1.6           # mild boost so dark frames open visible
        self._autoc = False        # autocontrast on (set by [Auto])
        self.dotR = R
        self.undo_stack = []       # (image, coords-snapshot) per edit gesture
        self._spacepan = False
        self._remaining = 0
        self._dirty = False        # unsaved edits since last save
        self._saved_time = ""
        self._build_ui()
        self._bind()
        if not self.load_next_folder():
            messagebox.showinfo("Done", "No folders need labeling.")
            root.after(50, root.destroy)

    # ---------- UI ----------
    def _mk_slider(self, parent, label, lo, hi, val, cb):
        ttk.Label(parent, text=label).pack(side="left", padx=(8, 1))
        s = ttk.Scale(parent, from_=lo, to=hi, value=val, length=90,
                      command=lambda v: cb(float(v)))
        s.pack(side="left")
        return s

    def _build_ui(self):
        tb = ttk.Frame(self.root); tb.pack(fill="x", side="top")
        ttk.Label(tb, text="Tool:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(8, 2))
        self.toolvar = tk.StringVar(value="edit")
        for _name, _label in (("edit", "✎ Edit / label  (E)"), ("pan", "✋ Pan  (H)"),
                              ("zoom", "🔍 Zoom-box  (Z)")):
            ttk.Radiobutton(tb, text=_label, value=_name, variable=self.toolvar,
                            command=lambda: self.set_tool(self.toolvar.get())).pack(side="left", padx=2)
        ttk.Label(tb, text="    wheel=zoom · right-drag / hold-Space-drag = pan",
                  foreground="#666").pack(side="left", padx=8)
        ttk.Button(tb, text="?  Help", command=self.show_help).pack(side="right", padx=8)
        self.folder_box = ttk.Combobox(tb, state="readonly", width=28)
        self.folder_box.pack(side="right", padx=2)
        self.folder_box.bind("<<ComboboxSelected>>", self.on_pick_folder)
        ttk.Label(tb, text="Subject:", font=("Segoe UI", 9, "bold")).pack(side="right", padx=(8, 2))

        adj = ttk.Frame(self.root); adj.pack(fill="x", side="top")
        ttk.Label(adj, text="Image:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(8, 2))
        self._sl_b = self._mk_slider(adj, "Bright", -100, 100, self.bright, self._set_bright)
        self._sl_c = self._mk_slider(adj, "Contrast", 30, 300, self.contrast * 100, self._set_contrast)
        self._sl_g = self._mk_slider(adj, "Gamma", 30, 300, self.gamma * 100, self._set_gamma)
        ttk.Button(adj, text="Auto", command=self.auto_adjust).pack(side="left", padx=(6, 2))
        ttk.Button(adj, text="Reset img", command=self.reset_adjust).pack(side="left", padx=2)
        ttk.Label(adj, text="    Dot size:").pack(side="left", padx=(12, 1))
        ttk.Scale(adj, from_=2, to=16, value=self.dotR, length=80,
                  command=lambda v: self._set_dot(float(v))).pack(side="left")

        main = ttk.Frame(self.root); main.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(main, bg="#111", width=820, height=820, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(main, width=180); side.pack(side="right", fill="y")
        ttk.Label(side, text="BODYPARTS  (click / 1-9)", font=("Segoe UI", 9, "bold")).pack(pady=(8, 4))
        self.bp_rows = []
        for n, bp in enumerate(BPS, 1):
            row = tk.Frame(side, cursor="hand2"); row.pack(fill="x", padx=6, pady=1)
            sw = tk.Canvas(row, width=14, height=14, highlightthickness=0); sw.pack(side="left")
            sw.create_oval(2, 2, 12, 12, fill=COLORS[bp], outline="")
            lab = tk.Label(row, text=f"{n} {bp}", anchor="w", width=14); lab.pack(side="left")
            dot = tk.Label(row, text="•", fg="#666"); dot.pack(side="right")
            for w in (row, sw, lab, dot):
                w.bind("<Button-1>", lambda e, k=n - 1: self.set_active(k))
            self.bp_rows.append((row, lab, dot))

        bottom = ttk.Frame(self.root); bottom.pack(fill="x", side="bottom")
        nav = ttk.Frame(bottom); nav.pack(fill="x", pady=2)
        ttk.Button(nav, text="◀ Prev (b)", command=self.prev).pack(side="left", padx=3)
        self.counter = ttk.Label(nav, text="", width=18, anchor="center"); self.counter.pack(side="left")
        ttk.Button(nav, text="Next (n) ▶", command=self.next).pack(side="left", padx=3)
        ttk.Button(nav, text="Save", command=self.save).pack(side="left", padx=12)
        ttk.Button(nav, text="Undo (Ctrl+Z)", command=self.undo).pack(side="left", padx=3)
        ttk.Button(nav, text="Reset frame (r)", command=self.reset_frame).pack(side="left", padx=3)
        ttk.Button(nav, text="Delete pt (Del)", command=self.delete_sel).pack(side="left", padx=3)
        ttk.Button(nav, text="Folder done ✓", command=self.folder_done).pack(side="left", padx=12)
        ttk.Button(nav, text="–", width=3, command=lambda: self.zoom_by(1 / 1.25)).pack(side="left", padx=1)
        ttk.Button(nav, text="+", width=3, command=lambda: self.zoom_by(1.25)).pack(side="left", padx=1)
        ttk.Button(nav, text="Fit (f)", command=self.fit).pack(side="left", padx=3)
        ttk.Label(nav, text="Pan:").pack(side="left", padx=(8, 1))
        for _t, _dx, _dy in (("◄", PAN, 0), ("▲", 0, PAN), ("▼", 0, -PAN), ("►", -PAN, 0)):
            ttk.Button(nav, text=_t, width=3,
                       command=lambda dx=_dx, dy=_dy: self.pan(dx, dy)).pack(side="left", padx=1)
        self.savelbl = ttk.Label(nav, text="", font=("Segoe UI", 10, "bold")); self.savelbl.pack(side="right", padx=10)
        self.status = ttk.Label(bottom, text="", anchor="w"); self.status.pack(fill="x")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _bind(self):
        c = self.canvas
        c.bind("<Button-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_motion)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Button-3>", lambda e: setattr(self, "_panning", (e.x, e.y)))
        c.bind("<B3-Motion>", self.on_pan)
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Configure>", self._on_configure)
        # bind_all so keys (Delete, 1-9, n/b, Ctrl+Z, ...) fire even when a slider/button has focus
        self.root.bind_all("<Key>", self.on_key)
        self.root.bind_all("<KeyPress-space>", self._space_down)
        self.root.bind_all("<KeyRelease-space>", self._space_up)

    def _on_configure(self, e):
        if not self._fitted and self.base is not None and self.canvas.winfo_width() > 50:
            self.fit()
        else:
            self.draw()

    def set_tool(self, name):
        self.tool = name
        try:
            self.toolvar.set(name)
        except Exception:
            pass
        self._set_cursor()

    def _set_cursor(self):
        cur = "fleur" if self._spacepan else \
            {"edit": "crosshair", "pan": "fleur", "zoom": "tcross"}.get(self.tool, "arrow")
        try:
            self.canvas.config(cursor=cur)
        except Exception:
            pass

    # ---------- image adjustment (display only) ----------
    def _set_bright(self, v): self.bright = v; self._apply_adjust(); self.draw()
    def _set_contrast(self, v): self.contrast = v / 100.0; self._apply_adjust(); self.draw()
    def _set_gamma(self, v): self.gamma = max(0.1, v / 100.0); self._apply_adjust(); self.draw()
    def _set_dot(self, v): self.dotR = int(round(v)); self.draw()

    def _apply_adjust(self):
        if self.base is None:
            self.adj = None; return
        img = self.base
        if self._autoc:
            img = ImageOps.autocontrast(img, cutoff=1)
        if abs(self.bright) > 0.5:
            img = ImageEnhance.Brightness(img).enhance(1 + self.bright / 100.0)
        if abs(self.contrast - 1) > 0.01:
            img = ImageEnhance.Contrast(img).enhance(self.contrast)
        if abs(self.gamma - 1) > 0.01:
            inv = 1.0 / max(0.1, self.gamma)
            lut = [int(min(255, 255 * ((i / 255.0) ** inv))) for i in range(256)] * 3
            img = img.point(lut)
        self.adj = img

    def _sync_sliders(self):
        for s, val in ((self._sl_b, self.bright), (self._sl_c, self.contrast * 100),
                       (self._sl_g, self.gamma * 100)):
            try: s.set(val)
            except Exception: pass

    def auto_adjust(self):
        self._autoc = True; self.bright = 0.0; self.contrast = 1.0; self.gamma = 2.2
        self._apply_adjust(); self._sync_sliders(); self.draw()

    def reset_adjust(self):
        self._autoc = False; self.bright = 0.0; self.contrast = 1.0; self.gamma = 1.0
        self._apply_adjust(); self._sync_sliders(); self.draw()

    # ---------- folders / data ----------
    def load_next_folder(self):
        folders = list_folders()
        if not folders:
            return False
        self._remaining = len(folders)
        self.load_folder(folders[0])
        return True

    def _refresh_folder_menu(self):
        folders = list_folders()
        self._remaining = len(folders)
        self.folder_box["values"] = folders
        if self.folder:
            self.folder_box.set(self.folder)

    def on_pick_folder(self, _evt=None):
        name = self.folder_box.get()
        if name == self.folder or name not in list_folders():
            if self.folder:
                self.folder_box.set(self.folder)
            return
        try:
            self.save()
            self.load_folder(name)
        except Exception as e:
            if self.folder:
                self.folder_box.set(self.folder)
            messagebox.showerror("Load failed", f"Could not open {name}:\n{e}")

    def load_folder(self, name):
        self.folder = name
        self.dir = os.path.join(LDIR, self.folder)
        self.imgs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(self.dir, "img*.png")))
        cd = pd.read_hdf(os.path.join(self.dir, f"CollectedData_{SCORER}.h5"))
        sc = cd.columns.get_level_values(0)[0]
        # Index entries come in two shapes: a flat single-level path string
        # ("labeled-data\\<session>\\img###.png", written by this labeler / the extractor) OR a
        # 3-level DLC MultiIndex tuple ("labeled-data", "<session>", "img###.png"). Key both by the
        # bare image filename so existing labels match the frames on disk either way.
        by = {_img_key(i): i for i in cd.index}
        self.coords = {}
        for im in self.imgs:
            r = by.get(im); d = {}
            for bp in BPS:
                if r is not None:
                    x = cd.loc[r, (sc, bp, 'x')]; y = cd.loc[r, (sc, bp, 'y')]
                    d[bp] = (float(x), float(y)) if pd.notna(x) and pd.notna(y) else None
                else:
                    d[bp] = None
            self.coords[im] = d
        self.prefill = {im: dict(v) for im, v in self.coords.items()}
        self.i = 0; self.sel = None; self.undo_stack = []
        self.load_frame()
        self._refresh_folder_menu()

    def load_frame(self):
        self.base = Image.open(os.path.join(self.dir, self.imgs[self.i])).convert("RGB")
        self.sel = None
        self._fitted = False
        self._apply_adjust()       # carry the brightness/gamma settings to the new frame
        self.fit()

    def fit(self):
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            self.root.after(40, self.fit); return
        if self.base:
            self.zoom = min(cw / self.base.width, ch / self.base.height)
            self.ox = (cw - self.base.width * self.zoom) / 2
            self.oy = (ch - self.base.height * self.zoom) / 2
        self._fitted = True
        self.draw()

    # ---------- transforms ----------
    def to_canvas(self, x, y):
        return x * self.zoom + self.ox, y * self.zoom + self.oy

    def to_image(self, cx, cy):
        return (cx - self.ox) / self.zoom, (cy - self.oy) / self.zoom

    # ---------- draw ----------
    def draw(self):
        c = self.canvas; c.delete("all")
        src = self.adj if self.adj is not None else self.base
        if src is None:
            return
        zw, zh = int(src.width * self.zoom), int(src.height * self.zoom)
        if zw > 0 and zh > 0:
            self.tkimg = ImageTk.PhotoImage(src.resize((zw, zh), Image.BILINEAR))
            c.create_image(self.ox, self.oy, anchor="nw", image=self.tkimg)
        im = self.imgs[self.i]
        # skeleton (always on) — drawn under the keypoints
        for a, b in SKELETON:
            pa = self.coords[im].get(a); pb = self.coords[im].get(b)
            if pa and pb:
                ax, ay = self.to_canvas(*pa); bx, by = self.to_canvas(*pb)
                c.create_line(ax, ay, bx, by, fill="#ff4444", width=2)
        for bp in BPS:
            p = self.coords[im][bp]
            if not p:
                continue
            cx, cy = self.to_canvas(*p)
            cur = (bp == BPS[self.active]); seld = (bp == self.sel)
            rr = self.dotR + (3 if cur or seld else 0)
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill=COLORS[bp],
                          outline="red" if seld else ("white" if cur else "black"),
                          width=3 if seld else (2 if cur else 1))
            c.create_text(cx + rr + 2, cy, text=bp, anchor="w", fill="white", font=("Segoe UI", 8))
        self._refresh_side()
        n = sum(1 for bp in BPS if self.coords[im][bp])
        self.counter.config(text=f"Frame {self.i+1}/{len(self.imgs)}")
        self.status.config(text=f"{self.folder}  ·  frame {self.i+1}/{len(self.imgs)}  ·  {n}/9 labeled"
                                f"  ·  editing: {self.active+1} {BPS[self.active]}"
                                f"  ·  {self._remaining} folder(s) left  ·  zoom {self.zoom:.1f}x")
        self._refresh_savelbl()

    def _refresh_savelbl(self):
        if self._dirty:
            self.savelbl.config(text="●  UNSAVED edits  (Ctrl+S)", foreground="#cc6600")
        else:
            t = f"  {self._saved_time}" if self._saved_time else ""
            self.savelbl.config(text=f"✓  all saved{t}", foreground="#1a9a1a")

    def _refresh_side(self):
        im = self.imgs[self.i]
        for k, (row, lab, dot) in enumerate(self.bp_rows):
            present = self.coords[im][BPS[k]] is not None
            dot.config(text="●" if present else "○", fg="#3c3" if present else "#a33")
            if k == self.active:
                row.config(bg="#cde"); lab.config(bg="#cde", font=("Segoe UI", 9, "bold"))
            else:
                row.config(bg=self.root.cget("bg")); lab.config(bg=self.root.cget("bg"), font=("Segoe UI", 9))

    # ---------- undo ----------
    def _push_undo(self):
        im = self.imgs[self.i]
        self.undo_stack.append((im, dict(self.coords[im])))
        if len(self.undo_stack) > 80:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            self.status.config(text="nothing to undo"); return
        im, snap = self.undo_stack.pop()
        if im in self.coords:
            self.coords[im] = snap; self._dirty = True
            if self.imgs[self.i] != im and im in self.imgs:
                self.i = self.imgs.index(im); self.load_frame()
            else:
                self.sel = None; self.draw()

    # ---------- actions ----------
    def set_active(self, k):
        self.active = k
        im = self.imgs[self.i]
        self.sel = BPS[k] if self.coords[im][BPS[k]] else None
        self.set_tool("edit")
        self.draw()

    def hit(self, cx, cy):
        im = self.imgs[self.i]; best = None; bd = HIT
        ix, iy = self.to_image(cx, cy)
        for bp in BPS:
            p = self.coords[im][bp]
            if p:
                d = ((p[0] - ix) ** 2 + (p[1] - iy) ** 2) ** 0.5
                if d < bd:
                    bd = d; best = bp
        return best

    def on_press(self, e):
        self.canvas.focus_set()                # so keyboard (Delete, 1-9, ...) goes to the canvas
        if self._spacepan or self.tool == "pan":
            self._panning = (e.x, e.y); return
        if self.tool == "zoom":
            self._zbox = (e.x, e.y); self._zrect = None; return
        # --- edit tool ---
        h = self.hit(e.x, e.y)
        if h:                                  # grab the existing keypoint under the cursor -> drag it
            self._push_undo()
            self.sel = h; self.active = BPS.index(h); self.dragging = True
        elif self.coords[self.imgs[self.i]][BPS[self.active]] is None:
            # empty click + the picked bodypart is MISSING -> place it (won't move a present point)
            self._push_undo()
            self.coords[self.imgs[self.i]][BPS[self.active]] = self.to_image(e.x, e.y)
            self.sel = BPS[self.active]; self.dragging = True; self._dirty = True
        else:
            self.sel = None
        self.draw()

    def on_motion(self, e):
        if (self._spacepan or self.tool == "pan") and self._panning:
            self.ox += e.x - self._panning[0]; self.oy += e.y - self._panning[1]
            self._panning = (e.x, e.y); self.draw()
        elif self.tool == "zoom" and self._zbox:
            if self._zrect:
                self.canvas.delete(self._zrect)
            self._zrect = self.canvas.create_rectangle(self._zbox[0], self._zbox[1], e.x, e.y,
                                                       outline="#33ccff", width=2, dash=(4, 2))
        elif self.dragging and self.sel:
            self.coords[self.imgs[self.i]][self.sel] = self.to_image(e.x, e.y)
            self._dirty = True; self.draw()

    def on_release(self, e):
        if self.tool == "zoom" and self._zbox and not self._spacepan:
            x0, y0 = self._zbox; x1, y1 = e.x, e.y
            if self._zrect:
                self.canvas.delete(self._zrect); self._zrect = None
            self._zbox = None
            if abs(x1 - x0) > 12 and abs(y1 - y0) > 12:
                self._zoom_to_box(x0, y0, x1, y1)
            else:
                self._zoom_about(e.x, e.y, 1.6)
        if self._spacepan or self.tool == "pan":
            self._panning = None
        self.dragging = False

    def _zoom_to_box(self, x0, y0, x1, y1):
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        ix0, iy0 = self.to_image(min(x0, x1), min(y0, y1))
        ix1, iy1 = self.to_image(max(x0, x1), max(y0, y1))
        bw = max(1.0, ix1 - ix0); bh = max(1.0, iy1 - iy0)
        self.zoom = max(0.2, min(12.0, min(cw / bw, ch / bh)))
        cx = (ix0 + ix1) / 2; cy = (iy0 + iy1) / 2
        self.ox = cw / 2 - cx * self.zoom; self.oy = ch / 2 - cy * self.zoom
        self.draw()

    def on_pan(self, e):
        if self._panning:
            self.ox += e.x - self._panning[0]; self.oy += e.y - self._panning[1]
            self._panning = (e.x, e.y); self.draw()

    def on_wheel(self, e):
        d = getattr(e, "delta", 0)
        self._zoom_about(e.x, e.y, 1.2 if d > 0 else 1 / 1.2)

    def _zoom_about(self, cx, cy, f):
        ix, iy = self.to_image(cx, cy)
        self.zoom = max(0.2, min(12.0, self.zoom * f))
        self.ox = cx - ix * self.zoom; self.oy = cy - iy * self.zoom
        self.draw()

    def zoom_by(self, f):
        self._zoom_about(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, f)

    def pan(self, dx, dy):
        self.ox += dx; self.oy += dy; self.draw()

    def _space_down(self, e):
        if not self._spacepan:
            self._spacepan = True; self._set_cursor()

    def _space_up(self, e):
        self._spacepan = False; self._panning = None; self._set_cursor()

    def delete_sel(self):
        if self.sel:
            self._push_undo()
            self.coords[self.imgs[self.i]][self.sel] = None; self.sel = None; self._dirty = True; self.draw()

    def reset_frame(self):
        self._push_undo()
        self.coords[self.imgs[self.i]] = dict(self.prefill[self.imgs[self.i]])
        self.sel = None; self._dirty = True; self.draw()

    def next(self):
        self.save(); self.i = min(self.i + 1, len(self.imgs) - 1); self.load_frame()

    def prev(self):
        self.save(); self.i = max(self.i - 1, 0); self.load_frame()

    def show_help(self):
        messagebox.showinfo("PixelPaws labeler — shortcuts",
            "TOOLS:\n"
            "  E = Edit/label   H = Pan   Z = Zoom-box\n\n"
            "EDIT:\n"
            "  • drag a dot to move it\n"
            "  • pick a bodypart (1-9 or side panel), then click empty space to place it\n"
            "  • Del / Backspace / Delete pt = mark the selected point off-frame\n\n"
            "VIEW:\n"
            "  • mouse wheel = zoom (on cursor)   +/- = zoom\n"
            "  • right-drag OR hold Space + drag = pan   arrows / ◄▲▼► = pan\n"
            "  • F = fit / reset view\n\n"
            "IMAGE (dark frames):\n"
            "  • Brightness / Contrast / Gamma sliders   Auto = make a dark frame readable\n\n"
            "FRAMES:\n"
            "  • n = next   b = prev  (both auto-save)\n"
            "  • r = reset frame to model prefill   Ctrl+Z = undo   s = save\n"
            "  • Folder done ✓ = finish this folder, advance to the next")

    def on_key(self, e):
        k = e.keysym
        if (e.state & 0x4) and k in ("z", "Z"):    # Ctrl+Z
            self.undo(); return
        if k in [str(n) for n in range(1, 10)]:
            self.set_active(int(k) - 1)
        elif k == "n":
            self.next()
        elif k == "b":
            self.prev()
        elif k == "Left":
            self.pan(PAN, 0)
        elif k == "Right":
            self.pan(-PAN, 0)
        elif k == "Up":
            self.pan(0, PAN)
        elif k == "Down":
            self.pan(0, -PAN)
        elif k in ("Delete", "BackSpace", "d"):
            self.delete_sel()
        elif k == "r":
            self.reset_frame()
        elif k in ("f", "F", "Home"):
            self.fit()
        elif k == "s":
            self.save()
        elif k in ("plus", "equal", "KP_Add"):
            self.zoom_by(1.25)
        elif k in ("minus", "underscore", "KP_Subtract"):
            self.zoom_by(1 / 1.25)
        elif k in ("e", "E"):
            self.set_tool("edit")
        elif k in ("h", "H"):
            self.set_tool("pan")
        elif k in ("z", "Z"):
            self.set_tool("zoom")
        elif k in ("question", "F1"):
            self.show_help()
        elif k in ("q", "Escape"):
            self.on_close()

    # ---------- save ----------
    def save(self):
        if not self.folder:
            return
        rows, idx = [], []
        for im in self.imgs:
            row = []
            for bp in BPS:
                p = self.coords[im][bp]
                row += [p[0], p[1]] if p else [np.nan, np.nan]
            rows.append(row); idx.append(os.path.join("labeled-data", self.folder, im))
        cd = pd.DataFrame(rows, columns=COLS, index=idx)
        # Save is purely ADDITIVE: it writes EVERY current frame (all 9 coords each) and removes
        # nothing. Saving mid-folder (e.g. frame 7/13) or re-saving later is fully idempotent and can
        # never drop an as-yet-unlabeled frame. Empty (all-NaN) frames are purged ONLY on the explicit
        # "Folder done" finalize (see _purge_empty), so an incremental Ctrl+S is always safe.
        cd.to_hdf(os.path.join(self.dir, f"CollectedData_{SCORER}.h5"), key="df_with_missing", mode="w")
        cd.to_csv(os.path.join(self.dir, f"CollectedData_{SCORER}.csv"))
        self._dirty = False
        self._saved_time = time.strftime("%H:%M:%S")
        try:
            self._refresh_savelbl()
            self.status.config(text=f"✓ saved all {len(cd)} frames of {self.folder} at {self._saved_time}")
        except Exception:
            pass

    def _purge_empty(self):
        """Drop frames with ZERO labeled points: move each image to _removed_empty and remove it from
        the working set. Called ONLY on the explicit Folder-done finalize -- never on incremental save --
        so a mid-folder Ctrl+S can't discard a frame you simply haven't labeled yet. Returns #purged."""
        empties = [im for im in self.imgs if all(self.coords[im][bp] is None for bp in BPS)]
        if not empties:
            return 0
        rem = os.path.join(LDIR, "_removed_empty", self.folder); os.makedirs(rem, exist_ok=True)
        for im in empties:
            src = os.path.join(self.dir, im)
            if os.path.isfile(src):
                dst = os.path.join(rem, im)
                # the PNG must leave self.dir, else it reappears as a blank frame on next load;
                # clear any stale same-named copy first, delete the source if the move still fails.
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.move(src, dst)
                except Exception:
                    try: os.remove(src)
                    except Exception: pass
        self.imgs = [im for im in self.imgs if im not in empties]
        self.coords = {im: self.coords[im] for im in self.imgs}
        self.i = min(self.i, max(0, len(self.imgs) - 1))
        return len(empties)

    def folder_done(self):
        n = self._purge_empty()
        self.save()
        mk = os.path.join(self.dir, MARKER)
        if os.path.isfile(mk):
            try: os.remove(mk)
            except Exception: pass
        if not self.load_next_folder():
            messagebox.showinfo("Done", "All folders labeled! Tell Claude to merge + retrain.")
            self.root.destroy()

    def on_close(self):
        try:
            if self._dirty:      # only persist on close when there are actual unsaved edits —
                self.save()      # never rewrite a folder that was merely opened and viewed
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1100x920")
    Labeler(root)
    root.mainloop()
    return 0


def _selftest():
    """Headless check that the (possibly frozen) runtime can round-trip CollectedData via
    PyTables — the highest-risk path in a packaged build. Prints SELFTEST OK / FAIL, exits."""
    import tempfile
    try:
        d = tempfile.mkdtemp()
        h5 = os.path.join(d, f"CollectedData_{SCORER}.h5")
        df = pd.DataFrame([[1.0, 2.0] * len(BPS)], columns=COLS,
                          index=[os.path.join("labeled-data", "selftest", "img000.png")])
        df.to_hdf(h5, key="df_with_missing", mode="w")
        df.to_csv(os.path.join(d, f"CollectedData_{SCORER}.csv"))
        back = pd.read_hdf(h5)
        ok = back.shape == df.shape
        msg = f"SELFTEST {'OK' if ok else 'FAIL'} rows={len(back)} cols={back.shape[1]} scorer={SCORER}"
        rc = 0 if ok else 1
    except Exception as e:
        msg = f"SELFTEST FAIL {type(e).__name__}: {e}"
        rc = 1
    # Write the result file FIRST — in a --windowed build sys.stdout is None, so print() must not
    # be the thing that records the outcome.
    out = os.environ.get("PIXELPAWS_SELFTEST_OUT",
                         os.path.join(_base_dir(), "_selftest_result.txt"))
    try:
        open(out, "w", encoding="utf-8").write(msg + "\n")
    except Exception:
        pass
    try:
        print(msg)
    except Exception:
        pass
    return rc


def _check():
    """Headless check that the folder queue resolves in a (possibly frozen) build: report how
    many labeled-data folders list_folders() finds under the resolved LDIR. Writes + prints."""
    try:
        fs = list_folders()
        # Also load the first folder's labels the same way the GUI does, and count how many frames
        # actually resolve a labeled point — this catches index-format mismatches (flat vs 3-level)
        # that would otherwise make existing labels silently invisible.
        detail = ""
        if fs:
            d = os.path.join(LDIR, fs[0])
            imgs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "img*.png")))
            cd = pd.read_hdf(os.path.join(d, f"CollectedData_{SCORER}.h5"))
            sc = cd.columns.get_level_values(0)[0]
            by = {_img_key(i): i for i in cd.index}
            matched = sum(1 for im in imgs if im in by
                          and bool(pd.notna(cd.loc[by[im], sc]).any()))
            detail = f" | first-folder frames={len(imgs)} with-labels={matched}"
        msg = f"CHECK OK ldir={LDIR} folders={len(fs)} first={fs[0] if fs else '-'}{detail}"
    except Exception as e:
        msg = f"CHECK FAIL {type(e).__name__}: {e} ldir={LDIR}"
    out = os.environ.get("PIXELPAWS_SELFTEST_OUT",
                         os.path.join(_base_dir(), "_selftest_result.txt"))
    try:
        open(out, "w", encoding="utf-8").write(msg + "\n")
    except Exception:
        pass
    try:
        print(msg)
    except Exception:
        pass
    return 0 if msg.startswith("CHECK OK") else 1


if __name__ == "__main__":
    # A PyInstaller-frozen exe that pulls in multiprocessing (via numpy/pandas/tables) will
    # re-execute this whole program in each spawned child unless freeze_support() short-circuits
    # them — without it the app fork-bombs itself. Must be the first thing in __main__.
    import multiprocessing as _mp
    _mp.freeze_support()
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--check" in sys.argv:
        raise SystemExit(_check())
    raise SystemExit(main())
