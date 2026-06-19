"""Regenerate Scratching + Left_licking threshold-curve PDFs WITHOUT
the y-grid lines (show_grid=False), then build the 2x2 combined
composite (tight gap between threshold and SHAP).

Reuses:
  - Scratching cache pkl (OOF, X, y)  — fast, no re-CV
  - Left_licking BAREfoot train pkl   — runs 3-fold session-level
    GroupKFold to produce OOF, then re-renders threshold curve only
"""
from __future__ import annotations
import json, pickle, sys, time, urllib.request, uuid, warnings
from pathlib import Path

import joblib
import numpy as np
import fitz

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "research"))

warnings.filterwarnings("ignore")

import regen_scratching_plots as rsp
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

SCRATCH_PROJECT = Path(
    r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected"
)
SCRATCH_CACHE = (SCRATCH_PROJECT / "Classifiers" / "plots" /
                 "regen_20260429_140020" / "_regen_cache.pkl")
SCRATCH_TRAIN = (SCRATCH_PROJECT / "Classifiers" / "training_data" /
                 "Scratching_train_set.pkl")
SCRATCH_SHAP_PDF = (SCRATCH_PROJECT / "Classifiers" / "plots" /
                    "pdf_exports" / "PixelPaws_Scratching_panelD_shap.pdf")

LICK_TRAIN = Path(
    r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/"
    r"PixelPaws_Classifiers/HomeRig/Left_licking_train_set.pkl"
)
LICK_PANEL_DIR = Path(
    r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/"
    r"PixelPaws_Classifiers/HomeRig/licking_plasma_panels"
)
LICK_SHAP_PDF = LICK_PANEL_DIR / "PixelPaws_Licking_panelD_shap.pdf"

# Session boundaries for licking honest CV (matches licking_honest_correct.py)
LICK_N_S1, LICK_N_S2 = 49855, 21153
SEED = 42

OUT_DIR = LICK_PANEL_DIR

# Output PDFs (no-grid versions)
SCRATCH_THR_NG = OUT_DIR / "PixelPaws_Scratching_panelC_threshold_nogrid.pdf"
LICK_THR_NG    = OUT_DIR / "PixelPaws_Licking_panelC_threshold_nogrid.pdf"

# --- Composite layout knobs ---
ROW_H  = 200
THR_W  = round(ROW_H * 1.241, 2)
SHAP_W = round(ROW_H * 1.982, 2)
GAP_X  = 4
GAP_Y  = 10
PAD_L  = 4
PAD_R  = 4
PAD_T  = 4
PAD_B  = 4
PAGE_W = PAD_L + THR_W + GAP_X + SHAP_W + PAD_R
PAGE_H = PAD_T + 2 * ROW_H + GAP_Y + PAD_B

WEBHOOK = (
    ""
)


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(files):
        mime = {
            ".png": "image/png", ".pdf": "application/pdf",
            ".svg": "image/svg+xml",
        }.get(p.suffix.lower(), "application/octet-stream")
        body += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="file{i}"; '
            f'filename="{p.name}"'.encode(),
            f"Content-Type: {mime}".encode(), b"",
            p.read_bytes(),
        ]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "PixelPaws-ThresholdNoGrid/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def fit_rect(bb_w, bb_h, dest):
    dw, dh = dest.width, dest.height
    scale = min(dw / bb_w, dh / bb_h)
    new_w = bb_w * scale
    new_h = bb_h * scale
    cx = (dest.x0 + dest.x1) / 2
    cy = (dest.y0 + dest.y1) / 2
    return fitz.Rect(
        cx - new_w / 2, cy - new_h / 2,
        cx + new_w / 2, cy + new_h / 2,
    )


def place(page, src_pdf, dest):
    src = fitz.open(src_pdf)
    sp = src[0]
    fit = fit_rect(sp.rect.width, sp.rect.height, dest)
    page.show_pdf_page(fit, src, 0)
    src.close()
    return fit


def regen_scratching_threshold_nogrid():
    step("loading scratching regen cache")
    with open(SCRATCH_CACHE, "rb") as f:
        cache = pickle.load(f)
    with open(SCRATCH_TRAIN, "rb") as f:
        ts = pickle.load(f)
    y = np.asarray(ts.get("y", ts.get("labels")))
    step(f"  rows={len(y):,}  pos={int(y.sum()):,}")
    rsp.plot_threshold_curves(y, cache["oof"], "Scratching",
                              str(SCRATCH_THR_NG), show_grid=False)
    step(f"  wrote {SCRATCH_THR_NG.name}")


def regen_licking_threshold_nogrid():
    step("loading BAREfoot Left_licking train set")
    train = joblib.load(LICK_TRAIN)
    X, y = train["X"], np.asarray(train["y"], dtype=int)
    step(f"  rows={len(X):,}  features={X.shape[1]}  pos={int(y.sum()):,}")

    groups = np.zeros(len(X), dtype=int)
    groups[:LICK_N_S1] = 0
    groups[LICK_N_S1:LICK_N_S1 + LICK_N_S2] = 1
    groups[LICK_N_S1 + LICK_N_S2:] = 2

    step("running 3-fold session-level GroupKFold for OOF")
    gkf = GroupKFold(n_splits=3)
    oof = np.zeros(len(y), dtype=np.float32)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        t0 = time.time()
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw,
            tree_method="hist", random_state=SEED, n_jobs=-1,
            verbosity=0, use_label_encoder=False, eval_metric="logloss",
        )
        clf.fit(X.iloc[tr], y[tr])
        oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
        step(f"  fold {fold}/3 done in {time.time()-t0:.1f}s")

    rsp.plot_threshold_curves(y, oof, "Left_licking",
                              str(LICK_THR_NG), show_grid=False)
    step(f"  wrote {LICK_THR_NG.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCRATCH_THR_NG.exists():
        regen_scratching_threshold_nogrid()
    else:
        step(f"scratching no-grid threshold pdf exists, skipping regen")

    if not LICK_THR_NG.exists():
        regen_licking_threshold_nogrid()
    else:
        step(f"licking no-grid threshold pdf exists, skipping regen")

    for p in (SCRATCH_THR_NG, SCRATCH_SHAP_PDF, LICK_THR_NG, LICK_SHAP_PDF):
        if not p.exists():
            step(f"! missing {p}")
            return 1

    step(f"page geometry: {PAGE_W:.1f} x {PAGE_H:.1f}")
    out = fitz.open()
    page = out.new_page(width=PAGE_W, height=PAGE_H)

    # Row 1 — Scratching
    row1_y = PAD_T
    thr_dest = fitz.Rect(PAD_L, row1_y, PAD_L + THR_W, row1_y + ROW_H)
    shap_x0 = PAD_L + THR_W + GAP_X
    shap_dest = fitz.Rect(shap_x0, row1_y, shap_x0 + SHAP_W, row1_y + ROW_H)
    place(page, SCRATCH_THR_NG, thr_dest)
    place(page, SCRATCH_SHAP_PDF, shap_dest)

    # Row 2 — Left_licking
    row2_y = PAD_T + ROW_H + GAP_Y
    thr_dest2 = fitz.Rect(PAD_L, row2_y, PAD_L + THR_W, row2_y + ROW_H)
    shap_dest2 = fitz.Rect(shap_x0, row2_y, shap_x0 + SHAP_W, row2_y + ROW_H)
    place(page, LICK_THR_NG, thr_dest2)
    place(page, LICK_SHAP_PDF, shap_dest2)

    out_pdf = OUT_DIR / "Scratching_Licking_combined_panels_nogrid.pdf"
    out.save(out_pdf, deflate=True, garbage=4)

    zoom = 240 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix_w = page.get_pixmap(matrix=mat, alpha=False)
    pix_t = page.get_pixmap(matrix=mat, alpha=True)
    out_png_w = OUT_DIR / "Scratching_Licking_combined_panels_nogrid_white.png"
    out_png_t = OUT_DIR / "Scratching_Licking_combined_panels_nogrid.png"
    pix_w.save(out_png_w)
    pix_t.save(out_png_t)
    out.close()
    step(f"wrote {out_pdf.name}, {out_png_w.name}, {out_png_t.name}")

    head = (
        "**Scratching + Left_licking combined panels — gridlines removed.**\n"
        f"Page {PAGE_W:.0f}×{PAGE_H:.0f}.  Row 1 = Scratching, "
        "Row 2 = Left_licking.  Per row: threshold curve (left, "
        "*no y-grid*) + SHAP beeswarm+bar (right).  Horizontal gap "
        f"between threshold and SHAP = {GAP_X}pt.  Plasma palette "
        "preserved (Recall=orange, Precision=pink, F1=purple, "
        "beeswarm=plasma Low→High, importance bars=purple)."
    )
    discord_upload(head, [out_png_w, out_png_t, out_pdf])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
