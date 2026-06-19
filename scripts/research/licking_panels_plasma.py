"""Regenerate Left_licking threshold + SHAP panels using the *exact*
scratching plasma palette/style (regen_scratching_plots.plot_threshold_curves
and plot_shap_panel). Then composite them side-by-side to match the
scratching layout the user shared.
"""
from __future__ import annotations
import json, sys, time, urllib.request, uuid, warnings
from pathlib import Path

import joblib
import numpy as np
import fitz

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "research"))

warnings.filterwarnings("ignore")

# Reuse the scratching panel functions verbatim — same plasma palette,
# same beeswarm density jitter, same bar colors, same layout.
from regen_scratching_plots import (
    plot_threshold_curves,
    plot_shap_panel,
)

from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

LICK_TRAIN = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/PixelPaws_Classifiers/HomeRig/Left_licking_train_set.pkl")
OUT_DIR    = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/PixelPaws_Classifiers/HomeRig/licking_plasma_panels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LICK_N_S1, LICK_N_S2 = 49855, 21153
SEED = 42

WEBHOOK = ("")


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml"
                }.get(p.suffix.lower(), "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-LickingPlasma/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def fit_rect(bb_w, bb_h, dest):
    dw, dh = dest.width, dest.height
    scale = min(dw / bb_w, dh / bb_h)
    new_w = bb_w * scale
    new_h = bb_h * scale
    cx = (dest.x0 + dest.x1) / 2
    cy = (dest.y0 + dest.y1) / 2
    return fitz.Rect(cx - new_w / 2, cy - new_h / 2,
                     cx + new_w / 2, cy + new_h / 2)


def main():
    step("loading BAREfoot-downsampled Left_licking train set")
    train = joblib.load(LICK_TRAIN)
    X, y = train["X"], np.asarray(train["y"], dtype=int)
    step(f"  rows={len(X):,}  features={X.shape[1]}  pos={int(y.sum()):,}")

    # Session boundaries for honest CV OOF (matches licking_honest_correct.py)
    groups = np.zeros(len(X), dtype=int)
    groups[:LICK_N_S1] = 0
    groups[LICK_N_S1:LICK_N_S1 + LICK_N_S2] = 1
    groups[LICK_N_S1 + LICK_N_S2:] = 2

    step("computing 3-fold session-level GroupKFold OOF")
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

    step("training final model on all data for SHAP")
    spw_all = (y == 0).sum() / max((y == 1).sum(), 1)
    final = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw_all,
        tree_method="hist", random_state=SEED, n_jobs=-1,
        verbosity=0, use_label_encoder=False, eval_metric="logloss",
    )
    final.fit(X, y)

    # ----- Panel C (threshold curves, plasma palette) -----
    thr_svg = OUT_DIR / "PixelPaws_Licking_panelC_threshold.svg"
    thr_png = OUT_DIR / "PixelPaws_Licking_panelC_threshold.png"
    thr_pdf = OUT_DIR / "PixelPaws_Licking_panelC_threshold.pdf"
    step(f"rendering {thr_svg.name}")
    plot_threshold_curves(y, oof, "Left_licking", str(thr_svg))
    plot_threshold_curves(y, oof, "Left_licking", str(thr_png))
    plot_threshold_curves(y, oof, "Left_licking", str(thr_pdf))

    # ----- Panel D (SHAP beeswarm + bar, plasma palette) -----
    shap_svg = OUT_DIR / "PixelPaws_Licking_panelD_shap.svg"
    shap_png = OUT_DIR / "PixelPaws_Licking_panelD_shap.png"
    shap_pdf = OUT_DIR / "PixelPaws_Licking_panelD_shap.pdf"
    step(f"rendering {shap_svg.name}")
    plot_shap_panel(final, X, "Left_licking", str(shap_svg),
                    n_top=12, sample_n=12000)
    plot_shap_panel(final, X, "Left_licking", str(shap_png),
                    n_top=12, sample_n=12000)
    plot_shap_panel(final, X, "Left_licking", str(shap_pdf),
                    n_top=12, sample_n=12000)

    # ----- Composite: threshold (left) + SHAP (right), matching the
    # scratching layout the user shared -----
    THR_W, THR_H = 360, 230
    SHAP_W, SHAP_H = 640, 230
    GAP_X = 12
    PAD = 6
    out_w = PAD + THR_W + GAP_X + SHAP_W + PAD
    out_h = PAD + max(THR_H, SHAP_H) + PAD

    out = fitz.open()
    page = out.new_page(width=out_w, height=out_h)

    src_thr  = fitz.open(thr_pdf)
    src_shap = fitz.open(shap_pdf)
    thr_page  = src_thr[0]
    shap_page = src_shap[0]

    thr_dest = fitz.Rect(PAD, PAD, PAD + THR_W, PAD + THR_H)
    thr_fit  = fit_rect(thr_page.rect.width, thr_page.rect.height, thr_dest)
    page.show_pdf_page(thr_fit, src_thr, 0)

    shap_x0 = PAD + THR_W + GAP_X
    shap_dest = fitz.Rect(shap_x0, PAD, shap_x0 + SHAP_W, PAD + SHAP_H)
    shap_fit  = fit_rect(shap_page.rect.width, shap_page.rect.height, shap_dest)
    page.show_pdf_page(shap_fit, src_shap, 0)

    out_pdf_final = OUT_DIR / "PixelPaws_Licking_threshold_shap_plasma.pdf"
    out.save(out_pdf_final, deflate=True, garbage=4)

    zoom = 240 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix_w = page.get_pixmap(matrix=mat, alpha=False)
    pix_t = page.get_pixmap(matrix=mat, alpha=True)
    out_png_w = OUT_DIR / "PixelPaws_Licking_threshold_shap_plasma_white.png"
    out_png_t = OUT_DIR / "PixelPaws_Licking_threshold_shap_plasma.png"
    pix_w.save(out_png_w)
    pix_t.save(out_png_t)

    out_svg = OUT_DIR / "PixelPaws_Licking_threshold_shap_plasma.svg"
    out_svg.write_text(page.get_svg_image(matrix=fitz.Matrix(1, 1)),
                       encoding="utf-8")

    src_thr.close(); src_shap.close(); out.close()
    step(f"wrote {out_pdf_final.name}, {out_png_w.name}, {out_png_t.name}, {out_svg.name}")

    head = (
        "**Left_licking — threshold + SHAP panels (plasma palette, scratching style)**\n"
        "Both panels generated with the exact same `regen_scratching_plots` "
        "code (plasma colormap, density-jittered beeswarm, purple bars), "
        "so this is colour-consistent with the Scratching figure.\n"
        "Left: Precision / Recall / F1 vs threshold (3-fold session-level "
        "GroupKFold OOF, deployed pkl methodology).\n"
        "Right: Top-12 SHAP beeswarm (Low→High feature value) + mean(|SHAP|) bar chart."
    )
    # Skip the giant SVGs (>20 MB each because of embedded SHAP point cloud)
    # — Discord rejects payloads >8 MB. Upload composite PNG + composite PDF
    # + the small panel PDFs.
    discord_upload(head, [out_png_w, out_png_t, out_pdf_final,
                          thr_pdf, thr_png])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
