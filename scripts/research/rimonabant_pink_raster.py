"""
Render a Scratching evaluation raster in the user's pink/red styling:
  Panel 1 — per-frame raster, both Human and Model in pink/magenta
  Panel 2 — 2x2 confusion matrix (RdPu)
  Panel 3 — 10-s-binned ETHOGRAM-style heatmap (continuous color
            intensity rather than the existing bar plot)

Picks the labeled session whose F1 is closest to 0.81 to match the
screenshot. Outputs PNG + PDF + SVG and posts to #results.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

PROJECT = Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected")
LABELS_DIR = PROJECT / "labels"
FEATURES_DIR = PROJECT / "features"
VIDEOS_DIR = PROJECT / "videos"
CLF_PKL = PROJECT / "Classifiers" / "PixelPaws_Scratching.pkl"
SHUFFLE = 9
TARGET_F1 = 0.81

OUT_DIR = PROJECT / "Classifiers" / "plots" / "pink_rasters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord_upload(content: str, files: list[Path]) -> None:
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml"}.get(
            p.suffix.lower(), "application/octet-stream"
        )
        body += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
            f"Content-Type: {mime}".encode(), b"",
            p.read_bytes(),
        ]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-PinkRaster/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def load_labels_csv(label_path: Path, n_frames: int):
    """Convert a PixelPaws Labels.csv to a per-frame binary array for
    Scratching."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(label_path)
    y = np.zeros(n_frames, dtype=np.int8)
    # Common formats: rows with (start_frame, end_frame, behavior) or
    # per-frame binary columns. Detect dynamically.
    cols_lower = [c.lower() for c in df.columns]
    if "behavior" in cols_lower and "start_frame" in cols_lower and "end_frame" in cols_lower:
        c_b = df.columns[cols_lower.index("behavior")]
        c_s = df.columns[cols_lower.index("start_frame")]
        c_e = df.columns[cols_lower.index("end_frame")]
        for _, row in df.iterrows():
            beh = str(row[c_b]).strip().lower()
            if "scratch" not in beh:
                continue
            s = int(row[c_s]); e = int(row[c_e])
            s = max(0, min(s, n_frames - 1))
            e = max(0, min(e, n_frames))
            y[s:e + 1] = 1
        return y
    # Per-frame binary fallback
    sc_cols = [c for c in df.columns if "scratch" in c.lower()]
    if sc_cols:
        col = sc_cols[0]
        arr = df[col].astype(int).to_numpy()
        m = min(len(arr), n_frames)
        y[:m] = arr[:m].astype(np.int8)
        return y
    return y


def predict_and_eval(session_stem: str):
    """Returns (y_true, y_pred, fps) for one session, or None if data missing."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score
    from prediction_pipeline import predict_with_xgboost, augment_features_post_cache

    pkls = sorted(FEATURES_DIR.glob(f"{session_stem}_features_*.pkl"))
    if not pkls:
        return None
    pkl = pkls[-1]
    with open(pkl, "rb") as f:
        X = pickle.load(f)
    n_frames = len(X)

    h5s = list(VIDEOS_DIR.glob(f"{session_stem}*shuffle{SHUFFLE}*_filtered.h5")) \
        + list(VIDEOS_DIR.glob(f"{session_stem}*_filtered.h5"))
    h5 = str(h5s[0]) if h5s else ""

    try:
        cd = joblib.load(CLF_PKL)
    except Exception:
        with open(CLF_PKL, "rb") as f:
            cd = pickle.load(f)
    model = cd["clf_model"]
    thresh = float(cd.get("best_thresh", 0.5))
    try:
        X_aug = augment_features_post_cache(X.copy(), cd, model, h5)
    except Exception:
        X_aug = X
    # Backfill missing Ego_* columns from non-ego counterparts
    if hasattr(model, "feature_names_in_"):
        need = [f for f in model.feature_names_in_ if f not in X_aug.columns]
        for f in need:
            if f.startswith("Ego_"):
                nonego = f[len("Ego_"):]
                if nonego in X_aug.columns:
                    X_aug[f] = X_aug[nonego]
    try:
        y_proba = predict_with_xgboost(
            model, X_aug,
            calibrator=cd.get("prob_calibrator"),
            fold_models=cd.get("fold_models"),
        )
    except Exception as e:
        step(f"  ! predict failed for {session_stem}: {e}")
        return None
    y_pred = (y_proba >= thresh).astype(int)
    # Post-process — local implementation, no imports
    mb = int(cd.get("min_bout", 1))
    mg = int(cd.get("max_gap", 0))

    def find_bouts(y):
        bouts = []; in_b = False; start = 0
        for i, v in enumerate(y):
            if v and not in_b:
                in_b = True; start = i
            elif not v and in_b:
                in_b = False; bouts.append((start, i - 1))
        if in_b:
            bouts.append((start, len(y) - 1))
        return bouts

    def post(y, min_bout=1, max_gap=0):
        y = y.astype(int).copy()
        if max_gap > 0:
            bouts = find_bouts(y)
            for i in range(len(bouts) - 1):
                gap = bouts[i + 1][0] - bouts[i][1] - 1
                if 0 < gap <= max_gap:
                    y[bouts[i][1] + 1: bouts[i + 1][0]] = 1
        if min_bout > 1:
            for s, e in find_bouts(y):
                if e - s + 1 < min_bout:
                    y[s: e + 1] = 0
        return y
    y_pred = post(y_pred, mb, mg)

    # Ground truth from labels CSV (Scratching)
    lab_csv = LABELS_DIR / f"{session_stem}_Labels.csv"
    if not lab_csv.is_file():
        return None
    y_true = load_labels_csv(lab_csv, n_frames)

    # FPS
    fps = 60.0
    try:
        import cv2
        vids = list(VIDEOS_DIR.glob(f"{session_stem}*.mp4"))
        if vids:
            cap = cv2.VideoCapture(str(vids[0]))
            f = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if f > 0:
                fps = float(f)
    except Exception:
        pass

    return y_true, y_pred, fps


def render_pink_raster(y_true, y_pred, fps: float, session: str,
                       out_png: Path, out_pdf: Path, out_svg: Path):
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix
    from scipy.stats import pearsonr
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    # Metrics
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm_raw = confusion_matrix(y_true, y_pred, labels=[0, 1])
    rs = cm_raw.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    cm_norm = cm_raw / rs

    # 10-s bins
    bin_frames = max(1, int(10 * fps))
    n_frames = len(y_true)
    n_bins = max(1, int(np.ceil(n_frames / bin_frames)))
    human_s = np.zeros(n_bins); model_s = np.zeros(n_bins)
    for k in range(n_bins):
        sl = slice(k * bin_frames, (k + 1) * bin_frames)
        human_s[k] = y_true[sl].sum() / fps
        model_s[k] = y_pred[sl].sum() / fps
    r_val, _ = pearsonr(human_s, model_s) if n_bins > 1 else (float("nan"), None)

    fig = plt.figure(figsize=(16, 4), constrained_layout=True)
    gs  = fig.add_gridspec(1, 3, width_ratios=[5, 2, 4], wspace=0.35)
    ax_raster, ax_cm, ax_bins = gs.subplots()

    PINK_LIGHT = "#F8BBD0"   # Human band
    PINK_DARK  = "#C2185B"   # Model band
    PINK_BG    = "#FFFFFF"

    # ── Panel 1: per-frame raster, both rows in pink shades ──
    def bouts(arr):
        padded = np.concatenate([[0], arr.astype(int), [0]])
        d = np.diff(padded)
        s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
        return list(zip(s.tolist(), (e - s).tolist()))
    ax_raster.broken_barh(bouts(y_true), (2.6, 0.8), facecolors=PINK_LIGHT,
                          edgecolors=PINK_LIGHT)
    ax_raster.broken_barh(bouts(y_pred), (1.2, 0.8), facecolors=PINK_DARK,
                          edgecolors=PINK_DARK)
    ax_raster.set_yticks([1.6, 3.0])
    ax_raster.set_yticklabels(["Model", "Human"], color="#777")
    ax_raster.set_ylim(0.8, 3.8)
    ax_raster.set_xlim(0, n_frames)
    ax_raster.set_xlabel("Frames")
    ax_raster.set_title(session, fontsize=10, color="#444")
    for sp in ("top", "right", "left"):
        ax_raster.spines[sp].set_visible(False)
    ax_raster.tick_params(left=False, colors="#777")

    # ── Panel 2: confusion matrix (RdPu) ──
    im = ax_cm.imshow(cm_norm, cmap="RdPu", vmin=0, vmax=1)
    for row in range(2):
        for col in range(2):
            v = cm_norm[row, col]
            ax_cm.text(col, row, f"{v:.2f}",
                       ha="center", va="center", fontsize=10,
                       color="white" if v > 0.5 else "black")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["0", "1"]); ax_cm.set_yticklabels(["0", "1"])
    ax_cm.set_xlabel("Predicted label", color="#555")
    ax_cm.set_ylabel("True label", color="#555")
    ax_cm.set_title(f"F1-Score: {f1:.2f}", fontsize=10, color="#444")

    # ── Panel 3: 10-s-binned ETHOGRAM (heatmap), two rows in RdPu ──
    M = np.vstack([human_s, model_s])   # (2, n_bins)
    vmax = max(M.max(), 1e-9)
    ax_bins.imshow(M, aspect="auto", cmap="RdPu", vmin=0, vmax=vmax,
                   extent=[0, n_bins * 10, 1.5, -0.5])
    ax_bins.set_yticks([0, 1])
    ax_bins.set_yticklabels(["Human", "Model"], color="#777")
    ax_bins.set_xlabel("Time (sec)")
    ax_bins.set_title(f"R = {r_val:.2f}" if not np.isnan(r_val) else "R = n/a",
                      fontsize=10, color="#444")
    for sp in ("top", "right", "left"):
        ax_bins.spines[sp].set_visible(False)
    ax_bins.tick_params(left=False, colors="#777")

    fig.savefig(out_png, format="png", dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return f1, r_val


def main() -> int:
    import numpy as np
    # Discover labeled sessions
    sessions = []
    for csv in sorted(LABELS_DIR.glob("*_Labels.csv")):
        sessions.append(csv.stem.replace("_Labels", ""))
    step(f"labeled sessions: {len(sessions)} -> {sessions}")

    # Evaluate each, then pick the closest to TARGET_F1
    candidates = []
    for sess in sessions:
        step(f"evaluating {sess}...")
        res = predict_and_eval(sess)
        if res is None:
            step(f"  ! could not evaluate")
            continue
        y_true, y_pred, fps = res
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, zero_division=0)
        step(f"  F1 = {f1:.3f}  fps={fps}")
        candidates.append((sess, y_true, y_pred, fps, f1))

    if not candidates:
        step("! no sessions evaluated successfully"); return 1

    # Closest to target F1
    pick = min(candidates, key=lambda c: abs(c[4] - TARGET_F1))
    step(f"\npicked: {pick[0]}  (F1={pick[4]:.3f})")

    out_png = OUT_DIR / f"PixelPaws_Scratching_PinkRaster_{pick[0]}.png"
    out_pdf = out_png.with_suffix(".pdf")
    out_svg = out_png.with_suffix(".svg")
    f1, r_val = render_pink_raster(
        pick[1], pick[2], pick[3], pick[0],
        out_png, out_pdf, out_svg,
    )
    step(f"\n-> {out_png}")
    step(f"-> {out_pdf}")
    step(f"-> {out_svg}")

    head = (
        f"**Scratching evaluation raster — pink styling**\n"
        f"Session `{pick[0]}` (F1 = {f1:.2f}, R = {r_val:.2f}). "
        "Both Human and Model rows in pink (Human light / Model dark), "
        "RdPu confusion matrix, 10-s-binned ethogram on the right as "
        "a heatmap rather than bars. PNG + PDF (editable text, "
        "Illustrator-ready) + SVG."
    )
    discord_upload(head, [out_png, out_pdf, out_svg])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
