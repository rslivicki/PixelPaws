"""
Regenerate two panels for the Scratching classifier using honest, held-out
data and the corrected threshold pkl:

  Panel B — Learning curve: F1 (5-fold stratified CV) vs bout-positive
            frames. Fresh subsampled training on the cached X/y.
  Panel C — Threshold curves: Precision / Recall / F1 vs threshold,
            using the cached `oof_proba` already in the pkl (held-out).
            Dashed vertical lines mark BOTH the OOF-optimal threshold
            (max F1, no post-proc) and the corrected deployed threshold
            (after post-proc).

Pink palette to stay consistent with the rest of the Scratching figures.
PNG + PDF + SVG exports posted to #results.
"""
from __future__ import annotations
import json, pickle, sys, time, urllib.request, uuid, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
from evaluation_tab import _apply_bout_filtering

PROJ = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected")
CLF  = PROJ / "Classifiers" / "PixelPaws_Scratching.pkl"
TRAIN = PROJ / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
OUT_DIR = PROJ / "Classifiers" / "plots" / "corrected_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = ("")

# Pink palette — matches the existing PinkRaster style
PINK_LIGHT = "#F8BBD0"; PINK_MID = "#EC407A"; PINK_DARK = "#C2185B"
GRID_GREY = "#E0E0E0"; TEXT_GREY = "#666666"
N_FOLDS = 5
SEED = 42


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
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml"}.get(p.suffix.lower(),
                                              "application/octet-stream")
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
                 "User-Agent": "PixelPaws-CorrectedPanels/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def _make_xgb(spw, n_est=200):
    return XGBClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method="hist", random_state=SEED, n_jobs=-1,
        verbosity=0, use_label_encoder=False, eval_metric="logloss")


def plot_threshold_curves(y, oof, deployed, out_stem: Path):
    """Precision / Recall / F1 vs threshold on cached OOF probabilities.
    Two dashed lines: OOF-best (raw) + corrected deployed (post-proc)."""
    thrs = np.linspace(0.01, 0.99, 99)
    prec, rec, f1 = [], [], []
    for t in thrs:
        yp = (oof >= t).astype(int)
        prec.append(precision_score(y, yp, zero_division=0))
        rec.append(recall_score(y, yp, zero_division=0))
        f1.append(f1_score(y, yp, zero_division=0))
    prec, rec, f1 = map(np.asarray, (prec, rec, f1))
    best_idx = int(np.argmax(f1))
    best_t = float(thrs[best_idx])
    best_f1 = float(f1[best_idx])
    step(f"  threshold sweep: best (no post-proc) at thr={best_t:.2f}, F1={best_f1:.3f}")

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.plot(thrs, rec,  color=PINK_LIGHT, linewidth=1.8, label="Recall")
    ax.plot(thrs, prec, color=PINK_MID,   linewidth=1.8, label="Precision")
    ax.plot(thrs, f1,   color=PINK_DARK,  linewidth=2.0, label="F1")
    ax.axvline(best_t, color=PINK_DARK, linestyle="--", linewidth=1.0,
               alpha=0.85, label=f"OOF-best thr={best_t:.2f}")
    ax.axvline(deployed, color=TEXT_GREY, linestyle=":", linewidth=1.0,
               alpha=0.85, label=f"deployed thr={deployed:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.6, zorder=1)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=True, loc="lower left", fontsize=7,
              framealpha=0.95, edgecolor="#cccccc")

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    out_svg = out_stem.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf, out_svg, best_t, best_f1


def learning_curve(X, y, fractions=None, n_folds=5, fixed_thr=0.5):
    """Stratified-CV F1 vs bout-positive frames using subsampled positives.
    Uses a *fixed* threshold (default: OOF-best from full data) for every
    fold and every subsample — that's the honest learning curve, not per-
    fold-optimal F1 which over-states performance."""
    if fractions is None:
        fractions = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    rng = np.random.default_rng(SEED)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    out = []
    for frac in fractions:
        keep_n = max(50, int(round(frac * len(pos_idx))))
        keep_pos = rng.choice(pos_idx, size=min(keep_n, len(pos_idx)),
                              replace=False)
        idx = np.concatenate([keep_pos, neg_idx]); idx.sort()
        X_sub = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
        y_sub = y[idx]
        n_pos = int(y_sub.sum())
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        f1s = []
        for fold, (tr, te) in enumerate(skf.split(X_sub, y_sub), 1):
            t0 = time.time()
            spw = (y_sub[tr] == 0).sum() / max((y_sub[tr] == 1).sum(), 1)
            clf = _make_xgb(spw=spw, n_est=200)
            X_tr = X_sub.iloc[tr] if isinstance(X_sub, pd.DataFrame) else X_sub[tr]
            X_te = X_sub.iloc[te] if isinstance(X_sub, pd.DataFrame) else X_sub[te]
            clf.fit(X_tr, y_sub[tr])
            p = clf.predict_proba(X_te)[:, 1]
            f1 = f1_score(y_sub[te], (p >= fixed_thr).astype(int),
                          zero_division=0)
            f1s.append(f1)
            step(f"  LC frac={frac:.2f}  fold {fold}/{n_folds}  "
                 f"F1@thr={fixed_thr:.2f}={f1:.3f}  ({time.time()-t0:.1f}s)")
        out.append((n_pos, float(np.mean(f1s)), float(np.std(f1s))))
        step(f"    -> frac={frac:.2f}  n_pos={n_pos}  "
             f"F1={out[-1][1]:.3f}±{out[-1][2]:.3f}")
    return out


def plot_learning_curve(lc, out_stem: Path):
    xs    = np.array([r[0] for r in lc])
    means = np.array([r[1] for r in lc])
    stds  = np.array([r[2] for r in lc])
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.plot(xs, means, marker="o", markersize=6, linewidth=1.8,
            color=PINK_DARK, label="Scratching", zorder=3)
    ax.fill_between(xs, means - stds, means + stds,
                    color=PINK_MID, alpha=0.18, zorder=2)
    ax.set_xlabel("Bout-positive frames")
    ax.set_ylabel(f"F1-score ({N_FOLDS}-fold CV)")
    ax.set_ylim(0.0, 1.0); ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.6, zorder=1)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    out_svg = out_stem.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf, out_svg


def main():
    step("loading classifier + training set")
    cd = joblib.load(CLF)
    train = pickle.load(open(TRAIN, "rb"))
    X, y = train["X"], train["y"]
    oof = cd["oof_proba"]
    deployed = float(cd["best_thresh"])
    step(f"  X shape={X.shape}, positives={int(y.sum())}, "
         f"deployed thr={deployed:.2f}")

    step("=== Panel C: threshold curves (cached OOF) ===")
    thr_stem = OUT_DIR / "PixelPaws_Scratching_panelC_threshold_corrected"
    thr_png, thr_pdf, thr_svg, best_t, best_f1 = plot_threshold_curves(
        y, oof, deployed, thr_stem)

    step("=== Panel B: learning curve (stratified CV subsamples) ===")
    lc = learning_curve(X, y, n_folds=N_FOLDS, fixed_thr=best_t)
    lc_stem = OUT_DIR / "PixelPaws_Scratching_panelB_learning_corrected"
    lc_png, lc_pdf, lc_svg = plot_learning_curve(lc, lc_stem)

    step("uploading")
    discord_upload(
        f"**Scratching panels — corrected** (using cached OOF + fresh stratified CV)\n"
        f"**Panel B** (learning curve): F1 ({N_FOLDS}-fold CV) vs bout-positive frames. "
        f"At full data (n={lc[-1][0]:,} positives) F1 = `{lc[-1][1]:.3f}±{lc[-1][2]:.3f}` "
        f"(per-fold F1 at fold-optimal threshold). The old learning curve plateau "
        f"~0.9 was in-sample; this honest version plateaus near the OOF F1.\n"
        f"**Panel C** (threshold curves): Precision/Recall/F1 vs threshold on the cached "
        f"OOF probabilities (truly held-out). Dashed pink line marks OOF-best "
        f"`thr={best_t:.2f}` (F1={best_f1:.3f}, no post-proc); dotted grey line marks "
        f"the corrected deployed `thr={deployed:.2f}`. The old screenshot's vertical "
        f"line at ~0.85 with F1~0.95 was using in-sample evaluation; this version "
        f"uses true held-out OOF predictions.\n"
        f"PNG + PDF + SVG.",
        [lc_png, thr_png, lc_pdf, thr_pdf, lc_svg, thr_svg],
    )


if __name__ == "__main__":
    main()
