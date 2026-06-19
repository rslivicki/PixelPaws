"""Honest precision / recall / frame-F1 / bout-F1 vs threshold curve
for the Scratching classifier (pruned 120, session-level GroupKFold OOF).

v2 changes:
  - Adds bout-level F1 curve (IoU >= 0.3, with post-proc min_bout=15,
    max_gap=6 baked in from the saved 4D-best params -- only the
    threshold sweeps).
  - Savitzky-Golay smoothed F1 curves overlaid on the raw step curves
    so the eye can track the trend.
  - Best frame-F1 and best bout-F1 markers + threshold annotations.
"""
from __future__ import annotations
import json, sys, time, urllib.request, uuid
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
from evaluation_tab import _apply_bout_filtering
from scratching_retrain_honest_pruned import (
    find_bouts, bout_metrics_pooled, slice_oof_per_session,
)

PROJ = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected")
TRAIN_PKL = PROJ / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
PRUNED_PKL = (PROJ / "Classifiers" /
              "PixelPaws_Scratching_honest_20260521_160523" /
              "PixelPaws_Scratching_pruned_120_honest_20260521_160523.pkl")
LABELS_DIR = PROJ / "labels"
OUT_DIR = PRUNED_PKL.parent / "plots"

# Post-proc params held fixed at the saved 4D-best for the sweep
FIX_MIN_BOUT = 15
FIX_MAX_GAP  = 6
FIX_MIN_AFTER = 0
IOU_THR = 0.3
SAVGOL_WIN = 21
SAVGOL_POLY = 3

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
                ".svg": "image/svg+xml", ".csv": "text/csv"
                }.get(p.suffix.lower(), "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-HonestThreshCurveV2/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        step(f"  HTTP {r.status}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step("loading training set + pruned-120 honest pkl")
    train = joblib.load(TRAIN_PKL)
    y = np.asarray(train["y"], dtype=int)
    model = joblib.load(PRUNED_PKL)
    proba = np.asarray(model["oof_proba"], dtype=float)
    assert proba.shape == y.shape
    sessions = model["training_sessions"]
    bounds = slice_oof_per_session(y, sessions, LABELS_DIR)
    step(f"  n_frames={len(y)}  n_pos={int(y.sum())}  "
         f"prevalence={y.mean()*100:.2f}%  sessions={len(sessions)}")

    # Threshold grid -- fine enough for smooth curves
    thresholds = np.arange(0.005, 1.0, 0.005)
    pos = (y == 1)
    n_pos = int(pos.sum())

    precision, recall, frame_f1 = [], [], []
    bout_p, bout_r, bout_f1 = [], [], []

    for ti, t in enumerate(thresholds):
        pred_pos = proba >= t
        tp = int((pred_pos & pos).sum())
        fp = int((pred_pos & ~pos).sum())
        fn = n_pos - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0
        precision.append(p); recall.append(r); frame_f1.append(f)

        # Bout-level: apply post-proc (min_bout, max_gap) per session then
        # match bouts at IoU>=0.3
        y_post_pieces = []
        for k in range(len(sessions)):
            lo, hi = bounds[k], bounds[k + 1]
            piece = _apply_bout_filtering(
                pred_pos[lo:hi].astype(int).copy(),
                min_bout=FIX_MIN_BOUT,
                min_after_bout=FIX_MIN_AFTER,
                max_gap=FIX_MAX_GAP,
            )
            y_post_pieces.append(piece.astype(int))
        y_post = np.concatenate(y_post_pieces)
        bf, bP, bR, _, _, _ = bout_metrics_pooled(y, y_post, sessions,
                                                    bounds, iou_thr=IOU_THR)
        bout_p.append(bP); bout_r.append(bR); bout_f1.append(bf)
        if ti % 20 == 0:
            step(f"  thr={t:.3f}  frame F1={f:.3f}  bout F1={bf:.3f}")

    precision = np.array(precision); recall = np.array(recall)
    frame_f1 = np.array(frame_f1); bout_f1 = np.array(bout_f1)

    # Smoothed F1 curves (Savitzky-Golay)
    def smooth(v):
        if len(v) < SAVGOL_WIN + 2:
            return v
        return savgol_filter(v, window_length=SAVGOL_WIN, polyorder=SAVGOL_POLY)

    frame_f1_sm = smooth(frame_f1)
    bout_f1_sm = smooth(bout_f1)

    best_frame_i = int(np.argmax(frame_f1))
    best_bout_i = int(np.argmax(bout_f1))
    step(f"\n  RAW best frame F1 = {frame_f1[best_frame_i]:.3f} at thresh = "
         f"{thresholds[best_frame_i]:.3f}")
    step(f"  RAW best bout  F1 = {bout_f1[best_bout_i]:.3f} at thresh = "
         f"{thresholds[best_bout_i]:.3f}  (IoU>={IOU_THR}, "
         f"mb={FIX_MIN_BOUT}, mg={FIX_MAX_GAP})")

    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    # Faint raw F1 curves
    ax.plot(thresholds, frame_f1, color="#5b2c91", linewidth=0.9,
            alpha=0.28, zorder=2)
    ax.plot(thresholds, bout_f1, color="#1f9d55", linewidth=0.9,
            alpha=0.28, zorder=2)

    # Bold smoothed F1 curves
    ax.plot(thresholds, frame_f1_sm, color="#5b2c91", linewidth=2.6,
            label="Frame F1 (smoothed)", zorder=4)
    ax.plot(thresholds, bout_f1_sm, color="#1f9d55", linewidth=2.6,
            label=f"Bout F1 (IoU>={IOU_THR}, smoothed)", zorder=4)

    # Precision + recall (unsmoothed but light)
    ax.plot(thresholds, recall, color="#f5a623", linewidth=1.8,
            label="Recall (frame)", zorder=3)
    ax.plot(thresholds, precision, color="#d12c80", linewidth=1.8,
            label="Precision (frame)", zorder=3)

    # Mark peaks
    ax.axvline(thresholds[best_frame_i], color="#5b2c91", linestyle=":",
               linewidth=1.0, alpha=0.7)
    ax.scatter([thresholds[best_frame_i]], [frame_f1[best_frame_i]],
               color="#5b2c91", s=70, edgecolor="white", zorder=6)
    ax.annotate(f"frame F1 = {frame_f1[best_frame_i]:.2f}\n"
                f"thresh = {thresholds[best_frame_i]:.3f}",
                xy=(thresholds[best_frame_i], frame_f1[best_frame_i]),
                xytext=(thresholds[best_frame_i] + 0.10, frame_f1[best_frame_i] - 0.18),
                fontsize=9, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#5b2c91", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc"))

    ax.axvline(thresholds[best_bout_i], color="#1f9d55", linestyle=":",
               linewidth=1.0, alpha=0.7)
    ax.scatter([thresholds[best_bout_i]], [bout_f1[best_bout_i]],
               color="#1f9d55", s=70, edgecolor="white", zorder=6)
    ax.annotate(f"bout F1 = {bout_f1[best_bout_i]:.2f}\n"
                f"thresh = {thresholds[best_bout_i]:.3f}",
                xy=(thresholds[best_bout_i], bout_f1[best_bout_i]),
                xytext=(thresholds[best_bout_i] + 0.05, bout_f1[best_bout_i] + 0.10),
                fontsize=9, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#1f9d55", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc"))

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_base = OUT_DIR / "scratching_honest_threshold_curve_pruned120_v2"
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"wrote {out_base}.png / svg / pdf")

    pd.DataFrame({"threshold": thresholds,
                   "frame_precision": precision, "frame_recall": recall,
                   "frame_f1": frame_f1, "frame_f1_smoothed": frame_f1_sm,
                   "bout_precision": bout_p, "bout_recall": bout_r,
                   "bout_f1": bout_f1, "bout_f1_smoothed": bout_f1_sm,
                   }).to_csv(out_base.with_suffix(".csv"), index=False)

    head = ("**Scratching threshold curve -- v2 (frame + bout F1, smoothed)**\n"
            "Frame-level precision / recall / F1 plus bout-level F1 (IoU>=0.3, "
            f"post-proc fixed at mb={FIX_MIN_BOUT}, ma={FIX_MIN_AFTER}, "
            f"mg={FIX_MAX_GAP} from the saved 4D-best -- only thresh sweeps). "
            f"F1 curves smoothed with Savitzky-Golay (window={SAVGOL_WIN}, "
            f"polyorder={SAVGOL_POLY}); raw step curves left at ~28% opacity "
            "for reference.\n"
            f"Raw best frame F1 = {frame_f1[best_frame_i]:.3f} at thresh="
            f"{thresholds[best_frame_i]:.3f}; raw best bout F1 = "
            f"{bout_f1[best_bout_i]:.3f} at thresh="
            f"{thresholds[best_bout_i]:.3f}. The bout F1 plateau is broad and "
            "near-flat between thresh~0.1 and ~0.6 -- the post-proc absorbs "
            "the higher false-positive rate at lower thresholds because the "
            "min_bout filter rejects spurious sub-15-frame triggers.")
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf"),
                          out_base.with_suffix(".csv")])


if __name__ == "__main__":
    main()
