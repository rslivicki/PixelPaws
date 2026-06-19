"""Honest precision/recall/F1 vs threshold curve for the Scratching
classifier (pruned 120, session-level GroupKFold OOF).

Reads the pooled OOF probabilities and labels saved by
`scratching_retrain_honest_pruned.py` and draws clean curves with the
F1-optimal threshold marked.

Replaces the misleading old curve where F1 peaked near thresh=0.85 due
to StratifiedKFold(shuffle=True) leaking bouts across folds. The
honest curve has F1 peaking around thresh=0.125 because the OOF
probabilities are computed on truly-held-out sessions.
"""
from __future__ import annotations
import json, time, urllib.request, uuid
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

PROJ = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected")
TRAIN_PKL = PROJ / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
PRUNED_PKL = (PROJ / "Classifiers" /
              "PixelPaws_Scratching_honest_20260521_160523" /
              "PixelPaws_Scratching_pruned_120_honest_20260521_160523.pkl")
OUT_DIR = PRUNED_PKL.parent / "plots"

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
                 "User-Agent": "PixelPaws-HonestThreshCurve/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        step(f"  HTTP {r.status}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step("loading training set + pruned-120 honest pkl")
    train = joblib.load(TRAIN_PKL)
    y = np.asarray(train["y"], dtype=int)
    model = joblib.load(PRUNED_PKL)
    proba = np.asarray(model["oof_proba"], dtype=float)
    assert proba.shape == y.shape, f"shape mismatch: proba {proba.shape} vs y {y.shape}"
    step(f"  n_frames={len(y)}  n_positive={int(y.sum())}  "
         f"prevalence={y.mean()*100:.2f}%")

    # Frame-level threshold sweep across thresholds 0.005...0.995 (step 0.005)
    thresholds = np.arange(0.005, 1.0, 0.005)
    pos = (y == 1)
    n_pos = int(pos.sum())
    precision, recall, f1 = [], [], []
    for t in thresholds:
        pred_pos = proba >= t
        tp = int((pred_pos & pos).sum())
        fp = int((pred_pos & ~pos).sum())
        fn = n_pos - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0
        precision.append(p); recall.append(r); f1.append(f)
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)
    best_idx = int(np.argmax(f1))
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1[best_idx])
    step(f"  frame-level best F1 = {best_f1:.3f}  at thresh = {best_thresh:.3f}")

    # 4D-best (from pkl) for visual reference
    bp = model["oof_best_params"]
    step(f"  saved 4D-best OOF: thresh={bp['thresh']}  min_bout={bp['min_bout']}  "
         f"min_after={bp['min_after_bout']}  max_gap={bp['max_gap']}  F1={bp['f1']:.3f}")

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.plot(thresholds, recall, color="#f5a623", linewidth=2.2, label="Recall")
    ax.plot(thresholds, precision, color="#d12c80", linewidth=2.2, label="Precision")
    ax.plot(thresholds, f1, color="#5b2c91", linewidth=2.6, label="F1 Score")

    # Mark F1-optimal threshold
    ax.axvline(best_thresh, color="#888", linestyle=":", linewidth=1.4, alpha=0.85)
    ax.scatter([best_thresh], [best_f1], color="#5b2c91", s=70,
               edgecolor="white", zorder=5)
    ax.annotate(f"best F1 = {best_f1:.2f}\nthresh = {best_thresh:.3f}",
                xy=(best_thresh, best_f1),
                xytext=(best_thresh + 0.12, best_f1 - 0.15),
                fontsize=10, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#5b2c91", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#cccccc"))

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_base = OUT_DIR / "scratching_honest_threshold_curve_pruned120"
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"wrote {out_base}.png / svg / pdf")

    pd.DataFrame({"threshold": thresholds, "precision": precision,
                   "recall": recall, "f1": f1}).to_csv(
        out_base.with_suffix(".csv"), index=False)

    head = ("**Scratching classifier -- HONEST precision/recall/F1 vs threshold**\n"
            "Built from the pooled session-level GroupKFold OOF probabilities "
            "saved in the pruned-120 honest retrain pkl. Replaces the older "
            "in-sample / StratifiedKFold-leak curve where F1 peaked near "
            f"thresh=0.85. Honest frame-level best F1 = {best_f1:.3f} at "
            f"thresh = {best_thresh:.3f}. With the full 4D post-proc "
            f"(min_bout / min_after / max_gap) the saved best is F1=0.802 "
            f"at thresh=0.125, mb=15, ma=0, mg=6.\n"
            "Vertical dotted line + dot marks the frame-level F1-optimal "
            "threshold. PNG/SVG/PDF + sweep CSV attached.")
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf"),
                          out_base.with_suffix(".csv")])


if __name__ == "__main__":
    main()
