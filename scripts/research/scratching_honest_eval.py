"""
Honest per-frame + per-bout F1 evaluation for the Scratching classifier.

Uses the cached OOF probabilities from the deployed pkl (which were
generated with session-level CV inside _real_training — held-out by
session, not by frame, so no bout leakage).

For each of the 5 training sessions, computes:
  - Per-frame F1 at the deployed threshold + post-proc
  - Per-bout F1 with multiple temporal-IoU thresholds (0.1, 0.3, 0.5)

Aggregates and posts to #results.
"""
from __future__ import annotations
import json, pickle, sys, time, urllib.request, uuid
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
from sklearn.metrics import f1_score, precision_score, recall_score

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
from evaluation_tab import _apply_bout_filtering

PROJ = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected")
CLF  = PROJ / "Classifiers" / "PixelPaws_Scratching.pkl"
TRAIN = PROJ / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
LABELS_DIR = PROJ / "labels"
OUT_DIR = PROJ / "Classifiers" / "plots" / "honest_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = ("")

PINK_LIGHT = "#F8BBD0"; PINK_MID = "#EC407A"; PINK_DARK = "#C2185B"
GRID_GREY = "#E0E0E0"


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
                ".svg": "image/svg+xml", ".csv": "text/csv"}.get(
                    p.suffix.lower(), "application/octet-stream")
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
                 "User-Agent": "PixelPaws-HonestEval/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def find_bouts(y):
    """Return list of (start, end_inclusive) bout indices."""
    if len(y) == 0:
        return []
    padded = np.concatenate([[0], y.astype(int), [0]])
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def bout_f1(y_true, y_pred, iou_thr=0.5):
    """Bout-level F1: a predicted bout matches a true bout if their
    temporal IoU >= iou_thr. Returns (F1, P, R, TP, FP, FN, n_true, n_pred)."""
    true_bouts = find_bouts(y_true)
    pred_bouts = find_bouts(y_pred)
    n_true = len(true_bouts)
    n_pred = len(pred_bouts)

    def iou(a, b):
        s = max(a[0], b[0])
        e = min(a[1], b[1])
        inter = max(0, e - s + 1)
        union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
        return inter / union if union > 0 else 0.0

    # Greedy matching: for each true bout, find best pred match >= iou_thr
    matched_true = set()
    matched_pred = set()
    pairs = []
    for ti, tb in enumerate(true_bouts):
        best_pi, best_iou = -1, 0.0
        for pi, pb in enumerate(pred_bouts):
            if pi in matched_pred:
                continue
            v = iou(tb, pb)
            if v > best_iou:
                best_iou = v
                best_pi = pi
        if best_pi >= 0 and best_iou >= iou_thr:
            matched_true.add(ti)
            matched_pred.add(best_pi)
            pairs.append((ti, best_pi, best_iou))

    TP = len(matched_true)
    FN = n_true - TP
    FP = n_pred - len(matched_pred)
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return f1, P, R, TP, FP, FN, n_true, n_pred


def slice_oof_per_session(y_global, training_sessions, labels_dir):
    """Same logic as scratching_per_session_rasters: slice OOF by matching
    cumulative positives to each session's Labels.csv positives."""
    pos_per = {}
    for s in training_sessions:
        df = pd.read_csv(labels_dir / f"{s}_Labels.csv")
        col = next((c for c in df.columns if c.lower() == "scratching"), None)
        pos_per[s] = int(df[col].sum()) if col else 0

    pos_idx = np.where(y_global == 1)[0]
    cum_target = 0
    bounds = [0]
    for s in training_sessions[:-1]:
        cum_target += pos_per[s]
        last_pos = pos_idx[cum_target - 1]
        if cum_target < len(pos_idx):
            first_next = pos_idx[cum_target]
        else:
            first_next = len(y_global)
        b = (last_pos + first_next) // 2 + 1
        bounds.append(b)
    bounds.append(len(y_global))
    return bounds, pos_per


def main():
    step("loading classifier + training set")
    cd = joblib.load(CLF)
    train = pickle.load(open(TRAIN, "rb"))
    y = train["y"]; oof = cd["oof_proba"]
    sessions = list(cd["training_sessions"])
    thr = float(cd["best_thresh"]); mb = int(cd["min_bout"])
    ma = int(cd["min_after_bout"]); mg = int(cd["max_gap"])
    step(f"  deployed params: thr={thr:.2f}  mb={mb}  ma={ma}  mg={mg}")

    bounds, pos_per = slice_oof_per_session(y, sessions, LABELS_DIR)

    rows = []
    iou_thresholds = [0.1, 0.3, 0.5]

    for k, sess in enumerate(sessions):
        lo, hi = bounds[k], bounds[k + 1]
        y_s = y[lo:hi]; proba_s = oof[lo:hi]
        y_pred = (proba_s >= thr).astype(int)
        y_pred = _apply_bout_filtering(y_pred.copy(), mb, ma, mg)

        # Per-frame metrics
        fm_f1 = f1_score(y_s, y_pred, zero_division=0)
        fm_p  = precision_score(y_s, y_pred, zero_division=0)
        fm_r  = recall_score(y_s, y_pred, zero_division=0)
        TP = int(((y_s == 1) & (y_pred == 1)).sum())
        FP = int(((y_s == 0) & (y_pred == 1)).sum())
        FN = int(((y_s == 1) & (y_pred == 0)).sum())

        row = {
            "session": sess,
            "n_frames": int(len(y_s)),
            "n_pos": int(y_s.sum()),
            "pos_pct": float(y_s.sum() / len(y_s) * 100),
            "frame_F1": fm_f1, "frame_P": fm_p, "frame_R": fm_r,
            "frame_TP": TP, "frame_FP": FP, "frame_FN": FN,
        }

        # Per-bout metrics at multiple IoU thresholds
        for iou_t in iou_thresholds:
            bf1, bp, br, tp, fp, fn, n_t, n_p = bout_f1(y_s, y_pred, iou_t)
            row[f"bout_F1_iou{iou_t:.1f}"]  = bf1
            row[f"bout_P_iou{iou_t:.1f}"]   = bp
            row[f"bout_R_iou{iou_t:.1f}"]   = br
            row[f"bout_TP_iou{iou_t:.1f}"]  = tp
            row[f"bout_FP_iou{iou_t:.1f}"]  = fp
            row[f"bout_FN_iou{iou_t:.1f}"]  = fn
            if iou_t == 0.5:
                row["n_true_bouts"] = n_t
                row["n_pred_bouts"] = n_p

        rows.append(row)
        step(f"  {sess}  n={len(y_s):,}  pos={int(y_s.sum()):,}  "
             f"frame F1={fm_f1:.3f}  bout F1(iou=0.5)={row['bout_F1_iou0.5']:.3f}  "
             f"true_bouts={row['n_true_bouts']}  pred_bouts={row['n_pred_bouts']}")

    # Aggregated (pooled across all sessions)
    y_pred_all = (oof >= thr).astype(int)
    y_pred_all = _apply_bout_filtering(y_pred_all.copy(), mb, ma, mg)
    pooled_fm_f1 = f1_score(y, y_pred_all, zero_division=0)
    pooled_fm_p = precision_score(y, y_pred_all, zero_division=0)
    pooled_fm_r = recall_score(y, y_pred_all, zero_division=0)

    pooled_row = {
        "session": "POOLED",
        "n_frames": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_pct": float(y.sum() / len(y) * 100),
        "frame_F1": pooled_fm_f1, "frame_P": pooled_fm_p, "frame_R": pooled_fm_r,
        "frame_TP": int(((y == 1) & (y_pred_all == 1)).sum()),
        "frame_FP": int(((y == 0) & (y_pred_all == 1)).sum()),
        "frame_FN": int(((y == 1) & (y_pred_all == 0)).sum()),
    }
    for iou_t in iou_thresholds:
        # For pooled bout F1, compute per-session bout metrics and sum
        # the TP/FP/FN, then compute F1 — this is the right way to pool
        # bout-level F1 since bouts don't span sessions.
        TP = sum(r[f"bout_TP_iou{iou_t:.1f}"] for r in rows)
        FP = sum(r[f"bout_FP_iou{iou_t:.1f}"] for r in rows)
        FN = sum(r[f"bout_FN_iou{iou_t:.1f}"] for r in rows)
        P = TP / (TP + FP) if (TP + FP) else 0.0
        R = TP / (TP + FN) if (TP + FN) else 0.0
        f1v = 2 * P * R / (P + R) if (P + R) else 0.0
        pooled_row[f"bout_F1_iou{iou_t:.1f}"] = f1v
        pooled_row[f"bout_P_iou{iou_t:.1f}"]  = P
        pooled_row[f"bout_R_iou{iou_t:.1f}"]  = R
        pooled_row[f"bout_TP_iou{iou_t:.1f}"] = TP
        pooled_row[f"bout_FP_iou{iou_t:.1f}"] = FP
        pooled_row[f"bout_FN_iou{iou_t:.1f}"] = FN
    pooled_row["n_true_bouts"] = sum(r["n_true_bouts"] for r in rows)
    pooled_row["n_pred_bouts"] = sum(r["n_pred_bouts"] for r in rows)

    rows.append(pooled_row)

    df = pd.DataFrame(rows)
    csv_out = OUT_DIR / "PixelPaws_Scratching_honest_eval_per_frame_and_per_bout.csv"
    df.to_csv(csv_out, index=False)
    step(f"  -> {csv_out}")
    print("\n" + df.to_string(index=False))

    # Render a compact summary figure: frame F1 vs bout F1 across IoU thresholds
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    metrics = ["frame_F1", "bout_F1_iou0.1", "bout_F1_iou0.3", "bout_F1_iou0.5"]
    labels  = ["Per-frame", "Per-bout (IoU≥0.1)", "Per-bout (IoU≥0.3)", "Per-bout (IoU≥0.5)"]
    sess_ids = [r["session"] for r in rows[:-1]] + ["POOLED"]
    x = np.arange(len(sess_ids))
    width = 0.20
    colors = [PINK_DARK, PINK_MID, "#E91E63", PINK_LIGHT]
    for i, (m, lbl, c) in enumerate(zip(metrics, labels, colors)):
        vals = [r[m] for r in rows]
        ax.bar(x + (i - 1.5) * width, vals, width, label=lbl, color=c,
               edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_Rimonabant", "") for s in sess_ids],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.5, alpha=0.7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=False, ncol=2)

    out_png = OUT_DIR / "PixelPaws_Scratching_honest_eval_per_frame_and_per_bout.png"
    out_pdf = out_png.with_suffix(".pdf")
    out_svg = out_png.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    pooled = rows[-1]
    discord_upload(
        f"**Scratching classifier — HONEST per-frame + per-bout F1**\n"
        f"Using session-level CV cached OOF probabilities (the right way: no "
        f"frame-level leakage between bouts) and the corrected deployed params "
        f"`thr={thr:.2f} mb={mb} mg={mg}`.\n\n"
        f"**Per-frame metrics (pooled across 5 sessions):**\n"
        f"  F1 = `{pooled['frame_F1']:.3f}`  P = `{pooled['frame_P']:.3f}`  "
        f"R = `{pooled['frame_R']:.3f}`\n"
        f"**Per-bout metrics (matched by temporal IoU):**\n"
        f"  IoU≥0.1: F1=`{pooled['bout_F1_iou0.1']:.3f}`  "
        f"P=`{pooled['bout_P_iou0.1']:.3f}`  R=`{pooled['bout_R_iou0.1']:.3f}`\n"
        f"  IoU≥0.3: F1=`{pooled['bout_F1_iou0.3']:.3f}`  "
        f"P=`{pooled['bout_P_iou0.3']:.3f}`  R=`{pooled['bout_R_iou0.3']:.3f}`\n"
        f"  IoU≥0.5: F1=`{pooled['bout_F1_iou0.5']:.3f}`  "
        f"P=`{pooled['bout_P_iou0.5']:.3f}`  R=`{pooled['bout_R_iou0.5']:.3f}`\n"
        f"  True bouts: {pooled['n_true_bouts']}  Predicted bouts: {pooled['n_pred_bouts']}\n\n"
        f"PNG + PDF + SVG + per-session CSV.",
        [out_png, out_pdf, out_svg, csv_out],
    )


if __name__ == "__main__":
    main()
