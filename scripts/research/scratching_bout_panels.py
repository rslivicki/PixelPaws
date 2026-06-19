"""
Honest per-bout learning curve + threshold sweep for Scratching.

Panel B (learning curve, bout-level):
  - Subsample positive *bouts* from each session, keep all negatives
  - Session-level GroupKFold (each fold = one session held-out)
  - Plot bout-level F1 (IoU >= 0.5) vs total bout-positive frames

Panel C (threshold curves, bout-level):
  - Cached OOF probabilities (already session-level, honest)
  - Sweep threshold 0.01..0.99
  - Compute bout-level precision/recall/F1 (IoU >= 0.5) at each threshold
  - Apply post-processing (min_bout, max_gap) before bout extraction
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
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
from evaluation_tab import _apply_bout_filtering

PROJ = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected")
CLF  = PROJ / "Classifiers" / "PixelPaws_Scratching.pkl"
TRAIN = PROJ / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
LABELS_DIR = PROJ / "labels"
OUT_DIR = PROJ / "Classifiers" / "plots" / "bout_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = ("")

PINK_LIGHT = "#F8BBD0"; PINK_MID = "#EC407A"; PINK_DARK = "#C2185B"
PINK_VLIGHT = "#FCE4EC"; GRID_GREY = "#E0E0E0"; TEXT_GREY = "#666666"
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
                 "User-Agent": "PixelPaws-BoutPanels/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def find_bouts(y):
    if len(y) == 0:
        return []
    padded = np.concatenate([[0], y.astype(int), [0]])
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def bout_pr_f1(y_true, y_pred, iou_thr=0.3):
    """Bout-level (precision, recall, F1) at given IoU threshold."""
    true_bouts = find_bouts(y_true)
    pred_bouts = find_bouts(y_pred)
    n_t = len(true_bouts); n_p = len(pred_bouts)

    def iou(a, b):
        s = max(a[0], b[0]); e = min(a[1], b[1])
        inter = max(0, e - s + 1)
        union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
        return inter / union if union > 0 else 0.0

    matched_pred = set()
    TP = 0
    for tb in true_bouts:
        best_pi, best_iou = -1, 0.0
        for pi, pb in enumerate(pred_bouts):
            if pi in matched_pred: continue
            v = iou(tb, pb)
            if v > best_iou:
                best_iou = v; best_pi = pi
        if best_pi >= 0 and best_iou >= iou_thr:
            matched_pred.add(best_pi); TP += 1
    FP = n_p - len(matched_pred)
    FN = n_t - TP
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, f1, TP, FP, FN


def slice_oof_per_session(y_global, training_sessions, labels_dir):
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


def _make_xgb(spw, n_est=200):
    return XGBClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method="hist", random_state=SEED, n_jobs=-1,
        verbosity=0, use_label_encoder=False, eval_metric="logloss")


def panel_c_threshold_curves(y_global, oof, bounds, sessions, deployed_thr,
                              mb, ma, mg, out_stem: Path):
    """Bout-level Precision/Recall/F1 vs threshold (IoU=0.5), with post-proc."""
    thrs = np.linspace(0.02, 0.95, 47)
    P, R, F = [], [], []
    for t in thrs:
        # Apply threshold + post-proc per session, then compute pooled
        # bout TP/FP/FN
        TP_all = FP_all = FN_all = 0
        for k in range(len(sessions)):
            lo, hi = bounds[k], bounds[k + 1]
            y_s = y_global[lo:hi]
            yp = (oof[lo:hi] >= t).astype(int)
            yp = _apply_bout_filtering(yp.copy(), mb, ma, mg)
            _, _, _, tp, fp, fn = bout_pr_f1(y_s, yp, iou_thr=0.3)
            TP_all += tp; FP_all += fp; FN_all += fn
        prec = TP_all / (TP_all + FP_all) if (TP_all + FP_all) else 0.0
        rec  = TP_all / (TP_all + FN_all) if (TP_all + FN_all) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        P.append(prec); R.append(rec); F.append(f1)

    P, R, F = map(np.asarray, (P, R, F))
    best_idx = int(np.argmax(F))
    best_t = float(thrs[best_idx])
    best_f1 = float(F[best_idx])
    step(f"  bout threshold sweep: best at thr={best_t:.2f}, bout-F1(IoU=0.5)={best_f1:.3f}")

    fig, ax = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    ax.plot(thrs, R, color=PINK_LIGHT, linewidth=1.8, label="Bout Recall (IoU≥0.3)")
    ax.plot(thrs, P, color=PINK_MID,   linewidth=1.8, label="Bout Precision (IoU≥0.3)")
    ax.plot(thrs, F, color=PINK_DARK,  linewidth=2.2, label="Bout F1 (IoU≥0.3)")
    ax.axvline(best_t, color=PINK_DARK, linestyle="--", linewidth=1.0,
               alpha=0.85, label=f"best thr={best_t:.2f}")
    ax.axvline(deployed_thr, color=TEXT_GREY, linestyle=":", linewidth=1.0,
               alpha=0.85, label=f"deployed thr={deployed_thr:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Bout-level score")
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


def subsample_positive_bouts(y, bouts, frac, rng):
    """Keep `frac` of the bouts; return new y vector with only those
    bouts' positives. (Negatives stay; bouts not selected get zeroed.)"""
    n_keep = max(2, int(round(frac * len(bouts))))
    n_keep = min(n_keep, len(bouts))
    keep_idx = rng.choice(len(bouts), size=n_keep, replace=False)
    keep_set = set(int(i) for i in keep_idx)
    new_y = np.zeros_like(y)
    for i, (s, e) in enumerate(bouts):
        if i in keep_set:
            new_y[s:e + 1] = 1
    return new_y, n_keep


def panel_b_learning_curve(X, y_global, sessions, bounds, session_ids,
                            best_thr, mb, ma, mg, fractions=None):
    """Session-level GroupKFold learning curve, bout-level F1 at IoU=0.5.
    For each fraction: keep that fraction of TRUE BOUTS per session; run
    5-fold GroupKFold (each session-as-test); compute per-fold bout F1;
    aggregate."""
    if fractions is None:
        fractions = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    rng = np.random.default_rng(SEED)
    out = []

    # Bouts per session (from full y)
    bouts_per_sess = []
    for k in range(len(sessions)):
        lo, hi = bounds[k], bounds[k + 1]
        b_s = find_bouts(y_global[lo:hi])
        bouts_per_sess.append([(s + lo, e + lo) for s, e in b_s])
    total_bouts = sum(len(b) for b in bouts_per_sess)
    step(f"  total true bouts across 5 sessions: {total_bouts}")

    for frac in fractions:
        # Subsample bouts per session
        y_sub = np.zeros_like(y_global)
        total_kept_bouts = 0
        for k in range(len(sessions)):
            bouts = bouts_per_sess[k]
            n_keep = max(2, int(round(frac * len(bouts))))
            n_keep = min(n_keep, len(bouts))
            keep_idx = rng.choice(len(bouts), size=n_keep, replace=False)
            for i in keep_idx:
                s, e = bouts[i]
                y_sub[s:e + 1] = 1
            total_kept_bouts += n_keep
        n_pos_frames = int(y_sub.sum())

        step(f"  frac={frac:.2f}  kept_bouts={total_kept_bouts}  "
             f"pos_frames={n_pos_frames}  -> training 5 folds...")

        # Session-level GroupKFold (5 folds, each session held out once)
        gkf = GroupKFold(n_splits=len(sessions))
        f1s = []
        for fold, (tr, te) in enumerate(gkf.split(X, y_sub, groups=session_ids), 1):
            t0 = time.time()
            spw = (y_sub[tr] == 0).sum() / max((y_sub[tr] == 1).sum(), 1)
            clf = _make_xgb(spw=spw, n_est=200)
            X_tr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
            X_te = X.iloc[te] if isinstance(X, pd.DataFrame) else X[te]
            clf.fit(X_tr, y_sub[tr])
            p = clf.predict_proba(X_te)[:, 1]
            yp = (p >= best_thr).astype(int)
            yp = _apply_bout_filtering(yp.copy(), mb, ma, mg)
            _, _, bf1, _, _, _ = bout_pr_f1(y_sub[te], yp, iou_thr=0.3)
            f1s.append(bf1)
            step(f"    fold {fold}/{len(sessions)}  bout-F1={bf1:.3f}  "
                 f"({time.time()-t0:.1f}s)")
        out.append((n_pos_frames, total_kept_bouts,
                    float(np.mean(f1s)), float(np.std(f1s))))
        step(f"  -> frac={frac:.2f}  mean bout-F1={out[-1][2]:.3f}±{out[-1][3]:.3f}")
    return out


def plot_panel_b(lc, out_stem: Path):
    xs    = np.array([r[0] for r in lc])  # bout-positive frames
    means = np.array([r[2] for r in lc])
    stds  = np.array([r[3] for r in lc])
    fig, ax = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    ax.plot(xs, means, marker="o", markersize=6, linewidth=2.0,
            color=PINK_DARK, label="Scratching (bout F1, IoU≥0.3)", zorder=3)
    ax.fill_between(xs, means - stds, means + stds,
                    color=PINK_MID, alpha=0.18, zorder=2)
    ax.set_xlabel("Bout-positive frames")
    ax.set_ylabel("Bout-F1 (5-fold session-CV)")
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
    sessions = list(cd["training_sessions"])
    thr = float(cd["best_thresh"]); mb = int(cd["min_bout"])
    ma = int(cd["min_after_bout"]); mg = int(cd["max_gap"])

    bounds, pos_per = slice_oof_per_session(y, sessions, LABELS_DIR)
    step(f"  session boundaries: {bounds}")

    # Build session_ids vector for GroupKFold
    session_ids = np.zeros(len(y), dtype=int)
    for k in range(len(sessions)):
        session_ids[bounds[k]:bounds[k + 1]] = k

    step("=== Panel C: bout-level threshold sweep (cached OOF) ===")
    pc_stem = OUT_DIR / "PixelPaws_Scratching_panelC_bout_threshold_iou0.3"
    pc_png, pc_pdf, pc_svg, best_t, best_bf1 = panel_c_threshold_curves(
        y, oof, bounds, sessions, thr, mb, ma, mg, pc_stem)

    discord_upload(
        f"**Scratching classifier — HONEST per-bout threshold curves** "
        f"(session-level CV cached OOF, bout F1 at IoU≥0.3)\n"
        f"Best at thr=`{best_t:.2f}` (bout-F1=`{best_bf1:.3f}`). "
        f"Deployed at thr=`{thr:.2f}`, mb={mb}, mg={mg}. Post-processing "
        f"is applied before bout extraction. PNG + PDF + SVG.",
        [pc_png, pc_pdf, pc_svg],
    )


if __name__ == "__main__":
    main()
