"""Honest threshold curve + SHAP analysis for the Left_licking classifier.

The deployed Left_licking pkl doesn't carry OOF probabilities or
session-CV models, so this script rebuilds a per-session view of the
training data and runs honest LOVO GroupKFold:
  - Load per-session features from FeatureCache
  - Reconstruct y per session from BORIS bout exports
  - Concatenate with explicit group labels
  - 3-fold session-level GroupKFold (LOVO across 3 training sessions)
  - Collect pooled OOF probabilities
  - Render the smoothed precision/recall/F1-vs-threshold curve
  - Train final model on all sessions, run SHAP, render beeswarm + bar
"""
from __future__ import annotations
import json, sys, time, urllib.request, uuid, warnings
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURE_DIR = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/251114_Formalin_encoded/Cut/FeatureCache")
LABELS_DIR  = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/251124_Formalin_classifier_generation/Targets/Boris_Exports")
DEPLOYED_PKL = Path(r"E:/RSVIDS/Blackbox/2512_Blackbox_Formalin_Oxy/Left_paws/Classifiers/PixelPaws_Left_licking.pkl")
OUT_DIR = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/PixelPaws_Classifiers/HomeRig/licking_honest_plots")

SESSIONS = ["S1", "S2", "S3"]
SEED = 42
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
                 "User-Agent": "PixelPaws-LickingHonest/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def load_per_session():
    """Returns (X_list, y_list, n_frames_list) per session."""
    X_list, y_list = [], []
    for s in SESSIONS:
        feat_pkl = FEATURE_DIR / f"251126_Formalin_F_{s}_Cut_features.pkl"
        d = joblib.load(feat_pkl)
        X = d["X"] if isinstance(d, dict) else d
        n_frames = X.shape[0]
        step(f"  {s}: {n_frames} frames, {X.shape[1]} features")

        # Build y from BORIS bout file
        label_file = LABELS_DIR / f"251114_Formalin_{s}_Licking_barefoot_labels.csv"
        bouts = pd.read_csv(label_file)
        y = np.zeros(n_frames, dtype=int)
        for _, row in bouts.iterrows():
            start, end = int(row["frame_start"]), int(row["frame_end"])
            end = min(end, n_frames - 1)
            y[start:end + 1] = 1
        step(f"    bouts: {len(bouts)}, positive frames: {int(y.sum())} "
             f"({y.mean()*100:.2f}%)")
        X_list.append(X)
        y_list.append(y)
    return X_list, y_list


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step("loading per-session features and labels")
    X_list, y_list = load_per_session()

    # Concatenate with explicit group labels
    X_all = pd.concat(X_list, ignore_index=True)
    y_all = np.concatenate(y_list)
    groups = np.concatenate([np.full(len(yi), i) for i, yi in enumerate(y_list)])
    step(f"combined: {X_all.shape[0]} frames, {X_all.shape[1]} features, "
         f"prevalence {y_all.mean()*100:.2f}%")

    # Honest 3-fold session-level GroupKFold
    gkf = GroupKFold(n_splits=3)
    oof = np.zeros(len(y_all), dtype=np.float32)
    fold_f1s = []
    for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
        t0 = time.time()
        spw = (y_all[tr] == 0).sum() / max((y_all[tr] == 1).sum(), 1)
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw,
            tree_method="hist", random_state=SEED, n_jobs=-1,
            verbosity=0, use_label_encoder=False, eval_metric="logloss",
        )
        clf.fit(X_all.iloc[tr], y_all[tr])
        proba = clf.predict_proba(X_all.iloc[te])[:, 1]
        oof[te] = proba
        f1 = f1_score(y_all[te], (proba >= 0.5).astype(int), zero_division=0)
        fold_f1s.append(float(f1))
        held = int(groups[te][0])
        step(f"  fold {fold}/3 (test=session {SESSIONS[held]}): F1@0.5={f1:.3f} "
             f"({time.time()-t0:.1f}s)")
    step(f"per-fold F1@0.5: {fold_f1s}  mean={np.mean(fold_f1s):.3f}")

    # Threshold sweep
    thresholds = np.arange(0.005, 1.0, 0.005)
    pos = (y_all == 1)
    n_pos = int(pos.sum())
    precision, recall, frame_f1 = [], [], []
    for t in thresholds:
        pred_pos = oof >= t
        tp = int((pred_pos & pos).sum())
        fp = int((pred_pos & ~pos).sum())
        fn = n_pos - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0
        precision.append(p); recall.append(r); frame_f1.append(f)

    precision = np.array(precision); recall = np.array(recall); frame_f1 = np.array(frame_f1)
    sm = lambda v: savgol_filter(v, window_length=SAVGOL_WIN, polyorder=SAVGOL_POLY)
    p_sm, r_sm, f_sm = sm(precision), sm(recall), sm(frame_f1)
    best_i = int(np.argmax(frame_f1))
    step(f"frame-level best F1 = {frame_f1[best_i]:.3f} at thresh = {thresholds[best_i]:.3f}")

    # Threshold curve plot
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.plot(thresholds, r_sm, color="#f5a623", linewidth=2.4, label="Recall")
    ax.plot(thresholds, p_sm, color="#d12c80", linewidth=2.4, label="Precision")
    ax.plot(thresholds, f_sm, color="#5b2c91", linewidth=2.8, label="F1 Score")
    ax.axvline(thresholds[best_i], color="#888", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.scatter([thresholds[best_i]], [frame_f1[best_i]], color="#5b2c91", s=70,
               edgecolor="white", zorder=5)
    ax.annotate(f"best F1 = {frame_f1[best_i]:.2f}\nthresh = {thresholds[best_i]:.3f}",
                xy=(thresholds[best_i], frame_f1[best_i]),
                xytext=(thresholds[best_i] + 0.10, frame_f1[best_i] - 0.18),
                fontsize=10, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#5b2c91", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    thr_base = OUT_DIR / "licking_honest_threshold_curve"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(thr_base.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    step(f"wrote {thr_base}.png/svg/pdf")

    pd.DataFrame({"threshold": thresholds, "precision": precision, "recall": recall,
                  "f1": frame_f1}).to_csv(thr_base.with_suffix(".csv"), index=False)

    # Train final model on all sessions for SHAP
    step("training final model on all sessions for SHAP")
    spw_all = (y_all == 0).sum() / max((y_all == 1).sum(), 1)
    clf_final = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw_all,
        tree_method="hist", random_state=SEED, n_jobs=-1,
        verbosity=0, use_label_encoder=False, eval_metric="logloss",
    )
    clf_final.fit(X_all, y_all)

    # SHAP analysis
    step("computing SHAP values (sampled)")
    import shap
    # Sample to speed up SHAP (use stratified sample of ~5000 frames)
    rng = np.random.RandomState(SEED)
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]
    n_pos_sample = min(2500, len(pos_idx))
    n_neg_sample = min(2500, len(neg_idx))
    sample_idx = np.concatenate([
        rng.choice(pos_idx, n_pos_sample, replace=False),
        rng.choice(neg_idx, n_neg_sample, replace=False),
    ])
    X_sample = X_all.iloc[sample_idx]
    explainer = shap.TreeExplainer(clf_final)
    shap_vals = explainer.shap_values(X_sample)
    step(f"SHAP values shape: {shap_vals.shape}")

    # SHAP beeswarm
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_vals, X_sample, max_display=12, show=False,
                      plot_size=(12, 6))
    plt.title("Feature Importance -- Left_licking", fontsize=12, fontweight="bold")
    plt.tight_layout()
    shap_base = OUT_DIR / "licking_shap_beeswarm"
    for ext in ("png", "svg", "pdf"):
        plt.savefig(shap_base.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close()
    step(f"wrote {shap_base}.png/svg/pdf")

    # SHAP bar (mean |SHAP|)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sample, max_display=12, plot_type="bar",
                      show=False, plot_size=(10, 6))
    plt.title("Mean |SHAP value| -- Left_licking", fontsize=12, fontweight="bold")
    plt.tight_layout()
    bar_base = OUT_DIR / "licking_shap_bar"
    for ext in ("png", "svg", "pdf"):
        plt.savefig(bar_base.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close()
    step(f"wrote {bar_base}.png/svg/pdf")

    head = ("**Left_licking classifier -- honest threshold curve + SHAP**\n"
            f"Rebuilt per-session training data from FeatureCache + BORIS bout "
            f"exports ({len(X_list)} sessions, {X_all.shape[0]} frames, "
            f"{X_all.shape[1]} features, prevalence {y_all.mean()*100:.2f}%). "
            f"Honest 3-fold session-level GroupKFold (LOVO).\n"
            f"Per-fold F1 @ thresh=0.5: {[round(x,3) for x in fold_f1s]} "
            f"(mean = {np.mean(fold_f1s):.3f}).\n"
            f"Frame-level best F1 = {frame_f1[best_i]:.3f} at thresh = "
            f"{thresholds[best_i]:.3f}. SHAP computed on a stratified 5000-frame "
            f"sample (2500 +/2500 -) of the trained-on-all-sessions model. "
            f"Smoothed Savitzky-Golay (win={SAVGOL_WIN}, poly={SAVGOL_POLY}).")
    discord_upload(head, [
        thr_base.with_suffix(".png"), thr_base.with_suffix(".svg"),
        thr_base.with_suffix(".pdf"), thr_base.with_suffix(".csv"),
        shap_base.with_suffix(".png"), shap_base.with_suffix(".svg"),
        shap_base.with_suffix(".pdf"),
        bar_base.with_suffix(".png"), bar_base.with_suffix(".svg"),
        bar_base.with_suffix(".pdf"),
    ])


if __name__ == "__main__":
    main()
