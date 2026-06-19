"""
Honest head-to-head: does SHAP pruning (600 -> 120 features) improve
held-out F1 for Scratching?

Trains a 120-feature model using **session-level GroupKFold** (correct
for behavioral time series — no frame-level leakage). Compares per-frame
and per-bout F1 against the deployed 600-feature pkl (whose OOF
probabilities are already from session-level CV via _real_training).
"""
from __future__ import annotations
import json, pickle, sys, time, urllib.request, uuid, warnings
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
from evaluation_tab import _apply_bout_filtering

PROJ = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected")
CLF_DEPLOYED = PROJ / "Classifiers" / "PixelPaws_Scratching.pkl"
CLF_PRUNED   = PROJ / "Classifiers" / "PixelPaws_Scratching_pruned_120_20260429_153755.pkl"
TRAIN = PROJ / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
LABELS_DIR = PROJ / "labels"
OUT_DIR = PROJ / "Classifiers" / "plots" / "pruning_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = ("")
SEED = 42
PINK_LIGHT = "#F8BBD0"; PINK_MID = "#EC407A"; PINK_DARK = "#C2185B"


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
                 "User-Agent": "PixelPaws-PruneCompare/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def find_bouts(y):
    if len(y) == 0: return []
    padded = np.concatenate([[0], y.astype(int), [0]])
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0].tolist(),
                    (np.where(d == -1)[0] - 1).tolist()))


def slice_oof_per_session(y_global, training_sessions, labels_dir):
    pos_per = {}
    for s in training_sessions:
        df = pd.read_csv(labels_dir / f"{s}_Labels.csv")
        col = next((c for c in df.columns if c.lower() == "scratching"), None)
        pos_per[s] = int(df[col].sum()) if col else 0
    pos_idx = np.where(y_global == 1)[0]
    cum_target = 0; bounds = [0]
    for s in training_sessions[:-1]:
        cum_target += pos_per[s]
        last_pos = pos_idx[cum_target - 1]
        first_next = pos_idx[cum_target] if cum_target < len(pos_idx) else len(y_global)
        bounds.append((last_pos + first_next) // 2 + 1)
    bounds.append(len(y_global))
    return bounds


def make_xgb(spw, n_est=300):
    return XGBClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method="hist", random_state=SEED, n_jobs=-1,
        verbosity=0, use_label_encoder=False, eval_metric="logloss")


def session_cv_oof(X, y, session_ids, n_est=300, label=""):
    """Session-level GroupKFold OOF probabilities."""
    gkf = GroupKFold(n_splits=len(np.unique(session_ids)))
    oof = np.zeros(len(y), dtype=np.float32)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=session_ids), 1):
        t0 = time.time()
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = make_xgb(spw=spw, n_est=n_est)
        X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        X_te = X.iloc[te] if hasattr(X, "iloc") else X[te]
        clf.fit(X_tr, y[tr])
        oof[te] = clf.predict_proba(X_te)[:, 1]
        held = int(session_ids[te][0])
        step(f"  [{label}] fold {fold}/5 (test=session {held}) "
             f"({time.time()-t0:.1f}s)")
    return oof


def bout_metrics_pooled(y, oof_pred, sessions, bounds, iou_thr=0.3):
    """Compute pooled bout-level F1 from binary predictions per session."""
    def iou(a, b):
        s = max(a[0], b[0]); e = min(a[1], b[1])
        inter = max(0, e - s + 1)
        union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
        return inter / union if union > 0 else 0.0
    TP = FP = FN = 0
    for k in range(len(sessions)):
        lo, hi = bounds[k], bounds[k + 1]
        tb = find_bouts(y[lo:hi])
        pb = find_bouts(oof_pred[lo:hi])
        matched = set(); tp = 0
        for tbb in tb:
            best_i, best_v = -1, 0.0
            for i, pbb in enumerate(pb):
                if i in matched: continue
                v = iou(tbb, pbb)
                if v > best_v: best_v = v; best_i = i
            if best_i >= 0 and best_v >= iou_thr:
                matched.add(best_i); tp += 1
        TP += tp; FP += len(pb) - len(matched); FN += len(tb) - tp
    P = TP/(TP+FP) if (TP+FP) else 0.0
    R = TP/(TP+FN) if (TP+FN) else 0.0
    f1 = 2*P*R/(P+R) if (P+R) else 0.0
    return f1, P, R, TP, FP, FN


def evaluate(y, oof, sessions, bounds, mb=20, ma=0, mg=0):
    """Sweep threshold, find OOF-best F1, then report frame + bout F1."""
    thrs = np.linspace(0.05, 0.95, 19)
    best_f1, best_t = -1.0, 0.5
    for t in thrs:
        yp = (oof >= t).astype(int)
        yp = _apply_bout_filtering(yp.copy(), mb, ma, mg)
        f1 = f1_score(y, yp, zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    yp = (oof >= best_t).astype(int)
    yp = _apply_bout_filtering(yp.copy(), mb, ma, mg)
    frame_f1 = f1_score(y, yp, zero_division=0)
    frame_p = precision_score(y, yp, zero_division=0)
    frame_r = recall_score(y, yp, zero_division=0)
    bf1, bp, br, tp_b, fp_b, fn_b = bout_metrics_pooled(
        y, yp, sessions, bounds, iou_thr=0.3)
    return {
        "best_thr": float(best_t),
        "frame_F1": float(frame_f1),
        "frame_P": float(frame_p),
        "frame_R": float(frame_r),
        "bout_F1_iou0.3": float(bf1),
        "bout_P_iou0.3": float(bp),
        "bout_R_iou0.3": float(br),
        "bout_TP": int(tp_b), "bout_FP": int(fp_b), "bout_FN": int(fn_b),
    }


def main():
    step("loading classifier + training set")
    deployed = joblib.load(CLF_DEPLOYED)
    pruned   = joblib.load(CLF_PRUNED)
    train = pickle.load(open(TRAIN, "rb"))
    X, y = train["X"], train["y"]
    sessions = list(deployed["training_sessions"])
    bounds = slice_oof_per_session(y, sessions, LABELS_DIR)
    session_ids = np.zeros(len(y), dtype=int)
    for k in range(len(sessions)):
        session_ids[bounds[k]:bounds[k + 1]] = k

    # Full-feature: reuse cached deployed OOF
    step("\n=== EVAL: 600 features (cached session-CV OOF) ===")
    oof_full = deployed["oof_proba"]
    res_full = evaluate(y, oof_full, sessions, bounds, mb=20, ma=0, mg=0)
    step(f"  frame F1={res_full['frame_F1']:.3f} (thr={res_full['best_thr']:.2f}) "
         f"bout F1(IoU=0.3)={res_full['bout_F1_iou0.3']:.3f}")

    # 120-feature: retrain with session-CV
    step("\n=== EVAL: 120 features (SHAP-pruned, retrain w/ GroupKFold) ===")
    top120 = list(pruned["selected_feature_cols"])
    avail = [str(f) for f in top120 if str(f) in X.columns]
    missing = [str(f) for f in top120 if str(f) not in X.columns]
    step(f"  top-120 features: {len(avail)} available, {len(missing)} missing")
    X_pruned = X[avail]
    step(f"  retraining 5-fold GroupKFold on n={X_pruned.shape[0]:,} x {X_pruned.shape[1]}...")
    t0 = time.time()
    oof_pruned = session_cv_oof(X_pruned, y, session_ids, n_est=300, label="P120")
    step(f"  total retrain time: {time.time() - t0:.1f}s")
    res_pruned = evaluate(y, oof_pruned, sessions, bounds, mb=20, ma=0, mg=0)
    step(f"  frame F1={res_pruned['frame_F1']:.3f} (thr={res_pruned['best_thr']:.2f}) "
         f"bout F1(IoU=0.3)={res_pruned['bout_F1_iou0.3']:.3f}")

    df = pd.DataFrame([
        {"model": "All 600 features", **res_full},
        {"model": "Pruned 120 features", **res_pruned},
    ])
    csv_out = OUT_DIR / "scratching_pruning_compare.csv"
    df.to_csv(csv_out, index=False)
    print("\n" + df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    x = np.arange(2)
    width = 0.35
    frame = [res_full["frame_F1"], res_pruned["frame_F1"]]
    bout  = [res_full["bout_F1_iou0.3"], res_pruned["bout_F1_iou0.3"]]
    ax.bar(x - width/2, frame, width, label="Per-frame F1", color=PINK_DARK,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, bout,  width, label="Per-bout F1 (IoU≥0.3)", color=PINK_LIGHT,
           edgecolor="black", linewidth=0.5)
    for i, v in enumerate(frame):
        ax.text(i - width/2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(bout):
        ax.text(i + width/2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["All 600 features", "Pruned 120 features"])
    ax.set_ylabel("F1 score (honest session-CV)")
    ax.set_ylim(0, 1.0); ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.set_title("SHAP pruning effect on honest F1", fontweight="bold")

    out_png = OUT_DIR / "scratching_pruning_compare.png"
    out_pdf = out_png.with_suffix(".pdf")
    out_svg = out_png.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    discord_upload(
        f"**Scratching — does SHAP pruning improve honest F1?**\n"
        f"Both models retrained with session-level GroupKFold (5 sessions, each "
        f"held out once). Same hyperparams, same post-processing (mb=20, mg=0).\n\n"
        f"**All 600 features:** frame F1=`{res_full['frame_F1']:.3f}`  "
        f"bout F1(IoU=0.3)=`{res_full['bout_F1_iou0.3']:.3f}`  "
        f"best thr=`{res_full['best_thr']:.2f}`\n"
        f"**Pruned 120 features:** frame F1=`{res_pruned['frame_F1']:.3f}`  "
        f"bout F1(IoU=0.3)=`{res_pruned['bout_F1_iou0.3']:.3f}`  "
        f"best thr=`{res_pruned['best_thr']:.2f}`\n\n"
        f"Δ frame F1 = `{res_pruned['frame_F1'] - res_full['frame_F1']:+.3f}`  "
        f"Δ bout F1 = `{res_pruned['bout_F1_iou0.3'] - res_full['bout_F1_iou0.3']:+.3f}`\n"
        f"PNG + PDF + SVG + CSV.",
        [out_png, out_pdf, out_svg, csv_out],
    )


if __name__ == "__main__":
    main()
