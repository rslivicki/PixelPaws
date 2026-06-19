"""
Retrain the Scratching classifier with the honest pipeline:
  1. Session-level GroupKFold (no frame leakage)
  2. Full 600-feature XGBoost + OOF + post-proc sweep
  3. SHAP on final 600-feature model -> top 120 features
  4. Pruned 120-feature retrain + OOF + post-proc sweep
  5. Side-by-side honest evaluation (frame + bout F1 at IoU 0.3 + 0.5)
  6. Save NEW pkls alongside existing (no clobbering)
  7. Render comparison panel + post to #results
"""
from __future__ import annotations
import json, pickle, shutil, sys, time, urllib.request, uuid, warnings
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
from datetime import datetime as _dt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
from evaluation_tab import _apply_bout_filtering

PROJ = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected")
CLF_CURRENT = PROJ / "Classifiers" / "PixelPaws_Scratching.pkl"
TRAIN = PROJ / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
LABELS_DIR = PROJ / "labels"
CLF_DIR = PROJ / "Classifiers"
RUN_TS = _dt.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = CLF_DIR / f"PixelPaws_Scratching_honest_{RUN_TS}"
PLOTS = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

WEBHOOK = ("")
SEED = 42
PINK_LIGHT = "#F8BBD0"; PINK_MID = "#EC407A"; PINK_DARK = "#C2185B"

# Post-proc grid (same as evaluation_tab's standard)
THRESHOLDS = np.arange(0.05, 0.96, 0.025)
MIN_BOUTS  = [1, 2, 3, 5, 8, 12, 15, 20]
MIN_AFTER  = [0, 1, 3, 5]
MAX_GAPS   = [0, 2, 4, 5, 6, 10, 15]


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
                ".svg": "image/svg+xml", ".csv": "text/csv",
                ".pkl": "application/octet-stream"}.get(
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
                 "User-Agent": "PixelPaws-HonestPruned/1.0"})
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


def bout_metrics_pooled(y, y_pred, sessions, bounds, iou_thr=0.3):
    def iou(a, b):
        s = max(a[0], b[0]); e = min(a[1], b[1])
        inter = max(0, e - s + 1)
        union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
        return inter / union if union > 0 else 0.0
    TP = FP = FN = 0
    for k in range(len(sessions)):
        lo, hi = bounds[k], bounds[k + 1]
        tb = find_bouts(y[lo:hi])
        pb = find_bouts(y_pred[lo:hi])
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
    return (2*P*R/(P+R) if (P+R) else 0.0), P, R, TP, FP, FN


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


def make_xgb(spw, n_est=400):
    return XGBClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method="hist", random_state=SEED, n_jobs=-1,
        verbosity=0, use_label_encoder=False, eval_metric="logloss")


def session_cv_train(X, y, session_ids, n_est=400, label=""):
    """Session-level GroupKFold: train one model per held-out session, return
    OOF probs + fold models + per-fold F1 (at thr=0.5)."""
    gkf = GroupKFold(n_splits=len(np.unique(session_ids)))
    oof = np.zeros(len(y), dtype=np.float32)
    fold_models = []
    fold_f1s = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=session_ids), 1):
        t0 = time.time()
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = make_xgb(spw=spw, n_est=n_est)
        X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        X_te = X.iloc[te] if hasattr(X, "iloc") else X[te]
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)[:, 1]
        oof[te] = proba
        f1 = f1_score(y[te], (proba >= 0.5).astype(int), zero_division=0)
        fold_models.append(clf); fold_f1s.append(float(f1))
        held = int(session_ids[te][0])
        step(f"  [{label}] fold {fold}/5 (test=session {held}) "
             f"F1@0.5={f1:.3f}  ({time.time()-t0:.1f}s)")
    return oof, fold_models, fold_f1s


def sweep_post_proc(oof, y):
    """Joint 4D grid search."""
    best = {"thresh": 0.5, "min_bout": 1, "min_after_bout": 0,
            "max_gap": 0, "f1": -1.0}
    for t in THRESHOLDS:
        yr = (oof >= t).astype(int)
        for mb in MIN_BOUTS:
            for ma in MIN_AFTER:
                for mg in MAX_GAPS:
                    if mb == 1 and ma == 0 and mg == 0:
                        yf = yr
                    else:
                        yf = _apply_bout_filtering(yr.copy(), mb, ma, mg)
                    f = f1_score(y, yf, zero_division=0)
                    if f > best["f1"]:
                        best = {"thresh": float(round(t, 3)),
                                "min_bout": mb, "min_after_bout": ma,
                                "max_gap": mg, "f1": float(f)}
    return best


def evaluate_with_post(y, oof, best_params, sessions, bounds):
    mb, ma, mg = best_params["min_bout"], best_params["min_after_bout"], best_params["max_gap"]
    thr = best_params["thresh"]
    yp = (oof >= thr).astype(int)
    yp = _apply_bout_filtering(yp.copy(), mb, ma, mg)
    frame_f1 = f1_score(y, yp, zero_division=0)
    frame_p = precision_score(y, yp, zero_division=0)
    frame_r = recall_score(y, yp, zero_division=0)
    bf1_03, bp_03, br_03, *_ = bout_metrics_pooled(y, yp, sessions, bounds, iou_thr=0.3)
    bf1_05, bp_05, br_05, *_ = bout_metrics_pooled(y, yp, sessions, bounds, iou_thr=0.5)
    return {
        "thresh": thr, "min_bout": mb, "min_after_bout": ma, "max_gap": mg,
        "frame_F1": float(frame_f1), "frame_P": float(frame_p), "frame_R": float(frame_r),
        "bout_F1_iou0.3": float(bf1_03), "bout_P_iou0.3": float(bp_03), "bout_R_iou0.3": float(br_03),
        "bout_F1_iou0.5": float(bf1_05), "bout_P_iou0.5": float(bp_05), "bout_R_iou0.5": float(br_05),
    }


def compute_shap_topk(model, X, y, k=120, sample_n=8000):
    """Compute SHAP importance and return top-k feature names."""
    import shap
    rng = np.random.default_rng(SEED)
    if len(X) > sample_n:
        idx = rng.choice(len(X), size=sample_n, replace=False)
        X_s = X.iloc[idx]
    else:
        X_s = X
    step(f"    SHAP on {len(X_s):,} rows × {X_s.shape[1]} features...")
    t0 = time.time()
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X_s)
    if isinstance(sv, list): sv = sv[1]
    importance = np.abs(sv).mean(axis=0)
    step(f"    SHAP done in {time.time()-t0:.1f}s")
    order = np.argsort(importance)[::-1]
    feat_names = list(X.columns)
    return [feat_names[i] for i in order[:k]], importance[order[:k]]


def save_pkl(model, X_cols, oof, best_params, fold_f1s, fold_models,
              behavior, training_sessions, calibrator, out_path: Path,
              note=""):
    cd = {
        "clf_model": model,
        "Behavior_type": behavior,
        "selected_feature_cols": X_cols if len(X_cols) < 600 else None,
        "best_thresh": best_params["thresh"],
        "min_bout": best_params["min_bout"],
        "min_after_bout": best_params["min_after_bout"],
        "max_gap": best_params["max_gap"],
        "ui_min_bout": best_params["min_bout"],
        "ui_min_after_bout": best_params["min_after_bout"],
        "ui_max_gap": best_params["max_gap"],
        "bp_pixbrt_list": ["hrpaw", "hlpaw", "snout"],
        "square_size": [40, 40, 15],
        "pix_threshold": 0.4,
        "include_optical_flow": False,
        "bp_optflow_list": [],
        "training_sessions": list(training_sessions),
        "cv_f1_scores": list(fold_f1s),
        "mean_cv_f1": float(np.mean(fold_f1s)),
        "std_cv_f1": float(np.std(fold_f1s)),
        "oof_best_f1": best_params["f1"],
        "oof_best_params": best_params,
        "oof_proba": oof,
        "prob_calibrator": calibrator,
        "fold_models": fold_models,
        "use_egocentric": True,
        "use_lag_features": True,
        "use_contact_features": True,
        "contact_threshold": 15.0,
        "use_multiscale_features": False,
        "optuna_best_params": None,
        "scale_pos_weight": float((np.array(oof) >= 0).sum()),  # placeholder
        "hmm_log_trans": None,
        "hmm_log_prior": None,
        "honest_training_note": (
            f"Session-level GroupKFold (no frame leakage). {note}"),
    }
    joblib.dump(cd, out_path)


def main():
    step("loading training data + cached classifier")
    deployed = joblib.load(CLF_CURRENT)
    train = pickle.load(open(TRAIN, "rb"))
    X, y = train["X"], train["y"]
    sessions = list(deployed["training_sessions"])
    bounds = slice_oof_per_session(y, sessions, LABELS_DIR)
    session_ids = np.zeros(len(y), dtype=int)
    for k in range(len(sessions)):
        session_ids[bounds[k]:bounds[k + 1]] = k
    step(f"  data shape: {X.shape}  positives: {int(y.sum())}  "
         f"sessions: {len(sessions)}")

    # ── (A) Full 600-feature, session-CV train ──────────────────────
    step("\n=== (A) FULL 600-feature: GroupKFold session-CV training ===")
    oof_full, fold_models_full, fold_f1s_full = session_cv_train(
        X, y, session_ids, n_est=400, label="FULL600")
    step(f"  CV F1 per-fold: {[round(f,3) for f in fold_f1s_full]}  "
         f"mean={np.mean(fold_f1s_full):.3f}±{np.std(fold_f1s_full):.3f}")

    # Isotonic calibration on full OOF
    calib_full = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calib_full.fit(oof_full, y)
    oof_full_cal = np.clip(calib_full.predict(oof_full), 0, 1).astype(np.float32)

    step("  sweeping post-proc (4D grid)...")
    best_full = sweep_post_proc(oof_full_cal, y)
    step(f"  best params (calibrated): {best_full}")

    eval_full = evaluate_with_post(y, oof_full_cal, best_full, sessions, bounds)
    step(f"  EVAL FULL: frame F1={eval_full['frame_F1']:.3f} "
         f"bout F1(IoU=0.3)={eval_full['bout_F1_iou0.3']:.3f}")

    # Train final 600-feature model on ALL data
    step("  training final 600-feature model on full data...")
    spw_full = (y == 0).sum() / max((y == 1).sum(), 1)
    final_full = make_xgb(spw=spw_full, n_est=500); final_full.fit(X, y)

    # ── (B) SHAP -> top 120 features ────────────────────────────────
    step("\n=== (B) SHAP feature importance + prune to top 120 ===")
    top120, top120_importance = compute_shap_topk(final_full, X, y, k=120)
    step(f"  top 5: {top120[:5]}")
    step(f"  bottom 5 (of top 120): {top120[-5:]}")

    # ── (C) Pruned 120-feature, session-CV retrain ──────────────────
    step("\n=== (C) PRUNED 120-feature: GroupKFold session-CV retrain ===")
    X_pruned = X[top120]
    oof_pruned, fold_models_pruned, fold_f1s_pruned = session_cv_train(
        X_pruned, y, session_ids, n_est=400, label="PRUNED120")
    step(f"  CV F1 per-fold: {[round(f,3) for f in fold_f1s_pruned]}  "
         f"mean={np.mean(fold_f1s_pruned):.3f}±{np.std(fold_f1s_pruned):.3f}")

    calib_pruned = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calib_pruned.fit(oof_pruned, y)
    oof_pruned_cal = np.clip(calib_pruned.predict(oof_pruned), 0, 1).astype(np.float32)

    step("  sweeping post-proc (4D grid)...")
    best_pruned = sweep_post_proc(oof_pruned_cal, y)
    step(f"  best params (calibrated): {best_pruned}")

    eval_pruned = evaluate_with_post(y, oof_pruned_cal, best_pruned, sessions, bounds)
    step(f"  EVAL PRUNED: frame F1={eval_pruned['frame_F1']:.3f} "
         f"bout F1(IoU=0.3)={eval_pruned['bout_F1_iou0.3']:.3f}")

    step("  training final 120-feature model on full data...")
    final_pruned = make_xgb(spw=spw_full, n_est=500); final_pruned.fit(X_pruned, y)

    # ── (D) Save BOTH pkls in the run folder (no clobbering of deployed) ──
    full_pkl_path = RUN_DIR / f"PixelPaws_Scratching_AllFeatures_honest_{RUN_TS}.pkl"
    pruned_pkl_path = RUN_DIR / f"PixelPaws_Scratching_pruned_120_honest_{RUN_TS}.pkl"
    save_pkl(final_full, list(X.columns), oof_full_cal, best_full,
              fold_f1s_full, fold_models_full, "Scratching",
              sessions, calib_full, full_pkl_path,
              note="full 600 features, session-level GroupKFold + isotonic calibration")
    save_pkl(final_pruned, top120, oof_pruned_cal, best_pruned,
              fold_f1s_pruned, fold_models_pruned, "Scratching",
              sessions, calib_pruned, pruned_pkl_path,
              note="SHAP-pruned top 120, session-level GroupKFold + isotonic calibration")
    step(f"  saved: {full_pkl_path.name}")
    step(f"  saved: {pruned_pkl_path.name}")

    # ── (E) Comparison plot + CSV ───────────────────────────────────
    rows = [
        {"model": "Deployed (cached OOF)",
         "frame_F1": 0.722, "bout_F1_iou0.3": 0.812, "bout_F1_iou0.5": 0.649,
         "thresh": 0.50, "mb": 20, "ma": 0, "mg": 0,
         "cv_mean": 0.641, "cv_std": 0.073},
        {"model": "Honest 600 features",
         "frame_F1": eval_full["frame_F1"], "bout_F1_iou0.3": eval_full["bout_F1_iou0.3"],
         "bout_F1_iou0.5": eval_full["bout_F1_iou0.5"],
         "thresh": best_full["thresh"], "mb": best_full["min_bout"],
         "ma": best_full["min_after_bout"], "mg": best_full["max_gap"],
         "cv_mean": float(np.mean(fold_f1s_full)), "cv_std": float(np.std(fold_f1s_full))},
        {"model": "Honest pruned 120",
         "frame_F1": eval_pruned["frame_F1"], "bout_F1_iou0.3": eval_pruned["bout_F1_iou0.3"],
         "bout_F1_iou0.5": eval_pruned["bout_F1_iou0.5"],
         "thresh": best_pruned["thresh"], "mb": best_pruned["min_bout"],
         "ma": best_pruned["min_after_bout"], "mg": best_pruned["max_gap"],
         "cv_mean": float(np.mean(fold_f1s_pruned)), "cv_std": float(np.std(fold_f1s_pruned))},
    ]
    df = pd.DataFrame(rows)
    csv_out = PLOTS / "scratching_honest_compare.csv"
    df.to_csv(csv_out, index=False)
    print("\n" + df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    x = np.arange(3); width = 0.27
    f_vals = [r["frame_F1"] for r in rows]
    b03 = [r["bout_F1_iou0.3"] for r in rows]
    b05 = [r["bout_F1_iou0.5"] for r in rows]
    ax.bar(x - width, f_vals, width, label="Per-frame F1", color=PINK_DARK,
           edgecolor="black", linewidth=0.5)
    ax.bar(x, b03, width, label="Per-bout F1 (IoU≥0.3)", color=PINK_MID,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width, b05, width, label="Per-bout F1 (IoU≥0.5)", color=PINK_LIGHT,
           edgecolor="black", linewidth=0.5)
    for i, (a, b, c) in enumerate(zip(f_vals, b03, b05)):
        ax.text(i - width, a + 0.012, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i,         b + 0.012, f"{b:.3f}", ha="center", fontsize=7)
        ax.text(i + width, c + 0.012, f"{c:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in rows], rotation=12, ha="right")
    ax.set_ylabel("F1 score (honest session-CV)")
    ax.set_ylim(0, 1.0); ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.set_title("Scratching — honest retrain comparison", fontweight="bold")

    out_png = PLOTS / "scratching_honest_compare.png"
    out_pdf = out_png.with_suffix(".pdf")
    out_svg = out_png.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    discord_upload(
        f"**Scratching — full honest pipeline (session-CV + SHAP pruning)**\n"
        f"All training done with **session-level GroupKFold** (each Rim session "
        f"held out once) + isotonic calibration + 4D post-proc sweep.\n\n"
        f"| Model | Frame F1 | Bout F1 (IoU 0.3) | Bout F1 (IoU 0.5) | CV mean |\n"
        f"|---|---|---|---|---|\n"
        f"| Deployed (cached) | 0.722 | 0.812 | 0.649 | 0.641 |\n"
        f"| Honest 600 features | {eval_full['frame_F1']:.3f} | "
        f"{eval_full['bout_F1_iou0.3']:.3f} | {eval_full['bout_F1_iou0.5']:.3f} | "
        f"{np.mean(fold_f1s_full):.3f} |\n"
        f"| Honest pruned 120 | **{eval_pruned['frame_F1']:.3f}** | "
        f"**{eval_pruned['bout_F1_iou0.3']:.3f}** | "
        f"**{eval_pruned['bout_F1_iou0.5']:.3f}** | "
        f"**{np.mean(fold_f1s_pruned):.3f}** |\n\n"
        f"Both new pkls saved to `{RUN_DIR.name}/` (NOT overwriting deployed). "
        f"Pruned model has full SHAP-derived feature subset + calibrated probs + "
        f"GroupKFold OOF. PNG + PDF + SVG + CSV.",
        [out_png, out_pdf, out_svg, csv_out],
    )


if __name__ == "__main__":
    main()
