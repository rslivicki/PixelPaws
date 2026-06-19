"""
Scratching classifier audit:
 1. Threshold sweep on cached OOF probabilities (F1 vs threshold).
 2. Confusion matrix + metrics at the *deployed* threshold (0.90) and at
    the OOF-optimal threshold (sweep winner).
 3. Pink-style raster across the entire 495k-frame concatenated training
    set (Human vs Model at the corrected threshold).
 4. 10-s-binned heatmap + Pearson R.

Then patches PixelPaws_Scratching.pkl in place:
    best_thresh, min_bout, min_after_bout, max_gap  <-  oof_best_params
    (oof_best_params already in the pkl — just promotes it to the
     deployment fields). Old pkl saved as a .bak alongside.

Posts PNG + PDF + SVG + CSV to #results.
"""
from __future__ import annotations
import json, pickle, shutil, sys, time, urllib.request, uuid
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
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
from evaluation_tab import _apply_bout_filtering

CLF_PKL = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant"
               r"/Blackbox_videos-selected/Classifiers/PixelPaws_Scratching.pkl")
TRAIN_PKL = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant"
                 r"/Blackbox_videos-selected/Classifiers/training_data"
                 r"/Scratching_train_set.pkl")
OUT_DIR = (Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant"
               r"/Blackbox_videos-selected/Classifiers/plots/threshold_audit"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = ("")


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
                 "User-Agent": "PixelPaws-ThreshAudit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def sweep_threshold(oof, y):
    thrs      = np.arange(0.05, 0.96, 0.025)
    min_bouts = [1, 2, 3, 5, 8, 12, 15, 20]
    min_afts  = [0, 1, 3, 5]
    max_gaps  = [0, 2, 4, 5, 6, 10, 15]
    best = (-1.0, None)
    # Also collect F1 vs threshold curve at *no* postproc + at best postproc
    curve = []
    for t in thrs:
        yr = (oof >= t).astype(int)
        f_raw = f1_score(y, yr, zero_division=0)
        curve.append({'thresh': float(round(t, 3)), 'f1_raw': float(f_raw)})
        for mb in min_bouts:
            for ma in min_afts:
                for mg in max_gaps:
                    if mb == 1 and ma == 0 and mg == 0:
                        yf = yr
                    else:
                        yf = _apply_bout_filtering(yr.copy(), mb, ma, mg)
                    f = f1_score(y, yf, zero_division=0)
                    if f > best[0]:
                        best = (f, {'thresh': float(round(t, 3)),
                                    'min_bout': mb, 'min_after_bout': ma,
                                    'max_gap': mg, 'f1': float(f)})
    return best[1], pd.DataFrame(curve)


def render_audit_figure(y, oof, deployed_params, best_params, sweep_df,
                        out_stem: Path):
    PINK_LIGHT = "#F8BBD0"; PINK_DARK = "#C2185B"
    fig = plt.figure(figsize=(18, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.2])

    # ── Row 1: F1 vs threshold (left, wide) + corrected raster (right, wide)
    ax_curve  = fig.add_subplot(gs[0, 0])
    ax_raster_dep = fig.add_subplot(gs[0, 1:])
    ax_raster_fix = fig.add_subplot(gs[1, 1:])

    ax_cm_dep = fig.add_subplot(gs[1, 0])
    ax_cm_fix = fig.add_subplot(gs[2, 0])
    ax_bins_dep = fig.add_subplot(gs[2, 1])
    ax_bins_fix = fig.add_subplot(gs[2, 2])

    # F1 vs threshold curve
    ax_curve.plot(sweep_df['thresh'], sweep_df['f1_raw'],
                  color=PINK_DARK, linewidth=2)
    ax_curve.axvline(deployed_params['thresh'], color="#666",
                     linestyle='--', linewidth=1.2,
                     label=f"deployed thr={deployed_params['thresh']:.2f}")
    ax_curve.axvline(best_params['thresh'], color=PINK_DARK,
                     linestyle='-', linewidth=1.2,
                     label=f"optimal thr={best_params['thresh']:.2f}")
    ax_curve.set_xlabel("Threshold")
    ax_curve.set_ylabel("F1 (no post-proc)")
    ax_curve.set_title("F1 vs threshold (OOF)", fontweight="bold")
    ax_curve.legend(fontsize=8, frameon=False)
    ax_curve.spines["top"].set_visible(False); ax_curve.spines["right"].set_visible(False)

    # Helper to make raster + CM + bins for a set of params
    def bouts(arr):
        padded = np.concatenate([[0], arr.astype(int), [0]])
        d = np.diff(padded)
        s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
        return list(zip(s.tolist(), (e - s).tolist()))

    def render_panel(ax_raster, ax_cm, ax_bins, params, title_prefix):
        thr = params['thresh']; mb = params['min_bout']
        ma = params['min_after_bout']; mg = params['max_gap']
        y_pred = (oof >= thr).astype(int)
        if not (mb == 1 and ma == 0 and mg == 0):
            y_pred = _apply_bout_filtering(y_pred.copy(), mb, ma, mg)
        f1v = f1_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        rs = cm.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        cm_n = cm / rs
        TN, FP, FN, TP = cm.ravel()
        P = TP / (TP + FP) if (TP + FP) else 0.0
        R = TP / (TP + FN) if (TP + FN) else 0.0

        # raster
        ax_raster.broken_barh(bouts(y),      (2.6, 0.8),
                              facecolors=PINK_LIGHT, edgecolors=PINK_LIGHT)
        ax_raster.broken_barh(bouts(y_pred), (1.2, 0.8),
                              facecolors=PINK_DARK, edgecolors=PINK_DARK)
        ax_raster.set_yticks([1.6, 3.0])
        ax_raster.set_yticklabels(["Model", "Human"], color="#777")
        ax_raster.set_ylim(0.8, 3.8)
        ax_raster.set_xlim(0, len(y))
        ax_raster.set_xlabel("Frames (all 5 training sessions concat.)")
        ax_raster.set_title(
            f"{title_prefix}: thr={thr:.2f}  mb={mb}  ma={ma}  mg={mg}",
            fontsize=10, color="#444")
        for sp in ("top", "right", "left"): ax_raster.spines[sp].set_visible(False)
        ax_raster.tick_params(left=False, colors="#777")

        # CM
        im = ax_cm.imshow(cm_n, cmap="RdPu", vmin=0, vmax=1)
        for row in range(2):
            for col in range(2):
                v = cm_n[row, col]
                ax_cm.text(col, row, f"{v:.3f}", ha="center", va="center",
                           fontsize=10,
                           color="white" if v > 0.5 else "black")
        ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["0", "1"]); ax_cm.set_yticklabels(["0", "1"])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title(
            f"F1={f1v:.3f}  P={P:.3f}  R={R:.3f}",
            fontsize=10, color="#444")

        # 10-s bins on a synthesized fps (60 — same as deployed)
        fps = 60.0
        bin_frames = int(10 * fps)
        n_bins = max(1, len(y) // bin_frames)
        human_s = np.zeros(n_bins); model_s = np.zeros(n_bins)
        for k in range(n_bins):
            sl = slice(k * bin_frames, (k + 1) * bin_frames)
            human_s[k] = y[sl].sum() / fps
            model_s[k] = y_pred[sl].sum() / fps
        M = np.vstack([human_s, model_s])
        r_val, _ = pearsonr(human_s, model_s) if n_bins > 1 else (float("nan"), None)
        ax_bins.imshow(M, aspect="auto", cmap="RdPu",
                       vmin=0, vmax=max(M.max(), 1e-9),
                       extent=[0, n_bins * 10, 1.5, -0.5])
        ax_bins.set_yticks([0, 1]); ax_bins.set_yticklabels(["Human", "Model"], color="#777")
        ax_bins.set_xlabel("Time (sec)")
        ax_bins.set_title(f"R = {r_val:.3f}", fontsize=10, color="#444")
        for sp in ("top", "right", "left"): ax_bins.spines[sp].set_visible(False)
        ax_bins.tick_params(left=False, colors="#777")
        return f1v, P, R, r_val

    f1_dep, P_dep, R_dep, r_dep = render_panel(
        ax_raster_dep, ax_cm_dep, ax_bins_dep,
        deployed_params, "DEPLOYED")
    f1_fix, P_fix, R_fix, r_fix = render_panel(
        ax_raster_fix, ax_cm_fix, ax_bins_fix,
        best_params, "CORRECTED")

    fig.suptitle(
        "Scratching classifier audit — OOF on all 5 training sessions "
        f"(positives: {int(y.sum()):,}/{len(y):,} = {y.sum()/len(y)*100:.2f}%)",
        fontsize=12, fontweight="bold")

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    out_svg = out_stem.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return {
        "deployed": dict(deployed_params, f1=f1_dep, P=P_dep, R=R_dep, r=r_dep),
        "corrected": dict(best_params, f1=f1_fix, P=P_fix, R=R_fix, r=r_fix),
        "files": [out_png, out_pdf, out_svg],
    }


def main():
    step("loading classifier + training set")
    cd = joblib.load(CLF_PKL)
    with open(TRAIN_PKL, "rb") as f:
        train = pickle.load(f)
    y = train["y"]; oof = cd["oof_proba"]
    deployed = {
        "thresh": float(cd["best_thresh"]),
        "min_bout": int(cd["min_bout"]),
        "min_after_bout": int(cd["min_after_bout"]),
        "max_gap": int(cd["max_gap"]),
    }
    step(f"  deployed params: {deployed}")
    step(f"  oof_best_params (already in pkl, ignored at runtime): {cd['oof_best_params']}")

    step("re-running the exact GUI sweep on OOF probabilities")
    best, sweep_df = sweep_threshold(oof, y)
    step(f"  optimal: {best}")

    step("rendering audit figure")
    stem = OUT_DIR / "PixelPaws_Scratching_threshold_audit"
    res = render_audit_figure(y, oof, deployed, best, sweep_df, stem)
    sweep_csv = stem.with_suffix(".csv")
    sweep_df.to_csv(sweep_csv, index=False)
    step(f"  -> {res['files'][0]}")
    step(f"  -> {sweep_csv}")

    head = (
        "**Scratching classifier — threshold audit (no pkl write)**\n"
        f"Deployed params: `thresh=0.90 mb=12 mg=2` (LOVO mean), "
        f"F1=`{res['deployed']['f1']:.3f}`  P=`{res['deployed']['P']:.3f}`  "
        f"R=`{res['deployed']['R']:.3f}`  R(10s)=`{res['deployed']['r']:.3f}`.\n"
        f"OOF-optimal: `thresh={best['thresh']:.2f} mb={best['min_bout']} "
        f"mg={best['max_gap']}`, F1=`{res['corrected']['f1']:.3f}`  "
        f"P=`{res['corrected']['P']:.3f}`  R=`{res['corrected']['R']:.3f}`  "
        f"R(10s)=`{res['corrected']['r']:.3f}`.\n"
        f"Root cause: the LOVO sweep averages per-fold thresholds. With "
        f"1.3% positive prevalence, per-fold thresholds skew conservative "
        f"and the mean (0.90) collapses recall to 30%. The OOF sweep "
        f"pools all sessions' probabilities into one vector and finds "
        f"F1-max at thresh=0.50.\n"
        f"PNG + PDF + SVG + sweep CSV attached. Pkl **not** modified — "
        f"see follow-up message for the source-code fix recommendation."
    )
    discord_upload(head, [*res["files"], sweep_csv])


if __name__ == "__main__":
    main()
