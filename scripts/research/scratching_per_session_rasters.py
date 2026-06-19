"""
Per-session pink rasters for the Scratching classifier, using cached OOF
probabilities at the corrected threshold (thresh=0.50, mb=20).

Slices the 495,594-frame OOF vector into per-session chunks by matching
cumulative positives to the per-session positive counts read from each
session's Labels.csv.
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
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
from evaluation_tab import _apply_bout_filtering

PROJ = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected")
CLF  = PROJ / "Classifiers" / "PixelPaws_Scratching.pkl"
TRAIN = PROJ / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
LABELS_DIR = PROJ / "labels"
OUT_DIR = PROJ / "Classifiers" / "plots" / "per_session_rasters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = ("")
FPS = 60.0
PINK_LIGHT = "#F8BBD0"; PINK_DARK = "#C2185B"


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
                 "User-Agent": "PixelPaws-PerSessRaster/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def slice_oof_per_session(y_global, training_sessions):
    """Find session boundaries in the concatenated training set by
    matching cumulative positives to each session's Labels.csv positives.
    The exact negative-only boundary is undetermined (some negative
    frames were dropped during training); we place it halfway between
    the last positive of session k and the first positive of session
    k+1."""
    pos_per = {}
    for s in training_sessions:
        df = pd.read_csv(LABELS_DIR / f"{s}_Labels.csv")
        col = next((c for c in df.columns if c.lower() == "scratching"), None)
        pos_per[s] = int(df[col].sum()) if col else 0

    cumpos = np.cumsum(y_global)
    pos_idx = np.where(y_global == 1)[0]
    if pos_idx.sum() == 0:
        raise RuntimeError("no positives in OOF y")

    cum_target = 0
    bounds = [0]
    for s in training_sessions[:-1]:
        cum_target += pos_per[s]
        # last positive index for this session
        last_pos = pos_idx[cum_target - 1]
        # first positive of next session (>= cum_target)
        if cum_target < len(pos_idx):
            first_next = pos_idx[cum_target]
        else:
            first_next = len(y_global)
        b = (last_pos + first_next) // 2 + 1
        bounds.append(b)
    bounds.append(len(y_global))
    return bounds, pos_per


def render_session_raster(session, y, proba, thresh, mb, ma, mg,
                           out_png, out_pdf, out_svg):
    y_pred = (proba >= thresh).astype(int)
    if not (mb == 1 and ma == 0 and mg == 0):
        y_pred = _apply_bout_filtering(y_pred.copy(), mb, ma, mg)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    rs = cm.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    cm_n = cm / rs
    TN, FP, FN, TP = cm.ravel()
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0

    bin_frames = int(10 * FPS)
    n_bins = max(1, len(y) // bin_frames)
    human_s = np.zeros(n_bins); model_s = np.zeros(n_bins)
    for k in range(n_bins):
        sl = slice(k * bin_frames, (k + 1) * bin_frames)
        human_s[k] = y[sl].sum() / FPS
        model_s[k] = y_pred[sl].sum() / FPS
    r_val, _ = (pearsonr(human_s, model_s) if n_bins > 1 and
                model_s.std() > 0 and human_s.std() > 0
                else (float("nan"), None))

    def bouts(arr):
        padded = np.concatenate([[0], arr.astype(int), [0]])
        d = np.diff(padded)
        s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
        return list(zip(s.tolist(), (e - s).tolist()))

    fig = plt.figure(figsize=(16, 4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[5, 2, 4], wspace=0.35)
    ax_raster, ax_cm, ax_bins = gs.subplots()

    ax_raster.broken_barh(bouts(y),      (2.6, 0.8),
                          facecolors=PINK_LIGHT, edgecolors=PINK_LIGHT)
    ax_raster.broken_barh(bouts(y_pred), (1.2, 0.8),
                          facecolors=PINK_DARK, edgecolors=PINK_DARK)
    ax_raster.set_yticks([1.6, 3.0])
    ax_raster.set_yticklabels(["Model", "Human"], color="#777")
    ax_raster.set_ylim(0.8, 3.8)
    ax_raster.set_xlim(0, len(y))
    ax_raster.set_xlabel("Frame")
    ax_raster.set_title(
        f"{session}   n={len(y):,}  pos={int(y.sum()):,} "
        f"({y.sum()/len(y)*100:.2f}%)",
        fontsize=10, color="#444")
    for sp in ("top", "right", "left"): ax_raster.spines[sp].set_visible(False)
    ax_raster.tick_params(left=False, colors="#777")

    im = ax_cm.imshow(cm_n, cmap="RdPu", vmin=0, vmax=1)
    for row in range(2):
        for col in range(2):
            v = cm_n[row, col]
            ax_cm.text(col, row, f"{v:.3f}", ha="center", va="center",
                       fontsize=10,
                       color="white" if v > 0.5 else "black")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["0", "1"]); ax_cm.set_yticklabels(["0", "1"])
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    ax_cm.set_title(f"F1={f1:.3f}  P={P:.3f}  R={R:.3f}",
                    fontsize=10, color="#444")

    M = np.vstack([human_s, model_s])
    ax_bins.imshow(M, aspect="auto", cmap="RdPu",
                   vmin=0, vmax=max(M.max(), 1e-9),
                   extent=[0, n_bins * 10, 1.5, -0.5])
    ax_bins.set_yticks([0, 1])
    ax_bins.set_yticklabels(["Human", "Model"], color="#777")
    ax_bins.set_xlabel("Time (sec)")
    ax_bins.set_title(f"R = {r_val:.3f}" if not np.isnan(r_val) else "R = n/a",
                      fontsize=10, color="#444")
    for sp in ("top", "right", "left"): ax_bins.spines[sp].set_visible(False)
    ax_bins.tick_params(left=False, colors="#777")

    fig.savefig(out_png, format="png", dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return dict(f1=f1, P=P, R=R, r=r_val,
                n=len(y), pos=int(y.sum()),
                TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN))


def main():
    step("loading classifier + training set")
    cd = joblib.load(CLF)
    train = pickle.load(open(TRAIN, "rb"))
    y = train["y"]; oof = cd["oof_proba"]
    sessions = list(cd["training_sessions"])
    thr = float(cd["best_thresh"]); mb = int(cd["min_bout"])
    ma = int(cd["min_after_bout"]); mg = int(cd["max_gap"])
    step(f"  using corrected params: thresh={thr:.2f}  mb={mb}  ma={ma}  mg={mg}")

    bounds, pos_per = slice_oof_per_session(y, sessions)
    step(f"  per-session positives (from Labels.csv): {pos_per}")
    step(f"  inferred frame boundaries in OOF vector: {bounds}")

    summary = []
    files = []
    for k, sess in enumerate(sessions):
        lo, hi = bounds[k], bounds[k + 1]
        y_s     = y[lo:hi]
        proba_s = oof[lo:hi]
        # sanity: positives in this slice should match the label CSV
        slice_pos = int(y_s.sum())
        ok = "OK" if slice_pos == pos_per[sess] else f"mismatch (slice={slice_pos} vs label={pos_per[sess]})"
        step(f"  {sess}: frames={hi-lo:,}  positives={slice_pos:,}  [{ok}]")

        out_png = OUT_DIR / f"PixelPaws_Scratching_PinkRaster_{sess}.png"
        out_pdf = out_png.with_suffix(".pdf")
        out_svg = out_png.with_suffix(".svg")
        m = render_session_raster(sess, y_s, proba_s, thr, mb, ma, mg,
                                   out_png, out_pdf, out_svg)
        summary.append({"session": sess, **m})
        files += [out_png, out_pdf, out_svg]
        step(f"     F1={m['f1']:.3f}  P={m['P']:.3f}  R={m['R']:.3f}  R10s={m['r']:.3f}")

    sum_df = pd.DataFrame(summary)
    sum_csv = OUT_DIR / "PixelPaws_Scratching_PerSession_summary.csv"
    sum_df.to_csv(sum_csv, index=False)
    step(f"\nsummary CSV: {sum_csv}")
    print(sum_df.to_string(index=False))

    head = (
        f"**Per-session Scratching rasters — corrected params** "
        f"(thresh={thr:.2f}, mb={mb}, mg={mg})\n"
        f"OOF probabilities sliced per session by matching cumulative "
        f"positives to each session's Labels.csv. Three panels each: "
        f"raster (Human light pink / Model dark pink) | RdPu confusion "
        f"matrix with F1/P/R | 10-s-binned heatmap with Pearson R.\n\n"
        + sum_df.to_string(index=False)
    )
    # First message: all 5 PNGs (Discord limits attachments per post, so
    # send PNGs together, then PDFs+SVGs+CSV in a follow-up).
    pngs = [Path(f) for f in files if f.suffix == ".png"]
    discord_upload(head, pngs + [sum_csv])
    pdfs_svgs = [Path(f) for f in files if f.suffix in {".pdf", ".svg"}]
    if pdfs_svgs:
        discord_upload(
            "PDF/SVG companions (editable text, Illustrator-ready).",
            pdfs_svgs[:9])  # split to stay under Discord per-msg limit
        if len(pdfs_svgs) > 9:
            discord_upload("...(continued)", pdfs_svgs[9:])


if __name__ == "__main__":
    main()
