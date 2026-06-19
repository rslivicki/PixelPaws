"""Combined ethogram + confusion matrix + time-bin figure for
Scratching (purple) and Licking (teal), styled after the example
multi-behavior reference figure.

Two rows. Each row: 3 panels.
  Row 1 (Scratching, purple): raster | confusion matrix | 10-s bin heatmap
  Row 2 (Licking,    teal):   raster | confusion matrix | 10-s bin heatmap

For each behavior we use the held-out OOF probabilities from the
honest CV reproduction so the F1/CM/R values reflect the deployed
classifier's true generalization behavior.
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
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))
from evaluation_tab import _apply_bout_filtering

# ---------- paths ----------
SCRATCH_PROJ  = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected")
SCRATCH_CLF   = SCRATCH_PROJ / "Classifiers" / "PixelPaws_Scratching.pkl"
SCRATCH_TRAIN = SCRATCH_PROJ / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
SCRATCH_LBLS  = SCRATCH_PROJ / "labels"

LICK_TRAIN = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/PixelPaws_Classifiers/HomeRig/Left_licking_train_set.pkl")
LICK_N_S1, LICK_N_S2 = 49855, 21153   # verified session boundaries

OUT_DIR = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/PixelPaws_Classifiers/HomeRig/scratch_lick_combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60.0
SEED = 42
BIN_SEC = 5  # 5-s time bins

WEBHOOK = ("")

# ---------- palettes ----------
# Scratching = green gradient, Licking = blue gradient.
SCRATCH_CMAP = LinearSegmentedColormap.from_list(
    "scratch_green",
    ["#ffffff", "#d9efd2", "#9bd49a", "#4ca85f", "#1b5e2a", "#0a3514"])
SCRATCH_LIGHT = "#9bd49a"
SCRATCH_DARK  = "#1b5e2a"

LICK_CMAP = LinearSegmentedColormap.from_list(
    "lick_blue",
    ["#ffffff", "#d6e6f4", "#9bc4e8", "#4a8ec7", "#1f4e8a", "#0a2540"])
LICK_LIGHT = "#9bc4e8"
LICK_DARK  = "#1f4e8a"


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def densest_window(y, win_frames):
    """Return (start, stop) of the win_frames-length window with the
    most positives. If the session is shorter than win_frames, return
    the whole session."""
    if len(y) <= win_frames:
        return 0, len(y)
    # Cumulative-sum trick: positives in window [i, i+win) = c[i+win] - c[i]
    c = np.concatenate([[0], np.cumsum(y)])
    counts = c[win_frames:] - c[:-win_frames]
    i = int(np.argmax(counts))
    return i, i + win_frames


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
                ".svg": "image/svg+xml", ".csv": "text/csv"
                }.get(p.suffix.lower(), "application/octet-stream")
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
                 "User-Agent": "PixelPaws-CombinedScratchLick/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


# ---------- session slicing ----------
def slice_scratching_oof():
    """Return (session_name, y, oof, thresh, mb) for the chosen scratching session."""
    cd = joblib.load(SCRATCH_CLF)
    train = pickle.load(open(SCRATCH_TRAIN, "rb"))
    y_all = train["y"]; oof_all = cd["oof_proba"]
    sessions = list(cd["training_sessions"])
    thr = float(cd["best_thresh"]); mb = int(cd["min_bout"])
    ma = int(cd["min_after_bout"]); mg = int(cd["max_gap"])

    pos_per = {}
    for s in sessions:
        df = pd.read_csv(SCRATCH_LBLS / f"{s}_Labels.csv")
        col = next((c for c in df.columns if c.lower() == "scratching"), None)
        pos_per[s] = int(df[col].sum()) if col else 0

    pos_idx = np.where(y_all == 1)[0]
    cum_target = 0
    bounds = [0]
    for s in sessions[:-1]:
        cum_target += pos_per[s]
        last_pos = pos_idx[cum_target - 1]
        first_next = pos_idx[cum_target] if cum_target < len(pos_idx) else len(y_all)
        bounds.append((last_pos + first_next) // 2 + 1)
    bounds.append(len(y_all))

    # Pick the session with the most positives so the raster is densely
    # populated (good for the figure)
    best_idx = int(np.argmax([pos_per[s] for s in sessions]))
    sess = sessions[best_idx]
    lo, hi = bounds[best_idx], bounds[best_idx + 1]
    step(f"  scratching: picked {sess} (frames={hi-lo:,}, pos={int(y_all[lo:hi].sum()):,})")
    return sess, y_all[lo:hi], oof_all[lo:hi], thr, mb, ma, mg


def compute_licking_oof():
    """Reproduce honest 3-fold GroupKFold OOF for licking, return (sess, y, oof)."""
    step("  licking: loading BAREfoot train set + reproducing OOF")
    train = joblib.load(LICK_TRAIN)
    X, y = train["X"], np.asarray(train["y"], dtype=int)
    groups = np.zeros(len(X), dtype=int)
    groups[:LICK_N_S1] = 0
    groups[LICK_N_S1:LICK_N_S1 + LICK_N_S2] = 1
    groups[LICK_N_S1 + LICK_N_S2:] = 2

    gkf = GroupKFold(n_splits=3)
    oof = np.zeros(len(y), dtype=np.float32)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        t0 = time.time()
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw,
            tree_method="hist", random_state=SEED, n_jobs=-1,
            verbosity=0, use_label_encoder=False, eval_metric="logloss",
        )
        clf.fit(X.iloc[tr], y[tr])
        oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
        step(f"    fold {fold}/3 (test=S{int(groups[te][0])+1}) done in {time.time()-t0:.1f}s")

    # Pick S1 — longest session (~13.8 min) so the full raster shows the
    # most data; user has previously seen S2 and S3, so cycle to S1.
    pick = 0  # 0=S1, 1=S2, 2=S3
    pos_per_sess = [int(y[groups == g].sum()) for g in range(3)]
    sess = f"S{pick+1}"
    sl = (groups == pick)
    step(f"  licking: picked {sess} (frames={int(sl.sum()):,}, pos={pos_per_sess[pick]:,})")
    return sess, y[sl], oof[sl]


# ---------- one row (raster | CM | bins) ----------
def bouts(arr):
    padded = np.concatenate([[0], arr.astype(int), [0]])
    d = np.diff(padded)
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
    return list(zip(s.tolist(), (e - s).tolist()))


def render_row(axes, session, y, proba, thresh, mb, ma, mg,
               light, dark, cmap, label_long):
    ax_raster, ax_cm, ax_bins = axes

    # Full session — no windowing.
    y_pred = (proba >= thresh).astype(int)
    if not (mb == 1 and ma == 0 and mg == 0):
        y_pred = _apply_bout_filtering(y_pred.copy(), mb, ma, mg)

    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    rs = cm.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    cm_n = cm / rs

    # 5-s bins over the full session
    bin_frames = int(BIN_SEC * FPS)
    n_bins = max(1, len(y) // bin_frames)
    human_s = np.zeros(n_bins); model_s = np.zeros(n_bins)
    for k in range(n_bins):
        sl = slice(k * bin_frames, (k + 1) * bin_frames)
        human_s[k] = y[sl].sum() / FPS
        model_s[k] = y_pred[sl].sum() / FPS
    r_val = float("nan")
    if n_bins > 1 and human_s.std() > 0 and model_s.std() > 0:
        r_val, _ = pearsonr(human_s, model_s)

    # ----- raster -----
    ax_raster.broken_barh(bouts(y),      (2.6, 0.8), facecolors=light, edgecolors=light)
    ax_raster.broken_barh(bouts(y_pred), (1.2, 0.8), facecolors=dark,  edgecolors=dark)
    ax_raster.set_yticks([1.6, 3.0])
    ax_raster.set_yticklabels(["Model", "Human"], color="#666")
    ax_raster.set_ylim(0.8, 3.8)
    ax_raster.set_xlim(0, len(y))
    ax_raster.set_xlabel("Frame")
    ax_raster.set_ylabel(label_long, fontsize=10, color="#222", fontweight="bold")
    for sp in ("top", "right", "left"):
        ax_raster.spines[sp].set_visible(False)
    ax_raster.tick_params(left=False, colors="#666")

    # ----- confusion matrix (aspect="auto" lets it fill the row height) -----
    im = ax_cm.imshow(cm_n, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for row in range(2):
        for col in range(2):
            v = cm_n[row, col]
            ax_cm.text(col, row, f"{v:.3f}", ha="center", va="center",
                       fontsize=10, color="white" if v > 0.5 else "black")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["0", "1"]); ax_cm.set_yticklabels(["0", "1"])
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    ax_cm.set_title(f"F1-Score: {f1:.3f}", fontsize=10, color="#333", pad=4)

    # ----- 5-s bin heatmap — Human (top) and Model (bottom) rendered as
    # two separate strips with a visible gap so they read like the two
    # raster rows on the left -----
    H = human_s[np.newaxis, :]
    M = model_s[np.newaxis, :]
    vmax = max(human_s.max(), model_s.max(), 1e-9)
    T = n_bins * BIN_SEC
    # y-extents: Human strip in [0, 0.45], gap, Model strip in [0.55, 1.0]
    im_h = ax_bins.imshow(H, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                          extent=[0, T, 1.0, 0.55])
    im_m = ax_bins.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                          extent=[0, T, 0.45, 0.0])
    ax_bins.set_ylim(0.0, 1.0)
    ax_bins.set_yticks([0.225, 0.775])
    ax_bins.set_yticklabels(["Model", "Human"], color="#666")
    ax_bins.set_xlim(0, T)
    ax_bins.set_xlabel("Time (sec)")
    ax_bins.text(0.99, 1.04, f"R = {r_val:.2f}" if not np.isnan(r_val) else "R = n/a",
                 transform=ax_bins.transAxes, ha="right", va="bottom",
                 fontsize=10, color="#333")
    for sp in ("top", "right", "left"):
        ax_bins.spines[sp].set_visible(False)
    ax_bins.tick_params(left=False, colors="#666")
    cbar2 = plt.colorbar(im_h, ax=ax_bins, fraction=0.035, pad=0.015,
                         shrink=0.85)
    cbar2.outline.set_visible(False)
    cbar2.ax.tick_params(labelsize=7)
    cbar2.set_label(f"sec active per {BIN_SEC}-s bin", fontsize=8, color="#333")

    return dict(f1=f1, R=r_val, n=len(y), pos=int(y.sum()))


def main():
    step("=== scratching ===")
    s_sess, s_y, s_oof, s_thr, s_mb, s_ma, s_mg = slice_scratching_oof()
    step(f"  scratching params: thresh={s_thr:.2f}, mb={s_mb}, ma={s_ma}, mg={s_mg}")

    step("=== licking ===")
    l_sess, l_y, l_oof = compute_licking_oof()
    # Use the deployed pkl threshold = 0.5 (matches reported CV F1)
    l_thr, l_mb, l_ma, l_mg = 0.50, 1, 0, 0

    # ----- figure -----
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[6, 1.6, 3.4], height_ratios=[1, 1],
                          wspace=0.28, hspace=0.55,
                          left=0.05, right=0.985, top=0.94, bottom=0.10)
    row1 = [fig.add_subplot(gs[0, j]) for j in range(3)]
    row2 = [fig.add_subplot(gs[1, j]) for j in range(3)]

    m1 = render_row(row1, s_sess, s_y, s_oof, s_thr, s_mb, s_ma, s_mg,
                    SCRATCH_LIGHT, SCRATCH_DARK, SCRATCH_CMAP, "Scratching")
    m2 = render_row(row2, l_sess, l_y, l_oof, l_thr, l_mb, l_ma, l_mg,
                    LICK_LIGHT, LICK_DARK, LICK_CMAP, "Licking")

    out_base = OUT_DIR / "scratching_licking_ethogram_figure"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    step(f"wrote {out_base}.png/pdf/svg")

    head = (
        "**Scratching + Licking — combined ethogram / confusion / time-bin figure**\n"
        f"Row 1 = Scratching (purple), session **{s_sess}**: "
        f"F1 = **{m1['f1']:.3f}**, R(10-s bins) = **{m1['R']:.2f}** "
        f"(n={m1['n']:,} frames, pos={m1['pos']:,}). "
        f"Cached OOF, thresh={s_thr:.2f}, mb={s_mb}, mg={s_mg}.\n"
        f"Row 2 = Left_licking (teal), session **{l_sess}**: "
        f"F1 = **{m2['f1']:.3f}**, R(10-s bins) = **{m2['R']:.2f}** "
        f"(n={m2['n']:,} frames, pos={m2['pos']:,}). "
        "Honest 3-fold session-level GroupKFold OOF, thresh=0.50."
    )
    discord_upload(head, [
        out_base.with_suffix(".png"),
        out_base.with_suffix(".pdf"),
        out_base.with_suffix(".svg"),
    ])


if __name__ == "__main__":
    main()
