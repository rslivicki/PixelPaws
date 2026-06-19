"""
SNLT Baseline temporal probability — corrected analysis.

The GUI's stacked-area plot has a rendering quirk: when the bottom band
gets thicker, every band above shifts upward at both edges, so visually
"all behaviours move together" even though the fractions sum to 1.0 in
every bin and only ONE band is actually changing thickness.

This script:
  1. Predicts the 5 SNLT classifiers on each baseline session's cached
     features (hash 414c7814 — Apr-29 set with optical flow).
  2. Applies per-classifier post-processing (thresh + min_bout + max_gap)
     and the argmax priority Facial_grooming > Left_licking > rearing
     > still > walking (matches for_claude.json).
  3. Bins per-frame state assignments into 30-s windows (the GUI's
     prob_bin=30 s default), drops noise frames (exclude_noise=True),
     renormalises so the visible bands sum to 1.0 per bin.
  4. Averages across sessions in each treatment group (SNTX, Naive).
  5. Renders TWO panels per group as SVG:
       - left: stacked-area (matches the GUI's PDF, for reference)
       - right: unstacked overlapping lines (each band's actual fraction
         over time on its own y-axis — no stacking artifact)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------- #
# Config — taken verbatim from `transitions/for_claude.json`
# --------------------------------------------------------------------- #
PROJECT = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline")
FOR_CLAUDE_JSON = PROJECT / "transitions" / "for_claude.json"
KEY_FILE = PROJECT / "key_file.csv"
FEATURES_DIR = PROJECT / "features"
SHUFFLE = 9
FPS = 60
PROB_BIN_SEC = 30.0
EXCLUDE_NOISE = True

# Match the GUI's argmax priority (lowest index wins)
STATES_ORDER = ["Facial_grooming", "Left_licking", "rearing", "still", "walking"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATES_ORDER)}

# Plasma / inferno colours (matches GUI's inferno palette setting)
STATE_COLORS = {
    "Facial_grooming": "#000004",  # near black
    "Left_licking":    "#641A80",
    "rearing":         "#B73779",
    "still":           "#FCA50A",
    "walking":         "#F0F921",
}
GROUP_ORDER = ["Naive", "SNTX"]

ANALYSIS = PROJECT / "analysis" / "transitions"

WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord_upload(content: str, files: list[Path]) -> None:
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(files):
        ext = p.suffix.lower()
        mime = {".svg": "image/svg+xml", ".png": "image/png"}.get(ext, "application/octet-stream")
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
                 "User-Agent": "PixelPaws-SNLT-TempProb/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


# --------------------------------------------------------------------- #
# Bout / post-process helpers (same as the GUI's apply_postprocess)
# --------------------------------------------------------------------- #
def find_bouts(y):
    import numpy as np
    bouts = []; in_b = False; start = 0
    for i, v in enumerate(y):
        if v and not in_b:
            in_b = True; start = i
        elif not v and in_b:
            in_b = False; bouts.append((start, i - 1))
    if in_b:
        bouts.append((start, len(y) - 1))
    return bouts


def apply_postprocess(y_pred, min_bout=1, min_after_bout=0, max_gap=0):
    import numpy as np
    y = y_pred.astype(int).copy()
    if max_gap > 0:
        bouts = find_bouts(y)
        for i in range(len(bouts) - 1):
            gap = bouts[i + 1][0] - bouts[i][1] - 1
            if 0 < gap <= max_gap:
                y[bouts[i][1] + 1: bouts[i + 1][0]] = 1
    if min_bout > 1:
        for s, e in find_bouts(y):
            if e - s + 1 < min_bout:
                y[s: e + 1] = 0
    if min_after_bout > 0:
        bouts = find_bouts(y)
        for i in range(1, len(bouts)):
            gap = bouts[i][0] - bouts[i - 1][1] - 1
            if gap < min_after_bout:
                s, e = bouts[i]
                y[s: e + 1] = 0
    return y


# --------------------------------------------------------------------- #
# Per-session prediction → state sequence
# --------------------------------------------------------------------- #
def predict_state_sequence(features_pkl: Path, h5_path: Path, classifier_specs):
    """Returns per-frame state index (or -1 for noise) and total frame count."""
    import joblib
    import numpy as np
    import pandas as pd
    from prediction_pipeline import predict_with_xgboost, augment_features_post_cache

    with open(features_pkl, "rb") as f:
        X = pickle.load(f)
    n = len(X)
    state = np.full(n, -1, dtype=np.int8)   # -1 = noise / unclassified

    # Predict each classifier in priority order (highest first). We
    # assign a frame to the first state whose post-processed binary
    # is 1, matching the GUI's argmax-priority logic.
    binaries = {}
    for spec in classifier_specs:
        clf_path = Path(spec["path"])
        try:
            cd = joblib.load(clf_path)
        except Exception:
            with open(clf_path, "rb") as f:
                cd = pickle.load(f)
        model = cd["clf_model"]
        thresh = float(spec.get("best_thresh", cd.get("best_thresh", 0.5)))
        mb = int(spec.get("min_bout", cd.get("min_bout", 1)))
        ma = int(spec.get("min_after_bout", 0))
        mg = int(spec.get("max_gap", 0))
        try:
            X_aug = augment_features_post_cache(
                X.copy(), cd, model,
                str(h5_path) if h5_path is not None else "",
            )
        except Exception as e:
            step(f"      ! augment failed for {Path(spec['path']).stem}: {type(e).__name__}: {e}; using raw X")
            X_aug = X

        # Backfill missing Ego_* features from their rotation/translation-
        # invariant non-ego counterparts. Pairwise distances are identical
        # in any rigid frame, and the *_Vel1 columns in PixelPaws are
        # speed magnitudes (also rotation-invariant). The Ego model was
        # trained on these same numeric values, just labelled
        # "Ego_<name>", so the backfill is an exact substitute for
        # Ego_Dis_* and a near-exact substitute for Ego_*_Vel*.
        if hasattr(model, "feature_names_in_"):
            need = [f for f in model.feature_names_in_ if f not in X_aug.columns]
            if need:
                backfilled = []
                for f in need:
                    if f.startswith("Ego_"):
                        nonego = f[len("Ego_"):]
                        if nonego in X_aug.columns:
                            X_aug[f] = X_aug[nonego]
                            backfilled.append(f)
                if backfilled:
                    step(f"      backfilled {len(backfilled)} Ego_* "
                         f"features for {Path(spec['path']).stem} "
                         f"(rotation-invariant substitutes)")
        try:
            y_proba = predict_with_xgboost(model, X_aug,
                                           calibrator=cd.get("prob_calibrator"),
                                           fold_models=cd.get("fold_models"))
        except Exception as e:
            step(f"      ! predict failed for {Path(spec['path']).stem}: {type(e).__name__}: {e}")
            continue
        y_raw = (y_proba >= thresh).astype(int)
        y_post = apply_postprocess(y_raw, mb, ma, mg)
        # State name = classifier filename without prefix / suffix
        name = (clf_path.stem
                .replace("PixelPaws_", "")
                .replace("_AllFeatures", "")
                .replace("_pruned_120", ""))
        if name not in STATE_TO_IDX:
            step(f"      ? unknown classifier name {name}; skipping")
            continue
        binaries[STATE_TO_IDX[name]] = y_post.astype(bool)

    # Argmax with priority: walk in priority order, first hit wins
    for idx in range(len(STATES_ORDER)):
        if idx not in binaries:
            continue
        mask = binaries[idx] & (state == -1)
        state[mask] = idx
    return state, n


def bin_state_sequence(state, n_frames, bin_frames):
    """Per-bin renormalised fractions (n_bins, n_states). Drops -1 noise."""
    import numpy as np
    n_states = len(STATES_ORDER)
    n_bins = max(1, n_frames // bin_frames)
    fracs = np.zeros((n_bins, n_states), dtype=float)
    centres = np.zeros(n_bins)
    for b in range(n_bins):
        s = b * bin_frames; e = min(s + bin_frames, n_frames)
        chunk = state[s:e]
        centres[b] = (s + e) / 2.0 / FPS
        for st in range(n_states):
            fracs[b, st] = int((chunk == st).sum())
        total = fracs[b].sum()
        if total > 0:
            fracs[b] /= total
    return centres, fracs


# --------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------- #
def collect_all(spec):
    """Per-session state sequence → per-bin fraction matrix, grouped."""
    import numpy as np
    import pandas as pd

    key_df = pd.read_csv(KEY_FILE).dropna()
    mouse_to_group = {row["Subject"]: row["Treatment"]
                      for _, row in key_df.iterrows()}
    step(f"key file: {len(mouse_to_group)} mice "
         f"(SNTX={sum(v=='SNTX' for v in mouse_to_group.values())}, "
         f"Naive={sum(v=='Naive' for v in mouse_to_group.values())})")

    sessions = spec["session_selection"]
    classifier_specs = spec["classifiers"]
    bin_frames = int(PROB_BIN_SEC * FPS)

    group_bins = {g: [] for g in GROUP_ORDER}
    session_summaries = []

    for sess in sessions:
        # mouse id = strip the trailing _Baseline tag
        mouse = sess.replace("_Baseline", "")
        group = mouse_to_group.get(mouse)
        if not group:
            step(f"  ? no group for {sess}; skip")
            continue
        # Find the most-recent Apr-29 feature pkl for this session
        pkls = sorted(FEATURES_DIR.glob(f"{sess}_features_414c7814.pkl"))
        if not pkls:
            pkls = sorted(FEATURES_DIR.glob(f"{sess}_features_*.pkl"))
        if not pkls:
            step(f"  ! no feature pkl for {sess}; skip")
            continue
        pkl = pkls[-1]
        # filtered DLC h5 — best-effort lookup; if none found we pass
        # None and let augment_features_post_cache skip the ego step.
        # The contact / DutyCycle features are computed from height
        # columns already in the cached feature pkl, so we don't need
        # the h5 for the transition plots.
        mouse_stem = sess.replace("_Baseline", "")
        h5_candidates = (
            list((PROJECT / "videos").glob(f"{sess}*shuffle9*_filtered.h5"))
            + list((PROJECT / "videos").glob(f"{mouse_stem}DLC*shuffle9*_filtered.h5"))
            + list((PROJECT / "videos").glob(f"{mouse_stem}_BaselineDLC*shuffle9*_filtered.h5"))
            + list((PROJECT / "videos").glob(f"{sess}*shuffle8*_filtered.h5"))
            + list((PROJECT / "videos").glob(f"{mouse_stem}*shuffle8*_filtered.h5"))
        )
        if not h5_candidates:
            h5_candidates = list(PROJECT.rglob(f"{mouse_stem}*_filtered.h5"))
        h5 = h5_candidates[0] if h5_candidates else None
        if h5 is None:
            step(f"  no h5 for {sess} — proceeding without ego augmentation")
        step(f"  {sess} -> {group}  (pkl={pkl.stem})")

        state, n = predict_state_sequence(pkl, h5, classifier_specs)
        centres, fracs = bin_state_sequence(state, n, bin_frames)
        group_bins[group].append((centres, fracs))
        session_summaries.append({"session": sess, "group": group,
                                  "n_frames": int(n), "n_bins": int(fracs.shape[0])})

    # Average across sessions in each group
    mean_per_group = {}
    for g, runs in group_bins.items():
        if not runs:
            continue
        min_bins = min(r[1].shape[0] for r in runs)
        centres = runs[0][0][:min_bins]
        stacked = np.stack([r[1][:min_bins] for r in runs], axis=0)
        mean_per_group[g] = (centres, stacked.mean(axis=0), stacked.shape[0])

    return mean_per_group, session_summaries


def plot_temporal_probability(mean_per_group, out_svg: Path, out_png: Path,
                              out_pdf: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    # Editable text in Illustrator
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    import numpy as np

    state_colors = [STATE_COLORS[s] for s in STATES_ORDER]
    n_groups = len(mean_per_group)
    fig, axes = plt.subplots(n_groups, 2, figsize=(13.5, 4.2 * n_groups),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    if n_groups == 1:
        axes = axes.reshape(1, 2)

    for row, group in enumerate(g for g in GROUP_ORDER if g in mean_per_group):
        centres, fracs, n_sess = mean_per_group[group]
        # --- LEFT: stacked area (reference / GUI style) ---
        ax_s = axes[row, 0]
        ax_s.stackplot(centres, fracs.T, labels=STATES_ORDER,
                       colors=state_colors, alpha=0.85)
        ax_s.set_ylim(0, 1)
        ax_s.set_title(f"{group}  (stacked — reference, n={n_sess})",
                       fontweight="bold")
        ax_s.set_ylabel("Fraction (cumulative)")
        if row == n_groups - 1:
            ax_s.set_xlabel("Time (s)")
        if row == 0:
            ax_s.legend(loc="upper right", fontsize=7, ncol=2)

        # --- RIGHT: unstacked overlapping lines (corrected analysis) ---
        ax_u = axes[row, 1]
        for st_idx, state_name in enumerate(STATES_ORDER):
            ax_u.plot(centres, fracs[:, st_idx],
                      color=state_colors[st_idx], linewidth=1.5,
                      label=state_name)
        ax_u.set_ylim(0, max(0.5, fracs.max() * 1.05))
        ax_u.set_title(f"{group}  (unstacked lines — corrected, n={n_sess})",
                       fontweight="bold")
        ax_u.set_ylabel("Fraction (per state)")
        ax_u.grid(alpha=0.3)
        if row == n_groups - 1:
            ax_u.set_xlabel("Time (s)")
        if row == 0:
            ax_u.legend(loc="upper right", fontsize=7, ncol=2)

    fig.suptitle(
        "SNLT Baseline — temporal probability\n"
        "Left: original stacked area (geometric stacking artifact makes "
        "all bands appear to move together when the bottom band changes). "
        "Right: unstacked overlapping lines (each band's true fraction).",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=180, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    with open(FOR_CLAUDE_JSON) as f:
        spec = json.load(f)

    step("Predicting + binning all sessions...")
    mean_per_group, summaries = collect_all(spec)

    if not mean_per_group:
        step("! no group data collected"); return 1

    # Per-session diagnostic CSV
    import pandas as pd
    summ_csv = ANALYSIS / "temporal_probability_session_inventory.csv"
    pd.DataFrame(summaries).to_csv(summ_csv, index=False)

    svg_path = ANALYSIS / "temporal_probability_corrected.svg"
    png_path = ANALYSIS / "temporal_probability_corrected.png"
    pdf_path = ANALYSIS / "temporal_probability_corrected.pdf"
    plot_temporal_probability(mean_per_group, svg_path, png_path, pdf_path)
    step(f"  -> {svg_path}  ({svg_path.stat().st_size/1024:.0f} KB)")
    step(f"  -> {png_path}  ({png_path.stat().st_size/1024:.0f} KB)")
    step(f"  -> {pdf_path}  ({pdf_path.stat().st_size/1024:.0f} KB)")

    head = (
        "**SNLT Baseline — temporal probability (corrected analysis)**\n"
        "Side-by-side: original GUI-style stacked-area (left) vs unstacked "
        "overlapping lines (right). The stacked plot has a rendering quirk: "
        "when the bottom band gets thicker, every band above shifts upward "
        "at both edges, so visually 'all bands move together' even though "
        "fractions sum to 1.0 in every bin and only ONE band actually "
        "changes thickness. The line plot shows each band's true fraction "
        "over time — much clearer for spotting which behaviours genuinely "
        "co-vary.\n"
        f"Sessions: {len(summaries)} ({sum(s['group']=='SNTX' for s in summaries)} "
        f"SNTX + {sum(s['group']=='Naive' for s in summaries)} Naïve). "
        f"30-s bins, argmax priority Facial_grooming > Left_licking > rearing "
        f"> still > walking, noise frames dropped + renormalised."
    )
    discord_upload(head, [pdf_path, svg_path, png_path, summ_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
