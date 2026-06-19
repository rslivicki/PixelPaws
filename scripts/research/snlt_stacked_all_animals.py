"""
SNLT Baseline temporal-probability stacked-area plot for ALL animals.

Renders just the stacked-area panels (matches the previous screenshot
format): Naïve on top, SNTX on bottom, both averaged across every
animal in each treatment group. 30-s bins, argmax priority
Facial_grooming > Left_licking > rearing > still > walking.
"""
from __future__ import annotations
import json, pickle, sys, time, urllib.request, uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the prediction pipeline from the corrected script
from snlt_temporal_probability_corrected import (
    predict_state_sequence, bin_state_sequence,
    STATES_ORDER, STATE_COLORS,
    PROJECT, FEATURES_DIR, KEY_FILE, FOR_CLAUDE_JSON,
    PROB_BIN_SEC, FPS,
)

GROUP_ORDER = ["Naive", "SNTX"]
ANALYSIS = PROJECT / "analysis" / "transitions"
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
                 "User-Agent": "PixelPaws-SNLT-AllAnimals/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def main():
    step("loading session selection + key")
    spec = json.load(open(FOR_CLAUDE_JSON))
    sessions = spec["session_selection"]
    classifier_specs = spec["classifiers"]

    key_df = pd.read_csv(KEY_FILE).dropna()
    mouse_to_group = {row["Subject"]: row["Treatment"] for _, row in key_df.iterrows()}

    bin_frames = int(PROB_BIN_SEC * FPS)
    group_bins = {g: [] for g in GROUP_ORDER}
    summaries = []

    for sess in sessions:
        mouse = sess.replace("_Baseline", "")
        group = mouse_to_group.get(mouse)
        if not group:
            step(f"  ? no group for {sess}; skip"); continue
        pkls = sorted(FEATURES_DIR.glob(f"{sess}_features_414c7814.pkl"))
        if not pkls:
            pkls = sorted(FEATURES_DIR.glob(f"{sess}_features_*.pkl"))
        if not pkls:
            step(f"  ! no feature pkl for {sess}; skip"); continue
        pkl = pkls[-1]
        # h5 optional (Ego_* will be backfilled in predict_state_sequence)
        h5_candidates = (
            list((PROJECT / "videos").glob(f"{sess}*shuffle9*_filtered.h5"))
            + list((PROJECT / "videos").glob(f"{mouse}DLC*shuffle9*_filtered.h5"))
            + list((PROJECT / "videos").glob(f"{mouse}_BaselineDLC*shuffle9*_filtered.h5"))
            + list((PROJECT / "videos").glob(f"{sess}*shuffle8*_filtered.h5"))
        )
        if not h5_candidates:
            h5_candidates = list(PROJECT.rglob(f"{mouse}*_filtered.h5"))
        # Fall back to raw (unfiltered) DLC h5 if no _filtered.h5 exists.
        # augment_features_post_cache reads positions/likelihoods which
        # are present in both — the filtering is only a smoothing pass.
        if not h5_candidates:
            h5_candidates = (
                list((PROJECT / "videos").glob(f"{sess}*shuffle9*.h5"))
                + list((PROJECT / "videos").glob(f"{mouse}DLC*shuffle9*.h5"))
                + list((PROJECT / "videos").glob(f"{mouse}_BaselineDLC*shuffle9*.h5"))
                + list((PROJECT / "videos").glob(f"{mouse}DLC*shuffle8*.h5"))
            )
            # Don't accept full / meta pickles
            h5_candidates = [p for p in h5_candidates
                             if "_full" not in p.name and "_meta" not in p.name]
        h5 = h5_candidates[0] if h5_candidates else None

        step(f"  {sess} -> {group}  (pkl={pkl.stem}{'  h5=' + h5.name if h5 else '  no h5'})")
        state, n = predict_state_sequence(pkl, h5, classifier_specs)
        centres, fracs = bin_state_sequence(state, n, bin_frames)
        group_bins[group].append((centres, fracs))
        summaries.append({"session": sess, "group": group,
                          "n_frames": int(n), "n_bins": int(fracs.shape[0])})

    summ_df = pd.DataFrame(summaries)
    step("\nper-session inventory:")
    print(summ_df.to_string(index=False))

    # Average per group, then renormalise so every displayed bin sums to 1.
    # Per-session fracs already sum to 1 within bins that have ANY classified
    # frame, but bins where one or more sessions had only noise frames pull
    # the across-session mean below 1. Renormalising the group mean per bin
    # makes the stacked area top reach exactly 1.0 (the residual is implicit
    # noise, redistributed proportionally over the 5 states).
    mean_per_group = {}
    for g, runs in group_bins.items():
        if not runs: continue
        min_bins = min(r[1].shape[0] for r in runs)
        centres = runs[0][0][:min_bins]
        stacked = np.stack([r[1][:min_bins] for r in runs], axis=0)
        mean_fracs = stacked.mean(axis=0)
        row_sum = mean_fracs.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        mean_fracs = mean_fracs / row_sum
        mean_per_group[g] = (centres, mean_fracs, stacked.shape[0])

    state_colors = [STATE_COLORS[s] for s in STATES_ORDER]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True,
                              sharex=True)
    for row, group in enumerate(GROUP_ORDER):
        if group not in mean_per_group:
            axes[row].set_visible(False); continue
        centres, fracs, n_sess = mean_per_group[group]
        ax = axes[row]
        ax.stackplot(centres, fracs.T, labels=STATES_ORDER,
                     colors=state_colors, alpha=0.92)
        ax.set_xlim(centres[0], centres[-1])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        ax.set_title(f"{group}  (n = {n_sess})", fontweight="bold")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=True,
                      framealpha=0.95, ncol=1, edgecolor="#cccccc")
        if row == 1:
            ax.set_xlabel("Time (s)")

    out_dir = ANALYSIS
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "temporal_probability_all_animals"
    inv_csv = out_dir / "temporal_probability_all_animals_inventory.csv"
    summ_df.to_csv(inv_csv, index=False)

    fig.savefig(stem.with_suffix(".png"), format="png", dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)

    n_naive = (summ_df["group"] == "Naive").sum()
    n_sntx  = (summ_df["group"] == "SNTX").sum()
    step(f"\n-> {stem}.{{png,pdf,svg}}")

    discord_upload(
        f"**SNLT Baseline — stacked-area temporal probability (ALL animals)**\n"
        f"Naïve (n={n_naive}) on top, SNTX (n={n_sntx}) on bottom. {PROB_BIN_SEC:.0f}-s bins, "
        f"argmax priority Facial_grooming > Left_licking > rearing > still > walking, "
        f"noise frames dropped + renormalised. Each panel is the per-bin "
        f"mean fraction across every animal in the group. PNG + PDF + SVG + "
        f"per-session inventory CSV.",
        [stem.with_suffix(".png"), stem.with_suffix(".pdf"),
         stem.with_suffix(".svg"), inv_csv],
    )


if __name__ == "__main__":
    main()
