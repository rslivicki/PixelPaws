"""Cross-classifier feature-importance report (report only — no trimming, no retraining).

Sweeps every trained classifier (project + global encyclopedia), scores each feature's
importance, maps features to the same named families used by fig_shap_families, and writes:

    paper_figures/shap_report/per_feature_importance.csv   feature x classifier, importance + family
    paper_figures/shap_report/family_importance.csv        family x classifier, % of total importance
    paper_figures/shap_report/shap_family_importance.png|pdf|svg   stacked-bar summary

Per classifier the importance is computed as mean(|tree-SHAP|) via XGBoost native
`pred_contribs` when an exact training matrix (a `*_train_set.pkl`) can be found for it;
otherwise it falls back to XGBoost **gain** importance (always available from the booster,
needs no data). The method used is recorded per classifier so the report is honest about
which rows are SHAP vs gain.

Run in the PixelPaws GUI Python (needs numpy/pandas/matplotlib/xgboost/joblib; the `shap`
library is NOT required — we use the booster's native pred_contribs):

    python scripts/research/shap_importance_report.py --project "E:/path/to/ProjectFolder"

Options:
    --project DIR        project folder (its classifiers/ dir is swept); optional
    --train-set-dir DIR  extra dir searched recursively for *_train_set.pkl (enables SHAP)
    --out DIR            output dir (default: <repo>/paper_figures/shap_report)
    --max-rows N         row subsample cap for the SHAP pass (default 40000)
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Named feature families — mirrors scripts/research/fig_shap_families.py::family (keep in sync).
FAMILIES = ["Paw brightness", "Velocity/accel", "Body angle",
            "Inter-part distance", "Height/elevation", "Visibility"]
FAMILY_COLOR = {"Paw brightness": "#f2a900", "Velocity/accel": "#4a8ec7",
                "Body angle": "#9b59b6", "Inter-part distance": "#2ecc71",
                "Height/elevation": "#e67e22", "Visibility": "#95a5a6"}


def family(f: str) -> str:
    fl = str(f).lower()
    if "pix" in fl or "bright" in fl:
        return "Paw brightness"
    if "vel" in fl or "jerk" in fl or "speed" in fl or "accel" in fl or "d/dt" in fl:
        return "Velocity/accel"
    if "ang" in fl or "ori" in fl:
        return "Body angle"
    if "height" in fl or "elev" in fl or "surfacez" in fl:
        return "Height/elevation"
    if "dis" in fl or "ego" in fl:
        return "Inter-part distance"
    if "inframe" in fl or fl.startswith("any") or "_any" in fl:
        return "Visibility"
    return "Visibility"  # residual (tracking/misc) folded in, matching fig_shap_families


def _load_pickle(path):
    """joblib first (handles compressed bundles), then plain pickle."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as fh:
            return pickle.load(fh)


def discover_classifiers(project: str | None):
    """{display_name -> pkl path} for project + encyclopedia classifiers.

    Mirrors PixelPaws_GUI._pf_list_classifiers (PixelPaws_GUI.py:8599)."""
    out = {}
    roots = []
    if project:
        roots.append((os.path.join(project, "classifiers"), ""))
    roots.append((os.path.join(REPO, "pixelpaws_global_classifier_encyclopedia",
                               "classifiers"), " (encyclopedia)"))
    for root, tag in roots:
        for p in sorted(glob.glob(os.path.join(root, "classifier_*.pkl"))):
            name = os.path.basename(p)[len("classifier_"):-len(".pkl")]
            out.setdefault(f"{name}{tag}", p)
    return out


def _model_cols(model, bundle):
    """Ordered feature names the model was trained on."""
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    for k in ("selected_feature_cols", "feature_names"):
        v = bundle.get(k) if isinstance(bundle, dict) else None
        if v:
            return list(v)
    return None


def _find_train_matrix(clf_path, bundle, extra_dir):
    """Best-effort locate the exact trained matrix X (a *_train_set.pkl dict with 'X').

    Searches next to the classifier, its training_data/ siblings, and an optional extra dir.
    Returns a DataFrame or None."""
    search = []
    cdir = os.path.dirname(clf_path)
    search += glob.glob(os.path.join(cdir, "*_train_set.pkl"))
    search += glob.glob(os.path.join(cdir, "**", "*_train_set.pkl"), recursive=True)
    if extra_dir:
        search += glob.glob(os.path.join(extra_dir, "**", "*_train_set.pkl"), recursive=True)
    for p in sorted(set(search), key=lambda x: os.path.getmtime(x) if os.path.isfile(x) else 0,
                    reverse=True):
        try:
            d = _load_pickle(p)
            X = d.get("X") if isinstance(d, dict) else None
            if X is not None and getattr(X, "shape", (0,))[0] > 0:
                return X
        except Exception:
            continue
    return None


def _importance_for(model, bundle, cols, X, max_rows):
    """Return (importance_vector aligned to cols, method_str).

    SHAP (mean|pred_contribs|) if X covers cols; else XGBoost gain importance."""
    import xgboost as xgb
    # SHAP path — needs every trained column present in X.
    if X is not None and all(c in getattr(X, "columns", []) for c in cols):
        try:
            Xr = X[cols]
            if hasattr(Xr, "replace"):
                Xr = Xr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if Xr.shape[0] > max_rows:
                Xr = Xr.sample(max_rows, random_state=42)
            booster = model.get_booster()
            contribs = booster.predict(xgb.DMatrix(Xr, feature_names=cols), pred_contribs=True)
            return np.abs(np.asarray(contribs)[:, :-1]).mean(axis=0), "shap"
        except Exception as e:
            print(f"    SHAP failed ({e}); using gain importance", flush=True)
    # Gain fallback — always available from a fitted XGBoost model, no data needed.
    try:
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")  # {feat_name: gain}
        # booster feature names may be f0.. if not set; align via feature_names_in_ order.
        bnames = booster.feature_names
        if bnames and len(bnames) == len(cols):
            gain = np.array([score.get(bn, 0.0) for bn in bnames], dtype=float)
        else:
            gain = np.array([score.get(c, 0.0) for c in cols], dtype=float)
        if gain.sum() > 0:
            return gain, "gain"
    except Exception:
        pass
    fi = np.asarray(getattr(model, "feature_importances_", np.zeros(len(cols))), dtype=float)
    return fi, "gain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--train-set-dir", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "paper_figures", "shap_report"))
    ap.add_argument("--max-rows", type=int, default=40000)
    args = ap.parse_args()

    import pandas as pd

    classifiers = discover_classifiers(args.project)
    if not classifiers:
        print("No classifiers found. Pass --project <folder> (and the encyclopedia is searched "
              "automatically).", flush=True)
        return
    print(f"Found {len(classifiers)} classifier(s).", flush=True)

    per_feature_rows = []          # dict rows for the per-feature CSV
    family_by_clf = {}             # display -> {family: pct}
    method_by_clf = {}
    covered, gain_fallback, skipped = [], [], []

    for display, path in classifiers.items():
        try:
            bundle = _load_pickle(path)
        except Exception as e:
            print(f"  [skip] {display}: cannot load ({e})", flush=True)
            skipped.append(display); continue
        model = bundle.get("clf_model") if isinstance(bundle, dict) else bundle
        if model is None:
            print(f"  [skip] {display}: no clf_model in bundle", flush=True)
            skipped.append(display); continue
        cols = _model_cols(model, bundle)
        if not cols:
            print(f"  [skip] {display}: no feature names on model/bundle", flush=True)
            skipped.append(display); continue

        X = _find_train_matrix(path, bundle, args.train_set_dir)
        imp, method = _importance_for(model, bundle, cols, X, args.max_rows)
        if imp is None or np.asarray(imp).sum() <= 0:
            print(f"  [skip] {display}: zero/undefined importance", flush=True)
            skipped.append(display); continue

        imp = np.asarray(imp, dtype=float)
        total = imp.sum() or 1.0
        fam_pct = {fam: 0.0 for fam in FAMILIES}
        for c, v in zip(cols, imp):
            fam = family(c)
            fam_pct[fam] += 100.0 * float(v) / total
            per_feature_rows.append(dict(classifier=display, feature=c, family=fam,
                                         importance=float(v),
                                         pct_within_classifier=100.0 * float(v) / total,
                                         method=method))
        family_by_clf[display] = fam_pct
        method_by_clf[display] = method
        (covered if method == "shap" else gain_fallback).append(display)
        print(f"  [ok]  {display}: {len(cols)} features, method={method}", flush=True)

    if not family_by_clf:
        print("No classifiers produced importances; nothing written.", flush=True)
        return

    os.makedirs(args.out, exist_ok=True)

    # per-feature CSV (descending importance within each classifier)
    pf = pd.DataFrame(per_feature_rows).sort_values(
        ["classifier", "pct_within_classifier"], ascending=[True, False])
    pf_path = os.path.join(args.out, "per_feature_importance.csv")
    pf.to_csv(pf_path, index=False)

    # family x classifier matrix
    fam_df = pd.DataFrame(family_by_clf).reindex(FAMILIES).fillna(0.0)  # rows=family, cols=clf
    fam_df.loc["method"] = [method_by_clf[c] for c in fam_df.columns]
    fam_path = os.path.join(args.out, "family_importance.csv")
    fam_df.to_csv(fam_path)

    # stacked-bar figure (family % per classifier)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        clfs = list(family_by_clf.keys())
        x = np.arange(len(clfs))
        bottom = np.zeros(len(clfs))
        fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(clfs) + 3), 5.2), constrained_layout=True)
        for fam in FAMILIES:
            vals = np.array([family_by_clf[c][fam] for c in clfs])
            ax.bar(x, vals, bottom=bottom, color=FAMILY_COLOR[fam], label=fam,
                   width=0.7, ec="white", lw=0.5)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n[{method_by_clf[c]}]" for c in clfs],
                           rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Feature importance (% of total)")
        ax.set_ylim(0, 100)
        ax.set_title("What drives each classifier (family importance)", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        stem = os.path.join(args.out, "shap_family_importance")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(stem + "." + ext, dpi=300 if ext == "png" else None, bbox_inches="tight")
        fig_note = stem + ".png"
    except Exception as e:
        fig_note = f"(figure skipped: {e})"

    print("\n=== SHAP / importance report ===", flush=True)
    print(f"classifiers covered (SHAP): {len(covered)}", flush=True)
    print(f"gain-importance fallback:   {len(gain_fallback)}  {gain_fallback}", flush=True)
    print(f"skipped:                    {len(skipped)}  {skipped}", flush=True)
    print(f"wrote: {pf_path}", flush=True)
    print(f"wrote: {fam_path}", flush=True)
    print(f"wrote: {fig_note}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
