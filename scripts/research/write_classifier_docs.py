"""Write per-classifier .md docs for all entries in manifest.json."""
import json
from pathlib import Path

ENCY = Path(r"E:/PixelPaws/pixelpaws_global_classifier_encyclopedia")
DOCS_DIR = ENCY / "docs"
DOCS_DIR.mkdir(exist_ok=True)

manifest = json.loads((ENCY / "manifest.json").read_text())

for c in manifest["classifiers"]:
    name = c["name"]
    tier = c["tier"]
    status = c["status"]
    tier_name = {1: "KEEP", 2: "BOUT_USEFUL", 3: "BELOW_GATE"}[tier]
    md_path = DOCS_DIR / f"{name}.md"
    if md_path.exists() and tier == 1:
        # Keep existing Tier 1 docs as-is (they have more info)
        continue
    fs = c["feature_schema"]
    metrics_path = ENCY / c["performance_dir"] / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    bo_best = metrics.get("bout_optimization", {}).get("best", {})
    dur_err = bo_best.get("mean_bout_duration_error_sec")
    dur_err_str = f"{dur_err:.2f}s" if isinstance(dur_err, (int, float)) else "n/a"

    md = f"""# {name}

**Tier {tier}: {tier_name}**

## Behaviour definition
{c['definition']}

## Performance
- **Frame F1 (CV)**: {c['frame_f1']:.3f} ± {c['frame_f1_std']:.3f}
- **Bout F1 (post-optimized)**: {c['bout_f1']:.3f}
- **Bout-count Pearson r**: {c['bout_count_pearson_r']:.3f}
- **Mean bout duration error**: {dur_err_str}

## Recommended post-processing
- `best_thresh`: {c['best_thresh']:.2f}
- `min_bout`: {c['min_bout']}
- `min_after_bout`: {c['min_after_bout']}
- `max_gap`: {c['max_gap']}

These were chosen by joint sweep over (threshold, min_bout, max_gap) on
out-of-fold probabilities to maximize bout-level F1 at IoU≥0.5.

## Training data
- Sessions: {c['n_train_sessions']}
- Positive frames: {c['n_pos_frames']:,}

## Feature schema
- Pose v{fs['pose_feature_version']} + brightness v{fs['brightness_feature_version']}
- Optical flow: {fs['include_optical_flow']}
- square_size: {fs['square_size']}
- brightness body parts: {fs['bp_pixbrt_list']}

## Status notes
{('This classifier passes the 0.85 frame F1 inclusion gate and is '
  'production-ready.' if tier == 1 else
  'Frame-level F1 is below the 0.85 gate, but bout-level performance '
  'is strong (bout F1 ≥ 0.50 + bout-count r ≥ 0.70). Use for "how many '
  'bouts per session" questions where bout count/timing matters more '
  'than per-frame accuracy.' if tier == 2 else
  'Frame F1 < 0.85 AND bout-level criteria not met. Shipped for '
  'transparency but not recommended for production analyses. The '
  'main limitations are usually: (a) small training data, (b) high '
  'per-session label variance, or (c) the behaviour overlaps visually '
  'with adjacent behaviours.')}

## See also
- `../classifiers/classifier_{name}.pkl` — model artefact (load with joblib)
- `../performance_results/{name}/training_metrics.json` — full metrics + sweep results
- `../performance_results/{name}/oof.npz` — out-of-fold probabilities (for plot regen / re-sweep)
"""
    md_path.write_text(md)
    print(f"wrote docs/{name}.md (Tier {tier})")
