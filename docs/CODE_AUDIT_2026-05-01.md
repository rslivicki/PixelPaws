# PixelPaws Code Audit — 2026-05-01

Author: in-session walkthrough by Claude (not a delegated agent). Complements the UX audit at `C:\Users\Gereau\.claude\projects\E--PixelPaws\memory\project_ux_audit_2026_05_01.md` — that one looked at layout/widget/theming. This one looks at **code-level** issues found by reading specific functions.

Scope: I read `unsupervised_tab.py`, `evaluation_tab.py`, `prediction_pipeline.py`, `transitions_tab.py`, `feature_cache.py`, `pose_features.py`, and used grep to find cross-cutting patterns. I did NOT exhaustively read the larger files (`PixelPaws_GUI.py`, `gait_limb_tab.py`, `analysis_tab.py`) — flag follow-ups for those.

---

## P0 — silently broken behaviour, fix today

### 1. `_apply_bout_filtering` ignores `min_after_bout` everywhere

**Where**: `evaluation_tab.py:51-85`

**What**: The function signature accepts `min_after_bout` as a parameter, but the function body never references it. Every site that calls this function — there are **18 of them** across `PixelPaws_GUI.py`, `evaluation_tab.py`, `bout_eval.py`, `_compute_transitions_formoxy.py` — passes a value, all of which are silently dropped.

```python
def _apply_bout_filtering(y_pred, min_bout, min_after_bout, max_gap):
    """Apply bout filtering: min-bout removal and gap bridging."""
    y_filtered = y_pred.copy()

    # --- min_bout: remove bouts shorter than threshold ---
    in_bout = False
    bout_start = 0
    for i in range(len(y_filtered)):
        if y_filtered[i] == 1 and not in_bout:
            ...
    # --- max_gap: bridge short gaps between bouts ---
    if max_gap > 0:
        ...
    return y_filtered
    # NO min_after_bout LOGIC
```

The classifier .pkl format stores `min_after_bout`, the evaluation tab UI exposes it, the eval auto-tuner sweeps over `[0, 1, 3, 5]` (line 1317), the post-processing optimization research doc explicitly compares it against competitors. **None of those efforts are doing anything.** Whatever ratio of post-bout suppression frames the user picks, the predicted label sequence is identical.

**Why it matters**: `min_after_bout` is a documented, exposed knob. Users have been tuning a noop for some unknown length of time. The eval tab also reports per-config metrics that are identical across all min_after_bout values (which the user may have noticed and dismissed as "no effect for this dataset").

**Fix**:

```python
# After the min_bout pass, add a min_after_bout pass that suppresses
# bouts when there's another bout within `min_after_bout` frames AFTER
# the current one ends (or BEFORE it starts — confirm intent with the
# user / classifier_training.py docstring).
if min_after_bout > 0:
    bouts = _extract_bouts(y_filtered)  # list of (start, end_exclusive)
    for i in range(1, len(bouts)):
        prev_end = bouts[i - 1][1]
        cur_start = bouts[i][0]
        if (cur_start - prev_end) < min_after_bout:
            y_filtered[bouts[i][0]:bouts[i][1]] = 0
```

The exact semantic — "no new bout within N frames of the end of the previous one" — should be confirmed against the README / classifier_training docstring before shipping. Add a unit test on `evaluation_tab.py` that proves at least one input pair `(min_after_bout=0, min_after_bout=5)` produces *different* outputs for some y_pred. The test would have caught this.

### 2. UnsupervisedTab `_video_ext_var` AttributeError on every project switch

**Where**: `unsupervised_tab.py:220-223` and `unsupervised_tab.py:315`

**What**: When `umap-learn`/`hdbscan` are not installed, `_build_ui` returns early at line 222–223 before `_build_sessions_panel` is called. `_video_ext_var = tk.StringVar(value='.mp4')` is created inside `_build_sessions_panel` (line 315), so it never exists in the missing-deps branch. Any later project change fires `_scan_sessions` (line 916) which reads `self._video_ext_var.get()` and crashes — **silently**, into the Tk callback void.

```python
def _build_ui(self):
    if not UMAP_HDBSCAN_AVAILABLE:
        self._build_missing_deps_ui()
        return                              # ← returns BEFORE _video_ext_var is created
    ...
    self._build_sessions_panel(left_frame)   # this is where _video_ext_var lives
```

**Why it matters**: This was hit during the `_capture_gui_screenshots.py` run today. Users may not see the traceback because `tkinter.Tk.report_callback_exception` writes to stderr, which is invisible from the .exe build. They just see the Discover tab not updating.

**Fix** (5 min):

```python
def __init__(self, parent, app):
    super().__init__(parent)
    self.app = app
    # ... other init ...
    self._video_ext_var = tk.StringVar(value='.mp4')   # hoist out of _build_sessions_panel
    self._build_ui()
```

Then change `_build_sessions_panel` to just reference the existing var instead of re-creating it. Same pattern for any other StringVar/IntVar that lives in a panel-builder method — do an audit of the file for that.

### 3. 109 silent `except Exception: pass` blocks

**Where**: 19 files; concentrated in `PixelPaws_GUI.py` (24), `gait_limb_tab.py` (27), and `unsupervised_tab.py` (11).

**What**: A pattern like

```python
try:
    self._render_legend()
except Exception:
    pass
```

swallows every failure mode — including the `_video_ext_var` AttributeError above, which is the reason that bug went un-noticed long enough for me to discover it accidentally. It also hides:
- Type errors from refactoring drift (e.g. `clf_data['Behavior_type']` becoming a list instead of a string in one code path)
- Permission errors on write paths
- File-not-found errors that should be visible

**Why it matters**: Bugs accumulate. The April delta audit already flagged some of these; the count has not gone down.

**Fix** (incremental, not all at once): when you touch any of these blocks for an unrelated reason, replace `pass` with at minimum `traceback.print_exc()` or `log_fn(f"...failed: {e}")` so the failure is visible. Audit the 24 instances in `PixelPaws_GUI.py` specifically — they're in the hot path for the GUI's user-facing operations.

A grep for the audit:

```bash
grep -rn "except Exception:" --include="*.py" | grep -A1 ": *$" | grep -B1 "pass"
```

### 4. `_palette_var` default is `'deep'`, but `_SEQUENTIAL` doesn't include it

**Where**: `transitions_tab.py:800` and `transitions_tab.py:2754, 2958`

**What**:

```python
self._palette_var = tk.StringVar(value='deep')              # line 800
...
_SEQUENTIAL = {'magma','plasma','viridis','inferno','cividis','turbo'}
_heatmap_cmap = self._palette_var.get() if self._palette_var.get() in _SEQUENTIAL else 'YlOrRd'
```

`'deep'` is a seaborn discrete palette name; it's not in `_SEQUENTIAL`, so the default heatmap palette is **always** the YlOrRd fallback. The user gets YlOrRd until they explicitly pick another palette from the dropdown.

This was already in the May 1 UX audit (verified there), restating because the fix is one line and it's the source of "the heatmaps don't match the inferno theme" complaints.

**Fix**:

```python
self._palette_var = tk.StringVar(value='inferno')   # was 'deep'
```

---

## P1 — significant, fix this week

### 5. 214 hardcoded `Arial` font references; no central font constant

**Where**: spread across `PixelPaws_GUI.py` (46), `gait_limb_tab.py` (65), `analysis_tab.py` (62), `feature_schematic.py` (16), and 6 other files.

**What**: every `font=('Arial', 10)` is a literal. Switching to the system font, supporting non-Latin scripts, or changing the body size requires editing 214 lines across 10 files.

**Fix**:

```python
# In ui_utils.py
FONT_BODY    = ('Segoe UI', 10)
FONT_HEADER  = ('Segoe UI', 14, 'bold')
FONT_MONO    = ('Consolas', 11)
FONT_SMALL   = ('Segoe UI', 9)
```

Then a one-shot codemod:

```bash
ruff check --select=ALL --fix-only --unsafe-fixes  # or use rope/jedi
# Or:  sed -i "s/font=('Arial', 10)/font=FONT_BODY/g" *.py
```

Do this in a dedicated PR with a single review focus.

### 6. Two ad-hoc `find_session_triplets` import-fallback stubs

**Where**: `body_contact_tab.py:46-50` and `gait_limb_tab.py:82-86`. Both:

```python
try:
    from evaluation_tab import find_session_triplets
except ImportError:
    def find_session_triplets(folder, **kw):
        return []
```

**What**: Identical fallback block. If `evaluation_tab.py` ever fails to import, both tabs silently get an empty session list with no error message — the user sees an empty session list and assumes the project has no sessions.

**Fix**: move `find_session_triplets` into `io_utils.py` (which already exists) so the import never has to come from `evaluation_tab` — that file is doing too many things anyway. Or, at minimum, log the ImportError before installing the stub:

```python
except ImportError as _e:
    print(f"[body_contact_tab] WARNING: could not import find_session_triplets: {_e}")
    def find_session_triplets(folder, **kw):
        return []
```

### 7. `predict_with_xgboost` uses bare `print()` instead of `log_fn`

**Where**: `prediction_pipeline.py:559-634`

**What**: The function logs feature-selection messages, calibration warnings, and fold-averaging notices via `print()`:

```python
print(f"  Selected {len(model.feature_names_in_)} features for prediction")
print(f"  Averaged {len(fold_probas)} models (final + {len(fold_probas) - 1} fold)")
print(f"  Warning: calibrator failed ({_cal_err}); using raw probabilities")
```

These messages are useful — calibrator failures are silent disasters — but in the GUI they go to the controlling console, which is hidden when the GUI is launched from a shortcut. Inside `_compute_transitions_formoxy.py` they showed up because we ran from a shell, but the GUI Predict tab user sees nothing.

**Fix**: accept `log_fn` parameter (matching the rest of the pipeline functions):

```python
def predict_with_xgboost(model, X, calibrator=None, fold_models=None,
                         log_fn=None):
    _log = log_fn if log_fn else print
    ...
    _log(f"  Selected {len(model.feature_names_in_)} features for prediction")
```

Update all 18+ call sites. Same fix pattern applies to `compute_brightness_category_b` and `compute_normalized_distances`, which already accept log_fn but their callers don't always pass one.

### 8. `_apply_bout_filtering` is a Python loop on 290k-frame sessions

**Where**: `evaluation_tab.py:51-85`

**What**: The min_bout pass and max_gap pass both walk every frame in Python. For a 30-min session at 60 fps (~108k frames) running across 5 classifiers and 12 sessions × 4 LOVO folds, that's ~5M Python-level iterations per evaluation run. Adds 10–30 s per eval cycle.

**Fix**: vectorize using `numpy` run-length encoding (similar to what `bout_eval.py:find_bouts` already does):

```python
def _apply_bout_filtering_vec(y_pred, min_bout, min_after_bout, max_gap):
    y = np.asarray(y_pred, dtype=np.int8).copy()
    if y.size == 0:
        return y
    # Find run boundaries
    diff = np.diff(np.concatenate([[0], y, [0]]))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    lens   = ends - starts
    # min_bout — zero out short positive runs
    short = lens < min_bout
    for s, e in zip(starts[short], ends[short]):
        y[s:e] = 0
    # max_gap — bridge short zero gaps between adjacent positive runs
    ...
```

A 5-line numpy version typically runs ~50× faster on long arrays.

---

## P2 — code-quality, fix this month

### 9. `augment_features_post_cache` is six try-blocks deep, all silent

**Where**: `prediction_pipeline.py:641-790`

The function has six independent try/except blocks for ego, contact, lag, multi-timescale, brightness Category B, and normalized distances. Each catches Exception, logs via `log_fn`, and continues. If three of six fail silently the model still gets called with whatever survived — which is exactly how the SNLT classifiers ended up "needing 7 features" in today's debugging.

**Fix**: collect all failures into a list, raise a single `FeatureAugmentationError` at the end with the full failure list, AND let the caller decide whether to proceed with partial features or abort. The classifier-portability check at the top of `prediction_pipeline.py` should also be extended to verify augmented column coverage, not just base coverage.

### 10. `feature_cache.compute_hash` does NOT include `multiscale_*` config

**Where**: `feature_cache.py:50-74`

The hash dict at line 57 includes `bp_include_list`, `bp_pixbrt_list`, `square_size`, `pix_threshold`, `pose_feature_version`, `brightness_feature_version`, `include_optical_flow`, `bp_optflow_list`, optionally `compute_silhouette` and `silhouette_floor`.

It does NOT include `multiscale_windows_ms`, `multiscale_stats`, `use_multiscale_features`. Those settings affect the *augmented* feature set, not the cache base, so technically the hash doesn't need to track them — but the *training-time* cache also stores the augmented columns, so two classifiers trained with different multiscale windows on the same project will collide. (The April delta audit flagged "multi-timescale cross-session bleed" — this is one face of that.)

**Fix**: add a `pose_feature_schema_hash` field that's a content-addressed hash of the literal column-name list produced by `extract_all_features`. Then the cache hash captures schema changes regardless of which config knob caused them.

### 11. `_compute_transitions_formoxy.py` is a one-off script in the repo root

This one's mine, but worth listing: I wrote a 360-line standalone script today to side-step the cache-upgrade chain. The proper fix is to teach `prediction_pipeline._load_features_for_prediction` how to derive `|d/dt(Pix_*)|` and `Log10(Pix_*/Pix_*)` from raw `Pix_*` columns (the same logic I implemented in `_compute_transitions_formoxy.derive_brightness_features`). Then nobody else has to write a side-stepping script the next time this happens.

**Fix**: move `derive_brightness_features` into `brightness_features.py` as a public function, then have `augment_features_post_cache` call it BEFORE `compute_brightness_category_b`. Also delete `_compute_transitions_formoxy.py` once that's wired up — its existence is a documentation smell.

---

## P3 — nits

- **Repo root clutter**: `regen_scratching_plots.py`, `regen_flinching_*.py`, `bout_eval.py`, `prune_and_save_post.py`, `_compute_transitions_formoxy.py`, `_plot_transitions_gui_style.py`, `_render_transitions_pair.py`, `_render_transitions_triple.py`, `_render_snlt_classifier_f1.py`, `_replot_no_s6.py`, `_capture_gui_screenshots.py`, `feature_schematic.py`, `belly_groundtruth_check.py`, `silhouette_*.py` are all research/one-off scripts in the same folder as production GUI modules. Move them to `scripts/` or `research/`.
- **Three remaining bare `except:` blocks** in `analyze_batch_results.py`, `brightness_preview.py`, `brightness_features.py`. Convert to `except Exception:` minimum.
- **`PixelPaws_GUI.py.vscode-overwrite-20260428-2133.bak`** is checked in. `.gitignore` it or delete.
- **`from pose_features import POSE_FEATURE_VERSION` ImportError fallback hard-codes `5`** (`feature_cache.py:28`). If the constant moves or renames, the fallback silently uses a stale number. Either fail loudly or `raise` from the except.

---

## Quick wins (under 1 hour total)

1. **Fix `min_after_bout`** semantic in `_apply_bout_filtering` + add a 3-line unit test. ~30 min.
2. **Hoist `_video_ext_var`** in `unsupervised_tab.py.__init__`. 5 min.
3. **Default palette `'inferno'`** in `transitions_tab.py:800`. 1 min.
4. **Delete `.bak` file** + add to `.gitignore`. 1 min.
5. **Move research scripts to `scripts/`** — git mv, not delete. ~15 min.

That's ~52 minutes for five concrete improvements that ship today.

---

## Follow-ups not investigated here

- `PixelPaws_GUI.py` (12k lines) — too big for this pass; needs a dedicated walk through the predict/train/eval methods specifically.
- `gait_limb_tab.py` (553k lines, 27 silent except blocks) — likely a similar pattern of swallowed errors. Worth its own audit.
- `analysis_tab.py` (247k lines) — the prior April UX audit flagged peri-stimulus alignment gaps but I didn't read the file directly.
- The 7-tier directory fallback search in `feature_cache.find_any_cache` — works, but is hard to reason about. Worth diagramming.
