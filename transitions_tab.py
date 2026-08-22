"""
transitions_tab.py - PixelPaws Behavioral State Transition Analysis
====================================================================
Computes transition probability matrices between behavioral states,
visualizes ethograms, heatmaps, directed network graphs, and group
comparisons.  Includes latent behavioral state discovery via k-means
clustering of windowed transition matrices, state occupancy analysis,
and PCA-based continuous indices (inspired by LUPE, Nature 2026).

State sources:
  1. Unsupervised clusters from the Discover tab (primary)
  2. Supervised classifier predictions from results/ (optional)
"""

import os
import glob
import re
import hashlib
import json
import pickle
import threading
import traceback
import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog

# ---------------------------------------------------------------------------
# Optional: matplotlib
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional: networkx for directed graph
# ---------------------------------------------------------------------------
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from evaluation_tab import find_session_triplets
    _FIND_SESSION_TRIPLETS_AVAILABLE = True
except ImportError:
    find_session_triplets = None
    _FIND_SESSION_TRIPLETS_AVAILABLE = False


def _robust_load(path):
    """Load a .pkl that may be joblib+LZ4 (canonical caches + encyclopedia classifiers) OR plain
    pickle (project-local GUI artifacts). joblib.load reads both, so it's the universal loader."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as _fh:
            return pickle.load(_fh)

try:
    from feature_cache import FeatureCacheManager as _TransFeatureCacheManager
    _TRANS_FEATURE_CACHE_AVAILABLE = True
except ImportError:
    _TransFeatureCacheManager = None
    _TRANS_FEATURE_CACHE_AVAILABLE = False


from ui_utils import _bind_tight_layout_on_resize, FONT_FAMILY, ToolTip
from io_utils import atomic_pickle_save, get_git_sha

# ═══════════════════════════════════════════════════════════════════════════
# Saved-session (computed results) persistence
# ═══════════════════════════════════════════════════════════════════════════
# Bumped only when the on-disk result schema changes incompatibly. Loading a
# file with a newer schema than this is refused.
SESSION_SCHEMA_VERSION = 1

# The compute-result members serialized into a saved session (all plain
# numpy/dict/str - no tk vars, figures, or XGBoost models). Stored WITHOUT the
# leading underscore; save/load add it. Keep save & load in sync via this list.
# `state_seqs`, `states`, `state_labels` are the required minimum to redraw.
RESULT_FIELDS = [
    'state_seqs', 'state_seqs_full', 'states', 'state_labels', 'session_subjects',
    'matrices', 'windowed', 'temporal_probs', 'group_matrices', 'group_sem',
    'group_subject_matrices', 'effective_fps', 'frame_probs',
    # unsupervised views (retired but cheap arrays - best-effort)
    'occupancy', 'latent_centroids', 'session_latent_map', 'n_latent',
    'group_occupancy', 'group_occupancy_sem', 'pca_scores', 'pca_loadings',
    'merge_info',
]

# ═══════════════════════════════════════════════════════════════════════════
# Color model - shared palette constants
# ═══════════════════════════════════════════════════════════════════════════
# Wong (Nature Methods) / Okabe-Ito 8-color colorblind-safe categorical palette.
_OKABE_ITO = ['#000000', '#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
# Perceptually-uniform sequential cmaps allowed for probability heatmaps.
_SEQ_CMAPS = {'magma', 'plasma', 'viridis', 'inferno', 'cividis', 'turbo',
              'YlOrRd', 'YlGnBu', 'Blues', 'Greens', 'Greys'}
# Zero-centered diverging cmap for group-difference heatmaps.
_DIVERGING_CMAP = 'RdBu_r'

# Results view menu - grouped with non-selectable '- header -' rows.
_VIEW_MENU = [
    '- Composition -', 'State Usage', 'State Composition',
    '- Transitions -', 'Transition Matrix', 'Transition Graph',
    'Group Transition Graphs', 'Group Matrices', 'Transition Difference',
    '- Sequencing -', 'Sequencing Networks', 'Sequencing Difference',
    'Sequencing Ordination',
    '- Temporal -', 'Ethogram', 'Composition Over Time',
    'Occupancy Over Time', 'Behavior × Group', 'Transition Timeline',
]
# Old view name → new, so saved sessions/configs still resolve.
_VIEW_ALIASES = {
    'Occupancy': 'State Usage',
    'Heatmap': 'Transition Matrix',
    'Network': 'Transition Graph',
    'Group Networks': 'Group Transition Graphs',
    'Group Comparison': 'Group Matrices',
    'Temporal Probability': 'Composition Over Time',
    'Behavior Over Time': 'Occupancy Over Time',
    'Timeline': 'Transition Timeline',
}

def _g(var, default=None):
    """Safe tk-var .get() with a fallback (used by the Methods text)."""
    try:
        return var.get()
    except Exception:
        return default


# Per-view "how this graph is calculated" prose. `{...}` fields are filled with
# the live settings by TransitionsTab._methods_text(). Keyed by canonical view.
_METHODS_STATIC = {
    'State Usage':
        "STATE OCCUPANCY (per-behaviour comparison across treatment groups)\n"
        "For each animal, the occupancy of a given behaviour was defined as the "
        "percentage of analysed frames assigned to that state, O = (N_state ÷ "
        "N_total) × 100, where N_state is the number of frames in the state and "
        "N_total the total number of analysed frames; this yields a single occupancy "
        "value per animal per behaviour. Individual animals are plotted as points, the "
        "bar denotes the group mean, and error bars denote {error}. Behaviours are "
        "ordered along the abscissa by {sort} (under 'Group difference' ordering, "
        "behaviours are ranked by the magnitude of the largest between-group difference "
        "relative to the reference).\n\n"
        "Statistical analysis. For each behaviour, occupancy was compared across "
        "treatment groups using {test}; the resulting p-values were adjusted for "
        "multiple comparisons across the set of behaviours using {correction}, with "
        "significance assessed at α = {alpha}. Adjusted p-values (q) are annotated "
        "beneath each behaviour and significance is denoted * q<0.05, ** q<0.01, "
        "*** q<0.001. Nonparametric tests are the default, reflecting the typically "
        "non-normal distribution and small sample sizes of behavioural occupancy data. "
        "The {ref} group served as the reference for pairwise contrasts.",
    'State Composition':
        "BEHAVIOURAL TIME BUDGET (compositional summary)\n"
        "For each animal, the proportion of session time spent in every behaviour was "
        "computed as in the occupancy analysis. Per-animal proportions were averaged "
        "within each treatment group and displayed as a single 100%-stacked bar per "
        "group, in which segment heights represent the group-mean proportion of time "
        "allocated to each behaviour and sum to unity. This view summarises the overall "
        "allocation of behaviour across groups; formal per-behaviour statistical "
        "comparisons are provided in the State Usage view.",
    'Transition Matrix':
        "FIRST-ORDER MARKOV TRANSITION MATRIX\n"
        "Behavioural sequencing was summarised by a first-order Markov transition "
        "matrix estimated for each session and averaged across sessions. Transitions "
        "were enumerated {transition_mode}. Transition counts were row-normalised so "
        "that entry T(i, j) = P(next state = j | current state = i) and each row sums "
        "to one; the matrix shown is the across-session mean of the per-session "
        "matrices. {norm}{zerodiag}\n\n"
        "Interpretation. Each row is the outgoing conditional distribution from a "
        "behaviour; off-diagonal structure identifies preferred behavioural sequences.",
    'Transition Graph':
        "TRANSITION GRAPH (directed-network representation)\n"
        "The mean transition matrix (estimated {transition_mode}) is rendered as a "
        "directed graph in which nodes represent behavioural states and directed edges "
        "represent transitions. Node area is scaled to overall state occupancy and edge "
        "width is scaled to the corresponding transition probability; edges with "
        "probability below a small display threshold are omitted for legibility, and "
        "node colour encodes behavioural identity.",
    'Group Transition Graphs':
        "GROUP-WISE TRANSITION GRAPHS\n"
        "The directed transition network was constructed separately for each treatment "
        "group from that group's mean transition matrix (estimated {transition_mode}), "
        "with node area scaled to state occupancy and edge width to transition "
        "probability as described for the Transition Graph. A common node layout is "
        "applied across all panels so that homologous nodes occupy identical positions, "
        "permitting direct visual comparison of sequencing structure across groups.",
    'Group Matrices':
        "GROUP-WISE TRANSITION MATRICES\n"
        "For each treatment group, per-animal first-order Markov transition matrices "
        "(estimated {transition_mode}, row-normalised) were averaged to yield a "
        "group-mean matrix with entry T(i, j) = P(next state = j | current state = i). "
        "{norm}Matrices are displayed side by side on a common colour scale. "
        "Cell-wise statistical comparison of transitions against a reference group is "
        "provided in the Transition Difference view.",
    'Transition Difference':
        "DIFFERENTIAL TRANSITION ANALYSIS\n"
        "For each non-reference group, the element-wise difference between its "
        "group-mean transition matrix and that of the reference group ({ref}) is "
        "displayed as a divergent heatmap centred at zero; positive values (warmer "
        "colours) indicate transitions more probable than in the reference and negative "
        "values indicate transitions less probable.\n\n"
        "Statistical analysis. For every matrix cell, the per-animal transition "
        "probabilities of the group and of the reference were compared with a two-sided "
        "Mann-Whitney U test, and the resulting p-values were adjusted across all cells "
        "of the matrix using {correction}; cells surviving the α = {alpha} threshold are "
        "annotated with significance markers (* q<0.05, ** q<0.01, *** q<0.001).",
    'Ethogram':
        "ETHOGRAM\n"
        "The assigned behavioural state is displayed as a colour-coded raster with one "
        "row per session and time on the abscissa; each column encodes the state "
        "occupied at the corresponding time. For rendering, sequences exceeding 3,000 "
        "samples were decimated to 3,000 columns (each column representing the state of "
        "a short contiguous interval), so sub-column flicker may not be resolved. Rows "
        "are ordered by treatment group.",
    'Composition Over Time':
        "TEMPORAL BEHAVIOURAL COMPOSITION\n"
        "Each session was partitioned into non-overlapping {bin}-second bins, and "
        "within each bin the fraction of frames occupied by each state was computed "
        "(fractions sum to one within a bin). Per-bin fractions were averaged across "
        "the animals of a treatment group and displayed as a stacked-area plot, one "
        "panel per group, depicting the evolution of the behavioural time budget over "
        "the session.",
    'Occupancy Over Time':
        "TIME-RESOLVED STATE OCCUPANCY\n"
        "State occupancy was computed within a sliding window of {win} s advanced in "
        "{step}-second steps; within each window the fraction of frames in each state "
        "was measured to yield a per-animal time course. Time courses were averaged "
        "across the animals of a treatment group (line, group mean; shaded band, SEM), "
        "with one panel per group. This view requires the sliding-window time mode.",
    'Behavior × Group':
        "BEHAVIOUR-WISE TEMPORAL TRAJECTORIES (groups compared)\n"
        "For a given behaviour, each session was partitioned into {bin}-second bins and "
        "the per-bin fraction of time in that behaviour was computed; trajectories were "
        "aligned to a common time axis and averaged across the animals of a treatment "
        "group (line, group mean; shaded band, SEM), with groups distinguished by "
        "colour. The 'All behaviours (grid)' option arranges one panel per behaviour, "
        "whereas selecting a single behaviour yields an enlarged single panel; the Time "
        "(min) fields restrict the displayed interval (e.g. to isolate an acute versus a "
        "delayed phase of a response).",
    'Transition Timeline':
        "TIME-RESOLVED TRANSITION PROBABILITY\n"
        "For a user-selected ordered pair of behaviours (i → j), the conditional "
        "transition probability P(j | i) was estimated within a sliding window of "
        "{win} s advanced in {step}-second steps, yielding a per-animal time course. "
        "Time courses were averaged across the animals of a treatment group (line, "
        "group mean; shaded band, SEM). This view requires the sliding-window time mode.",
}


# ═══════════════════════════════════════════════════════════════════════════
# Hover-help text - single source of truth for the tab's tooltips
# ═══════════════════════════════════════════════════════════════════════════
# Plain-language, one-or-two-sentence descriptions shown on mouse-over via
# ui_utils.ToolTip (theme-aware; wraps at 320px). Keyed by short slug; attach
# with self._tip(widget, 'slug'). Add a slug here, reference it at the widget.
_HELP = {
    # Time Windows
    'full_session':   "Analyze the whole session as one block - a single transition "
                      "matrix / occupancy summary per session.",
    'time_range':     "Analyze only a fixed slice of the session, from Start to End "
                      "(seconds). Everything outside the range is ignored.",
    'sliding_windows': "Compute a separate result for each short window of time as it "
                      "slides across the session - shows how behavior changes over the "
                      "session (e.g. habituation). Set the window length and step below.",
    'window_sec':     "Length of each analysis window, in seconds. Larger windows are "
                      "smoother but give fewer time points.",
    'step_sec':       "How far the window advances between measurements. Smaller than the "
                      "window ⇒ overlapping windows (more, smoother points); equal to the "
                      "window ⇒ back-to-back, non-overlapping bins.",
    'range_start':    "Start of the analyzed slice, in seconds from the recording start.",
    'range_end':      "End of the analyzed slice, in seconds from the recording start.",
    'crop_first':     "Ignore everything after the first N minutes - useful to equalize "
                      "recordings of different lengths before comparing groups.",
    'prob_bin':       "Bin width (seconds) for the Temporal Probability view: each bin "
                      "reports the fraction of time spent in each state.",
    # State assignment
    'assign_priority': "When two or more classifiers fire on the same frame, the one "
                       "higher in the priority list below wins that frame.",
    'assign_argmax':  "When multiple classifiers fire on the same frame, the one with the "
                      "highest probability wins that frame.",
    'priority_list':  "Order classifiers by priority (top = highest). Used only in "
                      "'Priority ranking' mode to break ties between simultaneous behaviors.",
    # Settings
    'smooth_ms':      "Merge state flickers shorter than this many milliseconds into the "
                      "surrounding state, removing single-frame noise before counting "
                      "transitions.",
    'exclude_other':  "Drop the 'Other' state (frames where no classifier fired) from the "
                      "analysis, so transitions and occupancy are computed only between "
                      "named behaviors.",
    'cache_only':     "Use only feature caches that already exist on disk. Sessions whose "
                      "features would need re-extraction (a slow video re-read) are skipped "
                      "and listed, instead of being extracted.",
    'zero_diag':      "Zero the diagonal of the transition matrix so self-transitions "
                      "(a state following itself) don't dominate the display.",
    # State source / classifiers / preset
    'preset_source':  "Load a ready-made set of classifiers: your project's classifiers, "
                      "the shared encyclopedia, or both (deduped, project wins). Loads them "
                      "for review - you then click Compute.",
    'add_classifier': "Add one or more classifier .pkl files to the set used to label "
                      "each frame's state.",
    # Views / palettes
    'open_graph_window': "Open the results in a large, resizable window (recommended) - plots "
                      "fill the window with room for group panels, plus group order/colors, "
                      "per-view axis limits, and Save Figure.",
    'view_combo':     "Choose what to plot from the computed results - occupancy, "
                      "ethograms, transition matrices/networks, temporal probabilities, "
                      "group comparisons, and more.",
    'heat_palette':   "Color map for the transition-matrix heatmap.",
    'bot_palette':    "Color map assigning one color per behavior in the Behavior-Over-Time "
                      "view.",
    # Key file / sessions
    'key_file':       "A CSV/XLSX with Subject and Treatment columns that maps each session "
                      "to its subject and group. Enables the group-comparison views. "
                      "Auto-found and loaded when you open a project.",
    'key_autofind':   "Search the project folder for a valid key file (one with Subject and "
                      "Treatment columns) and load it.",
    'session_group':  "The Treatment group this session's subject belongs to, from the key "
                      "file. Shows '-' until a key file is loaded.",
    # Run
    'compute':        "Run the loaded classifiers on the selected sessions and compute "
                      "transitions + occupancy. Honors 'Use cached features only'.",
    'save_session':   "Save this completed run - all computed results plus the settings that "
                      "produced them - to one file, so you can reopen the plots later without "
                      "re-running the classifiers.",
    'load_session':   "Load a previously saved session and redraw every view instantly, with no "
                      "recompute. The last run is also auto-saved per project.",
}


# ═══════════════════════════════════════════════════════════════════════════
# Transition computation (pure functions, no GUI dependency)
# ═══════════════════════════════════════════════════════════════════════════

def compute_transition_matrix(state_seq, states=None, normalize=True,
                              zero_diagonal=False):
    """Compute transition matrix from a 1-D integer state sequence.

    Parameters
    ----------
    state_seq : array-like of int
    states : list of int, optional - ordered state IDs (rows/cols).
        If None, derived from unique values in *state_seq*.
    normalize : bool - row-normalize to probabilities.
    zero_diagonal : bool - zero self-transition diagonal.

    Returns
    -------
    matrix : np.ndarray (n_states, n_states)
    states : list of int - ordered state IDs matching rows/cols
    """
    seq = np.asarray(state_seq, dtype=int)
    if states is None:
        states = sorted(set(seq))
    state_to_idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    mat = np.zeros((n, n), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        if a in state_to_idx and b in state_to_idx:
            mat[state_to_idx[a], state_to_idx[b]] += 1
    if zero_diagonal:
        np.fill_diagonal(mat, 0)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = mat / row_sums
    return mat, states


def compute_windowed_transitions(state_seq, fps, window_sec, step_sec,
                                 states=None, zero_diagonal=False, mode='frame'):
    """Slide a window across *state_seq* and compute a transition matrix
    for each window position.

    Returns list of (time_center_sec, matrix) tuples.
    """
    seq = np.asarray(state_seq, dtype=int)
    if states is None:
        states = sorted(set(seq))
    win_frames = max(1, int(round(window_sec * fps)))
    step_frames = max(1, int(round(step_sec * fps)))
    results = []
    start = 0
    while start + win_frames <= len(seq):
        chunk = seq[start:start + win_frames]
        if mode == 'bout':
            mat, _ = compute_bout_transition_matrix(
                chunk, states=states, normalize=True)
        else:
            mat, _ = compute_transition_matrix(chunk, states=states,
                                               normalize=True,
                                               zero_diagonal=zero_diagonal)
        center = (start + win_frames / 2) / fps
        results.append((center, mat))
        start += step_frames
    return results, states


def smooth_state_sequence(seq, min_frames):
    """Remove short bouts (< min_frames) by replacing them with the
    surrounding state."""
    if min_frames <= 1:
        return seq
    out = seq.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        bout_len = j - i
        if bout_len < min_frames:
            # Replace with previous state if possible, else next
            replacement = out[i - 1] if i > 0 else (out[j] if j < n else out[i])
            out[i:j] = replacement
        i = j
    return out


def extract_bouts(state_seq):
    """Convert per-frame state sequence to ordered list of bout dicts."""
    if len(state_seq) == 0:
        return []
    bouts, current, start = [], state_seq[0], 0
    for i in range(1, len(state_seq)):
        if state_seq[i] != current:
            bouts.append({'state': current, 'start': start, 'end': i - 1})
            current, start = state_seq[i], i
    bouts.append({'state': current, 'start': start, 'end': len(state_seq) - 1})
    return bouts


def compute_bout_transition_matrix(state_seq, states=None, normalize=True):
    """Transition matrix over bout-to-bout switches. Diagonal is always 0."""
    bouts = extract_bouts(state_seq)
    if states is None:
        states = sorted(set(state_seq))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n))
    for a, b in zip(bouts[:-1], bouts[1:]):
        if a['state'] in idx and b['state'] in idx:
            counts[idx[a['state']], idx[b['state']]] += 1
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        counts = counts / row_sums
    return counts, states


def compute_temporal_probabilities(state_seq, fps, bin_sec, states):
    """Bin *state_seq* into time bins and compute fraction per state.

    Returns
    -------
    time_centers : np.ndarray of float (n_bins,)
    prob_matrix  : np.ndarray (n_bins, n_states)
    """
    seq = np.asarray(state_seq, dtype=int)
    bin_frames = max(1, int(round(bin_sec * fps)))
    n_bins = max(1, len(seq) // bin_frames)
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    prob = np.zeros((n_bins, n_states))
    centers = np.zeros(n_bins)
    for b in range(n_bins):
        start = b * bin_frames
        end = min(start + bin_frames, len(seq))
        chunk = seq[start:end]
        centers[b] = (start + end) / 2.0 / fps
        for frame_val in chunk:
            idx = state_to_idx.get(frame_val)
            if idx is not None:
                prob[b, idx] += 1
        total = prob[b].sum()
        if total > 0:
            prob[b] /= total
    return centers, prob


def cluster_transition_matrices(windowed_dict, k, states, n_init=100):
    """K-means on flattened windowed transition matrices (LUPE method).

    Parameters
    ----------
    windowed_dict : dict  {session: [(time_center, matrix), ...]}
    k : int               number of latent states
    states : list         ordered state IDs
    n_init : int          KMeans n_init

    Returns
    -------
    centroids : np.ndarray (k, n_states, n_states) - centroid matrices
    session_latent_map : dict {session: list of int} - latent state ID per window
    """
    from sklearn.cluster import KMeans

    n_states = len(states)
    flat_rows = []
    session_indices = []  # (session_name, window_idx)
    for session, wresults in windowed_dict.items():
        for wi, (t, mat) in enumerate(wresults):
            flat_rows.append(mat.ravel())
            session_indices.append((session, wi))

    X = np.vstack(flat_rows)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
    labels = km.fit_predict(X)

    # Build per-session map
    session_latent_map = {s: [] for s in windowed_dict}
    for (session, wi), lbl in zip(session_indices, labels):
        session_latent_map[session].append(int(lbl))

    # Reshape centroids
    centroids = km.cluster_centers_.reshape(k, n_states, n_states)
    return centroids, session_latent_map


def compute_state_occupancy(session_latent_map, n_latent_states):
    """Fractional occupancy of each latent state per session.

    Returns
    -------
    dict {session: np.ndarray of shape (n_latent_states,)}
    """
    occupancy = {}
    for session, latent_ids in session_latent_map.items():
        arr = np.array(latent_ids, dtype=int)
        counts = np.bincount(arr, minlength=n_latent_states).astype(float)
        total = counts.sum()
        if total > 0:
            counts /= total
        occupancy[session] = counts
    return occupancy


def pca_on_occupancy(occupancy_dict):
    """PCA on stacked occupancy vectors.

    Returns
    -------
    pca_model : sklearn PCA
    scores_dict : dict {session: (pc1, pc2)}
    loadings : np.ndarray (n_components, n_latent_states)
    """
    from sklearn.decomposition import PCA

    sessions = sorted(occupancy_dict.keys())
    X = np.vstack([occupancy_dict[s] for s in sessions])
    n_comp = min(2, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)
    scores_dict = {}
    for i, s in enumerate(sessions):
        pc1 = float(scores[i, 0]) if n_comp >= 1 else 0.0
        pc2 = float(scores[i, 1]) if n_comp >= 2 else 0.0
        scores_dict[s] = (pc1, pc2)
    return pca, scores_dict, pca.components_


def reduce_clusters(processed_seqs, states, target_n):
    """Merge fine-grained clusters into target_n meta-clusters using
    agglomerative clustering on transition count profiles.

    Parameters
    ----------
    processed_seqs : dict {session_name: np.array of int}
    states : list of int - ordered state IDs
    target_n : int - desired number of meta-clusters

    Returns
    -------
    mapping : dict {old_id: new_id}
    new_states : list of int - sorted new state IDs (0..target_n-1)
    merge_info : dict {new_id: list of old_ids}
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    # Pool all sequences and compute raw transition count matrix
    state_to_idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n), dtype=float)
    for seq in processed_seqs.values():
        for a, b in zip(seq[:-1], seq[1:]):
            if a in state_to_idx and b in state_to_idx:
                counts[state_to_idx[a], state_to_idx[b]] += 1

    # Each cluster's profile = its row in the count matrix
    profiles = counts.copy()

    # Agglomerative clustering on profiles
    if n <= target_n:
        # Nothing to reduce
        mapping = {s: i for i, s in enumerate(states)}
        new_states = list(range(len(states)))
        merge_info = {i: [s] for i, s in enumerate(states)}
        return mapping, new_states, merge_info

    Z = linkage(profiles, method='ward')
    labels = fcluster(Z, t=target_n, criterion='maxclust')
    # labels is 1-indexed; convert to 0-indexed
    labels = labels - 1

    # Build mapping: old cluster ID -> new meta-cluster ID
    mapping = {}
    merge_info = {}
    for idx, old_id in enumerate(states):
        new_id = int(labels[idx])
        mapping[old_id] = new_id
        if new_id not in merge_info:
            merge_info[new_id] = []
        merge_info[new_id].append(old_id)

    new_states = sorted(merge_info.keys())
    return mapping, new_states, merge_info


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# TransitionVideoPreview
# ═══════════════════════════════════════════════════════════════════════════

class TransitionVideoPreview:
    """Multi-state video preview with per-frame probability graph.
    Adapted from SideBySidePreview for n-class transitions output."""

    # seaborn/tab10-style colors for states (index 0 = Other = grey)
    _STATE_COLORS_BGR = [
        (128, 128, 128),  # 0 = Other
        (214,  39,  40),  # 1
        ( 31, 119, 180),  # 2
        ( 44, 160,  44),  # 3
        (148, 103, 189),  # 4
        (140,  86,  75),  # 5
        (227, 119, 194),  # 6
        (188, 189,  34),  # 7
        ( 23, 190, 207),  # 8
        (255, 127,  14),  # 9
    ]
    _STATE_COLORS_HEX = [
        '#808080', '#d62728', '#1f77b4', '#2ca02c',
        '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#ff7f0e',
    ]

    def __init__(self, parent, video_path, state_seq, prob_matrix,
                 state_names, session_name):
        self.parent = parent
        self.video_path = video_path
        self.state_seq = np.asarray(state_seq, dtype=int)
        self.prob_matrix = prob_matrix  # (n_frames × n_classifiers) or None
        self.state_names = state_names  # index 0 = "Other"
        self.session_name = session_name

        try:
            import cv2 as _cv2
            self._cv2 = _cv2
        except ImportError:
            messagebox.showerror("Missing dependency",
                "OpenCV (cv2) is required for video preview.\n"
                "Install with: pip install opencv-python", parent=parent)
            return

        self.cap = self._cv2.VideoCapture(video_path)
        self.fps = self.cap.get(self._cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self._last_read_frame = -1
        self._canvas_image_id = None
        self.graph_window_obj = None
        self.graph_window_var = tk.IntVar(value=500)
        self.graph_redraw_counter = 0
        self.graph_redraw_interval = 5

        self._build_ui()
        self.window.after(100, self.update_frame)

    # ── UI build ──────────────────────────────────────────────────────
    def _build_ui(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Video Preview \u2014 {self.session_name}")
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        w, h = int(sw * 0.72), int(sh * 0.72)
        self.window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Controls row
        ctrl = ttk.Frame(self.window)
        ctrl.pack(fill='x', padx=6, pady=4)
        self.play_btn = ttk.Button(ctrl, text="\u25b6 Play", command=self.toggle_play)
        self.play_btn.pack(side='left', padx=2)
        ttk.Button(ctrl, text="\u23ee -100", command=lambda: self.jump(-100)).pack(side='left', padx=2)
        ttk.Button(ctrl, text="\u25c4 -10",  command=lambda: self.jump(-10)).pack(side='left', padx=2)
        ttk.Button(ctrl, text="\u25ba +10",  command=lambda: self.jump(10)).pack(side='left', padx=2)
        ttk.Button(ctrl, text="\u23ed +100", command=lambda: self.jump(100)).pack(side='left', padx=2)
        for lbl, spd in [("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("5x", 5.0)]:
            ttk.Button(ctrl, text=lbl, width=5,
                       command=lambda s=spd: self.set_speed(s)).pack(side='left', padx=1)
        if self.prob_matrix is not None:
            ttk.Button(ctrl, text="\U0001f4c8 Prob Graph",
                       command=self.open_graph_window).pack(side='right', padx=6)

        # Frame label
        self.frame_label = ttk.Label(ctrl, text="0 / 0")
        self.frame_label.pack(side='right', padx=8)

        # Video canvas
        self.canvas_video = tk.Canvas(self.window, bg='black')
        self.canvas_video.pack(fill='both', expand=True, padx=4, pady=2)

        # Scrub slider
        slider_row = ttk.Frame(self.window)
        slider_row.pack(fill='x', padx=6, pady=(0, 4))
        self.slider = ttk.Scale(slider_row, from_=0, to=max(0, self.total_frames - 1),
                                orient='horizontal', command=self._on_slider)
        self.slider.pack(fill='x', expand=True)

        # State color timeline bar
        self.timeline_canvas = tk.Canvas(self.window, height=16, bg='#eee')
        self.timeline_canvas.pack(fill='x', padx=6, pady=(0, 4))
        self.timeline_canvas.bind('<Button-1>', self._on_timeline_click)
        self.window.after(200, self._draw_timeline)

    # ── Frame display ─────────────────────────────────────────────────
    def update_frame(self):
        cv2 = self._cv2
        if self.current_frame != self._last_read_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        self._last_read_frame = self.current_frame

        if ret:
            # State lookup uses state_seq length (may be downsampled)
            n_state = len(self.state_seq)
            if n_state > 0 and self.total_frames > 0:
                fi_state = min(int(self.current_frame * n_state / self.total_frames), n_state - 1)
            else:
                fi_state = 0
            # Prob lookup uses prob_matrix length (always full-length)
            n_prob = len(self.prob_matrix) if self.prob_matrix is not None else 0
            if n_prob > 0 and self.total_frames > 0:
                fi_prob = min(int(self.current_frame * n_prob / self.total_frames), n_prob - 1)
            else:
                fi_prob = 0

            state_id = int(self.state_seq[fi_state])
            state_name = (self.state_names[state_id]
                          if state_id < len(self.state_names) else f"State {state_id}")
            color_bgr = self._get_color_bgr(state_id)

            # Overlay: state name
            cv2.putText(frame, state_name,
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color_bgr, 3)
            if self.prob_matrix is not None and fi_prob < n_prob:
                probs = self.prob_matrix[fi_prob]  # (n_classifiers,)
                argmax_ci = int(np.argmax(probs))   # 0-based index of highest-prob classifier
                for ci, p in enumerate(probs):
                    sname = (self.state_names[ci + 1]
                             if ci + 1 < len(self.state_names) else f"State {ci+1}")
                    is_assigned = (state_id == ci + 1)
                    is_best     = (ci == argmax_ci)
                    if is_assigned and is_best:
                        marker = '>'    # assigned AND highest prob (normal)
                    elif is_assigned:
                        marker = '<'    # assigned but NOT highest prob  (priority/gap-fill override)
                    elif is_best:
                        marker = '*'    # highest prob but NOT assigned
                    else:
                        marker = ' '
                    cv2.putText(frame, f"{marker} {sname}: {p:.2f}",
                                (20, 100 + ci * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                self._get_color_bgr(ci + 1), 2)
                # Legend - only shown when assigned ≠ argmax
                if state_id != 0 and state_id != argmax_ci + 1:
                    legend_y = 100 + len(probs) * 35 + 8
                    cv2.putText(frame, "< assigned   * highest prob",
                                (20, legend_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                (200, 200, 200), 1)
            cv2.putText(frame, f"Frame {self.current_frame}",
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self._show_frame(frame)

        self.slider.set(self.current_frame)
        self.frame_label.config(text=f"{self.current_frame} / {self.total_frames}")
        self._update_timeline_marker()
        self._maybe_update_graph()

    def _show_frame(self, frame):
        from PIL import Image, ImageTk
        cv2 = self._cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cw = self.canvas_video.winfo_width() or 640
        ch = self.canvas_video.winfo_height() or 480
        h, w = frame_rgb.shape[:2]
        aspect = w / h
        if cw / ch > aspect:
            nh, nw = ch, int(ch * aspect)
        else:
            nw, nh = cw, int(cw / aspect)
        frame_resized = cv2.resize(frame_rgb, (nw, nh))
        photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        x, y = (cw - nw) // 2, (ch - nh) // 2
        if self._canvas_image_id is None:
            self.canvas_video.delete('all')
            self._canvas_image_id = self.canvas_video.create_image(
                x, y, anchor='nw', image=photo)
        else:
            self.canvas_video.coords(self._canvas_image_id, x, y)
            self.canvas_video.itemconfig(self._canvas_image_id, image=photo)
        self.canvas_video.image = photo  # prevent GC

    # ── Timeline bar ──────────────────────────────────────────────────
    def _draw_timeline(self):
        self.timeline_canvas.delete('all')
        w = self.timeline_canvas.winfo_width() or 800
        n = len(self.state_seq)
        if n == 0 or w < 2:
            return
        ds = max(1, n // w)
        for i in range(0, n, ds):
            sid = int(self.state_seq[i])
            color = (self._STATE_COLORS_HEX[sid % len(self._STATE_COLORS_HEX)])
            x = int(i / n * w)
            self.timeline_canvas.create_line(x, 0, x, 16, fill=color, width=max(1, ds))
        self._update_timeline_marker()

    def _update_timeline_marker(self):
        self.timeline_canvas.delete('marker')
        w = self.timeline_canvas.winfo_width() or 800
        x = int(self.current_frame / self.total_frames * w) if self.total_frames > 0 else 0
        self.timeline_canvas.create_line(x, 0, x, 16, fill='white', width=2, tags='marker')

    def _on_timeline_click(self, event):
        w = self.timeline_canvas.winfo_width() or 800
        frame = int(event.x / w * self.total_frames)
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        self.update_frame()

    # ── Playback ──────────────────────────────────────────────────────
    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="\u23f8 Pause" if self.playing else "\u25b6 Play")
        if self.playing:
            self._play_loop()

    def _play_loop(self):
        if not self.playing:
            return
        if self.current_frame >= self.total_frames - 1:
            self.playing = False
            self.play_btn.config(text="\u25b6 Play")
            return
        self.current_frame += 1
        self.update_frame()
        delay = max(1, int(1000 / (self.fps * self.playback_speed)))
        self.window.after(delay, self._play_loop)

    def set_speed(self, spd):
        self.playback_speed = spd

    def jump(self, delta):
        self.current_frame = max(0, min(self.current_frame + delta, self.total_frames - 1))
        self.update_frame()

    def _on_slider(self, val):
        frame = int(float(val))
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_frame()

    # ── Probability graph window ──────────────────────────────────────
    def open_graph_window(self):
        if self.graph_window_obj and self.graph_window_obj.winfo_exists():
            self.graph_window_obj.lift()
            self._update_graph()
            return
        self.graph_window_obj = tk.Toplevel(self.window)
        self.graph_window_obj.title(f"Probability Graph \u2014 {self.session_name}")
        sw = self.graph_window_obj.winfo_screenwidth()
        sh = self.graph_window_obj.winfo_screenheight()
        w, h = int(sw * 0.75), int(sh * 0.55)
        self.graph_window_obj.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        ctrl = ttk.Frame(self.graph_window_obj)
        ctrl.pack(fill='x', padx=5, pady=4)
        ttk.Label(ctrl, text="Window:").pack(side='left', padx=2)
        ttk.Spinbox(ctrl, from_=100, to=10000, increment=100,
                    textvariable=self.graph_window_var, width=7).pack(side='left')
        ttk.Label(ctrl, text="frames").pack(side='left', padx=2)
        ttk.Button(ctrl, text="Refresh", command=self._update_graph).pack(side='left', padx=8)

        self.graph_frame_lbl = ttk.Label(ctrl, text="", width=18)
        self.graph_frame_lbl.pack(side='right', padx=6)
        self.graph_scrollbar = ttk.Scale(
            ctrl, from_=0, to=self.total_frames - 1,
            orient='horizontal', command=self._on_graph_scroll)
        self.graph_scrollbar.pack(side='right', fill='x', expand=True, padx=5)

        graph_frame = ttk.Frame(self.graph_window_obj)
        graph_frame.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.graph_fig = Figure(figsize=(12, 3.5), dpi=100, facecolor='white',
                                constrained_layout=True)
        self.graph_ax = self.graph_fig.add_subplot(111)
        self.graph_canvas_widget = FigureCanvasTkAgg(self.graph_fig, master=graph_frame)
        self.graph_canvas_widget.get_tk_widget().pack(fill='both', expand=True)
        self.graph_canvas_widget.mpl_connect('button_press_event', self._on_graph_click)
        _bind_tight_layout_on_resize(self.graph_canvas_widget, self.graph_fig)
        self._update_graph()

    def _update_graph(self):
        if not self.graph_window_obj or not self.graph_window_obj.winfo_exists():
            return
        if self.playing:
            self.graph_redraw_counter += 1
            if self.graph_redraw_counter < self.graph_redraw_interval:
                return
            self.graph_redraw_counter = 0

        try:
            self.graph_ax.clear()
            # Compute feature-frame index the same way as update_frame (proportional mapping)
            n_feat = len(self.prob_matrix)
            if n_feat > 0 and self.total_frames > 0:
                fi = min(int(self.current_frame * n_feat / self.total_frames), n_feat - 1)
            else:
                fi = 0

            half = self.graph_window_var.get() // 2
            feat_start = max(0, fi - half)
            feat_end   = min(n_feat, fi + half)
            feat_indices = np.arange(feat_start, feat_end)
            # Convert feature-frame indices → video-frame numbers for x-axis
            video_frames = (feat_indices * self.total_frames / n_feat).astype(int) if self.total_frames > 0 else feat_indices

            # State lookup MUST use the state-sequence length: state_seq may be
            # downsampled/shorter than prob_matrix, so indexing it with the
            # prob-scaled `fi` overruns it (crashing the redraw and freezing the
            # graph out of sync with the video). Mirror update_frame's mapping.
            n_state = len(self.state_seq)
            if n_state > 0 and self.total_frames > 0:
                fi_state = min(int(self.current_frame * n_state / self.total_frames),
                               n_state - 1)
            else:
                fi_state = 0
            current_state = int(self.state_seq[fi_state]) if n_state else 0

            n_clf = self.prob_matrix.shape[1]
            for ci in range(n_clf):
                sname = (self.state_names[ci + 1]
                         if ci + 1 < len(self.state_names) else f"State {ci+1}")
                color = (self._STATE_COLORS_HEX[(ci + 1) % len(self._STATE_COLORS_HEX)])
                lw = 2.5 if current_state == ci + 1 else 1.0
                alpha = 1.0 if current_state == ci + 1 else 0.5
                self.graph_ax.plot(video_frames, self.prob_matrix[feat_start:feat_end, ci],
                                   color=color, linewidth=lw, alpha=alpha,
                                   label=sname, zorder=3)

            self.graph_ax.axvline(x=self.current_frame, color='black',
                                  linewidth=2, linestyle='--',
                                  label='Current frame', zorder=4)
            self.graph_ax.set_xlabel('Frame (video)')
            self.graph_ax.set_ylabel('Probability')
            self.graph_ax.set_ylim(-0.05, 1.05)
            x_left  = int(video_frames[0])  if len(video_frames) else 0
            x_right = int(video_frames[-1]) if len(video_frames) else self.total_frames
            self.graph_ax.set_xlim(x_left, x_right)
            state_lbl = (self.state_names[current_state]
                         if current_state < len(self.state_names)
                         else f"State {current_state}")
            self.graph_ax.set_title(
                f"{self.session_name}  \u2014  Frame {self.current_frame}  \u2014  "
                f"Current state: {state_lbl}", fontsize=11)
            self.graph_ax.grid(True, alpha=0.3)
            self.graph_ax.legend(loc='upper right', fontsize=9, ncol=2)
            self.graph_canvas_widget.draw()

            if hasattr(self, 'graph_scrollbar'):
                self.graph_scrollbar.set(self.current_frame)
            if hasattr(self, 'graph_frame_lbl'):
                self.graph_frame_lbl.config(
                    text=f"{self.current_frame} / {self.total_frames}")
        except Exception as e:
            print(f"Graph update error: {e}")

    def _maybe_update_graph(self):
        if (self.graph_window_obj and self.graph_window_obj.winfo_exists()
                and self.prob_matrix is not None):
            self._update_graph()

    def _on_graph_click(self, event):
        if event.inaxes != self.graph_ax:
            return
        self.current_frame = max(0, min(int(event.xdata), self.total_frames - 1))
        self.update_frame()

    def _on_graph_scroll(self, val):
        frame = int(float(val))
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_frame()

    # ── Helpers ───────────────────────────────────────────────────────
    def _get_color_bgr(self, state_id):
        return self._STATE_COLORS_BGR[state_id % len(self._STATE_COLORS_BGR)]

    def _on_close(self):
        self.playing = False
        self.cap.release()
        if self.graph_window_obj and self.graph_window_obj.winfo_exists():
            self.graph_window_obj.destroy()
        self.window.destroy()


# TransitionsTab  (ttk.Frame)
# ═══════════════════════════════════════════════════════════════════════════

class TransitionsTab(ttk.Frame):

    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui

        # Internal state
        self._state_seqs = {}       # {session_name: np.array of int} - analysis-time, may be capped
        self._state_seqs_full = {}  # {session_name: np.array of int} - uncapped, for UI 1:1 lookup
        self._states = []           # ordered state IDs
        self._state_labels = {}     # {state_id: user label}  e.g. {0: "Still"}
        self._matrices = {}         # {session_name: (matrix, states)}
        self._windowed = {}         # {session_name: [(t, mat), ...]}
        self._group_matrices = {}   # {group: mean_matrix}
        self._group_sem = {}        # {group: sem_matrix}
        self._group_subject_matrices = {}  # {group: [matrix, ...]} - individual subjects
        self._frame_probs = {}  # {session_name: np.ndarray (n_frames × n_classifiers)}
        self._key_df = None
        self._session_subjects = {} # {session_name: subject}
        self._merge_info = None     # {new_id: [old_ids]} from cluster reduction
        self._model_bundle = None   # full model.pkl contents for summary view
        self._worker_thread = None
        self._stop_event = threading.Event()

        # Results graph window (pop-out, Analysis-style). `_fig`/`_canvas` are the
        # ACTIVE render target and are swapped between the inline and window figures.
        self._graph_win = None
        self._win_fig = None
        self._win_canvas = None
        self._inline_fig = None
        self._inline_canvas = None
        # Group ordering / per-treatment colors for grouped views (None → dose order).
        self._group_order = None
        self._group_colors = {}
        self._loading_session = False   # True while restoring (suppresses auto-save)
        self._saved_items = []          # [(label, path)] for the saved-session combo
        # Per-view axis limits, e.g. {'Temporal Probability': {'ymin':0,'ymax':1}}.
        self._axis_limits = {}

        # Latent state discovery (LUPE method)
        self._latent_centroids = None    # (k, n_states, n_states)
        self._session_latent_map = {}    # {session: [latent_state_id per window]}
        self._n_latent = 0
        self._occupancy = {}             # {session: array(k,)} fractional occupancy
        self._pca_model = None
        self._pca_scores = {}            # {session: (pc1, pc2)}
        self._pca_loadings = None
        self._temporal_probs = {}        # {session: (time_centers, prob_matrix)}
        self._group_occupancy = {}       # {treatment: mean_array}
        self._group_occupancy_sem = {}   # {treatment: sem_array}

        # Supervised prediction state
        self._loaded_classifiers = []   # list of clf_data dicts
        self._priority_order = []       # indices into _loaded_classifiers, in priority order
        self._trans_sessions = []       # list of session dicts from find_session_triplets
        self._trans_session_checked = {}  # {session_name: BooleanVar}
        self._pending_session_selection = None  # set by _load_config, consumed by _scan_trans_sessions
        self._effective_fps = 25.0
        self._assign_mode = tk.StringVar(value='priority')
        # 'inferno' is in _SEQUENTIAL (the heatmap whitelist) and matches
        # the project's house palette. Pre-2026-05-01 the default was
        # 'deep' (a discrete palette), which silently fell through to
        # YlOrRd via _heatmap_cmap's else-branch.
        # `_palette_var` is the STATE palette (categorical/sequential sampled to
        # one color per state); `_heat_palette_var` is the sequential palette for
        # probability heatmaps. Split so a categorical pick never silently no-ops
        # on a heatmap (old code fell through to 'YlOrRd').
        self._palette_var  = tk.StringVar(value='inferno')
        self._bot_palette_var = tk.StringVar(value='tab10')  # legacy; unused
        self._heat_palette_var = tk.StringVar(value='inferno')
        self._show_annot_var = tk.BooleanVar(value=True)
        self._show_sig_var = tk.BooleanVar(value=False)
        # Real time (True) shows the actual recording clock on time-axis views; when
        # False the axis is re-based so the display window starts at 0.
        self._realtime_axis_var = tk.BooleanVar(value=True)
        # Ethogram display bin (s): 0 = full native resolution; raise it to coarsen
        # the raster (each column = majority state over that many seconds). Kept
        # separate from the fraction-plot bin so the ethogram stays high-fidelity
        # by default.
        self._etho_bin_var = tk.DoubleVar(value=0)
        self._transition_mode = tk.StringVar(value='bout')
        # Comparison controls (graph window)
        self._sort_mode_var = tk.StringVar(value='Usage ↓')
        self._norm_mode_var = tk.StringVar(value='Row-normalized')
        self._ref_group_var = tk.StringVar(value='(auto)')    # (auto) = vehicle
        self._group_palette_mode_var = tk.StringVar(value='categorical')
        # Statistics (configurable; Auto adapts to the design)
        self._stat_test_var = tk.StringVar(value='Auto')
        self._stat_correction_var = tk.StringVar(value='BH-FDR')
        self._stat_alpha_var = tk.StringVar(value='0.05')
        self._error_mode_var = tk.StringVar(value='SEM')      # or 95% CI (bootstrap)
        # Behavior × Group view controls
        self._behav_var = tk.StringVar(value='All behaviors (grid)')
        self._trend_tmin_var = tk.StringVar(value='')         # display window (min)
        self._trend_tmax_var = tk.StringVar(value='')

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        # Scrollable container
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)
        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        sf = self._scroll_frame

        # Title
        ttk.Label(sf, text="Behavioral State Transitions",
                  font=(FONT_FAMILY, 14, 'bold')).pack(anchor='w', padx=20, pady=(15, 5))
        ttk.Label(sf, text="Compute transition probabilities between behavioral "
                  "states and compare across treatment groups.",
                  wraplength=700).pack(anchor='w', padx=20, pady=(0, 10))

        # ── One-click preset ──────────────────────────────────────────
        preset_frame = ttk.Frame(sf)
        preset_frame.pack(fill='x', padx=20, pady=(0, 6))
        ttk.Label(preset_frame, text="Preset:").pack(side='left')
        self._preset_source_var = tk.StringVar(value='')
        preset_combo = ttk.Combobox(
            preset_frame, textvariable=self._preset_source_var, state='readonly',
            width=30, values=[
                'Project + encyclopedia (deduped)',
                'Encyclopedia only',
                'Project only',
            ])
        preset_combo.pack(side='left', padx=(6, 0))
        preset_combo.set('Select classifier set…')
        preset_combo.bind('<<ComboboxSelected>>', self._preset_load_from_dropdown)
        self._tip(preset_combo, 'preset_source')
        ttk.Label(preset_frame,
                  text="Loads classifiers on all sessions - then click Compute.",
                  foreground='gray').pack(side='left', padx=10)

        # ── State from classifiers ────────────────────────────────────
        # Supervised (trained-classifier) path only - the unsupervised/LUPE path is retired
        # for this release. _unsup_frame/_run_combo are still created (unpacked) so any legacy
        # references remain valid.
        src_frame = ttk.LabelFrame(sf, text="State from classifiers", padding=10)
        src_frame.pack(fill='x', padx=20, pady=5)

        self._source_var = tk.StringVar(value='supervised')
        self._unsup_frame = ttk.Frame(src_frame)
        self._run_combo = ttk.Combobox(self._unsup_frame, state='readonly', width=30)

        self._sup_frame = ttk.Frame(src_frame)
        self._sup_frame.pack(fill='both', expand=True)

        # -- Classifiers sub-section --
        clf_section = ttk.LabelFrame(self._sup_frame, text="Classifiers", padding=5)
        clf_section.pack(fill='x', pady=(5, 3))

        clf_list_frame = ttk.Frame(clf_section)
        clf_list_frame.pack(fill='x')
        self._clf_listbox = tk.Listbox(clf_list_frame, height=4, selectmode='single',
                                       font=('Courier', 8), width=80)
        _clf_scroll = ttk.Scrollbar(clf_list_frame, command=self._clf_listbox.yview)
        _clf_hscroll = ttk.Scrollbar(clf_list_frame, orient='horizontal',
                                     command=self._clf_listbox.xview)
        self._clf_listbox.configure(yscrollcommand=_clf_scroll.set,
                                    xscrollcommand=_clf_hscroll.set)
        self._clf_listbox.pack(side='left', fill='both', expand=True)
        _clf_scroll.pack(side='right', fill='y')
        _clf_hscroll.pack(side='bottom', fill='x')
        self._clf_listbox.bind('<Double-ButtonRelease-1>',
                               lambda e: self._edit_clf_settings())

        # Pick-from-project/encyclopedia dropdown + Add (mirrors the Run Classifiers tab).
        pick_row = ttk.Frame(clf_section)
        pick_row.pack(fill='x', pady=(3, 0))
        ttk.Label(pick_row, text="Classifier:").pack(side='left')
        self._clf_pick_options = {}   # {display: path}
        self._clf_pick_var = tk.StringVar()
        self._clf_pick_combo = ttk.Combobox(pick_row, textvariable=self._clf_pick_var,
                                            state='readonly', width=40)
        self._clf_pick_combo.pack(side='left', fill='x', expand=True, padx=5)
        self._tip(self._clf_pick_combo, 'add_classifier')
        ttk.Button(pick_row, text="🔄", width=3,
                   command=self._refresh_clf_pick_options).pack(side='left', padx=2)
        ttk.Button(pick_row, text="➕ Add",
                   command=self._add_selected_classifier).pack(side='left', padx=2)

        clf_btn_row = ttk.Frame(clf_section)
        clf_btn_row.pack(fill='x', pady=(3, 0))
        ttk.Button(clf_btn_row, text="📁 Browse…",
                   command=self._add_classifier).pack(side='left', padx=(0, 5))
        ttk.Button(clf_btn_row, text="Remove",
                   command=self._remove_classifier).pack(side='left')
        ttk.Button(clf_btn_row, text="Edit Settings",
                   command=self._edit_clf_settings).pack(side='left', padx=5)
        self._refresh_clf_pick_options()

        # -- Sessions sub-section --
        sess_section = ttk.LabelFrame(self._sup_frame, text="Sessions", padding=5)
        sess_section.pack(fill='x', pady=(3, 3))

        sess_btn_row = ttk.Frame(sess_section)
        sess_btn_row.pack(fill='x', pady=(0, 3))
        ttk.Button(sess_btn_row, text="Refresh Sessions",
                   command=self._scan_trans_sessions).pack(side='left', padx=(0, 5))
        ttk.Button(sess_btn_row, text="Select All",
                   command=self._trans_select_all).pack(side='left', padx=(0, 5))
        ttk.Button(sess_btn_row, text="Deselect All",
                   command=self._trans_deselect_all).pack(side='left')

        # -- Key file (auto-found on project open; drives the Group column
        #    and all group-comparison views) --
        kf_row = ttk.Frame(sess_section)
        kf_row.pack(fill='x', pady=(0, 3))
        kf_lbl = ttk.Label(kf_row, text="Key file:")
        kf_lbl.pack(side='left', padx=(0, 4))
        self._tip(kf_lbl, 'key_file')
        self._key_file_var = tk.StringVar()
        kf_entry = ttk.Entry(kf_row, textvariable=self._key_file_var, width=38)
        kf_entry.pack(side='left', padx=(0, 5))
        self._tip(kf_entry, 'key_file')
        kf_auto = ttk.Button(kf_row, text="Auto-find",
                             command=self._autofind_key_file)
        kf_auto.pack(side='left', padx=2)
        self._tip(kf_auto, 'key_autofind')
        ttk.Button(kf_row, text="Browse",
                   command=self._browse_key_file).pack(side='left', padx=2)
        ttk.Button(kf_row, text="Load",
                   command=self._load_key_file).pack(side='left', padx=2)
        self._key_status = ttk.Label(sess_section, text="No key file loaded",
                                     foreground='gray')
        self._key_status.pack(anchor='w', pady=(0, 3))

        trans_tree_frame = ttk.Frame(sess_section)
        trans_tree_frame.pack(fill='x', pady=(0, 3))
        self._trans_tree = ttk.Treeview(
            trans_tree_frame,
            columns=("check", "session", "group", "video"),
            show="headings",
            selectmode="none",
            height=6,
        )
        self._trans_tree.heading("check", text="✓")
        self._trans_tree.heading("session", text="Session Name")
        self._trans_tree.heading("group", text="Group")
        self._trans_tree.heading("video", text="Video")
        self._trans_tree.column("check", width=30, anchor="center", stretch=False)
        self._trans_tree.column("session", width=230, anchor="w")
        self._trans_tree.column("group", width=90, anchor="w")
        self._trans_tree.column("video", width=220, anchor="w")
        _trans_tree_sb = ttk.Scrollbar(trans_tree_frame, orient='vertical',
                                       command=self._trans_tree.yview)
        self._trans_tree.configure(yscrollcommand=_trans_tree_sb.set)
        self._trans_tree.pack(side='left', fill='x', expand=True)
        _trans_tree_sb.pack(side='right', fill='y')
        self._trans_tree.bind("<ButtonRelease-1>", self._on_trans_tree_click)

        # -- State assignment mode --
        assign_section = ttk.LabelFrame(self._sup_frame, text="State Assignment",
                                        padding=5)
        assign_section.pack(fill='x', pady=(3, 3))

        assign_row = ttk.Frame(assign_section)
        assign_row.pack(fill='x')
        _rb_prio = ttk.Radiobutton(assign_row, text="Priority ranking",
                        variable=self._assign_mode, value='priority',
                        command=self._toggle_assign_mode)
        _rb_prio.pack(side='left', padx=(0, 15))
        self._tip(_rb_prio, 'assign_priority')
        _rb_argmax = ttk.Radiobutton(assign_row, text="Best wins (argmax)",
                        variable=self._assign_mode, value='argmax',
                        command=self._toggle_assign_mode)
        _rb_argmax.pack(side='left')
        self._tip(_rb_argmax, 'assign_argmax')

        self._priority_frame = ttk.Frame(assign_section)
        self._priority_frame.pack(fill='x', pady=(3, 0))
        ttk.Label(self._priority_frame,
                  text="Highest priority wins when multiple classifiers fire",
                  foreground='gray').pack(anchor='w', padx=(20, 0))
        prio_list_row = ttk.Frame(self._priority_frame)
        prio_list_row.pack(fill='x', padx=(20, 0), pady=(3, 0))
        self._priority_listbox = tk.Listbox(prio_list_row, height=4,
                                            selectmode='single',
                                            font=('Courier', 8))
        self._priority_listbox.pack(side='left', fill='x', expand=True)
        self._tip(self._priority_listbox, 'priority_list')
        prio_btn_col = ttk.Frame(prio_list_row)
        prio_btn_col.pack(side='left', padx=(5, 0))
        ttk.Button(prio_btn_col, text="Up",
                   command=self._move_priority_up).pack(fill='x', pady=(0, 3))
        ttk.Button(prio_btn_col, text="Down",
                   command=self._move_priority_down).pack(fill='x')

        # -- Transition mode --
        tmode_section = ttk.LabelFrame(self._sup_frame, text="Transition Mode",
                                       padding=5)
        tmode_section.pack(fill='x', pady=(3, 5))
        tmode_row = ttk.Frame(tmode_section)
        tmode_row.pack(fill='x')
        ttk.Radiobutton(tmode_row, text="Per-bout",
                        variable=self._transition_mode, value='bout').pack(
                            side='left', padx=(0, 15))
        ttk.Radiobutton(tmode_row, text="Per-frame (LUPE style)",
                        variable=self._transition_mode, value='frame').pack(
                            side='left')

        # Legacy supervised vars (for fallback _load_supervised_states path)
        self._results_var = tk.StringVar()
        self._behavior_list_frame = ttk.Frame(src_frame)
        self._behavior_vars = {}  # {name: BooleanVar}

        # ── Cluster Labels (optional rename table) ────────────────────
        lbl_frame = ttk.LabelFrame(sf, text="Cluster Labels (optional)", padding=10)
        lbl_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(lbl_frame,
                  text="Rename clusters for readability. Leave blank to keep default names.",
                  wraplength=600).pack(anchor='w')
        self._label_table_frame = ttk.Frame(lbl_frame)
        self._label_table_frame.pack(fill='x', pady=5)
        self._label_entries = {}  # {state_id: Entry widget}

        # ── Settings ──────────────────────────────────────────────────
        set_frame = ttk.LabelFrame(sf, text="Settings", padding=10)
        set_frame.pack(fill='x', padx=20, pady=5)

        row = 0
        ttk.Label(set_frame, text="FPS:").grid(row=row, column=0, sticky='w')
        self._fps_var = tk.IntVar(value=60)
        ttk.Spinbox(set_frame, from_=1, to=240, textvariable=self._fps_var,
                     width=6).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Button(set_frame, text="Auto-detect",
                   command=self._auto_detect_fps).grid(row=row, column=2, sticky='w', padx=5)

        row += 1
        self._downsample_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(set_frame, text="Downsample to 20 Hz - mode of every N frames (LUPE)",
                        variable=self._downsample_var).grid(
            row=row, column=0, columnspan=3, sticky='w', pady=(2, 0))

        row += 1
        _sm_lbl = ttk.Label(set_frame, text="Min state duration (ms):")
        _sm_lbl.grid(row=row, column=0, sticky='w')
        self._tip(_sm_lbl, 'smooth_ms')
        self._smooth_ms_var = tk.IntVar(value=100)
        _sm_sp = ttk.Spinbox(set_frame, from_=0, to=5000, increment=50,
                     textvariable=self._smooth_ms_var, width=6)
        _sm_sp.grid(row=row, column=1, sticky='w', padx=5)
        self._tip(_sm_sp, 'smooth_ms')

        # Exclude-noise (unsupervised) retired: keep the var, no control.
        self._exclude_noise_var = tk.BooleanVar(value=False)

        row += 1
        self._exclude_other_var = tk.BooleanVar(value=False)
        _exo_cb = ttk.Checkbutton(set_frame, text="Exclude 'Other' state (frames with no behavior) from analysis",
                        variable=self._exclude_other_var)
        _exo_cb.grid(row=row, column=0, columnspan=3, sticky='w')
        self._tip(_exo_cb, 'exclude_other')

        row += 1
        self._zero_diag_var = tk.BooleanVar(value=False)
        _zero_cb = ttk.Checkbutton(set_frame, text="Zero self-transitions on diagonal",
                        variable=self._zero_diag_var)
        _zero_cb.grid(row=row, column=0, columnspan=3, sticky='w')
        self._tip(_zero_cb, 'zero_diag')

        row += 1
        self._cache_only_var = tk.BooleanVar(value=False)
        _cache_cb = ttk.Checkbutton(set_frame,
                        text="Use cached features only (skip sessions needing video re-extraction)",
                        variable=self._cache_only_var)
        _cache_cb.grid(row=row, column=0, columnspan=3, sticky='w')
        self._tip(_cache_cb, 'cache_only')

        # Cluster reduction (unsupervised) retired: keep vars/frame, no controls.
        self._reduce_clusters_var = tk.BooleanVar(value=False)
        self._target_clusters_var = tk.IntVar(value=10)
        self._reduce_sub_frame = ttk.Frame(set_frame)
        self._reduce_status_label = ttk.Label(self._reduce_sub_frame, text="")

        row += 1
        _pb_lbl = ttk.Label(set_frame, text="Temporal prob bin (s):")
        _pb_lbl.grid(row=row, column=0, sticky='w')
        self._tip(_pb_lbl, 'prob_bin')
        self._prob_bin_var = tk.DoubleVar(value=30)
        _pb_sp = ttk.Spinbox(set_frame, from_=5, to=600, increment=5,
                     textvariable=self._prob_bin_var, width=6)
        _pb_sp.grid(row=row, column=1, sticky='w', padx=5)
        self._tip(_pb_sp, 'prob_bin')

        # ── Time Windows ──────────────────────────────────────────────
        tw_frame = ttk.LabelFrame(sf, text="Time Windows", padding=10)
        tw_frame.pack(fill='x', padx=20, pady=5)

        self._time_mode_var = tk.StringVar(value='sliding')
        _rb_full = ttk.Radiobutton(tw_frame, text="Full session",
                        variable=self._time_mode_var, value='full',
                        command=self._toggle_time)
        _rb_full.grid(row=0, column=0, sticky='w')
        self._tip(_rb_full, 'full_session')
        _rb_range = ttk.Radiobutton(tw_frame, text="Time range",
                        variable=self._time_mode_var, value='range',
                        command=self._toggle_time)
        _rb_range.grid(row=1, column=0, sticky='w')
        self._tip(_rb_range, 'time_range')
        _rb_slide = ttk.Radiobutton(tw_frame, text="Sliding windows",
                        variable=self._time_mode_var, value='sliding',
                        command=self._toggle_time)
        _rb_slide.grid(row=2, column=0, sticky='w')
        self._tip(_rb_slide, 'sliding_windows')

        # Range sub-frame
        self._range_frame = ttk.Frame(tw_frame)
        _lbl_start = ttk.Label(self._range_frame, text="Start (s):")
        _lbl_start.pack(side='left', padx=(20, 2))
        self._tip(_lbl_start, 'range_start')
        self._range_start_var = tk.DoubleVar(value=0)
        _sp_start = ttk.Spinbox(self._range_frame, from_=0, to=99999, increment=30,
                     textvariable=self._range_start_var, width=8)
        _sp_start.pack(side='left', padx=2)
        self._tip(_sp_start, 'range_start')
        _lbl_end = ttk.Label(self._range_frame, text="End (s):")
        _lbl_end.pack(side='left', padx=(10, 2))
        self._tip(_lbl_end, 'range_end')
        self._range_end_var = tk.DoubleVar(value=1800)
        _sp_end = ttk.Spinbox(self._range_frame, from_=0, to=99999, increment=30,
                     textvariable=self._range_end_var, width=8)
        _sp_end.pack(side='left', padx=2)
        self._tip(_sp_end, 'range_end')

        # Sliding sub-frame
        self._sliding_frame = ttk.Frame(tw_frame)
        _lbl_win = ttk.Label(self._sliding_frame, text="Window (s):")
        _lbl_win.pack(side='left', padx=(20, 2))
        self._tip(_lbl_win, 'window_sec')
        self._win_sec_var = tk.DoubleVar(value=30)
        _sp_win = ttk.Spinbox(self._sliding_frame, from_=1, to=600, increment=10,
                     textvariable=self._win_sec_var, width=8)
        _sp_win.pack(side='left', padx=2)
        self._tip(_sp_win, 'window_sec')
        _lbl_step = ttk.Label(self._sliding_frame, text="Step (s):")
        _lbl_step.pack(side='left', padx=(10, 2))
        self._tip(_lbl_step, 'step_sec')
        self._step_sec_var = tk.DoubleVar(value=10)
        _sp_step = ttk.Spinbox(self._sliding_frame, from_=1, to=600, increment=10,
                     textvariable=self._step_sec_var, width=8)
        _sp_step.pack(side='left', padx=2)
        self._tip(_sp_step, 'step_sec')

        # Row 3: optional duration cap
        self._dur_limit_var = tk.BooleanVar(value=False)
        self._dur_limit_min = tk.DoubleVar(value=30.0)
        dur_row = ttk.Frame(tw_frame)
        dur_row.grid(row=3, column=0, columnspan=3, sticky='w', pady=(4, 0))
        _crop_cb = ttk.Checkbutton(dur_row, text="Crop to first",
                        variable=self._dur_limit_var)
        _crop_cb.pack(side='left')
        self._tip(_crop_cb, 'crop_first')
        _crop_sp = ttk.Spinbox(dur_row, from_=1, to=9999, increment=5,
                    textvariable=self._dur_limit_min, width=6)
        _crop_sp.pack(side='left', padx=2)
        self._tip(_crop_sp, 'crop_first')
        ttk.Label(dur_row, text="min").pack(side='left')

        # (The former "Behavior Over Time palette" control was removed - all
        # state-colored views now share the single State-colors palette.)

        # Latent State Discovery (LUPE, unsupervised) retired for this release - keep the vars.
        self._discover_latent_var = tk.BooleanVar(value=False)
        self._n_latent_var = tk.IntVar(value=6)

        # ── Run ──────────────────────────────────────────────────────
        run_frame = ttk.LabelFrame(sf, text="Run", padding=10)
        run_frame.pack(fill='x', padx=20, pady=5)

        btn_row = ttk.Frame(run_frame)
        btn_row.pack(fill='x')
        _compute_btn = ttk.Button(btn_row, text="Compute Transitions",
                   command=self._start_compute)
        _compute_btn.pack(side='left', padx=5)
        self._tip(_compute_btn, 'compute')
        self._stop_btn = ttk.Button(btn_row, text="\u25a0  Stop",
                                    command=self._stop_compute, state='disabled')
        self._stop_btn.pack(side='left', padx=2)
        ttk.Separator(btn_row, orient='vertical').pack(side='left', fill='y', padx=8)
        _save_sess_btn = ttk.Button(btn_row, text="💾 Save Session",
                                    command=self._save_session)
        _save_sess_btn.pack(side='left', padx=2)
        self._tip(_save_sess_btn, 'save_session')
        _load_sess_btn = ttk.Button(btn_row, text="📂 Load Session",
                                    command=self._load_session)
        _load_sess_btn.pack(side='left', padx=2)
        self._tip(_load_sess_btn, 'load_session')
        ttk.Separator(btn_row, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(btn_row, text="Save Config",
                   command=self._save_config).pack(side='left', padx=2)
        ttk.Button(btn_row, text="Load Config",
                   command=self._load_config).pack(side='left', padx=2)
        self._progress = ttk.Progressbar(btn_row, mode='indeterminate', length=200)
        self._progress.pack(side='left', padx=10)

        self._log = scrolledtext.ScrolledText(run_frame, height=6, state='disabled',
                                              wrap='word')
        self._log.pack(fill='x', pady=(5, 0))

        # ── Results ──────────────────────────────────────────────────
        res_frame = ttk.LabelFrame(sf, text="Results", padding=10)
        res_frame.pack(fill='both', padx=20, pady=5, expand=True)

        # Saved sessions - pick a previous run and reload it (no recompute).
        saved_row = ttk.Frame(res_frame)
        saved_row.pack(fill='x', pady=(0, 5))
        ttk.Label(saved_row, text="Saved sessions:").pack(side='left', padx=(0, 5))
        self._saved_combo = ttk.Combobox(saved_row, state='readonly', width=34, values=[])
        self._saved_combo.pack(side='left', padx=(0, 4))
        self._saved_load_btn = ttk.Button(saved_row, text="Load",
                                          command=self._load_selected_session,
                                          state='disabled')
        self._saved_load_btn.pack(side='left', padx=2)
        ttk.Button(saved_row, text="Save…",
                   command=self._save_current_session_named).pack(side='left', padx=2)
        self._saved_del_btn = ttk.Button(saved_row, text="Delete",
                                         command=self._delete_selected_session,
                                         state='disabled')
        self._saved_del_btn.pack(side='left', padx=2)

        # View selector
        view_row = ttk.Frame(res_frame)
        view_row.pack(fill='x', pady=(0, 5))
        _open_gw_btn = ttk.Button(view_row, text="📈 Open Graph Window",
                                  command=self._open_graph_window)
        _open_gw_btn.pack(side='left', padx=(0, 10))
        self._tip(_open_gw_btn, 'open_graph_window')
        ttk.Label(view_row, text="View:").pack(side='left', padx=(0, 5))
        self._view_var = tk.StringVar(value='State Usage')
        self._last_view = 'State Usage'
        self._view_combo = ttk.Combobox(
            view_row, textvariable=self._view_var, state='readonly', width=22,
            values=list(_VIEW_MENU))
        self._view_combo.pack(side='left', padx=5)
        self._view_combo.bind('<<ComboboxSelected>>', self._on_view_changed)
        self._tip(self._view_combo, 'view_combo')

        ttk.Label(view_row, text="Palette:").pack(side='left', padx=(10, 2))
        self._palette_combo = ttk.Combobox(
            view_row, textvariable=self._palette_var, state='readonly', width=12,
            values=['deep', 'colorblind', 'muted', 'bright', 'dark',
                    'tab10', 'tab20', 'Set1', 'Set2', 'Dark2', 'Paired',
                    'magma', 'plasma', 'viridis', 'inferno', 'cividis', 'turbo'])
        self._palette_combo.pack(side='left', padx=2)
        self._palette_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())
        self._tip(self._palette_combo, 'heat_palette')

        ttk.Checkbutton(view_row, text="Cell values",
                        variable=self._show_annot_var,
                        command=self._refresh_plot).pack(side='left', padx=(8, 2))
        self._sig_cb = ttk.Checkbutton(view_row, text="Sig. markers",
                                       variable=self._show_sig_var,
                                       command=self._refresh_plot)
        self._sig_cb.pack(side='left', padx=(8, 2))

        # Video preview controls
        self._preview_session_var = tk.StringVar()
        self._preview_session_combo = ttk.Combobox(
            view_row, textvariable=self._preview_session_var,
            state='readonly', width=20)
        self._preview_session_combo.pack(side='left', padx=(16, 2))
        ttk.Button(view_row, text="\u25b6 Video Preview",
                   command=self._open_video_preview).pack(side='left', padx=(2, 4))

        # State pair selector (for timeline view)
        self._pair_frame = ttk.Frame(view_row)
        ttk.Label(self._pair_frame, text="Transition:").pack(side='left', padx=(10, 2))
        self._pair_combo = ttk.Combobox(self._pair_frame, state='readonly', width=25)
        self._pair_combo.pack(side='left', padx=2)
        self._pair_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        # The graphs live in the pop-out Graph Window (opened automatically after
        # Compute / load). The main page shows a stats TABLE instead of a cramped
        # inline plot. `_fig`/`_canvas` are still created as the ACTIVE render
        # target (swapped to the window's figure while it is open), but the inline
        # canvas is NOT packed - it renders off-screen when the window is closed.
        self._fig = plt.figure(figsize=(9, 5), constrained_layout=True) if MATPLOTLIB_AVAILABLE else None
        self._canvas = None
        if MATPLOTLIB_AVAILABLE:
            self._canvas = FigureCanvasTkAgg(self._fig, master=res_frame)  # not packed
            self._inline_fig = self._fig
            self._inline_canvas = self._canvas

        # Stats summary table (per-behavior % time by group; graphs in the window).
        ttk.Label(res_frame,
                  text="Per-behavior % time by group (mean ± SEM). Graphs open in "
                       "the Graph Window.",
                  foreground='gray').pack(anchor='w', pady=(2, 2))
        _stats_holder = ttk.Frame(res_frame)
        _stats_holder.pack(fill='both', expand=True)
        self._stats_tree = ttk.Treeview(_stats_holder, show='headings', height=10)
        _stats_vsb = ttk.Scrollbar(_stats_holder, orient='vertical',
                                   command=self._stats_tree.yview)
        self._stats_tree.configure(yscrollcommand=_stats_vsb.set)
        self._stats_tree.pack(side='left', fill='both', expand=True)
        _stats_vsb.pack(side='right', fill='y')

        # ── Export ───────────────────────────────────────────────────
        exp_frame = ttk.LabelFrame(sf, text="Export", padding=10)
        exp_frame.pack(fill='x', padx=20, pady=(5, 20))

        ttk.Button(exp_frame, text="Export Matrices (CSV)",
                   command=self._export_matrices).pack(side='left', padx=5)
        ttk.Button(exp_frame, text="Export State Sequences (CSV)",
                   command=self._export_sequences).pack(side='left', padx=5)
        ttk.Button(exp_frame, text="Export Figure (PNG)",
                   command=lambda: self._export_figure('png')).pack(side='left', padx=5)
        ttk.Button(exp_frame, text="Export Figure (PDF)",
                   command=lambda: self._export_figure('pdf')).pack(side='left', padx=5)

        self._toggle_time()

        # Refresh the saved-session dropdown when the user opens this tab (not on
        # project open). SidebarNav.bind appends, so this is non-destructive.
        try:
            self.app.notebook.bind('<<NotebookTabChanged>>', self._on_tab_shown)
        except Exception:
            pass
        try:
            self._refresh_saved_sessions()
        except Exception:
            pass

    def _on_tab_shown(self, event=None):
        """Fires on any main-tab change; refresh the saved-session dropdown when
        Transitions becomes active (no more reload prompt)."""
        try:
            if self.app.notebook.select() != "🔀 Transitions":
                return
        except Exception:
            return
        self._refresh_saved_sessions()

    # ------------------------------------------------------------------
    # Toggle helpers
    # ------------------------------------------------------------------

    def _toggle_source(self):
        if self._source_var.get() == 'unsupervised':
            self._unsup_frame.grid(row=2, column=0, columnspan=3, sticky='ew',
                                   pady=(5, 0))
            self._sup_frame.grid_forget()
        else:
            self._unsup_frame.grid_forget()
            self._sup_frame.grid(row=2, column=0, columnspan=3, sticky='ew',
                                 pady=(5, 0))

    def _toggle_reduce(self):
        if self._reduce_clusters_var.get():
            self._reduce_sub_frame.grid()
        else:
            self._reduce_sub_frame.grid_remove()

    def _toggle_assign_mode(self):
        if self._assign_mode.get() == 'priority':
            self._priority_frame.pack(fill='x', pady=(3, 0))
        else:
            self._priority_frame.pack_forget()

    # ------------------------------------------------------------------
    # Supervised classifier helpers
    # ------------------------------------------------------------------

    def _add_classifier_path(self, path):
        """Load one classifier .pkl and append it to the list. Returns the loaded behavior name
        (or None on failure). Skips duplicates already in the list by path."""
        if any(cd.get('_path') == path for cd in self._loaded_classifiers):
            return None
        try:
            clf_data = _robust_load(path)
            if 'clf_model' not in clf_data:
                self._log_msg(f"Skipping {os.path.basename(path)}: no clf_model key")
                return None
            clf_data['_path'] = path
            # Normalize Behavior_type to a string (some classifiers store a list of behaviors)
            # so all downstream uses (listbox, state labels, preview) are safe.
            _bt = clf_data.get('Behavior_type')
            if _bt is not None and not isinstance(_bt, str):
                clf_data['Behavior_type'] = ('+'.join(map(str, _bt))
                                             if isinstance(_bt, (list, tuple)) else str(_bt))
            self._loaded_classifiers.append(clf_data)
            self._priority_order.append(len(self._loaded_classifiers) - 1)
            beh = clf_data.get('Behavior_type', '?')
            self._log_msg(f"Loaded: {beh} from {os.path.basename(path)}")
            return beh
        except Exception as e:
            self._log_msg(f"Error loading {os.path.basename(path)}: {e}")
            return None

    def _add_classifier(self):
        folder = self.app.current_project_folder.get()
        clf_dir = os.path.join(folder, 'classifiers') if folder else None
        paths = filedialog.askopenfilenames(
            title="Select Classifier .pkl file(s)",
            initialdir=clf_dir if clf_dir and os.path.isdir(clf_dir) else folder,
            filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        for path in paths:
            self._add_classifier_path(path)
        self._update_clf_listbox()
        self._update_priority_listbox()

    # ------------------------------------------------------------------
    # Classifier discovery (project + global + encyclopedia) + preset
    # ------------------------------------------------------------------
    def _encyclopedia_dirs(self):
        """Candidate 'classifier encyclopedia' folders (shipped next to the app / near the
        global classifiers folder)."""
        cands = [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'pixelpaws_global_classifier_encyclopedia')]
        try:
            from user_config import get_global_classifiers_folder
            gcf = get_global_classifiers_folder()
            if gcf:
                cands.append(os.path.join(os.path.dirname(gcf),
                                          'pixelpaws_global_classifier_encyclopedia'))
        except Exception:
            pass
        return [c for c in dict.fromkeys(cands) if os.path.isdir(c)]

    def _discover_classifiers(self):
        """Return {display: path} for project + global + encyclopedia classifiers."""
        import glob as _g
        opts = {}
        try:
            proj = self.app.current_project_folder.get()
        except Exception:
            proj = ''
        if proj and os.path.isdir(os.path.join(proj, 'classifiers')):
            for full in sorted(_g.glob(os.path.join(proj, 'classifiers', '**', '*.pkl'), recursive=True)):
                opts[f"[Project] {os.path.basename(full)}"] = full
        try:
            from user_config import get_global_classifiers_folder
            gcf = get_global_classifiers_folder()
            if gcf and os.path.isdir(gcf):
                for full in sorted(_g.glob(os.path.join(gcf, '**', '*.pkl'), recursive=True)):
                    opts[f"[Global] {os.path.basename(full)}"] = full
        except Exception:
            pass
        for enc in self._encyclopedia_dirs():
            man = os.path.join(enc, 'manifest.json')
            try:
                if os.path.isfile(man):
                    import json
                    with open(man, encoding='utf-8') as fh:
                        m = json.load(fh)
                    for c in m.get('classifiers', []):
                        rel = c.get('path')
                        full = os.path.join(enc, rel) if rel else None
                        if full and os.path.isfile(full):
                            opts[f"[Encyclopedia] {c.get('name', os.path.basename(full))}"] = full
                else:
                    for full in sorted(_g.glob(os.path.join(enc, 'classifiers', '*.pkl'))):
                        opts[f"[Encyclopedia] {os.path.basename(full)}"] = full
            except Exception:
                pass
        return opts

    def _refresh_clf_pick_options(self):
        combo = getattr(self, '_clf_pick_combo', None)
        if combo is None:
            return
        opts = self._discover_classifiers()
        self._clf_pick_options = opts
        combo['values'] = list(opts.keys())
        if opts and not self._clf_pick_var.get():
            combo.current(0)

    def _add_selected_classifier(self):
        disp = self._clf_pick_var.get()
        path = self._clf_pick_options.get(disp)
        if not path:
            messagebox.showinfo("No classifier",
                                "Pick a classifier from the dropdown first (🔄 to refresh).")
            return
        if self._add_classifier_path(path):
            self._update_clf_listbox()
            self._update_priority_listbox()
        else:
            messagebox.showinfo("Not added",
                                "That classifier is already in the list (or has no model).")

    def _preset_load_from_dropdown(self, event=None):
        """Dropdown handler: load the chosen classifier set (deduped) WITHOUT starting
        the compute - the user reviews the list / settings, then clicks Compute."""
        label = (self._preset_source_var.get() or '').strip()
        source = {
            'Project + encyclopedia (deduped)': 'both',
            'Encyclopedia only':                'encyclopedia',
            'Project only':                     'project',
        }.get(label)
        if source is None:
            return
        self._preset_compute_from_classifiers(source, start=False)

    def _preset_compute_from_classifiers(self, source='both', start=True):
        """Load classifiers (deduped by behavior - project wins) and select all sessions.
        When `start` is True, also kick off the transitions + occupancy compute.
        `source`: 'both' (project + encyclopedia), 'encyclopedia' (encyclopedia only),
        or 'project' (project only)."""
        opts = self._discover_classifiers()
        # Filter by requested source.
        if source == 'encyclopedia':
            opts = {d: p for d, p in opts.items() if d.startswith('[Encyclopedia]') or d.startswith('[Global]')}
            _src_desc = "the encyclopedia / global folder"
        elif source == 'project':
            opts = {d: p for d, p in opts.items() if d.startswith('[Project]')}
            _src_desc = "the project"
        else:
            _src_desc = "the project or the encyclopedia"
        if not opts:
            messagebox.showinfo("No classifiers",
                                f"No classifiers found in {_src_desc}.\n\n"
                                "Train/import classifiers into <project>/classifiers, or set a "
                                "global classifiers folder / ship the encyclopedia.")
            return

        def _rank(disp):
            return 0 if disp.startswith('[Project]') else (1 if disp.startswith('[Global]') else 2)

        chosen = {}   # behavior -> (rank, path)
        for disp, path in opts.items():
            try:
                cd = _robust_load(path)
            except Exception:
                continue
            if 'clf_model' not in cd:
                continue
            beh = cd.get('Behavior_type') or os.path.basename(path)
            if not isinstance(beh, str):   # some classifiers store a list of behaviors
                beh = '+'.join(map(str, beh)) if isinstance(beh, (list, tuple)) else str(beh)
            r = _rank(disp)
            if beh not in chosen or r < chosen[beh][0]:
                chosen[beh] = (r, path)

        # Fresh load of the deduped set.
        self._loaded_classifiers = []
        self._priority_order = []
        for beh, (_r, path) in sorted(chosen.items()):
            self._add_classifier_path(path)
        self._update_clf_listbox()
        self._update_priority_listbox()
        if not self._loaded_classifiers:
            messagebox.showinfo("No classifiers", "Could not load any classifiers.")
            return

        self._source_var.set('supervised')
        self._scan_trans_sessions(silent=True)
        self._trans_select_all()
        if start:
            self._log_msg(f"Preset: {len(self._loaded_classifiers)} classifier(s) "
                          f"({', '.join(sorted(chosen.keys()))}) → computing…")
            self._start_compute()
        else:
            self._log_msg(
                f"Loaded {len(self._loaded_classifiers)} classifier(s) "
                f"({', '.join(sorted(chosen.keys()))}). "
                f"Review settings, then click Compute.")

    def _tip(self, widget, key):
        """Attach a hover tooltip to *widget* using the _HELP text for *key*.

        Theme-aware (ui_utils.ToolTip); safe no-op if the widget is None.
        """
        if widget is None:
            return
        try:
            ToolTip(widget, _HELP.get(key, key))
        except Exception:
            pass

    def _session_group(self, name):
        """Treatment group for a session (via the key file), or None."""
        if getattr(self, '_key_df', None) is None:
            return None
        subj = (getattr(self, '_session_subjects', None) or {}).get(name, name)
        try:
            row = self._key_df[self._key_df['Subject'] == subj]
            if not row.empty:
                return str(row.iloc[0]['Treatment'])
        except Exception:
            pass
        return None

    def _refresh_session_groups(self):
        """Resolve each scanned session to its subject/group and fill the tree's
        Group column. Also pre-populates _session_subjects so group-aware views
        work before a compute (compute rebuilds it too). Safe with no key file
        (every row shows '-')."""
        tree = getattr(self, '_trans_tree', None)
        if tree is None:
            return
        # Map session -> subject from the currently scanned sessions.
        self._session_subjects = {
            s['session_name']: self._resolve_subject(s['session_name'])
            for s in getattr(self, '_trans_sessions', []) or []
        }
        for iid in tree.get_children():
            grp = self._session_group(iid) or '-'
            tree.set(iid, 'group', grp)

    def _plot_occupancy(self):
        """Bar chart of whole-session state occupancy (% time in each behavior), mean ± SEM
        across sessions; grouped by treatment when a key file is loaded."""
        import numpy as np
        seqs = getattr(self, '_state_seqs', None)
        ax = self._fig.add_subplot(111)
        if not seqs:
            ax.text(0.5, 0.5, "Run 'Compute from classifiers' first.",
                    ha='center', va='center', transform=ax.transAxes)
            return
        states = list(self._states)
        labels = [self._state_labels.get(s, str(s)) for s in states]

        # Per-session % time in each state (ignore excluded frames marked -1).
        occ = {}
        for name, seq in seqs.items():
            arr = np.asarray(seq)
            arr = arr[arr != -1]
            if len(arr) == 0:
                continue
            occ[name] = np.array([np.mean(arr == s) for s in states]) * 100.0
        if not occ:
            ax.text(0.5, 0.5, "No data.", ha='center', va='center', transform=ax.transAxes)
            return

        cmap = self._get_cmap()
        x = np.arange(len(states))
        groups = {}
        for name in occ:
            g = self._session_group(name)
            groups.setdefault(g, []).append(occ[name])

        if len(groups) > 1 or (len(groups) == 1 and None not in groups):
            # Grouped bars by treatment.
            gkeys = self._ordered_groups(groups.keys())
            n_g = len(gkeys)
            width = 0.8 / n_g
            for gi, g in enumerate(gkeys):
                mat = np.vstack(groups[g])
                means = mat.mean(0)
                sems = mat.std(0, ddof=1) / np.sqrt(len(mat)) if len(mat) > 1 else np.zeros(len(states))
                ax.bar(x + (gi - (n_g - 1) / 2) * width, means, width, yerr=sems, capsize=3,
                       color=self._group_color(g, gi),
                       label=f"{g if g is not None else 'ungrouped'} (n={len(mat)})")
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1),
                      borderaxespad=0)
        else:
            mat = np.vstack(list(occ.values()))
            means = mat.mean(0)
            sems = mat.std(0, ddof=1) / np.sqrt(len(mat)) if len(mat) > 1 else np.zeros(len(states))
            ax.bar(x, means, yerr=sems, capsize=4,
                   color=[cmap(i) for i in range(len(states))])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('% of time')
        ax.set_title(f'State occupancy (% time) - {len(occ)} session(s)')
        ax.grid(axis='y', alpha=0.3)

    # ------------------------------------------------------------------
    # Composition views (State Usage / State Composition)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Display time window - restrict any view to [tmin, tmax] minutes live,
    # re-deriving occupancy / transition matrices from the windowed sequences.
    # ------------------------------------------------------------------

    def _win_bounds(self, n):
        """(i0, i1) frame indices for the current display window (minutes)."""
        try:
            fps = float(self._effective_fps) or 25.0
        except Exception:
            fps = 25.0
        def _f(v):
            try:
                s = v.get().strip()
                return float(s) if s else None
            except Exception:
                return None
        tmin = _f(self._trend_tmin_var) if hasattr(self, '_trend_tmin_var') else None
        tmax = _f(self._trend_tmax_var) if hasattr(self, '_trend_tmax_var') else None
        i0 = int(round(tmin * 60 * fps)) if tmin else 0
        i1 = int(round(tmax * 60 * fps)) if tmax else n
        i0 = max(0, min(i0, n))
        i1 = max(i0, min(i1, n))
        return i0, i1

    def _win_seq(self, seq):
        """Return `seq` restricted to the current display time window."""
        a = np.asarray(seq)
        i0, i1 = self._win_bounds(len(a))
        return a[i0:i1] if (i0 > 0 or i1 < len(a)) else a

    def _apply_window_xlim(self, unit='s'):
        """Crop the x-axis of time-series panels to the display window. `unit` is
        the axis unit ('s' or 'min'); the window fields are in minutes."""
        def _f(v):
            try:
                s = v.get().strip(); return float(s) if s else None
            except Exception:
                return None
        tmn, tmx = _f(self._trend_tmin_var), _f(self._trend_tmax_var)
        if tmn is None and tmx is None:
            return
        scale = 60.0 if unit == 's' else 1.0
        lo = tmn * scale if tmn is not None else None
        hi = tmx * scale if tmx is not None else None
        zero_based = not _g(self._realtime_axis_var, True)
        from matplotlib.ticker import FuncFormatter
        for ax in (self._fig.axes if self._fig else []):
            if getattr(ax, 'images', None):
                continue
            x0, x1 = ax.get_xlim()
            L = lo if lo is not None else x0
            H = hi if hi is not None else x1
            ax.set_xlim(L, H)
            if zero_based:
                # Relabel ticks so the window start reads as 0 (data untouched).
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda v, _p, off=L: f"{v - off:g}"))

    def _session_matrix(self, seq):
        """Transition matrix over the windowed sequence, honoring the current
        transition mode (bout/frame) and zero-diagonal setting."""
        s = self._win_seq(seq)
        n = len(self._states)
        if len(s) < 2:
            return np.zeros((n, n))
        try:
            if _g(self._transition_mode, 'bout') == 'bout':
                mat, _ = compute_bout_transition_matrix(
                    s, states=self._states, normalize=True)
            else:
                mat, _ = compute_transition_matrix(
                    s, states=self._states, normalize=True,
                    zero_diagonal=_g(self._zero_diag_var, False))
        except Exception:
            mat = np.zeros((n, n))
        return mat

    def _matrices_win(self):
        """{session: transition matrix} recomputed over the display window."""
        return {name: self._session_matrix(seq)
                for name, seq in (getattr(self, '_state_seqs', None) or {}).items()}

    def _group_matrices_win(self):
        """(group_means, group_subject_matrices) over the display window."""
        subj = {}
        for name, m in self._matrices_win().items():
            g = self._session_group(name)
            if g is None:
                continue
            subj.setdefault(g, []).append(m)
        means = {g: np.mean(np.stack(ms), axis=0) for g, ms in subj.items() if ms}
        return means, subj

    def _state_usage_occ(self):
        """Per-session % time in each state (over the display window), grouped by
        treatment.

        Returns (occ, groups) where occ = {session: np.array(%per state)} and
        groups = {treatment_or_None: [session, ...]} in dose order."""
        occ = {}
        for name, seq in (getattr(self, '_state_seqs', None) or {}).items():
            arr = self._win_seq(seq)
            arr = arr[arr != -1]
            if len(arr) == 0:
                continue
            occ[name] = np.array([np.mean(arr == s) for s in self._states]) * 100.0
        groups = {}
        for name in occ:
            groups.setdefault(self._session_group(name), []).append(name)
        return occ, groups

    def _plot_state_usage(self):
        """Per-state usage comparison (MoSeq/SuperPlot style): per-animal dots +
        group mean ± SEM, group-colored, sortable, with FDR significance stars."""
        ax = self._fig.add_subplot(111)
        occ, groups = self._state_usage_occ()
        if not occ:
            ax.text(0.5, 0.5, "Run 'Compute from classifiers' first.",
                    ha='center', va='center', transform=ax.transAxes)
            return
        n_states = len(self._states)
        # {group: array(n_states, n_animals)}
        gkeys = self._ordered_groups([g for g in groups if g is not None]) or \
            self._ordered_groups(groups.keys())
        vals = {}
        for g in gkeys:
            sess = groups.get(g, [])
            if not sess:
                continue
            vals[g] = np.column_stack([occ[s] for s in sess])  # (n_states, n_animals)
        if not vals:
            # single ungrouped set
            vals = {'All': np.column_stack([occ[s] for s in occ])}
            gkeys = ['All']
        else:
            gkeys = [g for g in gkeys if g in vals]

        # State display order.
        overall = np.array([np.concatenate([vals[g][si] for g in gkeys]).mean()
                            for si in range(n_states)])
        mode = self._sort_mode_var.get()
        if mode.startswith('Usage'):
            order = list(np.argsort(-overall))
        elif mode.startswith('Group'):
            ref = self._reference_group(gkeys)
            nonref = [g for g in gkeys if g != ref]
            if ref in vals and nonref:
                diff = np.array([
                    max(abs(vals[g][si].mean() - vals[ref][si].mean())
                        for g in nonref)
                    for si in range(n_states)])
                order = list(np.argsort(-diff))
            else:
                order = list(range(n_states))
        else:
            order = list(range(n_states))

        gpal = self._group_palette(gkeys)
        x = np.arange(n_states)
        n_g = len(gkeys)
        width = 0.8 / max(n_g, 1)
        rng = np.random.RandomState(0)
        for gi, g in enumerate(gkeys):
            col = gpal.get(g, 'gray')
            means = np.array([vals[g][si].mean() for si in order])
            arr = vals[g]
            n_an = arr.shape[1]
            errs = np.array([self._group_err(arr[si]) for si in order])
            xpos = x + (gi - (n_g - 1) / 2) * width
            ax.bar(xpos, means, width * 0.9, color=col, alpha=0.55,
                   yerr=errs, capsize=2, label=f"{g} (n={n_an})", zorder=1)
            # per-animal dots
            for k, si in enumerate(order):
                jitter = (rng.rand(n_an) - 0.5) * width * 0.6
                ax.scatter(np.full(n_an, xpos[k]) + jitter, arr[si],
                           s=14, color=col, edgecolor='black', linewidth=0.4,
                           zorder=3)
        # Per-state stats: q-value text (transparency) + stars (significance).
        if self._show_sig_var.get() and n_g >= 2:
            stats = self._perstate_stats(vals)
            top = max((float(vals[g].max()) for g in gkeys), default=0.0)
            for k, si in enumerate(order):
                st = stats[si] if si < len(stats) else {}
                q = st.get('q')
                star = st.get('star', '')
                if q is not None and not np.isnan(q):
                    ax.text(x[k], top * 1.02 + 1, f"q={q:.2g}", ha='center',
                            va='bottom', fontsize=6, color='gray')
                if star:
                    ax.text(x[k], top * 1.02 + 4, star, ha='center',
                            va='bottom', fontsize=11)
        labels = [self._state_labels.get(self._states[si], str(self._states[si]))
                  for si in order]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        _err_lbl = '95% CI' if self._error_mode_var.get().startswith('95') else 'SEM'
        ax.set_ylabel('% of time')
        ax.set_title(f'State usage (% time) - per-animal, mean ± {_err_lbl}')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1),
                  borderaxespad=0)

    def _plot_state_composition(self):
        """100%-stacked mean behavioral composition per group (state-colored)."""
        ax = self._fig.add_subplot(111)
        occ, groups = self._state_usage_occ()
        if not occ:
            ax.text(0.5, 0.5, "Run 'Compute from classifiers' first.",
                    ha='center', va='center', transform=ax.transAxes)
            return
        gkeys = self._ordered_groups(groups.keys())
        colors = self._state_palette()
        x = np.arange(len(gkeys))
        bottoms = np.zeros(len(gkeys))
        for si, s in enumerate(self._states):
            heights = []
            for g in gkeys:
                sess = groups.get(g, [])
                if sess:
                    m = np.mean([occ[n][si] for n in sess])
                else:
                    m = 0.0
                heights.append(m)
            heights = np.array(heights)
            ax.bar(x, heights, bottom=bottoms, color=colors[si],
                   label=self._state_labels.get(s, str(s)))
            bottoms += heights
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gkeys], rotation=20, ha='right')
        ax.set_ylabel('% of time')
        ax.set_title('Behavioral composition by group')
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1),
                  borderaxespad=0)

    def _remove_classifier(self):
        sel = self._clf_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        removed_clf_idx = idx
        del self._loaded_classifiers[removed_clf_idx]
        # Rebuild priority order: remove reference to removed index,
        # and decrement any index > removed
        self._priority_order = [
            (i - 1 if i > removed_clf_idx else i)
            for i in self._priority_order
            if i != removed_clf_idx
        ]
        self._update_clf_listbox()
        self._update_priority_listbox()

    def _edit_clf_settings(self):
        sel = self._clf_listbox.curselection()
        if not sel:
            messagebox.showinfo("Select classifier",
                                "Click a classifier first, then Edit Settings.")
            return
        idx = sel[0]
        cd = self._loaded_classifiers[idx]
        self._open_clf_edit_dialog(idx, cd)

    def _open_clf_edit_dialog(self, idx, cd):
        dlg = tk.Toplevel(self)
        dlg.title("Edit Classifier Settings")
        dlg.resizable(False, False)
        dlg.grab_set()

        bname = cd.get('Behavior_type', f'Classifier {idx+1}')
        ttk.Label(dlg, text=f"Editing: {bname}",
                  font=(FONT_FAMILY, 10, 'bold')).grid(
            row=0, column=0, columnspan=2, padx=15, pady=(12, 8), sticky='w')

        fields = [
            ("Threshold (0-1):",         'best_thresh',    0.5,
             dict(from_=0.0, to=1.0, increment=0.01, format='%.3f')),
            ("Min bout (frames):",       'min_bout',       0,
             dict(from_=0, to=9999, increment=1)),
            ("Min after bout (frames):", 'min_after_bout', 0,
             dict(from_=0, to=9999, increment=1)),
            ("Max gap (frames):",        'max_gap',        0,
             dict(from_=0, to=9999, increment=1)),
        ]
        vars_ = {}
        for row, (label, key, default, spin_kw) in enumerate(fields, start=1):
            ttk.Label(dlg, text=label).grid(row=row, column=0, sticky='w',
                                            padx=(15, 5), pady=3)
            val = cd.get(key, default)
            if key == 'best_thresh':
                v = tk.DoubleVar(value=round(float(val), 3))
            else:
                v = tk.IntVar(value=int(val))
            vars_[key] = v
            ttk.Spinbox(dlg, textvariable=v, width=9, **spin_kw).grid(
                row=row, column=1, sticky='w', padx=(0, 15), pady=3)

        def _reset():
            path = cd.get('_path', '')
            if not path or not os.path.isfile(path):
                messagebox.showwarning("No file",
                    "Original .pkl path not found - cannot reset.", parent=dlg)
                return
            try:
                orig = _robust_load(path)
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=dlg)
                return
            for _, key, default, _ in fields:
                val = orig.get(key, default)
                if key == 'best_thresh':
                    vars_[key].set(round(float(val), 3))
                else:
                    vars_[key].set(int(val))

        def _ok():
            for _, key, _, _ in fields:
                try:
                    self._loaded_classifiers[idx][key] = vars_[key].get()
                except Exception:
                    pass
            self._update_clf_listbox()
            self._update_priority_listbox()
            dlg.destroy()

        btn_row = ttk.Frame(dlg)
        btn_row.grid(row=len(fields)+1, column=0, columnspan=2,
                     pady=(8, 12), padx=15, sticky='e')
        ttk.Button(btn_row, text="Reset to Defaults",
                   command=_reset).pack(side='left', padx=(0, 20))
        ttk.Button(btn_row, text="Cancel",
                   command=dlg.destroy).pack(side='left', padx=5)
        ttk.Button(btn_row, text="OK",
                   command=_ok).pack(side='left', padx=5)

        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        sw, sh = dlg.winfo_screenwidth(), dlg.winfo_screenheight()
        dlg.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    def _update_clf_listbox(self):
        self._clf_listbox.delete(0, 'end')
        for cd in self._loaded_classifiers:
            bname  = cd.get('Behavior_type', '?')
            if not isinstance(bname, str):   # some classifiers store a list of behaviors
                bname = '+'.join(map(str, bname)) if isinstance(bname, (list, tuple)) else str(bname)
            thresh = cd.get('best_thresh', 0.5)
            mb     = cd.get('min_bout', 0)
            mab    = cd.get('min_after_bout', 0)
            mg     = cd.get('max_gap', 0)
            self._clf_listbox.insert('end',
                f"{bname:<20} thresh={thresh:.3f}  "
                f"min_bout={mb}  min_after={mab}  max_gap={mg}")

    def _scan_trans_sessions(self, silent=False):
        """Scan project folder for sessions using find_session_triplets."""
        proj = self.app.current_project_folder.get()
        if not proj:
            if not silent:
                messagebox.showwarning("No project",
                                       "Select a project folder first.")
            return
        if not _FIND_SESSION_TRIPLETS_AVAILABLE:
            if not silent:
                messagebox.showerror("Error",
                                     "Cannot import find_session_triplets.")
            return

        sessions = find_session_triplets(proj, require_labels=False,
                                         recursive=True)
        self._trans_sessions = sessions

        # Rebuild treeview
        self._trans_tree.delete(*self._trans_tree.get_children())
        self._trans_session_checked.clear()

        for s in sessions:
            name = s['session_name']
            video_name = os.path.basename(s.get('video_path', '') or '')
            var = tk.BooleanVar(value=True)
            self._trans_session_checked[name] = var
            self._trans_tree.insert('', 'end', iid=name,
                                    values=("✓", name, "-", video_name))

        self._apply_pending_session_selection()
        self._refresh_session_groups()

        if not silent:
            self._log_msg(f"Found {len(sessions)} session(s) in project.")
        elif sessions:
            self._log_msg(f"Found {len(sessions)} session(s) in project.")

    def _on_trans_tree_click(self, event):
        """Toggle checkbox when user clicks a row in the session treeview."""
        tree = self._trans_tree
        region = tree.identify_region(event.x, event.y)
        if region not in ("cell", "tree"):
            return
        row_id = tree.identify_row(event.y)
        if not row_id or row_id not in self._trans_session_checked:
            return
        bvar = self._trans_session_checked[row_id]
        bvar.set(not bvar.get())
        vals = list(tree.item(row_id, "values"))
        vals[0] = "✓" if bvar.get() else ""
        tree.item(row_id, values=vals)

    def _trans_select_all(self):
        for name, bvar in self._trans_session_checked.items():
            bvar.set(True)
            vals = list(self._trans_tree.item(name, "values"))
            vals[0] = "✓"
            self._trans_tree.item(name, values=vals)

    def _trans_deselect_all(self):
        for name, bvar in self._trans_session_checked.items():
            bvar.set(False)
            vals = list(self._trans_tree.item(name, "values"))
            vals[0] = ""
            self._trans_tree.item(name, values=vals)

    def _apply_pending_session_selection(self):
        """Apply _pending_session_selection to the treeview if sessions are loaded."""
        if self._pending_session_selection is None:
            return
        if not self._trans_session_checked:
            return  # sessions not yet scanned - will be applied after _scan_trans_sessions
        for name, bvar in self._trans_session_checked.items():
            checked = name in self._pending_session_selection
            bvar.set(checked)
            vals = list(self._trans_tree.item(name, "values"))
            vals[0] = "✓" if checked else ""
            self._trans_tree.item(name, values=vals)
        self._pending_session_selection = None  # consumed

    def _update_priority_listbox(self):
        self._priority_listbox.delete(0, 'end')
        for idx in self._priority_order:
            cd = self._loaded_classifiers[idx]
            bname = cd.get('Behavior_type', '?')
            thresh = cd.get('best_thresh', 0.5)
            self._priority_listbox.insert('end',
                                          f"{bname}  (thresh={thresh:.3f})")

    def _move_priority_up(self):
        sel = self._priority_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        i = sel[0]
        self._priority_order[i], self._priority_order[i - 1] = (
            self._priority_order[i - 1], self._priority_order[i])
        self._update_priority_listbox()
        self._priority_listbox.selection_set(i - 1)

    def _move_priority_down(self):
        sel = self._priority_listbox.curselection()
        if not sel or sel[0] >= len(self._priority_order) - 1:
            return
        i = sel[0]
        self._priority_order[i], self._priority_order[i + 1] = (
            self._priority_order[i + 1], self._priority_order[i])
        self._update_priority_listbox()
        self._priority_listbox.selection_set(i + 1)

    def _toggle_time(self):
        self._range_frame.grid_forget()
        self._sliding_frame.grid_forget()
        self._pair_frame.pack_forget()
        mode = self._time_mode_var.get()
        if mode == 'range':
            self._range_frame.grid(row=1, column=1, columnspan=2, sticky='w', padx=10)
        elif mode == 'sliding':
            self._sliding_frame.grid(row=2, column=1, columnspan=2, sticky='w', padx=10)

    # ------------------------------------------------------------------
    # Scan unsupervised runs
    # ------------------------------------------------------------------

    def _scan_runs(self):
        folder = self.app.current_project_folder.get()
        if not folder:
            messagebox.showwarning("No project", "Select a project folder first.")
            return
        unsup_dir = os.path.join(folder, 'unsupervised')
        if not os.path.isdir(unsup_dir):
            self._run_combo['values'] = []
            self._run_combo.set('')
            self._log_msg("No unsupervised/ directory found in project.")
            return
        runs = sorted(d for d in os.listdir(unsup_dir)
                      if os.path.isdir(os.path.join(unsup_dir, d)))
        self._run_combo['values'] = runs
        if runs:
            self._run_combo.set(runs[0])
        self._log_msg(f"Found {len(runs)} Discover run(s): {', '.join(runs)}")

    # ------------------------------------------------------------------
    # Supervised source helpers
    # ------------------------------------------------------------------

    def _browse_results(self):
        folder = self.app.current_project_folder.get()
        init_dir = os.path.join(folder, 'results') if folder else None
        chosen = filedialog.askdirectory(title="Select Results Folder",
                                         initialdir=init_dir)
        if chosen:
            self._results_var.set(chosen)
            self._scan_supervised()

    def _scan_supervised(self):
        folder = self._results_var.get()
        if not folder or not os.path.isdir(folder):
            return
        behaviors = set()
        for dirpath, _, filenames in os.walk(folder):
            for f in filenames:
                if f.endswith('.csv') and 'prediction' in f.lower():
                    bname = self._extract_behavior_name(f)
                    if bname:
                        behaviors.add(bname)
        # Clear old checkboxes
        for w in self._behavior_list_frame.winfo_children():
            w.destroy()
        self._behavior_vars.clear()
        if behaviors:
            self._behavior_list_frame.grid(row=3, column=0, columnspan=3,
                                            sticky='ew', pady=(5, 0))
            ttk.Label(self._behavior_list_frame,
                      text="Select behaviors:").pack(anchor='w', padx=20)
            for b in sorted(behaviors):
                var = tk.BooleanVar(value=True)
                self._behavior_vars[b] = var
                ttk.Checkbutton(self._behavior_list_frame, text=b,
                                variable=var).pack(anchor='w', padx=30)
            self._log_msg(f"Found {len(behaviors)} behavior(s): "
                          f"{', '.join(sorted(behaviors))}")
        else:
            self._log_msg("No prediction CSVs found.")

    @staticmethod
    def _extract_behavior_name(filename):
        """Extract behavior name from prediction filename."""
        name = filename.replace('.csv', '')
        for suffix in ['_predictions', '_prediction', '_pred']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        parts = name.split('_')
        if 'PixelPaws' in parts:
            idx = parts.index('PixelPaws')
            behavior_parts = parts[idx + 1:]
            if behavior_parts:
                return '_'.join(behavior_parts)
        return None

    # ------------------------------------------------------------------
    # Key file
    # ------------------------------------------------------------------

    def _discover_key_file(self, folder):
        """Recursively find a valid key file in *folder*, or return None.

        A valid key file is a .csv/.xlsx whose header contains both 'Subject'
        and 'Treatment' columns (same contract as _load_key_file). Mirrors
        gait_limb_tab._scan_key_files. Prediction/bout exports are skipped.
        Candidates are ranked: name contains 'key' first, then shortest path.
        """
        if not folder or not os.path.isdir(folder):
            return None
        _SKIP = {'__pycache__', '.git', '.claude', 'node_modules', '.idea'}
        _PRED_KW = ('prediction', 'predictions', 'pred', 'bout', 'bouts')
        candidates = []
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in sorted(dirs)
                       if d not in _SKIP and not d.startswith('.')]
            for fname in files:
                fl = fname.lower()
                if not fl.endswith(('.csv', '.xlsx')):
                    continue
                if any(kw in fl for kw in _PRED_KW):
                    continue
                full = os.path.join(root, fname)
                try:
                    if full.endswith('.xlsx'):
                        cols = pd.read_excel(full, nrows=0).columns.tolist()
                    else:
                        cols = pd.read_csv(full, nrows=0).columns.tolist()
                    if 'Subject' not in cols or 'Treatment' not in cols:
                        continue
                    # exported results tables carry Subject+Treatment too
                    try:
                        from project_config import _KEY_RESULTS_COLS
                    except Exception:
                        _KEY_RESULTS_COLS = set()
                    if (_KEY_RESULTS_COLS & set(cols)) and 'key' not in fl:
                        continue
                    candidates.append(full)
                except Exception:
                    pass
        if not candidates:
            return None
        candidates.sort(key=lambda p: (
            0 if 'key' in os.path.basename(p).lower() else 1, len(p)))
        return candidates[0]

    def _autofind_key_file(self):
        """Search the project for a valid key file and load it (button / auto)."""
        folder = self.app.current_project_folder.get() or ''
        path = self._discover_key_file(folder)
        if path:
            self._key_file_var.set(path)
            self._load_key_file()
        else:
            self._key_status.config(
                text="No key file found in project (needs Subject + Treatment columns)",
                foreground='orange')

    def _browse_key_file(self):
        folder = self.app.current_project_folder.get() or ''
        path = filedialog.askopenfilename(
            title="Select Key File",
            initialdir=folder,
            filetypes=[("CSV/Excel", "*.csv *.xlsx"), ("All", "*.*")])
        if path:
            self._key_file_var.set(path)
            self._load_key_file()

    def _load_key_file(self):
        path = self._key_file_var.get()
        if not path or not os.path.isfile(path):
            return
        try:
            if path.endswith('.xlsx'):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            if 'Subject' not in df.columns or 'Treatment' not in df.columns:
                messagebox.showerror("Invalid",
                                     "Key file must have Subject and Treatment columns.")
                return
            df['Subject'] = df['Subject'].astype(str)
            self._key_df = df
            treatments = df['Treatment'].unique()
            self._key_status.config(
                text=f"Loaded: {len(df)} subjects, "
                     f"{len(treatments)} group(s): {', '.join(map(str, treatments))}",
                foreground='green')
            # Update the sessions' Group column now that a key file is available.
            self._refresh_session_groups()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load key file:\n{e}")

    # ------------------------------------------------------------------
    # FPS auto-detect
    # ------------------------------------------------------------------

    def _auto_detect_fps(self):
        folder = self.app.current_project_folder.get()
        vid_dir = os.path.join(folder, 'videos') if folder else ''
        path = filedialog.askopenfilename(
            title="Select a video to detect FPS",
            initialdir=vid_dir if os.path.isdir(vid_dir) else folder,
            filetypes=[("Video", "*.mp4 *.avi *.mov"), ("All", "*.*")])
        if not path:
            return
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                self._fps_var.set(int(round(fps)))
                self._log_msg(f"Detected FPS: {fps:.2f} -> set to {int(round(fps))}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not detect FPS:\n{e}")

    # ------------------------------------------------------------------
    # Cluster label editing
    # ------------------------------------------------------------------

    def _populate_label_table(self, states):
        """Build or rebuild the editable rename table for discovered states."""
        for w in self._label_table_frame.winfo_children():
            w.destroy()
        self._label_entries.clear()

        ttk.Label(self._label_table_frame, text="ID", width=6,
                  font=(FONT_FAMILY, 9, 'bold')).grid(row=0, column=0)
        ttk.Label(self._label_table_frame, text="Label", width=20,
                  font=(FONT_FAMILY, 9, 'bold')).grid(row=0, column=1)

        for i, sid in enumerate(states):
            ttk.Label(self._label_table_frame,
                      text=str(sid), width=6).grid(row=i + 1, column=0)
            entry = ttk.Entry(self._label_table_frame, width=20)
            entry.grid(row=i + 1, column=1, padx=5, pady=1)
            # Pre-fill with existing label or merge info
            if sid in self._state_labels:
                entry.insert(0, self._state_labels[sid])
            elif (self._merge_info is not None and sid in self._merge_info
                  and len(self._merge_info[sid]) > 1):
                old_ids = self._merge_info[sid]
                n = len(old_ids)
                if n <= 4:
                    detail = ','.join(map(str, old_ids))
                else:
                    detail = f"{n} clusters"
                entry.insert(0, f"Meta {sid} ({detail})")
            self._label_entries[sid] = entry

    def _read_label_entries(self):
        """Harvest the label table into self._state_labels."""
        self._state_labels = {}
        for sid, entry in self._label_entries.items():
            txt = entry.get().strip()
            if txt:
                self._state_labels[sid] = txt

    def _state_name(self, sid):
        """Return user label or default cluster name."""
        return self._state_labels.get(sid, f"Cluster {sid}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_msg(self, msg):
        def _append():
            self._log.config(state='normal')
            self._log.insert('end', msg + '\n')
            self._log.see('end')
            self._log.config(state='disabled')
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.app.root.after(0, _append)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_unsupervised_states(self, run_name):
        """Load per-session cluster IDs from a Discover run."""
        folder = self.app.current_project_folder.get()
        run_dir = os.path.join(folder, 'unsupervised', run_name)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Try loading from per-session CSV files
        csv_files = glob.glob(os.path.join(run_dir, '*_cluster_ids.csv'))
        seqs = {}
        if csv_files:
            for csv_path in csv_files:
                fname = os.path.basename(csv_path)
                session = fname.replace('_cluster_ids.csv', '')
                df = pd.read_csv(csv_path)
                seqs[session] = df['cluster_id'].values
            self._log_msg(f"Loaded {len(seqs)} session(s) from CSVs")
            # Best-effort: also load model.pkl for embedding data
            model_path = os.path.join(run_dir, 'model.pkl')
            if os.path.isfile(model_path):
                try:
                    with open(model_path, 'rb') as fh:
                        self._model_bundle = pickle.load(fh)
                    self._log_msg("Loaded model bundle (embedding available)")
                except Exception:
                    self._model_bundle = None
            else:
                self._model_bundle = None
        else:
            # Fall back to model bundle
            model_path = os.path.join(run_dir, 'model.pkl')
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"No cluster_ids CSVs or model.pkl in {run_dir}")
            with open(model_path, 'rb') as fh:
                bundle = pickle.load(fh)
            self._model_bundle = bundle
            labels = bundle.get('cluster_labels')
            row_map = bundle.get('sessions')
            if labels is None or row_map is None:
                raise ValueError("Model bundle missing cluster_labels or sessions")
            for session, (start, end) in row_map.items():
                seqs[session] = labels[start:end].copy()
            self._log_msg(f"Loaded {len(seqs)} session(s) from model bundle")

        return seqs

    def _load_supervised_states(self):
        """Combine binary prediction CSVs into multi-class state sequences."""
        folder = self._results_var.get()
        if not folder or not os.path.isdir(folder):
            raise FileNotFoundError("Results folder not set or not found")

        selected = [b for b, v in self._behavior_vars.items() if v.get()]
        if not selected:
            raise ValueError("No behaviors selected")

        # Collect prediction files per behavior
        behavior_files = {b: [] for b in selected}
        for dirpath, _, filenames in os.walk(folder):
            for f in filenames:
                if not f.endswith('.csv') or 'prediction' not in f.lower():
                    continue
                bname = self._extract_behavior_name(f)
                if bname in behavior_files:
                    behavior_files[bname].append(os.path.join(dirpath, f))

        # Group by session (subject)
        session_preds = {}  # {session: {behavior: array}}
        for bname, files in behavior_files.items():
            for fpath in files:
                fname = os.path.basename(fpath).replace('.csv', '')
                # Remove behavior suffix to get session name
                for suffix in ['_predictions', '_prediction', '_pred']:
                    if fname.endswith(suffix):
                        fname = fname[:-len(suffix)]
                        break
                # Remove behavior name from end to get base session
                if fname.endswith('_' + bname):
                    session = fname[:-len(bname) - 1]
                elif 'PixelPaws' in fname:
                    parts = fname.split('_')
                    idx = parts.index('PixelPaws')
                    session = '_'.join(parts[:idx])
                else:
                    session = fname

                df = pd.read_csv(fpath)
                # Look for probability or binary prediction column
                pred_col = None
                for c in df.columns:
                    cl = c.lower()
                    if 'probability' in cl or 'prob' in cl:
                        pred_col = c
                        break
                if pred_col is None:
                    for c in df.columns:
                        cl = c.lower()
                        if 'prediction' in cl or 'pred' in cl:
                            pred_col = c
                            break
                if pred_col is None:
                    continue

                if session not in session_preds:
                    session_preds[session] = {}
                session_preds[session][bname] = df[pred_col].values

        # Combine into multi-class: highest probability wins, "Idle" otherwise
        seqs = {}
        all_behaviors = sorted(selected)
        state_map = {b: i + 1 for i, b in enumerate(all_behaviors)}
        state_map_inv = {v: k for k, v in state_map.items()}
        idle_id = 0

        for session, bdict in session_preds.items():
            n_frames = max(len(v) for v in bdict.values())
            prob_matrix = np.zeros((n_frames, len(all_behaviors)))
            for bi, bname in enumerate(all_behaviors):
                if bname in bdict:
                    arr = bdict[bname]
                    prob_matrix[:len(arr), bi] = arr

            states = np.full(n_frames, idle_id, dtype=int)
            max_probs = prob_matrix.max(axis=1)
            active = max_probs > 0.5
            states[active] = prob_matrix[active].argmax(axis=1) + 1
            seqs[session] = states

        # Set up state labels
        self._state_labels = {idle_id: "Idle"}
        for bname, sid in state_map.items():
            self._state_labels[sid] = bname

        self._log_msg(f"Combined {len(all_behaviors)} behaviors into "
                      f"{len(seqs)} session(s)")
        return seqs

    def _predict_and_assign_states(self):
        """Run classifiers on checked sessions; assign per-frame state."""
        if not self._loaded_classifiers:
            raise ValueError("No classifiers loaded. Use 'Add Classifier'.")

        if not self._trans_sessions:
            raise ValueError("No sessions found. Click 'Refresh Sessions' first.")
        checked_sessions = [s for s in self._trans_sessions
                            if self._trans_session_checked.get(
                                s['session_name'], tk.BooleanVar()).get()]
        if not checked_sessions:
            raise ValueError("No sessions selected.")

        # Import utilities from parent modules
        try:
            from PixelPaws_GUI import PixelPaws_ExtractFeatures, \
                augment_features_post_cache, predict_with_xgboost, \
                _load_features_for_prediction
        except ImportError:
            raise ImportError(
                "Cannot import prediction utilities from PixelPaws_GUI.")
        try:
            from evaluation_tab import _apply_bout_filtering
        except ImportError:
            raise ImportError(
                "Cannot import _apply_bout_filtering from evaluation_tab.")

        assign_mode = self._assign_mode.get()
        seqs = {}
        self._skipped_sessions = []

        # State labels: 0 = Other, 1..N = behaviors in load order
        self._state_labels = {0: 'Other'}
        for bi, cd in enumerate(self._loaded_classifiers):
            self._state_labels[bi + 1] = cd.get('Behavior_type',
                                                  f'Behavior_{bi + 1}')

        # Build feature config from first classifier (all share same features)
        cd0 = self._loaded_classifiers[0]
        cfg = {
            'bp_include_list':      cd0.get('bp_include_list', None),
            'bp_pixbrt_list':       cd0.get('bp_pixbrt_list', []),
            'square_size':          cd0.get('square_size', [20]),
            'pix_threshold':        cd0.get('pix_threshold', 0.3),
            'include_optical_flow': cd0.get('include_optical_flow', False),
            'bp_optflow_list':      cd0.get('bp_optflow_list', []),
        }
        if _TRANS_FEATURE_CACHE_AVAILABLE:
            cfg_hash = _TransFeatureCacheManager.compute_hash(cfg)
        else:
            key_dict = {
                'bp_include_list':  cfg['bp_include_list'],
                'bp_pixbrt_list':   list(cfg['bp_pixbrt_list']),
                'square_size':      [int(x) for x in cfg['square_size']]
                                    if hasattr(cfg['square_size'], '__iter__')
                                    else [int(cfg['square_size'])],
                'pix_threshold':    round(float(cfg['pix_threshold']), 6),
                'include_optical_flow': bool(cfg['include_optical_flow']),
                'bp_optflow_list':  list(cfg['bp_optflow_list']),
            }
            cfg_hash = hashlib.md5(
                repr(key_dict).encode('utf-8')).hexdigest()[:8]

        proj = self.app.current_project_folder.get()
        cache_root = os.path.join(proj, 'features') if proj else None

        for session in checked_sessions:
            if self._stop_event.is_set():
                self._log_msg("  Stopped.")
                return {}
            session_name = session['session_name']
            dlc_path = session['pose_path']
            video_path = session.get('video_path', '') or ''

            if not dlc_path or not os.path.isfile(dlc_path):
                self._log_msg(f"  {session_name}: DLC file not found, skipping")
                continue

            self._log_msg(f"  Processing: {session_name}")

            try:
                # Prefer the exact-config cache (schema guaranteed to match these
                # classifiers); only fall back to any-hash match if it's absent.
                existing_cache = None
                if _TRANS_FEATURE_CACHE_AVAILABLE and cache_root:
                    os.makedirs(cache_root, exist_ok=True)
                    existing_cache = _TransFeatureCacheManager.find_cache(
                        session_name, cfg_hash, cache_root,
                        os.path.dirname(video_path), proj)
                    if not existing_cache:
                        existing_cache = _TransFeatureCacheManager.find_any_cache(
                            session_name, cache_root,
                            os.path.dirname(video_path), proj)

                save_path = (os.path.join(
                    cache_root,
                    f"{session_name}_features_{cfg_hash}.pkl")
                             if cache_root else None)

                def _extract():
                    return PixelPaws_ExtractFeatures(
                        pose_data_file=dlc_path,
                        video_file_path=video_path
                                        if os.path.isfile(video_path) else '',
                        bp_pixbrt_list=cd0.get('bp_pixbrt_list', []),
                        square_size=cd0.get('square_size', 20),
                        pix_threshold=cd0.get('pix_threshold', 50),
                        bp_include_list=cd0.get('bp_include_list', None),
                    )

                _cache_only = bool(getattr(self, '_cache_only_var', None) and self._cache_only_var.get())
                X = _load_features_for_prediction(
                    cache_file=existing_cache,
                    model=cd0.get('clf_model'),
                    extract_fn=(None if _cache_only else _extract),
                    save_path=save_path,
                    log_fn=self._log_msg,
                    dlc_path=dlc_path,
                    clf_data=cd0,
                )
            except Exception as e:
                self._log_msg(f"  {session_name}: feature extraction failed: "
                              f"{e}")
                self._skipped_sessions.append(session_name)
                continue

            if X is None:
                # cache-only mode + stale/missing cache → skip rather than re-read the video.
                self._log_msg(f"  Skipped {session_name}: features not cached "
                              f"(would need video re-extraction)")
                self._skipped_sessions.append(session_name)
                continue

            n_frames = len(X)
            prob_matrix = np.zeros((n_frames, len(self._loaded_classifiers)))
            binary_matrix = np.zeros(
                (n_frames, len(self._loaded_classifiers)), dtype=int)

            for bi, cd in enumerate(self._loaded_classifiers):
                try:
                    model = cd['clf_model']
                    X_aug = augment_features_post_cache(
                        X.copy(), cd, model, dlc_path,
                        log_fn=self._log_msg)
                    proba = predict_with_xgboost(
                        model, X_aug,
                        calibrator=(cd.get('prob_calibrator') or cd.get('calibrator')),
                        fold_models=cd.get('fold_models'))
                    prob_matrix[:len(proba), bi] = proba

                    thresh = cd.get('best_thresh', 0.5)
                    binary_raw = (proba >= thresh).astype(int)
                    min_bout = cd.get('min_bout', 0)
                    min_after = cd.get('min_after_bout', 0)
                    max_gap = cd.get('max_gap', 0)
                    binary_filt = _apply_bout_filtering(
                        binary_raw, min_bout, min_after, max_gap)
                    binary_matrix[:len(binary_filt), bi] = binary_filt
                except Exception as e:
                    self._log_msg(f"  {session_name} / "
                                  f"{cd.get('Behavior_type','?')}: {e}")

            # Assign states
            states_arr = np.zeros(n_frames, dtype=int)
            if assign_mode == 'priority':
                for frame_i in range(n_frames):
                    assigned = False
                    for rank_pos, clf_idx in enumerate(self._priority_order):
                        if binary_matrix[frame_i, clf_idx] == 1:
                            states_arr[frame_i] = clf_idx + 1
                            assigned = True
                            break
                    if not assigned:
                        states_arr[frame_i] = 0
            else:  # argmax
                for frame_i in range(n_frames):
                    best = int(np.argmax(prob_matrix[frame_i]))
                    best_clf = self._loaded_classifiers[best]
                    thresh = best_clf.get('best_thresh', 0.5)
                    if prob_matrix[frame_i, best] >= thresh:
                        states_arr[frame_i] = best + 1
                    else:
                        states_arr[frame_i] = 0

            seqs[session_name] = states_arr
            self._frame_probs[session_name] = prob_matrix.copy()
            self._log_msg(
                f"  {session_name}: {n_frames} frames assigned")

        if self._skipped_sessions:
            self._log_msg(
                f"Skipped {len(self._skipped_sessions)} session(s) needing "
                f"video re-extraction: "
                f"{', '.join(self._skipped_sessions)}")

        if not seqs:
            if self._skipped_sessions:
                raise ValueError(
                    "No sessions produced state sequences - all selected "
                    "sessions needed feature re-extraction and were skipped "
                    "('Use cached features only' is on). Uncheck it to extract, "
                    "or select sessions with cached features.")
            raise ValueError("No sessions produced valid state sequences.")
        return seqs

    # ------------------------------------------------------------------
    # Subject resolution
    # ------------------------------------------------------------------

    def _resolve_subject(self, session_name):
        """Match session name to a subject in the key file."""
        if self._key_df is None:
            return session_name
        stem = session_name
        for suffix in ['_predictions', '_prediction', '_pred', '_clusters']:
            stem = stem.replace(suffix, '')
        tokens = stem.split('_')
        for subj in self._key_df['Subject']:
            subj_str = str(subj).strip()
            if subj_str in tokens:
                return subj_str
            if f'_{subj_str}_' in f'_{stem}_':
                return subj_str
        return session_name

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def _notify_cache_status(self):
        """Scan checked sessions for stale caches; show one-time info dialog."""
        try:
            cd0 = self._loaded_classifiers[0]
            cfg = {
                'bp_include_list':      cd0.get('bp_include_list', None),
                'bp_pixbrt_list':       cd0.get('bp_pixbrt_list', []),
                'square_size':          cd0.get('square_size', [20]),
                'pix_threshold':        cd0.get('pix_threshold', 0.3),
                'include_optical_flow': cd0.get('include_optical_flow', False),
                'bp_optflow_list':      cd0.get('bp_optflow_list', []),
            }
            cfg_hash = _TransFeatureCacheManager.compute_hash(cfg)
            proj = self.app.current_project_folder.get()
            cache_root = os.path.join(proj, 'features') if proj else None
            if not cache_root:
                return

            checked_sessions = [s for s in self._trans_sessions
                                 if self._trans_session_checked.get(
                                     s['session_name'],
                                     tk.BooleanVar(value=False)).get()]

            stale, missing = [], []
            for session in checked_sessions:
                sname = session['session_name']
                vpath = session.get('video_path', '') or ''
                found = _TransFeatureCacheManager.find_any_cache(
                    sname, cache_root, os.path.dirname(vpath), proj)
                if found is None:
                    missing.append(sname)
                elif cfg_hash not in os.path.basename(found):
                    stale.append(sname)

            parts = []
            if stale:
                parts.append(
                    f"{len(stale)} session(s) have older feature caches \u2014 "
                    f"PixelPaws will attempt to upgrade them without re-reading video.")
            if missing:
                parts.append(
                    f"{len(missing)} session(s) have no cache \u2014 "
                    f"full feature extraction required.")
            if not parts:
                return   # everything is up-to-date, no dialog needed

            messagebox.showinfo(
                "Feature Cache Status",
                "\n\n".join(parts),
                parent=self)
        except Exception:
            pass   # never block compute on a scan failure

    def _stop_compute(self):
        self._stop_event.set()
        self._log_msg("Stop requested \u2014 aborting after current step...")

    def _start_compute(self):
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showwarning("Busy", "Computation already running.")
            return
        self._stop_event.clear()
        self._stop_btn.config(state='normal')
        self._progress.start()
        self._read_label_entries()

        # Pre-flight: detect stale feature caches and inform user
        if (self._source_var.get() == 'supervised'
                and self._loaded_classifiers
                and _TRANS_FEATURE_CACHE_AVAILABLE):
            self._notify_cache_status()

        self._worker_thread = threading.Thread(target=self._compute_thread,
                                               daemon=True)
        self._worker_thread.start()

    def _compute_thread(self):
        try:
            self._log_msg("--- Starting transition computation ---")
            self._frame_probs = {}

            # 1. Load state sequences
            if self._source_var.get() == 'unsupervised':
                run = self._run_combo.get()
                if not run:
                    raise ValueError("Select a Discover run first.")
                seqs = self._load_unsupervised_states(run)
            else:
                if self._loaded_classifiers:
                    seqs = self._predict_and_assign_states()
                else:
                    seqs = self._load_supervised_states()

            if not seqs:
                raise ValueError("No state sequences loaded.")

            # 2. Pre-process: smoothing + noise exclusion
            fps = self._fps_var.get()
            smooth_ms = self._smooth_ms_var.get()
            exclude_noise = self._exclude_noise_var.get()
            min_frames = max(1, round(smooth_ms * fps / 1000)) if smooth_ms > 0 else 0

            processed = {}
            for name, seq in seqs.items():
                if self._stop_event.is_set():
                    self._log_msg("Stopped.")
                    self.app.root.after(0, self._on_compute_done)
                    return
                s = seq.copy()
                if exclude_noise:
                    # Replace -1 (noise) with nearest valid state
                    valid_mask = s >= 0
                    if valid_mask.any() and not valid_mask.all():
                        # Forward fill then backward fill
                        valid_idx = np.where(valid_mask)[0]
                        for i in range(len(s)):
                            if s[i] < 0:
                                # Find nearest valid
                                dists = np.abs(valid_idx - i)
                                s[i] = s[valid_idx[dists.argmin()]]
                if min_frames > 0:
                    s = smooth_state_sequence(s, min_frames)
                # Downsample to ~20 Hz (LUPE style): take mode of every N frames
                if self._downsample_var.get() and fps > 20:
                    n = max(1, round(fps / 20))
                    trim = len(s) - len(s) % n
                    if trim > 0:
                        from scipy.stats import mode as scipy_mode
                        chunks = s[:trim].reshape(-1, n)
                        s = scipy_mode(chunks, axis=1).mode.flatten()
                    else:
                        s = s[:0]  # edge case: sequence shorter than one block
                processed[name] = s

            # Apply effective fps after downsampling
            if self._downsample_var.get() and fps > 20:
                n = max(1, round(fps / 20))
                fps = fps / n
            self._effective_fps = fps   # store for _plot_behavior_over_time

            # Always preserve a full-length copy so UI consumers (Video Preview)
            # can index 1:1 against video frames even when analysis pipelines
            # use a duration cap.
            self._state_seqs_full = dict(processed)

            # Duration cap (applies to all time modes)
            if self._dur_limit_var.get():
                cap_frames = int(self._dur_limit_min.get() * 60 * fps)
                processed = {name: s[:cap_frames] for name, s in processed.items()}
                self._log_msg(f"Duration cap: first {self._dur_limit_min.get():.0f} min "
                              f"({cap_frames} frames)")

            # Determine global state set
            all_states_set = set()
            for s in processed.values():
                all_states_set.update(s)
            states = sorted(all_states_set)

            # 2b. Cluster reduction (optional)
            self._merge_info = None
            reduce = self._reduce_clusters_var.get()
            target_n = self._target_clusters_var.get()

            if reduce and len(states) > target_n:
                self._log_msg(f"Reducing {len(states)} clusters -> "
                              f"{target_n} meta-clusters...")
                mapping, states, merge_info = reduce_clusters(
                    processed, states, target_n)
                # Remap all sequences
                for name in processed:
                    processed[name] = np.array(
                        [mapping[v] for v in processed[name]])
                self._merge_info = merge_info
                # Log the merge mapping
                for new_id, old_ids in sorted(merge_info.items()):
                    self._log_msg(f"  Meta-cluster {new_id}: "
                                  f"merged from {old_ids}")
                self._log_msg(f"Reduction complete: {len(states)} "
                              f"meta-clusters")
                # Update status label on main thread
                orig_n = len(all_states_set)
                self.app.root.after(0, lambda: self._reduce_status_label.config(
                    text=f"{orig_n} -> {len(states)} clusters"))
            elif reduce and len(states) <= target_n:
                self._log_msg(f"Already have {len(states)} clusters "
                              f"(<= target {target_n}), skipping reduction.")
                self.app.root.after(0, lambda: self._reduce_status_label.config(
                    text=f"{len(states)} clusters (no reduction needed)"))

            # Exclude 'Other' (state 0) if requested
            if self._exclude_other_var.get() and 0 in states:
                self._log_msg("Excluding 'Other' (state 0) from analysis...")
                for name in processed:
                    seq = processed[name]
                    processed[name] = np.where(seq == 0, -1, seq)
                states = [s for s in states if s != 0]
                self._state_seqs = processed   # update before matrix step
                self._states = states

            self._state_seqs = processed
            self._states = states
            self._session_subjects = {n: self._resolve_subject(n)
                                      for n in processed}

            # Populate label table on main thread
            self.app.root.after(0, lambda: self._populate_label_table(states))

            # 3. Time slicing
            time_mode = self._time_mode_var.get()
            sliced = {}
            if time_mode == 'range':
                start_f = int(round(self._range_start_var.get() * fps))
                end_f = int(round(self._range_end_var.get() * fps))
                for name, s in processed.items():
                    sliced[name] = s[start_f:end_f]
                self._log_msg(f"Time range: {self._range_start_var.get():.0f}s "
                              f"- {self._range_end_var.get():.0f}s")
            else:
                sliced = processed

            # 4. Compute matrices
            zero_diag = self._zero_diag_var.get()
            t_mode = self._transition_mode.get()
            self._matrices = {}
            for name, s in sliced.items():
                if t_mode == 'bout':
                    mat, _ = compute_bout_transition_matrix(
                        s, states=states, normalize=True)
                else:
                    mat, _ = compute_transition_matrix(
                        s, states=states, normalize=True,
                        zero_diagonal=zero_diag)
                self._matrices[name] = mat
                self._log_msg(f"  {name}: {len(s)} frames, "
                              f"{len(states)} states")

            # 5. Windowed transitions (if sliding mode)
            self._windowed = {}
            if time_mode == 'sliding':
                win_s = self._win_sec_var.get()
                step_s = self._step_sec_var.get()
                for name, s in processed.items():
                    wresults, _ = compute_windowed_transitions(
                        s, fps, win_s, step_s, states=states,
                        zero_diagonal=zero_diag, mode=t_mode)
                    self._windowed[name] = wresults
                self._log_msg(f"Sliding windows: {win_s:.0f}s window, "
                              f"{step_s:.0f}s step")

            # 5b. Latent state discovery (if enabled and sliding mode)
            discover_latent = self._discover_latent_var.get()
            n_latent = self._n_latent_var.get()
            self._latent_centroids = None
            self._session_latent_map = {}
            self._n_latent = 0
            self._occupancy = {}
            self._pca_model = None
            self._pca_scores = {}
            self._pca_loadings = None
            self._group_occupancy = {}
            self._group_occupancy_sem = {}

            if discover_latent and time_mode == 'sliding' and self._windowed:
                self._log_msg(f"Running k-means clustering (k={n_latent})...")
                centroids, session_latent_map = cluster_transition_matrices(
                    self._windowed, k=n_latent, states=states)
                self._latent_centroids = centroids
                self._session_latent_map = session_latent_map
                self._n_latent = n_latent

                # State occupancy fractions
                self._occupancy = compute_state_occupancy(
                    session_latent_map, n_latent)
                self._log_msg(f"Computed state occupancy for "
                              f"{len(self._occupancy)} sessions")

                # PCA on occupancy
                if len(self._occupancy) >= 2:
                    self._pca_model, self._pca_scores, self._pca_loadings = \
                        pca_on_occupancy(self._occupancy)
                    self._log_msg("PCA on occupancy complete")

                # Group aggregation of occupancy
                if self._key_df is not None:
                    grp_occ = {}  # {treatment: [array, ...]}
                    for name, occ in self._occupancy.items():
                        subj = self._session_subjects.get(name, name)
                        row = self._key_df[self._key_df['Subject'] == subj]
                        if not row.empty:
                            treatment = str(row.iloc[0]['Treatment'])
                            if treatment not in grp_occ:
                                grp_occ[treatment] = []
                            grp_occ[treatment].append(occ)
                    for grp, occs in grp_occ.items():
                        stack = np.stack(occs)
                        self._group_occupancy[grp] = stack.mean(axis=0)
                        self._group_occupancy_sem[grp] = (
                            stack.std(axis=0, ddof=1) / np.sqrt(len(occs))
                            if len(occs) > 1 else np.zeros(n_latent))
            elif discover_latent and time_mode != 'sliding':
                self._log_msg("NOTE: Latent state discovery requires "
                              "sliding windows mode.")

            # 5c. Temporal probability (always computed)
            prob_bin_sec = self._prob_bin_var.get()
            self._temporal_probs = {}
            for name, s in processed.items():
                self._temporal_probs[name] = compute_temporal_probabilities(
                    s, fps, prob_bin_sec, states)
            self._log_msg(f"Temporal probabilities computed "
                          f"(bin={prob_bin_sec:.0f}s)")

            # 6. Group aggregation
            self._group_matrices = {}
            self._group_sem = {}
            self._group_subject_matrices = {}
            if self._key_df is not None:
                groups = {}  # {treatment: [matrix, ...]}
                for name, mat in self._matrices.items():
                    subj = self._session_subjects.get(name, name)
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty:
                        treatment = str(row.iloc[0]['Treatment'])
                        if treatment not in groups:
                            groups[treatment] = []
                        groups[treatment].append(mat)

                for grp, mats in groups.items():
                    stack = np.stack(mats)
                    self._group_matrices[grp] = stack.mean(axis=0)
                    self._group_sem[grp] = (stack.std(axis=0, ddof=1) /
                                            np.sqrt(len(mats))
                                            if len(mats) > 1
                                            else np.zeros_like(mats[0]))
                    self._group_subject_matrices[grp] = list(mats)
                    self._log_msg(f"  Group '{grp}': {len(mats)} subjects")

            self._log_msg("--- Computation complete ---")
            self.app.root.after(0, self._on_compute_done)

        except Exception as e:
            self._log_msg(f"ERROR: {e}")
            traceback.print_exc()
            self.app.root.after(0, self._on_compute_done)

    def _on_compute_done(self):
        self._progress.stop()
        self._stop_btn.config(state='disabled')
        self._read_label_entries()
        # Populate pair selector for timeline
        if self._states:
            pairs = []
            for si in self._states:
                for sj in self._states:
                    if si != sj:
                        pairs.append(f"{self._state_name(si)} -> "
                                     f"{self._state_name(sj)}")
            self._pair_combo['values'] = pairs
            if pairs:
                self._pair_combo.set(pairs[0])
        self._refresh_behav_options()
        sessions_with_video = [
            s['session_name'] for s in self._trans_sessions
            if s.get('video') or s.get('video_path')
        ]
        self._preview_session_combo['values'] = sessions_with_video
        if sessions_with_video and not self._preview_session_var.get():
            self._preview_session_var.set(sessions_with_video[0])
        self._refresh_plot()
        # Persist the last completed run so it can be reloaded without a re-run.
        self._autosave_session()
        # Main page shows the stats; graphs open in the Graph Window automatically.
        self._show_results()

    # ------------------------------------------------------------------
    # Config Save / Load
    # ------------------------------------------------------------------

    def _config_dict(self):
        """Snapshot of all persisted settings (shared by _save_config and the
        saved-session bundle)."""
        return {
            'source':           self._source_var.get(),
            'assign_mode':      self._assign_mode.get(),
            'transition_mode':  self._transition_mode.get(),
            'fps':              self._fps_var.get(),
            'smooth_ms':        self._smooth_ms_var.get(),
            'exclude_noise':    self._exclude_noise_var.get(),
            'exclude_other':    self._exclude_other_var.get(),
            'cache_only':       self._cache_only_var.get(),
            'zero_diag':        self._zero_diag_var.get(),
            'reduce_clusters':  self._reduce_clusters_var.get(),
            'target_clusters':  self._target_clusters_var.get(),
            'prob_bin':         self._prob_bin_var.get(),
            'time_mode':        self._time_mode_var.get(),
            'range_start':      self._range_start_var.get(),
            'range_end':        self._range_end_var.get(),
            'win_sec':          self._win_sec_var.get(),
            'step_sec':         self._step_sec_var.get(),
            'downsample':       self._downsample_var.get(),
            'discover_latent':  self._discover_latent_var.get(),
            'n_latent':         self._n_latent_var.get(),
            'key_file':         self._key_file_var.get(),
            'view':             self._view_var.get(),
            'palette':          self._palette_var.get(),
            'heat_palette':     self._heat_palette_var.get(),
            'sort_mode':        self._sort_mode_var.get(),
            'norm_mode':        self._norm_mode_var.get(),
            'ref_group':        self._ref_group_var.get(),
            'group_palette_mode': self._group_palette_mode_var.get(),
            'stat_test':        self._stat_test_var.get(),
            'stat_correction':  self._stat_correction_var.get(),
            'stat_alpha':       self._stat_alpha_var.get(),
            'error_mode':       self._error_mode_var.get(),
            'show_annot':       self._show_annot_var.get(),
            'classifiers': [
                {
                    'path':           cd.get('_path', ''),
                    'best_thresh':    cd.get('best_thresh'),
                    'min_bout':       cd.get('min_bout'),
                    'min_after_bout': cd.get('min_after_bout'),
                    'max_gap':        cd.get('max_gap'),
                }
                for cd in self._loaded_classifiers
            ],
            'priority_order': list(self._priority_order),
            'session_selection': [
                name for name, var in self._trans_session_checked.items() if var.get()
            ],
        }

    def _save_config(self):
        proj = self.app.current_project_folder.get()
        init_dir = os.path.join(proj, 'transitions') if proj else '/'
        os.makedirs(init_dir, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Transitions Config",
            initialdir=init_dir,
            defaultextension='.json',
            filetypes=[('JSON config', '*.json'), ('All files', '*.*')])
        if not path:
            return
        cfg = self._config_dict()
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
        self._log_msg(f"Config saved to {path}")

    def _load_config(self):
        proj = self.app.current_project_folder.get()
        init_dir = os.path.join(proj, 'transitions') if proj else '/'
        path = filedialog.askopenfilename(
            parent=self,
            title="Load Transitions Config",
            initialdir=init_dir,
            filetypes=[('JSON config', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path) as f:
                cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Load Config", f"Could not read config: {e}", parent=self)
            return

        self._apply_config_dict(cfg)
        self._log_msg(f"Config loaded from {path}")

    def _apply_config_dict(self, cfg):
        """Apply a settings dict (from a saved config OR a saved session) to the
        tab's controls: scalar vars, key file, session selection, and classifiers.
        Shared by _load_config and _load_session. Missing keys are ignored;
        missing classifier files are reported but non-fatal."""
        # Apply scalar settings
        _sv = lambda key, var: var.set(cfg[key]) if key in cfg else None
        _sv('source',          self._source_var)
        _sv('assign_mode',     self._assign_mode)
        _sv('transition_mode', self._transition_mode)
        _sv('fps',             self._fps_var)
        _sv('smooth_ms',       self._smooth_ms_var)
        _sv('exclude_noise',   self._exclude_noise_var)
        _sv('exclude_other',   self._exclude_other_var)
        _sv('cache_only',      self._cache_only_var)
        _sv('zero_diag',       self._zero_diag_var)
        _sv('reduce_clusters', self._reduce_clusters_var)
        _sv('target_clusters', self._target_clusters_var)
        _sv('prob_bin',        self._prob_bin_var)
        _sv('time_mode',       self._time_mode_var)
        _sv('range_start',     self._range_start_var)
        _sv('range_end',       self._range_end_var)
        _sv('win_sec',         self._win_sec_var)
        _sv('step_sec',        self._step_sec_var)
        _sv('downsample',      self._downsample_var)
        _sv('discover_latent', self._discover_latent_var)
        _sv('n_latent',        self._n_latent_var)
        _sv('key_file',        self._key_file_var)
        # Auto-load the restored key file so _key_df / Group column populate.
        if self._key_file_var.get() and os.path.isfile(self._key_file_var.get()):
            self._load_key_file()
        _sv('view',            self._view_var)
        # Migrate any old view name (e.g. 'Occupancy') to its new label.
        self._view_var.set(_VIEW_ALIASES.get(self._view_var.get(), self._view_var.get()))
        self._last_view = self._view_var.get()
        _sv('palette',         self._palette_var)
        _sv('heat_palette',    self._heat_palette_var)
        _sv('sort_mode',       self._sort_mode_var)
        _sv('norm_mode',       self._norm_mode_var)
        _sv('ref_group',       self._ref_group_var)
        _sv('group_palette_mode', self._group_palette_mode_var)
        _sv('stat_test',       self._stat_test_var)
        _sv('stat_correction', self._stat_correction_var)
        _sv('stat_alpha',      self._stat_alpha_var)
        _sv('error_mode',      self._error_mode_var)
        _sv('show_annot',      self._show_annot_var)

        # Restore session selection
        saved_sel = cfg.get('session_selection')
        if saved_sel is not None:
            self._pending_session_selection = set(saved_sel)
            self._apply_pending_session_selection()

        # Re-load classifiers
        clf_entries = cfg.get('classifiers', [])
        if clf_entries:
            self._loaded_classifiers.clear()
            self._priority_order.clear()
            missing = []
            for i, entry in enumerate(clf_entries):
                fpath = entry.get('path', '')
                if not fpath or not os.path.isfile(fpath):
                    missing.append(fpath or f'(entry {i})')
                    continue
                try:
                    cd = _robust_load(fpath)
                    if 'clf_model' not in cd:
                        missing.append(fpath)
                        continue
                    cd['_path'] = fpath
                    for key in ('best_thresh', 'min_bout', 'min_after_bout', 'max_gap'):
                        if entry.get(key) is not None:
                            cd[key] = entry[key]
                    self._loaded_classifiers.append(cd)
                    self._priority_order.append(len(self._loaded_classifiers) - 1)
                except Exception as e:
                    missing.append(f"{fpath} ({e})")

            if missing:
                messagebox.showwarning(
                    "Load Config",
                    "Could not load classifier(s):\n" + "\n".join(missing),
                    parent=self)

            saved_order = cfg.get('priority_order', [])
            if len(saved_order) == len(self._loaded_classifiers):
                self._priority_order[:] = saved_order

            self._update_clf_listbox()

    # ------------------------------------------------------------------
    # Saved sessions (computed results - reload without re-running)
    # ------------------------------------------------------------------

    def _session_bundle(self):
        """Assemble a full-session dict: settings + all computed result members.

        Only plain numpy/dict/str result state is included (no tk vars, figures,
        or XGBoost models). The fitted PCA model is reduced to the one array a
        plot needs (explained_variance_ratio_)."""
        results = {name: getattr(self, '_' + name, None) for name in RESULT_FIELDS}
        pca_evr = None
        _pm = getattr(self, '_pca_model', None)
        if _pm is not None and hasattr(_pm, 'explained_variance_ratio_'):
            pca_evr = np.asarray(_pm.explained_variance_ratio_)
        return {
            'schema_version': SESSION_SCHEMA_VERSION,
            'git_sha':        get_git_sha(),
            'saved_at':       datetime.datetime.now().isoformat(timespec='seconds'),
            'app':            'pixelpaws-transitions',
            'settings':       self._config_dict(),
            'results':        results,
            'pca_evr':        pca_evr,
            'key_df':         getattr(self, '_key_df', None),
        }

    def _save_session(self):
        """Manual save of the current run (settings + results) to one file."""
        if not getattr(self, '_state_seqs', None):
            messagebox.showinfo("Save Session",
                                "Nothing to save yet - run Compute first.",
                                parent=self)
            return
        proj = self.app.current_project_folder.get()
        init_dir = os.path.join(proj, 'transitions') if proj else '/'
        os.makedirs(init_dir, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Transitions Session",
            initialdir=init_dir,
            defaultextension='.pkl',
            filetypes=[('Transitions session', '*.pkl'), ('All files', '*.*')])
        if not path:
            return
        try:
            atomic_pickle_save(self._session_bundle(), path)
            self._log_msg(f"Saved session → {path}")
        except Exception as e:
            messagebox.showerror("Save Session",
                                 f"Could not save session:\n{e}", parent=self)

    def _autosave_session(self):
        """Auto-save the last completed run as a new saved-session entry - unless
        we're mid-load (loading must never spawn a new entry)."""
        if getattr(self, '_loading_session', False):
            return
        if not getattr(self, '_state_seqs', None):
            return
        self._save_session_file('auto')

    def _load_session(self, path=None):
        """Load a saved session and redraw every view - no recompute."""
        if not path:
            proj = self.app.current_project_folder.get()
            init_dir = os.path.join(proj, 'transitions') if proj else '/'
            path = filedialog.askopenfilename(
                parent=self,
                title="Load Transitions Session",
                initialdir=init_dir,
                filetypes=[('Transitions session', '*.pkl'), ('All files', '*.*')])
        if not path:
            return
        try:
            bundle = _robust_load(path)
        except Exception as e:
            messagebox.showerror("Load Session",
                                 f"Could not read session:\n{e}", parent=self)
            return

        # --- validate schema + required results ---
        ver = bundle.get('schema_version') if isinstance(bundle, dict) else None
        if ver is None or ver > SESSION_SCHEMA_VERSION:
            messagebox.showerror(
                "Load Session",
                "This file is not a compatible transitions session "
                f"(schema {ver!r}; this app supports up to "
                f"{SESSION_SCHEMA_VERSION}).",
                parent=self)
            return
        results = bundle.get('results') or {}
        for req in ('state_seqs', 'states', 'state_labels'):
            if not results.get(req):
                messagebox.showerror(
                    "Load Session",
                    f"Session file is missing required results ('{req}').",
                    parent=self)
                return

        # --- restore settings/controls first (this may reload the key file
        #     and classifiers), then let the authoritative results win ---
        settings = bundle.get('settings')
        if isinstance(settings, dict):
            self._apply_config_dict(settings)

        for name in RESULT_FIELDS:
            if name in results:
                setattr(self, '_' + name, results[name])

        # Lightweight PCA stand-in - only .explained_variance_ratio_ is read.
        evr = bundle.get('pca_evr')
        self._pca_model = (SimpleNamespace(explained_variance_ratio_=evr)
                           if evr is not None else None)

        # Key-file grouping: keep the freshly-loaded key_df if the file was
        # still present; otherwise fall back to the one saved in the bundle.
        if getattr(self, '_key_df', None) is None and bundle.get('key_df') is not None:
            self._key_df = bundle['key_df']

        # --- redraw from the loaded state (populate label editor first so
        #     _on_compute_done's _read_label_entries stays consistent).
        #     _loading_session suppresses the autosave that _on_compute_done
        #     triggers, so loading never creates a new saved entry. ---
        self._populate_label_table(self._states)
        self._loading_session = True
        try:
            self._on_compute_done()
        finally:
            self._loading_session = False
        self._log_msg(
            f"Loaded session (saved {bundle.get('saved_at', '?')}) - "
            f"{len(self._state_seqs)} session(s), {len(self._states)} state(s).")

    # ------------------------------------------------------------------
    # Saved-session store + dropdown (multiple timestamped saves)
    # ------------------------------------------------------------------

    _SESSION_AUTO_KEEP = 15   # prune older auto_* saves beyond this (named_* kept)

    def _sessions_dir(self, proj=None):
        proj = proj or self.app.current_project_folder.get()
        if not proj:
            return None
        d = os.path.join(proj, 'transitions', 'sessions')
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            return None
        return d

    def _save_session_file(self, kind, name=None):
        d = self._sessions_dir()
        if not d:
            return
        n = len(getattr(self, '_state_seqs', None) or {})
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        if kind == 'named' and name:
            slug = re.sub(r'[^A-Za-z0-9._-]+', '-', name).strip('-') or 'session'
            fname = f'named_{slug}_{ts}_{n}sess.pkl'
        else:
            fname = f'auto_{ts}_{n}sess.pkl'
        try:
            atomic_pickle_save(self._session_bundle(), os.path.join(d, fname))
        except Exception as e:
            self._log_msg(f"(auto-save of session skipped: {e})")
            return
        self._prune_auto_sessions(d)
        self._refresh_saved_sessions(select_path=os.path.join(d, fname))

    def _prune_auto_sessions(self, d):
        try:
            autos = sorted(f for f in os.listdir(d)
                           if f.startswith('auto_') and f.endswith('.pkl'))
        except Exception:
            return
        for f in autos[:-self._SESSION_AUTO_KEEP]:
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass

    @staticmethod
    def _parse_session_fname(fn):
        """(sort_key, label) for a saved-session filename, or None if unrecognized."""
        _TS = r'(\d{8}-\d{6}(?:-\d{1,6})?)'
        m = re.match(r'auto_' + _TS + r'_(\d+)sess\.pkl$', fn)
        if m:
            ts, n = m.group(1), int(m.group(2))
            return ts, f"{TransitionsTab._fmt_ts(ts)} · {n} session{'s' if n != 1 else ''}"
        m = re.match(r'named_(.+)_' + _TS + r'_(\d+)sess\.pkl$', fn)
        if m:
            slug, ts, n = m.group(1), m.group(2), int(m.group(3))
            return ts, f"★ {slug} · {n} session{'s' if n != 1 else ''}"
        return None

    @staticmethod
    def _fmt_ts(ts):
        try:
            return datetime.datetime.strptime(
                ts[:15], '%Y%m%d-%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return ts

    def _list_saved_sessions(self):
        """[(label, path)] newest-first for the current project (+ legacy file)."""
        items = []
        d = self._sessions_dir()
        if d and os.path.isdir(d):
            for fn in os.listdir(d):
                parsed = self._parse_session_fname(fn)
                if parsed:
                    items.append((parsed[0], parsed[1], os.path.join(d, fn)))
        proj = self.app.current_project_folder.get()
        if proj:
            legacy = os.path.join(proj, 'transitions', '_last_session.pkl')
            if os.path.isfile(legacy):
                items.append(('00000000-000000', '(previous)', legacy))
        items.sort(key=lambda t: t[0], reverse=True)
        return [(label, path) for _sk, label, path in items]

    def _refresh_saved_sessions(self, select_path=None):
        """Repopulate the Saved-sessions combobox from disk (index-keyed)."""
        combo = getattr(self, '_saved_combo', None)
        if combo is None:
            return
        self._saved_items = self._list_saved_sessions()
        labels = [label for label, _ in self._saved_items]
        combo.config(values=labels)
        idx = 0
        if select_path is not None:
            idx = next((i for i, (_l, p) in enumerate(self._saved_items)
                        if os.path.abspath(p) == os.path.abspath(select_path)), 0)
        if labels:
            combo.current(idx)
        else:
            combo.set('')
        state = 'normal' if labels else 'disabled'
        for b in ('_saved_load_btn', '_saved_del_btn'):
            w = getattr(self, b, None)
            if w is not None:
                w.config(state=state)

    def _selected_session_path(self):
        combo = getattr(self, '_saved_combo', None)
        items = getattr(self, '_saved_items', [])
        if combo is None or not items:
            return None
        i = combo.current()
        return items[i][1] if 0 <= i < len(items) else None

    def _load_selected_session(self):
        path = self._selected_session_path()
        if not path:
            messagebox.showinfo("Load session", "No saved session selected.", parent=self)
            return
        self._load_session(path)

    def _save_current_session_named(self):
        if not getattr(self, '_state_seqs', None):
            messagebox.showinfo("Save session", "Run Compute first.", parent=self)
            return
        name = simpledialog.askstring(
            "Save session", "Name for this saved session:", parent=self)
        if name:
            self._save_session_file('named', name)

    def _delete_selected_session(self):
        path = self._selected_session_path()
        if not path:
            return
        label = self._saved_combo.get()
        if messagebox.askyesno("Delete session",
                               f"Delete this saved session?\n\n{label}", parent=self):
            try:
                os.remove(path)
            except OSError as e:
                self._log_msg(f"(could not delete: {e})")
            self._refresh_saved_sessions()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _on_view_changed(self, event=None):
        view = self._view_var.get()
        # Ignore non-selectable section headers - revert to the last real view.
        if view.startswith('-'):
            self._view_var.set(getattr(self, '_last_view', 'State Usage'))
            return
        self._last_view = view
        if view == 'Transition Timeline':
            self._pair_frame.pack(side='left', padx=10)
        else:
            self._pair_frame.pack_forget()
        self._toggle_view_controls()
        self._refresh_plot()

    # ------------------------------------------------------------------
    # Group ordering / colors (shared by all grouped views)
    # ------------------------------------------------------------------

    @staticmethod
    def _treatment_sort_key(t):
        """Dose-response order: vehicle/control first, then ascending numeric dose."""
        import re as _re
        _VEH = {'vehicle', 'veh', 'saline', 'control', 'ctrl', 'acsf',
                'water', 'pbs', 'naive'}
        tl = str(t).lower()
        if any(kw in tl for kw in _VEH):
            return (0, 0.0, tl)
        m = _re.search(r'(\d+\.?\d*)', str(t))
        return (1, float(m.group(1)), tl) if m else (2, 0.0, tl)

    def _ordered_groups(self, keys):
        """Return group keys in the user's chosen order, else dose-response order."""
        keys = list(keys)
        if self._group_order:
            ordered = [g for g in self._group_order if g in keys]
            ordered += [g for g in keys if g not in ordered]  # any not in saved order
            return ordered
        return sorted(keys, key=self._treatment_sort_key)

    # ------------------------------------------------------------------
    # Color model - one STATE palette (states = category) and one GROUP
    # palette (groups = category on the x-axis). See module _OKABE_ITO.
    # ------------------------------------------------------------------

    def _state_palette(self, n=None):
        """Return a list of `n` (default = n states) colors for STATE categories.

        Single source of truth for state coloring (ethogram, stacked-area,
        transition-graph nodes, occupancy-over-time). Honors seaborn names,
        'okabe-ito' (Wong colorblind-safe), and matplotlib cmap names.
        """
        n = int(n if n is not None else max(len(self._states), 1))
        name = self._palette_var.get()
        if name == 'okabe-ito':
            return [_OKABE_ITO[i % len(_OKABE_ITO)] for i in range(n)]
        _SNS = {'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'}
        if name in _SNS:
            return list(sns.color_palette(name, n_colors=n))
        cmap = plt.cm.get_cmap(name, n)
        return [cmap(i) for i in range(n)]

    def _heat_cmap(self):
        """Sequential colormap NAME for probability heatmaps (never silently
        dropped - replaces the old 'YlOrRd' fallback)."""
        name = self._heat_palette_var.get()
        return name if name in _SEQ_CMAPS else 'inferno'

    def _diverging_cmap(self):
        return _DIVERGING_CMAP

    def _norm_matrix(self, mat):
        """Apply the display normalization from `_norm_mode_var`. Matrices are
        stored row-normalized; 'Column-normalized' renormalizes columns for
        display (incoming-transition view). No recompute."""
        m = np.asarray(mat, dtype=float)
        if self._norm_mode_var.get().startswith('Column'):
            csum = m.sum(axis=0, keepdims=True)
            csum[csum == 0] = 1.0
            return m / csum
        return m

    def _group_palette(self, groups=None):
        """Ordered {group: color} for GROUP categories (grouped bars, dot plots,
        timeline, group means). Precedence: explicit override → dose gradient
        (when mode='gradient') → categorical Okabe-Ito by dose index."""
        ordered = self._ordered_groups(
            groups if groups is not None else self._group_matrices.keys())
        mode = self._group_palette_mode_var.get()
        out = {}
        # Vehicle-bucket groups (dose 0) render gray in gradient mode.
        non_veh = [g for g in ordered if self._treatment_sort_key(g)[0] != 0]
        if mode == 'gradient':
            try:
                cmap = plt.cm.get_cmap('viridis')
            except Exception:
                cmap = None
            for g in ordered:
                if g in non_veh and cmap is not None:
                    k = non_veh.index(g)
                    frac = 0.15 + 0.75 * (k / max(1, len(non_veh) - 1))
                    r, gr, b, _ = cmap(frac)
                    out[g] = '#%02x%02x%02x' % (int(r*255), int(gr*255), int(b*255))
                else:
                    out[g] = '#8a8a8a'
        else:
            for i, g in enumerate(ordered):
                out[g] = _OKABE_ITO[(i + 1) % len(_OKABE_ITO)]  # skip black
        # Explicit per-group overrides always win.
        for g in ordered:
            if g in self._group_colors:
                out[g] = self._group_colors[g]
        return out

    def _group_color(self, group, idx=0):
        """Stable per-treatment color from the group palette (dialog-driven)."""
        pal = self._group_palette()
        if group in pal:
            return pal[group]
        return _OKABE_ITO[(idx + 1) % len(_OKABE_ITO)]

    def _set_panel_figsize(self, n_panels, orient='v', base_w=11.0, base_h=5.0):
        """Legacy per-panel sizing hint. In the pop-out the figure is instead
        matched to the canvas widget (see _sync_active_fig_size) so it always
        fills the canvas; kept as a no-op-ish helper for API stability. Uses
        forward=False so it never resizes the window under the user."""
        if not self._fig:
            return
        # When rendering into the pop-out, defer entirely to the widget-size sync
        # (avoids a figure smaller than the canvas leaving a prior view's pixels
        # showing through).
        if self._win_fig is not None and self._fig is self._win_fig:
            return
        n = max(1, int(n_panels))
        if orient == 'v':
            self._fig.set_size_inches(base_w, max(base_h, 2.4 * n), forward=False)
        else:
            self._fig.set_size_inches(max(base_w, 3.6 * n), base_h, forward=False)

    def _sync_active_fig_size(self):
        """Match the active figure's size to its canvas widget so the Agg render
        fills the whole canvas - prevents a prior (larger) view's pixels from
        showing through when the new view's figure is smaller."""
        try:
            widget = self._canvas.get_tk_widget()
            widget.update_idletasks()
            pw, ph = widget.winfo_width(), widget.winfo_height()
            if pw > 1 and ph > 1:
                dpi = self._fig.get_dpi()
                self._fig.set_size_inches(pw / dpi, ph / dpi, forward=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pop-out graph window (roomy, Analysis-style)
    # ------------------------------------------------------------------

    def _open_graph_window(self):
        """Open (or raise) a large resizable window that hosts the results figure.

        While open, self._fig / self._canvas point at the window's figure/canvas,
        so every _plot_* renders into it unchanged."""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showinfo("Graphs", "matplotlib is not available.", parent=self)
            return
        if self._graph_win is not None and self._graph_win.winfo_exists():
            self._graph_win.lift()
            return
        win = tk.Toplevel(self)
        self._graph_win = win
        win.title("Transitions - Graphs")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = int(sw * 0.6), int(sh * 0.8)
        win.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        win.minsize(760, 520)
        win.protocol("WM_DELETE_WINDOW", self._close_graph_window)

        # Two rows so every control stays visible without needing an ultra-wide
        # window: row 1 = view + per-view display controls; row 2 = global time
        # window + action buttons.
        bar = ttk.Frame(win, padding=(8, 6, 8, 2))
        bar.pack(fill='x')
        bar2 = ttk.Frame(win, padding=(8, 0, 8, 6))
        bar2.pack(fill='x')
        self._graph_bar, self._graph_bar2 = bar, bar2
        ttk.Label(bar, text="View:").pack(side='left', padx=(0, 4))
        vcombo = ttk.Combobox(bar, textvariable=self._view_var, state='readonly',
                              width=22, values=list(_VIEW_MENU))
        vcombo.pack(side='left', padx=(0, 8))
        vcombo.bind('<<ComboboxSelected>>', self._on_view_changed)

        # Contextual control frames (shown per view by _toggle_view_controls).
        self._state_pal_frame = ttk.Frame(bar)
        ttk.Label(self._state_pal_frame, text="State colors:").pack(side='left', padx=(0, 2))
        _spc = ttk.Combobox(self._state_pal_frame, textvariable=self._palette_var,
                            state='readonly', width=10,
                            values=['okabe-ito', 'deep', 'colorblind', 'muted', 'bright',
                                    'dark', 'tab10', 'tab20', 'Set1', 'Set2', 'Dark2',
                                    'Paired', 'viridis', 'plasma', 'inferno'])
        _spc.pack(side='left'); _spc.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        self._heat_pal_frame = ttk.Frame(bar)
        ttk.Label(self._heat_pal_frame, text="Heatmap:").pack(side='left', padx=(0, 2))
        _hpc = ttk.Combobox(self._heat_pal_frame, textvariable=self._heat_palette_var,
                            state='readonly', width=9, values=sorted(_SEQ_CMAPS))
        _hpc.pack(side='left'); _hpc.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        self._sort_frame = ttk.Frame(bar)
        ttk.Label(self._sort_frame, text="Sort:").pack(side='left', padx=(0, 2))
        _sc = ttk.Combobox(self._sort_frame, textvariable=self._sort_mode_var,
                           state='readonly', width=15,
                           values=['Usage ↓', 'Group difference', 'Fixed'])
        _sc.pack(side='left'); _sc.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        self._norm_frame = ttk.Frame(bar)
        ttk.Label(self._norm_frame, text="Norm:").pack(side='left', padx=(0, 2))
        _nc = ttk.Combobox(self._norm_frame, textvariable=self._norm_mode_var,
                           state='readonly', width=16,
                           values=['Row-normalized', 'Column-normalized'])
        _nc.pack(side='left'); _nc.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        self._ref_frame = ttk.Frame(bar)
        ttk.Label(self._ref_frame, text="Reference:").pack(side='left', padx=(0, 2))
        self._ref_combo = ttk.Combobox(self._ref_frame, textvariable=self._ref_group_var,
                                        state='readonly', width=12, values=['(auto)'])
        self._ref_combo.pack(side='left')
        self._ref_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        # Statistics block (test + correction + alpha + error bars)
        self._stats_frame = ttk.Frame(bar)
        ttk.Label(self._stats_frame, text="Test:").pack(side='left', padx=(0, 2))
        _tc = ttk.Combobox(self._stats_frame, textvariable=self._stat_test_var,
                           state='readonly', width=17,
                           values=['Auto', 'Kruskal-Wallis', 'Mann-Whitney vs ref',
                                   'Parametric (t/ANOVA)'])
        _tc.pack(side='left'); _tc.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())
        ttk.Label(self._stats_frame, text="Correct:").pack(side='left', padx=(6, 2))
        _cc = ttk.Combobox(self._stats_frame, textvariable=self._stat_correction_var,
                           state='readonly', width=10,
                           values=['BH-FDR', 'Bonferroni', 'None'])
        _cc.pack(side='left'); _cc.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())
        ttk.Label(self._stats_frame, text="α:").pack(side='left', padx=(6, 2))
        _ae = ttk.Entry(self._stats_frame, textvariable=self._stat_alpha_var, width=5)
        _ae.pack(side='left'); _ae.bind('<Return>', lambda e: self._refresh_plot())
        ttk.Label(self._stats_frame, text="Error:").pack(side='left', padx=(6, 2))
        _ec = ttk.Combobox(self._stats_frame, textvariable=self._error_mode_var,
                           state='readonly', width=15,
                           values=['SEM', '95% CI (bootstrap)'])
        _ec.pack(side='left'); _ec.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        # Behavior picker (contextual - Behavior × Group only).
        self._trend_frame = ttk.Frame(bar)
        ttk.Label(self._trend_frame, text="Behavior:").pack(side='left', padx=(0, 2))
        self._behav_combo = ttk.Combobox(self._trend_frame, textvariable=self._behav_var,
                                         state='readonly', width=20,
                                         values=['All behaviors (grid)'])
        self._behav_combo.pack(side='left')
        self._behav_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        # GLOBAL analysis window + bin + Update (applies to every view; restricts
        # the analysis to a time slice and re-derives occupancy/matrices live).
        # Lives on row 2 with the action buttons.
        self._window_frame = ttk.Frame(bar2)
        ttk.Label(self._window_frame, text="Window (min):").pack(side='left', padx=(0, 2))
        _wf = ttk.Entry(self._window_frame, textvariable=self._trend_tmin_var, width=5)
        _wf.pack(side='left'); _wf.bind('<Return>', lambda e: self._refresh_plot())
        ttk.Label(self._window_frame, text="-").pack(side='left', padx=1)
        _wt = ttk.Entry(self._window_frame, textvariable=self._trend_tmax_var, width=5)
        _wt.pack(side='left'); _wt.bind('<Return>', lambda e: self._refresh_plot())
        self._bin_label = ttk.Label(self._window_frame, text="Bin (s):")
        self._bin_label.pack(side='left', padx=(8, 2))
        self._bin_spin = ttk.Spinbox(self._window_frame, from_=0, to=600, increment=5,
                                     textvariable=self._prob_bin_var, width=6,
                                     command=self._refresh_plot)
        self._bin_spin.pack(side='left')
        self._bin_spin.bind('<Return>', lambda e: self._refresh_plot())
        self._realtime_chk = ttk.Checkbutton(
            self._window_frame, text="Real time", variable=self._realtime_axis_var,
            command=self._refresh_plot)
        self._realtime_chk.pack(side='left', padx=(8, 0))
        ttk.Button(self._window_frame, text="↻ Update",
                   command=self._refresh_plot).pack(side='left', padx=(6, 0))
        self._window_frame.pack(side='left', padx=(0, 10))

        # Display toggles stay on row 1 next to the view's display controls.
        ttk.Checkbutton(bar, text="Cell values", variable=self._show_annot_var,
                        command=self._refresh_plot).pack(side='left', padx=4)
        ttk.Checkbutton(bar, text="Sig. markers", variable=self._show_sig_var,
                        command=self._refresh_plot).pack(side='left', padx=4)
        # Output actions sit at the far right of row 2 (first-packed side='right' =
        # right-most): Save Figure, then Export Data, then the editors.
        ttk.Button(bar2, text="💾 Save Figure",
                   command=self._save_graph_figure).pack(side='right', padx=3)
        ttk.Button(bar2, text="⬇ Export Data (CSV)",
                   command=self._export_view_csv).pack(side='right', padx=3)
        ttk.Button(bar2, text="ℹ Methods",
                   command=self._open_methods_dialog).pack(side='right', padx=3)
        ttk.Button(bar2, text="Axis…",
                   command=self._open_axis_dialog).pack(side='right', padx=3)
        ttk.Button(bar2, text="Order / Colors…",
                   command=self._open_group_order_dialog).pack(side='right', padx=3)

        # Size the window so the widest per-view control bar is fully visible.
        # Measure each view's actual control set (not all frames at once, which
        # over-counts) and take the max, then widen the window to fit - capped to
        # the screen. minsize stops it being dragged narrower than the controls.
        self._refresh_ref_groups()
        _saved_view = self._view_var.get()
        _max_bar = bar2.winfo_reqwidth()
        for _lbl in _VIEW_MENU:
            if _lbl.startswith('-'):
                continue
            self._view_var.set(_lbl)
            self._toggle_view_controls()
            win.update_idletasks()
            _max_bar = max(_max_bar, bar.winfo_reqwidth(), bar2.winfo_reqwidth())
        self._view_var.set(_saved_view)
        bar_w = _max_bar + 28
        fit_w = min(int(sw * 0.95), max(w, bar_w))
        win.geometry(f"{fit_w}x{h}+{max(0, (sw - fit_w) // 2)}+{max(0, (sh - h) // 2)}")
        win.minsize(min(bar_w, int(sw * 0.95)), 520)

        self._refresh_ref_groups()
        self._refresh_behav_options()
        self._toggle_view_controls()

        canvas_frame = ttk.Frame(win)
        canvas_frame.pack(fill='both', expand=True)
        self._win_fig = plt.figure(figsize=(11, 7), constrained_layout=True)
        self._win_canvas = FigureCanvasTkAgg(self._win_fig, master=canvas_frame)
        self._win_canvas.get_tk_widget().pack(fill='both', expand=True)
        tb = NavigationToolbar2Tk(self._win_canvas, win)
        tb.update()
        _bind_tight_layout_on_resize(self._win_canvas, self._win_fig)

        # Make the window figure the active render target.
        self._fig = self._win_fig
        self._canvas = self._win_canvas
        win.after(50, self._refresh_plot)

    def _refresh_ref_groups(self):
        """Populate the Reference-group combo from the current groups."""
        if not hasattr(self, '_ref_combo') or not self._ref_combo.winfo_exists():
            return
        try:
            groups = self._ordered_groups(self._group_matrices.keys())
        except Exception:
            groups = []
        self._ref_combo['values'] = ['(auto)'] + [str(g) for g in groups]

    def _refresh_behav_options(self):
        """Populate the Behavior × Group behavior picker (named behaviors only)."""
        if not hasattr(self, '_behav_combo') or not self._behav_combo.winfo_exists():
            return
        names = [self._state_name(s) for s in self._states if s != 0]
        self._behav_combo['values'] = ['All behaviors (grid)'] + names
        if self._behav_var.get() not in ('All behaviors (grid)', *names):
            self._behav_var.set('All behaviors (grid)')

    def _refresh_stats_table(self):
        """Main-page stats: per-behavior % time by group (mean ± SEM) + per-state
        FDR significance. The graphs themselves live in the Graph Window."""
        tree = getattr(self, '_stats_tree', None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        occ, groups = (self._state_usage_occ()
                       if getattr(self, '_state_seqs', None) else ({}, {}))
        if not occ:
            tree['columns'] = ('info',)
            tree.heading('info', text='')
            tree.column('info', width=420, anchor='w')
            tree.insert('', 'end',
                        values=("Run Compute (or load a session) to see stats.",))
            return
        gkeys = self._ordered_groups([g for g in groups if g is not None])
        vals = {}
        for g in gkeys:
            sess = groups.get(g, [])
            if sess:
                vals[g] = np.column_stack([occ[s] for s in sess])
        gkeys = [g for g in gkeys if g in vals]
        if not vals:
            vals = {'All': np.column_stack([occ[s] for s in occ])}
            gkeys = ['All']
        n_states = len(self._states)
        overall = np.array([np.concatenate([vals[g][si] for g in gkeys]).mean()
                            for si in range(n_states)])
        order = [si for si in np.argsort(-overall) if self._states[si] != 0]
        stats = self._perstate_stats(vals) if len(gkeys) >= 2 else None

        cols = ['behavior'] + [f'g{i}' for i in range(len(gkeys))] + ['q', 'sig']
        tree['columns'] = cols
        tree.heading('behavior', text='Behavior')
        tree.column('behavior', width=150, anchor='w')
        for i, g in enumerate(gkeys):
            tree.heading(f'g{i}', text=f'{g} (n={vals[g].shape[1]})')
            tree.column(f'g{i}', width=110, anchor='center')
        tree.heading('q', text='q'); tree.column('q', width=70, anchor='center')
        tree.heading('sig', text='sig'); tree.column('sig', width=50, anchor='center')

        for si in order:
            row = [self._state_name(self._states[si])]
            for g in gkeys:
                arr = vals[g][si]
                m = float(np.mean(arr))
                sem = (float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                       if len(arr) > 1 else 0.0)
                row.append(f"{m:.1f} ± {sem:.1f}")
            if stats and si < len(stats):
                q = stats[si].get('q')
                row.append(f"{q:.2g}" if q is not None and not np.isnan(q) else '')
                row.append(stats[si].get('star', ''))
            else:
                row += ['', '']
            tree.insert('', 'end', values=row)

    def _show_results(self):
        """Refresh the main-page stats and auto-open the Graph Window for plots."""
        self._refresh_stats_table()
        if MATPLOTLIB_AVAILABLE and getattr(self, '_state_seqs', None):
            self._open_graph_window()

    def _toggle_view_controls(self):
        """Show only the controls relevant to the current view (pop-out only)."""
        if self._graph_win is None or not self._graph_win.winfo_exists():
            return
        view = _VIEW_ALIASES.get(self._view_var.get(), self._view_var.get())
        order = [self._state_pal_frame, self._heat_pal_frame, self._sort_frame,
                 self._norm_frame, self._ref_frame, self._stats_frame,
                 self._trend_frame]
        show = {
            self._trend_frame: view == 'Behavior × Group',
            self._state_pal_frame: view in (
                'State Usage', 'State Composition', 'Ethogram',
                'Composition Over Time', 'Occupancy Over Time',
                'Transition Graph', 'Group Transition Graphs'),
            self._heat_pal_frame: view in ('Transition Matrix', 'Group Matrices'),
            self._sort_frame: view in ('State Usage', 'State Composition'),
            self._norm_frame: view in (
                'Transition Matrix', 'Group Matrices', 'Transition Difference'),
            self._ref_frame: view in ('Transition Difference', 'State Usage'),
            self._stats_frame: view in ('State Usage', 'Transition Difference'),
        }
        for f in order:
            f.pack_forget()
        for f in order:
            if show.get(f):
                f.pack(side='left', padx=(0, 6))
        # The Bin control only affects time-resolved views; bind it to the right
        # variable per view (the ethogram has its own resolution bin, 0 = Full) and
        # grey it out where it has no meaning so it never looks ignored.
        if getattr(self, '_bin_spin', None) is not None:
            if view == 'Ethogram':
                self._bin_spin.configure(textvariable=self._etho_bin_var,
                                         state='normal')
                self._bin_label.configure(text="Bin (s, 0=full):")
            elif view in ('Composition Over Time', 'Behavior × Group'):
                self._bin_spin.configure(textvariable=self._prob_bin_var,
                                         state='normal')
                self._bin_label.configure(text="Bin (s):")
            else:
                self._bin_spin.configure(textvariable=self._prob_bin_var,
                                         state='disabled')
                self._bin_label.configure(text="Bin (s):")
        self._refresh_ref_groups()

    def _close_graph_window(self):
        """Restore the inline figure as the active target and close the window."""
        self._fig = self._inline_fig
        self._canvas = self._inline_canvas
        self._win_fig = None
        self._win_canvas = None
        if self._graph_win is not None:
            try:
                self._graph_win.destroy()
            except Exception:
                pass
        self._graph_win = None
        self._refresh_plot()

    def _save_graph_figure(self):
        """Save the current figure to PNG/PDF/SVG at high DPI."""
        if not self._fig:
            return
        proj = self.app.current_project_folder.get()
        init_dir = os.path.join(proj, 'analysis', 'transitions') if proj else '/'
        try:
            os.makedirs(init_dir, exist_ok=True)
        except Exception:
            init_dir = '/'
        path = filedialog.asksaveasfilename(
            parent=self._graph_win or self,
            title="Save Figure",
            initialdir=init_dir,
            initialfile=f"{self._view_var.get().replace(' ', '_').lower()}",
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if not path:
            return
        try:
            self._fig.savefig(path, dpi=200, bbox_inches='tight')
            self._log_msg(f"Saved figure → {path}")
        except Exception as e:
            messagebox.showerror("Save Figure", f"Could not save:\n{e}",
                                 parent=self._graph_win or self)

    # ------------------------------------------------------------------
    # Methods description (how the current graph is computed)
    # ------------------------------------------------------------------

    def _methods_text(self, view):
        """Plain-language description of how `view` is calculated, with the
        current settings substituted in. Covers every view (generic fallback)."""
        view = _VIEW_ALIASES.get(view, view)
        groups = list(self._group_matrices.keys()) if self._group_matrices else []
        ref = self._reference_group(groups) if groups else '(vehicle)'
        n_groups = max(len(groups), 2)
        _test = self._effective_test(n_groups)
        test_desc = {
            'kruskal': "the Kruskal-Wallis one-way analysis of variance by ranks "
                       "(a nonparametric omnibus test across all groups)",
            'anova':   "a one-way analysis of variance (ANOVA)",
            'mw':      f"two-sided Mann-Whitney U tests against the {ref} group "
                       "(each other group vs. reference)",
            'welch':   f"Welch's unequal-variance t-tests against the {ref} group",
        }.get(_test, "a nonparametric test")
        corr = {'BH-FDR': "the Benjamini-Hochberg false-discovery-rate (FDR) procedure",
                'Bonferroni': "the Bonferroni method",
                'None': "no correction"}.get(
                    self._stat_correction_var.get(), "the Benjamini-Hochberg procedure")
        err = ("the 95% confidence interval of the group mean, estimated by "
               "percentile bootstrap (1,000 resamples of animals)"
               if self._error_mode_var.get().startswith('95')
               else "the standard error of the mean (SEM) across animals")
        norm = ("For display, the matrix shown here is column-normalized (incoming "
                "transitions): each column was rescaled to sum to unity. "
                if self._norm_mode_var.get().startswith('Column')
                else "")
        zerodiag = (" Self-transitions were additionally set to zero prior to display."
                    if self._zero_diag_var.get() else "")
        tmode = _g(self._transition_mode, 'bout')
        transition_mode = (
            "at the bout level: each state sequence was first reduced to an ordered "
            "list of behavioural bouts (maximal contiguous runs of a single state), and "
            "transitions were tallied between temporally adjacent bouts; consequently "
            "self-transitions cannot occur and the matrix diagonal is identically zero"
            if tmode == 'bout' else
            "at the frame level: transitions were tallied between every pair of "
            "consecutive frames, so the diagonal quantifies frame-to-frame persistence "
            "within a behaviour")
        fields = {
            'error': err, 'sort': self._sort_mode_var.get(),
            'test': test_desc, 'correction': corr,
            'alpha': self._stat_alpha_var.get(), 'ref': ref,
            'norm': norm, 'zerodiag': zerodiag, 'transition_mode': transition_mode,
            'bin': _g(self._prob_bin_var, 30), 'win': _g(self._win_sec_var, 30),
            'step': _g(self._step_sec_var, 10),
        }
        body = _METHODS_STATIC.get(view,
            f"{view}: a computed summary of the transition results.")
        try:
            body = body.format(**fields)
        except Exception:
            pass

        assign = ("each frame was assigned to the highest-priority active classifier, "
                  "with ties among simultaneously-active behaviours resolved by a "
                  "user-defined priority ranking"
                  if self._assign_mode.get() == 'priority'
                  else "each frame was assigned, in a winner-take-all fashion, to the "
                       "classifier with the maximum probability among those exceeding "
                       "their detection threshold")
        smooth = int(_g(self._smooth_ms_var, 100))
        ds = ""
        if _g(self._downsample_var, False):
            ds = (" State sequences were subsequently down-sampled to ~20 Hz by "
                  "majority vote (modal state over non-overlapping frame windows).")
        cap = ""
        if _g(self._dur_limit_var, False):
            cap = (" Analysis was restricted to the first "
                   f"{float(_g(self._dur_limit_min, 0)):.0f} min of each recording.")
        excl = (" Frames labelled 'Other' were excluded from all occupancy and "
                "transition calculations, so metrics are defined over the named "
                "behaviours only."
                if self._exclude_other_var.get() else "")
        try:
            fps = f"{float(self._effective_fps):.1f}"
        except Exception:
            fps = "the recording"
        preamble = (
            "BEHAVIOURAL STATE ASSIGNMENT AND PRE-PROCESSING\n"
            "Behaviour was classified frame-by-frame using the loaded set of XGBoost "
            "(gradient-boosted decision-tree) classifiers, one per behaviour; each "
            "classifier produced a per-frame probability that was thresholded at its "
            "operating point to yield a binary detection. Per-behaviour detections were "
            "refined by morphological bout filtering, in which bouts shorter than the "
            "classifier's minimum-bout duration were discarded and inter-bout gaps "
            "shorter than its maximum-gap duration were bridged, suppressing spurious "
            "single-frame events. A single behavioural state was then assigned to every "
            f"frame - {assign} - and frames lacking any active behaviour were labelled "
            f"'Other'. Residual state flicker was removed by merging any bout shorter "
            f"than {smooth} ms into the preceding state.{ds}{cap} All time-based "
            f"metrics were computed at an effective frame rate of {fps} fps. Animals "
            f"were assigned to treatment groups using the project key file.{excl} "
            "Colour palettes, group ordering and axis limits are display options only "
            "and do not affect any reported value.\n"
            "\n────────────────────────────────────────\n\n"
        )
        return preamble + body

    def _open_methods_dialog(self):
        view = self._view_var.get()
        dlg = tk.Toplevel(self._graph_win or self)
        dlg.title(f"Methods - {view}")
        dlg.transient(self._graph_win or self)
        try:
            dlg.geometry("720x580")
        except Exception:
            pass
        txt = scrolledtext.ScrolledText(dlg, wrap='word', font=(FONT_FAMILY, 10),
                                        padx=8, pady=8)
        txt.pack(fill='both', expand=True)
        txt.insert('1.0', self._methods_text(view))
        txt.config(state='disabled')
        btns = ttk.Frame(dlg, padding=6); btns.pack(fill='x')

        def _copy():
            try:
                dlg.clipboard_clear(); dlg.clipboard_append(self._methods_text(view))
            except Exception:
                pass
        ttk.Button(btns, text="Copy", command=_copy).pack(side='right', padx=3)
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side='right', padx=3)

    # ------------------------------------------------------------------
    # Export the current view's underlying data as CSV
    # ------------------------------------------------------------------

    def _view_dataframe(self, view):
        """Return a tidy DataFrame of the numbers behind `view`, or None."""
        view = _VIEW_ALIASES.get(view, view)
        states = list(self._states)
        labels = [self._state_name(s) for s in states]

        # --- per-animal occupancy (composition-style views) ---
        def _occ_long():
            occ, groups = self._state_usage_occ()
            if not occ:
                return None
            rows = []
            for name, pcts in occ.items():
                subj = self._session_subjects.get(name, name)
                grp = self._session_group(name)
                for si, s in enumerate(states):
                    if s == 0:
                        continue
                    rows.append({'Subject': subj, 'Group': grp,
                                 'Behavior': self._state_name(s),
                                 'Percent_time': round(float(pcts[si]), 4)})
            return pd.DataFrame(rows) if rows else None

        # --- group-mean occupancy over time (live-binned at the current Bin) ---
        def _temporal_long(single_behavior=None):
            tprob = self._temporal_probs_current()
            if not tprob:
                return None
            gmap = {n: self._session_group(n) for n in tprob}
            gkeys = self._ordered_groups({g for g in gmap.values() if g}) or ['All']
            if gkeys == ['All']:
                gmap = {n: 'All' for n in tprob}
            gsess = {g: [n for n, gg in gmap.items() if gg == g] for g in gkeys}
            sidx = {s: i for i, s in enumerate(states)}
            beh = ([s for s in states if s != 0] if single_behavior is None
                   else [s for s in states
                         if self._state_name(s) == single_behavior])
            rows = []
            for sid in beh:
                col = sidx[sid]
                for g in gkeys:
                    sess = gsess.get(g, [])
                    if not sess:
                        continue
                    all_t = sorted({round(float(c) / 60.0, 6)
                                    for n in sess for c in tprob[n][0]})
                    tpos = {t: k for k, t in enumerate(all_t)}
                    mat = np.full((len(sess), len(all_t)), np.nan)
                    for ri, n in enumerate(sess):
                        centers, pm = tprob[n]
                        for k, c in enumerate(centers):
                            j = tpos.get(round(float(c) / 60.0, 6))
                            if j is not None and col < pm.shape[1]:
                                mat[ri, j] = pm[k, col]
                    nv = np.sum(~np.isnan(mat), axis=0)
                    mean = np.nanmean(mat, axis=0)
                    sem = np.where(nv > 1, np.nanstd(mat, axis=0, ddof=1) / np.sqrt(nv), 0.0)
                    for k, t in enumerate(all_t):
                        rows.append({'Behavior': self._state_name(sid), 'Group': g,
                                     'Time_min': t, 'Mean': round(float(mean[k]), 5),
                                     'SEM': round(float(sem[k]), 5)})
            return pd.DataFrame(rows) if rows else None

        if view in ('State Usage', 'State Composition', 'Ethogram'):
            return _occ_long()
        if view == 'Transition Matrix':
            mats = self._matrices_win() or self._matrices
            if not mats:
                return None
            mean_mat = self._norm_matrix(np.mean(np.stack(list(mats.values())), 0))
            return pd.DataFrame(mean_mat, index=labels, columns=labels)
        if view in ('Group Matrices', 'Transition Graph', 'Group Transition Graphs'):
            gm = self._group_matrices_win()[0] or self._group_matrices
            if not gm:
                return None
            rows = []
            for g in self._ordered_groups(gm.keys()):
                m = np.asarray(gm[g])
                for i, si in enumerate(states):
                    for j, sj in enumerate(states):
                        rows.append({'Group': g, 'From': self._state_name(si),
                                     'To': self._state_name(sj),
                                     'Probability': round(float(m[i, j]), 5)})
            return pd.DataFrame(rows)
        if view == 'Transition Difference':
            gm, gsub = self._group_matrices_win()
            gm = gm or self._group_matrices
            gsub = gsub or self._group_subject_matrices
            if not gm:
                return None
            groups = self._ordered_groups(gm.keys())
            ref = self._reference_group(groups)
            nonref = [g for g in groups if g != ref]
            if not ref or not nonref:
                return None
            refm = np.asarray(gm[ref])
            rows = []
            for g in nonref:
                d = np.asarray(gm[g]) - refm
                q = (self._percell_stats(gsub, ref, g)
                     if ref in gsub and g in gsub else None)
                for i, si in enumerate(states):
                    for j, sj in enumerate(states):
                        rows.append({'Group': g, 'Reference': ref,
                                     'From': self._state_name(si),
                                     'To': self._state_name(sj),
                                     'Diff': round(float(d[i, j]), 5),
                                     'q': (round(float(q[i, j]), 5)
                                           if q is not None else '')})
            return pd.DataFrame(rows)
        if view == 'Composition Over Time':
            return _temporal_long()
        if view == 'Behavior × Group':
            sel = _g(self._behav_var, 'All behaviors (grid)')
            return _temporal_long(None if sel == 'All behaviors (grid)' else sel)
        if view == 'Occupancy Over Time':
            # group-mean sliding-window occupancy (only if _windowed exists)
            if not self._windowed:
                return _temporal_long()
            return _temporal_long()
        if view == 'Transition Timeline':
            if not self._windowed:
                return None
            pair = _g(self._pair_combo, '') if hasattr(self, '_pair_combo') else ''
            if ' -> ' not in pair:
                return None
            fn, tn = pair.split(' -> ', 1)
            idx = {self._state_name(s): i for i, s in enumerate(states)}
            if fn not in idx or tn not in idx:
                return None
            fi, ti = idx[fn], idx[tn]
            gmap = {n: self._session_group(n) for n in self._windowed}
            rows = []
            for name, wres in self._windowed.items():
                grp = gmap.get(name)
                for t, m in wres:
                    rows.append({'Group': grp, 'Session': name,
                                 'Time_s': float(t), 'From': fn, 'To': tn,
                                 'P': round(float(np.asarray(m)[fi, ti]), 5)})
            return pd.DataFrame(rows) if rows else None
        # Fallback: per-animal occupancy.
        return _occ_long()

    def _export_view_csv(self):
        view = self._view_var.get()
        try:
            df = self._view_dataframe(view)
        except Exception as e:
            messagebox.showerror("Export Data", f"Could not build data:\n{e}",
                                 parent=self._graph_win or self)
            return
        if df is None or len(df) == 0:
            messagebox.showinfo("Export Data",
                                f"No tabular data available for '{view}'.\n"
                                "Run Compute (or load a session) first.",
                                parent=self._graph_win or self)
            return
        init_dir = self._output_dir()
        square = view in ('Transition Matrix',)
        path = filedialog.asksaveasfilename(
            parent=self._graph_win or self,
            title="Export Data (CSV)",
            initialdir=init_dir,
            initialfile=f"{view.replace(' ', '_').replace('×', 'x').lower()}_data.csv",
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if not path:
            return
        try:
            df.to_csv(path, index=square)
            self._log_msg(f"Exported data → {path}")
            messagebox.showinfo("Export Data", f"Saved:\n{path}",
                                parent=self._graph_win or self)
        except Exception as e:
            messagebox.showerror("Export Data", f"Could not save:\n{e}",
                                 parent=self._graph_win or self)

    # ------------------------------------------------------------------
    # Group order / colors dialog
    # ------------------------------------------------------------------

    # Colormaps offered for the dose gradient (mirrors the Analysis tab).
    _GRAD_CMAPS = ['viridis', 'plasma', 'magma', 'inferno', 'cividis',
                   'Blues', 'Purples', 'Oranges', 'Greens', 'Reds',
                   'YlOrRd', 'YlGnBu', 'PuRd', 'BuGn', 'RdPu', 'GnBu',
                   'cool', 'hot', 'copper']

    def _open_group_order_dialog(self):
        """Reorder groups and set colors - Individual (per-group pick) or a dose
        Gradient (colormap + direction + vehicle), with a live preview. Mirrors
        the Analysis tab's order/colors dialog."""
        from tkinter import colorchooser
        import matplotlib.colors as mcolors
        groups = self._ordered_groups(self._group_matrices.keys()
                                      or {g for g in self._session_subjects
                                          for g in [self._session_group(g)] if g})
        groups = [g for g in groups if g is not None]
        if not groups:
            messagebox.showinfo("Order / Colors",
                                "No treatment groups available (load a key file and "
                                "compute first).", parent=self._graph_win or self)
            return

        dlg = tk.Toplevel(self._graph_win or self)
        dlg.title("Group order & colors")
        dlg.transient(self._graph_win or self)
        dlg.grab_set()

        order = list(groups)                        # live order (drag / Up-Down)
        picked = dict(self._group_colors)           # individual overrides {group: hex}
        color_mode = tk.StringVar(value=self._group_palette_mode_var.get() or 'categorical')
        grad_cmap = tk.StringVar(value='viridis')
        grad_dir = tk.StringVar(value='light_to_dark')
        _auto_veh = next((g for g in order if self._treatment_sort_key(g)[0] == 0), order[0])
        veh_var = tk.StringVar(value=_auto_veh)

        # ── Order (drag to reorder) ───────────────────────────────────────
        top = ttk.LabelFrame(dlg, text="Order  (drag to reorder - top = first)", padding=6)
        top.pack(fill='x', padx=8, pady=(8, 4))
        lb = tk.Listbox(top, height=min(8, len(order)), width=24,
                        selectmode='browse', exportselection=False)
        for g in order:
            lb.insert('end', g)
        lb.pack(side='left', fill='x', expand=True)
        ud = ttk.Frame(top); ud.pack(side='left', padx=6)

        def _sync_from_lb():
            order[:] = [lb.get(i) for i in range(lb.size())]
            _refresh_swatches()

        def _move(delta):
            sel = lb.curselection()
            if not sel:
                return
            i = sel[0]; j = i + delta
            if not (0 <= j < lb.size()):
                return
            it = lb.get(i); lb.delete(i); lb.insert(j, it)
            lb.selection_set(j); _sync_from_lb()

        ttk.Button(ud, text="↑", width=3, command=lambda: _move(-1)).pack(pady=2)
        ttk.Button(ud, text="↓", width=3, command=lambda: _move(1)).pack(pady=2)

        def _drag_start(e):
            i = lb.nearest(e.y); lb.selection_clear(0, 'end')
            lb.selection_set(i); lb.activate(i)
            lb.drag_data = {'index': i, 'item': lb.get(i)}

        def _drag_motion(e):
            i = lb.nearest(e.y)
            dd = getattr(lb, 'drag_data', None)
            if dd and i != dd['index']:
                lb.delete(dd['index']); lb.insert(i, dd['item'])
                dd['index'] = i; lb.selection_clear(0, 'end'); lb.selection_set(i)
                _sync_from_lb()
        lb.bind('<Button-1>', _drag_start)
        lb.bind('<B1-Motion>', _drag_motion)

        # ── Color mode ────────────────────────────────────────────────────
        mode_row = ttk.Frame(dlg); mode_row.pack(fill='x', padx=8)
        ttk.Label(mode_row, text="Colors:").pack(side='left')
        ttk.Radiobutton(mode_row, text="Individual", variable=color_mode,
                        value='categorical').pack(side='left', padx=4)
        ttk.Radiobutton(mode_row, text="Gradient (dose)", variable=color_mode,
                        value='gradient').pack(side='left', padx=4)

        container = ttk.Frame(dlg); container.pack(fill='x', padx=8, pady=4)

        # ── Individual: per-group swatch + Pick… ──────────────────────────
        indiv = ttk.Frame(container)
        indiv_sw = {}

        def _default_hex(g):
            return mcolors.to_hex(self._group_color(g, order.index(g) if g in order else 0))

        for g in groups:
            row = ttk.Frame(indiv); row.pack(fill='x', pady=1)
            sw = tk.Label(row, width=3, relief='solid', bd=1); sw.pack(side='left', padx=(0, 6))
            indiv_sw[g] = sw
            ttk.Label(row, text=str(g), width=14).pack(side='left')

            def _pick(gg=g):
                init = picked.get(gg) or _default_hex(gg)
                _, hx = colorchooser.askcolor(color=init, parent=dlg,
                                              title=f"Color for {gg}")
                if hx:
                    picked[gg] = hx; _refresh_swatches()
            ttk.Button(row, text="Pick…", width=6, command=_pick).pack(side='left', padx=2)

        # ── Gradient: colormap + direction + vehicle + preview ────────────
        grad = ttk.Frame(container)
        r1 = ttk.Frame(grad); r1.pack(fill='x', pady=2)
        ttk.Label(r1, text="Colormap:", width=11).pack(side='left')
        ttk.Combobox(r1, textvariable=grad_cmap, values=self._GRAD_CMAPS,
                     state='readonly', width=12).pack(side='left')
        r2 = ttk.Frame(grad); r2.pack(fill='x', pady=2)
        ttk.Label(r2, text="Direction:", width=11).pack(side='left')
        ttk.Radiobutton(r2, text="Light → Dark (↑dose)", variable=grad_dir,
                        value='light_to_dark').pack(side='left')
        ttk.Radiobutton(r2, text="Dark → Light", variable=grad_dir,
                        value='dark_to_light').pack(side='left', padx=8)
        r3 = ttk.Frame(grad); r3.pack(fill='x', pady=2)
        ttk.Label(r3, text="Vehicle:", width=11).pack(side='left')
        ttk.Combobox(r3, textvariable=veh_var, values=list(order),
                     state='readonly', width=12).pack(side='left')
        ttk.Label(r3, text="→ neutral gray", foreground='gray').pack(side='left', padx=4)
        ttk.Label(grad, text="Preview:").pack(anchor='w', pady=(4, 1))
        grad_sw = {}
        for g in groups:
            grow = ttk.Frame(grad); grow.pack(fill='x', pady=1)
            gs = tk.Label(grow, width=3, relief='solid', bd=1); gs.pack(side='left', padx=(0, 6))
            grad_sw[g] = gs
            ttk.Label(grow, text=str(g)).pack(side='left')

        def _grad_colors():
            veh = veh_var.get()
            try:
                cmap = plt.get_cmap(grad_cmap.get())
            except Exception:
                return {}
            non_veh = [g for g in order if g != veh]
            n = len(non_veh)
            pos = (np.linspace(0.35, 0.90, max(n, 1)) if grad_dir.get() == 'light_to_dark'
                   else np.linspace(0.90, 0.35, max(n, 1)))
            gc = {g: mcolors.to_hex(cmap(p)) for g, p in zip(non_veh, pos)}
            gc[veh] = '#d9d9d9'
            return gc

        def _refresh_swatches(*_):
            if color_mode.get() == 'gradient':
                gc = _grad_colors()
                for g, sw in grad_sw.items():
                    sw.config(bg=gc.get(g, '#cccccc'))
            else:
                for g, sw in indiv_sw.items():
                    sw.config(bg=picked.get(g) or _default_hex(g))

        def _toggle_mode(*_):
            if color_mode.get() == 'gradient':
                indiv.pack_forget(); grad.pack(fill='x')
            else:
                grad.pack_forget(); indiv.pack(fill='x')
            _refresh_swatches()
        color_mode.trace_add('write', _toggle_mode)
        grad_cmap.trace_add('write', _refresh_swatches)
        grad_dir.trace_add('write', _refresh_swatches)
        veh_var.trace_add('write', _refresh_swatches)

        # ── Buttons ───────────────────────────────────────────────────────
        btns = ttk.Frame(dlg, padding=8); btns.pack(fill='x')

        def _apply():
            self._group_order = list(order)
            self._group_palette_mode_var.set(color_mode.get())
            if color_mode.get() == 'gradient':
                self._group_colors = _grad_colors()
            else:
                self._group_colors = {g: c for g, c in picked.items() if c}
            dlg.destroy(); self._refresh_plot()

        def _reset():
            self._group_order = None
            self._group_colors = {}
            self._group_palette_mode_var.set('categorical')
            dlg.destroy(); self._refresh_plot()

        ttk.Button(btns, text="Apply", command=_apply).pack(side='right', padx=3)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right', padx=3)
        ttk.Button(btns, text="Reset to default", command=_reset).pack(side='left', padx=3)

        _toggle_mode()  # show the right frame + initial preview

    # ------------------------------------------------------------------
    # Per-view axis limits
    # ------------------------------------------------------------------

    def _open_axis_dialog(self):
        view = self._view_var.get()
        cur = self._axis_limits.get(view, {})
        dlg = tk.Toplevel(self._graph_win or self)
        dlg.title(f"Axis limits - {view}")
        dlg.transient(self._graph_win or self)
        dlg.grab_set()
        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill='both', expand=True)
        ttk.Label(frm, text="Leave blank for auto.").grid(
            row=0, column=0, columnspan=4, sticky='w', pady=(0, 6))
        entries = {}
        for i, (key, lab) in enumerate([('xmin', 'X min'), ('xmax', 'X max'),
                                        ('ymin', 'Y min'), ('ymax', 'Y max')]):
            r, c = divmod(i, 2)
            ttk.Label(frm, text=lab + ":").grid(row=r + 1, column=c * 2, sticky='e', padx=4, pady=3)
            e = ttk.Entry(frm, width=10)
            if cur.get(key) is not None:
                e.insert(0, str(cur[key]))
            e.grid(row=r + 1, column=c * 2 + 1, sticky='w', padx=4, pady=3)
            entries[key] = e

        def _apply():
            lim = {}
            for key, e in entries.items():
                txt = e.get().strip()
                if txt:
                    try:
                        lim[key] = float(txt)
                    except ValueError:
                        pass
            if lim:
                self._axis_limits[view] = lim
            else:
                self._axis_limits.pop(view, None)
            dlg.destroy()
            self._refresh_plot()

        btns = ttk.Frame(dlg, padding=(10, 0, 10, 10))
        btns.pack(fill='x')
        ttk.Button(btns, text="Apply", command=_apply).pack(side='right', padx=3)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right', padx=3)

    def _apply_axis_limits(self):
        """Apply the current view's saved x/y limits to cartesian axes only."""
        view = self._view_var.get()
        lim = self._axis_limits.get(view)
        if not lim or not self._fig:
            return
        # Skip image/network-heavy views where x/y limits are meaningless.
        if view in ('Transition Matrix', 'Transition Graph',
                    'Group Transition Graphs', 'Ethogram', 'Group Matrices',
                    'Transition Difference', 'State Composition',
                    'PCA', 'Meta-cluster Summary'):
            return
        for ax in self._fig.axes:
            if getattr(ax, 'images', None):
                continue
            if lim.get('xmin') is not None or lim.get('xmax') is not None:
                x0, x1 = ax.get_xlim()
                ax.set_xlim(lim.get('xmin', x0), lim.get('xmax', x1))
            if lim.get('ymin') is not None or lim.get('ymax') is not None:
                y0, y1 = ax.get_ylim()
                ax.set_ylim(lim.get('ymin', y0), lim.get('ymax', y1))

    def _refresh_plot(self):
        if not MATPLOTLIB_AVAILABLE or self._fig is None:
            return
        self._fig.clear()
        view = _VIEW_ALIASES.get(self._view_var.get(), self._view_var.get())
        _dispatch = {
            'State Usage':              self._plot_state_usage,
            'State Composition':        self._plot_state_composition,
            'Transition Matrix':        self._plot_heatmap,
            'Transition Graph':         self._plot_network,
            'Group Transition Graphs':  self._plot_group_networks,
            'Sequencing Networks':      self._plot_syntax_networks,
            'Sequencing Difference':    self._plot_syntax_difference,
            'Sequencing Ordination':    self._plot_syntax_ordination,
            'Group Matrices':           self._plot_group_matrices,
            'Transition Difference':    self._plot_transition_difference,
            'Ethogram':                 self._plot_ethogram,
            'Composition Over Time':    self._plot_temporal_probability,
            'Occupancy Over Time':      self._plot_behavior_over_time,
            'Behavior × Group':         self._plot_behavior_by_group,
            'Transition Timeline':      self._plot_timeline,
            # retired unsupervised views (not in the menu, kept for old configs)
            'Latent States':            self._plot_latent_states,
            'State Occupancy':          self._plot_state_occupancy,
            'PCA':                      self._plot_pca,
            'Meta-cluster Summary':     self._plot_meta_summary,
        }
        # Apply the display time window: recompute per-session/group transition
        # matrices (and crop sliding-window results) from the windowed sequences,
        # so EVERY matrix/graph view honors the window without per-plotter edits.
        # Occupancy-based views window themselves via _state_usage_occ. Restored
        # afterward so saved/exported state stays whole-session.
        _saved = (self._matrices, self._group_matrices,
                  self._group_subject_matrices, self._windowed)
        _win_active = False
        try:
            i0, i1 = self._win_bounds(10 ** 12)  # only to detect if a window is set
        except Exception:
            i0, i1 = 0, 10 ** 12
        windowed_display = (i0 > 0) or (i1 < 10 ** 12)
        try:
            if windowed_display and getattr(self, '_state_seqs', None):
                gm, gsub = self._group_matrices_win()
                if gm:
                    self._matrices = self._matrices_win()
                    self._group_matrices = gm
                    self._group_subject_matrices = gsub
                    _win_active = True
                # crop sliding-window results to [tmin, tmax] seconds
                def _f(v):
                    try:
                        s = v.get().strip(); return float(s) if s else None
                    except Exception:
                        return None
                tmn, tmx = _f(self._trend_tmin_var), _f(self._trend_tmax_var)
                if self._windowed:
                    lo = (tmn * 60.0) if tmn else -np.inf
                    hi = (tmx * 60.0) if tmx else np.inf
                    self._windowed = {
                        nm: [(t, m) for (t, m) in wr if lo <= t <= hi]
                        for nm, wr in _saved[3].items()}
            fn = _dispatch.get(view)
            if fn is not None:
                fn()
            else:
                ax = self._fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Unknown view: {view}", ha='center',
                        va='center', transform=ax.transAxes)
        except Exception as e:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center',
                    transform=ax.transAxes)
        finally:
            (self._matrices, self._group_matrices,
             self._group_subject_matrices, self._windowed) = _saved
        try:
            self._apply_axis_limits()
        except Exception:
            pass
        # Keep the figure the same size as its canvas so the render fills it
        # completely (no stale pixels from a prior, larger-figure view).
        self._sync_active_fig_size()
        self._canvas.draw()
        # Keep the main-page stats table in sync with the current settings.
        try:
            self._refresh_stats_table()
        except Exception:
            pass

    def _get_cmap(self):
        """State colormap - delegates to the single-source _state_palette()."""
        import matplotlib.colors as mcolors
        return mcolors.ListedColormap(self._state_palette())

    def _plot_ethogram(self):
        """Color-coded timeline bars per subject, grouped by treatment."""
        if not self._state_seqs:
            return
        # State sequences are sampled at the analysis fps (after any downsample),
        # not the raw video fps - use it so the time axis is correct.
        try:
            fps = float(self._effective_fps) or float(self._fps_var.get())
        except Exception:
            fps = float(self._fps_var.get() or 25.0)
        cmap = self._get_cmap()
        state_to_idx = {s: i for i, s in enumerate(self._states)}

        # Order sessions by group if key file loaded
        ordered = []
        if self._key_df is not None and self._group_matrices:
            for grp in self._ordered_groups(self._group_matrices.keys()):
                for name in sorted(self._state_seqs.keys()):
                    subj = self._session_subjects.get(name, name)
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty and str(row.iloc[0]['Treatment']) == grp:
                        ordered.append((grp, name))
        if not ordered:
            ordered = [('', name) for name in sorted(self._state_seqs.keys())]

        ax = self._fig.add_subplot(111)
        n_sessions = len(ordered)
        yticks, ylabels = [], []

        # Restrict to the display time window.
        _wseqs = {n: self._win_seq(self._state_seqs[n]) for _, n in ordered}
        max_frames = max((len(s) for s in _wseqs.values()), default=1)

        # The ethogram is a raster. At the default bin (0 = Full) it renders at
        # native fidelity - one column per frame, capped at 3000 columns for
        # display/speed. A non-zero ethogram bin coarsens it to that many seconds
        # per column. Either way each column is the majority (mode) state over its
        # span, so it is robust to single-frame flicker without discarding structure.
        try:
            etho_bin = float(_g(self._etho_bin_var, 0)) or 0.0
        except Exception:
            etho_bin = 0.0
        if etho_bin > 0:
            frames_per_col = max(1, int(round(etho_bin * fps)))
            n_cols = max(1, int(np.ceil(max_frames / frames_per_col)))
        else:
            n_cols = max_frames
        n_cols = max(1, min(n_cols, 3000))

        # Build RGBA image: white background = no data / excluded
        img = np.ones((n_sessions, n_cols, 4), dtype=float)

        for yi, (grp, name) in enumerate(ordered):
            seq = np.asarray(_wseqs[name])
            n_frames = len(seq)
            if n_frames:
                edges = np.linspace(0, n_frames, n_cols + 1).astype(int)
                for xi in range(n_cols):
                    a, b = edges[xi], edges[xi + 1]
                    seg = seq[a:b] if b > a else seq[min(a, n_frames - 1):
                                                       min(a, n_frames - 1) + 1]
                    if len(seg) == 0:
                        continue
                    vals, counts = np.unique(seg, return_counts=True)
                    idx = state_to_idx.get(int(vals[np.argmax(counts)]), -1)
                    if idx >= 0:
                        img[yi, xi] = cmap(idx)
            subj = self._session_subjects.get(name, name)
            yticks.append(yi)
            ylabels.append(f"{subj} ({grp})" if grp else subj)

        # X-axis in real recording time unless the user re-bases to zero. The
        # window start (minutes → seconds) is the offset of the cropped data.
        def _f(v):
            try:
                s = v.get().strip(); return float(s) if s else None
            except Exception:
                return None
        tmn = _f(self._trend_tmin_var) if hasattr(self, '_trend_tmin_var') else None
        realtime = _g(self._realtime_axis_var, True)
        x_start = (tmn * 60.0) if (realtime and tmn) else 0.0
        t_span = max_frames / fps
        ax.imshow(img, aspect='auto',
                  extent=[x_start, x_start + t_span, n_sessions - 0.5, -0.5],
                  interpolation='nearest')
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_title("Ethogram")

        # Legend - placed outside the image so it never covers data.
        patches = [mpatches.Patch(color=cmap(i), label=self._state_name(s))
                   for i, s in enumerate(self._states)]
        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, fontsize=7)

    def _plot_heatmap(self):
        """Transition probability heatmap (mean across all sessions)."""
        if not self._matrices:
            return
        # Average all session matrices
        all_mats = list(self._matrices.values())
        mean_mat = self._norm_matrix(np.mean(np.stack(all_mats), axis=0))

        n_states = len(self._states)
        labels = [self._state_name(s) for s in self._states]
        if n_states > 15:
            labels = [str(s) for s in self._states]
        show_annot = n_states <= 20 and self._show_annot_var.get()
        annot_size = max(5, 10 - n_states // 3) if show_annot else 7
        tick_size = max(5, 9 - n_states // 5)

        _shrink = min(0.85, max(0.35, n_states / (n_states + 6)))
        _heatmap_cmap = self._heat_cmap()
        ax = self._fig.add_subplot(111)
        sns.heatmap(mean_mat, annot=show_annot,
                    fmt='.1f' if n_states > 10 else '.2f',
                    annot_kws={'size': annot_size} if show_annot else {},
                    cmap=_heatmap_cmap,
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, vmin=0, vmax=1, square=n_states <= 20,
                    cbar_kws={'label': 'P(transition)', 'shrink': _shrink})
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        ax.set_title("Transition Probability Matrix (all subjects)")
        ax.tick_params(labelsize=tick_size)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    def _plot_network(self):
        """Directed graph: node size ~ time-in-state, edge width ~ P(transition)."""
        if not NETWORKX_AVAILABLE:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "networkx not installed.\npip install networkx",
                    ha='center', va='center', transform=ax.transAxes)
            return
        if not self._matrices:
            return

        all_mats = list(self._matrices.values())
        mean_mat = np.mean(np.stack(all_mats), axis=0)

        # Time-in-state fractions
        all_seqs = np.concatenate(list(self._state_seqs.values()))
        state_fracs = {}
        for i, s in enumerate(self._states):
            state_fracs[s] = (all_seqs == s).mean()

        G = nx.DiGraph()
        cmap = self._get_cmap()
        for i, si in enumerate(self._states):
            G.add_node(self._state_name(si), size=state_fracs.get(si, 0.01),
                       color=cmap(i))
        threshold = 0.02
        for i, si in enumerate(self._states):
            for j, sj in enumerate(self._states):
                if i != j and mean_mat[i, j] > threshold:
                    G.add_edge(self._state_name(si), self._state_name(sj),
                               weight=mean_mat[i, j])

        ax = self._fig.add_subplot(111)
        pos = nx.spring_layout(G, seed=42, k=2.0)
        node_sizes = [G.nodes[n]['size'] * 3000 + 200 for n in G.nodes]
        node_colors = [G.nodes[n]['color'] for n in G.nodes]
        edge_widths = [G.edges[e]['weight'] * 5 for e in G.edges]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colors, alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               alpha=0.6, edge_color='gray',
                               arrows=True, arrowsize=15,
                               connectionstyle='arc3,rad=0.15')
        # Edge labels
        edge_labels = {e: f"{G.edges[e]['weight']:.2f}" for e in G.edges
                       if G.edges[e]['weight'] > 0.05}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=6)
        ax.set_title("State Transition Network")
        ax.axis('off')

    # ------------------------------------------------------------------
    # Sequencing views - the manuscript's bout-level syntax analysis
    # (pipeline/syntax_analysis.py; quasi-independence residuals, rarefied
    # pooled networks, PCoA + PERMANOVA), run on the same priority-resolved
    # state sequences every other view uses.
    # ------------------------------------------------------------------

    _SYNTAX_GROUP_COLS = ["#8D99AE", "#CC79A7", "#3b528b", "#21918c",
                          "#f59f00", "#2f9e44"]

    def _syntax_cohort(self):
        """Build (cohort, ordered groups) from the current sequences, or (None, msg)."""
        from pipeline import syntax_analysis as SA
        seqs = getattr(self, '_state_seqs', None)
        if not seqs:
            return None, "Run a transitions compute first."
        # honor the display time window, as every matrix view does
        try:
            seqs = {name: self._win_seq(seq) for name, seq in seqs.items()}
        except Exception:
            pass
        group_of = {name: self._session_group(name) for name in seqs}
        if not any(group_of.values()):
            return None, ("Load a key file so sessions carry groups -\n"
                          "sequencing views compare groups.")
        named = [s for s in self._states if s != 0]      # unscored is closed over
        keep_idx = {sid: i for i, sid in enumerate(named)}
        labels = [self._state_name(s) for s in named]
        # paired design: within-animal permutation. The key file's Subject column defines
        # groups, so pairing needs its own column -- an 'Animal' (or 'Pair'/'Block') column
        # naming which rows are the same animal across conditions. Without one, fall back
        # to repeated subjects across sessions.
        subs = getattr(self, '_session_subjects', None) or {}
        strata_of = {n: subs.get(n, n) for n in seqs}
        kdf = getattr(self, '_key_df', None)
        if kdf is not None:
            pcol = next((c for c in ('Animal', 'Pair', 'Block') if c in kdf.columns), None)
            if pcol:
                m = {str(r['Subject']): str(r[pcol]) for _, r in kdf.iterrows()}
                strata_of = {n: m.get(str(subs.get(n, n)), subs.get(n, n)) for n in seqs}
        rep = len(set(strata_of.values())) < len([n for n in seqs if group_of.get(n)])
        cohort = SA.SyntaxCohort(seqs, group_of, keep_idx, labels,
                                 strata_of=strata_of if rep else None)
        if not cohort.sessions:
            return None, (f"No session carries ≥{SA.MIN_BOUTS} labelled transitions.\n"
                          "Sequencing needs bout-level data; check the bout filters.")
        groups = self._ordered_groups(cohort.groups)
        return (cohort, groups), None

    def _syntax_msg(self, msg):
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    def _plot_syntax_networks(self):
        """One pooled, rarefied transition network per group; colour = Pearson residual."""
        from pipeline import syntax_analysis as SA
        built, msg = self._syntax_cohort()
        if built is None:
            self._syntax_msg(msg)
            return
        cohort, groups = built
        n = len(groups)
        self._set_panel_figsize(n, 'h')
        axes = self._fig.subplots(1, n, squeeze=False)[0]
        for gi, g in enumerate(groups):
            col = self._SYNTAX_GROUP_COLS[gi % len(self._SYNTAX_GROUP_COLS)]
            SA.draw_network(axes[gi], cohort.pooled[g], cohort.labels,
                            f"{g}\n(rarefied to {int(cohort.pooled[g].sum())} transitions)",
                            col, SA.mono_cmap(col))
        if cohort.dropped:
            self._fig.text(0.01, 0.01,
                           f"excluded (<{SA.MIN_BOUTS} transitions): "
                           + ", ".join(cohort.dropped), fontsize=7, color='#888888')
        self._log_msg(f"Sequencing networks: {n} group(s), "
                      f"{len(cohort.sessions)} session(s), per-animal tables "
                      f"subsampled to {cohort.n_sub} transitions.")

    def _plot_syntax_difference(self):
        """What changed vs the reference group: every assessable change > MIN_DZ residual units."""
        from pipeline import syntax_analysis as SA
        built, msg = self._syntax_cohort()
        if built is None:
            self._syntax_msg(msg)
            return
        cohort, groups = built
        if len(groups) < 2:
            self._syntax_msg("Need at least two groups for a difference network.")
            return
        ref, others = groups[0], groups[1:]
        cmap = SA.div_cmap()
        self._set_panel_figsize(len(others), 'h')
        axes = self._fig.subplots(1, len(others), squeeze=False)[0]
        for gi, g in enumerate(others):
            vmax, n_ch, n_el = SA.draw_diff(axes[gi], cohort.pooled[ref],
                                            cohort.pooled[g], cohort.labels, cmap)
            axes[gi].set_title(f"{g} vs {ref}\n{n_ch} of {n_el} routes changed "
                               f"> {SA.MIN_DZ:.0f} residual units", fontsize=9, pad=4)
            self._log_msg(f"Sequencing difference {g} vs {ref}: {n_ch}/{n_el} routes "
                          f"past {SA.MIN_DZ:.0f} SD (solid strengthened, dashed weakened).")

    def _plot_syntax_ordination(self):
        """PCoA of per-animal residual tables, with PERMANOVA (Behrens-Fisher, exact p)."""
        from pipeline import syntax_analysis as SA
        import numpy as _np
        built, msg = self._syntax_cohort()
        if built is None:
            self._syntax_msg(msg)
            return
        cohort, groups = built
        xy, pcvar, _load = cohort.ordination()
        xyd = SA.dodge(xy)
        ax = self._fig.add_subplot(111)
        marks = ['o', 's', '^', 'D', 'v', 'P']
        for gi, g in enumerate(groups):
            m = cohort.groups_per_session == g
            col = self._SYNTAX_GROUP_COLS[gi % len(self._SYNTAX_GROUP_COLS)]
            mk = marks[gi % len(marks)]
            ax.scatter(xyd[m, 0], xyd[m, 1], s=46, marker=mk, facecolors='none',
                       edgecolors=col, linewidth=1.3, alpha=0.95, zorder=3, label=g)
            if m.sum() > 1:
                ax.scatter(*xy[m].mean(0), s=150, marker='+', color=col,
                           linewidth=1.8, zorder=4)
        try:
            F2, p, R2 = cohort.test()
            paired = cohort.strata is not None
            ax.set_title(f"p = {p:.4g}, R² = {R2:.2f}  (PERMANOVA"
                         + (", within-animal permutation)" if paired else ")"),
                         fontsize=10, pad=6)
        except Exception as ex:
            ax.set_title(f"PERMANOVA unavailable: {ex}", fontsize=9)
        ax.set_xlabel(f"PC1 ({pcvar[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({pcvar[1]:.0f}%)")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#DDDDDD')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(frameon=False, fontsize=8, loc='best')
        self._log_msg(f"Sequencing ordination: {len(cohort.sessions)} animals, "
                      f"PC1+PC2 carry {pcvar[0] + pcvar[1]:.0f}% of variance.")

    def _plot_group_networks(self):
        """One transition-network panel per group, side-by-side, shared layout."""
        if not NETWORKX_AVAILABLE:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "networkx not installed.\npip install networkx",
                    ha='center', va='center', transform=ax.transAxes)
            return
        if not self._group_matrices:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Load a key file and compute\nto see group networks.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        groups = self._ordered_groups(self._group_matrices.keys())
        n_groups = len(groups)
        threshold = 0.02

        # Map sessions to groups (mirrors the assembly at lines 2419-2428 used
        # to build self._group_matrices).
        group_sessions = {grp: [] for grp in groups}
        if self._key_df is not None:
            for name in self._matrices:
                subj = self._session_subjects.get(name, name)
                row = self._key_df[self._key_df['Subject'] == subj]
                if not row.empty:
                    grp = str(row.iloc[0]['Treatment'])
                    if grp in group_sessions:
                        group_sessions[grp].append(name)

        # Shared spring layout: union of every node and every above-threshold
        # edge across all groups, so node positions are identical across
        # panels and the eye can trace transitions group-to-group.
        union = nx.DiGraph()
        for s in self._states:
            union.add_node(self._state_name(s))
        for mat in self._group_matrices.values():
            for i, si in enumerate(self._states):
                for j, sj in enumerate(self._states):
                    if i != j and mat[i, j] > threshold:
                        union.add_edge(self._state_name(si),
                                       self._state_name(sj))
        pos = nx.spring_layout(union, seed=42, k=2.0)

        cmap = self._get_cmap()
        self._set_panel_figsize(n_groups, 'h')
        axes = self._fig.subplots(1, n_groups, squeeze=False)[0]

        for gi, grp in enumerate(groups):
            ax = axes[gi]
            mean_mat = self._group_matrices[grp]

            # Per-group state-time fractions. Prefer the uncapped sequences so
            # the analysis-time duration cap doesn't skew node sizes.
            sess_for_grp = group_sessions.get(grp, [])
            seqs = []
            for n in sess_for_grp:
                seq = self._state_seqs_full.get(n) if hasattr(
                    self, '_state_seqs_full') else None
                if seq is None:
                    seq = self._state_seqs.get(n)
                if seq is not None:
                    seqs.append(np.asarray(seq))
            all_seqs = np.concatenate(seqs) if seqs else np.array([])
            state_fracs = {
                s: (float((all_seqs == s).mean()) if all_seqs.size else 0.0)
                for s in self._states}

            G = nx.DiGraph()
            for i, si in enumerate(self._states):
                G.add_node(self._state_name(si),
                           size=state_fracs.get(si, 0.0),
                           color=cmap(i))
            for i, si in enumerate(self._states):
                for j, sj in enumerate(self._states):
                    if i != j and mean_mat[i, j] > threshold:
                        G.add_edge(self._state_name(si),
                                   self._state_name(sj),
                                   weight=mean_mat[i, j])

            node_sizes = [G.nodes[n]['size'] * 3000 + 200 for n in G.nodes]
            node_colors = [G.nodes[n]['color'] for n in G.nodes]
            edge_widths = [G.edges[e]['weight'] * 5 for e in G.edges]

            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                   node_color=node_colors, alpha=0.85)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                                   alpha=0.6, edge_color='gray',
                                   arrows=True, arrowsize=15,
                                   connectionstyle='arc3,rad=0.15')
            edge_labels = {e: f"{G.edges[e]['weight']:.2f}" for e in G.edges
                           if G.edges[e]['weight'] > 0.05}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                         font_size=6)
            ax.set_title(f"{grp}  (n={len(sess_for_grp)})", fontsize=10)
            ax.axis('off')

    # ------------------------------------------------------------------
    # Statistics - nonparametric per-state/per-cell tests with FDR
    # ------------------------------------------------------------------

    @staticmethod
    def _bh_fdr(pvals):
        """Benjamini-Hochberg FDR q-values for a 1-D array (NaN-safe)."""
        p = np.asarray(pvals, dtype=float)
        q = np.full(p.shape, np.nan)
        idx = np.where(~np.isnan(p))[0]
        if len(idx) == 0:
            return q
        pv = p[idx]
        order = np.argsort(pv)
        ranked = pv[order]
        m = len(pv)
        qv = ranked * m / (np.arange(1, m + 1))
        # enforce monotonicity from the largest p downward
        qv = np.minimum.accumulate(qv[::-1])[::-1]
        qv = np.clip(qv, 0, 1)
        out = np.empty(m)
        out[order] = qv
        q[idx] = out
        return q

    @staticmethod
    def _stars(p, alpha=0.05):
        """Significance stars for a (corrected) p/q value, gated at `alpha`."""
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ''
        if p >= alpha:
            return ''
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        return '*'

    def _alpha(self):
        try:
            return float(self._stat_alpha_var.get())
        except Exception:
            return 0.05

    def _correct_pvals(self, raw):
        """Apply the selected multiple-comparison correction across p-values."""
        raw = np.asarray(raw, dtype=float)
        mode = self._stat_correction_var.get()
        if mode == 'None':
            return raw.copy()
        if mode == 'Bonferroni':
            m = int(np.sum(~np.isnan(raw)))
            return np.clip(raw * max(m, 1), 0, 1)
        return self._bh_fdr(raw)   # BH-FDR (default)

    @staticmethod
    def _two_sample_p(a, b, test='mw'):
        """Two-sided p-value comparing two samples ('mw' = Mann-Whitney,
        'welch' = Welch's t). NaN when either sample has <2 points."""
        from scipy.stats import mannwhitneyu, ttest_ind
        a = np.asarray(a, float); b = np.asarray(b, float)
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            return np.nan
        try:
            if test == 'welch':
                return ttest_ind(a, b, equal_var=False).pvalue
            return mannwhitneyu(a, b, alternative='two-sided').pvalue
        except Exception:
            return np.nan

    def _group_err(self, values):
        """Error-bar half-width for a 1-D sample per `_error_mode_var`:
        SEM of the mean, or a 95% bootstrap CI half-width."""
        v = np.asarray(values, float)
        v = v[~np.isnan(v)]
        n = len(v)
        if n < 2:
            return 0.0
        if self._error_mode_var.get().startswith('95'):
            # percentile bootstrap CI of the mean (seeded → reproducible)
            rs = np.random.RandomState(12345)
            boot = np.array([rs.choice(v, n, replace=True).mean()
                             for _ in range(1000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            return (hi - lo) / 2.0
        return v.std(ddof=1) / np.sqrt(n)

    def _reference_group(self, groups):
        """Reference/vehicle group: explicit selection, else the dose-0 (vehicle)
        bucket, else the first ordered group."""
        groups = list(groups)
        ref = self._ref_group_var.get() if hasattr(self, '_ref_group_var') else ''
        if ref and ref in groups:
            return ref
        for g in self._ordered_groups(groups):
            if self._treatment_sort_key(g)[0] == 0:
                return g
        og = self._ordered_groups(groups)
        return og[0] if og else None

    def _effective_test(self, n_groups):
        """Resolve the Test control to a concrete per-state test given the design.
        Returns 'kruskal' | 'anova' | 'mw' | 'welch'."""
        t = self._stat_test_var.get()
        if t.startswith('Kruskal'):
            return 'kruskal'
        if t.startswith('Mann'):
            return 'mw'
        if t.startswith('Parametric') or t.startswith('Welch') or t.startswith('t-'):
            return 'anova' if n_groups >= 3 else 'welch'
        # Auto: nonparametric - omnibus KW for ≥3 groups, else MW vs reference.
        return 'kruskal' if n_groups >= 3 else 'mw'

    def _perstate_stats(self, values_by_group, reference=None):
        """Per-state group test honoring the Statistics controls (test +
        correction + alpha). `values_by_group`: {group: array(n_states,n_animals)}.
        Returns per state {p, q, star, test}."""
        from scipy.stats import kruskal, f_oneway
        groups = self._ordered_groups(values_by_group.keys())
        if not groups:
            return []
        n_states = values_by_group[groups[0]].shape[0]
        ref = reference or self._reference_group(groups)
        et = self._effective_test(len(groups))
        raw = np.full(n_states, np.nan)
        for si in range(n_states):
            samples = [np.asarray(values_by_group[g][si], float) for g in groups]
            samples = [s[~np.isnan(s)] for s in samples]
            valid = [s for s in samples if len(s) >= 2]
            if len(valid) < 2:
                continue
            try:
                if et == 'kruskal':
                    raw[si] = kruskal(*valid).pvalue
                elif et == 'anova':
                    raw[si] = f_oneway(*valid).pvalue
                else:  # pairwise (mw/welch) vs reference - smallest p
                    rmat = np.asarray(values_by_group.get(ref, np.empty((n_states, 0))), float)
                    rvec = rmat[si] if rmat.ndim == 2 and rmat.shape[1] else np.array([])
                    ps = [self._two_sample_p(values_by_group[g][si], rvec, et)
                          for g in groups if g != ref]
                    ps = [p for p in ps if not np.isnan(p)]
                    if ps:
                        raw[si] = min(ps)
            except Exception:
                pass
        q = self._correct_pvals(raw)
        a = self._alpha()
        return [{'p': raw[i], 'q': q[i], 'star': self._stars(q[i], a), 'test': et}
                for i in range(n_states)]

    def _percell_stats(self, mats_by_group, reference, target):
        """q-matrix comparing `target` vs `reference` per transition cell,
        honoring the Test + Correction controls (2-sample per cell)."""
        a = mats_by_group.get(target, [])
        b = mats_by_group.get(reference, [])
        n = a[0].shape[0] if a else (b[0].shape[0] if b else 0)
        if n == 0 or len(a) < 2 or len(b) < 2:
            return np.ones((n, n)) if n else np.ones((1, 1))
        use = 'welch' if self._effective_test(2) == 'welch' else 'mw'
        raw = np.full(n * n, np.nan)
        for k in range(n * n):
            i, j = divmod(k, n)
            raw[k] = self._two_sample_p([m[i, j] for m in a],
                                        [m[i, j] for m in b], use)
        return self._correct_pvals(raw).reshape(n, n)

    def _compute_sig_matrix(self, mats_a, mats_b):
        """Return Bonferroni-corrected p-value matrix (Mann-Whitney U, two-sided)."""
        from scipy.stats import mannwhitneyu
        n = mats_a[0].shape[0]
        pvals = np.ones((n, n))
        if len(mats_a) < 2 or len(mats_b) < 2:
            return pvals  # not enough subjects
        for i in range(n):
            for j in range(n):
                a = [m[i, j] for m in mats_a]
                b = [m[i, j] for m in mats_b]
                try:
                    _, p = mannwhitneyu(a, b, alternative='two-sided')
                    pvals[i, j] = p
                except Exception:
                    pass
        pvals = np.minimum(pvals * n * n, 1.0)  # Bonferroni
        return pvals

    def _plot_group_matrices(self):
        """Side-by-side per-group transition-matrix heatmaps (N groups, dose order)."""
        if not self._group_matrices:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Load a key file and compute\nto see group matrices.",
                    ha='center', va='center', transform=ax.transAxes)
            return
        groups = self._ordered_groups(self._group_matrices.keys())
        n_groups = len(groups)
        n_states = len(self._states)
        labels = [self._state_name(s) for s in self._states]
        if n_states > 15:
            labels = [str(s) for s in self._states]
        show_annot = n_states <= 15 and self._show_annot_var.get()
        annot_size = max(5, 9 - n_states // 3) if show_annot else 7
        tick_size = max(4, 8 - n_states // 5)
        use_square = n_states <= 15
        _shrink = min(0.85, max(0.35, n_states / (n_states + 6)))
        _heatmap_cmap = self._heat_cmap()
        self._set_panel_figsize(n_groups, 'h')
        axes = self._fig.subplots(1, n_groups, squeeze=False)[0]
        for gi, grp in enumerate(groups):
            mat = self._norm_matrix(self._group_matrices[grp])
            show_cbar = (gi == n_groups - 1)
            sns.heatmap(mat, annot=show_annot,
                        fmt='.1f' if n_states > 10 else '.2f',
                        annot_kws={'size': annot_size} if show_annot else {},
                        cmap=_heatmap_cmap,
                        xticklabels=labels, yticklabels=labels,
                        ax=axes[gi], vmin=0, vmax=1, square=use_square,
                        cbar=show_cbar,
                        cbar_kws={'shrink': _shrink} if show_cbar else {})
            axes[gi].set_title(grp, fontsize=10)
            axes[gi].set_xlabel("To")
            axes[gi].set_ylabel("From" if gi == 0 else "")
            axes[gi].tick_params(labelsize=tick_size)
            axes[gi].set_xticklabels(axes[gi].get_xticklabels(),
                                     rotation=45, ha='right')
            axes[gi].set_yticklabels(
                axes[gi].get_yticklabels() if gi == 0 else [], rotation=0)

    def _plot_transition_difference(self):
        """Diverging heatmap of (group − reference) transition probabilities,
        one panel per non-reference group (dose order), with FDR sig. stars."""
        if not self._group_matrices:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Load a key file and compute\nto see differences.",
                    ha='center', va='center', transform=ax.transAxes)
            return
        groups = self._ordered_groups(self._group_matrices.keys())
        ref = self._reference_group(groups)
        nonref = [g for g in groups if g != ref]
        if not ref or not nonref:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Need ≥2 groups (a reference + one other).",
                    ha='center', va='center', transform=ax.transAxes)
            return
        n_states = len(self._states)
        labels = [self._state_name(s) for s in self._states]
        if n_states > 15:
            labels = [str(s) for s in self._states]
        show_annot = n_states <= 15 and self._show_annot_var.get()
        annot_size = max(5, 9 - n_states // 3) if show_annot else 7
        tick_size = max(4, 8 - n_states // 5)
        use_square = n_states <= 15
        ref_mat = np.asarray(self._group_matrices[ref])
        diffs = {g: np.asarray(self._group_matrices[g]) - ref_mat for g in nonref}
        vmax = max((float(np.abs(d).max()) for d in diffs.values()), default=0.01)
        vmax = max(vmax, 0.01)
        self._set_panel_figsize(len(nonref), 'h')
        axes = self._fig.subplots(1, len(nonref), squeeze=False)[0]
        _fmt_signed = '{:+.1f}' if n_states > 10 else '{:+.2f}'
        for gi, grp in enumerate(nonref):
            d = diffs[grp]
            text = np.vectorize(lambda v: _fmt_signed.format(v))(d) if show_annot else False
            show_cbar = (gi == len(nonref) - 1)
            sns.heatmap(d, annot=text, fmt='' if show_annot else '',
                        annot_kws={'size': annot_size} if show_annot else {},
                        cmap=self._diverging_cmap(), center=0,
                        vmin=-vmax, vmax=vmax, xticklabels=labels,
                        yticklabels=labels if gi == 0 else [],
                        ax=axes[gi], square=use_square, cbar=show_cbar,
                        cbar_kws={'shrink': 0.7, 'label': f'Δ vs {ref}'} if show_cbar else {})
            axes[gi].set_title(f"{grp} − {ref}", fontsize=10)
            axes[gi].set_xlabel("To")
            axes[gi].set_ylabel("From" if gi == 0 else "")
            axes[gi].tick_params(labelsize=tick_size)
            axes[gi].set_xticklabels(axes[gi].get_xticklabels(), rotation=45, ha='right')
            # FDR-corrected significance stars vs reference.
            if self._show_sig_var.get() and ref in self._group_subject_matrices \
                    and grp in self._group_subject_matrices:
                q = self._percell_stats(self._group_subject_matrices, ref, grp)
                for i in range(n_states):
                    for j in range(n_states):
                        star = self._stars(q[i, j])
                        if star:
                            axes[gi].text(j + 0.5, i + 0.72, star, ha='center',
                                          va='center', fontsize=max(6, annot_size - 1),
                                          color='black', fontweight='bold')

    def _plot_timeline(self):
        """Line plot of P(state_i -> state_j) over time for sliding windows."""
        if not self._windowed:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Use 'Sliding windows' mode\nto see timeline.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Parse selected pair
        pair_text = self._pair_combo.get()
        if ' -> ' not in pair_text:
            return
        from_name, to_name = pair_text.split(' -> ', 1)
        # Find indices
        name_to_idx = {}
        for i, s in enumerate(self._states):
            name_to_idx[self._state_name(s)] = i
        if from_name not in name_to_idx or to_name not in name_to_idx:
            return
        fi, ti = name_to_idx[from_name], name_to_idx[to_name]

        ax = self._fig.add_subplot(111)

        # Group by treatment if key file loaded
        if self._key_df is not None and self._group_matrices:
            groups = {}  # {treatment: [(times, probs), ...]}
            for name, wresults in self._windowed.items():
                subj = self._session_subjects.get(name, name)
                row = self._key_df[self._key_df['Subject'] == subj]
                if not row.empty:
                    grp = str(row.iloc[0]['Treatment'])
                else:
                    grp = 'Unknown'
                if grp not in groups:
                    groups[grp] = []
                times = [t for t, _ in wresults]
                probs = [m[fi, ti] for _, m in wresults]
                groups[grp].append((times, probs))

            gpal = self._group_palette(groups.keys())
            for grp in self._ordered_groups(groups.keys()):
                traces = groups[grp]
                gcol = gpal.get(grp, 'gray')
                # Align to common time grid
                all_times = sorted(set(t for ts, _ in traces for t in ts))
                aligned = np.full((len(traces), len(all_times)), np.nan)
                for ri, (ts, ps) in enumerate(traces):
                    for t, p in zip(ts, ps):
                        idx = all_times.index(t)
                        aligned[ri, idx] = p
                mean = np.nanmean(aligned, axis=0)
                sem = (np.nanstd(aligned, axis=0, ddof=1) /
                       np.sqrt(np.sum(~np.isnan(aligned), axis=0)))
                sem = np.nan_to_num(sem)
                ax.plot(all_times, mean, label=grp, color=gcol, linewidth=2)
                ax.fill_between(all_times, mean - sem, mean + sem,
                                alpha=0.2, color=gcol)
        else:
            # Individual traces
            for name, wresults in self._windowed.items():
                times = [t for t, _ in wresults]
                probs = [m[fi, ti] for _, m in wresults]
                ax.plot(times, probs, label=self._session_subjects.get(name, name),
                        alpha=0.7)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("P(transition)")
        ax.set_title(f"Transition: {pair_text}")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    def _plot_behavior_over_time(self):
        """Mean ± SEM behavior occupancy over time, one subplot per group.
        Requires sliding-windows mode so _windowed is populated."""
        if not self._windowed:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Use 'Sliding windows' mode and re-compute\n"
                    "to see the Behavior Over Time plot.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        fps = self._effective_fps
        win_s  = self._win_sec_var.get()
        step_s = self._step_sec_var.get()
        win_frames  = max(1, int(round(win_s  * fps)))
        step_frames = max(1, int(round(step_s * fps)))

        # Compute per-session occupancy time-courses
        session_occ = {}
        for name, seq in self._state_seqs.items():
            s = np.asarray(seq, dtype=int)
            occ_series = []
            for start in range(0, len(s) - win_frames + 1, step_frames):
                chunk = s[start:start + win_frames]
                center_min = (start + win_frames / 2) / fps / 60.0
                fracs = {sid: float(np.sum(chunk == sid)) / len(chunk)
                         for sid in self._states}
                occ_series.append((center_min, fracs))
            session_occ[name] = occ_series

        # Determine groups
        if self._key_df is not None and self._group_matrices:
            groups = self._ordered_groups(self._group_matrices.keys())
            def _get_group(name):
                subj = self._session_subjects.get(name, name)
                row  = self._key_df[self._key_df['Subject'] == subj]
                return str(row.iloc[0]['Treatment']) if not row.empty else 'Unknown'
        else:
            groups = ['All']
            def _get_group(name): return 'All'

        n_groups = len(groups)
        self._set_panel_figsize(n_groups, 'h')
        axes = self._fig.subplots(1, n_groups, squeeze=False, sharey=True)[0]

        # States share the single canonical state palette.
        n_states = len(self._states)
        raw_colors = self._state_palette(n_states)

        for gi, grp in enumerate(groups):
            ax = axes[gi]
            grp_sessions = [n for n in session_occ if _get_group(n) == grp]
            if not grp_sessions:
                ax.set_title(grp)
                continue

            # Build common time grid
            all_times = sorted({t for name in grp_sessions
                                for t, _ in session_occ[name]})
            if not all_times:
                continue
            time_arr = np.array(all_times)

            for si, sid in enumerate(self._states):
                color = raw_colors[si % len(raw_colors)]
                sname = self._state_name(sid)

                # Align each session to the common time grid
                mat = np.full((len(grp_sessions), len(all_times)), np.nan)
                for ri, name in enumerate(grp_sessions):
                    t2f = {t: f.get(sid, 0.0) for t, f in session_occ[name]}
                    for ci_t, t in enumerate(all_times):
                        if t in t2f:
                            mat[ri, ci_t] = t2f[t]

                n_valid = np.sum(~np.isnan(mat), axis=0)
                mean    = np.nanmean(mat, axis=0)
                sem     = np.where(n_valid > 1,
                                   np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n_valid),
                                   0.0)

                ax.plot(time_arr, mean, color=color, linewidth=2, label=sname)
                ax.fill_between(time_arr, mean - sem, mean + sem,
                                alpha=0.2, color=color)

            ax.set_title(grp, fontsize=10)
            ax.set_xlabel("Time (min)")
            if gi == 0:
                ax.set_ylabel("Behaviour probability")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)

        # Shared legend - outside the last panel so it never covers the traces.
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[-1].legend(handles, labels, fontsize=8, loc='upper left',
                            bbox_to_anchor=(1.01, 1), borderaxespad=0)
        self._apply_window_xlim('min')

    def _temporal_probs_current(self):
        """Per-session (time_centers_s, prob[n_bins × n_states]) at the CURRENT bin
        width - re-binned live from _state_seqs so the graph-window Bin control
        updates instantly (no recompute). Falls back to the stored _temporal_probs."""
        seqs = getattr(self, '_state_seqs', None)
        if not seqs:
            return getattr(self, '_temporal_probs', {}) or {}
        try:
            bin_sec = float(_g(self._prob_bin_var, 30)) or 30.0
        except Exception:
            bin_sec = 30.0
        try:
            fps = float(self._effective_fps) or 25.0
        except Exception:
            fps = 25.0
        states = list(self._states)
        out = {}
        for name, s in seqs.items():
            try:
                out[name] = compute_temporal_probabilities(
                    np.asarray(s, dtype=int), fps, bin_sec, states)
            except Exception:
                pass
        return out or (getattr(self, '_temporal_probs', {}) or {})

    def _plot_behavior_by_group(self):
        """Per-behavior occupancy over time, GROUPS overlaid (group-colored mean ±
        SEM). Small-multiples grid (one panel per behavior) or a single behavior
        via the picker; a display time window (min) can restrict the x-range.
        Re-bins live at the current Bin width so it works in any time mode."""
        tp = self._temporal_probs_current()
        if not tp:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Run 'Compute from classifiers' first.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Groups (dose order) + session→group map.
        gmap = {n: self._session_group(n) for n in tp}
        gkeys = self._ordered_groups({g for g in gmap.values() if g})
        if not gkeys:
            gkeys = ['All']
            gmap = {n: 'All' for n in tp}
        gpal = self._group_palette(gkeys)
        group_sessions = {g: [n for n, gg in gmap.items() if gg == g] for g in gkeys}
        state_idx = {s: i for i, s in enumerate(self._states)}

        # Display time window (minutes); blank = auto/full.
        def _f(var):
            try:
                return float(var.get())
            except (ValueError, AttributeError):
                return None
        tmin = _f(self._trend_tmin_var)
        tmax = _f(self._trend_tmax_var)

        def _series(sid, sess):
            """Union-time-grid mean ± SEM (minutes) for state `sid` over `sess`."""
            col = state_idx.get(sid)
            if col is None or not sess:
                return None
            all_t = sorted({round(float(c) / 60.0, 6)
                            for n in sess for c in tp[n][0]})
            if not all_t:
                return None
            mat = np.full((len(sess), len(all_t)), np.nan)
            tpos = {t: k for k, t in enumerate(all_t)}
            for ri, n in enumerate(sess):
                centers, pm = tp[n]
                for k, c in enumerate(centers):
                    j = tpos.get(round(float(c) / 60.0, 6))
                    if j is not None and col < pm.shape[1]:
                        mat[ri, j] = pm[k, col]
            t = np.array(all_t)
            nv = np.sum(~np.isnan(mat), axis=0)
            mean = np.nanmean(mat, axis=0)
            sem = np.where(nv > 1, np.nanstd(mat, axis=0, ddof=1) / np.sqrt(nv), 0.0)
            keep = np.ones_like(t, dtype=bool)
            if tmin is not None:
                keep &= t >= tmin
            if tmax is not None:
                keep &= t <= tmax
            return t[keep], mean[keep], sem[keep]

        behaviors = [s for s in self._states if s != 0]   # named classifiers only
        sel = self._behav_var.get() if hasattr(self, '_behav_var') else 'All behaviors (grid)'

        def _draw(ax, sid):
            for g in gkeys:
                res = _series(sid, group_sessions[g])
                if res is None:
                    continue
                t, mean, sem = res
                col = gpal.get(g, 'gray')
                ax.plot(t, mean, color=col, linewidth=2, label=str(g))
                ax.fill_between(t, mean - sem, mean + sem, alpha=0.2, color=col)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)

        if sel and sel != 'All behaviors (grid)':
            # Single behavior - groups overlaid in one panel.
            sid = next((s for s in behaviors if self._state_name(s) == sel),
                       behaviors[0] if behaviors else 0)
            ax = self._fig.add_subplot(111)
            _draw(ax, sid)
            ax.set_title(self._state_name(sid), fontsize=11)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Fraction of time")
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1),
                      borderaxespad=0)
            return

        # Grid - one panel per behavior.
        if not behaviors:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No named behaviors to show.",
                    ha='center', va='center', transform=ax.transAxes)
            return
        n = len(behaviors)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        self._set_panel_figsize(nrows, 'v')
        axes = self._fig.subplots(nrows, ncols, squeeze=False,
                                  sharex=True, sharey=True)
        flat = axes.flat
        for ax, sid in zip(flat, behaviors):
            _draw(ax, sid)
            ax.set_title(self._state_name(sid), fontsize=9)
        for ax in list(flat)[n:]:
            ax.axis('off')
        # Shared axis labels + one group legend outside.
        self._fig.supxlabel("Time (min)", fontsize=9)
        self._fig.supylabel("Fraction of time", fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            self._fig.legend(handles, labels, fontsize=8, loc='upper right',
                             bbox_to_anchor=(0.99, 0.99))

    # ------------------------------------------------------------------
    # New LUPE-inspired visualizations
    # ------------------------------------------------------------------

    def _plot_temporal_probability(self):
        """Stacked area chart of behavior fractions over time (LUPE panel e).
        Re-bins live at the current Bin width."""
        tp = self._temporal_probs_current()
        if not tp:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Run Compute Transitions first.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        cmap = self._get_cmap()
        state_labels = [self._state_name(s) for s in self._states]
        colors = [cmap(i) for i in range(len(self._states))]

        # Group sessions by treatment if key file loaded
        if self._key_df is not None and self._group_matrices:
            groups = {}
            for name in tp:
                subj = self._session_subjects.get(name, name)
                row = self._key_df[self._key_df['Subject'] == subj]
                grp = str(row.iloc[0]['Treatment']) if not row.empty else 'Unknown'
                if grp not in groups:
                    groups[grp] = []
                groups[grp].append(name)

            sorted_groups = self._ordered_groups(groups.keys())
            n_groups = len(sorted_groups)
            self._set_panel_figsize(n_groups, 'v')
            axes = self._fig.subplots(n_groups, 1, squeeze=False)[:, 0]

            for gi, grp in enumerate(sorted_groups):
                ax = axes[gi]
                sessions = groups[grp]
                # Average temporal probs across sessions in group
                # Align to common time grid via interpolation
                all_centers = []
                all_probs = []
                for name in sessions:
                    centers, prob = tp[name]
                    all_centers.append(centers)
                    all_probs.append(prob)

                if len(sessions) == 1:
                    t = all_centers[0]
                    mean_prob = all_probs[0]
                else:
                    # Use the shortest common time range
                    min_len = min(len(c) for c in all_centers)
                    t = all_centers[0][:min_len]
                    stacked = np.stack([p[:min_len] for p in all_probs])
                    mean_prob = stacked.mean(axis=0)

                ax.stackplot(t, mean_prob.T, labels=state_labels,
                             colors=colors, alpha=0.85)
                ax.set_title(grp, fontsize=10)
                ax.set_ylabel("Fraction")
                ax.set_ylim(0, 1)
                if gi == n_groups - 1:
                    ax.set_xlabel("Time (s)")
                if gi == 0:
                    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                              borderaxespad=0, fontsize=7)
        else:
            # One subplot per session (max 6, then average)
            sessions = sorted(tp.keys())
            if len(sessions) <= 6:
                self._set_panel_figsize(len(sessions), 'v')
                axes = self._fig.subplots(len(sessions), 1,
                                          squeeze=False)[:, 0]
                for si, name in enumerate(sessions):
                    ax = axes[si]
                    centers, prob = tp[name]
                    ax.stackplot(centers, prob.T, labels=state_labels,
                                 colors=colors, alpha=0.85)
                    subj = self._session_subjects.get(name, name)
                    ax.set_title(subj, fontsize=9)
                    ax.set_ylim(0, 1)
                    if si == len(sessions) - 1:
                        ax.set_xlabel("Time (s)")
                    if si == 0:
                        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                                  borderaxespad=0, fontsize=7)
            else:
                # Average all
                ax = self._fig.add_subplot(111)
                min_len = min(len(tp[n][0]) for n in sessions)
                t = tp[sessions[0]][0][:min_len]
                stacked = np.stack(
                    [tp[n][1][:min_len] for n in sessions])
                mean_prob = stacked.mean(axis=0)
                ax.stackplot(t, mean_prob.T, labels=state_labels,
                             colors=colors, alpha=0.85)
                ax.set_title(f"Mean across {len(sessions)} sessions")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Fraction")
                ax.set_ylim(0, 1)
                ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                          borderaxespad=0, fontsize=7)
        self._apply_window_xlim('s')

    def _plot_latent_states(self):
        """Grid of centroid transition matrix heatmaps (LUPE panel h left)."""
        if self._latent_centroids is None:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Enable 'Discover latent behavioral states'\n"
                    "with sliding windows mode and re-compute.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        k = self._latent_centroids.shape[0]
        n_states = len(self._states)
        labels = [self._state_name(s) for s in self._states]

        # Adapt layout to state count
        show_annot = self._show_annot_var.get() and n_states <= 12
        use_square = n_states <= 15
        annot_size = max(5, 9 - n_states // 4) if show_annot else 7
        tick_size = max(4, 8 - n_states // 5)
        # Abbreviate labels if too many states
        if n_states > 10:
            labels = [str(s) for s in self._states]

        n_cols = 3 if k > 2 else k
        n_rows = int(np.ceil(k / n_cols))

        cmap = self._get_cmap()

        axes = self._fig.subplots(n_rows, n_cols, squeeze=False)
        for i in range(k):
            r, c = divmod(i, n_cols)
            ax = axes[r][c]
            mat = self._latent_centroids[i]
            sns.heatmap(mat, annot=show_annot,
                        fmt='.1f' if n_states > 8 else '.2f',
                        annot_kws={'size': annot_size} if show_annot else {},
                        cmap=cmap,
                        xticklabels=labels, yticklabels=labels,
                        ax=ax, vmin=0, vmax=1, square=use_square,
                        cbar=False)
            ax.set_title(f"Latent State {i + 1}", fontsize=9)
            ax.tick_params(labelsize=tick_size)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # Hide unused axes
        for i in range(k, n_rows * n_cols):
            r, c = divmod(i, n_cols)
            axes[r][c].set_visible(False)

    def _plot_state_occupancy(self):
        """Grouped bar chart of latent state occupancy (LUPE panel h right)."""
        if not self._occupancy:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Enable 'Discover latent behavioral states'\n"
                    "with sliding windows mode and re-compute.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        ax = self._fig.add_subplot(111)
        k = self._n_latent
        x = np.arange(k)
        state_labels = [f"LS {i + 1}" for i in range(k)]

        if self._group_occupancy:
            groups = sorted(self._group_occupancy.keys())
            n_g = len(groups)
            width = 0.7 / n_g
            grp_colors = plt.cm.tab10(np.linspace(0, 1, max(n_g, 1)))

            for gi, grp in enumerate(groups):
                offset = (gi - n_g / 2 + 0.5) * width
                means = self._group_occupancy[grp]
                sems = self._group_occupancy_sem[grp]
                ax.bar(x + offset, means, width, yerr=sems,
                       label=grp, color=grp_colors[gi], alpha=0.8,
                       capsize=3)

                # Overlay individual animal dots
                for name, occ in self._occupancy.items():
                    subj = self._session_subjects.get(name, name)
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty and str(row.iloc[0]['Treatment']) == grp:
                        jitter = np.random.uniform(-width * 0.3,
                                                   width * 0.3, k)
                        ax.scatter(x + offset + jitter, occ,
                                   color=grp_colors[gi], edgecolors='black',
                                   linewidths=0.5, s=20, zorder=5, alpha=0.7)

            ax.legend(fontsize=8)
        else:
            # No groups - show per-session bars
            sessions = sorted(self._occupancy.keys())
            n_s = len(sessions)
            width = 0.7 / max(n_s, 1)
            colors = plt.cm.tab20(np.linspace(0, 1, max(n_s, 1)))
            for si, name in enumerate(sessions):
                offset = (si - n_s / 2 + 0.5) * width
                occ = self._occupancy[name]
                subj = self._session_subjects.get(name, name)
                ax.bar(x + offset, occ, width, label=subj,
                       color=colors[si], alpha=0.8)
            if n_s <= 10:
                ax.legend(fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(state_labels)
        ax.set_xlabel("Latent State")
        ax.set_ylabel("Fraction of Time")
        ax.set_title("Latent State Occupancy")
        ax.set_ylim(bottom=0)

    def _plot_pca(self):
        """PCA scatter + loadings (LUPE panels i-l)."""
        if not self._pca_scores or self._pca_model is None:
            ax = self._fig.add_subplot(111)
            msg = ("Enable 'Discover latent behavioral states'\n"
                   "with sliding windows mode and re-compute.\n"
                   "(Requires >= 2 sessions)")
            ax.text(0.5, 0.5, msg, ha='center', va='center',
                    transform=ax.transAxes)
            return

        axes = self._fig.subplots(1, 2, squeeze=False)[0]
        ax_scatter = axes[0]
        ax_load = axes[1]

        # --- Scatter plot ---
        if self._key_df is not None:
            groups = {}
            for name, (pc1, pc2) in self._pca_scores.items():
                subj = self._session_subjects.get(name, name)
                row = self._key_df[self._key_df['Subject'] == subj]
                grp = str(row.iloc[0]['Treatment']) if not row.empty else 'Unknown'
                if grp not in groups:
                    groups[grp] = ([], [])
                groups[grp][0].append(pc1)
                groups[grp][1].append(pc2)

            sorted_groups = self._ordered_groups(groups.keys())
            grp_colors = plt.cm.tab10(np.linspace(0, 1, max(len(sorted_groups), 1)))
            for gi, grp in enumerate(sorted_groups):
                xs, ys = groups[grp]
                ax_scatter.scatter(xs, ys, label=grp, color=grp_colors[gi],
                                   s=60, edgecolors='black', linewidths=0.5,
                                   alpha=0.8)
                # 95% confidence ellipse
                if len(xs) >= 3:
                    from matplotlib.patches import Ellipse
                    mean_x, mean_y = np.mean(xs), np.mean(ys)
                    cov = np.cov(xs, ys)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = eigvals.argsort()[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]
                    angle = np.degrees(np.arctan2(eigvecs[1, 0],
                                                   eigvecs[0, 0]))
                    # 95% CI: chi2 with 2 dof at 0.05 = 5.991
                    scale = np.sqrt(5.991)
                    w = 2 * scale * np.sqrt(max(eigvals[0], 0))
                    h = 2 * scale * np.sqrt(max(eigvals[1], 0))
                    ell = Ellipse(xy=(mean_x, mean_y), width=w, height=h,
                                  angle=angle, facecolor=grp_colors[gi],
                                  alpha=0.15, edgecolor=grp_colors[gi],
                                  linewidth=1.5)
                    ax_scatter.add_patch(ell)

            ax_scatter.legend(fontsize=8)
        else:
            xs = [v[0] for v in self._pca_scores.values()]
            ys = [v[1] for v in self._pca_scores.values()]
            labels = [self._session_subjects.get(n, n)
                      for n in self._pca_scores]
            ax_scatter.scatter(xs, ys, s=60, edgecolors='black',
                               linewidths=0.5)
            for lbl, x, y in zip(labels, xs, ys):
                ax_scatter.annotate(lbl, (x, y), fontsize=7,
                                    textcoords='offset points',
                                    xytext=(5, 5))

        var_explained = self._pca_model.explained_variance_ratio_
        ax_scatter.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
        if len(var_explained) > 1:
            ax_scatter.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
        else:
            ax_scatter.set_ylabel("PC2")
        ax_scatter.set_title("PCA on Latent State Occupancy")
        ax_scatter.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax_scatter.axvline(0, color='gray', linewidth=0.5, linestyle='--')

        # --- Loadings bar chart ---
        k = self._pca_loadings.shape[1]
        x = np.arange(k)
        ls_labels = [f"LS {i + 1}" for i in range(k)]
        width = 0.35
        ax_load.bar(x - width / 2, self._pca_loadings[0], width,
                     label=f"PC1 ({var_explained[0]*100:.1f}%)",
                     color='steelblue')
        if self._pca_loadings.shape[0] >= 2:
            ax_load.bar(x + width / 2, self._pca_loadings[1], width,
                         label=f"PC2 ({var_explained[1]*100:.1f}%)",
                         color='coral')
        ax_load.set_xticks(x)
        ax_load.set_xticklabels(ls_labels, fontsize=8)
        ax_load.set_xlabel("Latent State")
        ax_load.set_ylabel("Loading")
        ax_load.set_title("PCA Loadings")
        ax_load.legend(fontsize=8)
        ax_load.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # ------------------------------------------------------------------
    # Meta-cluster Summary
    # ------------------------------------------------------------------

    def _build_meta_mapping(self):
        """Return dict {original_cluster_id: meta_cluster_id}, or None."""
        if self._merge_info is None:
            return None
        mapping = {}
        for new_id, old_ids in self._merge_info.items():
            for old_id in old_ids:
                mapping[old_id] = new_id
        return mapping

    def _plot_meta_summary(self):
        """UMAP scatter colored by meta-cluster + composition table."""
        has_embedding = (self._model_bundle is not None
                         and 'embedding' in self._model_bundle
                         and 'cluster_labels' in self._model_bundle)

        if not self._states:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data loaded.\nRun Compute Transitions first.",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
            return

        meta_mapping = self._build_meta_mapping()

        if has_embedding:
            axes = self._fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]})
            ax_scatter, ax_table = axes
            self._draw_meta_umap(ax_scatter, meta_mapping)
        else:
            ax_table = self._fig.add_subplot(111)
            if self._model_bundle is None:
                ax_table.set_title(
                    "No embedding available (requires model.pkl from Discover run)",
                    fontsize=10, fontstyle='italic')

        self._draw_composition_table(ax_table, meta_mapping)

    def _draw_meta_umap(self, ax, meta_mapping):
        """Draw UMAP scatter colored by meta-clusters (or original clusters)."""
        embedding = self._model_bundle['embedding']
        orig_labels = self._model_bundle['cluster_labels']

        # Remap labels to meta-clusters if reduction is active
        if meta_mapping is not None:
            plot_labels = np.array([meta_mapping.get(l, -1) for l in orig_labels])
        else:
            plot_labels = orig_labels.copy()

        # Subsample for performance (cap at 150k points)
        n_pts = len(embedding)
        max_pts = 150_000
        if n_pts > max_pts:
            idx = np.random.default_rng(42).choice(n_pts, max_pts, replace=False)
            embedding = embedding[idx]
            plot_labels = plot_labels[idx]

        cmap = self._get_cmap()
        unique_labels = sorted(set(plot_labels))

        for label in unique_labels:
            mask = plot_labels == label
            if label == -1:
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c='lightgrey', s=1, alpha=0.3, label='Noise',
                           rasterized=True)
            else:
                state_idx = self._states.index(label) if label in self._states else label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=[cmap(state_idx)], s=1, alpha=0.4,
                           label=self._state_name(label), rasterized=True)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        n_clusters = len([l for l in unique_labels if l != -1])
        if meta_mapping is not None:
            ax.set_title(f"UMAP colored by meta-clusters ({n_clusters})")
        else:
            ax.set_title(f"UMAP ({n_clusters} clusters)")
        ax.legend(loc='best', fontsize=6, markerscale=5, ncol=max(1, n_clusters // 10))

    def _draw_composition_table(self, ax, meta_mapping):
        """Draw composition table showing cluster details."""
        ax.axis('off')

        # Compute frame counts per state across all sessions
        total_frames = sum(len(s) for s in self._state_seqs.values())

        rows = []
        for sid in self._states:
            label = self._state_name(sid)
            if meta_mapping is not None and self._merge_info is not None:
                merged_from = self._merge_info.get(sid, [sid])
                merged_str = ', '.join(str(x) for x in merged_from)
            else:
                merged_from = [sid]
                merged_str = str(sid)

            # Count frames for this state across all sessions
            frame_count = sum(
                np.sum(seq == sid) for seq in self._state_seqs.values())
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            rows.append([str(sid), label, merged_str,
                         f"{frame_count:,}", f"{pct:.1f}%"])

        if not rows:
            ax.text(0.5, 0.5, "No states to display.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        col_labels = ['ID', 'Label', 'Merged From', 'Frames', '%']
        table = ax.table(cellText=rows, colLabels=col_labels,
                         loc='center', cellLoc='center')

        # Adaptive font size
        n_rows = len(rows)
        font_size = max(7, min(10, 14 - n_rows // 3))
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.3)

        # Style header row
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')

        # Color-code the ID column to match cluster colors
        cmap = self._get_cmap()
        for i, sid in enumerate(self._states):
            cell = table[i + 1, 0]
            state_idx = self._states.index(sid)
            rgba = cmap(state_idx)
            cell.set_facecolor((*rgba[:3], 0.3))

        if meta_mapping is not None:
            ax.set_title("Meta-cluster Composition", fontsize=11,
                         fontweight='bold', pad=10)
        else:
            ax.set_title("Cluster Composition", fontsize=11,
                         fontweight='bold', pad=10)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _output_dir(self):
        folder = self.app.current_project_folder.get()
        out = os.path.join(folder, 'analysis', 'transitions')
        os.makedirs(out, exist_ok=True)
        return out

    def _open_video_preview(self):
        session_name = self._preview_session_var.get()
        if not session_name:
            messagebox.showwarning("No session", "Select a session to preview.",
                                   parent=self)
            return
        if session_name not in self._state_seqs:
            messagebox.showwarning("No data",
                "Run computation first so state sequences are available.",
                parent=self)
            return

        # Resolve video path
        video_path = ''
        for s in self._trans_sessions:
            if s.get('session_name') == session_name:
                video_path = s.get('video') or s.get('video_path', '')
                break
        if not video_path or not os.path.isfile(video_path):
            messagebox.showwarning("Video not found",
                f"Cannot locate video file for '{session_name}'.", parent=self)
            return

        # Prefer the uncapped sequence so the Video Preview can index 1:1 with
        # video frames; fall back to the capped sequence for older runs that
        # didn't snapshot the full length.
        state_seq  = self._state_seqs_full.get(session_name,
                                               self._state_seqs[session_name])
        prob_mat   = self._frame_probs.get(session_name)   # may be None (unsupervised)
        state_names = ['Other'] + [
            cd.get('Behavior_type', f'State {i+1}')
            for i, cd in enumerate(self._loaded_classifiers)
        ]

        TransitionVideoPreview(
            parent=self,
            video_path=video_path,
            state_seq=state_seq,
            prob_matrix=prob_mat,
            state_names=state_names,
            session_name=session_name,
        )

    def _export_matrices(self):
        if not self._matrices:
            messagebox.showwarning("No data", "Run Compute Transitions first.")
            return
        out_dir = self._output_dir()
        labels = [self._state_name(s) for s in self._states]

        # Per-subject
        for name, mat in self._matrices.items():
            subj = self._session_subjects.get(name, name)
            df = pd.DataFrame(mat, index=labels, columns=labels)
            df.to_csv(os.path.join(out_dir, f"transition_matrix_{subj}.csv"))

        # Group means
        for grp, mat in self._group_matrices.items():
            df = pd.DataFrame(mat, index=labels, columns=labels)
            df.to_csv(os.path.join(out_dir, f"group_mean_matrix_{grp}.csv"))

        # Windowed
        for name, wresults in self._windowed.items():
            subj = self._session_subjects.get(name, name)
            rows = []
            for t, mat in wresults:
                row = {'time_center_s': t}
                for i, si in enumerate(self._states):
                    for j, sj in enumerate(self._states):
                        row[f"{self._state_name(si)}->{self._state_name(sj)}"] = mat[i, j]
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, f"windowed_transitions_{subj}.csv"),
                index=False)

        # Latent state exports
        if self._latent_centroids is not None:
            labels_ls = [self._state_name(s) for s in self._states]
            for i in range(self._latent_centroids.shape[0]):
                df = pd.DataFrame(self._latent_centroids[i],
                                  index=labels_ls, columns=labels_ls)
                df.to_csv(os.path.join(out_dir,
                                       f"latent_centroid_{i + 1}.csv"))

        if self._occupancy:
            rows = []
            for name, occ in self._occupancy.items():
                subj = self._session_subjects.get(name, name)
                treatment = ''
                if self._key_df is not None:
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty:
                        treatment = str(row.iloc[0]['Treatment'])
                r = {'session': name, 'subject': subj,
                     'treatment': treatment}
                for si in range(len(occ)):
                    r[f'state_{si + 1}_frac'] = occ[si]
                rows.append(r)
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, 'state_occupancy.csv'), index=False)

        if self._pca_scores:
            rows = []
            for name, (pc1, pc2) in self._pca_scores.items():
                subj = self._session_subjects.get(name, name)
                treatment = ''
                if self._key_df is not None:
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty:
                        treatment = str(row.iloc[0]['Treatment'])
                rows.append({'session': name, 'subject': subj,
                             'treatment': treatment,
                             'PC1': pc1, 'PC2': pc2})
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, 'pca_scores.csv'), index=False)

        self._log_msg(f"Matrices exported to {out_dir}")
        messagebox.showinfo("Export", f"Matrices saved to:\n{out_dir}")

    def _export_sequences(self):
        if not self._state_seqs:
            messagebox.showwarning("No data", "Run Compute Transitions first.")
            return
        out_dir = self._output_dir()
        fps = self._fps_var.get()
        for name, seq in self._state_seqs.items():
            subj = self._session_subjects.get(name, name)
            df = pd.DataFrame({
                'frame': range(len(seq)),
                'time_s': np.arange(len(seq)) / fps,
                'state_id': seq,
                'state_label': [self._state_name(s) for s in seq],
            })
            df.to_csv(os.path.join(out_dir, f"state_sequence_{subj}.csv"),
                      index=False)
        self._log_msg(f"Sequences exported to {out_dir}")
        messagebox.showinfo("Export", f"Sequences saved to:\n{out_dir}")

    def _export_figure(self, fmt='png'):
        if self._fig is None:
            return
        out_dir = self._output_dir()
        view = self._view_var.get().lower().replace(' ', '_')
        path = os.path.join(out_dir, f"{view}.{fmt}")
        self._fig.savefig(path, dpi=200, bbox_inches='tight')
        self._log_msg(f"Figure saved: {path}")
        messagebox.showinfo("Export", f"Figure saved to:\n{path}")

    # ------------------------------------------------------------------
    # Project change hook
    # ------------------------------------------------------------------

    def on_project_changed(self):
        """Called when the project folder changes."""
        self._scan_runs()
        # Auto-find AND auto-load a valid key file (Subject + Treatment columns).
        folder = self.app.current_project_folder.get()
        if folder:
            key_path = self._discover_key_file(folder)
            if key_path:
                self._key_file_var.set(key_path)
                self._load_key_file()
            # Auto-fill results folder (legacy supervised mode)
            results_dir = os.path.join(folder, 'results')
            if os.path.isdir(results_dir) and hasattr(self, '_results_var'):
                self._results_var.set(results_dir)
        # Refresh supervised session list (also fills the Group column)
        self._scan_trans_sessions(silent=True)
        # Refresh the saved-session dropdown to follow the new project.
        self._refresh_saved_sessions()
