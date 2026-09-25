"""Behavioural sequencing for the Transitions tab, ported from the PixelPaws manuscript.

This is the manuscript's validated pipeline (research/syntax_stats.py and
fig_syntax_network.py in the manuscript repo), carried over so the GUI's Sequencing views
and the paper's Figure 8 / S11 are the same computation:

  * per-animal BOUT sequences: runs collapsed to one entry, unscored frames closed over so an
    unscored stretch reads as a direct hand-off between the labelled states on either side
  * per-animal transition tables, each animal subsampled to a common number of transitions
    (uniform pair sampling averaged over `reps` draws), animals under `min_bouts` excluded
  * quasi-independence expected counts by iterative proportional fitting; Pearson residuals
    (residual units, NOT z-scores -- the margins are estimated and the cells correlated)
  * Euclidean distance between residual tables; PCoA (exact, since the distance is Euclidean
    in the flattened residual space); PERMANOVA with the Behrens-Fisher pseudo-F2
    (Anderson et al. 2017) and the permutation space enumerated exhaustively when small,
    so p is exact; restricted-within-animal permutation for paired designs
  * pooled group networks rarefied to a common transition total, width = traffic,
    colour = residual; a difference network drawing every assessable change past MIN_DZ

The fold-enrichment approach (observed / m_j/(1-m_i)) is deliberately absent: on synthetic
cohorts with no sequencing difference it reported one 12 times out of 12, because the ratio's
noise pattern tracks occupancy. See the manuscript module docstrings for the full history.
"""
from __future__ import annotations
from math import comb

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch

# Display rules, identical to the manuscript figures.
MIN_P = 0.06      # draw an edge only if it carries this share of a state's exits
LIM_Z = 6.0       # colour limit for the group networks, in residual units
MIN_DZ = 4.0      # difference network draws every assessable change above this
MIN_NODE = 12     # a state with fewer exits cannot be characterised; drawn dashed
MIN_BOUTS = 100   # animals contributing fewer transitions are excluded from the stats

# ----------------------------------------------------------------------------- statistics core


def pairs_of(seq):
    """Every consecutive (from, to) bout pair in a bout-identity sequence."""
    seq = np.asarray(seq)
    return np.c_[seq[:-1], seq[1:]] if seq.size >= 2 else np.empty((0, 2), int)


def counts_from(pairs, k):
    C = np.zeros((k, k))
    if len(pairs):
        np.add.at(C, (pairs[:, 0], pairs[:, 1]), 1)
    return C


def bout_sequence(state, keep_idx):
    """Ordered labelled bouts with unscored stretches closed over.

    `keep_idx` maps raw state id -> compact index, omitting states that should not
    participate (e.g. an unscored/other id). Runs collapse to one entry each, so the
    diagonal of any table built from this is empty by construction.
    """
    lab = np.array([keep_idx.get(int(v), -1) for v in np.asarray(state)])
    lab = lab[lab >= 0]
    if lab.size < 2:
        return np.empty(0, int)
    return lab[np.diff(lab, prepend=lab[0] - 1) != 0]


def subsampled_counts(seq, k, n_sub, rng, reps=100):
    """Transition counts from n_sub pairs sampled uniformly, averaged over `reps` draws.

    Pairs are sampled uniformly rather than as a contiguous window: a random window covers
    the middle of a session ~140x more often than either end, silently deleting session
    onset -- exactly the part carrying an acute manipulation.
    """
    pr = pairs_of(seq)
    if not len(pr):
        return np.zeros((k, k))
    n = int(min(n_sub, len(pr)))
    draws = []
    for _ in range(reps):
        idx = rng.choice(len(pr), size=n, replace=False)
        draws.append(counts_from(pr[idx], k))
    return np.mean(draws, axis=0)


def expected_qi(O, iters=300, eps=1e-9):
    """Expected counts under quasi-independence: both margins matched, diagonal zero."""
    k = O.shape[0]
    E = (~np.eye(k, dtype=bool)).astype(float)
    r, c = O.sum(1), O.sum(0)
    for _ in range(iters):
        rs = E.sum(1)
        E = E * np.where(rs > eps, r / np.maximum(rs, eps), 0.0)[:, None]
        cs = E.sum(0)
        E = E * np.where(cs > eps, c / np.maximum(cs, eps), 0.0)[None, :]
    return E


def residuals(O):
    """Pearson residuals from the quasi-independence fit, in residual units."""
    O = np.asarray(O, float)
    E = expected_qi(O)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (O - E) / np.sqrt(np.maximum(E, 1e-9))
    z[~np.isfinite(z)] = 0.0
    np.fill_diagonal(z, 0.0)
    return z


def pairwise_distance(counts_list):
    """Euclidean distance between animals in residual space. No row weighting."""
    R = [residuals(x) for x in counts_list]
    n = len(R)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.sqrt(((R[i] - R[j]) ** 2).sum()))
    return D


def pcoa(counts_list, n_axes=2):
    """Principal coordinates of the residual tables: (coords, % variance, loadings).

    Exact and deterministic, because the distance is Euclidean in the flattened residual
    space; iterative MDS buys nothing here except a seed dependence.
    """
    Z = np.array([residuals(c).ravel() for c in counts_list])
    Zc = Z - Z.mean(0)
    U, S_, Vt = np.linalg.svd(Zc, full_matrices=False)
    var = S_ ** 2
    var = var / max(var.sum(), 1e-12) * 100
    coords = U[:, :n_axes] * S_[:n_axes]
    return coords, var[:n_axes], Vt[:n_axes]


def _ss(D, y):
    n = len(y)
    SST = (D ** 2).sum() / (2 * n)
    SSW = 0.0
    for g in np.unique(y):
        idx = np.where(y == g)[0]
        if idx.size > 1:
            SSW += (D[np.ix_(idx, idx)] ** 2).sum() / (2 * idx.size)
    return SST, SSW


def _perm_space(y, blocks, max_exact):
    """Every distinct label assignment under the design, or None if too many to enumerate."""
    from itertools import combinations, permutations, product
    y = np.asarray(y)
    if blocks is None:
        lv = np.unique(y)
        if len(lv) != 2:
            return None
        idx = np.arange(len(y))
        n1 = int((y == lv[1]).sum())
        if comb(len(y), n1) > max_exact:
            return None
        out = []
        for pick in combinations(idx, n1):
            lab = np.full(len(y), lv[0])
            lab[list(pick)] = lv[1]
            out.append(lab)
        return out
    variants = [sorted(set(permutations(y[b]))) for b in blocks]
    total = 1
    for v in variants:
        total *= len(v)
        if total > max_exact:
            return None
    out = []
    for combo in product(*variants):
        lab = y.copy()
        for b, vals in zip(blocks, combo):
            lab[b] = vals
        out.append(lab)
    return out


def permanova_bf(D, y, n_perm=4999, rng=None, strata=None, max_exact=200_000):
    """Behrens-Fisher modified PERMANOVA (pseudo-F2; Anderson et al. 2017).

    Returns (F2, p, R2). Identical to classic PERMANOVA at equal group sizes; on unbalanced
    cohorts it estimates each group's dispersion separately, so the null is equality of
    centroids given possibly unequal dispersions. `strata` (one label per observation,
    normally the animal) restricts permutation within animal for paired designs. When the
    permutation space is enumerable the p value is exact.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y)
    n = len(y)
    blocks = None
    if strata is not None:
        strata = np.asarray(strata)
        blocks = [np.where(strata == s)[0] for s in np.unique(strata)]

        def shuffle(lab):
            out = lab.copy()
            for b in blocks:
                out[b] = rng.permutation(lab[b])
            return out
    else:
        def shuffle(lab):
            return rng.permutation(lab)

    def F2(lab):
        SST, SSW = _ss(D, lab)
        den = 0.0
        for g in np.unique(lab):
            idx = np.where(lab == g)[0]
            ni = idx.size
            if ni < 2:
                continue
            Vi = (D[np.ix_(idx, idx)] ** 2).sum() / 2 / (ni * (ni - 1))
            den += (1 - ni / n) * Vi
        return (SST - SSW) / den if den > 0 else np.nan

    obs = F2(y)
    space = _perm_space(y, blocks, max_exact)
    SST, SSW = _ss(D, y)
    if space is not None:
        null = np.array([F2(l) for l in space])
        return obs, float(np.mean(null >= obs)), (SST - SSW) / SST
    null = np.array([F2(shuffle(y)) for _ in range(n_perm)])
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1)), (SST - SSW) / SST


def dodge(xy, tol=0.018, r=0.024):
    """Spread near-coincident points onto a small ring so every animal stays countable."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return xy.copy()
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-9)
    out, used = xy.copy(), np.zeros(len(xy), bool)
    for i in range(len(xy)):
        if used[i]:
            continue
        grp = [j for j in range(len(xy))
               if not used[j] and np.hypot(*(xy[j] - xy[i])) < tol * span]
        used[grp] = True
        if len(grp) < 2:
            continue
        c = xy[grp].mean(0)
        for q, j in enumerate(grp):
            th = 2 * np.pi * q / len(grp)
            out[j] = c + r * span * np.array([np.cos(th), np.sin(th)])
    return out


# ----------------------------------------------------------------------------- colour helpers

# The manuscript's viridis-derived anchors; DIV endpoints match its diverging map.
DIV_LO, DIV_HI = "#3b528b", "#CC79A7"


def mono_cmap(hexc, name="mono"):
    """Single-hue sequential colormap: light tint (low) -> anchor -> dark (high)."""
    r, g, b = to_rgb(hexc)
    light = (r + (1 - r) * 0.86, g + (1 - g) * 0.86, b + (1 - b) * 0.86)
    dark = (r * 0.5, g * 0.5, b * 0.5)
    return LinearSegmentedColormap.from_list(name, [light, (r, g, b), dark])


def div_cmap(lo=None, hi=None, name="div"):
    """Diverging colormap for signed values: dark cool -> near-white (0) -> dark warm."""
    l, h = to_rgb(lo or DIV_LO), to_rgb(hi or DIV_HI)
    deep_l = tuple(c * 0.55 for c in l)
    deep_h = tuple(c * 0.55 for c in h)
    return LinearSegmentedColormap.from_list(
        name, [deep_l, l, (0.975, 0.975, 0.975), h, deep_h])


# ----------------------------------------------------------------------------- pooled networks


def rarefy_pooled(C, T, rng, reps=100):
    """Pooled table subsampled to T transitions, averaged over `reps` draws.

    A Pearson residual scales with the square root of its table total, so two groups with
    different amounts of behaviour carry different residual magnitudes for the same
    underlying preference. Rarefying both pooled tables to the common total removes that
    exactly, the same way the per-animal tables are already equalised.
    """
    flat = np.rint(np.asarray(C, float).ravel()).astype(np.int64)
    T = int(min(T, flat.sum()))
    if T <= 0:
        return np.zeros(np.shape(C))
    draws = [rng.multivariate_hypergeometric(flat, T) for _ in range(reps)]
    return np.mean(draws, axis=0).reshape(np.shape(C))


def draw_network(ax, counts, labels, title, node_col, cmap, zlim=LIM_Z, label_r=1.32,
                 min_p=MIN_P, min_node=MIN_NODE):
    """Pooled group network: edge width = traffic, edge colour = Pearson residual.

    Pooled, not mean-of-ratios -- averaging per-animal ratios destroys any transition most
    animals never performed. Width is the share of the source state's exits; colour is
    whether that beats what the margins predict. A fat neutral edge is common but
    unremarkable; a saturated thin one is rare but specific.
    """
    k = len(labels)
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, k, endpoint=False)
    pos = np.c_[np.cos(ang), np.sin(ang)]
    rows = counts.sum(1)
    P = counts / np.maximum(rows[:, None], 1e-9)
    occ = counts.sum(0) / max(counts.sum(), 1e-9)
    L = residuals(counts)
    ex = [(P[i, j], i, j) for i in range(k) for j in range(k)
          if i != j and P[i, j] >= min_p and rows[i] >= min_node]
    ex.sort()
    for v, i, j in ex:
        val = L[i, j] if np.isfinite(L[i, j]) else 0.0
        ax.add_patch(FancyArrowPatch(
            pos[i] * 0.82, pos[j] * 0.82, connectionstyle="arc3,rad=0.18",
            arrowstyle="-|>", mutation_scale=6 + 16 * v,
            linewidth=0.5 + 5.0 * v,
            color=cmap((np.clip(val, -zlim, zlim) + zlim) / (2 * zlim)),
            alpha=0.92, shrinkA=13, shrinkB=15, zorder=2))
    for i, name in enumerate(labels):
        r = 0.085 + 0.20 * np.sqrt(occ[i])
        thin = rows[i] < min_node
        ax.add_patch(Circle(pos[i] * 0.82, r, facecolor="#F4F4F4" if thin else "white",
                            edgecolor="#BBBBBB" if thin else node_col,
                            linestyle=(0, (2, 2)) if thin else "-",
                            linewidth=1.3 if thin else 1.7, zorder=4))
        ax.text(*(pos[i] * label_r), name, ha="center", va="center", fontsize=7.5,
                color="#999999" if thin else "black", zorder=5)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, pad=2)


def draw_diff(ax, c_ctrl, c_treat, labels, cmap, min_dz=MIN_DZ, sign="both",
              vmax=None, label_r=1.32, min_p=MIN_P, min_node=MIN_NODE):
    """One network of what changed: every assessable change above min_dz residual units.

    Both sides are residuals from their OWN margins, so a destination simply becoming more
    common cannot move an edge. The threshold is a display choice applied identically to
    every dataset, not a per-edge significance test; the PERMANOVA tests the complete
    residual profile at the animal level. Decreases are dashed as well as blue.
    Returns (vmax, n_changed, n_assessable).
    """
    k = len(labels)
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, k, endpoint=False)
    pos = np.c_[np.cos(ang), np.sin(ang)]

    def parts(C):
        rows = C.sum(1)
        return residuals(C), C / np.maximum(rows[:, None], 1e-9), rows

    Lc, Pc, rc = parts(c_ctrl)
    Lt, Pt, rt = parts(c_treat)
    D = Lt - Lc
    ok = (rc >= min_node) & (rt >= min_node)
    busy = np.maximum(Pc, Pt) >= min_p
    ex = [(abs(D[i, j]), D[i, j], i, j) for i in range(k) for j in range(k)
          if i != j and ok[i] and busy[i, j] and np.isfinite(D[i, j])]
    n_elig = len(ex)
    ex = sorted([e for e in ex if e[0] >= min_dz], reverse=True)
    if sign == "pos":
        ex = [e for e in ex if e[1] > 0]
    elif sign == "neg":
        ex = [e for e in ex if e[1] < 0]
    if vmax is None:
        vmax = max([e[0] for e in ex], default=min_dz)
    for mag, val, i, j in sorted(ex):
        ax.add_patch(FancyArrowPatch(
            pos[i] * 0.82, pos[j] * 0.82, connectionstyle="arc3,rad=0.18",
            arrowstyle="-|>", mutation_scale=11, linewidth=2.1,
            linestyle="-" if (sign != "both" or val > 0) else (0, (3.5, 2.0)),
            color=cmap((np.clip(val / vmax, -1, 1) + 1) / 2), alpha=0.95,
            shrinkA=13, shrinkB=15, zorder=2))
    occ = 0.5 * ((c_ctrl.sum(0) / max(c_ctrl.sum(), 1e-9))
                 + (c_treat.sum(0) / max(c_treat.sum(), 1e-9)))
    for i, name in enumerate(labels):
        r = 0.085 + 0.20 * np.sqrt(occ[i])
        ax.add_patch(Circle(pos[i] * 0.82, r, facecolor="white" if ok[i] else "#F4F4F4",
                            edgecolor="#555555" if ok[i] else "#BBBBBB",
                            linestyle="-" if ok[i] else (0, (2, 2)),
                            linewidth=1.7 if ok[i] else 1.3, zorder=4))
        ax.text(*(pos[i] * label_r), name, ha="center", va="center", fontsize=8,
                color="black" if ok[i] else "#999999", zorder=5)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    return vmax, len(ex), n_elig


# ----------------------------------------------------------------------------- cohort assembly


class SyntaxCohort:
    """Per-animal tables, pooled tables and statistics for one grouped cohort.

    seqs        {session: 1-D int array of per-frame state ids} -- ALREADY priority-resolved
    group_of    {session: group label}; sessions mapping to None are skipped
    keep_idx    {raw state id -> compact index}; states absent from it (e.g. unscored) are
                closed over as direct hand-offs
    labels      display name per compact index
    strata_of   optional {session: animal id} for paired designs (within-animal permutation)
    """

    def __init__(self, seqs, group_of, keep_idx, labels,
                 min_bouts=MIN_BOUTS, reps=100, seed=3, strata_of=None):
        self.labels = list(labels)
        self.k = len(self.labels)
        rng = np.random.default_rng(seed)
        bouts, dropped = {}, []
        for name, seq in seqs.items():
            g = group_of.get(name)
            if g is None:
                continue
            b = bout_sequence(seq, keep_idx)
            if b.size < min_bouts:              # floor counts BOUTS, as in the manuscript
                dropped.append(name)
                continue
            bouts[name] = (g, b)
        self.dropped = dropped
        self.sessions = list(bouts)
        self.groups_per_session = np.array([bouts[s][0] for s in self.sessions])
        if not self.sessions:
            self.counts = []
            return
        # manuscript convention: subsample every animal to (shortest bout sequence - 1)
        # transitions, i.e. exactly the least active animal's own pair count
        n_sub = min(bouts[s][1].size for s in self.sessions) - 1
        self.n_sub = int(max(n_sub, 1))
        self.counts = [subsampled_counts(bouts[s][1], self.k, self.n_sub, rng, reps)
                       for s in self.sessions]
        # pooled per group, rarefied to the common total so residual magnitudes compare
        self.groups = list(dict.fromkeys(self.groups_per_session))
        raw = {g: np.zeros((self.k, self.k)) for g in self.groups}
        for s in self.sessions:
            g, b = bouts[s]
            raw[g] += counts_from(pairs_of(b), self.k)
        T = int(min(raw[g].sum() for g in self.groups)) if self.groups else 0
        self.pooled = {g: rarefy_pooled(raw[g], T, rng) for g in self.groups}
        self.pooled_raw_totals = {g: int(raw[g].sum()) for g in self.groups}
        self.strata = (np.array([strata_of.get(s) for s in self.sessions])
                       if strata_of else None)

    def ordination(self):
        return pcoa(self.counts)

    def test(self):
        """(F2, p, R2) over all groups; exact p when the permutation space is enumerable."""
        D = pairwise_distance(self.counts)
        strata = self.strata if (self.strata is not None
                                 and not any(v is None for v in self.strata)) else None
        return permanova_bf(D, self.groups_per_session, strata=strata)


# ----------------------------------------------------------------------------- ordination décor


def ellipse(ax, xy, col, level=0.95, lw=1.6, ls="-"):
    """95% normal-theory ellipse from the group's own covariance (chi-square scaling).

    Describes the region expected to contain `level` of the group, NOT a confidence region
    for its centroid -- the inference is the PERMANOVA, never the ellipse. Groups under
    three points, or degenerate in one direction, get no ellipse rather than a needle.
    """
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return False
    mu = xy.mean(0)
    cov = np.cov(xy.T)
    v, vec = np.linalg.eigh(cov)
    o = v.argsort()[::-1]
    v, vec = v[o], vec[:, o]
    if v[1] <= 0 or v[1] / max(v[0], 1e-12) < 0.004:
        return False
    w, h = 2 * np.sqrt(v * chi2.ppf(level, 2))
    e = Ellipse(mu, w, h, angle=np.degrees(np.arctan2(*vec[:, 0][::-1])),
                facecolor="none", edgecolor=col, lw=lw, ls=ls, zorder=2)
    ax.add_artist(e)
    e.set_clip_path(ax.patch)
    # widen the DATA limits so the ellipse stays inside the frame: set_ylim is discarded
    # under aspect="equal", adjustable="datalim"
    hx, hy = np.sqrt(chi2.ppf(level, 2) * np.diag(cov))
    ax.scatter([mu[0] - hx, mu[0] + hx], [mu[1] - hy, mu[1] + hy], s=0, alpha=0, zorder=0)
    return True


def hull(ax, xy, col, lw=1.4, alpha=0.10):
    """Convex hull of a group, outlined and lightly filled."""
    from scipy.spatial import ConvexHull, QhullError
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return False
    try:
        h = ConvexHull(xy)
    except QhullError:
        return False
    pts = xy[np.append(h.vertices, h.vertices[0])]
    ax.fill(pts[:, 0], pts[:, 1], facecolor=col, alpha=alpha, lw=0, zorder=1)
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, zorder=2)
    return True
