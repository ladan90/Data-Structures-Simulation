"""
theory.py -- the uniform-baseline formulas of Section 3 (c = 3 is what we use; Lemma 3.3 holds for c = 2^alpha+1).

  tree_level_pmf  : Lemma 3.2(ii), distribution of the level returned by SRC-search on the 2D-Tree
  dag_level_pmf   : Lemma 3.3(ii), same for the c-2D-DAG (support L*-2 .. L*)
  lod_pmf         : the frame TQD_c(N, s) of Lemma 3.4, distribution of level_diff
  fit_baseline    : search (s0*, s1*) near the nominal values minimising the L2 distance to an empirical LOD
All formulas assume uniform occupancy and N_i = 2^(n_i+1) (dyadic), and the shape assumptions.
"""
import math
import numpy as np


def kappa(N, s):
    return int(math.floor(math.log2(N / s) + 1e-12))


def shape_params(N0, N1, s0, s1):
    k0, k1 = kappa(N0, s0), kappa(N1, s1)
    return k0, k1, min(2 * k0, 2 * k1 + 1)


def shape_ok(N0, N1, s0, s1, c=3):
    """Shape assumptions: 1 < s_i < N_i and 2^kappa_i <= N_i/(c-1)."""
    if not (1 < s0 < N0 and 1 < s1 < N1):
        return False
    k0, k1, _ = shape_params(N0, N1, s0, s1)
    return 2 ** k0 <= N0 / (c - 1) and 2 ** k1 <= N1 / (c - 1)


# ---------------------------------------------------------------- Lemma 3.2 (2D-Tree)
def tree_level_pmf(N0, N1, s0, s1):
    """P[level_Tree = l], l = 0..L*  (returns a numpy array of length L*+1)."""
    k0, k1, Ls = shape_params(N0, N1, s0, s1)
    D = (N0 - s0) * (N1 - s1)
    ge = [(N0 - 2 ** ((l + 1) // 2) * s0) * (N1 - 2 ** (l // 2) * s1) / D for l in range(Ls + 1)]   # P[level >= l]
    return np.array([ge[l] - ge[l + 1] for l in range(Ls)] + [ge[Ls]])


# ---------------------------------------------------------------- Lemma 3.3 (c-2D-DAG)
def mu_tail(N, s, d, c):
    """P[mu_i(Q) >= d]  (eq. mu-tail)."""
    if s * (c - 1) <= (c - 2) * N / 2 ** d:
        return 1.0
    return ((c - 1) * 2 ** d - (c - 2)) * (N / 2 ** d - s) / (N - s)


def dag_level_pmf(N0, N1, s0, s1, c=3):
    """P[level_cDAG = l], l = 0..L*  (array of length L*+1; mass only on L*-2, L*-1, L*)."""
    k0, k1, Ls = shape_params(N0, N1, s0, s1)
    p0 = mu_tail(N0, s0, -(-Ls // 2), c)
    p1 = mu_tail(N1, s1, Ls // 2, c)
    pmf = np.zeros(Ls + 1)
    pmf[Ls] = p0 * p1
    if Ls >= 1:
        pmf[Ls - 1] = p0 * (1 - p1) if Ls % 2 == 0 else (1 - p0) * p1
    if Ls >= 2:
        pmf[Ls - 2] = (1 - p0) if Ls % 2 == 0 else (1 - p1)
    return pmf


# ---------------------------------------------------------------- Lemma 3.4 (the frame TQD)
def eta_params(N0, N1, s0, s1, c=3):
    k0, k1, Ls = shape_params(N0, N1, s0, s1)
    e0 = max(0.0, (c - 1) * s0 - (c - 2) * N0 / 2 ** k0) if 1 <= k0 <= k1 + 1 else 0.0
    e1 = max(0.0, (c - 1) * s1 - (c - 2) * N1 / 2 ** k1) if 1 <= k1 <= k0 else 0.0
    return e0, e1


def lod_pmf(N0, N1, s0, s1, c=3):
    """P[level_diff = k], k = 0..L*  (array of length L*+1), the frame TQD_c(N, s)."""
    k0, k1, Ls = shape_params(N0, N1, s0, s1)
    e0, e1 = eta_params(N0, N1, s0, s1, c)
    D = (N0 - s0) * (N1 - s1)
    if Ls == 0:
        return np.array([1.0])
    W = {}
    if k0 <= k1:                                              # Case 1, L* = 2 k0
        k = k0; K = 2 ** k
        W[0] = (N0 - K * s0) * (N1 - K * s1) + 2 ** (k - 1) * ((N0 - K * s0) * e1 + (N1 - 2 ** (k - 1) * s1) * e0)
        for j in range(1, k):
            W[2 * j] = (2 ** (k - j) * (s0 - e0) * (N1 - 2 ** (k - j) * s1 - (K - 2 ** (k - j)) * e1)
                        + 2 ** (k - j - 1) * ((N0 - 2 ** (k - j) * s0 - (K - 2 ** (k - j)) * e0) * e1
                                              + (N1 - 2 ** (k - j - 1) * s1) * e0))
        for j in range(0, k - 1):
            W[2 * j + 1] = (2 ** (k - j - 1) * ((N0 - 2 ** (k - j) * s0 - (K - 2 ** (k - j)) * e0) * (s1 - e1)
                                                + (K - 2 ** (k - j - 1)) * (s0 - e0) * e1)
                            + 2 ** (k - j - 2) * (K - 2 ** (k - j - 1)) * s1 * e0)
        W[Ls - 1] = (N0 - 2 * s0 - (K - 2) * e0) * (s1 - e1) + (K - 1) * (s0 - e0) * e1
        W[Ls] = (s0 - e0) * (N1 - s1 - (K - 1) * e1)
    else:                                                     # Case 2, L* = 2 k1 + 1
        k = k1; K = 2 ** k
        W[0] = (N0 - 2 * K * s0) * (N1 - K * s1) + K * (N1 - K * s1) * e0 + 2 ** (k - 1) * (N0 - K * s0) * e1
        for j in range(1, k):
            W[2 * j] = (2 ** (k - j) * ((N0 - 2 ** (k - j + 1) * s0 - (2 * K - 2 ** (k - j + 1)) * e0) * (s1 - e1)
                                        + (N1 - 2 ** (k - j) * s1 - (K - 2 ** (k - j)) * e1) * e0)
                        + 2 ** (k - j - 1) * (N0 - 2 ** (k - j) * s0) * e1)
        for j in range(0, k):
            W[2 * j + 1] = (2 ** (k - j) * (s0 - e0) * (N1 - 2 ** (k - j) * s1 - (K - 2 ** (k - j)) * e1)
                            + 2 ** (k - j - 1) * ((2 * K - 2 ** (k - j)) * (s1 - e1) * e0
                                                  + (K - 2 ** (k - j - 1)) * s0 * e1))
        if k >= 1:
            W[Ls - 1] = (N0 - 2 * s0 - (2 * K - 2) * e0) * (s1 - e1) + (N1 - s1 - (K - 1) * e1) * e0
        W[Ls] = (s0 - e0) * (N1 - s1 - (K - 1) * e1)
    return np.array([W.get(i, 0.0) / D for i in range(Ls + 1)])


# ---------------------------------------------------------------- moments and bounds
def expectations(pmf):
    """(E[level_diff], E[AR]) when level_diff has the given pmf and AR = 2^level_diff (uniform dyadic case)."""
    k = np.arange(len(pmf))
    return float((k * pmf).sum()), float(((2.0 ** k) * pmf).sum())


def bound_level_overhead(c=3):
    return (c - 2) * (5 * c - 3) / (c - 1) ** 2


def ar_lower_bound(N0, N1, s0, s1):
    k0, k1, _ = shape_params(N0, N1, s0, s1)
    m = min(k0, k1)
    return max(1.0, 2 ** (m - 1) - (m + 2) / 4)


# ---------------------------------------------------------------- fit of the baseline to an empirical LOD
def l2(p, q):
    keys = set(p) | set(q)
    return math.sqrt(sum((p.get(k, 0.0) - q.get(k, 0.0)) ** 2 for k in keys))


def fit_baseline(emp_lod, N0, N1, s0_nom, s1_nom, c=3, rel=0.25, npts=41):
    """emp_lod: dict k -> probability. Search s0*, s1* in [(1-rel), (1+rel)] x nominal (npts values each).
    Returns dict with the best and the nominal parameters, their L2 distances and the baseline pmfs."""
    def dist(a, b):
        return l2(emp_lod, dict(enumerate(lod_pmf(N0, N1, a, b, c))))
    best = None
    for a in np.linspace((1 - rel) * s0_nom, (1 + rel) * s0_nom, npts):
        for b in np.linspace((1 - rel) * s1_nom, (1 + rel) * s1_nom, npts):
            if not shape_ok(N0, N1, a, b, c):
                continue
            d = dist(a, b)
            if best is None or d < best[0]:
                best = (d, a, b)
    nominal = dist(s0_nom, s1_nom) if shape_ok(N0, N1, s0_nom, s1_nom, c) else float("nan")
    d, a, b = best
    return {"s0": float(a), "s1": float(b), "l2": d, "nominal_l2": nominal,
            "lod": lod_pmf(N0, N1, a, b, c), "tree": tree_level_pmf(N0, N1, a, b), "dag": dag_level_pmf(N0, N1, a, b, c)}


def nominal_s(s_deg, N, M_deg):
    """Nominal s* (as in the 1D experiment: s / step, step = span / (distinct count - 1)): the side length expressed in
    domain positions when the N distinct positions are spread uniformly over the span M_deg (degrees)."""
    return int(round(s_deg / (M_deg / (N - 1))))


def fit_baseline_window(emp_lod, N0, N1, s0_nom, s1_nom, c=3, radius=500, step=10):
    """Like the 1D experiment: try s_i* = nominal_i + offset for offsets in [-radius, radius] (here with a given step,
    jointly for the two dimensions) and keep the pair whose baseline LOD has the smallest L2 distance to emp_lod."""
    def dist(a, b):
        return l2(emp_lod, dict(enumerate(lod_pmf(N0, N1, a, b, c))))
    offs = list(range(-radius, radius + 1, step))
    if 0 not in offs:
        offs.append(0)
    best = None
    for da in offs:
        for db in offs:
            a, b = s0_nom + da, s1_nom + db
            if not shape_ok(N0, N1, a, b, c):
                continue
            d = dist(a, b)
            if best is None or d < best[0]:
                best = (d, a, b)
    nominal = dist(s0_nom, s1_nom) if shape_ok(N0, N1, s0_nom, s1_nom, c) else float("nan")
    d, a, b = best
    return {"s0": int(a), "s1": int(b), "l2": d, "nominal_l2": nominal, "s0_nom": int(s0_nom), "s1_nom": int(s1_nom),
            "lod": lod_pmf(N0, N1, a, b, c), "tree": tree_level_pmf(N0, N1, a, b), "dag": dag_level_pmf(N0, N1, a, b, c)}
