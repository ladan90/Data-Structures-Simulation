"""
theory1d.py -- the closed-form results of Sec. 4 (uniform occupancy, N = M = 2^(n+1)) used as baselines.

  tree_level_pmf  : Lem. 1  (level returned by SRC-search on the 1D-Tree)
  dag_level_pmf   : Lem. 2  (level returned by SRC-search on the c-DAG)
  lod_pmf         : Thm. 1  (Level Overhead Distribution, the pmf of level_diff = level_cDAG - level_Tree)
  expectations    : E[level_diff] and E[AR] = E[2^level_diff] of an LOD
  bound_level_overhead / ar_lower_bound : Thm. 2 (i) and (ii)
  fit_baseline    : best uniform baseline I_{s*} for an empirical LOD (L2 metric), as in Sec. 5

For a real query length s with 1 < s <= N let kappa = floor(log2(N/s)).  The results hold under the
non-degeneracy condition c - 1 <= 2^(n-kappa+1) = N / 2^kappa (only needed for kappa >= 1).
"""
import math
import numpy as np


def _check_N(N):
    if N < 2 or N & (N - 1):
        raise ValueError("N must be a power of two, N = 2^(n+1)")


def kappa(N, s):
    return int(math.floor(math.log2(N / s) + 1e-12))


def is_valid(N, s, c):
    """1 < s <= N and (kappa = 0 or c - 1 <= N / 2^kappa)."""
    if not (1 < s <= N):
        return False
    k = kappa(N, s)
    return k == 0 or (c - 1) <= N / 2 ** k


def _require(N, s, c):
    _check_N(N)
    if not is_valid(N, s, c):
        raise ValueError(f"baseline undefined for N={N}, s={s}, c={c} (need 1<s<=N and c-1 <= N/2^kappa)")
    return kappa(N, s)


def tree_level_pmf(N, s):
    """P[level_Tree = l], l = 0..kappa (array of length kappa+1).  Lem. 1."""
    _check_N(N)
    k = kappa(N, s)
    D = N - s
    p = np.array([2.0 ** l * s / D for l in range(k)] + [(N - 2.0 ** k * s) / D])
    return p


def dag_level_pmf(N, s, c):
    """P[level_cDAG = l], l = 0..kappa (mass only on kappa-1 and kappa).  Lem. 2."""
    k = _require(N, s, c)
    p = np.zeros(k + 1)
    D = N - s
    Nk = N / 2.0 ** k
    if k == 0 or s * (c - 1) <= (c - 2) * Nk:                       # Case 1: always level kappa
        p[k] = 1.0
        return p
    p[k] = ((c - 1) * 2.0 ** k - (c - 2)) * (Nk - s) / D             # Case 2
    p[k - 1] = (c - 1) * (2.0 ** k - 1) * (s - (c - 2) * N / ((c - 1) * 2.0 ** k)) / D
    return p


def lod_pmf(N, s, c):
    """P[level_diff = k], k = 0..kappa (array of length kappa+1).  Thm. 1."""
    kap = _require(N, s, c)
    if kap == 0:
        return np.array([1.0])
    D = N - s
    Nk = N / 2.0 ** kap
    p = np.zeros(kap + 1)
    if s * (c - 1) <= (c - 2) * Nk:                                   # Case 1
        p[0] = (N - 2.0 ** kap * s) / D
        for k in range(1, kap + 1):
            p[k] = 2.0 ** (kap - k) * s / D
    else:                                                             # Case 2
        p[0] = (-(c - 4) * N / 2.0 + (c - 3) * 2.0 ** (kap - 1) * s) / D
        for k in range(1, kap):
            p[k] = ((c - 2) * N / 2.0 ** (k + 1) - (c - 3) * 2.0 ** (kap - k - 1) * s) / D
        p[kap] = (c - 2) * (Nk - s) / D
    return p


def expectations(pmf):
    """(E[level_diff], E[AR]) of a level-diff pmf, with AR = 2^level_diff."""
    k = np.arange(len(pmf))
    return float((k * pmf).sum()), float(((2.0 ** k) * pmf).sum())


def bound_level_overhead(c):
    """Thm. 2(i): E[mean level_diff] < 2 (c-2)/(c-1)."""
    return 2.0 * (c - 2) / (c - 1)


def ar_lower_bound(N, s):
    """Thm. 2(ii): E[mean AR] >= max{1, floor(log2(N/s))/2}."""
    return max(1.0, 0.5 * kappa(N, s))


def l2(p, q):
    keys = set(p) | set(q)
    return math.sqrt(sum((p.get(k, 0.0) - q.get(k, 0.0)) ** 2 for k in keys))


def nominal_s(s, N, timespan):
    """Nominal s* = round(s / step), step = timespan / (N - 1): the query length expressed in domain positions
    when the N distinct positions are spread uniformly over the span (as in the original 1D experiment)."""
    return int(round(s / (timespan / (N - 1))))


def fit_baseline(emp_lod, N, c, s_nom, radius=500, step=1):
    """emp_lod: dict k -> probability.  Try s* = s_nom + offset, offset in [-radius, radius] (every `step`),
    keep the valid s* whose baseline LOD has the smallest L2 distance to emp_lod.
    Returns the best and the nominal s*, their L2 distances and the baseline LODs."""
    def dist(s):
        return l2(emp_lod, dict(enumerate(lod_pmf(N, s, c))))
    offs = list(range(-radius, radius + 1, step))
    if 0 not in offs:
        offs.append(0)
    best = None
    for o in offs:
        s = s_nom + o
        if not is_valid(N, s, c):
            continue
        d = dist(s)
        if best is None or d < best[0]:
            best = (d, s)
    if best is None:
        raise ValueError("no valid baseline s* in the search window")
    d, s = best
    nominal = dist(s_nom) if is_valid(N, s_nom, c) else float("nan")
    return {"s_best": s, "l2_best": d, "s_nom": s_nom, "l2_nominal": nominal,
            "lod_best": lod_pmf(N, s, c), "lod_nominal": lod_pmf(N, s_nom, c) if is_valid(N, s_nom, c) else None}
