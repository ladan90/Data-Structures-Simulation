"""tests_theory.py -- checks theory.py against brute force on uniform dyadic grids (run: python tests_theory.py).
A full c-2D-DAG and 2D-Tree are built on the grid; every query start on a half-integer grid (which covers every
cell of the continuous start distribution) is run through the real SRC-search; the exact distributions are compared."""
import numpy as np
from structures import CDag2DFull
import theory as T


def brute(N0, N1, s0, s1, c=3):
    xs, ys = np.meshgrid(np.arange(N0), np.arange(N1), indexing="ij")
    S = CDag2DFull(xs.ravel().astype(float), ys.ravel().astype(float), ((0, N0), (0, N1)), c, verbose=False)
    lT, lD = [], []
    for i in range(int(N0 - s0)):
        for j in range(int(N1 - s1)):
            x0, x1 = i + 0.5, j + 0.5
            lT.append(S.tree_src(x0, x1, s0, s1)[1]); lD.append(S.dag_src(x0, x1, s0, s1)[1])
    lT, lD = np.array(lT), np.array(lD)
    L = T.shape_params(N0, N1, s0, s1)[2]
    pm = lambda a: np.bincount(a, minlength=L + 1)[:L + 1] / len(a)
    assert lT.max() <= L and lD.max() <= L
    return pm(lT), pm(lD), pm(lD - lT)


cases = [(16, 16, 3, 3), (32, 16, 5, 3), (16, 32, 3, 5), (32, 32, 5, 5), (32, 32, 7, 3), (64, 32, 9, 5), (32, 64, 3, 11), (64, 64, 13, 7)]
for N0, N1, s0, s1 in cases:
    assert T.shape_ok(N0, N1, s0, s1), (N0, N1, s0, s1)
    eT, eD, eL = brute(N0, N1, s0, s1)
    tT, tD, tL = T.tree_level_pmf(N0, N1, s0, s1), T.dag_level_pmf(N0, N1, s0, s1), T.lod_pmf(N0, N1, s0, s1)
    errs = [np.abs(eT - tT).max(), np.abs(eD - tD).max(), np.abs(eL - tL).max()]
    assert max(errs) < 1e-9, ((N0, N1, s0, s1), errs)
    assert abs(tT.sum() - 1) < 1e-9 and abs(tD.sum() - 1) < 1e-9 and abs(tL.sum() - 1) < 1e-9
    print(f"  N=({N0},{N1}) s=({s0},{s1}) L*={len(tL)-1}: Lemma 3.2, Lemma 3.3 and the LOD frame match brute force (max error {max(errs):.1e}); "
          f"E[level_diff]={T.expectations(tL)[0]:.4f} <= bound {T.bound_level_overhead():.2f}")
# fit recovers a known baseline
N0 = N1 = 2 ** 20
a, b = 29127.1, 14563.6
emp = dict(enumerate(T.lod_pmf(N0, N1, a, b)))
f = T.fit_baseline(emp, N0, N1, a, b)                       # true point is on the search grid
assert f["l2"] < 1e-12, f["l2"]
f2 = T.fit_baseline(emp, N0, N1, a * 1.05, b * 0.97)         # off-grid start: close, but only grid-accurate
assert f2["l2"] < 1e-3, f2["l2"]
print("  fit_baseline recovers a known baseline LOD (L2 = %.1e on-grid, %.1e off-grid)" % (f["l2"], f2["l2"]))
print("ALL THEORY TESTS PASSED")
