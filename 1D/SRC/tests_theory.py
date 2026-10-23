"""tests_theory.py -- checks theory1d.py against brute force (run: python src/tests_theory.py).
On the dense dyadic domain N = M = 2^(n+1) (every position holds one element) the real data structures are
built, every query start of a fine grid (it covers every cell of the continuous start distribution I_s) is run
through the real SRC-search, and the exact level / LOD distributions are compared with Lem. 1, Lem. 2, Thm. 1.
Integer s and half-integer s are tested (the paper allows real s)."""
import numpy as np
from structures1d import RangeDag1D
import theory1d as T


def brute(N, s, c, q):
    """Exact distributions with starts x_i = (i + 1/2)/q, i = 0 .. (N - s) q - 1  (q = 1 for integer s, 2 for half-integers)."""
    times = np.arange(N)
    tree = RangeDag1D(times, N, 2)
    dag = RangeDag1D(times, N, c)
    m = int(round((N - s) * q))
    lT, lD = [], []
    for i in range(m):
        x = (i + 0.5) / q
        lT.append(tree.tree_src(x, s)[1])
        lD.append(dag.dag_src(x, s)[1])
    lT, lD = np.array(lT), np.array(lD)
    K = T.kappa(N, s)
    pm = lambda a: np.bincount(a, minlength=K + 1)[:K + 1] / len(a)
    assert lT.max() <= K and lD.max() <= K, "returned level above kappa"
    return pm(lT), pm(lD), pm(lD - lT), (lD - lT).min()


cases = []
for N in (256, 1024):
    for c in (3, 5, 9):
        for s in (1.5, 2, 2.5, 3, 5, 6.5, 9, 12, 17, 24, 31, 33, 48, 63.5, 100, 130, 200, 255):
            if s < N and T.is_valid(N, s, c):
                cases.append((N, s, c))
checked = 0
worst = 0.0
for N, s, c in cases:
    q = 1 if float(s).is_integer() else 2
    eT, eD, eL, mn = brute(N, s, c, q)
    tT, tD, tL = T.tree_level_pmf(N, s), T.dag_level_pmf(N, s, c), T.lod_pmf(N, s, c)
    assert mn >= 0, "negative level difference"
    assert abs(tT.sum() - 1) < 1e-9 and abs(tD.sum() - 1) < 1e-9 and abs(tL.sum() - 1) < 1e-9, (N, s, c)
    assert tL.min() >= -1e-12, (N, s, c, tL)
    err = max(np.abs(eT - tT).max(), np.abs(eD - tD).max(), np.abs(eL - tL).max())
    worst = max(worst, err)
    assert err < 1e-9, (N, s, c, err)
    e_k, e_ar = T.expectations(tL)
    assert e_k < T.bound_level_overhead(c) + 1e-12, (N, s, c, e_k)            # Thm. 2(i)
    assert e_ar >= T.ar_lower_bound(N, s) - 1e-9, (N, s, c, e_ar)            # Thm. 2(ii)
    checked += 1
print(f"  Lem.1, Lem.2 and Thm.1 match brute force on {checked} cases (N in 256,1024; c in 3,5,9; integer and half-integer s); max error {worst:.1e}")
print("  Thm.2(i) and (ii) hold for every tested baseline")
# invalid parameters are rejected, not silently approximated
for bad in [(1024, 1.0, 3), (1024, 2000, 3), (1024, 2.0, 9)]:
    try:
        T.lod_pmf(*bad); raise SystemExit(f"accepted invalid baseline {bad}")
    except ValueError:
        pass
print("  invalid baselines (s<=1, s>N, degenerate) are rejected")
# the fit recovers a known baseline
N = 2 ** 22
emp = dict(enumerate(T.lod_pmf(N, 5003, 3)))
f = T.fit_baseline(emp, N, 3, 5000)
assert f["s_best"] == 5003 and f["l2_best"] < 1e-12
print("  fit_baseline recovers a known baseline (s* = 5003 found from nominal 5000)")
print("ALL THEORY TESTS PASSED")
