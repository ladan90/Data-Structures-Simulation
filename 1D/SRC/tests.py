"""tests.py -- correctness tests for structures1d.py and data_prep.py (run: python src/tests.py).

A. An INDEPENDENT reference builder (plain Python lists, real positions, no rank arithmetic shared with
   structures1d.py) is written directly from the definition of Sec. 3.  The nodes, the child lists and the
   embedded 1D-Tree of RangeDag1D must coincide with it.
B. Structural properties: n_v = |D ∩ range_v|; sizes strictly decrease along links; the children of a node cover
   its range; leaves are exactly the nodes with n_v <= 1; the 1D-Tree (c = 2) is embedded in every c-DAG.
C. SRC-search (Tree, exhaustive c-DAG) equals a brute-force optimum over ALL nodes containing Q
   (deepest, ties -> smallest n_v; and the 'minsize' rule of the 2D code), including the number of tied nodes.
D. level_diff >= 0 and AR >= 1 for every query; for N = 2^k also AR = 2^level_diff and both rules agree.
E. Fig. 3 of the paper (N = M = 16, 3-DAG): Q1 = [2,5), Q2 = [12,15).
F. data_prep.seconds_from_datetime is correct for ns / us / ms / s datetime columns.
"""
import random
import numpy as np
import pandas as pd
from structures1d import RangeDag1D


# ------------------------------------------------------------------ A. reference builder
def ref_build(pts, M, c):
    """levels[l] = {(a, b): (n_v, [child ranges in slot order])}  built from real positions (sorted list pts)."""
    levels = [{(0, M): None}]
    while True:
        nxt = {}
        for (a, b) in levels[-1]:
            inside = [p for p in pts if a <= p < b]
            r = len(inside)
            if r <= 1:
                levels[-1][(a, b)] = (r, [])
                continue
            h = -(-r // 2)
            if c == 2:
                spans = [(1, h), (h + 1, r)]
            else:
                step = r // (2 * (c - 1))
                spans = [(1, h)] + [(j * step + 1, j * step + h) for j in range(1, c - 1)] + [(h + 1, r)]
            kids = []
            for g, hh in spans:                                   # ranks g..hh (1-based)
                left = a if g == 1 else inside[g - 1]
                right = b if hh == r else inside[hh]
                if (left, right) not in kids:
                    kids.append((left, right))
                nxt.setdefault((left, right), None)
            levels[-1][(a, b)] = (r, kids)
        if not nxt:
            return levels
        levels.append(nxt)


def contains(rng, x, s):
    return rng[0] <= x and x + s <= rng[1]


def brute_src(levels, pts, x, s, rule="deepest"):
    cont = []
    for lv, L in enumerate(levels):
        for (a, b) in L:
            if contains((a, b), x, s):
                n = sum(1 for p in pts if a <= p < b)
                cont.append((lv, n))
    key = (lambda t: (-t[0], t[1])) if rule == "deepest" else (lambda t: (t[1], -t[0]))
    best = min(cont, key=key)
    return best[0], best[1], sum(1 for t in cont if t == best)


def make_points(rng, kind, n, M):
    if kind == "dense":
        return list(range(n))
    if kind == "random":
        return sorted(rng.sample(range(M), n))
    pts = set()                                                    # clustered
    while len(pts) < n:
        v = rng.gauss(M * 0.3, M * 0.04) if rng.random() < 0.7 else rng.uniform(0, M)
        pts.add(int(min(M - 1, max(0, v))))
    return sorted(pts)


def run_case(name, pts, M, c, nq, rng):
    S = RangeDag1D(pts, M, c)
    T2 = RangeDag1D(pts, M, 2)
    ref, ref2 = ref_build(pts, M, c), ref_build(pts, M, 2)
    # ---- A: same nodes and the same child lists
    for lv, L in enumerate(ref):
        for (a, b), (r, kids) in L.items():
            node = next(n for n in _nodes_at(S, lv) if n[3] == a and n[4] == b)
            assert node[2] - node[1] == r == sum(1 for p in pts if a <= p < b), f"{name}: n_v mismatch at {(a, b)}"
            got = [(ch[3], ch[4]) for ch in S.children(node)]
            assert got == kids, f"{name}: children differ at level {lv} {(a, b)}: {got} vs {kids}"
    assert sum(len(L) for L in ref) == sum(len(_nodes_at(S, lv)) for lv in range(len(ref))), f"{name}: node counts differ"
    # ---- B: properties and embedded tree
    tree_nodes = {(lv, a, b) for lv, L in enumerate(ref2) for (a, b) in L}
    dag_tree = _tree_nodes(S)
    assert dag_tree == tree_nodes, f"{name}: embedded Tree differs from the reference 1D-Tree (c = 2)"
    for lv, L in enumerate(ref):
        for (a, b), (r, kids) in L.items():
            if r <= 1:
                assert kids == [], f"{name}: leaf with children"
                continue
            assert kids and all(a <= ka and kb <= b for ka, kb in kids)
            assert sorted(kids)[0][0] == a and max(kb for _, kb in kids) == b, f"{name}: children do not cover range"
            for ka, kb in kids:
                assert sum(1 for p in pts if ka <= p < kb) < r, f"{name}: size does not decrease"
    # ---- C, D: SRC-search
    dyadic = len(pts) & (len(pts) - 1) == 0
    neg = pos = zero = 0
    for _ in range(nq):
        s = rng.choice([0.5, 1, 1.5, 2, 3, 5, 8.5, 13, 30, 60]) * M / 100.0
        s = max(s, 0.3)
        x = rng.uniform(0, M - s)
        nT, lT, szT, visT = S.tree_src(x, s)
        lt_ref = brute_src(ref2, pts, x, s)
        assert (lT, szT) == lt_ref[:2], f"{name}: Tree SRC differs from brute force at x={x}, s={s}"
        assert S.tree_src(x, s)[:3] == T2.dag_src(x, s)[:3], f"{name}: tree_src differs from exhaustive c=2 search"
        for rule in ("deepest", "minsize"):
            node, lD, szD, nbest, vis, tst = S.dag_src(x, s, rule)
            assert (lD, szD, nbest) == brute_src(ref, pts, x, s, rule), f"{name}: {rule} SRC differs from brute force at x={x}, s={s}"
        node, lD, szD, nbest, vis, tst = S.dag_src(x, s)
        assert lD - lT >= 0, f"{name}: negative level_diff"
        assert szT / szD >= 1.0, f"{name}: AR < 1"
        if dyadic:
            assert szT == szD * 2 ** (lD - lT), f"{name}: AR != 2^level_diff"
            assert S.dag_src(x, s, "deepest")[1:3] == S.dag_src(x, s, "minsize")[1:3]
        d = lD - lT
        neg += d < 0; pos += d > 0; zero += d == 0
    print(f"  [{name}] N={len(pts)} M={M} c={c}: nodes={sum(len(L) for L in ref)}; {nq} queries ok; level_diff 0/>0 = {zero}/{pos}")


def _nodes_at(S, level):
    cur = {S.root}
    for _ in range(level):
        nxt = set()
        for n in cur:
            nxt.update(S.children(n))
        cur = nxt
    return cur


def _tree_nodes(S):
    out, cur = set(), [S.root]
    while cur:
        nxt = []
        for n in cur:
            out.add((n[0], n[3], n[4]))
            kids = S.children(n)
            if kids:
                nxt.extend({kids[0], kids[-1]})                  # tree children = first and last child (w_1, w_c)
        cur = nxt
    return out


def test_figure3():
    S = RangeDag1D(list(range(16)), 16, 3)
    T = S.tree_src
    n, lv, sz, _ = T(2, 3)
    assert (n[3], n[4], lv) == (0, 8, 1), n                          # Q1: Tree returns [0,8)
    n, lv, sz, nb, _, _ = S.dag_src(2, 3)
    assert (n[3], n[4], lv, nb) == (2, 6, 2, 1), n                    # Q1: 3-DAG returns [2,6)
    n, lv, sz, _ = T(12, 3)
    assert (n[3], n[4], lv) == (12, 16, 2), n                         # Q2: Tree returns [12,16)
    n, lv, sz, nb, _, _ = S.dag_src(12, 3)
    assert (n[3], n[4], lv) == (12, 16, 2), n                         # Q2: 3-DAG returns [12,16)
    print("  [Fig. 3] Q1=[2,5): Tree [0,8), 3-DAG [2,6); Q2=[12,15): both [12,16). OK")


def test_seconds():
    base = pd.Series(pd.to_datetime(["2024-01-01 00:00:05", "2024-01-31 23:59:59", "2002-12-31 22:59:39"]))
    want = [1704067205, 1706745599, 1041375579]
    for unit in ("ns", "us", "ms", "s"):
        col = base.astype(f"datetime64[{unit}]")
        from data_prep import seconds_from_datetime
        assert list(seconds_from_datetime(col)) == want, unit
    print("  [data_prep] seconds_from_datetime exact for ns/us/ms/s datetime columns. OK")


if __name__ == "__main__":
    print("running tests ...")
    rng = random.Random(7)
    for kind, n, M in [("dense", 64, 64), ("dense", 128, 128), ("dense", 100, 100), ("random", 64, 300),
                       ("random", 90, 700), ("random", 128, 1000), ("clustered", 100, 2000), ("clustered", 128, 4000),
                       ("random", 37, 90), ("dense", 256, 256)]:
        pts = make_points(rng, kind, n, M)
        for c in (3, 5, 9):
            run_case(f"{kind} c={c}", pts, M, c, 120, rng)
    test_figure3()
    test_seconds()
    print("ALL TESTS PASSED")
