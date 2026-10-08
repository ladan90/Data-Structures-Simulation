"""
tests.py -- correctness tests for structures.py (run:  python tests.py [path/to/2D_Tree.py])

A. An INDEPENDENT reference builder (plain Python, real coordinates, no ranks, no numpy) is written
   directly from the definition of Sec. 2.1. The materialised structure CDag2DFull must have exactly the
   same nodes (level, rectangle, |D_v|) and the same child lists, and its tree part must equal the
   reference 2D-Tree (c = 2).
B. Structural properties: |D_v| = |D ∩ range_v|; sizes strictly decrease along edges; the children of a
   node cover its range; leaves are exactly the nodes with |D_v| <= 1; the 2D-Tree is embedded.
C. SRC-search (Tree, exhaustive DAG) equals a brute-force minimum over all nodes containing Q
   (including the number of minimisers and whether tied nodes have different levels);
   LazyCDag2D returns exactly the same answers as CDag2DFull (levels, sizes, ties, visited counts).
D. AR(Q) >= 1 for every query; negative level differences are counted and reported.
E. Figure-2 check on the 16x16 uniform grid (Q1, Q2 of the paper and the tied query Q3).
F. (optional) cross-check of the Tree against the user's 2D_Tree.py on integer data.
"""
import sys, math, random, importlib.util
import os
import numpy as np
from structures import CDag2DFull, LazyCDag2D

# ------------------------------------------------------------------ A. reference builder
def ref_build(points, domain, c):
    """Return levels: list of dict rect -> (size, [child rect per slot or None])."""
    (lo0, hi0), (lo1, hi1) = domain
    root = (lo0, hi0, lo1, hi1)
    levels = [{root: (len(points), None, points)}]
    while True:
        nxt = {}
        for rect, (sz, _, pts) in levels[-1].items():
            if sz <= 1:
                levels[-1][rect] = (sz, None, pts)
                continue
            lvl = len(levels) - 1
            dim = lvl % 2
            us = sorted({p[dim] for p in pts})
            if len(us) < 2:
                dim = 1 - dim
                us = sorted({p[dim] for p in pts})
            r = len(us)
            h = -(-r // 2)
            step = r // (2 * (c - 1))
            spans = [(1, h)] + [(j * step + 1, j * step + h) for j in range(1, c - 1)] + [(h + 1, r)]
            kids = []
            for g, hh in spans:
                lo, hi = (rect[0], rect[1]) if dim == 0 else (rect[2], rect[3])
                left = lo if g == 1 else us[g - 1]
                right = hi if hh == r else us[hh]
                crect = (left, right, rect[2], rect[3]) if dim == 0 else (rect[0], rect[1], left, right)
                cpts = [p for p in pts if us[g - 1] <= p[dim] <= us[hh - 1]]
                if crect not in nxt:
                    nxt[crect] = (len(cpts), None, cpts)
                kids.append(crect)
            levels[-1][rect] = (sz, kids, pts)
        if not nxt:
            break
        levels.append(nxt)
    return levels


def contains(rect, q):
    return rect[0] <= q[0] and q[1] <= rect[1] and rect[2] <= q[2] and q[3] <= rect[3]


def ref_src(levels_dag, levels_tree, q):
    cont = [(sz, lvl, rect) for lvl, L in enumerate(levels_dag) for rect, (sz, _, _) in L.items()
            if contains(rect, q)]
    m = min(c[0] for c in cont)
    mins = [c for c in cont if c[0] == m]
    dag = (max(c[1] for c in mins), m, len(mins), len({c[1] for c in mins}) > 1)
    contT = [(sz, lvl) for lvl, L in enumerate(levels_tree) for rect, (sz, _, _) in L.items() if contains(rect, q)]
    mT = min(c[0] for c in contT)
    tree = (max(c[1] for c in contT if c[0] == mT), mT)
    return tree, dag


# ------------------------------------------------------------------ datasets
def make_int_points(rng, kind, n, M0, M1):
    pts = set()
    if kind == "random":
        while len(pts) < n:
            pts.add((int(rng.integers(0, M0)), int(rng.integers(0, M1))))
    elif kind == "clustered":
        cl = rng.integers(0, [M0, M1], (4, 2))
        while len(pts) < n:
            cc = cl[rng.integers(0, 4)]
            p = np.clip(rng.normal(cc, [M0 / 12, M1 / 12]), 0, [M0 - 1, M1 - 1]).astype(int)
            pts.add((int(p[0]), int(p[1])))
    elif kind == "grid":
        pts = {(i, j) for i in range(M0) for j in range(M1)}
    return sorted(pts)


def run_case(name, pts, domain, c, nq, rng, shapes_frac, quiet=False):
    lat = np.array([p[0] for p in pts], float); lon = np.array([p[1] for p in pts], float)
    full = CDag2DFull(lat, lon, domain, c, verbose=False)
    lazy = LazyCDag2D(lat, lon, domain, c, cache_min_size=8)       # small threshold: exercises the cache
    ref = ref_build(pts, domain, c)
    reft = ref_build(pts, domain, 2)
    # ---- A: same nodes and edges
    ids = {}
    for k in range(full.n_nodes):
        rect = (full.A0[k], full.B0[k], full.A1[k], full.B1[k])
        lvl = int(full.LV[k])
        assert rect in ref[lvl], f"{name}: node {rect} level {lvl} not in reference"
        assert ref[lvl][rect][0] == int(full.SZ[k]), f"{name}: size mismatch at {rect}"
        ids[(lvl, rect)] = k
    assert full.n_nodes == sum(len(L) for L in ref), f"{name}: node counts differ"
    for lvl, L in enumerate(ref):
        for rect, (sz, kids, pts_) in L.items():
            k = ids[(lvl, rect)]
            row = list(full.CH[k])
            if sz <= 1:
                assert kids is None and all(x < 0 for x in row), f"{name}: leaf with children"
            else:
                got = [(full.A0[x], full.B0[x], full.A1[x], full.B1[x]) for x in row]
                assert got == kids, f"{name}: children differ at {rect}"
    tree_ids = {k for k in range(full.n_nodes) if full.INT[k]}
    ref_tree_nodes = {(lvl, rect) for lvl, L in enumerate(reft) for rect in L}
    assert {(int(full.LV[k]), (full.A0[k], full.B0[k], full.A1[k], full.B1[k])) for k in tree_ids} == ref_tree_nodes, \
        f"{name}: embedded tree differs from the reference 2D-Tree"
    # ---- B: properties
    P = np.stack([lat, lon], 1)
    for k in range(full.n_nodes):
        cnt = int(((P[:, 0] >= full.A0[k]) & (P[:, 0] < full.B0[k]) & (P[:, 1] >= full.A1[k]) & (P[:, 1] < full.B1[k])).sum())
        assert cnt == int(full.SZ[k]), f"{name}: |D_v| != |D ∩ range_v|"
        row = [x for x in full.CH[k] if x >= 0]
        for x in row:
            assert full.SZ[x] < full.SZ[k] and full.LV[x] == full.LV[k] + 1
            assert full.A0[x] >= full.A0[k] and full.B0[x] <= full.B0[k] and full.A1[x] >= full.A1[k] and full.B1[x] <= full.B1[k]
        if row:
            # children cover the range of the node
            for lo_a, hi_a in ((full.A0, full.B0), (full.A1, full.B1)):
                spans = sorted({(lo_a[x], hi_a[x]) for x in row})
                if any((lo_a[x], hi_a[x]) != (lo_a[k], hi_a[k]) for x in row):   # split along this dim
                    cover = spans[0][0]; top = spans[0][0]
                    for a, b in spans:
                        assert a <= top; top = max(top, b)
                    assert spans[0][0] == lo_a[k] and top == hi_a[k], f"{name}: children do not cover range"
    # ---- C, D: SRC
    neg = pos = zero = multi = 0
    (lo0, hi0), (lo1, hi1) = domain
    for _ in range(nq):
        fa, fb = shapes_frac[rng.integers(len(shapes_frac))]
        s0, s1 = fa * (hi0 - lo0), fb * (hi1 - lo1)
        x0 = rng.uniform(lo0, hi0 - s0); x1 = rng.uniform(lo1, hi1 - s1)
        q = (x0, x0 + s0, x1, x1 + s1)
        rt, rd = ref_src(ref, reft, q)
        nT, lT, sT, vT = full.tree_src(x0, x1, s0, s1)
        nD, lD, sD, nmin, tied, vis, tst = full.dag_src(x0, x1, s0, s1)
        assert (lT, sT) == rt, f"{name}: tree SRC differs from brute force"
        assert (lD, sD, nmin, tied) == rd, f"{name}: DAG SRC differs from brute force {(lD, sD, nmin, tied)} vs {rd}"
        kT, lT2, sT2, vT2 = lazy.tree_src(x0, x1, s0, s1)
        kD, lD2, sD2, nmin2, tied2, vis2, tst2 = lazy.dag_src(x0, x1, s0, s1)
        assert (lT2, sT2, vT2) == (lT, sT, vT), f"{name}: lazy tree SRC differs"
        assert (lD2, sD2, nmin2, tied2, vis2, tst2) == (lD, sD, nmin, tied, vis, tst), f"{name}: lazy DAG SRC differs"
        assert kD == (lD, full.A0[nD], full.B0[nD], full.A1[nD], full.B1[nD]), f"{name}: lazy and full return different nodes"
        assert kT == (lT, full.A0[nT], full.B0[nT], full.A1[nT], full.B1[nT]), f"{name}: lazy and full tree return different nodes"
        assert sT / sD >= 1.0, f"{name}: AR < 1"
        d = lD - lT
        neg += d < 0; pos += d > 0; zero += d == 0; multi += nmin > 1
    if not quiet:
        print(f"  [{name}] N={len(pts)} c={c}: nodes={full.n_nodes} (tree {full.n_tree_nodes}), odd-r nodes={full.stats['odd_r_nodes']}; "
              f"{nq} queries: level_diff <0/0/>0 = {neg}/{zero}/{pos}; queries with several minimisers = {multi}")
    return full


def test_figure2():
    pts = [(i, j) for i in range(16) for j in range(16)]
    lat = np.array([p[0] for p in pts], float); lon = np.array([p[1] for p in pts], float)
    for cls in (CDag2DFull, LazyCDag2D):
        S = cls(lat, lon, ((0, 16), (0, 16)), 3, **({"verbose": False} if cls is CDag2DFull else {}))
        r = S.tree_src(2, 2, 3, 3); assert (r[1], r[2]) == (2, 64)                    # Q1, 2D-Tree
        r = S.dag_src(2, 2, 3, 3); assert (r[1], r[2], r[3]) == (4, 16, 1)             # Q1, 3-2D-DAG, unique node
        r = S.tree_src(12, 12, 3, 3); assert (r[1], r[2]) == (4, 16)                   # Q2, 2D-Tree
        r = S.dag_src(12, 12, 3, 3); assert (r[1], r[2], r[3]) == (4, 16, 1)           # Q2, 3-2D-DAG
        r = S.tree_src(2, 2, 3, 2); assert (r[1], r[2]) == (2, 64)                     # Q3 = [2,5)x[2,4)
        r = S.dag_src(2, 2, 3, 2); assert (r[1], r[2], r[3]) == (4, 16, 2)             # two nodes of 16 elements
    print("  [figure 2] Q1: Tree level 2 (64) / DAG level 4 (16); Q2: both level 4 (16); Q3: two DAG nodes tie. OK")


def test_user_tree(path):
    spec = importlib.util.spec_from_file_location("user_tree", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rng = np.random.default_rng(5)
    for kind in ("random", "clustered"):
        pts = make_int_points(rng, kind, 300, 40, 50)
        T = mod.KDTree([tuple(p) for p in pts], cutoff=1, domain=(40, 50))
        lat = np.array([p[0] for p in pts], float); lon = np.array([p[1] for p in pts], float)
        S = CDag2DFull(lat, lon, ((0, 40), (0, 50)), 3, verbose=False)
        mine = {(int(S.LV[k]), (S.A0[k], S.B0[k], S.A1[k], S.B1[k])) for k in range(S.n_nodes) if S.INT[k]}
        theirs = {(n.depth, (n.bbox[0][0], n.bbox[0][1], n.bbox[1][0], n.bbox[1][1])) for n in T.nodes()}
        assert mine == theirs, "2D_Tree.py and structures.py build different trees"
        for _ in range(500):
            s0, s1 = rng.uniform(1, 12), rng.uniform(1, 15)
            x0, x1 = rng.uniform(0, 40 - s0), rng.uniform(0, 50 - s1)
            assert T.SRC([(x0, x0 + s0), (x1, x1 + s1)]).depth == S.tree_src(x0, x1, s0, s1)[1]
    print("  [2D_Tree.py] same tree and same SRC levels as structures.py on integer data. OK")


if __name__ == "__main__":
    print("running tests ...")
    rng = np.random.default_rng(7)
    frac = [(0.1, 0.1), (0.25, 0.25), (0.1, 0.3), (0.4, 0.3), (0.05, 0.05)]
    for kind, n, M0, M1 in [("random", 90, 24, 40), ("random", 101, 30, 30), ("clustered", 150, 40, 60), ("clustered", 211, 64, 64),
                            ("grid", 256, 16, 16), ("grid", 144, 12, 12)]:
        pts = make_int_points(rng, kind, n if kind != "grid" else 0, M0, M1)
        run_case(f"{kind} {M0}x{M1}", pts, ((0, M0), (0, M1)), 3, 400, rng, frac)
    # real-valued coordinates with an offset domain, like (lat, lon) in degrees
    pts = make_int_points(rng, "clustered", 160, 50, 80)
    pts_f = [(-90 + p[0] * 3.37, -180 + p[1] * 4.11) for p in pts]
    run_case("real-valued, offset domain", pts_f, ((-90.0, 90.0), (-180.0, 180.0)), 3, 400, rng, frac)
    # real Gowalla prefix (needs gowalla_D.npz from data_prep.py)
    try:
        d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gowalla_D.npz"))
        pts_g = list(zip(d["lat"][:300].tolist(), d["lon"][:300].tolist()))
        run_case("gowalla first 300", pts_g, ((-90.0, 90.0), (-180.0, 180.0)), 3, 400, rng, frac)
    except FileNotFoundError:
        print("  (data/gowalla_D.npz not found: run data_prep.py first; skipped the Gowalla-prefix case)")
    test_figure2()
    if len(sys.argv) > 1:
        test_user_tree(sys.argv[1])
    print("ALL TESTS PASSED")
