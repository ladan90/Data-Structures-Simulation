"""
structures.py  --  data-dependent 2D-Tree and c-2D-DAG (Section 2.1 of the EDBT submission)
and exhaustive SRC-search on both.

Definition implemented (paper, Sec. 2.1)
  * D is a set of points in the domain [lo0,hi0) x [lo1,hi1); root range = the whole domain.
  * Node v holds D_v = D ∩ range_v. Leaf iff |D_v| <= 1.
  * Split dimension of a node at level l: i(v) = l mod 2 if D_v has >= 2 distinct coordinates
    along l mod 2, otherwise 1 - (l mod 2).
  * u_1 < ... < u_r : distinct coordinates realized by D_v along i(v)   (r >= 2).
  * A child spanning ranks g..h has side [u_g, u_{h+1}) along i(v); the left endpoint is the
    parent's left endpoint if g = 1, the right endpoint is the parent's right endpoint if h = r.
    The side along the other dimension is inherited.
  * Tree children:   w_1 = ranks 1..ceil(r/2)            -> [lo, u_{ceil(r/2)+1})
                     w_c = ranks ceil(r/2)+1 .. r         -> [u_{ceil(r/2)+1}, hi)
    (the median value u_{ceil(r/2)+1} belongs to the right child, the left child is open there).
  * Augmented children w_2..w_{c-1}: w_{j+1} spans ceil(r/2) ranks starting at rank
    j*floor(r/(2(c-1))) + 1, j = 1..c-2   (for c = 3: one augmented child).
  * Level l+1 = the distinct rectangles that are children of level-l nodes; identical rectangles at
    the same level form ONE node linked to all of its parents.
  * The 2D-Tree is the sub-structure of tree children (w_1, w_c); it is embedded in the c-2D-DAG.
  * SRC-search returns a node containing Q with minimum |D_v|; ties are broken in favour of the
    deepest node. On the Tree the containing nodes form a root path (|D_v| strictly decreases).

Two implementations of the SAME structure:
  * CDag2DFull : materialises every node (needs memory ~ number of DAG nodes). Used for tests and
                 for the structure-size table on small prefixes of the data.
  * LazyCDag2D : implicit structure. A node is the pair (level, rectangle); its children are computed
                 on demand from its element set, and every SRC-search visits exactly the nodes that
                 contain Q (the nodes the exhaustive search of the paper examines). Used for N = 2^20.
NOTE (measured): on non-uniform data, identical rectangles from different parents almost never
coincide, so the literal definition of the c-2D-DAG has about N^1.5 nodes (see growth table), and
cannot be materialised for N = 2^20.

Query Q = [x0, x0+s0) x [x1, x1+s1) is contained in range_v  iff  A0<=x0, x0+s0<=B0, A1<=x1, x1+s1<=B1.
"""
import math
import time
from array import array
import numpy as np


class CDag2DFull:
    def __init__(self, lat, lon, domain=((-90.0, 90.0), (-180.0, 180.0)), c=3, verbose=True):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        assert lat.shape == lon.shape and lat.ndim == 1 and len(lat) >= 1
        (lo0, hi0), (lo1, hi1) = domain
        assert lo0 <= lat.min() and lat.max() < hi0, "latitude outside the domain"
        assert lo1 <= lon.min() and lon.max() < hi1, "longitude outside the domain"
        assert len(np.unique(np.stack([lat, lon], 1), axis=0)) == len(lat), "D must be a set"
        assert c >= 3 and (c - 1) & (c - 2) == 0 or c == 3, "c = 2^alpha + 1"
        self.c, self.domain, self.N = c, domain, len(lat)
        self.stats = {"odd_r_nodes": 0, "build_seconds": None}
        self._build(lat, lon, verbose)

    # ------------------------------------------------------------------ construction
    def _build(self, lat, lon, verbose):
        t0 = time.time()
        c, N = self.c, self.N
        (lo0, hi0), (lo1, hi1) = self.domain
        U0, R0 = np.unique(lat, return_inverse=True)          # distinct sorted coordinates, dense ranks
        U1, R1 = np.unique(lon, return_inverse=True)
        U = (U0, U1)
        R = (R0.astype(np.int32), R1.astype(np.int32))

        A0 = array("d", [lo0]); B0 = array("d", [hi0])
        A1 = array("d", [lo1]); B1 = array("d", [hi1])
        SZ = array("i", [N]); LV = array("i", [0])
        CH = array("i", [-1] * c)                              # c child slots per node (flattened)
        INT = bytearray([1])                                   # node belongs to the 2D-Tree
        cur_ids = [0]
        cur_idx = [np.arange(N, dtype=np.int32)]
        level = 0
        odd_r = 0
        while cur_ids:
            nxt = {}                                           # rect -> node id (this next level)
            nxt_ids, nxt_idx = [], []
            for nid, idx in zip(cur_ids, cur_idx):
                if len(idx) <= 1:
                    continue                                   # leaf
                dim = level & 1
                col = R[dim][idx]
                u = np.unique(col)
                if len(u) < 2:
                    dim ^= 1
                    col = R[dim][idx]
                    u = np.unique(col)
                r = len(u)
                assert r >= 2
                if r & 1:
                    odd_r += 1
                h = (r + 1) // 2                               # ceil(r/2)
                step = r // (2 * (c - 1))
                spans = [(1, h)]
                spans += [(j * step + 1, j * step + h) for j in range(1, c - 1)]
                spans.append((h + 1, r))                       # w_c : ranks ceil(r/2)+1 .. r
                uval = U[dim][u]                               # real coordinates u_1..u_r
                if dim == 0:
                    plo, phi = A0[nid], B0[nid]
                else:
                    plo, phi = A1[nid], B1[nid]
                in_tree = INT[nid]
                for slot, (g, hh) in enumerate(spans):
                    assert 1 <= g <= hh <= r
                    left = plo if g == 1 else float(uval[g - 1])
                    right = phi if hh == r else float(uval[hh])
                    if dim == 0:
                        key = (left, right, A1[nid], B1[nid])
                    else:
                        key = (A0[nid], B0[nid], left, right)
                    cid = nxt.get(key)
                    if cid is None:
                        mask = (col >= u[g - 1]) & (col <= u[hh - 1])
                        sub = idx[mask]
                        cid = len(SZ)
                        nxt[key] = cid
                        A0.append(key[0]); B0.append(key[1]); A1.append(key[2]); B1.append(key[3])
                        SZ.append(len(sub)); LV.append(level + 1)
                        CH.extend([-1] * c)
                        INT.append(0)
                        nxt_ids.append(cid); nxt_idx.append(sub)
                    CH[nid * c + slot] = cid
                    if in_tree and (slot == 0 or slot == c - 1):
                        INT[cid] = 1                           # tree child of a tree node
            cur_ids, cur_idx = nxt_ids, nxt_idx
            level += 1
            if verbose:
                print(f"  level {level:2d}: {len(cur_ids):8d} nodes   total {len(SZ):9d}", flush=True)
        self.A0 = np.frombuffer(A0, dtype=np.float64); self.B0 = np.frombuffer(B0, dtype=np.float64)
        self.A1 = np.frombuffer(A1, dtype=np.float64); self.B1 = np.frombuffer(B1, dtype=np.float64)
        self.SZ = np.frombuffer(SZ, dtype=np.int32)
        self.LV = np.frombuffer(LV, dtype=np.int32)
        self.CH = np.frombuffer(CH, dtype=np.int32).reshape(-1, c)
        self.INT = np.frombuffer(INT, dtype=np.uint8).astype(bool)
        self.n_nodes = len(self.SZ)
        self.stats["odd_r_nodes"] = odd_r
        self.stats["build_seconds"] = time.time() - t0
        self._finish()

    def _finish(self):
        c = self.c
        self.height = int(self.LV.max())
        self.n_tree_nodes = int(self.INT.sum())
        ch = self.CH
        # number of distinct edges (merged duplicates counted once)
        e = 0
        for k in range(len(ch)):
            row = ch[k]; row = row[row >= 0]
            e += len(set(row.tolist()))
        self.n_edges = e
        # python-level views for fast scalar access in SRC
        self._a0, self._b0, self._a1, self._b1 = self.A0, self.B0, self.A1, self.B1

    # ------------------------------------------------------------------ SRC on the 2D-Tree
    def tree_src(self, x0, x1, s0, s1):
        """Follow the unique path of tree nodes containing Q; return (node, level, size, visited)."""
        qa0, qb0, qa1, qb1 = x0, x0 + s0, x1, x1 + s1
        A0, B0, A1, B1, CH, c = self.A0, self.B0, self.A1, self.B1, self.CH, self.c
        n = 0
        visited = 1
        while True:
            moved = False
            for slot in (0, c - 1):                            # the two tree children
                ch = CH[n, slot]
                if ch >= 0 and A0[ch] <= qa0 and qb0 <= B0[ch] and A1[ch] <= qa1 and qb1 <= B1[ch]:
                    n = ch; visited += 1; moved = True
                    break
            if not moved:
                break
        return n, int(self.LV[n]), int(self.SZ[n]), visited

    # ------------------------------------------------------------------ exhaustive SRC on the c-2D-DAG
    def dag_src(self, x0, x1, s0, s1):
        """Exhaustive SRC: visit every node containing Q, return a node minimizing |D_v| (ties -> deepest).
        Returns (node, level, size, n_min, tie_levels_differ, visited, tested):
          n_min   = number of distinct containing nodes that attain the minimum size,
          visited = number of containing nodes examined, tested = number of containment tests."""
        qa0, qb0, qa1, qb1 = x0, x0 + s0, x1, x1 + s1
        A0, B0, A1, B1, CH, SZ, LV = self.A0, self.B0, self.A1, self.B1, self.CH, self.SZ, self.LV
        stack = [0]
        seen = {0}
        best = 0; best_sz = int(SZ[0]); best_lv = int(LV[0]); n_min = 1; min_levels = {best_lv}
        visited = 0; tested = 1
        while stack:
            n = stack.pop()
            visited += 1
            sz = int(SZ[n]); lv = int(LV[n])
            if sz < best_sz:
                best, best_sz, best_lv, n_min, min_levels = n, sz, lv, 1, {lv}
            elif sz == best_sz and n != best:
                n_min += 1; min_levels.add(lv)
                if lv > best_lv:
                    best, best_lv = n, lv
            for ch in CH[n]:
                if ch >= 0 and ch not in seen:
                    seen.add(ch); tested += 1
                    if A0[ch] <= qa0 and qb0 <= B0[ch] and A1[ch] <= qa1 and qb1 <= B1[ch]:
                        stack.append(ch)
        return best, best_lv, best_sz, n_min, len(min_levels) > 1, visited, tested

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        np.savez(path, A0=self.A0, B0=self.B0, A1=self.A1, B1=self.B1, SZ=self.SZ, LV=self.LV,
                 CH=self.CH, INT=self.INT, c=self.c, N=self.N, domain=np.array(self.domain),
                 odd_r=self.stats["odd_r_nodes"], build_seconds=self.stats["build_seconds"])

    @classmethod
    def load(cls, path):
        z = np.load(path)
        o = cls.__new__(cls)
        o.A0, o.B0, o.A1, o.B1 = z["A0"], z["B0"], z["A1"], z["B1"]
        o.SZ, o.LV, o.CH, o.INT = z["SZ"], z["LV"], z["CH"], z["INT"]
        o.c, o.N = int(z["c"]), int(z["N"])
        o.domain = tuple(map(tuple, z["domain"].tolist()))
        o.n_nodes = len(o.SZ)
        o.stats = {"odd_r_nodes": int(z["odd_r"]), "build_seconds": float(z["build_seconds"])}
        o._finish()
        return o

    def summary(self):
        return (f"N={self.N}  c={self.c}  levels={self.height + 1}  DAG nodes={self.n_nodes}  "
                f"Tree nodes={self.n_tree_nodes}  DAG edges={self.n_edges}  "
                f"nodes with odd r={self.stats['odd_r_nodes']}  build={self.stats['build_seconds']:.1f}s")


# ======================================================================================
class LazyCDag2D:
    """Implicit c-2D-DAG / 2D-Tree over D (same definition and same SRC semantics as CDag2DFull)."""

    def __init__(self, lat, lon, domain=((-90.0, 90.0), (-180.0, 180.0)), c=3, cache_min_size=2048):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        (lo0, hi0), (lo1, hi1) = domain
        assert lo0 <= lat.min() and lat.max() < hi0 and lo1 <= lon.min() and lon.max() < hi1
        assert len(np.unique(np.stack([lat, lon], 1), axis=0)) == len(lat), "D must be a set"
        self.c, self.N, self.domain, self.T = c, len(lat), domain, cache_min_size
        self.U0, R0 = np.unique(lat, return_inverse=True)
        self.U1, R1 = np.unique(lon, return_inverse=True)
        self.U = (self.U0, self.U1)
        self.R = (R0.astype(np.int32), R1.astype(np.int32))
        self.root_key = (0, float(lo0), float(hi0), float(lo1), float(hi1))
        self.cache = {}            # key -> [idx, ex]   (only nodes with |D_v| >= cache_min_size)
        self.n_expansions = 0

    # ---- children of a node (computed from its element set) --------------------------------
    def _expand(self, key, idx):
        """Return (dim, [(slot, child_key, child_size, ulo_rank, uhi_rank), ...])."""
        c = self.c
        level, a0, b0, a1, b1 = key
        dim = level & 1
        col = self.R[dim][idx]
        u, cnt = np.unique(col, return_counts=True)
        if len(u) < 2:
            dim ^= 1
            col = self.R[dim][idx]
            u, cnt = np.unique(col, return_counts=True)
        r = len(u)
        h = (r + 1) // 2
        step = r // (2 * (c - 1))
        spans = [(1, h)] + [(j * step + 1, j * step + h) for j in range(1, c - 1)] + [(h + 1, r)]
        cum = np.concatenate(([0], np.cumsum(cnt)))
        uval = self.U[dim][u]
        plo, phi = (a0, b0) if dim == 0 else (a1, b1)
        out = []
        for slot, (g, hh) in enumerate(spans):
            assert 1 <= g <= hh <= r
            left = plo if g == 1 else float(uval[g - 1])
            right = phi if hh == r else float(uval[hh])
            ck = (level + 1, left, right, a1, b1) if dim == 0 else (level + 1, a0, b0, left, right)
            out.append((slot, ck, int(cum[hh] - cum[g - 1]), int(u[g - 1]), int(u[hh - 1])))
        return dim, out

    def _expansion(self, key, size, idx):
        if size >= self.T:
            rec = self.cache.get(key)
            if rec is not None and rec[1] is not None:
                return rec[1]
        ex = self._expand(key, idx)
        self.n_expansions += 1
        if size >= self.T:
            rec = self.cache.get(key)
            if rec is None:
                self.cache[key] = [idx, ex]
            else:
                rec[1] = ex
        return ex

    def _child_idx(self, ckey, csize, pidx, dim, ulo, uhi):
        if csize >= self.T:
            rec = self.cache.get(ckey)
            if rec is not None:
                return rec[0]
        col = self.R[dim][pidx]
        sub = pidx[(col >= ulo) & (col <= uhi)]
        if csize >= self.T:
            self.cache[ckey] = [sub, None]
        return sub

    @staticmethod
    def _contains(key, qa0, qb0, qa1, qb1):
        return key[1] <= qa0 and qb0 <= key[2] and key[3] <= qa1 and qb1 <= key[4]

    # ---- SRC on the 2D-Tree ------------------------------------------------------------------
    def tree_src(self, x0, x1, s0, s1):
        """Return (key, level, size, visited). Follows the unique path of tree children containing Q."""
        qa0, qb0, qa1, qb1 = x0, x0 + s0, x1, x1 + s1
        key, size, idx = self.root_key, self.N, np.arange(self.N, dtype=np.int32)
        visited = 1
        while size > 1:
            dim, ex = self._expansion(key, size, idx)
            moved = False
            for slot in (0, self.c - 1):
                _, ck, cs, ulo, uhi = ex[slot]
                if self._contains(ck, qa0, qb0, qa1, qb1):
                    idx = self._child_idx(ck, cs, idx, dim, ulo, uhi)
                    key, size = ck, cs
                    visited += 1; moved = True
                    break
            if not moved:
                break
        return key, key[0], size, visited

    # ---- exhaustive SRC on the c-2D-DAG ------------------------------------------------------
    def dag_src(self, x0, x1, s0, s1):
        """Return (key, level, size, n_min, tie_levels_differ, visited, tested); same meaning as CDag2DFull."""
        qa0, qb0, qa1, qb1 = x0, x0 + s0, x1, x1 + s1
        root = self.root_key
        stack = [(root, self.N, np.arange(self.N, dtype=np.int32))]
        seen = {root}
        best, best_sz, best_lv = root, self.N, 0
        n_min, min_levels = 1, {0}
        visited, tested = 0, 1
        while stack:
            key, size, idx = stack.pop()
            visited += 1
            lv = key[0]
            if size < best_sz:
                best, best_sz, best_lv, n_min, min_levels = key, size, lv, 1, {lv}
            elif size == best_sz and key != best:
                n_min += 1; min_levels.add(lv)
                if lv > best_lv:
                    best, best_lv = key, lv
            if size <= 1:
                continue
            dim, ex = self._expansion(key, size, idx)
            for slot, ck, cs, ulo, uhi in ex:
                if ck not in seen:
                    seen.add(ck); tested += 1
                    if self._contains(ck, qa0, qb0, qa1, qb1):
                        stack.append((ck, cs, self._child_idx(ck, cs, idx, dim, ulo, uhi) if cs > 1 else None))
        return best, best_lv, best_sz, n_min, len(min_levels) > 1, visited, tested
