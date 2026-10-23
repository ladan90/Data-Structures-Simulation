"""
structures1d.py -- data-dependent 1D-Tree and c-DAG and SRC-search on both (Sec. 3 of the 1D paper).

Definition implemented (identical in spirit to 2D/src/structures.py, specialised to one dimension)
  * D = u_0 < u_1 < ... < u_{N-1}: sorted distinct integer positions (seconds) inside the domain [0, M).
  * A node v is the tuple (level, lo, hi, a, b):  D_v = {u_lo, ..., u_{hi-1}}  (n_v = hi - lo)  and
    range_v = [a, b),  with  D ∩ range_v = D_v.  The root is (0, 0, N, 0, M).  Leaf  <=>  n_v <= 1.
  * For a non-leaf node with r = n_v elements:  h = ceil(r/2),  step = floor(r / (2(c-1))).
        w_1 = first h elements,                       range [a, u_{lo+h})
        w_c = the remaining r - h elements,          range [u_{lo+h}, b)         (the two Tree children)
        w_{j+1} = h elements starting at element offset j*step  (j = 1..c-2), augmented children.
    A child covering the element offsets g..g'-1 has range [u_{lo+g}, u_{lo+g'}), except that its left
    (right) endpoint is that of the parent if g = 0 (g' = r).  If r < 2(c-1) then step = 0 and every
    augmented child coincides with w_1 (the node has only its two Tree children).
  * Nodes of the same level with the same range are identified (one node, several parents), exactly as
    "identical rectangles at the same level form ONE node" in the 2D code.
  * c = 2 gives the 1D-Tree, c = 2^alpha + 1 (alpha >= 1) the c-DAG.
  * SRC-search (Alg. 1 of the paper = Falzon et al., exhaustive, global 'explored' labels): visit every node
    whose range contains Q, return the DEEPEST one, ties broken in favour of the smallest n_v
    (rule="deepest", the definition of Sec. 3).  rule="minsize" (smallest n_v, ties -> deepest, as in the
    2D code) is available for tests; for N a power of two both rules return the same node set.

A query is Q = [x, x+s) with real x, s;  Q is contained in range_v  <=>  a <= x and x + s <= b.
"""
from array import array
import numpy as np


class RangeDag1D:
    def __init__(self, times, domain_end, c=3, domain_start=0):
        t = np.ascontiguousarray(np.asarray(times), dtype=np.int64)
        if t.ndim != 1 or len(t) < 1:
            raise ValueError("times must be a non-empty 1-D array")
        if len(t) > 1 and not np.all(t[1:] > t[:-1]):
            raise ValueError("times must be strictly increasing (sorted distinct positions)")
        if not (domain_start <= t[0] and t[-1] < domain_end):
            raise ValueError("elements must lie inside [domain_start, domain_end)")
        if not (c == 2 or (c >= 3 and ((c - 1) & (c - 2)) == 0)):
            raise ValueError("c must be 2 (1D-Tree) or 2^alpha + 1")
        self.t = array("q")
        self.t.frombytes(t.tobytes())
        self.N, self.c = len(t), c
        self.domain = (int(domain_start), int(domain_end))
        self.root = (0, 0, self.N, int(domain_start), int(domain_end))

    # ------------------------------------------------------------------ children
    def children(self, node):
        """Distinct children of `node`, in slot order w_1, w_2, ..., w_c ([] for a leaf)."""
        level, lo, hi, a, b = node
        r = hi - lo
        if r <= 1:
            return []
        t, c = self.t, self.c
        h = (r + 1) >> 1
        if c == 2:
            spans = ((0, h), (h, r))
        else:
            step = r // (2 * (c - 1))
            spans = [(0, h)] + [(j * step, j * step + h) for j in range(1, c - 1)] + [(h, r)]
        out = []
        lv = level + 1
        for g, e in spans:
            child = (lv, lo + g, lo + e, a if g == 0 else t[lo + g], b if e == r else t[lo + e])
            if child not in out:
                out.append(child)
        return out

    # ------------------------------------------------------------------ SRC on the embedded 1D-Tree
    def tree_src(self, x, s):
        """SRC-search on the 1D-Tree (the Tree children w_1, w_c): the containing nodes form one root path.
        Returns (node, level, n_v, visited)."""
        qa, qb = x, x + s
        t = self.t
        node = self.root
        visited = 1
        while True:
            lv, lo, hi, a, b = node
            r = hi - lo
            if r <= 1:
                break
            h = (r + 1) >> 1
            m = t[lo + h]
            if a <= qa and qb <= m:
                node = (lv + 1, lo, lo + h, a, m)
            elif m <= qa and qb <= b:
                node = (lv + 1, lo + h, hi, m, b)
            else:
                break
            visited += 1
        return node, node[0], node[2] - node[1], visited

    # ------------------------------------------------------------------ exhaustive SRC (Alg. 1)
    def dag_src(self, x, s, rule="deepest"):
        """Exhaustive SRC-search with global explored labels.
        Returns (node, level, n_v, n_best, visited, tested):
          n_best  = number of containing nodes tied with the returned one (same level and same n_v),
          visited = number of containing nodes expanded, tested = number of containment tests."""
        qa, qb = x, x + s
        root = self.root
        stack = [root]
        seen = {root}
        best, best_lv, best_sz, n_best = root, 0, self.N, 1
        visited, tested = 0, 1
        deepest = rule == "deepest"
        if not deepest and rule != "minsize":
            raise ValueError("rule must be 'deepest' or 'minsize'")
        children = self.children
        while stack:
            node = stack.pop()
            visited += 1
            lv = node[0]
            sz = node[2] - node[1]
            if deepest:
                better = lv > best_lv or (lv == best_lv and sz < best_sz)
            else:
                better = sz < best_sz or (sz == best_sz and lv > best_lv)
            if better:
                best, best_lv, best_sz, n_best = node, lv, sz, 1
            elif lv == best_lv and sz == best_sz and node != best:
                n_best += 1
            for ch in children(node):
                if ch not in seen:
                    seen.add(ch)
                    tested += 1
                    if ch[3] <= qa and qb <= ch[4]:
                        stack.append(ch)
        return best, best_lv, best_sz, n_best, visited, tested

    def count_in(self, x, s):
        """|D ∩ [x, x+s)| for the dataset held by this structure (binary search)."""
        from bisect import bisect_left
        return bisect_left(self.t, x + s) - bisect_left(self.t, x)
