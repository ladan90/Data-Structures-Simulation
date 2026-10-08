import sys, os, time, numpy as np, hashlib
from structures import LazyCDag2D
LAT, LON = (-90.0, 90.0), (-180.0, 180.0)
class S2(LazyCDag2D):
    def minimizers(self, x0, x1, s0, s1):
        qa0, qb0, qa1, qb1 = x0, x0 + s0, x1, x1 + s1
        root = self.root_key
        stack = [(root, self.N, np.arange(self.N, dtype=np.int32))]
        seen = {root}
        best_sz = self.N; mins = {}
        while stack:
            key, size, idx = stack.pop()
            if size < best_sz:
                best_sz = size; mins = {}
            if size == best_sz:
                mins[key] = hashlib.md5(np.sort(idx).tobytes()).hexdigest()
            if size <= 1: continue
            dim, ex = self._expansion(key, size, idx)
            for slot, ck, cs, ulo, uhi in ex:
                if ck not in seen:
                    seen.add(ck)
                    if self._contains(ck, qa0, qb0, qa1, qb1):
                        col = self.R[dim][idx]
                        sub = idx[(col >= ulo) & (col <= uhi)]
                        stack.append((ck, cs, sub))
        return best_sz, mins
def run(args, results_dir):
    shape, limit = args
    s0, s1 = shape
    d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gowalla_D.npz")); S = S2(d["lat"], d["lon"], (LAT, LON), c=3)
    z = np.load(os.path.join(results_dir, f"queries_{s0}x{s1}.npz"))
    nm = z['nmin']; idxs = np.where(nm > 1)[0][:limit]
    out = []; t0 = time.time()
    for i in idxs:
        sz, mins = S.minimizers(z['x0'][i], z['x1'][i], s0, s1)
        assert sz == z['szD'][i] and len(mins) == nm[i], (sz, z['szD'][i], len(mins), nm[i])
        lv = {k: k[0] for k in mins}; L = max(lv.values())
        deep = [k for k in mins if lv[k] == L]
        from collections import Counter
        cnt=np.array(list(Counter(mins.values()).values()),float); p=cnt/cnt.sum(); Hs=float(-(p*np.log2(p)).sum())
        out.append((i, len(mins), len(deep), len(cnt), Hs))
    return shape, np.array(out), time.time() - t0
if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10**9, help="max tied queries per shape")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    lim = a.limit
    shapes = [(5,5),(5,10),(10,10),(25,20)]
    with mp.Pool(4) as p:
        for shape, o, t in p.imap_unordered(run, [(s, lim, a.results) for s in shapes]):
            np.save(os.path.join(a.results, f"privacy_{shape[0]}x{shape[1]}.npy"), o); print(shape, len(o), "%.0fs" % t, flush=True)
