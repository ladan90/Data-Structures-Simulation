"""
run_experiment.py -- Step 2: query streams on the Gowalla elements (2D-Tree vs 3-2D-DAG).

For every query shape (lat side, lon side, in degrees) a stream of queries
    Q = [x0, x0+s0) x [x1, x1+s1),  x0 ~ U[-90, 90-s0],  x1 ~ U[-180, 180-s1]   (independent)
is processed by both structures (exhaustive SRC-search; DAG ties -> min |D_v|, then deepest).
ONLY NON-EMPTY QUERIES (|D ∩ Q| >= 1) ARE PROCESSED: queries are drawn i.i.d. as above, their exact |D ∩ Q| is counted,
empty ones are discarded (they are only counted, to report the empty fraction) and the stream consists of the
non-empty ones. Protocol (same as the 1D experiment, but counted in non-empty queries): an initial batch, then batches
of 500; after each batch the L2 distance between the Level Overhead Distributions (LODs) of the last two snapshots is
computed; once it is below EPS the stream is declared stable, and EXTRA more (non-empty) queries are issued.
Everything per query is saved (npz), so all figures can be recomputed offline.

Usage
    python run_experiment.py --smoke            # tiny run (a few seconds per shape): check the pipeline
    python run_experiment.py --jobs 4           # full run, one process per shape (default shapes below)
Outputs: results/queries_<s0>x<s1>.npz
"""
import argparse, os, sys, time
import numpy as np
from structures import LazyCDag2D

SHAPES = [(5, 5), (10, 10), (5, 10), (25, 20)]       # (lat side, lon side) in degrees
LAT, LON = (-90.0, 90.0), (-180.0, 180.0)
SEED = 66
INIT, BATCH, EPS, EXTRA, MAX_TOTAL = 5000, 500, 0.001, 120000, 400000
FIELDS = ["x0", "x1", "lvT", "szT", "lvD", "szD", "nmin", "tied", "visT", "visD", "tstD", "nin"]


def lod_dist(counts):
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()}


def l2(p, q):
    return float(np.sqrt(sum((p.get(k, 0.0) - q.get(k, 0.0)) ** 2 for k in set(p) | set(q))))


def make_counter(lat, lon, s0, s1):
    """Exact |D ∩ Q| for a batch of queries (Chebyshev ball query after scaling the axes). Needs scipy."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        sys.exit("scipy is required (pip install scipy)")
    tree = cKDTree(np.column_stack([lat / s0, lon / s1]))
    def count(x0, x1):
        centers = np.column_stack([(x0 + s0 / 2) / s0, (x1 + s1 / 2) / s1])
        return np.asarray(tree.query_ball_point(centers, r=0.5, p=np.inf, return_length=True))
    return count


class NonEmptyStream:
    """Draws i.i.d. uniform queries, discards the empty ones (counting them) and hands out non-empty ones in order."""
    def __init__(self, rng, s0, s1, count, lat, lon, chunk=20000):
        self.rng, self.s0, self.s1, self.count, self.lat, self.lon, self.chunk = rng, s0, s1, count, lat, lon, chunk
        self.buf = (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
        self.n_gen = 0
        self.n_empty = 0
        self.verified = False

    def take(self, m):
        while len(self.buf[0]) < m:
            x0 = self.rng.uniform(LAT[0], LAT[1] - self.s0, self.chunk)
            x1 = self.rng.uniform(LON[0], LON[1] - self.s1, self.chunk)
            nin = self.count(x0, x1)
            if not self.verified:                                        # brute-force check of |D ∩ Q|
                for i in range(60):
                    bf = int(((self.lat >= x0[i]) & (self.lat < x0[i] + self.s0) &
                              (self.lon >= x1[i]) & (self.lon < x1[i] + self.s1)).sum())
                    assert bf == nin[i], f"|D∩Q| mismatch {bf} vs {nin[i]}"
                self.verified = True
                print(f"[{self.s0:g}x{self.s1:g}] |D∩Q| counts verified against brute force", flush=True)
            keep = nin > 0
            self.n_gen += self.chunk
            self.n_empty += int((~keep).sum())
            self.buf = tuple(np.concatenate([b, a[keep]]) for b, a in zip(self.buf, (x0, x1, nin)))
        out = tuple(b[:m] for b in self.buf)
        self.buf = tuple(b[m:] for b in self.buf)
        return out


def run_shape(args):
    shape_idx, shape, init, batch, eps, extra, max_total, out_dir = args
    s0, s1 = shape
    d = np.load("gowalla_D.npz")
    lat, lon = d["lat"], d["lon"]
    S = LazyCDag2D(lat, lon, (LAT, LON), c=3)
    rng = np.random.default_rng([SEED, shape_idx])
    stream = NonEmptyStream(rng, s0, s1, make_counter(lat, lon, s0, s1), lat, lon)
    rec = {f: [] for f in FIELDS}
    counts = {}                      # level_diff -> number of (non-empty) queries
    n, prev, stable_at, dist = 0, None, None, float("nan")
    t0 = time.time()
    tag = f"{s0}x{s1}"
    while True:
        m = init if n == 0 else batch
        x0, x1, nin = stream.take(m)
        for i in range(m):
            a = S.tree_src(x0[i], x1[i], s0, s1)
            b = S.dag_src(x0[i], x1[i], s0, s1)
            assert a[2] >= b[2], "AR < 1 (embedding violated)"
            for f, v in zip(FIELDS, (x0[i], x1[i], a[1], a[2], b[1], b[2], b[3], b[4], a[3], b[5], b[6], nin[i])):
                rec[f].append(v)
            dd = b[1] - a[1]
            counts[dd] = counts.get(dd, 0) + 1
        n += m
        cur = lod_dist(counts)
        if prev is not None:
            dist = l2(prev, cur)
            if stable_at is None and dist < eps:
                stable_at = n
                print(f"[{tag}] STABLE at n={n} non-empty queries (L2={dist:.6f}); {extra} more follow", flush=True)
        prev = cur
        if (n // batch) % 20 == 0 or n == init:
            print(f"[{tag}] n={n:7d} non-empty (generated {stream.n_gen})  L2(last two LODs)={dist:.5f}  "
                  f"{time.time() - t0:6.0f}s", flush=True)
        if stable_at is not None and n >= stable_at + extra:
            break
        if n >= max_total:
            print(f"[{tag}] reached max_total={max_total} without full protocol (stable_at={stable_at})", flush=True)
            break
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"queries_{s0}x{s1}.npz")
    arr = {f: np.asarray(rec[f]) for f in FIELDS}
    np.savez(path, shape=np.array(shape), seed=SEED, init=init, batch=batch, eps=eps, extra=extra,
             stable_at=-1 if stable_at is None else stable_at, n=n,
             n_generated=stream.n_gen, n_empty=stream.n_empty, **arr)
    dl = arr["lvD"] - arr["lvT"]
    print(f"[{tag}] saved {path}: {n} non-empty queries (empty fraction {stream.n_empty / stream.n_gen:.3f} of {stream.n_gen} drawn), "
          f"stable_at={stable_at}, level_diff<0 in {(dl < 0).sum()} queries, mean level_diff={dl.mean():.3f}, "
          f"mean AR={(arr['szT'] / arr['szD']).mean():.2f}, time {time.time() - t0:.0f}s", flush=True)
    return tag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    ap.add_argument("--jobs", type=int, default=1, help="number of parallel processes (one per shape)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", type=int, nargs="*", help="indices of shapes to run (0..3)")
    a = ap.parse_args()
    if a.smoke:
        init, batch, eps, extra, max_total, out = 1000, 500, 0.001, 1000, 4000, "results_smoke"
    else:
        init, batch, eps, extra, max_total, out = INIT, BATCH, EPS, EXTRA, MAX_TOTAL, a.out
    jobs = [(i, sh, init, batch, eps, extra, max_total, out) for i, sh in enumerate(SHAPES)
            if a.only is None or i in a.only]
    t = time.time()
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(min(a.jobs, len(jobs))) as pool:
            pool.map(run_shape, jobs)
    else:
        for j in jobs:
            run_shape(j)
    print(f"done in {time.time() - t:.0f}s")
