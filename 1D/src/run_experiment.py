"""
run_experiment.py -- Step 2: query streams on the 1-D element sets (1D-Tree vs 3-DAG vs 5-DAG).

For every query length s (seconds) a stream of queries Q = [x, x+s), x i.i.d. uniform on I_s = [0, M-s] (real valued),
is answered by the 1D-Tree, the 3-DAG and the 5-DAG (exhaustive SRC-search, deepest containing node, ties -> smallest n_v).
Protocol (Sec. 8 of the paper): an initial batch, then batches of BATCH queries; after each batch the L2 distance between the
Level Overhead Distributions (LODs, cumulative since the first query) of the last two snapshots is computed for both
c-DAGs; once it is below EPS for BOTH c-DAGs at the same snapshot the stream is declared stable, and EXTRA more queries are
issued (at most MAX_TOTAL per s).  Every query is stored (npz), so all figures and tables are recomputed offline by plots.py.
By default empty queries (|D ∩ Q| = 0) are kept, as in the paper's 1D stream model; --nonempty-only discards them like the 2D code.

Usage
    python src/run_experiment.py --dataset gowalla --jobs 4 --out results
    python src/run_experiment.py --dataset nyc     --jobs 4 --out results
    python src/run_experiment.py --dataset gowalla --smoke        # tiny run (seconds) to test the pipeline
Outputs: results/<dataset>/queries_<s>.npz
"""
import argparse, os, sys, time
import numpy as np
from structures1d import RangeDag1D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
QUERY_LENGTHS = [60, 3600, 86400, 604800]          # 1 minute, 1 hour, 1 day, 1 week (seconds)
DATASET_ID = {"gowalla": 0, "nyc": 1}
SEED = 66
INIT, BATCH, EPS, EXTRA, MAX_TOTAL = 5000, 500, 0.001, 120000, 200000
FIELDS = ["x", "lvT", "szT", "lv3", "sz3", "lv5", "sz5", "nin", "visT", "vis3", "tst3", "vis5", "tst5", "nb3", "nb5"]


def lod_dist(counts):
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()}


def l2(p, q):
    return float(np.sqrt(sum((p.get(k, 0.0) - q.get(k, 0.0)) ** 2 for k in set(p) | set(q))))


def load_times(dataset):
    z = np.load(os.path.join(DATA, f"{dataset}_times.npz"))
    return z["times"], int(z["M"])


def run_stream(args):
    dataset, s_idx, s, init, batch, eps, extra, max_total, nonempty_only, out_dir = args
    times, M = load_times(dataset)
    N = len(times)
    dyadic = N & (N - 1) == 0
    tree = RangeDag1D(times, M, 3)                 # tree_src follows the Tree children of the 3-DAG
    dag3, dag5 = tree, RangeDag1D(times, M, 5)
    rng = np.random.default_rng([SEED, DATASET_ID[dataset], s_idx])
    tag = f"{dataset} s={s}"
    rec = {f: [] for f in FIELDS}
    c3, c5 = {}, {}                                 # level_diff -> number of queries
    n, prev3, prev5, stable_at = 0, None, None, None
    d3 = d5 = float("nan")
    n_gen = n_empty = 0
    t0 = time.time()
    while True:
        m = init if n == 0 else batch
        taken = 0
        while taken < m:
            need = m - taken
            x = rng.uniform(0.0, M - s, need)
            nin = np.searchsorted(times, x + s, "left") - np.searchsorted(times, x, "left")
            n_gen += need
            n_empty += int((nin == 0).sum())
            if nonempty_only:
                keep = nin > 0
                x, nin = x[keep], nin[keep]
            for xi, ni in zip(x.tolist(), nin.tolist()):
                nT, lT, szT, vT = tree.tree_src(xi, s)
                n3, l3, sz3, nb3, v3, t3 = dag3.dag_src(xi, s)
                n5, l5, sz5, nb5, v5, t5 = dag5.dag_src(xi, s)
                if l3 < lT or l5 < lT or szT < sz3 or szT < sz5:
                    raise AssertionError(f"[{tag}] embedding violated at x={xi}: levels {lT},{l3},{l5} sizes {szT},{sz3},{sz5}")
                if dyadic and (szT != sz3 << (l3 - lT) or szT != sz5 << (l5 - lT)):
                    raise AssertionError(f"[{tag}] AR != 2^level_diff at x={xi}")
                for f, v in zip(FIELDS, (xi, lT, szT, l3, sz3, l5, sz5, ni, vT, v3, t3, v5, t5, nb3, nb5)):
                    rec[f].append(v)
                k3, k5 = l3 - lT, l5 - lT
                c3[k3] = c3.get(k3, 0) + 1
                c5[k5] = c5.get(k5, 0) + 1
            taken += len(x)
        n += m
        cur3, cur5 = lod_dist(c3), lod_dist(c5)
        if prev3 is not None:
            d3, d5 = l2(prev3, cur3), l2(prev5, cur5)
            if stable_at is None and d3 < eps and d5 < eps:
                stable_at = n
                print(f"[{tag}] STABLE at n={n} (L2 3-DAG {d3:.6f}, 5-DAG {d5:.6f}); {extra} more follow", flush=True)
        prev3, prev5 = cur3, cur5
        if (n // batch) % 20 == 0 or n == init:
            print(f"[{tag}] n={n:7d}  L2 3-DAG {d3:.5f} 5-DAG {d5:.5f}  {time.time() - t0:6.0f}s", flush=True)
        if stable_at is not None and n >= stable_at + extra:
            break
        if n >= max_total:
            print(f"[{tag}] reached max_total={max_total} (stable_at={stable_at})", flush=True)
            break
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"queries_{s}.npz")
    arr = {f: np.asarray(rec[f]) for f in FIELDS}
    np.savez(path, s=s, N=N, M=M, seed=SEED, dataset=dataset, init=init, batch=batch, eps=eps, extra=extra,
             max_total=max_total, nonempty_only=nonempty_only, stable_at=-1 if stable_at is None else stable_at, n=n,
             n_generated=n_gen, n_empty=n_empty, timespan=int(times[-1]), **arr)
    d3_ = arr["lv3"] - arr["lvT"]; d5_ = arr["lv5"] - arr["lvT"]
    print(f"[{tag}] saved {path}: n={n}, stable_at={stable_at}, empty fraction {n_empty / n_gen:.4f}, "
          f"mean level T/3/5 = {arr['lvT'].mean():.2f}/{arr['lv3'].mean():.2f}/{arr['lv5'].mean():.2f}, "
          f"mean level_diff 3-DAG {d3_.mean():.3f} 5-DAG {d5_.mean():.3f}, time {time.time() - t0:.0f}s", flush=True)
    return tag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_ID), required=True)
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes (one per query length)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    ap.add_argument("--nonempty-only", action="store_true", help="discard queries with |D ∩ Q| = 0 (as the 2D code)")
    ap.add_argument("--only", type=int, nargs="*", help="indices (0..3) of the query lengths to run")
    a = ap.parse_args()
    if a.smoke:
        init, batch, eps, extra, max_total, out = 1000, 500, 0.001, 1000, 4000, os.path.join("results_smoke", a.dataset)
    else:
        init, batch, eps, extra, max_total, out = INIT, BATCH, EPS, EXTRA, MAX_TOTAL, os.path.join(a.out, a.dataset)
    jobs = [(a.dataset, i, s, init, batch, eps, extra, max_total, a.nonempty_only, out)
            for i, s in enumerate(QUERY_LENGTHS) if a.only is None or i in a.only]
    t = time.time()
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
            pool.map(run_stream, jobs)
    else:
        for j in jobs:
            run_stream(j)
    print(f"done in {time.time() - t:.0f}s")
