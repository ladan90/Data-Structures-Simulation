"""inspect_results.py -- prints a summary of every results/queries_*.npz file (AR split by empty / non-empty queries)."""
import glob, numpy as np
for path in sorted(glob.glob("results/queries_*.npz")):
    z = np.load(path)
    s0, s1 = z["shape"]
    lvT, lvD = z["lvT"], z["lvD"]
    szT, szD = z["szT"].astype(float), z["szD"].astype(float)
    dl = lvD - lvT; ar = szT / szD; ne = z["nin"] > 0
    print(f"\n=== {path}  shape (lat {s0} x lon {s1} degrees) ===")
    print(f"queries n={int(z['n'])}  stable at n={int(z['stable_at'])}  (init {int(z['init'])}, batch {int(z['batch'])}, eps {float(z['eps'])}, extra {int(z['extra'])}, seed {int(z['seed'])})")
    print(f"empty queries (|D∩Q|=0): {(~ne).mean():.3f}   mean |D∩Q|: {z['nin'].mean():.1f}")
    print(f"mean level  Tree {lvT.mean():.3f}   DAG {lvD.mean():.3f}")
    vals, cnt = np.unique(dl, return_counts=True)
    print("level_diff distribution:", {int(v): round(float(c) / len(dl), 4) for v, c in zip(vals, cnt)})
    print(f"mean level_diff: all stored {dl.mean():.3f} | non-empty {dl[ne].mean():.3f}   (negative in {(dl < 0).sum()} queries)")
    if "n_generated" in z.files:
        print(f"drawn queries {int(z['n_generated'])}, of which empty {int(z['n_empty'])} ({int(z['n_empty']) / int(z['n_generated']):.3f})")
    print(f"AR all queries     : mean {ar.mean():.1f}  median {np.median(ar):.2f}  geometric mean {np.exp(np.log(ar).mean()):.3f}  max {ar.max():.0f}")
    print(f"AR non-empty (A_G defined): mean {ar[ne].mean():.1f}  median {np.median(ar[ne]):.2f}  geometric mean {np.exp(np.log(ar[ne]).mean()):.3f}")
    if (~ne).any():
        print(f"AR empty queries   : mean {ar[~ne].mean():.1f}  median {np.median(ar[~ne]):.2f}")
    s = np.sort(ar)[::-1]; k = max(1, len(ar) // 100)
    print(f"top 1% of queries give {s[:k].sum() / ar.sum() * 100:.1f}% of the AR sum")
    print(f"queries with several minimisers: {(z['nmin'] > 1).mean():.3f}   tied nodes with different levels: {z['tied'].mean():.4f}")
    print(f"mean nodes visited: Tree {z['visT'].mean():.1f}   DAG {z['visD'].mean():.1f} (containment tests {z['tstD'].mean():.1f})")
