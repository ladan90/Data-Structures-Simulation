"""inspect_results.py -- quick textual summary of every results/<dataset>/queries_*.npz (run: python src/inspect_results.py)."""
import glob, numpy as np
for path in sorted(glob.glob("results/*/queries_*.npz"), key=lambda p: (p.split("/")[1], int(p.split("_")[-1][:-4]))):
    z = np.load(path, allow_pickle=True)
    lvT, l3, l5 = z["lvT"], z["lv3"], z["lv5"]
    d3, d5 = l3 - lvT, l5 - lvT
    print(f"\n=== {path}: dataset {z['dataset']}, s={int(z['s'])}, N={int(z['N'])}, M={int(z['M'])}")
    print(f"queries n={int(z['n'])}  stable at {int(z['stable_at'])}  (init {int(z['init'])}, batch {int(z['batch'])}, eps {float(z['eps'])}, extra {int(z['extra'])}, seed {int(z['seed'])})")
    print(f"empty queries: {int(z['n_empty'])}/{int(z['n_generated'])} = {int(z['n_empty'])/int(z['n_generated']):.4f}   nonempty_only={bool(z['nonempty_only'])}")
    print(f"mean level  Tree {lvT.mean():.3f}  3-DAG {l3.mean():.3f}  5-DAG {l5.mean():.3f}")
    for c, d, sz in ((3, d3, z['sz3']), (5, d5, z['sz5'])):
        v, k = np.unique(d, return_counts=True)
        print(f"  {c}-DAG level_diff pmf {dict(zip(v.tolist(), np.round(k/len(d), 4).tolist()))}  mean {d.mean():.3f}  negative: {(d < 0).sum()}  mean AR {(z['szT']/sz).mean():.3f}")
    print(f"mean nodes visited: Tree {z['visT'].mean():.1f}  3-DAG {z['vis3'].mean():.1f} (tests {z['tst3'].mean():.1f})  5-DAG {z['vis5'].mean():.1f} (tests {z['tst5'].mean():.1f})")
