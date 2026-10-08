"""summarize_privacy.py -- prints the node- and element-set-level entropy
statistics of Section 6 (Table 3) from the privacy_*.npy files produced by
analyze_privacy.py (run: python src/summarize_privacy.py [--results results/paper])."""
import argparse, os
import numpy as np

# columns of privacy_<s0>x<s1>.npy: [query index, |M(Q)|, deepest-level |M|,
# distinct element sets among minimizers, set-level entropy H_set (bits)]
N_TOTAL = {"5x5": 138500, "5x10": 136000, "10x10": 129500, "25x20": 133000}
P_TIED = {"5x5": 0.2048, "5x10": 0.1725, "10x10": 0.1297, "25x20": 0.0642}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/paper")
    a = ap.parse_args()
    print(f"{'shape':8s} {'P(|M|>1)':>9s} {'P(Hset>0)':>9s} {'mean Hset':>9s}")
    for s, n in N_TOTAL.items():
        p = os.path.join(a.results, f"privacy_{s}.npy")
        if not os.path.exists(p):
            print(f"{s:8s} (privacy_{s}.npy missing, run analyze_privacy.py)"); continue
        o = np.load(p); m, ds, hs = o[:, 1], o[:, 3], o[:, 4]
        p_tied = (np.load(os.path.join(a.results, f"queries_{s}.npz"))["nmin"] > 1).mean()
        print(f"{s:8s} {p_tied*100:8.2f}% {(ds>1).mean()*p_tied*100:8.2f}% {p_tied*hs.mean():9.3f}")
