"""
plots.py -- Step 3: figures and tables from results/<dataset>/queries_<s>.npz   (run: python src/plots.py --dataset gowalla)

  <ds>_level_dist_<s>.png/pdf      returned-level distributions (bars: 1D-Tree / 3-DAG / 5-DAG, lines: their empirical CDFs)
  <ds>_LDD_<c>dag_<s>.png/pdf      empirical LOD (purple bars) and the theoretical LOD of the best-fit uniform baseline (red)
  <ds>_stability_<s>.png/pdf       L2 distance of consecutive LODs against the number of queries
  <ds>_combined_level_diff_whisker.png/pdf   average level overhead, boxes = snapshots at/after stabilisation, dots = theory
  <ds>_combined_exp_ratio_whisker.png/pdf    average accuracy ratio (same layout; dots = E[2^k] of the best-fit baseline,
                                             grey squares = lower bound of Thm. 2(ii))
  <ds>_summary_table.csv           all numbers behind the figures (one row per query length and c)
Baseline: nominal s* = round(s / step), step = timespan/(N-1); s* = nominal +- FIT_RADIUS (step FIT_STEP) minimising the L2
distance between the empirical LOD and the theoretical LOD (Thm. 1).  Accuracy ratio = measured n_Tree / n_cDAG (equals 2^k).
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import theory1d as T

FIT_RADIUS, FIT_STEP = 500, 1
GREEN, BLUE, RED, PURPLE = "#2ca02c", "#1f4fd8", "#d62728", "#8e44ad"
COLORS = {3: ("lightblue", "blue", "darkblue"), 5: ("lightcoral", "red", "darkred")}


def load(results_dir):
    runs = []
    for p in glob.glob(os.path.join(results_dir, "queries_*.npz")):
        z = np.load(p, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        for k in ("s", "N", "M", "init", "batch", "stable_at", "n", "timespan", "n_generated", "n_empty"):
            d[k] = int(z[k])
        d["eps"] = float(z["eps"])
        d["dataset"] = str(z["dataset"])
        d["dl"] = {3: d["lv3"] - d["lvT"], 5: d["lv5"] - d["lvT"]}
        d["sz"] = {3: d["sz3"], 5: d["sz5"]}
        d["ar"] = {c: d["szT"].astype(float) / d["sz"][c].astype(float) for c in (3, 5)}
        d["empty_frac"] = d["n_empty"] / d["n_generated"]
        d["s_nom"] = T.nominal_s(d["s"], d["N"], d["timespan"])
        d["ends"] = np.arange(d["init"], len(d["lvT"]) + 1, d["batch"])
        d["emp_lod"], d["fit"], d["cum"], d["l2c"] = {}, {}, {}, {}
        for c in (3, 5):
            dl = d["dl"][c]
            assert dl.min() >= 0, "negative level difference in the stored results"
            cnt = np.bincount(dl)
            d["emp_lod"][c] = {int(k): float(v) / len(dl) for k, v in enumerate(cnt) if v}
            d["fit"][c] = T.fit_baseline(d["emp_lod"][c], d["N"], c, d["s_nom"], FIT_RADIUS, FIT_STEP)
            K = int(dl.max()) + 1
            ind = np.stack([np.cumsum(dl == k) for k in range(K)], 1)[d["ends"] - 1] / d["ends"][:, None]
            d["l2c"][c] = np.sqrt(((ind[1:] - ind[:-1]) ** 2).sum(1))
            d["cum"][c] = (np.cumsum(dl.astype(float))[d["ends"] - 1] / d["ends"],
                           np.cumsum(d["ar"][c])[d["ends"] - 1] / d["ends"])
        both = np.where((d["l2c"][3] < d["eps"]) & (d["l2c"][5] < d["eps"]))[0]
        rec = int(d["ends"][1:][both[0]]) if len(both) else -1
        assert rec == d["stable_at"], f"stable_at recomputed offline ({rec}) != stored ({d['stable_at']})"
        runs.append(d)
    runs.sort(key=lambda r: r["s"])
    return runs


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def label_s(s):
    return {60: "1 minute", 3600: "1 hour", 86400: "1 day", 604800: "1 week"}.get(s, f"{s} s")


def fig_levels(d, out, ds):
    K = int(max(d["lvT"].max(), d["lv3"].max(), d["lv5"].max()))
    x = np.arange(K + 1)
    P = [np.bincount(d[k], minlength=K + 1) / len(d[k]) for k in ("lvT", "lv3", "lv5")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.27
    for p, off, col, lab in zip(P, (-w, 0, w), (GREEN, BLUE, RED), ("1D-Tree", "3-DAG", "5-DAG")):
        ax.bar(x + off, p, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xlabel("returned level"); ax.set_ylabel("probability")
    ax2 = ax.twinx()
    for p, col, lab in zip(P, (GREEN, BLUE, RED), ("1D-Tree CDF", "3-DAG CDF", "5-DAG CDF")):
        ax2.plot(x, np.cumsum(p), color=col, lw=1.3, label=lab)
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("cumulative probability")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2_, loc="upper left", frameon=False, ncol=2)
    ax.set_title(f"{ds}: query length s = {d['s']} ({label_s(d['s'])}), {len(d['lvT'])} queries")
    fig.tight_layout(); save(fig, out, f"{ds}_level_dist_{d['s']}")


def fig_lod(d, c, out, ds):
    f = d["fit"][c]
    kmax = max(max(d["emp_lod"][c]), len(f["lod_best"]) - 1)
    x = np.arange(kmax + 1)
    emp = np.array([d["emp_lod"][c].get(int(k), 0.0) for k in x])
    th = np.array([f["lod_best"][k] if k < len(f["lod_best"]) else 0.0 for k in x])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x, emp, 0.6, color=PURPLE, alpha=0.85, label="Empirical")
    ax.plot(x, th, color=RED, marker="o", ms=3, lw=1.2, label=f"Theoretical (best s*={f['s_best']}, c={c})")
    ax.set_xticks(x); ax.set_xlabel(f"level difference k = level({c}-DAG) - level(1D-Tree)"); ax.set_ylabel("probability")
    ax.set_title(f"{ds}: s = {d['s']} ({label_s(d['s'])}), {len(d['lvT'])} queries")
    ax.text(0.98, 0.55, f"L2 distance = {f['l2_best']:.3f}", transform=ax.transAxes, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout(); save(fig, out, f"{ds}_LDD_{c}dag_{d['s']}")


def fig_stability(d, out, ds):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for c, col in ((3, BLUE), (5, RED)):
        ax.plot(d["ends"][1:], d["l2c"][c], color=col, lw=1.0, label=f"{c}-DAG")
    ax.axhline(d["eps"], color="k", ls="--", lw=0.9, label=f"epsilon = {d['eps']:g}")
    ax.axvline(d["stable_at"], color="gray", ls=":", label=f"stable at {d['stable_at']}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
    ax.set_xlabel("number of queries"); ax.set_ylabel("L2 distance of consecutive LODs")
    ax.set_title(f"{ds}: s = {d['s']}"); ax.legend(frameon=False)
    fig.tight_layout(); save(fig, out, f"{ds}_stability_{d['s']}")


def whisker(runs, out, ds, name, ylabel, which, dot, bound=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = np.arange(len(runs)); width = 0.35
    for c, shift in ((3, -width / 2), (5, width / 2)):
        face, edge, med = COLORS[c]
        data = [d["cum"][c][which][d["ends"] >= d["stable_at"]] for d in runs]
        ax.boxplot(data, positions=pos + shift, widths=width, patch_artist=True, whis=[0, 100],
                   boxprops=dict(facecolor=face, color=edge, linewidth=2), whiskerprops=dict(color=edge, linewidth=2),
                   capprops=dict(color=edge, linewidth=2), medianprops=dict(color=med, linewidth=2.5))
        for i, d in enumerate(runs):
            ax.plot(i + shift, dot(d, c), "o", color=edge, ms=10,
                    label=f"{c}-DAG vs. 1D-Tree theoretical" if i == 0 else None)
            if bound:
                ax.plot(i + shift, bound(d, c), "s", color="#555", ms=6, label="lower bound of Thm. 2(ii)" if (i == 0 and c == 3) else None)
    ax.set_xticks(pos); ax.set_xticklabels([str(d["s"]) for d in runs])
    ax.set_xlabel("Query Lengths", fontsize=14); ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f"{ds}: 3-DAG vs 5-DAG ({ylabel})", fontsize=15)
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); save(fig, out, name)


def at_window_edge(d, c):
    """True if the best-fit s* is an end point of the search window (the fit may want a value outside it)."""
    s_best, s_nom = d["fit"][c]["s_best"], d["s_nom"]
    cands = [s_nom + o for o in range(-FIT_RADIUS, FIT_RADIUS + 1, FIT_STEP) if T.is_valid(d["N"], s_nom + o, c)]
    return s_best in (min(cands), max(cands))


def table(runs, out, ds):
    rows = []
    for d in runs:
        for c in (3, 5):
            f = d["fit"][c]
            e_k, e_ar = T.expectations(f["lod_best"])
            kap = T.kappa(d["N"], f["s_best"])
            rows.append({
                "dataset": ds, "s": d["s"], "c": c, "N": d["N"], "queries": len(d["lvT"]), "stable_at": d["stable_at"],
                "empty_fraction": round(d["empty_frac"], 4),
                "mean_level_Tree": round(float(d["lvT"].mean()), 3), "mean_level_cDAG": round(float(d[f"lv{c}"].mean()), 3),
                "mean_level_diff": round(float(d["dl"][c].mean()), 4), "theory_E_level_diff": round(e_k, 4),
                "bound_level_diff": round(T.bound_level_overhead(c), 4),
                "mean_AR_measured": round(float(d["ar"][c].mean()), 4), "mean_2^k": round(float((2.0 ** d["dl"][c]).mean()), 4),
                "theory_E_AR": round(e_ar, 4), "AR_lower_bound_Thm2ii": round(T.ar_lower_bound(d["N"], f["s_best"]), 4),
                "nominal_s*": f["s_nom"], "best_s*": f["s_best"], "kappa_best": kap,
                "L2_best": round(f["l2_best"], 5), "L2_nominal": round(f["l2_nominal"], 5),
                "best_s*_at_window_edge": at_window_edge(d, c),
                "mean_nodes_visited_Tree": round(float(d["visT"].mean()), 2),
                "mean_nodes_visited_cDAG": round(float(d[f"vis{c}"].mean()), 2),
                "mean_containment_tests_cDAG": round(float(d[f"tst{c}"].mean()), 2),
                "max_containment_tests_cDAG": int(d[f"tst{c}"].max()),
                "frac_queries_with_tied_nodes": round(float((d[f"nb{c}"] > 1).mean()), 4)})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{ds}_summary_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"s={r['s']:>6} c={r['c']} | n={r['queries']} stable_at={r['stable_at']} empty={r['empty_fraction']:.3f} | levels T {r['mean_level_Tree']:.2f} "
              f"DAG {r['mean_level_cDAG']:.2f} | level_diff emp {r['mean_level_diff']:.3f} theory {r['theory_E_level_diff']:.3f} (< {r['bound_level_diff']:.2f}) | "
              f"AR emp {r['mean_AR_measured']:.2f} theory {r['theory_E_AR']:.2f} lb {r['AR_lower_bound_Thm2ii']:.2f} | "
              f"best s*={r['best_s*']} (nominal {r['nominal_s*']}) L2 {r['L2_best']:.3f}" + ("  [best s* at the edge of the search window]" if r["best_s*_at_window_edge"] else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gowalla", "nyc"], required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--fit-radius", type=int, default=500, help="best-fit s* is searched in nominal +- this radius")
    ap.add_argument("--fit-step", type=int, default=1)
    a = ap.parse_args()
    FIT_RADIUS, FIT_STEP = a.fit_radius, a.fit_step
    ds = a.dataset
    runs = load(os.path.join(a.results, ds))
    assert len(runs) == 4, f"expected 4 result files in {a.results}/{ds}, found {len(runs)}"
    out = os.path.join(a.out, ds)
    for d in runs:
        fig_levels(d, out, ds); fig_stability(d, out, ds)
        for c in (3, 5):
            fig_lod(d, c, out, ds)
    whisker(runs, out, ds, f"{ds}_combined_level_diff_whisker", "Expected Level Difference", 0,
            lambda d, c: T.expectations(d["fit"][c]["lod_best"])[0])
    whisker(runs, out, ds, f"{ds}_combined_exp_ratio_whisker", "Expected FP-Competitive Ratio", 1,
            lambda d, c: T.expectations(d["fit"][c]["lod_best"])[1],
            bound=lambda d, c: T.ar_lower_bound(d["N"], d["fit"][c]["s_best"]))
    table(runs, out, ds)
    print("figures and table written to", out)
