"""
plots.py -- figures and summary table from results/queries_*.npz  (run: python plots.py [--results results] [--out figures])
ALL QUANTITIES USE NON-EMPTY QUERIES ONLY (|D ∩ Q| >= 1); rows with nin == 0 (old result files) are dropped on loading.

figA_<s0>x<s1>        returned-level distribution: bars = empirical 2D-Tree / 3-2D-DAG, lines = their EMPIRICAL cumulative
                      distribution functions (right axis), as in Fig. 7 of the SOFSEM submission
figB_<s0>x<s1>        empirical Level Overhead Distribution (LOD) and the theoretical LOD of the best-fit uniform baseline
figC_level_overhead   whisker plot of the average level overhead (boxes: snapshots at/after stabilisation; red dot: theory)
figC_accuracy_ratio   whisker plot of the average accuracy ratio (same layout)
figD_<s0>x<s1>        L2 distance between the LODs of consecutive snapshots against the number of queries
summary_table.csv     numbers behind the figures
Baseline: nominal s_i* = round(s_i / (M_i/(N_i-1))) with N_i = 2^20, M = (180, 360) degrees (as s/step in the 1D experiment);
(s0*, s1*) = the pair within +-FIT_RADIUS of the nominal values (step FIT_STEP) minimising the L2 distance between the
empirical LOD and the theoretical LOD (Lemma 3.4). The red dots are the expectations of that theoretical LOD:
E[level_diff] = sum_k k pi_k and E[AR] = sum_k 2^k pi_k.
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import theory as T

N0 = N1 = 2 ** 20
M0, M1 = 180.0, 360.0
FIT_RADIUS, FIT_STEP = 500, 10
SHOW_AR_LOWER_BOUND = True            # grey squares = lower bound of Thm 3.5(ii) in the AR whisker plot
GREEN, BLUE, PURPLE, RED = "#2ca02c", "#1f4fd8", "#8e44ad", "#d62728"
PAPER = False                          # --paper: panel-size figures (3.45 in wide) for a 2x2 layout in a two-column figure*
SIZES = {"A": (7.2, 3.3), "B": (6.4, 3.3), "C": (5.4, 3.3), "D": (5.2, 3.1)}
PAPER_SIZES = {"A": (3.45, 2.55), "B": (3.45, 2.45), "C": (3.45, 2.45), "D": (3.45, 2.3)}


def set_mode(paper):
    global PAPER
    PAPER = paper
    if paper:
        plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                             "xtick.labelsize": 6.5, "ytick.labelsize": 7})
    else:
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                             "xtick.labelsize": 8, "ytick.labelsize": 8})


def fsize(key):
    return (PAPER_SIZES if PAPER else SIZES)[key]


def title_text(d, extra=""):
    if PAPER:
        return f"{d['s0']:g}° × {d['s1']:g}° (latitude × longitude)"
    return f"query side lengths: {sides_label(d)}  (non-empty queries, n = {d['n_total']})"


set_mode(False)


def shape_name(d):
    return f"{d['s0']:g}x{d['s1']:g}"


def sides_label(d):
    return f"{d['s0']:g}° latitude × {d['s1']:g}° longitude"


def load(results):
    runs = []
    for p in sorted(glob.glob(os.path.join(results, "queries_*.npz"))):
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        keep = d["nin"] > 0                                         # NON-EMPTY queries only
        for k in ("x0", "x1", "lvT", "szT", "lvD", "szD", "nmin", "tied", "visT", "visD", "tstD", "nin"):
            d[k] = d[k][keep]
        d["s0"], d["s1"] = [float(v) for v in z["shape"]]
        d["n_total"] = len(d["lvT"])
        d["dl"] = d["lvD"] - d["lvT"]
        d["ar"] = d["szT"].astype(float) / d["szD"].astype(float)
        if "n_generated" in z.files:
            d["empty_frac"] = float(z["n_empty"]) / float(z["n_generated"])
        else:
            d["empty_frac"] = float("nan")
        cnt = np.unique(d["dl"], return_counts=True)
        d["emp_lod"] = {int(k): v / len(d["dl"]) for k, v in zip(*cnt)}
        n0 = T.nominal_s(d["s0"], N0, M0); n1 = T.nominal_s(d["s1"], N1, M1)
        d["fit"] = T.fit_baseline_window(d["emp_lod"], N0, N1, n0, n1, radius=FIT_RADIUS, step=FIT_STEP)
        # snapshots (same quantities as in run_experiment.py): cumulative statistics after init, init+batch, ...
        init, batch = int(d["init"]), int(d["batch"])
        ends = np.arange(init, d["n_total"] + 1, batch)
        ks = np.arange(int(d["dl"].min()), int(d["dl"].max()) + 1)
        cum = np.stack([np.cumsum(d["dl"] == k)[ends - 1] for k in ks], 1) / ends[:, None]
        d["ends"] = ends
        d["l2_consec"] = np.sqrt(((cum[1:] - cum[:-1]) ** 2).sum(1))
        below = np.where(d["l2_consec"] < float(d["eps"]))[0]
        d["stable_at"] = int(ends[1:][below[0]]) if len(below) else int(ends[0])
        d["mean_dl"] = np.cumsum(d["dl"].astype(float))[ends - 1] / ends
        d["mean_ar"] = np.cumsum(d["ar"])[ends - 1] / ends
        runs.append(d)
    runs.sort(key=lambda r: (r["s0"], r["s1"]))
    return runs


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def pmf_from(levels, K):
    return np.bincount(levels, minlength=K + 1)[:K + 1] / len(levels)


def fig_levels(d, out):
    K = int(max(d["lvT"].max(), d["lvD"].max()))
    x = np.arange(K + 1)
    pT, pD = pmf_from(d["lvT"], K), pmf_from(d["lvD"], K)
    fig, ax = plt.subplots(figsize=fsize("A"))
    ax.bar(x - 0.2, pT, 0.38, color=GREEN, label="2D-Tree")
    ax.bar(x + 0.2, pD, 0.38, color=BLUE, label="3-2D-DAG")
    ax.set_xticks(x); ax.set_xlim(-0.8, K + 0.8)
    if PAPER:
        ax.set_xticklabels([str(k) if k % 2 == 0 else "" for k in x])        # label every second level (all ticks kept)
    ax.set_xlabel("returned level"); ax.set_ylabel("probability")
    ax.set_title(title_text(d))
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(pT), color="#0a5c0a", lw=1.2, marker="o", ms=2.5, label="2D-Tree CDF")
    ax2.plot(x, np.cumsum(pD), color="#0a1f80", lw=1.2, marker="s", ms=2.5, label="3-2D-DAG CDF")
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("cumulative probability")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.30 if PAPER else -0.17), frameon=False,
              ncol=2 if PAPER else 4, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(); save(fig, out, f"figA_{shape_name(d)}")


def fig_lod(d, out):
    f = d["fit"]
    kmin, kmax = int(d["dl"].min()), int(max(d["dl"].max(), len(f["lod"]) - 1))
    x = np.arange(kmin, kmax + 1)
    emp = np.array([d["emp_lod"].get(int(k), 0.0) for k in x])
    th = np.array([f["lod"][k] if 0 <= k < len(f["lod"]) else 0.0 for k in x])
    fig, ax = plt.subplots(figsize=fsize("B"))
    ax.bar(x, emp, 0.6, color=PURPLE, alpha=0.85, label="Empirical")
    ax.plot(x, th, color=RED, marker="o", ms=3, lw=1.2, label=f"Theoretical (best s*=({f['s0']}, {f['s1']}), c=3)")
    ax.set_xticks(x); ax.set_xlim(kmin - 0.8, kmax + 0.8)
    ax.set_xlabel("level difference  k = level(3-2D-DAG) − level(2D-Tree)"); ax.set_ylabel("probability")
    ax.set_title(title_text(d))
    ax.text(0.98, 0.55, f"L2 distance = {f['l2']:.3f}", transform=ax.transAxes, ha="right", va="center")
    ax.legend(frameon=False)
    fig.tight_layout(); save(fig, out, f"figB_{shape_name(d)}")


def whisker(runs, out, name, ylabel, getvals, getdot, logy=False, extra=None):
    fig, ax = plt.subplots(figsize=fsize("C"))
    for i, d in enumerate(runs):
        keep = d["ends"] >= d["stable_at"]
        ax.boxplot([getvals(d)[keep]], positions=[i], widths=0.5, whis=[0, 100], patch_artist=True,
                   boxprops=dict(facecolor="#bcd0f7", color=BLUE, linewidth=1.5),
                   whiskerprops=dict(color=BLUE, linewidth=1.5), capprops=dict(color=BLUE, linewidth=1.5),
                   medianprops=dict(color="#0a1f80", linewidth=1.8))
        ax.plot(i, getdot(d), "o", color=RED, ms=7, label="theoretical expected value" if i == 0 else None)
        if extra:
            extra(ax, i, d)
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([f"({d['s0']:g}°, {d['s1']:g}°)" for d in runs])
    ax.set_xlabel("query side lengths (latitude, longitude)"); ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if logy:
        ax.set_yscale("log")
    return fig, ax


def fig_whiskers(runs, out):
    def extra_lo(ax, i, d):
        if i == 0:
            ax.axhline(T.bound_level_overhead(), color="#555", ls=":", lw=1, label="bound of Thm 3.5(i)")
    fig, ax = whisker(runs, out, "figC_level_overhead", "expected level overhead",
                      lambda d: d["mean_dl"], lambda d: T.expectations(d["fit"]["lod"])[0], extra=extra_lo)
    ax.legend(frameon=False, loc="lower left"); fig.tight_layout(); save(fig, out, "figC_level_overhead")

    def extra_ar(ax, i, d):
        if SHOW_AR_LOWER_BOUND:
            ax.plot(i, T.ar_lower_bound(N0, N1, d["fit"]["s0"], d["fit"]["s1"]), "s", color="#555", ms=5,
                    label="lower bound of Thm 3.5(ii)" if i == 0 else None)
    fig, ax = whisker(runs, out, "figC_accuracy_ratio", "expected accuracy ratio",
                      lambda d: d["mean_ar"], lambda d: T.expectations(d["fit"]["lod"])[1], logy=True, extra=extra_ar)
    ax.legend(frameon=False, loc="lower left"); fig.tight_layout(); save(fig, out, "figC_accuracy_ratio")


def fig_stability(d, out):
    fig, ax = plt.subplots(figsize=fsize("D"))
    ax.plot(d["ends"][1:], d["l2_consec"], color=PURPLE, lw=1.0)
    ax.axhline(float(d["eps"]), color="k", ls="--", lw=0.9, label=f"γ = {float(d['eps']):g}")
    ax.axvline(d["stable_at"], color=RED, ls=":", lw=1.0, label=f"stable at {d['stable_at']}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
    ax.set_xlabel("number of (non-empty) queries"); ax.set_ylabel("L2 distance of consecutive LODs")
    ax.set_title(title_text(d))
    ax.legend(frameon=False)
    fig.tight_layout(); save(fig, out, f"figD_{shape_name(d)}")


def table(runs, out):
    rows = []
    for d in runs:
        f = d["fit"]; ar = d["ar"]
        e_ld, e_ar = T.expectations(f["lod"])
        rows.append({
            "side_lengths_deg(lat,lon)": f"({d['s0']:g},{d['s1']:g})", "non_empty_queries": d["n_total"], "stable_at": d["stable_at"],
            "empty_fraction_of_drawn": round(d["empty_frac"], 4),
            "mean_level_Tree": round(float(d["lvT"].mean()), 3), "mean_level_DAG": round(float(d["lvD"].mean()), 3),
            "mean_level_overhead": round(float(d["dl"].mean()), 3), "negative_level_diff_queries": int((d["dl"] < 0).sum()),
            "AR_mean": round(float(ar.mean()), 2), "AR_median": round(float(np.median(ar)), 2),
            "AR_geometric_mean": round(float(np.exp(np.log(ar).mean())), 3),
            "nominal_s0*": f["s0_nom"], "nominal_s1*": f["s1_nom"], "best_s0*": f["s0"], "best_s1*": f["s1"],
            "L2_best": round(f["l2"], 4), "L2_nominal": round(f["nominal_l2"], 4),
            "theory_E_level_overhead": round(e_ld, 3), "theory_E_AR": round(e_ar, 3),
            "bound_level_overhead": round(T.bound_level_overhead(), 3),
            "AR_lower_bound": round(T.ar_lower_bound(N0, N1, f["s0"], f["s1"]), 3),
            "frac_queries_with_ties": round(float((d["nmin"] > 1).mean()), 4),
            "frac_ties_different_levels": round(float(d["tied"].mean()), 4),
            "mean_nodes_visited_Tree": round(float(d["visT"].mean()), 1), "mean_nodes_visited_DAG": round(float(d["visD"].mean()), 1)})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["side_lengths_deg(lat,lon)"], "| n=%d stable_at=%d | nominal s*=(%s,%s) best s*=(%s,%s) L2 best %.3f nominal %.3f | "
              "level overhead: empirical %.3f theory %.3f | AR mean: empirical %.1f theory %.2f"
              % (r["non_empty_queries"], r["stable_at"], r["nominal_s0*"], r["nominal_s1*"], r["best_s0*"], r["best_s1*"],
                 r["L2_best"], r["L2_nominal"], r["mean_level_overhead"], r["theory_E_level_overhead"], r["AR_mean"], r["theory_E_AR"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="figures")
    ap.add_argument("--paper", action="store_true", help="panel-size figures for the paper (3.45 in wide, small fonts)")
    a = ap.parse_args()
    set_mode(a.paper)
    runs = load(a.results)
    assert len(runs) == 4, f"expected 4 result files in {a.results}, found {len(runs)}"
    for d in runs:
        fig_levels(d, a.out); fig_lod(d, a.out); fig_stability(d, a.out)
    fig_whiskers(runs, a.out); table(runs, a.out)
    print("figures and summary_table.csv written to", a.out)
