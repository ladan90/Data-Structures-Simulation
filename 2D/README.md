# Data Structures for Single-Range-Cover Search with Multi-Dimensional Range Queries (2D)

Code, data and results for the 2D experiments of **"Efficient Data Structures for Single-Range-Cover Search with Multi-Dimensional Range Queries"** (Kian and Kowalski).

The experiments compare SRC-search on a 2D-Tree (the two-dimensional KD-Tree) against SRC-search on a 3-2D-DAG (its overlapping augmentation, branching parameter `c = 3`) on the Gowalla check-in dataset, and evaluate:

* the **Returned Level Overhead** `level_diff(Q)` (Thm 3.5(i)),
* the **Accuracy Ratio** `AR(Q)` (Thm 3.5(ii)),
* the fit of the learned **uniform baseline** for the Level Overhead Distribution (LOD, Lemma 3.4 / Corollary 4.1),
* the **entropy of the returned node** under the random maximal rule
  (Section 6, Prop. 6.1).

## Repository layout

2D/
├── src/
│   ├── data_prep.py         Step 1: build D (N = 2^20 elements) from the raw Gowalla check-in file
│   ├── run_experiment.py    Step 2: query streams, exhaustive SRC-search on both structures
│   ├── structures.py        LazyCDag2D: the implicit 2D-Tree and 3-2D-DAG (shared by all steps)
│   ├── theory.py            Exact baseline LOD TQD_3(N*, s*) and the bounds of Theorem 3.5
│   ├── plots.py             Step 3: all paper figures (figA-figD) and summary_table.csv
│   ├── analyze_privacy.py   Step 4: minimizer sets M(Q) and element-set entropy (Section 6)
│   ├── summarize_privacy.py Prints the Table-3 statistics from the privacy_*.npy files
│   ├── inspect_results.py   Quick textual summary of every queries_*.npz
│   ├── tests.py             Correctness tests (brute-force SRC, 16x16 example of Fig. 2)
│   └── tests_theory.py      Cross-checks plots' baseline LOD against the formulas
├── data/
│   └── gowalla_D.npz        Element set D: the first 2^20 distinct (lat, lon) pairs
│                            in check-in-time order (lat, lon float arrays)
├── results/paper/           Per-query records of the four non-empty streams (one .npz per
│                            shape: x0, x1, lvT, szT, lvD, szD, nmin, tied, visited, nin)
│                            + privacy_*.npy (Section 6: |M(Q)|, deepest-level |M|,
│                              distinct element sets, H_set per tied query)
├── figures_paper/           The paper figures (PDF) and summary_table.csv
├── requirements.txt
└── README.md


## Data

`gowalla_D.npz` was built from the public Gowalla check-in file (`Gowalla_totalCheckins.txt`, 6,442,892 records, Feb 2009 - Oct 2010) as in Section 5.1 of the paper: drop 168 invalid rows and the placeholder (0, 0), keep the earliest occurrence of each distinct (lat, lon) pair in check-in-time order, take the first 2^20 pairs. The raw file is available from SNAP: https://snap.stanford.edu/data/loc-gowalla.html
`data_prep.py` regenerates `data/gowalla_D.npz` from it.

## Reproducing the paper results

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

# 0. (optional) correctness tests
python src/tests.py                 # brute-force vs. incremental SRC, Fig.-2 example

# 1. build D (only if you want to regenerate data/gowalla_D.npz from the raw SNAP file)
python src/data_prep.py path/to/Gowalla_totalCheckins.txt

# 2. run the four query streams (the paper streams took ~1-2 h per shape on 4 cores)
python src/run_experiment.py --jobs 4 --out results/paper

# 3. all paper figures + summary_table.csv
python src/plots.py --results results/paper --out figures_paper --paper

# 4. Section 6 statistics (sampled tied queries; the repository already ships the
#    outputs of the paper runs: privacy_*.npy in results/paper/)
python src/analyze_privacy.py --results results/paper --limit 1200
python src/summarize_privacy.py --results results/paper
```

Step 2 writes one `queries_<s0>x<s1>.npz` per shape with every per-query record; steps 3 and 4 read only these files, so all figures and tables can be recomputed offline. A tiny end-to-end check of the pipeline:
`python src/run_experiment.py --smoke`.

## Results shipped in this repository

`results/paper/` contains the exact per-query records of the four streams reported in the paper (138,500 / 136,000 / 129,500 / 133,000 non-empty queries; seed 66), 
so `plots.py` and `analyze_privacy.py` reproduce every number and figure of Sections 5 and 6 without re-running the streams.
`figures_paper/` contains the PDFs used in the paper and
`summary_table.csv` with all derived statistics (mean levels, level overheads, baseline fits, AR statistics, tie fractions, visited-node counts).

The random maximal rule of Section 6 is analyzed by `analyze_privacy.py`:
for every query with several minimizers it records |M(Q)|, the number of minimizers on the deepest level, the number of distinct element sets, and the entropy of the returned element set.


