# Single-Range-Cover Search on DAGs: 1D experiments

Code, data and results for Sec. 8 (Experimental Evaluation) of
L. Kian, D. R. Kowalski, *Single-Range-Cover Search on DAGs: Average-Case Analysis,
Efficient Implementations, Privacy Guarantees* (arXiv).

## Contents
| Path | Description |
|---|---|
| `src/structures1d.py` | 1D-Tree and c-DAG (Sec. 3) and exhaustive SRC-search |
| `src/theory1d.py` | uniform baselines: Lem. 1, Lem. 2, Thm. 1, Thm. 2 (Sec. 4) |
| `src/data_prep.py` | builds the element sets D from the raw datasets |
| `src/run_experiment.py` | query streams, stabilization, per-query records |
| `src/plots.py` | figures and summary tables |
| `src/inspect_results.py` | textual summary of all result files |
| `src/tests.py` | correctness tests of the structures and SRC-search |
| `src/tests_theory.py` | checks `theory1d.py` against brute force |
| `data/` | derived element sets `gowalla_times.npz`, `nyc_times.npz` |
| `results/<dataset>/queries_<s>.npz` | every query of the runs reported in the paper |
| `figures/<dataset>/` | figures and `<dataset>_summary_table.csv` (Tables 1 and 2) |

## Setup
Python 3 (tested with Python 3.13.7).
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the paper
Run all commands from this `1D/` folder.

1. Tests
   ```bash
   python src/tests.py
   python src/tests_theory.py
   ```
2. Data (optional: `data/*.npz` are included)
   ```bash
   python src/data_prep.py gowalla --download
   python src/data_prep.py nyc --download
   ```
   A local copy of the raw file can be used instead: `--input PATH`.
3. Experiments (the flag `--nonempty-only` is required to reproduce the paper)
   ```bash
   python src/run_experiment.py --dataset gowalla --jobs 4 --nonempty-only
   python src/run_experiment.py --dataset nyc     --jobs 4 --nonempty-only
   ```
   For a quick pipeline check, add `--smoke`; the output goes to `results_smoke/`
   and can be plotted with `python src/plots.py --dataset gowalla --results results_smoke`.
4. Figures and tables
   ```bash
   python src/plots.py --dataset gowalla
   python src/plots.py --dataset nyc
   python src/inspect_results.py
   ```

## Datasets
- **Gowalla.** Check-in times of `loc-gowalla_totalCheckins.txt.gz` from SNAP
  (https://snap.stanford.edu/data/loc-gowalla.html), 6,442,892 records
  (February 2009 to October 2010). Times are in seconds from the earliest check-in
  (4 February 2009, 05:17:38 UTC); D is the first N = 2^22 = 4,194,304 of the
  5,561,957 distinct times. The paper uses all rows; `--drop-invalid-location`
  additionally removes rows with invalid or (0,0) coordinates, as in our 2D
  experiments, and is not used in the paper.
- **NYC Yellow Taxi.** Pickup times (`tpep_pickup_datetime`) of
  `yellow_tripdata_2024-01.parquet` from the NYC Taxi and Limousine Commission
  (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), 2,964,624 records.
  The 18 records outside January 2024 are removed. Times are in seconds from
  1 January 2024, 00:00:00; D is the first N = 2^20 = 1,048,576 of the
  1,575,689 distinct times.

## Experimental parameters
| Parameter | Value |
|---|---|
| query lengths s | 60, 3600, 86400, 604800 seconds |
| query start x | uniform on [0, M - s], real-valued, i.i.d. |
| queries kept | non-empty only (D ∩ Q ≠ ∅) |
| structures | 1D-Tree, 3-DAG, 5-DAG (same queries for all three) |
| SRC-search | exhaustive; deepest containing node, ties broken by smallest n_v |
| initial batch / batch size | 5,000 / 500 |
| stabilization | L2 distance between the last two cumulative LODs < 0.001, for both c-DAGs at the same snapshot |
| queries after stabilization | 120,000 |
| maximum queries per length | 200,000 (never reached) |
| random seed | 66 |
| best-fit s* | integer within ±500 of round(s / step), step = max D / (N - 1) |

## Result files
Each `results/<dataset>/queries_<s>.npz` stores the run parameters
(`s, N, M, seed, init, batch, eps, extra, max_total, nonempty_only, stable_at, n,
n_generated, n_empty`) and, for every query:

| Field | Meaning |
|---|---|
| `x` | query start |
| `nin` | number of elements of D in the query |
| `lvT, szT` | returned level and node size n_v, 1D-Tree |
| `lv3, sz3` / `lv5, sz5` | returned level and node size n_v, 3-DAG / 5-DAG |
| `visT` | nodes visited on the 1D-Tree path |
| `vis3, vis5` | containing nodes expanded by the exhaustive search |
| `tst3, tst5` | containment tests performed by the exhaustive search |
| `nb3, nb5` | containing nodes tied with the returned one (same level and same n_v) |

## Citation
If you use this code, please cite the paper above.
