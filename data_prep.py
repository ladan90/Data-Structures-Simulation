"""
data_prep.py  --  Step 1 of the EDBT 2D experiments.

Builds the element set D from Gowalla_totalCheckins.txt:
  * clean invalid rows,
  * order check-ins by time (ascending; file order breaks ties),
  * keep the earliest occurrence of every distinct (lat, lon) pair,
  * take the first 2**20 such pairs.
Prints the domain, N, the distinct counts per dimension, the power-of-two N_i,
and the query-shape table. Saves D (in time order) to gowalla_D.npz.

Usage:  python data_prep.py  path/to/Gowalla_totalCheckins.txt
"""
import sys, math
import numpy as np
import pandas as pd

N_TARGET = 2 ** 20
LAT_DOMAIN = (-90.0, 90.0)      # root range along dimension 0 (half-open)
LON_DOMAIN = (-180.0, 180.0)    # root range along dimension 1 (half-open)
# query shapes (side in degrees): (lat side, lon side)
SHAPES = [(5, 5), (10, 10), (5, 10), (25, 20)]


def load_clean(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["user", "time", "lat", "lon", "loc"])
    df["row"] = np.arange(len(df))                      # original file order
    n_raw = len(df)
    df = df.dropna(subset=["time", "lat", "lon"])
    # valid rows: lat in (-90, 90), lon in [-180, 180), and not the (0, 0) placeholder
    bad = ((df.lat <= LAT_DOMAIN[0]) | (df.lat >= LAT_DOMAIN[1]) |
           (df.lon < LON_DOMAIN[0]) | (df.lon >= LON_DOMAIN[1]) |
           ((df.lat == 0) & (df.lon == 0)))
    df = df[~bad].copy()
    df["t"] = pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M:%SZ")
    return df, n_raw, int(bad.sum())


def first_elements(df, n=N_TARGET):
    # ascending time; ties broken by original file order (deterministic)
    df = df.sort_values(["t", "row"], kind="stable")
    first = df.drop_duplicates(subset=["lat", "lon"], keep="first")   # earliest occurrence
    n_distinct_all = len(first)
    D = first.iloc[:n]
    ties_at_cut = int((first.t == D.t.iloc[-1]).sum()) if len(D) else 0
    return D, n_distinct_all, ties_at_cut


def report(D, n_raw, n_bad, n_distinct_all, ties_at_cut):
    lat, lon = D.lat.values, D.lon.values
    N = len(D)
    N0, N1 = len(np.unique(lat)), len(np.unique(lon))
    p0, p1 = 2 ** round(math.log2(N0)), 2 ** round(math.log2(N1))
    M0 = LAT_DOMAIN[1] - LAT_DOMAIN[0]
    M1 = LON_DOMAIN[1] - LON_DOMAIN[0]
    print(f"raw rows                      : {n_raw}")
    print(f"removed rows (invalid/(0,0))  : {n_bad}")
    print(f"distinct (lat,lon), cleaned   : {n_distinct_all}")
    print(f"elements N = |D|              : {N}  (2^20 = {N_TARGET})")
    print(f"check-in time of the first/last element: {D.t.iloc[0]} / {D.t.iloc[-1]}")
    print(f"distinct pairs sharing the last timestamp (tie at the cut): {ties_at_cut}")
    print(f"distinct latitudes  N0 (data) : {N0}   log2 = {math.log2(N0):.4f}")
    print(f"distinct longitudes N1 (data) : {N1}   log2 = {math.log2(N1):.4f}")
    print(f"power-of-two N_0, N_1         : {p0}, {p1}   (n_0, n_1 = {int(math.log2(p0))-1}, {int(math.log2(p1))-1})")
    print(f"element extent  lat [{lat.min():.6f}, {lat.max():.6f}]  lon [{lon.min():.6f}, {lon.max():.6f}]")
    print(f"domain  [{LAT_DOMAIN[0]}, {LAT_DOMAIN[1]}) x [{LON_DOMAIN[0]}, {LON_DOMAIN[1]})   (M_0={M0}, M_1={M1})")
    print("\nquery shapes (lat side x lon side, degrees) and their uniform-baseline parameters:")
    print(f"{'shape':>10} {'s0*=s0*N0/M0':>14} {'s1*=s1*N1/M1':>14} {'kappa0':>7} {'kappa1':>7} {'L*':>4}  shape-assumption 2^k <= N_i/2")
    for s0, s1 in SHAPES:
        a, b = s0 * p0 / M0, s1 * p1 / M1
        k0, k1 = math.floor(math.log2(p0 / a)), math.floor(math.log2(p1 / b))
        ok = (2 ** k0 <= p0 / 2) and (2 ** k1 <= p1 / 2) and a > 1 and b > 1
        print(f"{str((s0, s1)):>10} {a:14.1f} {b:14.1f} {k0:7d} {k1:7d} {min(2*k0, 2*k1+1):4d}  {ok}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "Gowalla_totalCheckins.txt"
    df, n_raw, n_bad = load_clean(path)
    D, n_distinct_all, ties = first_elements(df)
    assert len(D) == N_TARGET, "fewer than 2^20 distinct elements after cleaning"
    assert len(D[["lat", "lon"]].drop_duplicates()) == len(D)      # D is a set
    report(D, n_raw, n_bad, n_distinct_all, ties)
    np.savez("gowalla_D.npz", lat=D.lat.values, lon=D.lon.values)
    print("\nsaved gowalla_D.npz  (arrays 'lat', 'lon', in check-in-time order)")
