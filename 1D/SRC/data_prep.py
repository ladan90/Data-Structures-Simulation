"""
data_prep.py -- Step 1: build the 1-D element sets D (distinct pickup / check-in times, in seconds).

  gowalla : check-in times of Gowalla (SNAP, loc-gowalla_totalCheckins.txt[.gz], 6,442,892 rows, Feb 2009 - Oct 2010).
            Times are converted to seconds, shifted so that the earliest check-in is 0, de-duplicated and sorted;
            D = the first N = 2^22 = 4,194,304 distinct times.
  nyc     : pickup times (tpep_pickup_datetime) of the NYC Yellow Taxi trips of January 2024
            (yellow_tripdata_2024-01.parquet).  Garbage timestamps outside January 2024 are removed, times are
            shifted so that the earliest valid pickup is 0, de-duplicated and sorted;
            D = the first N = 2^20 = 1,048,576 distinct times.

The domain is [0, M) with M = max(D) + 1 (the elements span the whole domain, as in the original 1D code).
Output: data/<name>_times.npz with  times (int64, strictly increasing), M, N, timespan = max(D), step = timespan/(N-1).

Usage
    python src/data_prep.py gowalla --input data/loc-gowalla_totalCheckins.txt.gz   (or --download)
    python src/data_prep.py nyc     --input data/yellow_tripdata_2024-01.parquet     (or --download)
    add  --drop-invalid-location  for Gowalla to also drop the 168 rows with invalid / (0,0) coordinates
         (the 2D experiment does this; the paper's 1D experiment does not - see README).
"""
import argparse, os, sys, urllib.request
import numpy as np
import pandas as pd


GOWALLA_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
NYC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
N_GOWALLA, N_NYC = 2 ** 22, 2 ** 20
EPOCH = pd.Timestamp("1970-01-01")
JAN_2024 = (int(pd.Timestamp("2024-01-01").timestamp()), int(pd.Timestamp("2024-02-01").timestamp()))  # [start, end)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def seconds_from_datetime(col):
    """datetime64 column (any unit: ns, us, ms, s) -> int64 seconds since 1970-01-01 (unit- and platform-safe)."""
    col = pd.to_datetime(col)
    return ((col - EPOCH) // pd.Timedelta(seconds=1)).astype("int64").to_numpy()


def finish(sec_all, n_target, name, extra=""):
    """sec_all: raw seconds (with repetitions) already filtered.  Shift to 0, de-duplicate, keep the first n_target."""
    sec_all = np.asarray(sec_all, dtype=np.int64)
    t0 = int(sec_all.min())
    distinct = np.unique(sec_all - t0)                       # sorted, distinct
    if len(distinct) < n_target:
        sys.exit(f"only {len(distinct)} distinct times available, need {n_target}")
    times = distinct[:n_target]
    N, timespan = len(times), int(times[-1])
    M = timespan + 1
    step = timespan / (N - 1)
    assert np.all(times[1:] > times[:-1]) and times[0] == 0
    path = os.path.join(DATA, f"{name}_times.npz")
    os.makedirs(DATA, exist_ok=True)
    np.savez(path, times=times, M=M, N=N, timespan=timespan, step=step, distinct_total=len(distinct),
             rows_used=len(sec_all), t0_epoch_seconds=t0)
    print(f"rows used                    : {len(sec_all):,}{extra}")
    print(f"distinct times (all)         : {len(distinct):,}")
    print(f"N = |D| (first distinct)     : {N:,}   (2^{int(np.log2(N))})")
    print(f"earliest time (epoch s / UTC): {t0} / {pd.to_datetime(t0, unit='s')}")
    print(f"timespan = max(D)            : {timespan:,} s  ({timespan / 86400:.2f} days);  domain [0, M), M = {M:,}")
    print(f"step = timespan/(N-1)        : {step:.4f} s per element")
    print("nominal s* = round(s/step)   : " + ", ".join(f"s={s}: {round(s / step)}" for s in (60, 3600, 86400, 604800)))
    print(f"saved {os.path.abspath(path)}")


def prep_gowalla(path, drop_invalid_location=False):
    cols = [1, 2, 3] if drop_invalid_location else [1]
    df = pd.read_csv(path, sep="\t", header=None, usecols=cols, compression="infer")
    df.columns = (["time", "lat", "lon"] if drop_invalid_location else ["time"])
    n_raw = len(df)
    df = df.dropna(subset=["time"])
    ts = pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M:%SZ")
    sec = seconds_from_datetime(ts)
    extra = ""
    if drop_invalid_location:
        bad = (df.lat.isna() | df.lon.isna() | (df.lat <= -90) | (df.lat >= 90) | (df.lon < -180) | (df.lon >= 180)
               | ((df.lat == 0) & (df.lon == 0))).to_numpy()
        sec = sec[~bad]
        extra = f"   (raw rows {n_raw:,}; dropped {int(bad.sum())} rows with invalid location)"
    finish(sec, N_GOWALLA, "gowalla" if not drop_invalid_location else "gowalla_clean", extra)


def prep_nyc(path):
    df = pd.read_parquet(path, columns=["tpep_pickup_datetime"])
    n_raw = len(df)
    df = df.dropna(subset=["tpep_pickup_datetime"])
    sec = seconds_from_datetime(df["tpep_pickup_datetime"])
    ok = (sec >= JAN_2024[0]) & (sec < JAN_2024[1])         # remove garbage timestamps (2002, 2008, Feb 2024 ...)
    print(f"raw rows {n_raw:,}; outside January 2024 (removed): {int((~ok).sum()):,}")
    finish(sec[ok], N_NYC, "nyc")


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["gowalla", "nyc"])
    ap.add_argument("--input", help="path of the raw file")
    ap.add_argument("--download", action="store_true", help="download the raw file into data/ first")
    ap.add_argument("--drop-invalid-location", action="store_true")
    a = ap.parse_args()
    default = {"gowalla": ("Gowalla_totalCheckins.txt", GOWALLA_URL),
               "nyc": ("yellow_tripdata_2024-01.parquet", NYC_URL)}[a.dataset]

    path = a.input or os.path.join(DATA, default[0])
    if a.download:
        download(default[1], path)
    if not os.path.exists(path):
        sys.exit(f"{path} not found: use --input PATH or --download")
    prep_gowalla(path, a.drop_invalid_location) if a.dataset == "gowalla" else prep_nyc(path)
