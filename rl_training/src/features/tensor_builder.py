"""
tensor_builder.py — Data loading, feature panel creation, resampling,
and state-tensor construction.

Supports building tensors at arbitrary frequencies by resampling the
base 1-minute panel before tensor construction.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import re
import sys
import traceback
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytz
from gdown import download as gdown_download
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def download_file(file_name: str, file_id: str, out_dir: str):
    """Download a file from Google Drive if not already cached."""
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{file_name}.csv")
    if os.path.exists(out_path):
        print(f"Skipping download. File {file_name} already exists in cache.")
        return out_path
    try:
        print(f"Downloading {file_name} -> {out_path}")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown_download(url, out_path, quiet=False, use_cookies=False, verify=True)
        print("Download complete.")
        return out_path
    except Exception as e:
        sys.stdout.write(f"Download failed for {file_name}: {e}\n")
        sys.stdout.flush()
        return None


def load_csv_to_df(
    path: str,
    sep: str = ",",
    timestamp_index_col: str | None = "datetime",
    encoding: str = "utf-8-sig",
    **kw,
) -> pd.DataFrame:
    """Load CSV into a DataFrame with datetime index."""
    print("     Loading dataset (ca. 10 seconds)...")
    head = pd.read_csv(path, sep=sep, encoding=encoding, nrows=0)
    if timestamp_index_col and timestamp_index_col in head.columns:
        kw["parse_dates"] = [timestamp_index_col]
    df = pd.read_csv(path, sep=sep, encoding=encoding, engine="pyarrow", **kw)
    df = df.set_index("datetime")
    print("     Dataset loaded.")
    return df


# ---------------------------------------------------------------------------
# Feature structure identification
# ---------------------------------------------------------------------------

def identify_assets_features_pairs(
    df: pd.DataFrame,
    single_asset_format: str,
    pair_feature_format: str,
) -> tuple[list[str], list[str], list[str], list[tuple[str, str]]]:
    """Identify assets, single-asset features, pair features, and pairs."""

    def _fmt_to_re(fmt: str) -> re.Pattern:
        esc = re.escape(fmt)
        def _repl(m):
            n = m.group(1)
            cc = r"[A-Za-z0-9_]+" if "FEATURE" in n.upper() else r"[A-Za-z0-9]+"
            return f"(?P<{n}>{cc})"
        esc = re.sub(r"\\\{(\w+)\\\}", _repl, esc)
        return re.compile(f"^{esc}$")

    sp = _fmt_to_re(single_asset_format)
    pp = _fmt_to_re(pair_feature_format)
    gp = re.compile(r"^(?P<ASSET>[A-Za-z0-9]+)_(?P<FEATURE>[A-Za-z0-9_]+)$")

    assets: Set[str] = set()
    single_feats: Set[str] = set()
    pair_feats: Set[str] = set()
    pairs: Set[Tuple[str, str]] = set()

    lit = None
    if "{FEATURE}" not in single_asset_format:
        lit = single_asset_format.replace("{ASSET}", "").lstrip("_")

    skip = {"timestamp", "datetime", "date"}
    for col in df.columns:
        if col in skip:
            continue
        m = pp.match(col)
        if m:
            a1, a2, f = m.group("ASSET1"), m.group("ASSET2"), m.group("FEATURE")
            assets.update((a1, a2))
            pairs.add(tuple(sorted((a1, a2))))
            pair_feats.add(f)
            continue
        m = sp.match(col)
        if m:
            assets.add(m.group("ASSET"))
            f = m.groupdict().get("FEATURE") or lit
            if f:
                single_feats.add(f)
            continue
        m = gp.match(col)
        if m:
            assets.add(m.group("ASSET"))
            single_feats.add(m.group("FEATURE"))

    return sorted(assets), sorted(single_feats), sorted(pair_feats), sorted(pairs)


# ---------------------------------------------------------------------------
# Feature panel
# ---------------------------------------------------------------------------

def create_feature_panel(
    df: pd.DataFrame,
    assets: list[str],
    single_asset_features: list[str],
    pair_features: list[str],
    asset_pairs: list[tuple[str, str]],
    pair_feature_format: str,
    timestamp_col: str | None = "datetime",
) -> pd.DataFrame:
    """Assemble a MultiIndex feature panel."""
    if timestamp_col and timestamp_col in df.columns:
        wdf = df.set_index(timestamp_col)
    else:
        wdf = df
        if not isinstance(wdf.index, pd.DatetimeIndex):
            raise ValueError("Need DatetimeIndex or timestamp_col.")

    if wdf.index.tz is None:
        wdf.index = wdf.index.tz_localize("UTC")
    elif wdf.index.tz != pytz.UTC:
        wdf.index = wdf.index.tz_convert("UTC")

    cols, series = [], []
    for asset in assets:
        for feat in single_asset_features:
            cn = f"{asset}_{feat}"
            if cn in wdf.columns:
                series.append(wdf[cn])
                cols.append((asset, feat))

    for a1, a2 in asset_pairs:
        for feat in pair_features:
            cn = pair_feature_format.format(ASSET1=a1, ASSET2=a2, FEATURE=feat)
            if cn in wdf.columns:
                series.append(wdf[cn])
                cols.append((f"{a1}_{a2}", feat))

    if not series:
        raise ValueError("No matching feature columns found.")

    panel = pd.concat(series, axis=1)
    panel.columns = pd.MultiIndex.from_tuples(cols, names=["asset", "feature"])
    return panel


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

_OHLCV_AGGS = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def _normalize_pandas_freq_alias(freq: str) -> str:
    """Normalize shorthand frequency aliases to pandas-safe values."""
    if not isinstance(freq, str):
        return freq
    f = freq.strip().lower()
    # In pandas, "m" is month-end; use "min" for minutes.
    m = re.fullmatch(r"(\d+)m", f)
    if m:
        return f"{m.group(1)}min"
    return freq


def resample_panel(
    panel: pd.DataFrame,
    target_freq: str,
    base_freq: str = "1m",
) -> pd.DataFrame:
    """
    Resample a 1-minute panel to a coarser frequency.

    OHLCV columns get appropriate aggregation (first/max/min/last/sum).
    Everything else is downsampled with 'last'.
    """
    target_freq_norm = _normalize_pandas_freq_alias(target_freq)
    base_freq_norm = _normalize_pandas_freq_alias(base_freq)

    if target_freq_norm == base_freq_norm:
        return panel

    # Determine aggregation per column
    agg_map = {}
    for col in panel.columns:
        feat_name = col[1].lower() if isinstance(col, tuple) else col.lower()
        agg_map[col] = _OHLCV_AGGS.get(feat_name, "last")

    resampled = panel.resample(target_freq_norm).agg(agg_map)
    resampled = resampled.dropna(how="all")
    return resampled


# ---------------------------------------------------------------------------
# Time intervals
# ---------------------------------------------------------------------------

def build_time_intervals(
    df: pd.DataFrame,
    window: str,
    step: Optional[str] = None,
    timestamp_col: str = "datetime",
    include_last_partial: bool = False,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    W = pd.Timedelta(window)
    S = pd.Timedelta(step) if step else W

    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col]).dropna().sort_values()
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index).dropna().sort_values()
    else:
        raise ValueError("No timestamp found.")

    if ts.empty:
        return []

    t_min, t_max = ts.iloc[0], ts.iloc[-1]
    intervals, cur = [], t_min
    while cur < t_max:
        end = cur + W
        if end <= t_max:
            intervals.append((cur, end))
        elif include_last_partial:
            intervals.append((cur, t_max))
            break
        else:
            break
        cur += S
    return intervals


def is_timeframe_valid(
    df: pd.DataFrame,
    pair: tuple[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    feature_name: str,
    pair_feature_format: str,
    timestamp_col: str | None = "datetime",
) -> bool:
    # Construct column identifier
    pair_str = f"{pair[0]}_{pair[1]}"
    
    # For MultiIndex columns, use tuple; otherwise use string
    if isinstance(df.columns, pd.MultiIndex):
        col = (pair_str, feature_name)
    else:
        col = pair_feature_format.format(ASSET1=pair[0], ASSET2=pair[1], FEATURE=feature_name)
    
    # Check if column exists in dataframe
    if col not in df.columns:
        return False
    
    ts = df[timestamp_col] if (timestamp_col and timestamp_col in df.columns) else df.index
    mask = (ts >= start) & (ts < end)
    if mask.sum() == 0:
        return False
    return not df.loc[mask, col].isna().any()


# ---------------------------------------------------------------------------
# State tensor construction
# ---------------------------------------------------------------------------

def build_state_tensor_for_interval(
    panel: pd.DataFrame,
    pair: tuple,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[list]]:
    """Build (X, R, VOL, timestamps) for one interval / pair."""
    asset1, asset2 = pair
    pair_str = f"{asset1}_{asset2}"

    pair_cols = [c for c in panel.columns if c[0] == pair_str]
    a1_cols = [c for c in panel.columns if c[0] == asset1]
    a2_cols = [c for c in panel.columns if c[0] == asset2]

    if not pair_cols:
        return None, None, None, None

    all_cols = pair_cols + a1_cols + a2_cols

    pair_feats = sorted({c[1] for c in pair_cols})
    a1_feats = sorted({c[1] for c in a1_cols})
    a2_feats = sorted({c[1] for c in a2_cols})

    if "spreadNorm" not in pair_feats:
        return None, None, None, None
    if "close" not in a1_feats or "close" not in a2_feats:
        return None, None, None, None

    all_feat_names = pair_feats + [f"{asset1}_{f}" for f in a1_feats] + [f"{asset2}_{f}" for f in a2_feats]
    n_feats = len(all_feat_names)

    # Ensure start and end are timezone-aware UTC Timestamps
    start = pd.Timestamp(start)
    start = start.tz_convert("UTC") if start.tz else start.tz_localize("UTC")
    end = pd.Timestamp(end)
    end = end.tz_convert("UTC") if end.tz else end.tz_localize("UTC")
    
    mask = (panel.index >= start) & (panel.index <= end)
    wdata = panel.loc[mask, all_cols]
    wts = panel.index[mask]

    if len(wdata) < lookback + 1:
        return None, None, None, None

    vals = wdata.values
    n_samples = len(wdata) - lookback

    X = np.zeros((n_samples, 1, n_feats, lookback), dtype=np.float32)
    R = np.zeros((n_samples, 2), dtype=np.float32)
    VOL = np.zeros((n_samples, 1), dtype=np.float32)

    col_idx = {c: i for i, c in enumerate(all_cols)}
    feat_idxs = []
    for f in pair_feats:
        feat_idxs.append(col_idx.get((pair_str, f), -1))
    for f in a1_feats:
        feat_idxs.append(col_idx.get((asset1, f), -1))
    for f in a2_feats:
        feat_idxs.append(col_idx.get((asset2, f), -1))

    spread_idx = col_idx.get((pair_str, "spreadNorm"), -1)
    c1_idx = col_idx.get((asset1, "close"), -1)
    c2_idx = col_idx.get((asset2, "close"), -1)

    for t in range(n_samples):
        for j, fi in enumerate(feat_idxs):
            if fi != -1:
                X[t, 0, j, :] = vals[t : t + lookback, fi]
        if c1_idx != -1 and c2_idx != -1:
            R[t, 0] = vals[t + lookback, c1_idx] - vals[t + lookback - 1, c1_idx]
            R[t, 1] = vals[t + lookback, c2_idx] - vals[t + lookback - 1, c2_idx]
        if spread_idx != -1:
            VOL[t, 0] = abs(vals[t + lookback - 1, spread_idx])

    timestamps = wts[lookback : lookback + n_samples].tolist()
    return X, R, VOL, timestamps


# ---------------------------------------------------------------------------
# Full tensor build + cache
# ---------------------------------------------------------------------------

def build_all_tensors(
    panel: pd.DataFrame,
    valid_intervals_per_pair: dict,
    lookback: int,
    cache_dir: str | None = None,
    config_for_hash: dict | None = None,
    asset_pairs: list | None = None,
):
    """
    Build X_all, R_all, VOL_all from panel and valid intervals.

    Returns (X_all, R_all, VOL_all, metadata_dict).
    """
    ensure_dir(cache_dir or ".")

    all_X, all_R, all_V = [], [], []
    all_timestamps = []
    ticker_set = set()
    all_features = set()
    total_samples = 0

    for pair, ivs in tqdm(valid_intervals_per_pair.items(), desc="Building tensors", unit="pair"):
        ticker_set.update(pair)
        for start, end in ivs:
            X, R, V, ts = build_state_tensor_for_interval(panel, pair, start, end, lookback)
            if X is not None:
                all_X.append(X)
                all_R.append(R)
                all_V.append(V)
                all_timestamps.extend(ts)
                total_samples += len(X)
                ps = f"{pair[0]}_{pair[1]}"
                all_features.update(c[1] for c in panel.columns if c[0] == ps)

    if not all_X:
        raise ValueError("No valid tensor data created.")

    n_feats = all_X[0].shape[2]
    lb = all_X[0].shape[3]

    if cache_dir:
        storage = cache_dir
        ensure_dir(storage)
        xp = os.path.join(storage, "defi_X_mmap.dat")
        rp = os.path.join(storage, "defi_R_mmap.dat")
        vp = os.path.join(storage, "defi_VOL_mmap.dat")

        X_all = np.memmap(xp, dtype="float32", mode="w+", shape=(total_samples, 1, n_feats, lb))
        R_all = np.memmap(rp, dtype="float32", mode="w+", shape=(total_samples, 2))
        V_all = np.memmap(vp, dtype="float32", mode="w+", shape=(total_samples, 1))

        off = 0
        for cx, cr, cv in zip(all_X, all_R, all_V):
            n = len(cx)
            X_all[off : off + n] = cx
            R_all[off : off + n] = cr
            V_all[off : off + n] = cv
            off += n
        X_all.flush()
        R_all.flush()
        V_all.flush()
    else:
        X_all = np.concatenate(all_X, axis=0)
        R_all = np.concatenate(all_R, axis=0)
        V_all = np.concatenate(all_V, axis=0)

    del all_X, all_R, all_V
    gc.collect()

    FEAT_ORDER = sorted(all_features)
    TICKER_ORDER = sorted(ticker_set)
    SAMPLE_TIMESTAMPS = pd.DatetimeIndex(all_timestamps)

    meta = {
        "FEAT_ORDER": FEAT_ORDER,
        "TICKER_ORDER": TICKER_ORDER,
        "SAMPLE_TIMESTAMPS": SAMPLE_TIMESTAMPS,
        "X_shape": X_all.shape,
        "R_shape": R_all.shape,
        "VOL_shape": V_all.shape if hasattr(V_all, "shape") else (total_samples, 1),
        "lookback": lb,
        "n_features": n_feats,
        "created_at": pd.Timestamp.now().isoformat(),
    }

    if cache_dir:
        if config_for_hash:
            cfg_str = json.dumps(config_for_hash, sort_keys=True, default=str)
            meta["config_hash"] = hashlib.md5(cfg_str.encode()).hexdigest()[:12]
            meta["config"] = config_for_hash
        with open(os.path.join(cache_dir, "tensor_metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)

    print(f"Tensors built: X={X_all.shape}, R={R_all.shape}, VOL={V_all.shape if hasattr(V_all,'shape') else 'N/A'}")
    return X_all, R_all, V_all, meta


def load_tensors_from_cache(
    cache_dir: str,
    config_hash: str | None = None,
) -> Optional[tuple]:
    """Load memory-mapped tensors from cache."""
    mp = os.path.join(cache_dir, "tensor_metadata.pkl")
    if not os.path.exists(mp):
        print("No cache metadata found.")
        return None
    try:
        with open(mp, "rb") as f:
            meta = pickle.load(f)
        if config_hash and meta.get("config_hash") != config_hash:
            print(f"Config hash mismatch: cached={meta.get('config_hash')}, current={config_hash}")
            return None
        paths = [
            os.path.join(cache_dir, "defi_X_mmap.dat"),
            os.path.join(cache_dir, "defi_R_mmap.dat"),
            os.path.join(cache_dir, "defi_VOL_mmap.dat"),
        ]
        if not all(os.path.exists(p) for p in paths):
            return None
        X = np.memmap(paths[0], dtype="float32", mode="r", shape=meta["X_shape"])
        R = np.memmap(paths[1], dtype="float32", mode="r", shape=meta["R_shape"])
        V = np.memmap(paths[2], dtype="float32", mode="r", shape=meta["VOL_shape"])
        print(f"Loaded cached tensors: X={X.shape}")
        return X, R, V, meta
    except Exception as e:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def slice_by_mask(
    X: np.ndarray,
    R: np.ndarray,
    VOL: np.ndarray,
    timestamps: pd.DatetimeIndex,
    mask: np.ndarray,
    time_index: pd.DatetimeIndex | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice tensors by a boolean mask on the panel time index."""
    ts = timestamps
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    elif ts.tz != pytz.UTC:
        ts = ts.tz_convert("UTC")

    if time_index is not None:
        mt = time_index[mask]
    else:
        mt = ts[mask] if len(mask) == len(ts) else ts

    if hasattr(mt, "tz") and mt.tz is None:
        mt = mt.tz_localize("UTC")
    elif hasattr(mt, "tz") and mt.tz != pytz.UTC:
        mt = mt.tz_convert("UTC")

    start, end = mt.min(), mt.max()
    sm = (ts >= start) & (ts <= end)
    idx = np.where(sm)[0]
    if len(idx) == 0:
        raise ValueError(f"No data in [{start}, {end}]")
    print(f"Slicing: {len(idx)} samples in [{start}, {end}]")
    return X[idx], R[idx], VOL[idx]
