# features.py
"""
Side-features for production profiles.
Focus: compact descriptors per well + robust scaling.
"""

from typing import List, Sequence, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ---------------------------
# Helpers
# ---------------------------
def _nanpolyfit(y: np.ndarray, x: np.ndarray, deg: int = 1) -> Tuple[float, ...]:
    """
    Polyfit with NaN handling.
    Parameters are (y, x, deg) to keep backward-compatibility with older code that called it this way.
    Returns coeffs like numpy.polyfit: highest degree first. For deg=1 returns (a, b) for y ≈ a*x + b.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < deg + 1:
        return tuple([np.nan] * (deg + 1))
    coeffs = np.polyfit(x[mask], y[mask], deg=deg)
    return tuple([float(c) for c in coeffs])


def _first_index_where(y: np.ndarray, threshold: float) -> Optional[int]:
    """
    Returns first index t where y[t] <= threshold. If threshold is nan or never reached, returns None.
    """
    if not np.isfinite(threshold):
        return None
    y = np.asarray(y, float)
    mask = np.isfinite(y) & (y <= threshold)
    idx = np.nonzero(mask)[0]
    return int(idx[0]) if idx.size else None


def _rolling_stat(y: np.ndarray, win: int, fn) -> float:
    """Mean of fn over all rolling windows of length win. Ignores windows with too many NaN."""
    y = np.asarray(y, float)
    if win <= 0 or y.size < win:
        return np.nan
    vals = []
    for i in range(y.size - win + 1):
        seg = y[i : i + win]
        finite = np.isfinite(seg)
        if finite.sum() < max(2, win // 2):
            continue
        vals.append(fn(seg[finite]))
    return float(np.mean(vals)) if vals else np.nan


def _slope_over_window(y: np.ndarray, win: int) -> float:
    """
    Average slope over all rolling windows of length win using linear polyfit.
    """
    y = np.asarray(y, float)
    if win <= 1 or y.size < win:
        return np.nan
    slopes = []
    x_full = np.arange(y.size, dtype=float)
    for i in range(y.size - win + 1):
        seg = y[i : i + win]
        xs = x_full[i : i + win]
        a, b = _nanpolyfit(seg, xs, deg=1)  # returns (a, b) for y ≈ a*x + b
        if np.isfinite(a):
            slopes.append(a)
    return float(np.mean(slopes)) if slopes else np.nan


def _znorm_series(y: np.ndarray) -> np.ndarray:
    """
    Z-normalization along time, ignoring NaN. Zero std -> return original.
    """
    y = np.asarray(y, float)
    mu = np.nanmean(y)
    sigma = np.nanstd(y)
    if not (np.isfinite(sigma) and sigma > 0):
        return y.copy()
    z = (y - mu) / sigma
    return z


# ---------------------------
# Main API
# ---------------------------
def compute_side_features(
    panel_long: pd.DataFrame,
    T: Optional[int] = None,
    *,
    min_valid_ratio: Optional[float] = None,
    trend_znorm: bool = True,
) -> pd.DataFrame:
    """
    Build compact descriptors for each well over horizon [0 .. T-1].
    If T is None: use all available time steps per well.
    Optionally filters rows by valid_ratio >= min_valid_ratio.

    Required columns in panel_long:
      well_name, t, r_oil_s, r_oil_norm, wc, gor, dr_oil_norm
    """
    need = {"well_name", "t", "r_oil_s", "r_oil_norm", "wc", "gor", "dr_oil_norm"}
    missing = need - set(panel_long.columns)
    if missing:
        raise ValueError(f"compute_side_features: missing columns: {missing}")

    feats: List[pd.DataFrame] = []

    for w, g in panel_long.groupby("well_name", sort=False):
        g = g.copy()
        g = g.sort_values("t")
        if T is not None:
            g = g[g["t"].between(0, T - 1)]
        # vectors
        oil = g["r_oil_s"].to_numpy(float)
        oil_norm = g["r_oil_norm"].to_numpy(float)
        wc = g["wc"].to_numpy(float)
        gor = g["gor"].to_numpy(float)
        d_oil = g["dr_oil_norm"].to_numpy(float)

        # Effective horizon length for valid_ratio denominator
        if T is not None:
            denom_T = int(T)
        else:
            # max t + 1 if present else length
            if len(g):
                tmax = g["t"].max()
                denom_T = int(tmax) + 1 if np.isfinite(tmax) else len(oil)
            else:
                denom_T = 0

        # Base peaks
        peak = np.nanmax(oil) if oil.size else np.nan
        t_peak = int(np.nanargmax(oil)) if (oil.size and np.isfinite(peak)) else np.nan

        # Times to reach 50% and 20% of peak
        half = 0.5 * peak if np.isfinite(peak) else np.nan
        fifth = 0.2 * peak if np.isfinite(peak) else np.nan
        t_half = _first_index_where(oil, half)
        t_20 = _first_index_where(oil, fifth)

        # Early and mid aggregates
        m6 = int(min(6, oil_norm.size))
        m12 = int(min(12, oil_norm.size))
        early_mean6 = float(np.nanmean(oil_norm[:m6])) if m6 else np.nan
        early_mean12 = float(np.nanmean(oil_norm[:m12])) if m12 else np.nan

        # Trend features on z-normalized series if trend_znorm=True
        oil_for_trend = _znorm_series(oil_norm) if trend_znorm else oil_norm
        wc_for_trend = _znorm_series(wc) if trend_znorm else wc
        gor_for_trend = _znorm_series(gor) if trend_znorm else gor

        early_trend6 = np.nan
        if m6 >= 3:
            a, b = _nanpolyfit(oil_for_trend[:m6], np.arange(m6, dtype=float), deg=1)
            early_trend6 = float(a) if np.isfinite(a) else np.nan
        mid_trend12 = _slope_over_window(oil_for_trend, 12)

        # Volatility of first derivative
        vol_doil_w6 = _rolling_stat(d_oil, win=6, fn=np.std)
        vol_doil_w12 = _rolling_stat(d_oil, win=12, fn=np.std)

        # Water cut and GOR levels and trends
        wc_mean12 = float(np.nanmean(wc[:m12])) if m12 else np.nan
        wc_trend12 = _slope_over_window(wc_for_trend, 12)
        gor_trend12 = _slope_over_window(gor_for_trend, 12)

        # Plateau length: consecutive months from start with oil_norm >= 0.9
        plateau_len = 0
        for v in oil_norm:
            if not np.isfinite(v) or v < 0.9:
                break
            plateau_len += 1

        # Missingness ratio over the intended horizon
        valid_ratio = float(np.isfinite(oil).sum() / max(1, denom_T))

        row = pd.DataFrame(
            {
                "well_name": [w],
                "peak_oil": [peak],
                "t_peak": [t_peak],
                "t_half": [t_half],
                "t_20": [t_20],
                "early_mean6": [early_mean6],
                "early_mean12": [early_mean12],
                "early_trend6": [early_trend6],
                "mid_trend12": [mid_trend12],
                "vol_doil_w6": [vol_doil_w6],
                "vol_doil_w12": [vol_doil_w12],
                "wc_mean12": [wc_mean12],
                "wc_trend12": [wc_trend12],
                "gor_trend12": [gor_trend12],
                "plateau_len": [plateau_len],
                "valid_ratio": [valid_ratio],
            }
        )
        feats.append(row)

    feats_df = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()

    # Optional filtering by valid_ratio
    if min_valid_ratio is not None and "valid_ratio" in feats_df.columns and not feats_df.empty:
        feats_df = feats_df[feats_df["valid_ratio"] >= float(min_valid_ratio)].reset_index(drop=True)

    return feats_df


def scale_features(
    feats_df: pd.DataFrame,
    exclude: Sequence[str] = ("well_name",),
    *,
    min_valid_ratio: Optional[float] = None,
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Robust scale numeric columns, optionally filtering by valid_ratio first.
    """
    if feats_df is None or feats_df.empty:
        return feats_df, RobustScaler()

    if min_valid_ratio is not None and "valid_ratio" in feats_df.columns:
        feats_df = feats_df[feats_df["valid_ratio"] >= float(min_valid_ratio)].reset_index(drop=True)

    cols = [c for c in feats_df.columns if c not in set(exclude)]
    # keep only numeric
    num_cols = [c for c in cols if np.issubdtype(feats_df[c].dtype, np.number)]
    scaler = RobustScaler()
    X = scaler.fit_transform(feats_df[num_cols].astype(float).values)
    feats_scaled = feats_df.copy()
    for i, c in enumerate(num_cols):
        feats_scaled[c] = X[:, i]
    return feats_scaled, scaler
