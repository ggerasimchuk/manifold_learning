# -*- coding: utf-8 -*-
"""Production Behavior Manifold — Step 2: compact side features."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def _nanpolyfit(y: np.ndarray, x: Optional[np.ndarray] = None, deg: int = 1) -> Tuple[float, ...]:
    """Linear regression over data with NaNs. Returns coefficients."""
    y = np.asarray(y, float)
    if x is None:
        x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < deg + 1:
        return tuple([np.nan] * (deg + 1))
    coeffs = np.polyfit(x[mask], y[mask], deg=deg)
    return tuple(coeffs)


def _first_index_where(y: np.ndarray, cond) -> Optional[int]:
    idx = np.nonzero(cond(y))[0]
    return int(idx[0]) if idx.size else None


def _rolling_stat(y: np.ndarray, win: int, fn) -> float:
    """Aggregate statistic of all rolling windows of size win."""
    y = np.asarray(y, float)
    if win <= 0 or len(y) < win:
        return np.nan
    vals = []
    for i in range(len(y) - win + 1):
        seg = y[i:i + win]
        if np.isfinite(seg).sum() < max(2, win // 2):
            continue
        vals.append(fn(seg[np.isfinite(seg)]))
    return float(np.mean(vals)) if vals else np.nan


def _slope_over_window(y: np.ndarray, win: int) -> float:
    if win <= 1 or len(y) < win:
        return np.nan
    slopes = []
    for i in range(len(y) - win + 1):
        seg = y[i:i + win]
        if np.isfinite(seg).sum() < max(2, win // 2):
            continue
        a, b = _nanpolyfit(seg, np.arange(win), deg=1)
        slopes.append(a)
    return float(np.mean(slopes)) if slopes else np.nan


def compute_side_features(panel_long: pd.DataFrame, T: int = 36) -> pd.DataFrame:
    """Build per-well descriptors over horizon [0..T)."""
    need = {"well_name", "t", "r_oil_s", "r_oil_norm", "wc", "gor", "dr_oil_norm"}
    missing = need - set(panel_long.columns)
    if missing:
        raise ValueError(f"compute_side_features: missing columns: {missing}")

    feats: List[pd.DataFrame] = []
    for w, g in panel_long.groupby("well_name", sort=False):
        g = g.copy()
        g = g[g["t"].between(0, T - 1)]
        g = g.sort_values("t")
        oil = g["r_oil_s"].to_numpy(float)
        oil_norm = g["r_oil_norm"].to_numpy(float)
        wc = g["wc"].to_numpy(float)
        gor = g["gor"].to_numpy(float)
        d_oil = g["dr_oil_norm"].to_numpy(float)

        peak = np.nanmax(oil) if oil.size else np.nan
        t_peak = int(np.nanargmax(oil)) if (oil.size and np.isfinite(peak)) else np.nan

        half = 0.5 * peak if np.isfinite(peak) else np.nan
        fifth = 0.2 * peak if np.isfinite(peak) else np.nan
        t_half = _first_index_where(oil, lambda y: np.isfinite(half) and (y <= half))
        t_20 = _first_index_where(oil, lambda y: np.isfinite(fifth) and (y <= fifth))

        m6 = int(min(6, len(oil)))
        m12 = int(min(12, len(oil)))
        early_mean6 = float(np.nanmean(oil_norm[:m6])) if m6 else np.nan
        early_mean12 = float(np.nanmean(oil_norm[:m12])) if m12 else np.nan
        early_trend6, _ = _nanpolyfit(oil_norm[:m6], np.arange(m6), 1) if m6 >= 3 else (np.nan, np.nan)
        mid_trend12 = _slope_over_window(oil_norm, 12)

        vol_doil_w6 = _rolling_stat(d_oil, win=6, fn=np.std)
        vol_doil_w12 = _rolling_stat(d_oil, win=12, fn=np.std)

        wc_mean12 = float(np.nanmean(wc[:m12])) if m12 else np.nan
        wc_trend12 = _slope_over_window(wc, 12)
        gor_trend12 = _slope_over_window(gor, 12)

        plateau_len = 0
        for v in oil_norm:
            if not np.isfinite(v) or v < 0.9:
                break
            plateau_len += 1

        valid_ratio = float(np.isfinite(oil).sum() / max(1, T))

        row = pd.DataFrame({
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
        })
        feats.append(row)

    feats_df = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    return feats_df


def scale_features(feats_df: pd.DataFrame, exclude: Sequence[str] = ("well_name",)) -> Tuple[pd.DataFrame, RobustScaler]:
    if feats_df.empty:
        return feats_df, RobustScaler()
    cols = [c for c in feats_df.columns if c not in exclude]
    scaler = RobustScaler()
    X = scaler.fit_transform(feats_df[cols].astype(float).values)
    feats_scaled = feats_df.copy()
    for i, c in enumerate(cols):
        feats_scaled[c] = X[:, i]
    return feats_scaled, scaler
