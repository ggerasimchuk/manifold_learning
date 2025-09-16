"""Feature engineering utilities for the PBM project."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

__all__ = [
    "SideFeatureConfig",
    "compute_side_features",
    "scale_features",
    "prepare_scaled_features",
]


# ---------------------------------------------------------------------------
# Helper primitives
# ---------------------------------------------------------------------------

def _nanpolyfit(y: np.ndarray, x: Optional[np.ndarray] = None, deg: int = 1) -> Tuple[float, ...]:
    """Линейная регрессия по ряду с NaN. Возвращает коэффициенты."""

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
    """fn по всем скользящим окнам длины ``win`` (NaN пропускаются)."""

    y = np.asarray(y, float)
    if win <= 0 or len(y) < win:
        return np.nan
    vals: List[float] = []
    for i in range(len(y) - win + 1):
        seg = y[i : i + win]
        if np.isfinite(seg).sum() < max(2, win // 2):
            continue
        vals.append(fn(seg[np.isfinite(seg)]))
    return float(np.mean(vals)) if vals else np.nan


def _slope_over_window(y: np.ndarray, win: int) -> float:
    if win <= 1 or len(y) < win:
        return np.nan
    slopes: List[float] = []
    for i in range(len(y) - win + 1):
        seg = y[i : i + win]
        if np.isfinite(seg).sum() < max(2, win // 2):
            continue
        a, _ = _nanpolyfit(seg, np.arange(win), deg=1)
        slopes.append(a)
    return float(np.mean(slopes)) if slopes else np.nan


def _window_mean(values: np.ndarray, length: int) -> float:
    if length <= 0:
        return float("nan")
    return float(np.nanmean(values[:length]))


def _plateau(oil_norm: np.ndarray, threshold: float) -> int:
    count = 0
    for value in oil_norm:
        if not np.isfinite(value) or value < threshold:
            break
        count += 1
    return count


def _validate_columns(panel_long: pd.DataFrame) -> None:
    need = {"well_name", "t", "r_oil_s", "r_oil_norm", "wc", "gor", "dr_oil_norm"}
    missing = need - set(panel_long.columns)
    if missing:
        raise ValueError(f"compute_side_features: не хватает колонок: {missing}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SideFeatureConfig:
    """Parameters controlling the auxiliary feature engineering step."""

    horizon: int = 36
    early_window: int = 6
    mid_window: int = 12
    plateau_threshold: float = 0.9

    def clamp_early(self, length: int) -> int:
        return int(min(self.early_window, length))

    def clamp_mid(self, length: int) -> int:
        return int(min(self.mid_window, length))


def _single_well_features(g: pd.DataFrame, cfg: SideFeatureConfig) -> Dict[str, float]:
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

    m_early = cfg.clamp_early(len(oil_norm))
    m_mid = cfg.clamp_mid(len(oil_norm))
    early_mean6 = _window_mean(oil_norm, m_early)
    early_mean12 = _window_mean(oil_norm, m_mid)
    early_trend6, _ = _nanpolyfit(oil_norm[:m_early], np.arange(m_early), 1) if m_early >= 3 else (np.nan, np.nan)
    mid_trend12 = _slope_over_window(oil_norm, cfg.mid_window)

    vol_doil_w6 = _rolling_stat(d_oil, win=cfg.early_window, fn=np.std)
    vol_doil_w12 = _rolling_stat(d_oil, win=cfg.mid_window, fn=np.std)

    wc_mean12 = _window_mean(wc, m_mid)
    wc_trend12 = _slope_over_window(wc, cfg.mid_window)
    gor_trend12 = _slope_over_window(gor, cfg.mid_window)

    plateau_len = _plateau(oil_norm, cfg.plateau_threshold)
    valid_ratio = float(np.isfinite(oil).sum() / max(1, cfg.horizon))

    return {
        "peak_oil": float(peak),
        "t_peak": t_peak,
        "t_half": t_half,
        "t_20": t_20,
        "early_mean6": early_mean6,
        "early_mean12": early_mean12,
        "early_trend6": float(early_trend6),
        "mid_trend12": mid_trend12,
        "vol_doil_w6": vol_doil_w6,
        "vol_doil_w12": vol_doil_w12,
        "wc_mean12": wc_mean12,
        "wc_trend12": wc_trend12,
        "gor_trend12": gor_trend12,
        "plateau_len": plateau_len,
        "valid_ratio": valid_ratio,
    }


def compute_side_features(
    panel_long: pd.DataFrame,
    T: Optional[int] = None,
    config: Optional[SideFeatureConfig] = None,
) -> pd.DataFrame:
    """Compute per-well summary features on the ``[0, horizon)`` window."""

    cfg = config or SideFeatureConfig(horizon=T or SideFeatureConfig().horizon)
    if T is not None:
        cfg = SideFeatureConfig(
            horizon=T,
            early_window=cfg.early_window,
            mid_window=cfg.mid_window,
            plateau_threshold=cfg.plateau_threshold,
        )

    _validate_columns(panel_long)

    rows: List[Dict[str, float]] = []
    for well_name, g in panel_long.groupby("well_name", sort=False):
        gg = g[g["t"].between(0, cfg.horizon - 1)].sort_values("t")
        if gg.empty:
            continue
        stats = _single_well_features(gg, cfg)
        stats["well_name"] = str(well_name)
        rows.append(stats)

    return pd.DataFrame(rows)


def scale_features(
    feats_df: pd.DataFrame,
    exclude: Sequence[str] = ("well_name",),
) -> Tuple[pd.DataFrame, RobustScaler]:
    if feats_df.empty:
        return feats_df, RobustScaler()
    cols = [c for c in feats_df.columns if c not in exclude]
    scaler = RobustScaler()
    X = scaler.fit_transform(feats_df[cols].astype(float).values)
    feats_scaled = feats_df.copy()
    for i, c in enumerate(cols):
        feats_scaled[c] = X[:, i]
    return feats_scaled, scaler


def prepare_scaled_features(
    panel_long: pd.DataFrame,
    T: Optional[int] = None,
    config: Optional[SideFeatureConfig] = None,
    exclude: Sequence[str] = ("well_name",),
) -> Tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Convenience wrapper returning raw and scaled feature tables."""

    feats = compute_side_features(panel_long, T=T, config=config)
    feats_scaled, scaler = scale_features(feats, exclude=exclude)
    return feats, feats_scaled, scaler
