"""Forecasting utilities for PBM prefix → suffix completion."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ForecastMatrices",
    "ForecastResult",
    "ForecastMetrics",
    "build_prefix_scaled_channel",
    "make_matrices",
    "knn_forecast",
    "multioutput_forecast",
    "evaluate_forecasts",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ForecastMatrices:
    prefix: np.ndarray
    suffix: np.ndarray
    full: np.ndarray

    def full_rows(self) -> np.ndarray:
        """Indices of wells that have observations for the entire horizon."""

        full_length = self.full.shape[1]
        return np.where(np.isfinite(self.full).sum(axis=1) >= full_length)[0]

    @property
    def n_wells(self) -> int:
        return self.full.shape[0]

    @property
    def suffix_horizon(self) -> int:
        return self.suffix.shape[1]


@dataclass(slots=True)
class ForecastResult:
    predictions: np.ndarray
    train_indices: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)

    def copy(self) -> "ForecastResult":
        return ForecastResult(np.array(self.predictions, copy=True), np.array(self.train_indices, copy=True), dict(self.metadata))


@dataclass(slots=True)
class ForecastMetrics:
    rmse: float
    smape: float
    n_eval: int

    def as_dict(self) -> Dict[str, float]:
        return {"rmse": self.rmse, "smape": self.smape, "n_eval": self.n_eval}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_prefix_scaled_channel(
    panel_long: pd.DataFrame,
    wells: Sequence[str],
    T: int,
    T_pref: int,
    q: float = 0.90,
    eps: float = 1e-9,
    clip_max: float = 3.0,
    rate_col: str = "r_oil_s",
    out_col: str = "r_oil_pref_norm",
) -> pd.DataFrame:
    """Create leakage-free, prefix-scaled channel for the target rate."""

    pl = panel_long.copy()
    pl = pl[pl["t"].between(0, T - 1)].copy()
    scales = (
        pl[pl["t"] < T_pref]
        .groupby("well_name")[rate_col]
        .quantile(q)
        .rename("scale")
        .reset_index()
    )
    pl = pl.merge(scales, on="well_name", how="left")
    pl[out_col] = (pl[rate_col] / (pl["scale"].abs() + eps)).clip(lower=0, upper=clip_max)
    return pl.drop(columns=["scale"])


def make_matrices(
    panel_long: pd.DataFrame,
    wells: Sequence[str],
    T: int,
    T_pref: int,
    channel: str = "r_oil_pref_norm",
    target_col: str = "r_oil_s",
) -> ForecastMatrices:
    """Return matrices ``(prefix, suffix, full)`` aligned to the same well order."""

    idx = {w: i for i, w in enumerate(wells)}
    N = len(wells)
    prefix = np.full((N, T_pref), np.nan)
    full = np.full((N, T), np.nan)

    for w, g in panel_long.groupby("well_name", sort=False):
        i = idx.get(w)
        if i is None:
            continue
        gg = g.sort_values("t")
        t = gg["t"].to_numpy()
        if len(t) == 0:
            continue
        values = gg[channel].to_numpy()
        mask_pref = (t >= 0) & (t < T_pref)
        if mask_pref.any():
            prefix[i, t[mask_pref]] = values[mask_pref]
        vy = gg[target_col].to_numpy()
        mask_full = (t >= 0) & (t < T)
        if mask_full.any():
            full[i, t[mask_full]] = vy[mask_full]

    suffix = full[:, T_pref:T]
    return ForecastMatrices(prefix=prefix, suffix=suffix, full=full)


# ---------------------------------------------------------------------------
# Forecasting models
# ---------------------------------------------------------------------------


def knn_forecast(
    matrices: ForecastMatrices,
    T_pref: int,
    K: int = 15,
    metric: str = "euclidean",
) -> ForecastResult:
    """Nearest neighbour completion with per-neighbour amplitude alignment."""

    from sklearn.neighbors import NearestNeighbors

    X_pref = matrices.prefix
    Y_full = matrices.full
    suffix_horizon = matrices.suffix_horizon

    indices = matrices.full_rows()
    if len(indices) < 3:
        raise ValueError("Not enough wells with full horizon for KNN.")

    nbrs = NearestNeighbors(n_neighbors=min(K + 1, len(indices)), metric=metric).fit(X_pref[indices])
    dists, knn_idx = nbrs.kneighbors(X_pref[indices])

    preds = np.full((matrices.n_wells, suffix_horizon), np.nan)
    neighbours_used: List[np.ndarray] = []

    for row_local, (row_knn, row_d) in enumerate(zip(knn_idx, dists)):
        neigh = [j for j, d in zip(row_knn, row_d) if d > 0][:K]
        if not neigh:
            neigh = [row_knn[0]]
        neigh_global = indices[neigh]
        ref_idx = indices[row_local]
        y_ref_pref = Y_full[ref_idx, :T_pref]
        suffixes = []
        for g in neigh_global:
            y_nei_pref = Y_full[g, :T_pref]
            s = _align_scale(y_ref_pref, y_nei_pref)
            suffixes.append(s * Y_full[g, T_pref:])
        suffixes = np.vstack(suffixes)
        preds[ref_idx] = np.nanmedian(suffixes, axis=0)
        neighbours_used.append(neigh_global)

    return ForecastResult(predictions=preds, train_indices=indices, metadata={"neighbors": neighbours_used})


def _align_scale(y_ref_pref: np.ndarray, y_nei_pref: np.ndarray, eps: float = 1e-9) -> float:
    num = np.nansum(y_ref_pref * y_nei_pref)
    den = np.nansum(y_nei_pref ** 2) + eps
    scale = num / den
    if not np.isfinite(scale):
        scale = 1.0
    return float(scale)


def multioutput_forecast(
    matrices: ForecastMatrices,
    panel_long: pd.DataFrame,
    wells: Sequence[str],
    T: int,
    T_pref: int,
    random_state: int = 43,
) -> ForecastResult:
    """ElasticNet multi-output baseline on compact prefix features."""

    from sklearn.linear_model import ElasticNetCV
    from sklearn.multioutput import MultiOutputRegressor

    feats = _fallback_prefix_features(panel_long, wells, T_pref)
    idx = {w: i for i, w in enumerate(wells)}
    X = np.full((len(wells), feats.shape[1] - 1), np.nan)
    for _, r in feats.iterrows():
        i = idx[r["well_name"]]
        X[i] = r.drop(labels=["well_name"]).to_numpy(dtype=float)

    indices = matrices.full_rows()
    if len(indices) < 3:
        raise ValueError("Not enough wells with full horizon for MultiOutput.")

    Y = matrices.full[indices, T_pref:T]
    model = MultiOutputRegressor(ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=random_state))
    model.fit(X[indices], Y)

    preds = np.full((len(wells), matrices.suffix_horizon), np.nan)
    preds[indices] = model.predict(X[indices])
    return ForecastResult(predictions=preds, train_indices=indices, metadata={"model": model, "features": feats})


def _fallback_prefix_features(panel_long: pd.DataFrame, wells: Sequence[str], T_pref: int) -> pd.DataFrame:
    rows = []
    for w, g in panel_long[panel_long["t"].between(0, T_pref - 1)].groupby("well_name", sort=False):
        gg = g.sort_values("t")
        t = gg["t"].to_numpy().astype(float)
        y = gg.get("r_oil_pref_norm", gg.get("r_oil_s")).to_numpy()
        wc = gg.get("wc", pd.Series([np.nan] * len(gg))).to_numpy()
        mu, sd = np.nanmean(y), np.nanstd(y)
        if len(t) >= 2 and np.nanstd(t) > 0:
            A = np.vstack([t, np.ones_like(t)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            slope = 0.0
            intercept = float(y[0]) if len(y) else 0.0
        curv = np.nanmean(np.diff(y, 2)) if len(y) >= 3 else 0.0
        rows.append(
            {
                "well_name": w,
                "y_mean": float(mu),
                "y_std": float(sd),
                "y_slope": float(slope),
                "y_intercept": float(intercept),
                "y_curv": float(curv),
                "wc_mean": float(np.nanmean(wc)),
                "wc_std": float(np.nanstd(wc)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def evaluate_forecasts(Y_true: np.ndarray, Y_pred: np.ndarray) -> ForecastMetrics:
    """RMSE и sMAPE по строкам, где и факт, и прогноз без NaN."""

    mask = np.isfinite(Y_pred).all(axis=1) & np.isfinite(Y_true).all(axis=1)
    n_eval = int(mask.sum())
    if n_eval == 0:
        return ForecastMetrics(float("nan"), float("nan"), 0)
    diff = Y_pred[mask] - Y_true[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    smape = float(
        np.nanmean(2 * np.abs(Y_pred[mask] - Y_true[mask]) / (np.abs(Y_pred[mask]) + np.abs(Y_true[mask]) + 1e-9))
    )
    return ForecastMetrics(rmse, smape, n_eval)
