"""Utilities for prefix→suffix forecasting on well production profiles."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ForecastConfig:
    """Configuration controlling the forecasting workflow."""

    total_horizon: int
    prefix_horizon: int = 20
    prefix_quantile: float = 0.90
    prefix_eps: float = 1e-9
    prefix_clip_max: float = 3.0
    rate_column: str = "r_oil_s"
    normalized_prefix_column: str = "r_oil_pref_norm"
    target_column: str = "r_oil_s"
    knn_neighbors: int = 15
    elasticnet_random_state: int = 43

    @property
    def suffix_horizon(self) -> int:
        return self.total_horizon - self.prefix_horizon


@dataclass
class ForecastResult:
    """Container with matrices, predictions and evaluation results."""

    config: ForecastConfig
    wells: Sequence[str]
    panel_long: pd.DataFrame
    prefix_matrix: np.ndarray
    suffix_true: np.ndarray
    full_target: np.ndarray
    predictions: Dict[str, np.ndarray]
    metrics: Dict[str, Dict[str, float]]
    details: Dict[str, Dict[str, object]]

    def build_full_prediction(self, model_name: str) -> np.ndarray:
        if model_name not in self.predictions:
            raise KeyError(f"Unknown model '{model_name}'")
        full = np.array(self.full_target, copy=True)
        full[:, self.config.prefix_horizon : self.config.total_horizon] = self.predictions[model_name]
        return full

    def full_predictions(self) -> Dict[str, np.ndarray]:
        return {name: self.build_full_prediction(name) for name in self.predictions}


def build_prefix_scaled_channel(
    panel_long: pd.DataFrame,
    wells: Sequence[str],
    cfg: ForecastConfig,
) -> pd.DataFrame:
    """Create leakage-free, prefix-scaled rate channel."""
    pl = panel_long.copy()
    pl = pl[pl["t"].between(0, cfg.total_horizon - 1)].copy()
    scales = (
        pl[pl["t"] < cfg.prefix_horizon]
        .groupby("well_name")[cfg.rate_column]
        .quantile(cfg.prefix_quantile)
        .rename("scale")
        .reset_index()
    )
    pl = pl.merge(scales, on="well_name", how="left")
    denom = pl["scale"].abs() + cfg.prefix_eps
    pl[cfg.normalized_prefix_column] = (
        pl[cfg.rate_column] / denom
    ).clip(lower=0, upper=cfg.prefix_clip_max)
    return pl.drop(columns=["scale"])


def make_matrices(
    panel_long: pd.DataFrame,
    wells: Sequence[str],
    cfg: ForecastConfig,
    *,
    channel: Optional[str] = None,
    target_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``X_pref [N, T_pref]``, ``Y_suffix [N, T-T_pref]`` and ``Y_full [N, T]``."""
    channel = channel or cfg.normalized_prefix_column
    target_col = target_col or cfg.target_column
    well_to_idx = {w: i for i, w in enumerate(wells)}
    n_wells = len(wells)
    X = np.full((n_wells, cfg.prefix_horizon), np.nan)
    Y_full = np.full((n_wells, cfg.total_horizon), np.nan)
    for well, grp in panel_long.groupby("well_name", sort=False):
        i = well_to_idx.get(well)
        if i is None:
            continue
        g = grp.sort_values("t")
        t = g["t"].to_numpy(int)
        if t.size == 0:
            continue
        mask_pref = (t >= 0) & (t < cfg.prefix_horizon)
        if mask_pref.any():
            X[i, t[mask_pref]] = g[channel].to_numpy(float)[mask_pref]
        mask_full = (t >= 0) & (t < cfg.total_horizon)
        if mask_full.any():
            Y_full[i, t[mask_full]] = g[target_col].to_numpy(float)[mask_full]
    Y_suffix = Y_full[:, cfg.prefix_horizon : cfg.total_horizon]
    return X, Y_suffix, Y_full


def _align_scale(y_ref_pref: np.ndarray, y_nei_pref: np.ndarray, eps: float = 1e-9) -> float:
    num = np.nansum(y_ref_pref * y_nei_pref)
    den = np.nansum(y_nei_pref ** 2) + eps
    scale = num / den
    return float(scale) if np.isfinite(scale) else 1.0


def knn_forecast(
    X_pref: np.ndarray,
    Y_full: np.ndarray,
    T_pref: int,
    K: int = 15,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Neighbour-based suffix completion with per-neighbour amplitude alignment."""
    from sklearn.neighbors import NearestNeighbors

    n_wells, pref_in = X_pref.shape
    if pref_in != T_pref:
        raise ValueError("Prefix horizon mismatch")
    mask_full = np.isfinite(Y_full).sum(axis=1) >= Y_full.shape[1]
    train_idx = np.where(mask_full)[0]
    if len(train_idx) < 3:
        raise ValueError("Not enough wells with full horizon for KNN.")
    nn = NearestNeighbors(n_neighbors=min(K + 1, len(train_idx)), metric="euclidean").fit(X_pref[train_idx])
    dists, indices = nn.kneighbors(X_pref[train_idx])
    T_suffix = Y_full.shape[1] - T_pref
    preds = np.full((len(train_idx), T_suffix), np.nan)
    neighbours_used: List[np.ndarray] = []
    for r, (row_knn, row_dist) in enumerate(zip(indices, dists)):
        neigh = [j for j, d in zip(row_knn, row_dist) if d > 0][:K]
        if not neigh:
            neigh = [row_knn[0]]
        global_neigh = train_idx[neigh]
        y_ref_pref = Y_full[train_idx[r], :T_pref]
        suffixes = []
        for g in global_neigh:
            y_nei_pref = Y_full[g, :T_pref]
            scale = _align_scale(y_ref_pref, y_nei_pref)
            suffixes.append(scale * Y_full[g, T_pref:])
        preds[r] = np.nanmedian(np.vstack(suffixes), axis=0)
        neighbours_used.append(global_neigh)
    Y_pred = np.full((n_wells, T_suffix), np.nan)
    Y_pred[train_idx] = preds
    return Y_pred, {"train_indices": train_idx, "neighbors": neighbours_used}


def _fallback_prefix_features(panel_long: pd.DataFrame, wells: Sequence[str], T_pref: int) -> pd.DataFrame:
    """Compact prefix features for linear models."""
    rows: List[Dict[str, float]] = []
    prefix = panel_long[panel_long["t"].between(0, T_pref - 1)]
    for well, grp in prefix.groupby("well_name", sort=False):
        g = grp.sort_values("t")
        t = g["t"].to_numpy(float)
        y = g["r_oil_pref_norm"].to_numpy(float)
        wc = g.get("wc", pd.Series([np.nan] * len(g))).to_numpy(float)
        mu, sigma = np.nanmean(y), np.nanstd(y)
        if len(t) >= 2 and np.nanstd(t) > 0:
            A = np.vstack([t, np.ones_like(t)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            slope = 0.0
            intercept = float(y[0]) if len(y) else 0.0
        curvature = float(np.nanmean(np.diff(y, 2))) if len(y) >= 3 else 0.0
        rows.append(
            {
                "well_name": well,
                "y_mean": float(mu),
                "y_std": float(sigma),
                "y_slope": float(slope),
                "y_intercept": float(intercept),
                "y_curv": curvature,
                "wc_mean": float(np.nanmean(wc)),
                "wc_std": float(np.nanstd(wc)),
            }
        )
    return pd.DataFrame(rows)


def multioutput_forecast(
    panel_long: pd.DataFrame,
    wells: Sequence[str],
    T: int,
    T_pref: int,
    Y_full: np.ndarray,
    *,
    random_state: int = 43,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """ElasticNetCV-based multi-output regression on compact prefix features."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.multioutput import MultiOutputRegressor

    feats = _fallback_prefix_features(panel_long, wells, T_pref)
    if feats.empty:
        raise ValueError("Prefix feature table is empty.")
    well_to_idx = {w: i for i, w in enumerate(wells)}
    X = np.full((len(wells), feats.shape[1] - 1), np.nan)
    for _, row in feats.iterrows():
        X[well_to_idx[row["well_name"]]] = row.drop(labels=["well_name"]).to_numpy(float)
    mask_full = np.isfinite(Y_full).sum(axis=1) >= T
    train_idx = np.where(mask_full)[0]
    if len(train_idx) < 3:
        raise ValueError("Not enough wells with full horizon for MultiOutput.")
    Y = Y_full[train_idx, T_pref:T]
    model = MultiOutputRegressor(
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=random_state)
    )
    model.fit(X[train_idx], Y)
    Y_pred = np.full((len(wells), T - T_pref), np.nan)
    Y_pred[train_idx] = model.predict(X[train_idx])
    return Y_pred, {"train_indices": train_idx, "model": model, "features": feats}


def evaluate_forecasts(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE and sMAPE over wells with complete predictions."""
    mask = np.isfinite(Y_pred).all(axis=1) & np.isfinite(Y_true).all(axis=1)
    n_eval = int(mask.sum())
    if n_eval == 0:
        return {"rmse": float("nan"), "smape": float("nan"), "n_eval": 0}
    diff = Y_pred[mask] - Y_true[mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    smape = float(
        np.nanmean(
            2 * np.abs(Y_pred[mask] - Y_true[mask])
            / (np.abs(Y_pred[mask]) + np.abs(Y_true[mask]) + 1e-9)
        )
    )
    return {"rmse": rmse, "smape": smape, "n_eval": n_eval}


def prepare_forecast_inputs(
    panel_long: pd.DataFrame,
    wells: Sequence[str],
    cfg: ForecastConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build prefix-normalised panel and matrices used by the models."""
    normalized = build_prefix_scaled_channel(panel_long, wells, cfg)
    X_pref, Y_suffix, Y_full = make_matrices(normalized, wells, cfg)
    return normalized, X_pref, Y_suffix, Y_full


def run_forecast_pipeline(
    panel_long: pd.DataFrame,
    wells: Sequence[str],
    cfg: ForecastConfig,
) -> ForecastResult:
    """Run both forecasting baselines and evaluate them."""
    normalized, X_pref, Y_suffix, Y_full = prepare_forecast_inputs(panel_long, wells, cfg)
    predictions: Dict[str, np.ndarray] = {}
    details: Dict[str, Dict[str, object]] = {}

    knn_pred, knn_info = knn_forecast(X_pref, Y_full, cfg.prefix_horizon, K=cfg.knn_neighbors)
    predictions["knn"] = knn_pred
    details["knn"] = knn_info

    enet_pred, enet_info = multioutput_forecast(
        normalized,
        wells,
        cfg.total_horizon,
        cfg.prefix_horizon,
        Y_full,
        random_state=cfg.elasticnet_random_state,
    )
    predictions["elasticnet"] = enet_pred
    details["elasticnet"] = enet_info

    metrics = {name: evaluate_forecasts(Y_suffix, pred) for name, pred in predictions.items()}
    return ForecastResult(
        config=cfg,
        wells=wells,
        panel_long=normalized,
        prefix_matrix=X_pref,
        suffix_true=Y_suffix,
        full_target=Y_full,
        predictions=predictions,
        metrics=metrics,
        details=details,
    )


def save_forecast_artifacts(
    result: ForecastResult,
    output_dir: str,
    *,
    include_full_series: bool = True,
    n_examples: int = 6,
    random_state: int = 59,
) -> Dict[str, object]:
    """Persist numpy arrays, CSV exports and a lightweight HTML report."""
    os.makedirs(output_dir, exist_ok=True)
    cfg = result.config
    paths: Dict[str, object] = {"output_dir": output_dir}

    suffix_true_path = os.path.join(output_dir, "Y_suffix_true.npy")
    np.save(suffix_true_path, result.suffix_true)
    paths["suffix_true_npy"] = suffix_true_path

    suffix_cols = [f"m{t}" for t in range(cfg.prefix_horizon + 1, cfg.total_horizon + 1)]
    pred_suffix_paths: Dict[str, str] = {}
    pred_npy_paths: Dict[str, str] = {}
    for name, pred in result.predictions.items():
        npy_path = os.path.join(output_dir, f"Y_pred_{name}.npy")
        np.save(npy_path, pred)
        pred_npy_paths[name] = npy_path
        df = pd.DataFrame(pred, columns=suffix_cols)
        df.insert(0, "well_name", result.wells)
        csv_path = os.path.join(output_dir, f"pred_{name}.csv")
        df.to_csv(csv_path, index=False)
        pred_suffix_paths[name] = csv_path
    paths["pred_suffix_npy"] = pred_npy_paths
    paths["pred_suffix_csv"] = pred_suffix_paths

    metrics_df = (
        pd.DataFrame(
            [
                {"model": name, **metrics}
                for name, metrics in result.metrics.items()
            ]
        )
        .sort_values("model")
        .reset_index(drop=True)
    )
    metrics_csv = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    paths["metrics_csv"] = metrics_csv

    full_predictions = result.full_predictions() if include_full_series else {}
    full_cols = [f"m{t+1}" for t in range(cfg.total_horizon)]
    pred_full_paths: Dict[str, str] = {}
    for name, full in full_predictions.items():
        df = pd.DataFrame(full, columns=full_cols)
        df.insert(0, "well_name", result.wells)
        csv_path = os.path.join(output_dir, f"pred_full_{name}.csv")
        df.to_csv(csv_path, index=False)
        pred_full_paths[name] = csv_path
    if pred_full_paths:
        paths["pred_full_csv"] = pred_full_paths

    if include_full_series and full_predictions:
        long_records: List[Dict[str, object]] = []
        for i, well in enumerate(result.wells):
            for t in range(cfg.total_horizon):
                row = {
                    "well_name": well,
                    "t": t,
                    "segment": "observed" if t < cfg.prefix_horizon else "forecast",
                    "y_true": float(result.full_target[i, t]) if np.isfinite(result.full_target[i, t]) else np.nan,
                }
                for name, full in full_predictions.items():
                    row[f"y_pred_{name}"] = (
                        float(full[i, t]) if np.isfinite(full[i, t]) else np.nan
                    )
                long_records.append(row)
        long_df = pd.DataFrame(long_records)
        long_csv = os.path.join(output_dir, "pred_long.csv")
        long_df.to_csv(long_csv, index=False)
        paths["pred_long_csv"] = long_csv

    example_plots: List[str] = []
    if include_full_series and full_predictions:
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(random_state)
        reference = next(iter(result.predictions.values()))
        valid_idx = np.where(np.isfinite(reference).all(axis=1))[0]
        if valid_idx.size:
            chosen = rng.choice(valid_idx, size=min(n_examples, valid_idx.size), replace=False)
            time = np.arange(cfg.total_horizon)
            for idx in chosen:
                for name, full in full_predictions.items():
                    fig, ax = plt.subplots()
                    ax.plot(time, result.full_target[idx], label="true")
                    ax.plot(time, full[idx], label="pred")
                    ax.axvline(cfg.prefix_horizon - 1, linestyle="--", color="gray", linewidth=0.8)
                    ax.set_title(f"{name} — {result.wells[idx]}")
                    ax.set_xlabel("month index")
                    ax.set_ylabel(cfg.target_column)
                    ax.legend()
                    path = os.path.join(output_dir, f"example_{name}_{idx}.png")
                    fig.savefig(path, dpi=140, bbox_inches="tight")
                    plt.close(fig)
                    example_plots.append(path)
        paths["example_plots"] = example_plots

    html_path = build_forecast_report(output_dir, metrics_df, example_plots, cfg)
    paths["report_html"] = html_path
    return paths


def build_forecast_report(
    out_dir: str,
    metrics: pd.DataFrame,
    example_plots: Iterable[str],
    cfg: ForecastConfig,
) -> str:
    """Generate a very small HTML report with metrics and embedded plots."""
    html = [
        "<html><head><meta charset='utf-8'><title>Forecast Report</title></head><body>",
        f"<h2>Forecast evaluation (prefix {cfg.prefix_horizon} → total {cfg.total_horizon})</h2>",
        f"<p>Generated: {datetime.utcnow().isoformat()}Z</p>",
        metrics.to_html(index=False, border=1, justify="center"),
        "<h3>Examples</h3>",
    ]
    for path in example_plots:
        html.append(f"<img src='{os.path.basename(path)}' style='max-width:640px;display:block;margin-bottom:10px;' />")
    html.append("</body></html>")
    html_path = os.path.join(out_dir, "forecast_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return html_path

