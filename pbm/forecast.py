
"""
forecast_addon.py — Utilities for prefix→suffix forecasting on well production profiles.

Assumptions:
- You have a long-format DataFrame `panel_long` with columns: well_name, t (0..T-1), r_oil_s, wc, gor.
- The preprocessing aligned profiles so t=0 is production start (already done in your PBM Step 1).
- `out` is a dict from your pipeline with keys: "panel_long", "wells_used", "config" with {"T": int}.

This module provides:
- build_prefix_scaled_channel(): scale r_oil_s by prefix-quantile for leakage-free comparison
- make_matrices(): construct prefix matrix X_pref and full target matrix Y_full
- knn_forecast(): neighbor-based completion with per-neighbor amplitude alignment
- multioutput_forecast(): elastic-net multi-output baseline on compact prefix features
- evaluate_forecasts(): RMSE and sMAPE metrics
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

def build_prefix_scaled_channel(panel_long: pd.DataFrame, wells: list, T:int, T_pref:int,
                                q:float=0.90, eps:float=1e-9, clip_max:float=3.0,
                                rate_col:str="r_oil_s", out_col:str="r_oil_pref_norm") -> pd.DataFrame:
    """Create leakage-free, prefix-scaled oil-rate channel (normalized by prefix q-quantile)."""
    pl = panel_long.copy()
    pl = pl[pl["t"].between(0, T-1)].copy()
    # compute prefix scale per well
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

def make_matrices(panel_long: pd.DataFrame, wells: list, T:int, T_pref:int,
                  channel:str="r_oil_pref_norm", target_col:str="r_oil_s") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X_pref [N, T_pref], Y_suffix [N, T-T_pref], Y_full [N, T] for the given channels."""
    idx = {w:i for i,w in enumerate(wells)}
    N = len(wells)
    X = np.full((N, T_pref), np.nan)
    Yfull = np.full((N, T), np.nan)
    for w, g in panel_long.groupby("well_name", sort=False):
        i = idx.get(w, None)
        if i is None: continue
        gg = g.sort_values("t")
        t = gg["t"].to_numpy()
        if len(t)==0: continue
        # fill prefix channel
        v = gg[channel].to_numpy()
        mask_pref = (t >= 0) & (t < T_pref)
        if mask_pref.any():
            X[i, t[mask_pref]] = v[mask_pref]
        # fill full target
        vy = gg[target_col].to_numpy()
        mask_full = (t >= 0) & (t < T)
        if mask_full.any():
            Yfull[i, t[mask_full]] = vy[mask_full]
    Ysuffix = Yfull[:, T_pref:T]
    return X, Ysuffix, Yfull

def _align_scale(y_ref_pref: np.ndarray, y_nei_pref: np.ndarray, eps:float=1e-9) -> float:
    """Least-squares scale factor to match neighbor prefix to reference prefix."""
    num = np.nansum(y_ref_pref * y_nei_pref)
    den = np.nansum(y_nei_pref ** 2) + eps
    s = num / den
    if not np.isfinite(s):
        s = 1.0
    return float(s)

def knn_forecast(X_pref: np.ndarray, Y_full: np.ndarray, T_pref:int, K:int=15) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """KNN completion with per-neighbor amplitude alignment on prefix. Returns pred_suffix [N, T-T_pref]."""
    from sklearn.neighbors import NearestNeighbors
    N, T_pref_ = X_pref.shape
    assert T_pref_ == T_pref
    # Use only wells with complete full horizon for training & evaluation
    mask_full = np.isfinite(Y_full).sum(axis=1) >= Y_full.shape[1]
    I = np.where(mask_full)[0]
    if len(I) < 3:
        raise ValueError("Not enough wells with full horizon for KNN.")
    nbrs = NearestNeighbors(n_neighbors=min(K+1, len(I)), metric="euclidean").fit(X_pref[I])
    dists, knn_idx = nbrs.kneighbors(X_pref[I])
    T_suffix = Y_full.shape[1] - T_pref
    pred = np.zeros((len(I), T_suffix))
    used = []
    for r, (row_knn, row_d) in enumerate(zip(knn_idx, dists)):
        # exclude self (distance ~ 0)
        neigh = [j for j,d in zip(row_knn, row_d) if d>0][:K]
        if not neigh:
            # fallback: take the single closest even if self
            neigh = [row_knn[0]]
        # gather neighbors (global indices)
        neigh_g = I[neigh]
        # align each neighbor by LS scale over prefix (on raw target y since it's in original units)
        y_ref_pref = Y_full[I[r], :T_pref]
        suffixes = []
        for g in neigh_g:
            y_nei_pref = Y_full[g, :T_pref]
            s = _align_scale(y_ref_pref, y_nei_pref)
            y_nei_suffix = s * Y_full[g, T_pref:]
            suffixes.append(y_nei_suffix)
        suffixes = np.vstack(suffixes)
        pred[r] = np.nanmedian(suffixes, axis=0)
        used.append(neigh_g)
    # Build a full-N predictions array filled with nan, then place known rows
    Ypred_fullN = np.full((N, T_suffix), np.nan)
    Ypred_fullN[I] = pred
    return Ypred_fullN, {"train_indices": I, "neighbors": used}

def _safe_import_build_side_features():
    try:
        # Inherit from the user's notebook if defined
        from build_side_features import build_side_features  # unlikely
        return build_side_features
    except Exception:
        return None

def _fallback_prefix_features(panel_long: pd.DataFrame, wells:list, T_pref:int) -> pd.DataFrame:
    """Compact features from prefix window: mean, std, slope, curvature, wc stats."""
    rows = []
    for w, g in panel_long[panel_long["t"].between(0, T_pref-1)].groupby("well_name", sort=False):
        gg = g.sort_values("t")
        t = gg["t"].to_numpy().astype(float)
        y = gg["r_oil_pref_norm"].to_numpy()
        wc = gg.get("wc", pd.Series([np.nan]*len(gg))).to_numpy()
        # basic stats
        mu, sd = np.nanmean(y), np.nanstd(y)
        # slope via linear fit
        if len(t) >= 2 and np.nanstd(t) > 0:
            A = np.vstack([t, np.ones_like(t)]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            m, b = 0.0, float(y[0]) if len(y) else 0.0
        # curvature (2nd diff)
        if len(y) >= 3:
            curv = np.nanmean(np.diff(y,2))
        else:
            curv = 0.0
        rows.append({
            "well_name": w,
            "y_mean": mu,
            "y_std": sd,
            "y_slope": m,
            "y_intercept": b,
            "y_curv": curv,
            "wc_mean": float(np.nanmean(wc)),
            "wc_std": float(np.nanstd(wc)),
        })
    return pd.DataFrame(rows)

def multioutput_forecast(panel_long: pd.DataFrame, wells:list, T:int, T_pref:int,
                         Y_full: np.ndarray, random_state:int=43) -> Tuple[np.ndarray, Dict]:
    """ElasticNetCV-based multioutput regression on compact prefix features."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.multioutput import MultiOutputRegressor
    # Build compact features
    feats = _fallback_prefix_features(panel_long, wells, T_pref)
    # align feature rows with wells order
    idx = {w:i for i,w in enumerate(wells)}
    X = np.full((len(wells), feats.shape[1]-1), np.nan)
    for _, r in feats.iterrows():
        i = idx[r["well_name"]]
        X[i] = r.drop(labels=["well_name"]).to_numpy(dtype=float)
    # training mask (full horizon)
    mask_full = np.isfinite(Y_full).sum(axis=1) >= T
    I = np.where(mask_full)[0]
    if len(I) < 3:
        raise ValueError("Not enough wells with full horizon for MultiOutput.")
    Y = Y_full[I, T_pref:T]
    model = MultiOutputRegressor(ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, random_state=random_state))
    model.fit(X[I], Y)
    Ypred = np.full((len(wells), T-T_pref), np.nan)
    Ypred[I] = model.predict(X[I])
    return Ypred, {"train_indices": I, "model": model, "features": feats}

def evaluate_forecasts(Y_true, Y_pred):
    """RMSE и sMAPE по строкам, где и факт, и прогноз без NaN; без зависимости от 'squared'."""
    import numpy as np
    mask = np.isfinite(Y_pred).all(axis=1) & np.isfinite(Y_true).all(axis=1)
    n_eval = int(mask.sum())
    if n_eval == 0:
        return {"rmse": float("nan"), "smape": float("nan"), "n_eval": 0}
    diff = Y_pred[mask] - Y_true[mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    smape = float(np.nanmean(2*np.abs(Y_pred[mask]-Y_true[mask])/(np.abs(Y_pred[mask])+np.abs(Y_true[mask])+1e-9)))
    return {"rmse": rmse, "smape": smape, "n_eval": n_eval}
