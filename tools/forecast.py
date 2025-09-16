
"""
forecast_addon.py — Utilities for prefix→suffix forecasting on well production profiles.

Assumptions:
- You have a long-format DataFrame `panel_long` with columns: well_name, t (0..T-1), r_oil_s, wc, gor.
- The preprocessing aligned profiles so t=0 is production start (already done in your PBM Step 1).
- `out` is a dict from your pipeline with keys: "panel_long", "wells_used", "config" with {"T": int}.

This module provides:
- build_prefix_scaled_channel(): scale r_oil_s by prefix-quantile for leakage-free comparison
- make_matrices(): construct prefix matrix X_pref and full target matrix Y_full
- vote_cluster_by_prefix(): assign clusters to prefix-only wells via neighbor voting
- knn_forecast(): neighbor-based completion with per-neighbor amplitude alignment
- multioutput_forecast(): elastic-net multi-output baseline on compact prefix features
- evaluate_forecasts(): RMSE and sMAPE metrics
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Optional, Sequence, Tuple

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

def vote_cluster_by_prefix(
    X_pref: np.ndarray,
    wells: Sequence[str],
    df_map: pd.DataFrame,
    target_indices: Optional[Sequence[int]] = None,
    K_vote: int = 7,
    allow_noise: bool = True,
    cluster_col: str = "cluster",
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, object]]]:
    """Assign clusters to wells using prefix-space kNN voting.

    Parameters
    ----------
    X_pref : np.ndarray
        Prefix matrix of shape [N, T_pref].
    wells : sequence of str
        Well names aligned with rows of ``X_pref``.
    df_map : pd.DataFrame
        Output of ``cluster_hdbscan``/``cluster_gmm_bic`` with columns
        ``well_name`` and ``cluster`` at minimum.
    target_indices : sequence of int, optional
        Indices of wells (rows in ``X_pref``/``wells``) that require cluster
        assignment. If ``None``, all wells absent in ``df_map`` are used.
    K_vote : int, default 7
        Number of nearest neighbours (in prefix space) to use for majority vote.
    allow_noise : bool, default True
        Whether to keep cluster ``-1`` (HDBSCAN noise) in the vote. If False and
        all neighbours are noise, the function falls back to using them anyway.
    cluster_col : str, default ``"cluster"``
        Column in ``df_map`` containing cluster labels.

    Returns
    -------
    assignments : pd.DataFrame
        Columns ``well_name``, ``cluster_vote``, ``vote_conf`` and
        ``n_neighbors``.
    details : dict
        Mapping ``row_index -> {neighbor_indices, neighbor_wells,``
        ``neighbor_clusters, neighbor_distances}`` to aid debugging.
    """

    if cluster_col not in df_map.columns:
        raise ValueError(f"DataFrame df_map must contain column '{cluster_col}'")

    cluster_map = df_map.set_index("well_name")[cluster_col].to_dict()
    all_indices = list(range(len(wells)))
    if target_indices is None:
        target_indices = [i for i in all_indices if wells[i] not in cluster_map]
    else:
        target_indices = [int(i) for i in target_indices]

    if not target_indices:
        return pd.DataFrame(columns=["well_name", "cluster_vote", "vote_conf", "n_neighbors"]), {}

    candidates = [i for i in all_indices if wells[i] in cluster_map]
    if not candidates:
        raise ValueError("No wells with known clusters available for voting.")

    K_eff = max(1, min(int(K_vote), len(candidates)))
    nbrs = NearestNeighbors(metric="euclidean")
    nbrs.fit(X_pref[candidates])

    rows = []
    details: Dict[int, Dict[str, object]] = {}
    for idx in target_indices:
        sample = X_pref[idx].reshape(1, -1)
        dists, inds = nbrs.kneighbors(sample, n_neighbors=K_eff, return_distance=True)
        local_inds = inds[0]
        neigh_global = np.array([candidates[j] for j in local_inds], dtype=int)
        neigh_clusters = np.array([cluster_map.get(wells[g], np.nan) for g in neigh_global], dtype=float)
        neigh_dist = dists[0].astype(float)

        valid_mask = ~np.isnan(neigh_clusters)
        if not valid_mask.any():
            votes_idx = neigh_global[:0]
            votes_clusters = neigh_clusters[:0]
            votes_dist = neigh_dist[:0]
        else:
            neigh_global = neigh_global[valid_mask]
            neigh_clusters = neigh_clusters[valid_mask]
            neigh_dist = neigh_dist[valid_mask]
            if not allow_noise:
                non_noise = neigh_clusters != -1
                if non_noise.any():
                    neigh_global = neigh_global[non_noise]
                    neigh_clusters = neigh_clusters[non_noise]
                    neigh_dist = neigh_dist[non_noise]
            votes_idx = neigh_global[:K_eff]
            votes_clusters = neigh_clusters[:K_eff]
            votes_dist = neigh_dist[:K_eff]

        if votes_clusters.size:
            counts = Counter(int(c) for c in votes_clusters)
            best_cluster = None
            best_count = -1
            for cl, cnt in counts.items():
                if cnt > best_count or (cnt == best_count and (best_cluster is None or cl < best_cluster)):
                    best_cluster = cl
                    best_count = cnt
            total_votes = sum(counts.values())
            vote_conf = float(best_count / total_votes) if total_votes else 0.0
            cluster_vote = float(best_cluster) if best_cluster is not None else np.nan
        else:
            cluster_vote = np.nan
            vote_conf = 0.0

        rows.append({
            "well_name": wells[idx],
            "cluster_vote": cluster_vote,
            "vote_conf": vote_conf,
            "n_neighbors": int(votes_clusters.size),
        })
        details[int(idx)] = {
            "neighbor_indices": votes_idx.tolist(),
            "neighbor_wells": [wells[i] for i in votes_idx],
            "neighbor_clusters": votes_clusters.astype(float).tolist(),
            "neighbor_distances": votes_dist.tolist(),
        }

    assignments = pd.DataFrame(rows)
    return assignments, details

def knn_forecast(
    X_pref: np.ndarray,
    Y_full: np.ndarray,
    T_pref: int,
    K: int = 15,
    target_indices: Optional[Sequence[int]] = None,
    candidate_pools: Optional[Dict[int, Sequence[int]]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """KNN completion with per-neighbour amplitude alignment on prefix.

    Parameters
    ----------
    X_pref : np.ndarray
        Prefix matrix of shape [N, T_pref].
    Y_full : np.ndarray
        Full target matrix [N, T]. Rows may contain NaNs beyond the observed
        prefix (e.g. for forecast wells).
    T_pref : int
        Length of the prefix window.
    K : int, default 15
        Number of neighbours used for suffix aggregation (median).
    target_indices : sequence of int, optional
        Rows to forecast. Default: wells with full horizon (mask_full).
    candidate_pools : dict, optional
        Mapping ``target_index -> iterable of candidate neighbour indices``. If
        not provided the full training set is used for every target.

    Returns
    -------
    Y_pred : np.ndarray
        Array [N, T-T_pref] with NaNs for rows that were not predicted.
    info : dict
        Contains ``train_indices``, ``target_indices`` and ``neighbors``.
    """

    N, T_pref_ = X_pref.shape
    if T_pref_ != T_pref:
        raise ValueError("T_pref does not match X_pref shape")

    T_total = Y_full.shape[1]
    T_suffix = T_total - T_pref

    mask_full = np.isfinite(Y_full).sum(axis=1) >= T_total
    train_idx = np.where(mask_full)[0]
    if len(train_idx) < 1:
        raise ValueError("Not enough wells with full horizon for KNN.")

    if target_indices is None:
        target_indices = train_idx.tolist()
    else:
        target_indices = [int(i) for i in target_indices]

    Ypred_fullN = np.full((N, T_suffix), np.nan)
    neighbors_info: Dict[int, Dict[str, np.ndarray]] = {}

    base_nbrs: Optional[NearestNeighbors] = None
    train_idx_sorted = np.array(train_idx, dtype=int)
    base_nbrs = NearestNeighbors(metric="euclidean")
    base_nbrs.fit(X_pref[train_idx_sorted])

    train_set = set(train_idx_sorted.tolist())

    for idx in target_indices:
        if idx < 0 or idx >= N:
            continue

        pool = train_idx_sorted
        use_base = True
        if candidate_pools is not None and idx in candidate_pools:
            pool_candidates = sorted(set(int(p) for p in candidate_pools[idx]))
            pool_filtered = [p for p in pool_candidates if p in train_set]
            if pool_filtered:
                pool = np.array(pool_filtered, dtype=int)
                use_base = np.array_equal(pool, train_idx_sorted)
            else:
                pool = train_idx_sorted
                use_base = True
        else:
            pool = train_idx_sorted
            use_base = True

        if pool.size == 0:
            continue

        n_neighbors = min(max(K + 1, 1), pool.size)
        sample = X_pref[idx].reshape(1, -1)
        if use_base:
            base_dists, base_inds = base_nbrs.kneighbors(sample, n_neighbors=n_neighbors, return_distance=True)
            neigh_global = train_idx_sorted[base_inds[0]]
            dist_vals = base_dists[0]
            source_pool = train_idx_sorted
            source_inds = base_inds
            source_dists = base_dists
        else:
            local_nbrs = NearestNeighbors(metric="euclidean")
            local_nbrs.fit(X_pref[pool])
            local_dists, local_inds = local_nbrs.kneighbors(sample, n_neighbors=n_neighbors, return_distance=True)
            neigh_global = pool[local_inds[0]]
            dist_vals = local_dists[0]
            source_pool = pool
            source_inds = local_inds
            source_dists = local_dists

        mask_self = neigh_global != idx
        neigh_global = neigh_global[mask_self]
        dist_vals = dist_vals[mask_self]

        if neigh_global.size == 0 and not use_base:
            # fallback to global neighbour set (without cluster restriction)
            base_dists, base_inds = base_nbrs.kneighbors(sample, n_neighbors=n_neighbors, return_distance=True)
            neigh_global = train_idx_sorted[base_inds[0]]
            dist_vals = base_dists[0]
            source_pool = train_idx_sorted
            source_inds = base_inds
            source_dists = base_dists
            mask_self = neigh_global != idx
            neigh_global = neigh_global[mask_self]
            dist_vals = dist_vals[mask_self]

        if neigh_global.size == 0:
            neigh_global = source_pool[source_inds[0][:1]]
            dist_vals = source_dists[0][:1]

        neigh_global = neigh_global[:K]
        dist_vals = dist_vals[:K]
        if neigh_global.size == 0:
            continue

        y_ref_pref = Y_full[idx, :T_pref]
        if not np.isfinite(y_ref_pref).any():
            continue

        suffixes = []
        for g in neigh_global:
            y_nei_pref = Y_full[g, :T_pref]
            s = _align_scale(y_ref_pref, y_nei_pref)
            suffixes.append(s * Y_full[g, T_pref:])

        if not suffixes:
            continue

        suffixes = np.vstack(suffixes)
        Ypred_fullN[idx] = np.nanmedian(suffixes, axis=0)
        neighbors_info[int(idx)] = {
            "neighbors": neigh_global.astype(int),
            "distances": dist_vals.astype(float),
        }

    info = {
        "train_indices": train_idx_sorted,
        "target_indices": np.array(target_indices, dtype=int),
        "neighbors": neighbors_info,
    }
    return Ypred_fullN, info

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