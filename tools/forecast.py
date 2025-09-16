
"""
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

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .clustering import ClusterConfig, approximate_membership, cluster_hdbscan
from .manifold import ManifoldConfig, embed_umap_euclid, transform_prefix_to_Z
from .matrix_utils import _collect_matrix

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


def _prepare_df_map(df_map: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    df = df_map.reset_index(drop=True).copy()
    if len(df) != n_rows:
        raise ValueError("df_map должно иметь столько же строк, сколько профилей в X_pref")
    if "cluster" not in df.columns:
        raise ValueError("df_map должен содержать колонку 'cluster'")
    if "prob" not in df.columns:
        df["prob"] = 1.0
    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = np.nan
    if "well" not in df.columns and "well_name" in df.columns:
        df["well"] = df["well_name"]
    return df


def _resolve_train_indices(Y_full: np.ndarray, train_indices: Optional[np.ndarray]) -> np.ndarray:
    mask_full = np.isfinite(Y_full).sum(axis=1) >= Y_full.shape[1]
    if train_indices is None:
        donors = np.where(mask_full)[0]
    else:
        donors = np.asarray(train_indices, dtype=int)
        donors = donors[(donors >= 0) & (donors < Y_full.shape[0])]
        donors = donors[mask_full[donors]]
    if donors.size == 0:
        raise ValueError("Not enough wells with full horizon for KNN.")
    return donors


def _apply_anomaly_filter(donors: np.ndarray, df_map: pd.DataFrame, quantile: float) -> Tuple[np.ndarray, float]:
    if donors.size == 0:
        return donors, float("nan")
    scores = df_map.loc[donors, "anomaly_score"].to_numpy(dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        return donors, float("nan")
    threshold = float(np.nanquantile(scores[finite], quantile))
    keep = (~finite) | (scores <= threshold)
    filtered = donors[keep]
    if filtered.size == 0:
        filtered = donors
    return filtered, threshold


def _ordered_donors_by_prefix(X_pref: np.ndarray, donors: np.ndarray, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    target_vec = np.nan_to_num(np.asarray(X_pref[target_idx], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    donor_mat = np.nan_to_num(np.asarray(X_pref[donors], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    dists = np.linalg.norm(donor_mat - target_vec, axis=1)
    order = np.argsort(dists)
    return donors[order], dists[order]


def knn_forecast_cluster_restricted(
    X_pref: np.ndarray,
    Y_full: np.ndarray,
    T_pref: int,
    df_map: pd.DataFrame,
    train_indices: Optional[np.ndarray] = None,
    K: int = 15,
    min_incluster: int = 5,
    mix_global_frac: float = 0.3,
    anomaly_quantile: float = 0.95,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Кластер-aware kNN с ограничением пула доноров и fallback на глобальный поиск."""

    N, T_pref_ = X_pref.shape
    if T_pref_ != T_pref:
        raise ValueError("Несовпадение длины префикса")
    if Y_full.shape[0] != N:
        raise ValueError("Y_full и X_pref должны иметь одинаковое число строк")

    df_loc = _prepare_df_map(df_map, N)
    donors = _resolve_train_indices(Y_full, train_indices)
    donors, anomaly_threshold = _apply_anomaly_filter(donors, df_loc, anomaly_quantile)

    T_suffix = Y_full.shape[1] - T_pref
    Y_pred = np.full((N, T_suffix), np.nan)

    clusters = df_loc["cluster"].to_numpy()
    probs = df_loc["prob"].to_numpy(dtype=float)

    min_incluster = max(1, min(min_incluster, len(donors)))
    debug_rows: List[Optional[Dict[str, object]]] = [None] * N

    for i in range(N):
        if not np.isfinite(Y_full[i, :T_pref]).any():
            continue
        ordered, ordered_dists = _ordered_donors_by_prefix(X_pref, donors, i)
        if ordered.size == 0:
            continue

        target_cluster = clusters[i] if i < len(clusters) else -1
        target_prob = probs[i] if i < len(probs) else 1.0
        in_cluster = []
        out_cluster = []
        for cand, dist in zip(ordered, ordered_dists):
            if cand == i:
                continue
            if np.isfinite(target_cluster) and target_cluster >= 0 and clusters[cand] == target_cluster:
                in_cluster.append((int(cand), float(dist)))
            else:
                out_cluster.append((int(cand), float(dist)))

        need_fallback = (
            len(in_cluster) < min_incluster
            or len(in_cluster) < K
            or not np.isfinite(target_prob)
            or (np.isfinite(target_prob) and target_prob < 0.5)
        )

        neighbors: List[int] = []
        prefix_dists: Dict[int, float] = {}
        scales: List[float] = []

        if in_cluster:
            n_take_cluster = min(len(in_cluster), K if not need_fallback else max(min_incluster, K - int(math.ceil(K * mix_global_frac))))
            for cand, dist in in_cluster[:n_take_cluster]:
                if len(neighbors) >= K:
                    break
                neighbors.append(cand)
                prefix_dists[cand] = dist
        if need_fallback:
            n_global = max(0, K - len(neighbors))
            for cand, dist in out_cluster:
                if cand in neighbors:
                    continue
                neighbors.append(cand)
                prefix_dists[cand] = dist
                if len(neighbors) >= K:
                    break

        if not neighbors:
            # fallback: взять ближайших глобально
            for cand, dist in out_cluster + in_cluster:
                if cand == i:
                    continue
                neighbors.append(cand)
                prefix_dists[cand] = dist
                if len(neighbors) >= K:
                    break

        neighbors = neighbors[:K]
        if not neighbors:
            continue

        y_ref_pref = Y_full[i, :T_pref]
        suffixes = []
        for cand in neighbors:
            y_nei_pref = Y_full[cand, :T_pref]
            s = _align_scale(y_ref_pref, y_nei_pref)
            scales.append(float(s))
            suffixes.append(s * Y_full[cand, T_pref:])
        suffixes = np.vstack(suffixes)
        Y_pred[i] = np.nanmedian(suffixes, axis=0)

        debug_rows[i] = {
            "neighbors": neighbors,
            "neighbors_in_cluster": [c for c in neighbors if np.isfinite(target_cluster) and target_cluster >= 0 and clusters[c] == target_cluster],
            "neighbors_global": [c for c in neighbors if not (np.isfinite(target_cluster) and target_cluster >= 0 and clusters[c] == target_cluster)],
            "fallback_used": bool(need_fallback),
            "target_cluster": target_cluster,
            "target_prob": float(target_prob),
            "prefix_distances": prefix_dists,
            "scales": scales,
            "anomaly_threshold": anomaly_threshold,
        }

    debug_info = {
        "train_indices": donors,
        "selection": debug_rows,
    }
    return Y_pred, debug_info


def knn_forecast_cluster_weighted(
    X_pref: np.ndarray,
    Y_full: np.ndarray,
    T_pref: int,
    Z: np.ndarray,
    df_map: pd.DataFrame,
    K: int = 15,
    alpha_map: float = 2.0,
    w_cluster_match: float = 1.0,
    anomaly_quantile: float = 0.95,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Кластер-aware kNN с весами по карте и совпадению кластера."""

    N, T_pref_ = X_pref.shape
    if T_pref_ != T_pref:
        raise ValueError("Несовпадение длины префикса")
    if Y_full.shape[0] != N:
        raise ValueError("Y_full и X_pref должны иметь одинаковое число строк")
    if Z is None:
        raise ValueError("Для взвешенного варианта необходимо передать координаты Z")
    if len(Z) != N:
        raise ValueError("Z должно иметь столько же строк, сколько X_pref")

    df_loc = _prepare_df_map(df_map, N)
    donors = _resolve_train_indices(Y_full, None)
    donors, anomaly_threshold = _apply_anomaly_filter(donors, df_loc, anomaly_quantile)

    T_suffix = Y_full.shape[1] - T_pref
    Y_pred = np.full((N, T_suffix), np.nan)

    clusters = df_loc["cluster"].to_numpy()
    Z = np.asarray(Z, dtype=float)

    debug_rows: List[Optional[Dict[str, object]]] = [None] * N

    for i in range(N):
        if not np.isfinite(Y_full[i, :T_pref]).any():
            continue
        ordered, ordered_dists = _ordered_donors_by_prefix(X_pref, donors, i)
        if ordered.size == 0:
            continue

        neighbors: List[int] = []
        weights: List[float] = []
        prefix_dists: Dict[int, float] = {}
        scales: List[float] = []

        for cand, dist in zip(ordered, ordered_dists):
            if cand == i:
                continue
            y_nei_pref = Y_full[cand, :T_pref]
            s = _align_scale(Y_full[i, :T_pref], y_nei_pref)
            scales.append(float(s))
            suffix = s * Y_full[cand, T_pref:]
            if not neighbors:
                suffixes = suffix[np.newaxis, :]
            else:
                suffixes = np.vstack([suffixes, suffix])
            dist_umap = 0.0
            if np.all(np.isfinite(Z[i])) and np.all(np.isfinite(Z[cand])):
                dist_umap = float(np.linalg.norm(Z[i] - Z[cand]))
            w = math.exp(-alpha_map * dist_umap)
            if np.isfinite(clusters[i]) and clusters[i] >= 0 and clusters[cand] == clusters[i]:
                w *= 1.0 + w_cluster_match
            weights.append(float(w))
            neighbors.append(int(cand))
            prefix_dists[int(cand)] = float(dist)
            if len(neighbors) >= K:
                break

        if not neighbors:
            continue

        weights_arr = np.asarray(weights, dtype=float)
        if not np.isfinite(weights_arr).any() or weights_arr.sum() <= 0:
            weights_arr = np.ones_like(weights_arr)
        else:
            weights_arr = np.nan_to_num(weights_arr, nan=0.0, posinf=0.0, neginf=0.0)
        Y_pred[i] = np.average(suffixes, axis=0, weights=weights_arr)

        debug_rows[i] = {
            "neighbors": neighbors,
            "weights": weights_arr.tolist(),
            "prefix_distances": prefix_dists,
            "scales": scales,
            "target_cluster": clusters[i] if i < len(clusters) else np.nan,
            "anomaly_threshold": anomaly_threshold,
        }

    debug_info = {
        "train_indices": donors,
        "selection": debug_rows,
    }
    return Y_pred, debug_info

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


def _baseline_knn_single(
    X_pref: np.ndarray,
    Y_full: np.ndarray,
    T_pref: int,
    donors: np.ndarray,
    target_idx: int,
    K: int,
) -> Tuple[np.ndarray, List[int]]:
    donors = np.asarray(donors, dtype=int)
    donors = donors[(donors >= 0) & (donors < X_pref.shape[0])]
    if donors.size == 0:
        return np.full((Y_full.shape[1] - T_pref,), np.nan), []
    ordered, _ = _ordered_donors_by_prefix(X_pref, donors, target_idx)
    neighbors: List[int] = []
    suffixes = []
    y_ref_pref = Y_full[target_idx, :T_pref]
    for cand in ordered:
        if cand == target_idx:
            continue
        neighbors.append(int(cand))
        s = _align_scale(y_ref_pref, Y_full[cand, :T_pref])
        suffixes.append(s * Y_full[cand, T_pref:])
        if len(neighbors) >= K:
            break
    if not neighbors:
        return np.full((Y_full.shape[1] - T_pref,), np.nan), []
    suffixes = np.vstack(suffixes)
    return np.nanmedian(suffixes, axis=0), neighbors


def benchmark_cluster_aware(
    panel_long: pd.DataFrame,
    wells: list,
    T: int,
    n_list: Sequence[int] = (3, 6, 9, 12),
    K: int = 15,
    variant: str = "restricted",
    df_map: Optional[pd.DataFrame] = None,
    Z: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """LOO-бенчмарк kNN и кластер-aware варианта по наборам префиксов."""

    variant_key = variant.lower()
    if variant_key not in {"restricted", "weighted"}:
        raise ValueError("variant должен быть 'restricted' или 'weighted'")

    wells = list(wells)
    if not wells:
        return pd.DataFrame(columns=["well", "n", "method", "rmse", "smape"])

    panel = panel_long[panel_long["well_name"].isin(wells)].copy()
    panel = panel[panel["t"].between(0, T - 1)]

    manifold_cfg = ManifoldConfig()
    tensor_candidates = list(dict.fromkeys(manifold_cfg.channels + ("gor", "dr_oil_norm", "r_oil_norm", "wc")))
    tensor_channels = [c for c in tensor_candidates if c in panel.columns]
    missing = [c for c in manifold_cfg.channels if c not in tensor_channels]
    if missing:
        raise ValueError(f"panel_long не содержит каналов для UMAP: {missing}")

    tensor_mats = [_collect_matrix(panel, wells, ch, T) for ch in tensor_channels]
    X_tensor = np.stack(tensor_mats, axis=-1).astype(float)

    results: List[Dict[str, object]] = []
    N = len(wells)

    for n in n_list:
        panel_pref = build_prefix_scaled_channel(panel, wells, T, T_pref=int(n))
        X_pref, _, Y_full = make_matrices(panel_pref, wells, T, T_pref=int(n))
        T_suffix = T - int(n)

        for test_idx, well in enumerate(wells):
            y_true_suffix = Y_full[test_idx, int(n):]
            if not np.isfinite(y_true_suffix).all():
                continue

            train_idx = np.array([j for j in range(N) if j != test_idx], dtype=int)
            if train_idx.size < 3:
                continue

            donors_mask = np.isfinite(Y_full[train_idx]).sum(axis=1) >= T
            donors_idx = train_idx[donors_mask]
            if donors_idx.size == 0:
                continue

            n_neighbors = max(2, min(manifold_cfg.n_neighbors, len(train_idx) - 1))
            Z_train, umap_model = embed_umap_euclid(
                X_tensor[train_idx],
                tensor_channels,
                manifold_cfg.channels,
                n_neighbors=n_neighbors,
                min_dist=manifold_cfg.min_dist,
                n_components=manifold_cfg.n_components,
                random_state=manifold_cfg.random_state,
            )

            cluster_cfg = ClusterConfig(
                min_cluster_size=max(2, min(len(train_idx) // 2, 25)),
                min_samples=max(1, min(len(train_idx) // 2, 10)),
                allow_single_cluster=True,
            )
            try:
                cluster_res = cluster_hdbscan(Z_train, [wells[j] for j in train_idx], cfg=cluster_cfg)
                df_map_train = cluster_res["df_map"].reset_index(drop=True)
                clusterer = cluster_res["clusterer"]
            except Exception:
                df_map_train = pd.DataFrame(
                    {
                        "well_name": [wells[j] for j in train_idx],
                        "well": [wells[j] for j in train_idx],
                        "x": Z_train[:, 0],
                        "y": Z_train[:, 1],
                        "cluster": np.full(len(train_idx), -1),
                        "prob": np.ones(len(train_idx)),
                        "lof_score": np.zeros(len(train_idx)),
                        "dist_medoid": np.zeros(len(train_idx)),
                        "anomaly_score": np.zeros(len(train_idx)),
                    }
                )
                clusterer = None

            Z_target = transform_prefix_to_Z(
                umap_model,
                X_tensor[test_idx, : int(n), :],
                tensor_channels,
                manifold_cfg.channels,
            )
            if clusterer is not None:
                membership = approximate_membership(clusterer, Z_target)
                target_cluster = int(membership["labels"][0])
                target_prob = float(membership["prob"][0])
            else:
                target_cluster = -1
                target_prob = 0.0

            cols = list(df_map_train.columns)
            target_row = {c: np.nan for c in cols}
            if "well_name" in target_row:
                target_row["well_name"] = well
            if "well" in target_row:
                target_row["well"] = well
            target_row["x"] = float(Z_target[0, 0]) if Z_target.size else np.nan
            target_row["y"] = float(Z_target[0, 1]) if Z_target.size else np.nan
            target_row["cluster"] = target_cluster
            target_row["prob"] = target_prob
            if "anomaly_score" in target_row:
                target_row["anomaly_score"] = np.nan
            if "lof_score" in target_row:
                target_row["lof_score"] = np.nan
            if "dist_medoid" in target_row:
                target_row["dist_medoid"] = np.nan

            df_map_fold = pd.concat([df_map_train, pd.DataFrame([target_row])], ignore_index=True)
            Z_fold = np.vstack([Z_train, Z_target])
            X_pref_fold = np.vstack([X_pref[train_idx], X_pref[test_idx]])
            Y_full_fold = np.vstack([Y_full[train_idx], Y_full[test_idx]])
            Y_full_fold[-1, int(n):] = np.nan

            donors_local = np.where(np.isfinite(Y_full_fold[: len(train_idx)]).sum(axis=1) >= T)[0]
            if donors_local.size == 0:
                continue

            baseline_pred, _ = _baseline_knn_single(
                X_pref_fold,
                Y_full_fold,
                T_pref=int(n),
                donors=donors_local,
                target_idx=len(train_idx),
                K=K,
            )
            if np.isfinite(baseline_pred).all():
                metrics = evaluate_forecasts(y_true_suffix.reshape(1, -1), baseline_pred.reshape(1, -1))
                results.append(
                    {
                        "well": well,
                        "n": int(n),
                        "method": "baseline",
                        "rmse": metrics["rmse"],
                        "smape": metrics["smape"],
                    }
                )

            if variant_key == "restricted":
                min_incluster = max(1, min(5, donors_local.size))
                pred_variant_full, _ = knn_forecast_cluster_restricted(
                    X_pref_fold,
                    Y_full_fold,
                    T_pref=int(n),
                    df_map=df_map_fold,
                    train_indices=donors_local,
                    K=K,
                    min_incluster=min_incluster,
                )
                variant_pred = pred_variant_full[len(train_idx)]
            else:
                pred_variant_full, _ = knn_forecast_cluster_weighted(
                    X_pref_fold,
                    Y_full_fold,
                    T_pref=int(n),
                    Z=Z_fold,
                    df_map=df_map_fold,
                    K=K,
                )
                variant_pred = pred_variant_full[len(train_idx)]

            if np.isfinite(variant_pred).all():
                metrics_var = evaluate_forecasts(y_true_suffix.reshape(1, -1), variant_pred.reshape(1, -1))
                results.append(
                    {
                        "well": well,
                        "n": int(n),
                        "method": f"cluster-{variant_key}",
                        "rmse": metrics_var["rmse"],
                        "smape": metrics_var["smape"],
                    }
                )

    return pd.DataFrame(results)


def evaluate_forecasts(Y_true, Y_pred):
    """RMSE и sMAPE по строкам, где и факт, и прогноз без NaN; без зависимости от 'squared'."""
    mask = np.isfinite(Y_pred).all(axis=1) & np.isfinite(Y_true).all(axis=1)
    n_eval = int(mask.sum())
    if n_eval == 0:
        return {"rmse": float("nan"), "smape": float("nan"), "n_eval": 0}
    diff = Y_pred[mask] - Y_true[mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    smape = float(np.nanmean(2*np.abs(Y_pred[mask]-Y_true[mask])/(np.abs(Y_pred[mask])+np.abs(Y_true[mask])+1e-9)))
    return {"rmse": rmse, "smape": smape, "n_eval": n_eval}
