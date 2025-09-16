# -*- coding: utf-8 -*-
"""Tools for clustering embeddings produced by the PBM pipeline.

The original version of the project returned ad-hoc dictionaries from helper
functions which made the notebook code rather implicit.  The module now exposes
typed dataclasses so the intent of every step (clustering, anomaly scoring,
prototype construction) is explicit and easier to extend.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.mixture import GaussianMixture

# --- HDBSCAN ---
try:
    import hdbscan
except Exception as e:
    raise RuntimeError("Требуется пакет 'hdbscan'. Установите: pip install hdbscan")

# --- tslearn (опционально для барицентров) ---
try:
    from tslearn.barycenters import softdtw_barycenter, dtw_barycenter_averaging
    _TSLEARN_OK = True
except Exception:
    _TSLEARN_OK = False
    warnings.warn("tslearn не найден: прототипы будут как медианные кривые (без барицентров)")

try:
    from tqdm import tqdm
    _TQDM = True
except Exception:
    _TQDM = False

# ======================================================
# ---------------- КЛАСТЕРИЗАЦИЯ -----------------------
# ======================================================

__all__ = [
    "ClusterConfig",
    "ClusterResult",
    "cluster_hdbscan",
    "cluster_gmm_bic",
    "lof_anomaly_scores",
    "distance_to_medoid",
    "build_cluster_prototypes",
    "summarize_clusters",
    "assign_anomaly_scores",
]


@dataclass(slots=True)
class ClusterConfig:
    """Configuration for HDBSCAN clustering and anomaly scoring."""

    # HDBSCAN parameters
    min_cluster_size: int = 50
    min_samples: int = 12
    cluster_selection_epsilon: float = 0.0
    allow_single_cluster: bool = False
    metric: str = "euclidean"

    # Local Outlier Factor used later for anomaly scoring
    lof_neighbors: int = 30


@dataclass(slots=True)
class ClusterResult(Mapping[str, Any]):
    """Container returned by clustering helpers.

    The class acts as a read-only mapping to keep backwards compatibility with
    the previous dictionary-based API while providing convenient attribute
    access.  Additional information (e.g. the BIC table for the GMM variant) can
    be stored inside ``extras``.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    model: Any
    df_map: pd.DataFrame
    silhouette: float = float("nan")
    dbcv: float = float("nan")
    extras: Dict[str, Any] = field(default_factory=dict)

    _KEYS: tuple[str, ...] = ("labels", "probabilities", "model", "df_map", "silhouette", "dbcv", "extras")

    def __getitem__(self, key: str) -> Any:  # Mapping interface
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow ``dict`` representation of the result."""

        return {key: getattr(self, key) for key in self._KEYS}

    @property
    def n_clusters(self) -> int:
        """Number of non-noise clusters in the assignment."""

        return int(np.sum(np.unique(self.labels) >= 0))

    @property
    def has_noise(self) -> bool:
        """Whether HDBSCAN detected a noise cluster (label ``-1``)."""

        return bool(np.any(self.labels < 0))


def _build_cluster_frame(
    Z: np.ndarray, wells_sub: Sequence[str], labels: np.ndarray, probs: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_name": wells_sub,
            "x": Z[:, 0],
            "y": Z[:, 1],
            "cluster": labels,
            "prob": probs,
        }
    )


def _cluster_metrics(Z: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return ``(silhouette, dbcv)`` with graceful fallbacks for degenerate cases."""

    sil = float("nan")
    dbcv = float("nan")
    mask = labels >= 0
    try:
        if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
            sil = float(silhouette_score(Z[mask], labels[mask], metric="euclidean"))
    except Exception:
        sil = float("nan")

    try:
        dbcv = float(hdbscan.validity.validity_index(Z, labels, metric="euclidean"))
    except Exception:
        dbcv = float("nan")
    return sil, dbcv


def cluster_hdbscan(
    Z: np.ndarray,
    wells_sub: Sequence[str],
    cfg: Optional[ClusterConfig] = None,
) -> ClusterResult:
    """Cluster the embedding ``Z`` with HDBSCAN.

    Parameters
    ----------
    Z:
        Two dimensional embedding produced by the manifold step.
    wells_sub:
        Names of wells in the same order as ``Z`` rows.
    cfg:
        Optional :class:`ClusterConfig` instance.  ``None`` produces the
        defaults that were previously hard-coded.
    """

    if cfg is None:
        cfg = ClusterConfig()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        metric=cfg.metric,
        cluster_selection_epsilon=cfg.cluster_selection_epsilon,
        allow_single_cluster=cfg.allow_single_cluster,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(Z)
    probs = getattr(clusterer, "probabilities_", np.ones(len(labels)))
    df_map = _build_cluster_frame(Z, wells_sub, labels, probs)
    sil, dbcv = _cluster_metrics(Z, labels)
    return ClusterResult(labels=labels, probabilities=probs, model=clusterer, df_map=df_map, silhouette=sil, dbcv=dbcv)


def cluster_gmm_bic(
    Z: np.ndarray,
    wells_sub: Sequence[str],
    n_range: Sequence[int] = range(2, 11),
    random_state: int = 42,
) -> ClusterResult:
    """Cluster the embedding with :class:`GaussianMixture` using BIC selection."""

    best_bic = math.inf
    best_model: Optional[GaussianMixture] = None
    rows = []
    for k in n_range:
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gm.fit(Z)
        bic = gm.bic(Z)
        rows.append({"k": k, "bic": bic})
        if bic < best_bic:
            best_bic = bic
            best_model = gm

    if best_model is None:
        raise RuntimeError("GaussianMixture training failed for all candidate cluster counts.")

    bic_table = pd.DataFrame(rows)
    labels = best_model.predict(Z)
    proba = best_model.predict_proba(Z)
    probs = proba.max(axis=1)
    df_map = _build_cluster_frame(Z, wells_sub, labels, probs)
    sil, _ = _cluster_metrics(Z, labels)
    extras = {"best_k": int(best_model.n_components), "bic_table": bic_table, "proba": proba}
    return ClusterResult(labels=labels, probabilities=probs, model=best_model, df_map=df_map, silhouette=sil, extras=extras)


# ======================================================
# ------------------- АНОМАЛИИ -------------------------
# ======================================================

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).any():
        return np.zeros_like(scores)
    return (scores - np.nanmin(scores)) / (np.nanmax(scores) - np.nanmin(scores) + 1e-12)


def lof_anomaly_scores(Z: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    """Return Local Outlier Factor scores normalised to ``[0, 1]``."""

    if len(Z) < 2:
        return np.zeros(len(Z))
    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(Z) - 1), novelty=False)
    lof.fit(Z)
    scores = -lof.negative_outlier_factor_
    return _normalize_scores(scores)


def distance_to_medoid(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Расстояние точки до медоида своего кластера в координатах embedding.
    Для шума (-1) — расстояние до ближайшего кластера.
    """
    uniq = [c for c in np.unique(labels) if c >= 0]
    if not len(uniq):
        return np.full(len(labels), np.nan)
    medoids = {}
    for c in uniq:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        D = pairwise_distances(Z[idx], metric="euclidean")
        m = idx[np.argmin(D.sum(axis=0))]
        medoids[c] = Z[m]
    out = np.zeros(len(labels))
    for i, lab in enumerate(labels):
        if lab >= 0:
            mu = medoids.get(lab)
            out[i] = float(np.linalg.norm(Z[i] - mu)) if mu is not None else float("nan")
        else:
            if medoids:
                dmin = np.min([np.linalg.norm(Z[i] - mu) for mu in medoids.values()])
            else:
                dmin = float("nan")
            out[i] = float(dmin)
    return _normalize_scores(out)


# ======================================================
# --------------- ПРОТОТИПЫ КЛАСТЕРОВ ------------------
# ======================================================

def _collect_matrix(panel_long: pd.DataFrame, wells: Sequence[str], channel: str, T: int) -> np.ndarray:
    """Собирает матрицу [n_series, T] по указанному каналу с NaN.
    """
    rows = []
    for w in wells:
        g = panel_long.loc[panel_long["well_name"] == w, ["t", channel]].sort_values("t")
        v = np.full(T, np.nan, float)
        t = g["t"].to_numpy(int)
        vals = g[channel].to_numpy(float)
        t = t[(t >= 0) & (t < T)]
        vals = vals[: len(t)]
        v[t[: len(vals)]] = vals
        rows.append(v)
    return np.vstack(rows) if rows else np.empty((0, T))


def _barycenter_or_median(M: np.ndarray, method: str = "auto", gamma: float = 1.0, max_iter: int = 50) -> np.ndarray:
    """Возвращает барицентр (soft-DTW/DBA) или медиану по времени, если tslearn недоступен.
    NaN заполняются нулями перед барицентром (так как tslearn не поддерживает NaN).
    """
    if M.size == 0:
        return M
    if method == "auto":
        method = "softdtw" if _TSLEARN_OK else "median"
    if method in ("softdtw", "dba") and not _TSLEARN_OK:
        method = "median"

    if method == "median":
        return np.nanmedian(M, axis=0)
    M_filled = np.nan_to_num(M, nan=0.0)
    if method == "softdtw":
        try:
            return softdtw_barycenter(M_filled, gamma=gamma, max_iter=max_iter)
        except Exception:
            return np.nanmedian(M, axis=0)
    if method == "dba":
        try:
            return dtw_barycenter_averaging(M_filled, max_iter=max_iter)
        except Exception:
            return np.nanmedian(M, axis=0)
    # fallback
    return np.nanmedian(M, axis=0)


def build_cluster_prototypes(
    panel_long: pd.DataFrame,
    df_map: pd.DataFrame,
    channels: Sequence[str] = ("r_oil_s", "wc", "gor", "r_oil_norm"),
    T: int = 36,
    method: str = "auto",   # 'auto'|'softdtw'|'dba'|'median'
    gamma: float = 1.0,
    max_iter: int = 50,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Строит прототипы (по-канально) для каждого кластера (cluster>=0).
    Возвращает: {cluster_id: {channel: proto_vec_length_T}}
    Также можно использовать для отрисовки IQR/квантили — верните сырой M при необходимости.
    """
    # Скважины по кластерам
    cl2wells: Dict[int, List[str]] = {}
    for row in df_map.itertuples(index=False):
        if row.cluster >= 0:
            cl2wells.setdefault(int(row.cluster), []).append(str(row.well_name))

    res: Dict[int, Dict[str, np.ndarray]] = {}
    for cl, wells in cl2wells.items():
        res[cl] = {}
        for ch in channels:
            M = _collect_matrix(panel_long, wells, ch, T)
            proto = _barycenter_or_median(M, method=method, gamma=gamma, max_iter=max_iter)
            res[cl][ch] = proto
    return res


# ======================================================
# -------------- УТИЛИТЫ ДЛЯ ОТЧЁТА --------------------
# ======================================================

def summarize_clusters(df_map: pd.DataFrame | ClusterResult) -> pd.DataFrame:
    """Возвращает сводную таблицу: размер кластера, медиана prob, доля шума."""
    if isinstance(df_map, ClusterResult):
        df = df_map.df_map
    else:
        df = df_map
    total = len(df)
    noise_share = (df["cluster"].eq(-1)).mean() if total else np.nan
    rows = []
    for cl, g in df.groupby("cluster"):
        rows.append({
            "cluster": int(cl),
            "size": int(len(g)),
            "share": float(len(g) / total) if total else np.nan,
            "prob_median": float(np.median(g["prob"].values)) if len(g) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values(["cluster"]).reset_index(drop=True)
    out.attrs["noise_share"] = float(noise_share)
    return out


def assign_anomaly_scores(
    df_map: pd.DataFrame | ClusterResult,
    Z: np.ndarray,
    labels: Optional[np.ndarray] = None,
    lof_k: int = 30,
) -> pd.DataFrame:
    """Добавляет к df_map столбцы: lof_score [0..1], dist_medoid [0..1], anomaly_score."""

    if isinstance(df_map, ClusterResult):
        base_df = df_map.df_map.copy()
        if labels is None:
            labels = df_map.labels
    else:
        base_df = df_map.copy()
    if labels is None:
        if "cluster" not in base_df.columns:
            raise ValueError("labels must be provided when df_map lacks 'cluster' column")
        labels = base_df["cluster"].to_numpy()

    lof = lof_anomaly_scores(Z, n_neighbors=lof_k)
    dmed = distance_to_medoid(Z, labels)
    anom = 0.5 * (lof + dmed)
    out = base_df
    out["lof_score"] = lof
    out["dist_medoid"] = dmed
    out["anomaly_score"] = anom
    return out
