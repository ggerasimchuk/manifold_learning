# -*- coding: utf-8 -*-
"""
Production Behavior Manifold (PBM) — Шаг 4: Сегментация и аномалии
------------------------------------------------------------------
Зависимости (pip): numpy, pandas, scikit-learn, hdbscan, (опц.) tslearn

Входы из предыдущих шагов:
  - out: результат preprocess_profiles (Шаг 1)
      * out["panel_long"] — long-таблица после выравнивания t=0
      * out["X"] — тензор [N, T, C]
      * out["wells_used"] — список имён в порядке X
      * out["tensor_channels"] — имена каналов в X
      * out["config"]["T"] — горизонт T
  - Z, sub_idx: embedding и индексы скважин, вошедшие в него (из Шага 3)

Функционал:
  1) Кластеризация HDBSCAN на manifold (поддержка шума/аномалий)
  2) Альтернатива: GMM по BIC
  3) Аномалии: LOF + расстояние до медоида кластера
  4) Прототипы кластеров (медианные профили и, опционально, soft-DTW/DTW барицентры)
  5) Метрики качества: Silhouette (без шума), DBCV (для HDBSCAN)
  6) Удобные хелперы для сборки отчёта

Примечание: dtaidistance НЕ используется. Для soft-DTW/DBA — опционально tslearn.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from tools.common import collect_time_series, compute_cluster_medoids
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

@dataclass
class ClusterConfig:
    # HDBSCAN
    min_cluster_size: int = 50
    min_samples: int = 12
    cluster_selection_epsilon: float = 0.0
    allow_single_cluster: bool = False
    # LOF
    lof_neighbors: int = 30


def cluster_hdbscan(
    Z: np.ndarray,
    wells_sub: Sequence[str],
    cfg: Optional[ClusterConfig] = None,
    metric: str = "euclidean",
) -> Dict[str, object]:
    """Кластеризация HDBSCAN на низкоразмерных координатах Z.

    Возвращает dict:
      labels: np.array [Ns]
      probabilities: np.array [Ns]
      clusterer: HDBSCAN объект
      df_map: DataFrame(well_name, x, y, cluster, prob)
      silhouette: float | np.nan (по точкам с cluster>=0)
      dbcv: float | np.nan
    """
    if cfg is None:
        cfg = ClusterConfig()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        metric=metric,
        cluster_selection_epsilon=cfg.cluster_selection_epsilon,
        allow_single_cluster=cfg.allow_single_cluster,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(Z)
    probs = getattr(clusterer, "probabilities_", np.ones(len(labels)))

    # Карта
    df_map = pd.DataFrame({
        "well_name": wells_sub,
        "x": Z[:, 0],
        "y": Z[:, 1],
        "cluster": labels,
        "prob": probs,
    })

    # Метрики
    mask = labels >= 0
    if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
        try:
            sil = float(silhouette_score(Z[mask], labels[mask], metric="euclidean"))
        except Exception:
            sil = float("nan")
    else:
        sil = float("nan")

    try:
        dbcv = float(hdbscan.validity.validity_index(Z, labels, metric="euclidean"))
    except Exception:
        dbcv = float("nan")

    return {
        "labels": labels,
        "probabilities": probs,
        "clusterer": clusterer,
        "df_map": df_map,
        "silhouette": sil,
        "dbcv": dbcv,
    }


def cluster_gmm_bic(Z: np.ndarray, wells_sub: Sequence[str], n_range: Sequence[int] = range(2, 11)) -> Dict[str, object]:
    """Альтернативная кластеризация GMM с выбором числа компонент по BIC.
    Возвращает dict аналогичный HDBSCAN (labels, df_map, silhouette, model, best_k, bic_table).
    """
    best_bic = np.inf
    best = None
    rows = []
    for k in n_range:
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gm.fit(Z)
        bic = gm.bic(Z)
        rows.append({"k": k, "bic": bic})
        if bic < best_bic:
            best_bic = bic
            best = gm
    bic_table = pd.DataFrame(rows)
    labels = best.predict(Z)
    probs = best.predict_proba(Z).max(axis=1)
    df_map = pd.DataFrame({"well_name": wells_sub, "x": Z[:,0], "y": Z[:,1], "cluster": labels, "prob": probs})
    sil = float("nan")
    mask = labels >= 0
    if len(np.unique(labels[mask])) > 1:
        try:
            sil = float(silhouette_score(Z[mask], labels[mask], metric="euclidean"))
        except Exception:
            pass
    return {
        "labels": labels,
        "probabilities": probs,
        "model": best,
        "best_k": int(np.unique(labels).size),
        "df_map": df_map,
        "silhouette": sil,
        "bic_table": bic_table,
    }


# ======================================================
# ------------------- АНОМАЛИИ -------------------------
# ======================================================

def lof_anomaly_scores(Z: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    """Возвращает аномальность в [0..1], где 1 ~ сильная аномалия.
    На основе LocalOutlierFactor (scikit-learn), преобразуя negative_outlier_factor_.
    """
    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(Z) - 1), novelty=False)
    lof.fit(Z)
    # Чем более отрицателен negative_outlier_factor_, тем более аномальна точка.
    s = -lof.negative_outlier_factor_
    s = (s - np.min(s)) / (np.ptp(s) + 1e-12)
    return s


def distance_to_medoid(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Расстояние точки до медоида своего кластера в координатах embedding.
    Для шума (-1) — расстояние до ближайшего кластера.
    """
    medoid_idx = compute_cluster_medoids(Z, labels)
    if not medoid_idx:
        return np.full(len(labels), np.nan)
    medoids = {cl: Z[idx] for cl, idx in medoid_idx.items()}
    out = np.zeros(len(labels))
    for i, lab in enumerate(labels):
        if lab >= 0 and lab in medoids:
            out[i] = float(np.linalg.norm(Z[i] - medoids[lab]))
        else:
            # для шума — ближайший медоид
            out[i] = float(min(np.linalg.norm(Z[i] - mu) for mu in medoids.values()))
    # нормализация [0..1]
    out = (out - np.min(out)) / (np.ptp(out) + 1e-12)
    return out


# ======================================================
# --------------- ПРОТОТИПЫ КЛАСТЕРОВ ------------------
# ======================================================

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
            M = collect_time_series(panel_long, wells, ch, T)
            proto = _barycenter_or_median(M, method=method, gamma=gamma, max_iter=max_iter)
            res[cl][ch] = proto
    return res


# ======================================================
# -------------- УТИЛИТЫ ДЛЯ ОТЧЁТА --------------------
# ======================================================

def summarize_clusters(df_map: pd.DataFrame) -> pd.DataFrame:
    """Возвращает сводную таблицу: размер кластера, медиана prob, доля шума."""
    total = len(df_map)
    noise_share = (df_map["cluster"].eq(-1)).mean() if total else np.nan
    rows = []
    for cl, g in df_map.groupby("cluster"):
        rows.append({
            "cluster": int(cl),
            "size": int(len(g)),
            "share": float(len(g) / total) if total else np.nan,
            "prob_median": float(np.median(g["prob"].values)) if len(g) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values(["cluster"]).reset_index(drop=True)
    out.attrs["noise_share"] = float(noise_share)
    return out


def assign_anomaly_scores(df_map: pd.DataFrame, Z: np.ndarray, labels: np.ndarray, lof_k: int = 30) -> pd.DataFrame:
    """Добавляет к df_map столбцы: lof_score [0..1], dist_medoid [0..1], anomaly_score (среднее двух)."""
    lof = lof_anomaly_scores(Z, n_neighbors=lof_k)
    dmed = distance_to_medoid(Z, labels)
    anom = 0.5 * (lof + dmed)
    out = df_map.copy()
    out["lof_score"] = lof
    out["dist_medoid"] = dmed
    out["anomaly_score"] = anom
    return out

