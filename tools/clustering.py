# -*- coding: utf-8 -*-
"""
Функции:
- cluster_hdbscan: кластеризация HDBSCAN с валидацией входа, метриками, центрами и медоидами.
- cluster_gmm_bic: GMM с выбором k по BIC, стандартизация, гибкие covariance_type/reg_covar.
- assign_anomaly_scores: скоринг аномалий с rank-агрегацией LOF, dist-to-medoid и HDBSCAN outlier_scores.
- build_cluster_prototypes: прототипы кластеров (median или soft-DTW/DBA с весами prob), без замены NaN на 0.
- summarize_clusters: сводка по кластерам.
- _collect_matrix: сбор матрицы [N,T] по каналам из panel_long.

Улучшения:
1) Валидация Z и wells_sub.
2) Согласование метрик: silhouette с выбранной metric; DBCV считаем только при 'euclidean', иначе NaN.
3) Медоиды без O(n²): медоид — точка, ближайшая к центру масс; dist-to-medoid с защитой ptp==0.
4) Аномалии: добавлен HDBSCAN outlier_scores_, итог — ранговое среднее трёх сигналов.
5) Прототипы: per-time median по маске NaN; soft-DTW/DBA с весами prob и ограничением N_max.
6) GMM: StandardScaler, возврат best_k = model.n_components, опции covariance_type и reg_covar.
7) Производительность: меньше копий DataFrame, работа через массивы; строгие dtype df_map.
8) Защита краёв: кластеры размера 1, только шум, пустые медоиды и т.п.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

try:
    import hdbscan
    from hdbscan import validity as hdb_validity
    _HAVE_HDBSCAN = True
except Exception:
    _HAVE_HDBSCAN = False

# tslearn — опционально для soft-DTW/DBA
try:
    from tslearn.barycenters import softdtw_barycenter
    _HAVE_TSLEARN = True
except Exception:
    _HAVE_TSLEARN = False


# ============================ ВАЛИДАЦИЯ ============================

def _validate_embedding_and_wells(Z: np.ndarray, wells_sub: Iterable[str]) -> List[str]:
    if not isinstance(Z, np.ndarray):
        raise TypeError("Z должен быть numpy.ndarray")
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError("Z должен иметь форму [N, d] с d>=2")
    wells_sub = list(wells_sub)
    if len(wells_sub) != len(Z):
        raise ValueError("len(wells_sub) должен равняться Z.shape[0]")
    if len(set(wells_sub)) != len(wells_sub):
        raise ValueError("wells_sub содержит дубликаты")
    return wells_sub


def _as_float32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32, order="C")


# ============================ УТИЛИТЫ КЛАСТЕРОВ ============================

def _cluster_centers(Z: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    centers: Dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        if c < 0:
            continue
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        centers[int(c)] = Z[idx].mean(axis=0)
    return centers


def _cluster_medoids_via_center(Z: np.ndarray, labels: np.ndarray, centers: Dict[int, np.ndarray]) -> Dict[int, int]:
    """
    Медоид кластера = индекс точки, ближайшей к центру масс.
    O(N) на кластер, без полной матрицы расстояний.
    """
    medoids: Dict[int, int] = {}
    for c, mu in centers.items():
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        # расстояния до центра
        d = np.linalg.norm(Z[idx] - mu, axis=1)
        medoids[c] = int(idx[np.argmin(d)])
    return medoids


def _dist_to_cluster_medoids(Z: np.ndarray, labels: np.ndarray, medoids: Dict[int, int]) -> np.ndarray:
    """
    Для каждой точки считаем расстояние до медоида её кластера.
    Для шума берём минимум до любого медоида; если медоидов нет, возвращаем NaN.
    """
    n = len(Z)
    out = np.full(n, np.nan, dtype=np.float64)
    if not medoids:
        return out
    med_idx = np.array(list(medoids.values()), dtype=int)
    med_pts = Z[med_idx]
    for i in range(n):
        c = int(labels[i])
        if c >= 0 and c in medoids:
            out[i] = np.linalg.norm(Z[i] - Z[medoids[c]])
        else:
            # шум: минимальная дистанция до любого медоида
            out[i] = np.linalg.norm(Z[i] - med_pts, axis=1).min()
    return out


def _minmax01(x: np.ndarray) -> np.ndarray:
    """Нормировка в [0,1] с защитой ptp==0."""
    y = np.asarray(x, dtype=float)
    if not np.isfinite(y).any():
        return np.zeros_like(y)
    vmin = np.nanmin(y)
    vmax = np.nanmax(y)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(y)
    return (y - vmin) / (vmax - vmin)


# ============================ HDBSCAN ============================

@dataclass
class HDBSCANParams:
    min_cluster_size: int = 30
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.0
    alpha: float = 1.0
    allow_single_cluster: bool = False
    prediction_data: bool = True
    random_state: Optional[int] = None


def cluster_hdbscan(
    Z: np.ndarray,
    wells_sub: Iterable[str],
    params: Optional[HDBSCANParams] = None,
) -> Dict:
    """
    Возвращает:
    {
      'labels': np.ndarray[int],
      'prob': np.ndarray[float],
      'df_map': DataFrame(well_name,x,y,cluster,prob),
      'model': HDBSCAN,
      'silhouette': float|np.nan,
      'dbcv': float|np.nan,
      'centers': Dict[cluster -> np.ndarray],
      'medoids': Dict[cluster -> int],
    }
    """
    if not _HAVE_HDBSCAN:
        raise ImportError("hdbscan не установлен")

    wells_sub = _validate_embedding_and_wells(Z, wells_sub)
    Z = np.asarray(Z, dtype=np.float64, order="C")
    if params is None:
        params = HDBSCANParams()

    # Модель
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric=params.metric,
        cluster_selection_method=params.cluster_selection_method,
        cluster_selection_epsilon=params.cluster_selection_epsilon,
        alpha=params.alpha,
        allow_single_cluster=params.allow_single_cluster,
        prediction_data=params.prediction_data,
        core_dist_n_jobs=1,  # стабильность
        gen_min_span_tree=False,
    ).fit(Z)

    labels = clusterer.labels_.astype(np.int32, copy=False)
    prob = getattr(clusterer, "probabilities_", None)
    if prob is None:
        prob = np.ones(len(Z), dtype=np.float32)
    else:
        prob = _as_float32(prob)

    # Метрики качества
    unique_clusters = np.unique(labels[labels >= 0])
    silhouette = np.nan
    if unique_clusters.size >= 2 and np.unique(labels).size > 1:
        try:
            silhouette = float(silhouette_score(Z, labels, metric=params.metric))
        except Exception:
            silhouette = np.nan

    # DBCV надёжно вычисляем только в евклидовых координатах
    dbcv = np.nan
    if params.metric == "euclidean" and unique_clusters.size >= 2:
        try:
            dbcv = float(hdb_validity.validity_index(Z, labels))
        except Exception:
            dbcv = np.nan

    # Центры и медоиды
    centers = _cluster_centers(Z, labels)
    medoids = _cluster_medoids_via_center(Z, labels, centers)

    # df_map
    df_map = pd.DataFrame({
        "well_name": wells_sub,
        "x": _as_float32(Z[:, 0]),
        "y": _as_float32(Z[:, 1]),
        "cluster": labels.astype(np.int32, copy=False),
        "prob": _as_float32(prob),
    })

    return dict(
        labels=labels,
        prob=prob,
        df_map=df_map,
        model=clusterer,
        silhouette=silhouette,
        dbcv=dbcv,
        centers=centers,
        medoids=medoids,
    )


# ============================ GMM + BIC ============================

def cluster_gmm_bic(
    Z: np.ndarray,
    wells_sub: Iterable[str],
    k_range: Tuple[int, int] = (2, 12),
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    random_state: int = 0,
) -> Dict:
    """
    Возвращает:
    {
      'labels': np.ndarray[int],
      'prob': np.ndarray[float],
      'df_map': DataFrame(well_name,x,y,cluster,prob),
      'model': GaussianMixture,
      'scaler': StandardScaler,
      'bic_table': DataFrame[k, BIC],
      'best_k': int,
    }
    """
    wells_sub = _validate_embedding_and_wells(Z, wells_sub)
    Z = np.asarray(Z, dtype=np.float64, order="C")

    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)

    k_min, k_max = int(k_range[0]), int(k_range[1])
    if k_min < 1 or k_max <= k_min:
        raise ValueError("k_range должен быть (>=1, >min)")

    records = []
    models = {}
    for k in range(k_min, k_max + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        gm.fit(Zs)
        bic = gm.bic(Zs)
        records.append((k, float(bic)))
        models[k] = gm

    bic_table = pd.DataFrame(records, columns=["k", "bic"]).sort_values("k").reset_index(drop=True)
    best_k = int(bic_table.loc[bic_table["bic"].idxmin(), "k"])
    best = models[best_k]

    labels = best.predict(Zs).astype(np.int32, copy=False)
    resp = best.predict_proba(Zs).astype(np.float64, copy=False)
    prob = resp[np.arange(len(Zs)), labels]
    prob = _as_float32(prob)

    df_map = pd.DataFrame({
        "well_name": wells_sub,
        "x": _as_float32(Z[:, 0]),
        "y": _as_float32(Z[:, 1]),
        "cluster": labels.astype(np.int32, copy=False),
        "prob": prob,
    })

    return dict(
        labels=labels,
        prob=prob,
        df_map=df_map,
        model=best,
        scaler=scaler,
        bic_table=bic_table,
        best_k=int(best.n_components),
    )


# ============================ АНОМАЛИИ ============================

def assign_anomaly_scores(
    Z: np.ndarray,
    df_map: pd.DataFrame,
    labels: Optional[np.ndarray] = None,
    hdbscan_model: Optional[object] = None,
    lof_neighbors: Optional[int] = None,
) -> pd.DataFrame:
    """
    Итоговый скор аномалии = ранговое среднее трёх сигналов:
      1) LOF (чем выше, тем аномальнее),
      2) расстояние до медоида собственного кластера (или ближайшего медоида),
      3) outlier_scores_ из HDBSCAN (если есть, иначе игнорируется).
    Возвращает df_map с колонками: lof_score, dist_medoid, hdb_outlier, anomaly_score, anomaly_rank.
    """
    if labels is None:
        if "cluster" not in df_map.columns:
            raise ValueError("Нужны метки кластера в df_map['cluster'] или передайте labels")
        labels = df_map["cluster"].to_numpy(np.int32, copy=False)
    wells = df_map["well_name"].tolist()

    wells = _validate_embedding_and_wells(Z, wells)
    Z = np.asarray(Z, dtype=np.float64, order="C")
    n = len(Z)

    # Центры, медоиды, дистанции
    centers = _cluster_centers(Z, labels)
    medoids = _cluster_medoids_via_center(Z, labels, centers)
    dist_medoid = _dist_to_cluster_medoids(Z, labels, medoids)
    dist_medoid01 = _minmax01(dist_medoid)

    # LOF
    if lof_neighbors is None:
        lof_neighbors = max(10, min(50, int(np.sqrt(max(1, n)))))
    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination="auto", novelty=False)
    lof_scores = -lof.fit_predict(Z)  # не используется напрямую
    # Используем положительную версию фактора: чем больше, тем "аномальнее"
    lof_factor = -lof.negative_outlier_factor_
    lof01 = _minmax01(lof_factor)

    # HDBSCAN outlier_scores
    hdb_out = None
    if hdbscan_model is not None and hasattr(hdbscan_model, "outlier_scores_"):
        hdb_out = np.asarray(hdbscan_model.outlier_scores_, dtype=float)
        hdb_out = _minmax01(hdb_out)

    # Ранговое агрегирование
    # Преобразуем каждый сигнал в ранги [0,1]; затем среднее
    def _rank01(v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=float)
        idx = np.argsort(np.argsort(x))  # плотные ранги
        return idx / max(1, len(x) - 1)

    parts = [ _rank01(lof01), _rank01(dist_medoid01) ]
    if hdb_out is not None:
        parts.append(_rank01(hdb_out))

    stacked = np.vstack(parts)
    anomaly = stacked.mean(axis=0)
    rank = _rank01(anomaly)

    out = df_map.copy()
    out["lof_score"] = _as_float32(lof01)
    out["dist_medoid"] = dist_medoid.astype(np.float32)
    if hdb_out is not None:
        out["hdb_outlier"] = _as_float32(hdb_out)
    else:
        out["hdb_outlier"] = np.nan
    out["anomaly_score"] = _as_float32(anomaly)
    out["anomaly_rank"] = _as_float32(rank)
    return out


# ============================ ПРОТОТИПЫ ============================

def _collect_matrix(
    panel_long: pd.DataFrame,
    wells: Iterable[str],
    channel: str,
    T: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Собирает матрицу X [N,T] по списку скважин и маску валидности.
    Не заполняет NaN нулями.
    """
    wells = list(wells)
    N = len(wells)
    X = np.full((N, T), np.nan, dtype=np.float32)
    M = np.zeros((N, T), dtype=bool)

    pl = panel_long.loc[panel_long["well_name"].isin(wells), ["well_name", "t", channel]].copy()
    pl = pl[(pl["t"] >= 0) & (pl["t"] < T)]

    # Кодируем скважины для быстрого присваивания
    cat = pd.Categorical(pl["well_name"], categories=wells, ordered=False)
    ii = np.asarray(cat.codes, dtype=int)                     # <-- fix
    tt = pl["t"].to_numpy(dtype=int, copy=False)              # индексы времени
    vals = pd.to_numeric(pl[channel], errors="coerce").to_numpy()

    m = np.isfinite(vals) & (ii >= 0) & (tt >= 0) & (tt < T)  # защита от -1 и выхода за границы
    if m.any():
        ii, tt, vals = ii[m], tt[m], vals[m]
        X[ii, tt] = vals.astype(np.float32, copy=False)
        M[ii, tt] = True

    return X, M



def build_cluster_prototypes(
    panel_long: pd.DataFrame,
    df_map: pd.DataFrame,
    channel: str,
    T: int,
    method: str = "median",          # 'median' | 'softdtw'
    n_max: int = 400,                # ограничение размера кластера при soft-DTW
) -> Dict[int, np.ndarray]:
    """
    Возвращает словарь: cluster -> прототип длины T.
    - 'median': поэлементная медиана по доступным точкам (игнорируя NaN).
    - 'softdtw': soft-DTW/DBA (если доступен tslearn), веса = prob из df_map, сэмплирование до n_max.
    Шум (cluster==-1) не прототипируем.
    """
    if method not in ("median", "softdtw"):
        raise ValueError("method должен быть 'median' или 'softdtw'")

    if method == "softdtw" and not _HAVE_TSLEARN:
        raise ImportError("tslearn не установлен; недоступен метод 'softdtw'")

    prototypes: Dict[int, np.ndarray] = {}
    for c in sorted(df_map["cluster"].unique()):
        if c < 0:
            continue
        wells_c = df_map.loc[df_map["cluster"] == c, "well_name"].tolist()
        if len(wells_c) == 0:
            continue

        Xc, Mc = _collect_matrix(panel_long, wells_c, channel, T)

        if method == "median":
            # по-временному: медиана игнорируя NaN
            proto = np.nanmedian(Xc, axis=0)
            # если весь столбик NaN — заменим на линейную интерполяцию по времени из ближайших определённых
            if np.isnan(proto).any():
                idx = np.arange(T)
                m = ~np.isnan(proto)
                if m.any():
                    proto = np.interp(idx, idx[m], proto[m])
                else:
                    proto = np.zeros(T, dtype=np.float32)
            prototypes[int(c)] = proto.astype(np.float32, copy=False)
        else:
            # soft-DTW барицентр: отберём до n_max рядов, веса = prob
            rows = len(wells_c)
            if rows > n_max:
                # берём сэмпл с вероятностью пропорциональной prob
                weights = df_map.loc[df_map["cluster"] == c, "prob"].to_numpy()
                weights = weights / (weights.sum() + 1e-12)
                idx = np.random.choice(np.arange(rows), size=n_max, replace=False, p=weights)
                wells_sel = [wells_c[i] for i in idx]
                weights_sel = weights[idx]
                Xc, Mc = _collect_matrix(panel_long, wells_sel, channel, T)
                w = weights_sel
            else:
                w = df_map.loc[df_map["cluster"] == c, "prob"].to_numpy()
            # заполним пропуски кратко: по-временному медиана (только для входа в softdtw)
            Xfill = Xc.copy()
            col_med = np.nanmedian(Xfill, axis=0)
            # если где-то NaN, заменим на 0
            col_med = np.where(np.isfinite(col_med), col_med, 0.0)
            inds = np.where(~np.isfinite(Xfill))
            Xfill[inds] = np.take(col_med, inds[1])
            # tslearn ожидает [n_ts, T]
            proto = softdtw_barycenter(Xfill.astype(np.float64), weights=w.astype(np.float64))
            prototypes[int(c)] = proto.astype(np.float32, copy=False)

    return prototypes


# ============================ СВОДКА ============================

def summarize_clusters(df_map: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает сводку по кластерам: size, share, prob_median, is_noise.
    """
    n = len(df_map)
    if n == 0:
        return pd.DataFrame(columns=["cluster", "size", "share", "prob_median", "is_noise"])

    grp = df_map.groupby("cluster", sort=True)
    size = grp.size().rename("size")
    share = (size / float(n)).rename("share")
    prob_median = grp["prob"].median().rename("prob_median")
    out = pd.concat([size, share, prob_median], axis=1).reset_index()
    out["is_noise"] = out["cluster"] < 0
    return out


# ============================ ПРИМЕР ============================

if __name__ == "__main__":
    # Демонстрация на синтетике
    rng = np.random.default_rng(0)
    N = 800
    Z1 = rng.normal(loc=[-2, -2], scale=0.6, size=(N // 2, 2))
    Z2 = rng.normal(loc=[+2, +2], scale=0.6, size=(N // 2, 2))
    Z = np.vstack([Z1, Z2])
    wells = [f"W{i:04d}" for i in range(N)]

    if _HAVE_HDBSCAN:
        res = cluster_hdbscan(Z, wells)
        df_map = res["df_map"]
        df_map = assign_anomaly_scores(Z, df_map, labels=res["labels"], hdbscan_model=res["model"])
        print(df_map.head())
        print("silhouette:", res["silhouette"], "dbcv:", res["dbcv"])
        print(summarize_clusters(df_map).head())

    # GMM
    gmm = cluster_gmm_bic(Z, wells, k_range=(2, 6))
    print(gmm["bic_table"])
    print("best_k:", gmm["best_k"])
