"""
Production Behavior Manifold (PBM) — Шаг 2 (признаки) и Шаг 3 (Manifold)
-----------------------------------------------------------------------
Зависимости (pip): numpy, pandas, scikit-learn, umap-learn, fastdtw, tqdm (опц.)

Ожидается, что из Шага 1 у вас уже есть объект `out` с ключами:
  - out["panel_long"]: long-таблица по скважинам (после предобработки)
  - out["X"]: np.ndarray формы [n_wells, T, C] по каналам cfg.tensor_channels
  - out["wells_used"]: список имён скважин в том же порядке, что и ось 0 в X
  - out["tensor_channels"]: список имён каналов в X (например: ["r_oil_norm","wc","gor","dr_oil_norm"]) 

Этот файл добавляет:
  1) Шаг 2: компактные дескрипторы профилей (side features)
  2) Шаг 3: построение низкоразмерной "карты поведения" (UMAP) двумя способами:
       a) быстрый базовый (евклид на выровненных рядах)
       b) уточнённый с FastDTW, без полной O(N^2) — через доуточнение k ближайших пар

Примечание: библиотеку dtaidistance *не* используем. Для DTW берём fastdtw.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

# UMAP
try:
    import umap
except Exception as e:
    raise RuntimeError("Требуется пакет 'umap-learn'. Установите: pip install umap-learn")

# fastdtw
try:
    from fastdtw import fastdtw
except Exception:
    fastdtw = None
    warnings.warn("fastdtw не найден. DTW-вариант будет недоступен, останется базовый евклид.")

# tqdm (опционально)
try:
    from tqdm import tqdm
    _TQDM = True
except Exception:
    _TQDM = False


# ======================================================
# -------------------- ШАГ 2: ФИЧИ ---------------------
# ======================================================

def _nanpolyfit(y: np.ndarray, x: Optional[np.ndarray] = None, deg: int = 1) -> Tuple[float, ...]:
    """Лин. регрессия по ряду с NaN. Возвращает коэффициенты (последний — свободный член).
    deg=1 => (slope, intercept).
    """
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
    """Одно число: fn по всем скользящим окнам длины win, агрегация = среднее.
    Например, fn = np.std или пользовательская функция.
    """
    y = np.asarray(y, float)
    if win <= 0 or len(y) < win:
        return np.nan
    vals = []
    for i in range(len(y) - win + 1):
        seg = y[i:i+win]
        if np.isfinite(seg).sum() < max(2, win // 2):
            continue
        vals.append(fn(seg[np.isfinite(seg)]))
    return float(np.mean(vals)) if len(vals) else np.nan


def _slope_over_window(y: np.ndarray, win: int) -> float:
    if win <= 1 or len(y) < win:
        return np.nan
    slopes = []
    for i in range(len(y) - win + 1):
        seg = y[i:i+win]
        if np.isfinite(seg).sum() < max(2, win // 2):
            continue
        a, b = _nanpolyfit(seg, np.arange(win), deg=1)
        slopes.append(a)
    return float(np.mean(slopes)) if slopes else np.nan


def compute_side_features(panel_long: pd.DataFrame, T: int = 36) -> pd.DataFrame:
    """Строит компактные дескрипторы профиля по каждой скважине на горизонте [0..T).

    Требуемые колонки в panel_long: [well_name, t, r_oil_s, r_oil_norm, wc, gor, dr_oil_norm]
    Возвращает DataFrame: по одной строке на скважину.
    """
    need = {"well_name", "t", "r_oil_s", "r_oil_norm", "wc", "gor", "dr_oil_norm"}
    missing = need - set(panel_long.columns)
    if missing:
        raise ValueError(f"compute_side_features: не хватает колонок: {missing}")

    feats: List[pd.DataFrame] = []
    for w, g in panel_long.groupby("well_name", sort=False):
        g = g.copy()
        g = g[g["t"].between(0, T-1)]
        g = g.sort_values("t")
        # Векторы (длина может быть < T)
        oil = g["r_oil_s"].to_numpy(float)
        oil_norm = g["r_oil_norm"].to_numpy(float)
        wc = g["wc"].to_numpy(float)
        gor = g["gor"].to_numpy(float)
        d_oil = g["dr_oil_norm"].to_numpy(float)

        # Базовые величины
        peak = np.nanmax(oil) if oil.size else np.nan
        t_peak = int(np.nanargmax(oil)) if (oil.size and np.isfinite(peak)) else np.nan

        # Время до 50% и 20% от пика (decline speed)
        half = 0.5 * peak if np.isfinite(peak) else np.nan
        fifth = 0.2 * peak if np.isfinite(peak) else np.nan
        t_half = _first_index_where(oil, lambda y: np.isfinite(half) and (y <= half))
        t_20 = _first_index_where(oil, lambda y: np.isfinite(fifth) and (y <= fifth))

        # Усреднения по ранним/средним окнам
        m6 = int(min(6, len(oil)))
        m12 = int(min(12, len(oil)))
        early_mean6 = float(np.nanmean(oil_norm[:m6])) if m6 else np.nan
        early_mean12 = float(np.nanmean(oil_norm[:m12])) if m12 else np.nan
        early_trend6, _ = _nanpolyfit(oil_norm[:m6], np.arange(m6), 1) if m6 >= 3 else (np.nan, np.nan)
        mid_trend12 = _slope_over_window(oil_norm, 12)

        # Всплески/скачкообразность
        vol_doil_w6 = _rolling_stat(d_oil, win=6, fn=np.std)
        vol_doil_w12 = _rolling_stat(d_oil, win=12, fn=np.std)

        # Водоотдача и газовый фактор — уровни и тренды
        wc_mean12 = float(np.nanmean(wc[:m12])) if m12 else np.nan
        wc_trend12 = _slope_over_window(wc, 12)
        gor_trend12 = _slope_over_window(gor, 12)

        # Плато: сколько месяцев oil_norm >= 0.9 в начале профиля
        plateau_len = 0
        for v in oil_norm:
            if not np.isfinite(v) or v < 0.9:
                break
            plateau_len += 1

        # Системные пропуски
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


# ======================================================
# --------------- ШАГ 3: MANIFOLD (UMAP) ---------------
# ======================================================

@dataclass
class ManifoldConfig:
    # Какие каналы из тензора X брать для "формы"
    channels: Tuple[str, ...] = ("r_oil_norm", "wc")
    # DTW
    fastdtw_radius: int = 6
    k_refine: int = 40            # сколько ближайших по евклиду пар доуточнять DTW
    weights: Optional[Tuple[float, ...]] = (0.7, 0.3)  # веса каналов в многомерной метрике
    # UMAP
    n_neighbors: int = 30
    min_dist: float = 0.05
    n_components: int = 2
    random_state: int = 43


def _flatten_series_matrix(X: np.ndarray, channels_idx: Sequence[int]) -> np.ndarray:
    """Преобразует [N, T, C] -> [N, T*len(channels_idx)], NaN -> 0.
    Этот вектор используется ТОЛЬКО для быстрой евклидовой близости/UMAP-базы.
    """
    x = X[:, :, channels_idx]  # [N,T,C']
    N, T, C = x.shape
    x = np.transpose(x, (0, 2, 1)).reshape(N, C * T)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def embed_umap_euclid(X: np.ndarray, tensor_channels: Sequence[str], channels: Sequence[str],
                      n_neighbors: int = 30, min_dist: float = 0.05, n_components: int = 2,
                      random_state: int = 42) -> Tuple[np.ndarray, umap.UMAP]:
    """Быстрый базовый manifold: UMAP по евклиду на выровненных рядах.
    Возвращает (Z, umap_model).
    """
    ch_to_idx = {c: i for i, c in enumerate(tensor_channels)}
    idx = [ch_to_idx[c] for c in channels if c in ch_to_idx]
    X2 = _flatten_series_matrix(X, idx)
    model = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
        metric="euclidean", random_state=random_state
    )
    Z = model.fit_transform(X2)
    return Z, model


def _fastdtw_multichannel(A: np.ndarray, B: np.ndarray, weights: Optional[Sequence[float]] = None, radius: int = 6) -> float:
    """DTW для многоканальных рядов формы [T,C]. Использует L2 по каналам с весами.
    Отбрасывает позиции, где все каналы NaN. Заполняет оставшиеся NaN нулями.
    """
    if fastdtw is None:
        raise RuntimeError("fastdtw не установлен")
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Ожидается [T,C] для обоих рядов")
    # маска валидных позиций (хотя бы один канал не NaN)
    maskA = np.any(np.isfinite(A), axis=1)
    maskB = np.any(np.isfinite(B), axis=1)
    A = A[maskA]
    B = B[maskB]
    if A.size == 0 or B.size == 0:
        return float("inf")
    # NaN -> 0 (после выкидывания полностью пустых кадров)
    A = np.nan_to_num(A, nan=0.0)
    B = np.nan_to_num(B, nan=0.0)
    # взвешивание каналов
    if weights is not None:
        w = np.asarray(weights, float).reshape(1, -1)
        if w.shape[1] != A.shape[1]:
            raise ValueError("Длина weights должна совпадать с числом каналов")
        A = A * w
        B = B * w
    # метрика между точками — евклид в C-мерном пространстве
    dist, _ = fastdtw(A, B, radius=radius, dist=2)
    return float(dist)


def embed_umap_fastdtw(
    X: np.ndarray,
    tensor_channels: Sequence[str],
    channels: Sequence[str],
    cfg: Optional[ManifoldConfig] = None,
    sample_size: Optional[int] = None,
    candidate_knn: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, object]]:
    """UMAP с уточнением FastDTW для ближайших пар.

    Алгоритм:
      1) Строим евклидову матрицу расстояний E в пространстве выровненных рядов (быстро).
      2) Для каждой точки берём k_refine ближайших по E и пересчитываем расстояния DTW (fastdtw).
      3) Получаем полную матрицу D: для уточнённых пар D=DTW, иначе D=E (масштаб сопоставим, но разный — это ок для UMAP).
      4) UMAP(metric='precomputed') на D.

    Если sample_size задан, берём случайную подвыборку для ускорения.

    Возвращает: (Z, wells_sub, D, info)
    """
    if cfg is None:
        cfg = ManifoldConfig()

    rng = np.random.default_rng(cfg.random_state)
    N, T, Ctot = X.shape
    ch_to_idx = {c: i for i, c in enumerate(tensor_channels)}
    idx = [ch_to_idx[c] for c in channels if c in ch_to_idx]
    if len(idx) != len(channels):
        missing = [c for c in channels if c not in ch_to_idx]
        raise ValueError(f"В тензоре X нет каналов: {missing}")

    # Подвыборка
    all_idx = np.arange(N)
    if sample_size is not None and sample_size < N:
        sub_idx = rng.choice(all_idx, size=sample_size, replace=False)
    else:
        sub_idx = all_idx
    Xs = X[sub_idx][:, :, idx]  # [Ns, T, C']

    # База: евклид на флеттене
    Xflat = _flatten_series_matrix(Xs, list(range(len(idx))))  # [Ns, T*C']
    E = pairwise_distances(Xflat, metric="euclidean")  # [Ns,Ns]

    Ns = Xs.shape[0]
    D = E.copy()

    if fastdtw is None:
        warnings.warn("fastdtw недоступен — возвращаю UMAP по евклиду (без уточнения DTW)")
    else:
        # Список пар для уточнения — top-k по E для каждой точки
        k = cfg.k_refine if candidate_knn is None else candidate_knn
        nn = NearestNeighbors(n_neighbors=min(k+1, Ns), metric="euclidean")
        nn.fit(Xflat)
        dists, inds = nn.kneighbors(Xflat, return_distance=True)
        # inds[:,0] — сама точка; начнём с 1..
        pairs = set()
        for i in range(Ns):
            for j in inds[i, 1:]:
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                if a != b:
                    pairs.add((a, b))
        pairs = list(pairs)

        it = pairs
        if _TQDM:
            it = tqdm(pairs, desc=f"Recompute DTW for {len(pairs)} pairs (radius={cfg.fastdtw_radius})")

        for (i, j) in it:
            d = _fastdtw_multichannel(Xs[i], Xs[j], weights=cfg.weights, radius=cfg.fastdtw_radius)
            if not math.isfinite(d):
                continue
            D[i, j] = d
            D[j, i] = d

    # UMAP на предвычисленной матрице
    model = umap.UMAP(
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        n_components=cfg.n_components,
        metric="precomputed",
        random_state=cfg.random_state,
    )
    Z = model.fit_transform(D)

    info = {
        "Ns": int(Ns),
        "channels": list(channels),
        "k_refine": int(cfg.k_refine),
        "fastdtw_radius": int(cfg.fastdtw_radius),
        "umap_params": {
            "n_neighbors": cfg.n_neighbors,
            "min_dist": cfg.min_dist,
            "n_components": cfg.n_components,
            "random_state": cfg.random_state,
        }
    }
    return Z, sub_idx.tolist(), D, info


# ======================================================
# ======================================================

