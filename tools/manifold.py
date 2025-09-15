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
import umap
from fastdtw import fastdtw
import math
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    _TQDM = True
except Exception:
    _TQDM = False


# sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

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
) -> Tuple[np.ndarray, List[int], np.ndarray, Dict[str, object]]:
    """UMAP с уточнением FastDTW для ближайших пар.

    Алгоритм:
      1) Строим евклидову матрицу расстояний E в пространстве выровненных рядов (быстро).
      2) Для каждой точки берём k_refine ближайших по E и пересчитываем расстояния DTW (fastdtw).
      3) Получаем полную матрицу D: для уточнённых пар D=DTW, иначе D=E (масштаб сопоставим, но разный — это ок для UMAP).
      4) UMAP(metric='precomputed') на D.

    Если sample_size задан, берём случайную подвыборку для ускорения.

    Возвращает: (Z, sub_idx, D, info)
        sub_idx — индексы элементов исходного массива ``X``,
        попавшие в подвыборку (list[int]).
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
