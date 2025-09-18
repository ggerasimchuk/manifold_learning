
"""
Зачем нужен модуль
----------
Строит двумерные встраивания профилей скважин из тензора формы [N, T, C].
Два метода (опционально):
1) `embed_umap_euclid`: UMAP по евклиду на сплющенных рядах. (быстро)
2) `embed_umap_fastdtw`: UMAP на разреженном kNN-графе с рёбрами,
   длины которых уточнены многоканальным FastDTW. Масштабирование
   расстояний по медиане. Параллельный пересчёт DTW.
   (точнее)

Ключевые идеи
-------------
- Разделение быстрого поиска кандидатов (евклид) и точного пересчёта (DTW).
- Разреженный граф вместо плотной матрицы NxN. Память O(N·k).
- Учет временных сдвигов/растяжений через DTW. Близость по форме, а не
  только по значениям.
- Нормировка шкалы расстояний перед UMAP повышает устойчивость.

Входные данные
--------------
- `X: np.ndarray` формы [N, T, C].
  Массив временных рядов по скважинам.
  C — число исходных каналов (признаков на каждый момент времени).
  N — число объектов (скважин). Каждая скважина — отдельный ряд.
  T — число временных шагов (месяцев, дней и т. д.).

- `tensor_channels: Sequence[str]`.
  Список имён каналов длиной C, соответствующий последней оси X.
- `channels: Sequence[str]`.
  Имена поднабора каналов (C' ≤ C) для анализа.

Обработка пропусков
-------------------
- Евклид: перед flatten все NaN → 0.
- DTW: моменты времени, где во всех каналах NaN, удаляются. Частичные NaN → 0.
- Нулевой стандарт при z-нормализации: ряд не масштабируется.

Функции
-------
embed_umap_euclid(X, tensor_channels, channels,
                  n_neighbors=20, min_dist=0.1, n_components=2,
                  random_state=42, znorm_before_flatten=False)
    Быстрый базовый UMAP на flatten-представлении.
    Параметры:
      - `znorm_before_flatten`: опциональная z-нормализация по времени
        для каждого канала и ряда до сплющивания.
    Возвращает:
      - `Z: np.ndarray [N, n_components]`
      - `umap_model: umap.UMAP`

embed_umap_fastdtw(X, tensor_channels, channels,
                   cfg=None, sample_size=None, candidate_knn=None)
    UMAP на предвычисленном разреженном графе расстояний.
    Этапы:
      1) Подвыборка Ns из N (если задано).
      2) kNN-граф по евклиду на flatten (k = max(k_refine, n_neighbors)).
      3) Уточнение длин рёбер графа многоканальным FastDTW
         с опциональными весами каналов (weights нормируются до суммы 1).
         Параллелизм через joblib (`n_jobs`).
      4) Масштабирование всех ненулевых конечных расстояний по медиане.
      5) UMAP с `metric="precomputed"` на CSR-матрице.
    Параметры:
      - `cfg: ManifoldConfig` (см. ниже).
      - `sample_size`: размер подвыборки. По умолчанию берётся из `cfg`.
      - `candidate_knn`: число соседей в графе кандидатов. По умолчанию
        `max(cfg.k_refine, cfg.n_neighbors)`.
    Возвращает:
      - `Z: np.ndarray [Ns, n_components]`
      - `sub_idx: list[int]` индексы выбранных рядов из диапазона [0..N-1]
      - `D: scipy.sparse.csr_matrix` разреженная симметричная матрица
        расстояний для UMAP
      - `info: dict` диагностическая информация
      - `umap_model: umap.UMAP`

Класс конфигурации
------------------
ManifoldConfig:
- `channels: Sequence[str]` — имена каналов для анализа.
- `n_neighbors: int` — параметр UMAP (локальность графа).
- `min_dist: float` — параметр UMAP (компактность кластеров).
- `n_components: int` — размерность встраивания, обычно 2.
- `random_state: int` — воспроизводимость.
- `k_refine: int` — целевое число уточняемых DTW-рёбер на вершину.
- `fastdtw_radius: int` — радиус приближения FastDTW.
- `weights: Optional[Sequence[float]]` — веса каналов для DTW.
  Длина равна C'. Нормируются до суммы 1.
- `sample_size: int` — подвыборка для DTW-UMAP.
- `n_jobs: int` — параллелизм DTW (`-1` = все ядра, `1` = без параллелизма).
- `znorm_before_flatten: bool` — включить z-нормализацию перед евклидом.

Возвращаемые поля info
----------------------
- `method`: строка, идентифицирующая метод.
- `Ns`: число рядов в подвыборке.
- `candidate_knn`: фактический k для kNN-графа.
- `n_jobs`: число потоков/процессов для DTW.
- `pairs_total`: число рёбер-кандидатов.
- `pairs_replaced`: сколько рёбер пересчитано DTW.
- `replace_ratio`: доля пересчитанных рёбер.
- `sparse_graph_nnz`: число ненулевых элементов в CSR после уточнения.
- `edges_undirected`: число рёбер с учётом симметрии (nnz // 2).
- `fastdtw_available`: наличие библиотеки fastdtw.
- `median_euclid_before`: медиана евклидовых длин рёбер до пересчёта.
- `median_after_norm`: медиана после нормировки (обычно 1.0).
- `znorm_before_flatten`: использовалась ли z-нормализация.

Сложность
---------
- Память: O(Ns · k). Без плотных матриц NxN.
- Время:
  - Построение kNN-графа по евклиду: примерно O(Ns · k).
  - Уточнение DTW: O(числа рёбер) вызовов FastDTW.
  - Масштабируется по `n_jobs`.

Зависимости
-----------
Обязательные: `numpy`, `scipy`, `scikit-learn`, `umap-learn`.
Опциональные: `joblib` (параллелизм DTW), `fastdtw` (уточнение DTW), `tqdm` (прогресс).
При отсутствии `fastdtw` используется евклид без уточнения.

Особенности и ограничения
-------------------------
- Имя каждого канала в `channels` обязано присутствовать в `tensor_channels`.
- При `candidate_knn > Ns-1` k ограничивается `Ns-1`.
- Если ряд после фильтра полных NaN-позиций пуст, DTW даёт `inf`, ребро не заменяется.
- Z-нормализация применяется отдельно к каждому ряду и каналу по времени.
- Для очень больших Ns выбирайте умеренный k и/или увеличивайте `sample_size` осторожно.

Примеры
-------
Быстрый евклидовый UMAP:
Z_euclid, umap_e = embed_umap_euclid(
...     X, tensor_channels, channels=("r_oil_norm","wc"),
...     n_neighbors=20, min_dist=0.1, random_state=42,
...     znorm_before_flatten=True
... )

UMAP с FastDTW:
from manifold import ManifoldConfig, embed_umap_fastdtw
cfg = ManifoldConfig(
...     channels=("r_oil_norm","wc"),
...     n_neighbors=20,
...     k_refine=30,
...     n_jobs=-1,
...     znorm_before_flatten=True
... )
Z_dtw, sub_idx, D, info, umap_d = embed_umap_fastdtw(
...     X, tensor_channels, channels=("r_oil_norm","wc"),
...     cfg=cfg, sample_size=None
... )

Диагностика:
info["pairs_replaced"], info["replace_ratio"], D.nnz

Рекомендации по настройке
-------------------------
- Начните с `n_neighbors=20`, `k_refine=30`, `min_dist=0.1`.
- Если кластеры распадаются, повысите `n_neighbors`.
- Если DTW-этап долгий, уменьшите `k_refine` или примените подвыборку (`sample_size`).
- Для разномасштабных каналов используйте `weights` или включите `znorm_before_flatten`.

Безопасность воспроизводимости
------------------------------
- Фиксируйте `random_state` в `ManifoldConfig`.
- Логируйте `info` для последующего аудита качества графа.

"""

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Dict, List

import math
import numpy as np
import scipy.sparse as sp

# опционально: прогресс-бар
try:
    from tqdm import tqdm  # type: ignore
    _TQDM = True
except Exception:
    _TQDM = False

# опционально: параллельный пересчёт DTW
try:
    from joblib import Parallel, delayed  # type: ignore
    _JOBLIB = True
except Exception:
    _JOBLIB = False

# UMAP
try:
    import umap  # type: ignore
except Exception as e:
    raise ImportError("Пакет umap-learn не найден. Установите `pip install umap-learn`.") from e

# евклидовый kNN-граф
from sklearn.neighbors import NearestNeighbors

# опционально: fastdtw
try:
    from fastdtw import fastdtw  # type: ignore
    _FASTDWT_AVAILABLE = True
except Exception:
    _FASTDWT_AVAILABLE = False


# =========================
# Конфиг
# =========================
@dataclass
class ManifoldConfig:
    # каналы для анализа (имена должны быть в tensor_channels)
    channels: Sequence[str] = ("r_oil_norm",)
    # UMAP
    n_neighbors: int = 20
    min_dist: float = 0.1
    n_components: int = 2
    random_state: int = 42
    # уточнение DTW
    k_refine: int = 30          # сколько рёбер на вершину пересчитывать DTW
    fastdtw_radius: int = 1
    weights: Optional[Sequence[float]] = None  # веса каналов в многоканальном DTW
    # подвыборка
    sample_size: int = 800
    # параллелизм для DTW (-1 = все ядра, 1 = без параллелизма)
    n_jobs: int = -1
    # доп. опции
    znorm_before_flatten: bool = False  # (8) z-нормализация каждого ряда перед flatten


# =========================
# Утилиты
# =========================
def _check_and_get_channel_indices(
    tensor_channels: Sequence[str],
    channels: Sequence[str],
) -> List[int]:
    name_to_idx = {c: i for i, c in enumerate(tensor_channels)}
    missing = [c for c in channels if c not in name_to_idx]
    if missing:
        raise KeyError(f"Каналов нет в tensor_channels: {missing}. Доступно: {list(tensor_channels)}")
    return [name_to_idx[c] for c in channels]


def _select_channels(X: np.ndarray, ch_idx: Sequence[int]) -> np.ndarray:
    """
    X: [N, T, C_total] -> [N, T, C'] по выбранным каналам.
    """
    return X[:, :, ch_idx]


def _znorm_timewise(X_sel: np.ndarray) -> np.ndarray:
    """
    Z-нормализация по оси времени для каждого (n, c).
    NaN игнорируются при оценке среднего и std. Нулевой std -> оставляем как есть.
    """
    X = X_sel.astype(float, copy=True)
    # mean/std по времени
    mu = np.nanmean(X, axis=1, keepdims=True)
    sigma = np.nanstd(X, axis=1, keepdims=True)
    # безопасный делитель
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    X = (X - mu) / sigma_safe
    return X


def _flatten_rows(X_sel: np.ndarray, znorm: bool = False) -> np.ndarray:
    """
    [N, T, C'] -> [N, T*C'] с опцией z-нормализации и заменой NaN на 0.
    """
    if znorm:
        X_sel = _znorm_timewise(X_sel)
    N, T, C = X_sel.shape
    Xf = X_sel.reshape(N, T * C)
    Xf = np.asarray(Xf, dtype=float)
    Xf[np.isnan(Xf)] = 0.0
    return Xf


def _row_subsample(N: int, sample_size: Optional[int], random_state: int) -> np.ndarray:
    """
    Возвращает индексы подвыборки [Ns], случайно без повторов.
    Если sample_size None или >= N — возвращает np.arange(N).
    """
    if (sample_size is None) or (sample_size >= N):
        return np.arange(N, dtype=int)
    rs = np.random.RandomState(random_state)
    return np.sort(rs.choice(N, size=sample_size, replace=False).astype(int))


def _normalize_weights(weights: Sequence[float], C: int) -> np.ndarray:
    """
    (7) Проверка длины и нормировка весов до суммы 1.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != C:
        raise ValueError(f"Длина weights должна равняться числу каналов ({C}), получено {w.shape}")
    s = w.sum()
    if not (np.isfinite(s) and s > 0):
        raise ValueError("Сумма weights должна быть конечной и положительной.")
    return w / s


def _fastdtw_multichannel(
    A: np.ndarray,  # [T, C']
    B: np.ndarray,  # [T, C']
    weights: Optional[Sequence[float]] = None,
    radius: int = 1,
) -> float:
    """
    FastDTW по многоканальному ряду. На каждом шаге расстояние — L2 по каналам (с весами).
    Пропуски: если в момент времени во всех каналах NaN — шаг выкидываем. Иначе NaN->0.
    """
    if not _FASTDWT_AVAILABLE:
        # если fastdtw нет, вернём евклид в flatten как деградацию
        a = A.copy()
        b = B.copy()
        a[np.isnan(a)] = 0.0
        b[np.isnan(b)] = 0.0
        return float(np.linalg.norm(a.ravel() - b.ravel()))

    # фильтр шагов, где все каналы NaN
    maskA = ~np.all(np.isnan(A), axis=1)
    maskB = ~np.all(np.isnan(B), axis=1)
    A1 = A[maskA]
    B1 = B[maskB]

    if A1.shape[0] == 0 or B1.shape[0] == 0:
        return float("inf")

    A1 = A1.copy()
    B1 = B1.copy()
    A1[np.isnan(A1)] = 0.0
    B1[np.isnan(B1)] = 0.0

    if weights is not None:
        # здесь weights уже нормированы снаружи
        w = np.asarray(weights, dtype=float)
        A1 = A1 * w
        B1 = B1 * w

    # расстояние между векторами момента времени
    def _v2v(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return np.linalg.norm(u - v)

    dist, _path = fastdtw(A1, B1, radius=radius, dist=_v2v)
    return float(dist)


# =========================
# Встраивания
# =========================
def embed_umap_euclid(
    X: np.ndarray,
    tensor_channels: Sequence[str],
    channels: Sequence[str],
    n_neighbors: int = 20,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
    znorm_before_flatten: bool = False,
) -> Tuple[np.ndarray, "umap.UMAP"]:
    """
    Базовый UMAP по евклиду на flatten([T,C]).
    Возврат: (Z, umap_model)
    """
    ch_idx = _check_and_get_channel_indices(tensor_channels, channels)
    X_sel = _select_channels(X, ch_idx)              # [N, T, C']
    X_flat = _flatten_rows(X_sel, znorm=znorm_before_flatten)  # (8)

    model = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        n_components=int(n_components),
        metric="euclidean",
        random_state=int(random_state),
    )
    Z = model.fit_transform(X_flat)
    return Z, model


def embed_umap_fastdtw(
    X: np.ndarray,
    tensor_channels: Sequence[str],
    channels: Sequence[str],
    cfg: Optional[ManifoldConfig] = None,
    sample_size: Optional[int] = None,
    candidate_knn: Optional[int] = None,
) -> Tuple[np.ndarray, List[int], sp.csr_matrix, Dict[str, object], "umap.UMAP"]:
    """
    UMAP на предвычисленном графе расстояний:
      1) kNN-граф по евклиду на flatten
      2) пересчёт длин рёбер графа с FastDTW для выбранных пар
      3) нормировка по медиане и запуск UMAP(metric='precomputed') на sparse-матрице

    Возврат: (Z, sub_idx, dist_csr, info, umap_model)   # (5) возвращаем модель
    """
    if cfg is None:
        cfg = ManifoldConfig()

    # выбор каналов
    ch_idx = _check_and_get_channel_indices(tensor_channels, channels)
    X_sel = _select_channels(X, ch_idx)  # [N, T, C']
    Cprime = len(ch_idx)

    # подготовка и проверка весов (7)
    weights_norm = None
    if cfg.weights is not None:
        weights_norm = _normalize_weights(cfg.weights, Cprime)

    # подвыборка (5: sample_size читаем из cfg, если None)
    N = X_sel.shape[0]
    sub_idx = _row_subsample(N, sample_size if sample_size is not None else cfg.sample_size, cfg.random_state)
    Xs = X_sel[sub_idx]  # [Ns, T, C']
    Ns = Xs.shape[0]

    # базовая евклидовая близость по flatten, с опцией z-нормализации (8)
    X_flat = _flatten_rows(Xs, znorm=cfg.znorm_before_flatten)  # [Ns, T*C']

    # связка параметров кандидатов для уточнения (2)
    k_base = candidate_knn if candidate_knn is not None else max(int(cfg.k_refine), int(cfg.n_neighbors))
    k_base = int(min(max(1, k_base), Ns - 1))

    # kNN-граф (без self), симметризация
    nn = NearestNeighbors(n_neighbors=k_base, metric="euclidean")
    nn.fit(X_flat)
    G = nn.kneighbors_graph(X_flat, n_neighbors=k_base, mode="distance", include_self=False)  # CSR
    D = G.maximum(G.T).tolil()  # LIL для модификации
    Deuclid = D.tocsr(copy=True)  # (6) сохраним евклидовые длины для диагностики

    # перечень пар рёбер для уточнения
    rows, cols = D.nonzero()
    pairs = {(int(i), int(j)) if i < j else (int(j), int(i)) for i, j in zip(rows, cols) if i != j}
    pairs = sorted(pairs)

    # FastDTW-уточнение длин рёбер
    replaced = 0
    def _recompute_pair(p):
        i, j = p
        d = _fastdtw_multichannel(Xs[i], Xs[j], weights=weights_norm, radius=cfg.fastdtw_radius)
        return i, j, d

    if _FASTDWT_AVAILABLE and len(pairs) > 0:
        if _JOBLIB and cfg.n_jobs != 1:
            results = Parallel(n_jobs=cfg.n_jobs, prefer="processes")(
                delayed(_recompute_pair)(p) for p in pairs
            )
            for i, j, d in results:
                if math.isfinite(d):
                    D[i, j] = d
                    D[j, i] = d
                    replaced += 1
        else:
            it = tqdm(pairs, desc=f"DTW-edges {len(pairs)}", mininterval=0.2) if _TQDM else pairs
            for (i, j) in it:
                d = _fastdtw_multichannel(Xs[i], Xs[j], weights=weights_norm, radius=cfg.fastdtw_radius)
                if math.isfinite(d):
                    D[i, j] = d
                    D[j, i] = d
                    replaced += 1
    # если fastdtw нет — остаются евклидовые длины

    # в CSR и нормировка по медиане ненулевых и конечных значений (3)
    D = D.tocsr()
    med_before = None
    if Deuclid.nnz > 0:
        data0 = Deuclid.data
        finite0 = np.isfinite(data0) & (data0 > 0)
        if np.any(finite0):
            med_before = float(np.median(data0[finite0]))

    med_after = None
    if D.nnz > 0:
        data = D.data
        finite = np.isfinite(data) & (data > 0)
        if np.any(finite):
            med = np.median(data[finite])
            if np.isfinite(med) and med > 0:
                D.data = data / med
                med_after = 1.0  # после масштабирования медиана = 1
            else:
                med_after = float(med) if np.isfinite(med) else None

    # UMAP(metric='precomputed') на разреженной матрице расстояний
    model = umap.UMAP(
        n_neighbors=int(cfg.n_neighbors),
        min_dist=float(cfg.min_dist),
        n_components=int(cfg.n_components),
        metric="precomputed",
        random_state=int(cfg.random_state),
    )
    Z = model.fit_transform(D)

    # (6) расширенная диагностика
    undirected_edges = int(D.nnz // 2)
    info: Dict[str, object] = {
        "method": "UMAP+FastDTW-sparse",
        "Ns": int(Ns),
        "candidate_knn": int(k_base),
        "n_jobs": int(cfg.n_jobs),
        "pairs_total": int(len(pairs)),
        "pairs_replaced": int(replaced),
        "replace_ratio": float(replaced / max(1, len(pairs))),
        "sparse_graph_nnz": int(D.nnz),
        "edges_undirected": undirected_edges,
        "fastdtw_available": bool(_FASTDWT_AVAILABLE),
        "median_euclid_before": med_before,
        "median_after_norm": med_after,
        "znorm_before_flatten": bool(cfg.znorm_before_flatten),
    }
    # (5) возвращаем также модель
    return Z, sub_idx.tolist(), D, info, model
