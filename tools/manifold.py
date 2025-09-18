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
    n_neighbors=20, min_dist=0.1, n_components=2, random_state=42,
    znorm_before_flatten=False) -> (Z, umap_model)

embed_umap_fastdtw(X, tensor_channels, channels, cfg: ManifoldConfig,
    sample_size=None, candidate_knn=None) -> (Z, sub_idx, D_sparse, info, umap_model)
    Идея:
      1) На подмножестве рядов строим евклидов kNN-граф кандидатов.
      2) Пересчитываем верхушку рёбер (top-k_refine на вершину) многоканальным FastDTW
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
- `n_jobs: int` — параллельность в joblib. -1 — все ядра.
- `autotune: bool` — подбирать n_neighbors/min_dist сеточным поиском.
- `znorm_before_flatten: bool` — z-норм перед flatten для евклида.

Пример
------
Евклид:
from manifold import embed_umap_euclid
Z_euclid, umap_e = embed_umap_euclid(
...     X, tensor_channels, channels=("r_oil_norm","wc"),
...     n_neighbors=20
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
Все случайности определяются `random_state`.

"""

from typing import Sequence, Optional, Tuple, List, Dict
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

import math
try:
    import umap
except Exception:
    raise ImportError("Пакет umap-learn не найден. Установите `pip install umap-learn`.")

try:
    from fastdtw import fastdtw
    _FASTDWT_AVAILABLE = True
except Exception:
    _FASTDWT_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except Exception:
    _JOBLIB = False

try:
    from tqdm import tqdm
    _TQDM = True
except Exception:
    _TQDM = False


# =========================
# Утилиты
# =========================
def _check_and_get_channel_indices(tensor_channels: Sequence[str], channels: Sequence[str]) -> List[int]:
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
    NaN игнорируются при оценке среднего и std.
    FIX: Уточнение — при нулевом std ряд центрируется к нулю (становится нулём), а не «не масштабируется».
    """
    X = X_sel.astype(float, copy=True)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sigma = np.nanstd(X, axis=1, keepdims=True)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    X = (X - mu) / sigma_safe
    return X


def _flatten_rows(X_sel: np.ndarray, znorm: bool = False) -> np.ndarray:
    """
    [N, T, C'] -> [N, T*C'] с опцией z-нормализации и заменой NaN на 0.
    """
    X = X_sel
    if znorm:
        X = _znorm_timewise(X_sel)
    X = X.reshape(X.shape[0], -1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _row_subsample(N: int, sample_size: Optional[int], random_state: int) -> np.ndarray:
    """
    Индексы подвыборки по объектам.
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
    Требует fastdtw. Если fastdtw недоступен, эта функция не должна вызываться.
    """
    if not _FASTDWT_AVAILABLE:
        raise RuntimeError("fastdtw недоступен. Пересчёт DTW-ребер отключён.")

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
        w = np.asarray(weights, dtype=float)
        w_sqrt = np.sqrt(w)
        A1 = A1 * w_sqrt
        B1 = B1 * w_sqrt

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
    ch_idx = _check_and_get_channel_indices(tensor_channels, channels)
    X_sel = _select_channels(X, ch_idx)              # [N, T, C']
    X_flat = _flatten_rows(X_sel, znorm=znorm_before_flatten)

    model = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        n_components=int(n_components),
        metric="euclidean",
        random_state=int(random_state),
    )
    Z = model.fit_transform(X_flat)
    return Z, model


class ManifoldConfig:
    def __init__(
        self,
        channels: Optional[Sequence[str]] = None,
        n_neighbors: int = 20,
        min_dist: float = 0.1,
        n_components: int = 2,
        random_state: int = 42,
        k_refine: int = 30,
        fastdtw_radius: int = 1,
        weights: Optional[Sequence[float]] = None,
        sample_size: Optional[int] = None,
        n_jobs: int = -1,
        autotune: bool = True,
        znorm_before_flatten: bool = False,
    ):
        self.channels = channels
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.k_refine = int(k_refine)
        self.fastdtw_radius = int(fastdtw_radius)
        self.weights = weights
        self.sample_size = sample_size
        self.n_jobs = int(n_jobs)
        self.autotune = bool(autotune)
        self.znorm_before_flatten = bool(znorm_before_flatten)


def embed_umap_fastdtw(
    X: np.ndarray,
    tensor_channels: Sequence[str],
    channels: Sequence[str],
    cfg: Optional[ManifoldConfig] = None,
    sample_size: Optional[int] = None,
    candidate_knn: Optional[int] = None,
) -> Tuple[np.ndarray, List[int], sp.csr_matrix, Dict[str, object], "umap.UMAP"]:
    if cfg is None:
        cfg = ManifoldConfig()

    ch_idx = _check_and_get_channel_indices(tensor_channels, channels)
    X_sel = _select_channels(X, ch_idx)  # [N, T, C']
    Cprime = len(ch_idx)

    weights_norm = None
    if cfg.weights is not None:
        weights_norm = _normalize_weights(cfg.weights, Cprime)

    N = X_sel.shape[0]
    sub_idx = _row_subsample(N, sample_size if sample_size is not None else cfg.sample_size, cfg.random_state)
    Xs = X_sel[sub_idx]
    Ns = Xs.shape[0]

    X_flat = _flatten_rows(Xs, znorm=cfg.znorm_before_flatten)

    k_base_req = candidate_knn if candidate_knn is not None else max(int(cfg.k_refine), int(cfg.n_neighbors))
    k_base = int(min(max(1, k_base_req), max(1, Ns - 1)))

    nn = NearestNeighbors(n_neighbors=k_base, metric="euclidean")
    nn.fit(X_flat)
    G = nn.kneighbors_graph(X_flat, n_neighbors=k_base, mode="distance")  # CSR
    # NOTE: distances равны при наличии обоих направлений; maximum/minimum эквивалентны
    Deuclid = G.maximum(G.T).tocsr()  # симметризация

    # top-k_refine пар на строку
    if int(cfg.k_refine) < k_base:
        pairs_set = set()
        indptr = Deuclid.indptr
        indices = Deuclid.indices
        data = Deuclid.data
        k_ref = int(max(1, int(cfg.k_refine)))
        for i in range(Ns):
            s_i = indptr[i]
            e_i = indptr[i + 1]
            js = indices[s_i:e_i]
            ds = data[s_i:e_i]
            if ds.size == 0:
                continue
            order = np.argsort(ds)[:k_ref]
            for pos in order:
                j = int(js[pos])
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                pairs_set.add((a, b))
        pairs = sorted(pairs_set)
    else:
        rows, cols = Deuclid.nonzero()
        pairs = {(int(i), int(j)) if i < j else (int(j), int(i)) for i, j in zip(rows, cols) if i != j}
        pairs = sorted(pairs)

    D = Deuclid.tolil()
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

    # === НОРМАЛИЗАЦИЯ ПО МЕДИАНЕ ===
    med_before = None
    if Deuclid.nnz > 0:
        data0 = Deuclid.data
        finite0 = np.isfinite(data0) & (data0 > 0)
        if np.any(finite0):
            med_before = float(np.median(data0[finite0]))

    # FIX: сохранить фактический масштаб всех рёбер до нормировки и использовать далее
    D = D.tocsr()
    med_all = None
    if D.nnz > 0:
        finite = np.isfinite(D.data) & (D.data > 0)
        if np.any(finite):
            med_all = float(np.median(D.data[finite]))
            if np.isfinite(med_all) and med_all > 0:
                D.data = D.data / med_all  # теперь медиана == 1.0

    # --- САНИТИЗАЦИЯ + РЕМОНТ ГРАФА ---
    eps = 1e-3
    C = D.tocoo(copy=True)
    off = C.row != C.col
    bad = (~np.isfinite(C.data)) | (C.data <= 0)
    fix = off & bad
    if np.any(fix):
        C.data[fix] = eps
    D = C.tocsr()

    # диагональ нулевая
    D.setdiag(0.0)
    D.eliminate_zeros()

    # Если где-то степень < 2 — дозаполняем 2 ближайшими евклидовыми соседями
    added_edges_scaled = []
    row_nnz = np.diff(D.indptr)
    bad_rows = np.where(row_nnz < 2)[0]
    if bad_rows.size > 0:
        n2 = max(1, min(8, Ns - 1))
        nn2 = NearestNeighbors(n_neighbors=n2, metric="euclidean").fit(X_flat)
        dists2, idx2 = nn2.kneighbors(X_flat[bad_rows], return_distance=True)
        for r, i in enumerate(bad_rows):
            take = []
            for j, d in zip(idx2[r], dists2[r]):
                if j == i:
                    continue
                # FIX: приводим к той же шкале, что и D после нормировки, делим на med_all
                scale = (med_all if (med_all is not None and np.isfinite(med_all) and med_all > 0) else 1.0)
                take.append((int(j), float(max(d / scale, eps))))
                if len(take) >= 2:
                    break
            for j, d in take:
                D[i, j] = d
                D[j, i] = d
                added_edges_scaled.append(d)
        D = D.tocsr()
        row_nnz = np.diff(D.indptr)

    k_min = int(row_nnz.min()) if row_nnz.size else 0

    # --- ФОЛБЭК ---
    used_fallback = False
    tried: List[Tuple[int, float, float]] = []
    umap_n_eff = int(cfg.n_neighbors)
    umap_md_eff = float(cfg.min_dist)

    if Ns < 2 or k_min < 2:
        used_fallback = True
        k_fb = int(min(max(1, int(cfg.n_neighbors)), max(1, Ns - 1)))
        umap_n_eff = k_fb
        umap_md_eff = float(cfg.min_dist)
        model = umap.UMAP(
            n_neighbors=k_fb,
            min_dist=float(cfg.min_dist),
            n_components=int(cfg.n_components),
            metric="euclidean",
            random_state=int(cfg.random_state),
        )
        Z = model.fit_transform(X_flat)
    else:
        # --- АВТОТЮНИНГ параметров UMAP ---
        def _candidate_grid(Ns_: int, k_min_: int, cfg_) -> Tuple[List[int], List[float]]:
            base = {
                max(10, int(round((Ns_ ** 0.5) / 2))),
                int(round(Ns_ ** 0.5)),
                min(50, int(round((Ns_ ** 0.5) * 1.5))),
                int(cfg_.n_neighbors),
            }
            nn_c = sorted({max(2, min(int(b), k_min_)) for b in base if 2 <= b <= k_min_})
            if not nn_c:
                nn_c = [max(2, min(int(cfg_.n_neighbors), k_min_))]
            md_c = sorted({float(cfg_.min_dist), 0.05, 0.10, 0.30})
            md_c = [float(min(0.99, max(0.0, x))) for x in md_c]
            return nn_c, md_c

        autotune = bool(getattr(cfg, "autotune", True))
        if autotune:
            nn_cand, md_cand = _candidate_grid(Ns, k_min, cfg)

            # FIX: предвычисляем соседства по D для Jaccard один раз
            k_overlap = int(max(2, min(10, k_min)))
            indptr = D.indptr
            indices = D.indices
            data = D.data
            neigh_D = []
            for i in range(Ns):
                s_i = indptr[i]; e_i = indptr[i + 1]
                js = indices[s_i:e_i]
                ds = data[s_i:e_i]
                if ds.size == 0:
                    neigh_D.append(set())
                    continue
                order = np.argsort(ds)[:k_overlap]
                neigh_D.append(set(js[order].tolist()))

            def _jaccard_knn_overlap_precomputed(Z_emb, k):
                Ns_loc = Z_emb.shape[0]
                k_eff = int(max(1, min(int(k), Ns_loc - 1)))
                nn_emb = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(Z_emb)
                _, idx_emb = nn_emb.kneighbors(Z_emb, return_distance=True)
                vals = []
                for i in range(Ns_loc):
                    row = idx_emb[i]
                    row2 = [int(x) for x in row if int(x) != i][:k_eff]
                    b = set(row2)
                    a = neigh_D[i]
                    if not a and not b:
                        continue
                    u = len(a | b)
                    if u == 0:
                        continue
                    vals.append(len(a & b) / u)
                return float(np.mean(vals)) if len(vals) else float("nan")

            best = None
            best_pack = None
            for nn_ in nn_cand:
                for md_ in md_cand:
                    model_try = umap.UMAP(
                        n_neighbors=int(nn_),
                        min_dist=float(md_),
                        n_components=int(cfg.n_components),
                        metric="precomputed",
                        random_state=int(cfg.random_state),
                    )
                    try:
                        Z_try = model_try.fit_transform(D)
                        t = float(_jaccard_knn_overlap_precomputed(Z_try, k=k_overlap))
                        score = t
                    except Exception:
                        Z_try, t, score = None, float("nan"), -1.0
                    tried.append((nn_, md_, t))
                    if best is None or score > best:
                        best = score
                        best_pack = (model_try, Z_try, nn_, md_)
            if best_pack is None:
                umap_n = int(max(2, min(int(cfg.n_neighbors), k_min)))
                umap_md = float(cfg.min_dist)
                model = umap.UMAP(
                    n_neighbors=umap_n,
                    min_dist=umap_md,
                    n_components=int(cfg.n_components),
                    metric="precomputed",
                    random_state=int(cfg.random_state),
                )
                Z = model.fit_transform(D)
                umap_n_eff, umap_md_eff = umap_n, umap_md
            else:
                model, Z, umap_n, umap_md = best_pack
                umap_n_eff, umap_md_eff = int(umap_n), float(umap_md)
        else:
            umap_n = int(max(2, min(int(cfg.n_neighbors), k_min)))
            umap_md = float(cfg.min_dist)
            model = umap.UMAP(
                n_neighbors=umap_n,
                min_dist=umap_md,
                n_components=int(cfg.n_components),
                metric="precomputed",
                random_state=int(cfg.random_state),
            )
            Z = model.fit_transform(D)
            umap_n_eff, umap_md_eff = umap_n, umap_md

    # диагностика
    row_nnz = np.diff(D.indptr) if D.indptr.size else np.array([], dtype=int)
    undirected_edges = int(D.nnz // 2)
    info: Dict[str, object] = {
        "method": ("UMAP-euclid-fallback" if used_fallback else "UMAP+FastDTW-sparse"),
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
        # FIX: сохраняем фактическую медиану до нормировки для трассировки
        "median_all_before_norm": med_all,
        "eps_after_norm": 1e-3,
        "znorm_before_flatten": bool(cfg.znorm_before_flatten),
        "row_degree_min": int(row_nnz.min()) if row_nnz.size else 0,
        "row_degree_median": int(np.median(row_nnz)) if row_nnz.size else 0,
        "autotune": False if used_fallback else bool(getattr(cfg, "autotune", True)),
        "umap_n_neighbors_effective": int(umap_n_eff),
        "umap_min_dist_effective": float(umap_md_eff),
        "umap_n_neighbors_requested": int(cfg.n_neighbors),
        "umap_min_dist_requested": float(cfg.min_dist),
        "added_edges_scaled_by_median": bool(med_all is not None and np.isfinite(med_all) and med_all > 0),
        "added_edges_median": (float(np.median(added_edges_scaled)) if ('added_edges_scaled' in locals() and len(added_edges_scaled) > 0) else None),
        "autotune_trials": [
            {"n_neighbors": int(n), "min_dist": float(md), "jaccard": (None if (t is None or (isinstance(t, float) and not np.isfinite(t))) else float(t))}
            for (n, md, t) in tried
        ],
    }
    return Z, sub_idx.tolist(), D, info, model
