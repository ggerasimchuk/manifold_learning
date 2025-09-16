"""Utilities for building Production Behavior Manifold embeddings."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import umap

try:  # optional dependency — warn later if not available
    from fastdtw import fastdtw
except Exception:  # pragma: no cover - import guard
    fastdtw = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
    _TQDM = True
except Exception:  # pragma: no cover - tqdm is optional
    _TQDM = False

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


@dataclass
class ManifoldConfig:
    """Configuration for distance refinement and UMAP parameters."""

    channels: Tuple[str, ...] = ("r_oil_norm", "wc")
    fastdtw_radius: int = 6
    k_refine: int = 40
    weights: Optional[Tuple[float, ...]] = (0.7, 0.3)
    n_neighbors: int = 30
    min_dist: float = 0.05
    n_components: int = 2
    random_state: int = 43


@dataclass
class ManifoldEmbedding:
    """Bundle with embedding results for convenience."""

    embedding: np.ndarray
    umap_model: umap.UMAP
    subset_indices: np.ndarray
    distance_matrix: Optional[np.ndarray] = None
    info: Optional[Dict[str, object]] = None

    def wells_subset(self, wells: Sequence[str]) -> List[str]:
        """Return well names corresponding to ``subset_indices`` order."""
        return [wells[int(i)] for i in self.subset_indices]


def _resolve_channel_indices(
    tensor_channels: Sequence[str],
    channels: Sequence[str],
) -> List[int]:
    mapping = {c: i for i, c in enumerate(tensor_channels)}
    missing = [c for c in channels if c not in mapping]
    if missing:
        raise ValueError(f"В тензоре X нет каналов: {missing}")
    return [mapping[c] for c in channels]


def _flatten_series_matrix(X: np.ndarray, channels_idx: Sequence[int]) -> np.ndarray:
    """Convert ``[N, T, C]`` to ``[N, T*len(channels_idx)]`` with NaNs replaced by zeros."""
    idx = np.asarray(channels_idx, dtype=int)
    x = X[:, :, idx]
    n_samples, horizon, n_channels = x.shape
    x = np.transpose(x, (0, 2, 1)).reshape(n_samples, n_channels * horizon)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def embed_umap_euclid(
    X: np.ndarray,
    tensor_channels: Sequence[str],
    channels: Sequence[str],
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    n_components: int = 2,
    random_state: int = 42,
) -> ManifoldEmbedding:
    """Fast baseline manifold: UMAP with Euclidean distance on flattened profiles."""
    idx = _resolve_channel_indices(tensor_channels, channels)
    X_flat = _flatten_series_matrix(X, idx)
    model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=random_state,
    )
    embedding = model.fit_transform(X_flat)
    subset = np.arange(X_flat.shape[0], dtype=int)
    info = {
        "method": "euclidean",
        "channels": list(channels),
        "umap_params": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": n_components,
            "random_state": random_state,
        },
    }
    return ManifoldEmbedding(
        embedding=embedding,
        umap_model=model,
        subset_indices=subset,
        distance_matrix=None,
        info=info,
    )


def _fastdtw_multichannel(
    A: np.ndarray,
    B: np.ndarray,
    *,
    weights: Optional[Sequence[float]] = None,
    radius: int = 6,
) -> float:
    """Multi-channel DTW distance using FastDTW and optional channel weights."""
    if fastdtw is None:
        raise RuntimeError("fastdtw не установлен")
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Ожидается [T,C] для обоих рядов")
    mask_a = np.any(np.isfinite(A), axis=1)
    mask_b = np.any(np.isfinite(B), axis=1)
    A = A[mask_a]
    B = B[mask_b]
    if A.size == 0 or B.size == 0:
        return float("inf")
    A = np.nan_to_num(A, nan=0.0)
    B = np.nan_to_num(B, nan=0.0)
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(1, -1)
        if w.shape[1] != A.shape[1]:
            raise ValueError("Длина weights должна совпадать с числом каналов")
        A = A * w
        B = B * w
    dist, _ = fastdtw(A, B, radius=radius, dist=2)
    return float(dist)


def embed_umap_fastdtw(
    X: np.ndarray,
    tensor_channels: Sequence[str],
    channels: Sequence[str],
    cfg: Optional[ManifoldConfig] = None,
    *,
    sample_size: Optional[int] = None,
    candidate_knn: Optional[int] = None,
) -> ManifoldEmbedding:
    """UMAP embedding where a Euclidean graph is refined with multi-channel FastDTW."""
    if cfg is None:
        cfg = ManifoldConfig()

    idx = _resolve_channel_indices(tensor_channels, channels)
    rng = np.random.default_rng(cfg.random_state)
    n_samples = X.shape[0]
    all_idx = np.arange(n_samples)
    if sample_size is not None and sample_size < n_samples:
        sub_idx = np.sort(rng.choice(all_idx, size=sample_size, replace=False))
    else:
        sub_idx = all_idx

    X_subset = X[sub_idx][:, :, idx]
    flat_subset = _flatten_series_matrix(X_subset, np.arange(len(idx)))
    euclid_dist = pairwise_distances(flat_subset, metric="euclidean")
    refined_dist = euclid_dist.copy()

    if fastdtw is None:
        warnings.warn("fastdtw недоступен — возвращаю UMAP по евклиду (без уточнения DTW)")
    else:
        k = cfg.k_refine if candidate_knn is None else candidate_knn
        nn = NearestNeighbors(n_neighbors=min(k + 1, flat_subset.shape[0]), metric="euclidean")
        nn.fit(flat_subset)
        _, indices = nn.kneighbors(flat_subset, return_distance=True)
        pairs = set()
        for i, row in enumerate(indices):
            for j in row[1:]:  # пропускаем саму точку
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                if a != b:
                    pairs.add((a, b))
        iterator = list(pairs)
        if _TQDM:
            iterator = tqdm(iterator, desc=f"Recompute DTW for {len(pairs)} pairs (radius={cfg.fastdtw_radius})")
        for i, j in iterator:
            dist = _fastdtw_multichannel(
                X_subset[i],
                X_subset[j],
                weights=cfg.weights,
                radius=cfg.fastdtw_radius,
            )
            if math.isfinite(dist):
                refined_dist[i, j] = dist
                refined_dist[j, i] = dist

    model = umap.UMAP(
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        n_components=cfg.n_components,
        metric="precomputed",
        random_state=cfg.random_state,
    )
    embedding = model.fit_transform(refined_dist)

    info = {
        "method": "fastdtw",
        "Ns": int(X_subset.shape[0]),
        "channels": list(channels),
        "k_refine": int(cfg.k_refine),
        "fastdtw_radius": int(cfg.fastdtw_radius),
        "candidate_knn": int(candidate_knn) if candidate_knn is not None else int(cfg.k_refine),
        "umap_params": {
            "n_neighbors": cfg.n_neighbors,
            "min_dist": cfg.min_dist,
            "n_components": cfg.n_components,
            "random_state": cfg.random_state,
        },
    }
    return ManifoldEmbedding(
        embedding=embedding,
        umap_model=model,
        subset_indices=sub_idx.astype(int),
        distance_matrix=refined_dist,
        info=info,
    )

