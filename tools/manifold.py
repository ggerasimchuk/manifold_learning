"""Manifold construction utilities for PBM."""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import umap
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

try:  # optional, used only for DTW refinement
    from fastdtw import fastdtw
except Exception:  # pragma: no cover - dependency is optional
    fastdtw = None  # type: ignore

try:
    from tqdm import tqdm

    _TQDM = True
except Exception:  # pragma: no cover - tqdm is optional
    _TQDM = False

__all__ = [
    "ManifoldConfig",
    "ManifoldEmbedding",
    "embed_umap_euclid",
    "embed_umap_fastdtw",
]


@dataclass(slots=True)
class ManifoldConfig:
    channels: Tuple[str, ...] = ("r_oil_norm", "wc")
    fastdtw_radius: int = 6
    k_refine: int = 40
    weights: Optional[Tuple[float, ...]] = (0.7, 0.3)
    n_neighbors: int = 30
    min_dist: float = 0.05
    n_components: int = 2
    random_state: int = 43


@dataclass(slots=True)
class ManifoldEmbedding:
    embedding: np.ndarray
    wells: List[int]
    distance_matrix: np.ndarray
    model: umap.UMAP
    info: Dict[str, object]

    def __iter__(self):  # backwards compatibility with tuple unpacking
        yield self.embedding
        yield self.wells
        yield self.distance_matrix
        yield self.info


def _channel_indices(tensor_channels: Sequence[str], channels: Sequence[str]) -> List[int]:
    ch_to_idx = {c: i for i, c in enumerate(tensor_channels)}
    idx = [ch_to_idx[c] for c in channels if c in ch_to_idx]
    if len(idx) != len(channels):
        missing = [c for c in channels if c not in ch_to_idx]
        raise ValueError(f"В тензоре X нет каналов: {missing}")
    return idx


def _flatten_series_matrix(X: np.ndarray, channels_idx: Sequence[int]) -> np.ndarray:
    x = X[:, :, channels_idx]
    N, T, C = x.shape
    x = np.transpose(x, (0, 2, 1)).reshape(N, C * T)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def embed_umap_euclid(
    X: np.ndarray,
    tensor_channels: Sequence[str],
    channels: Sequence[str],
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, umap.UMAP]:
    """Быстрый базовый manifold: UMAP по евклиду на выровненных рядах."""

    idx = _channel_indices(tensor_channels, channels)
    X2 = _flatten_series_matrix(X, idx)
    model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=random_state,
    )
    Z = model.fit_transform(X2)
    return Z, model


def _fastdtw_multichannel(
    A: np.ndarray,
    B: np.ndarray,
    weights: Optional[Sequence[float]] = None,
    radius: int = 6,
) -> float:
    if fastdtw is None:
        raise RuntimeError("fastdtw не установлен")
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Ожидается [T,C] для обоих рядов")
    maskA = np.any(np.isfinite(A), axis=1)
    maskB = np.any(np.isfinite(B), axis=1)
    A = A[maskA]
    B = B[maskB]
    if A.size == 0 or B.size == 0:
        return float("inf")
    A = np.nan_to_num(A, nan=0.0)
    B = np.nan_to_num(B, nan=0.0)
    if weights is not None:
        w = np.asarray(weights, float).reshape(1, -1)
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
    sample_size: Optional[int] = None,
    candidate_knn: Optional[int] = None,
) -> ManifoldEmbedding:
    """UMAP с уточнением FastDTW для ближайших пар."""

    if cfg is None:
        cfg = ManifoldConfig()

    rng = np.random.default_rng(cfg.random_state)
    N = X.shape[0]
    idx = _channel_indices(tensor_channels, channels)

    all_idx = np.arange(N)
    if sample_size is not None and sample_size < N:
        sub_idx = rng.choice(all_idx, size=sample_size, replace=False)
    else:
        sub_idx = all_idx
    Xs = X[sub_idx][:, :, idx]

    Xflat = _flatten_series_matrix(Xs, list(range(len(idx))))
    E = pairwise_distances(Xflat, metric="euclidean")
    D = E.copy()

    if fastdtw is None:
        warnings.warn("fastdtw недоступен — возвращаю UMAP по евклиду (без уточнения DTW)")
    else:
        Ns = Xs.shape[0]
        k = cfg.k_refine if candidate_knn is None else candidate_knn
        nn = NearestNeighbors(n_neighbors=min(k + 1, Ns), metric="euclidean")
        nn.fit(Xflat)
        dists, inds = nn.kneighbors(Xflat, return_distance=True)
        pairs = set()
        for i in range(Ns):
            for j in inds[i, 1:]:
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                if a != b:
                    pairs.add((a, b))
        it = list(pairs)
        if _TQDM:
            it = tqdm(it, desc=f"Recompute DTW for {len(pairs)} pairs (radius={cfg.fastdtw_radius})")
        for i, j in it:
            d = _fastdtw_multichannel(Xs[i], Xs[j], weights=cfg.weights, radius=cfg.fastdtw_radius)
            if math.isfinite(d):
                D[i, j] = d
                D[j, i] = d

    model = umap.UMAP(
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        n_components=cfg.n_components,
        metric="precomputed",
        random_state=cfg.random_state,
    )
    Z = model.fit_transform(D)

    info = {
        "Ns": int(len(sub_idx)),
        "channels": list(channels),
        "k_refine": int(cfg.k_refine),
        "fastdtw_radius": int(cfg.fastdtw_radius),
        "umap_params": {
            "n_neighbors": cfg.n_neighbors,
            "min_dist": cfg.min_dist,
            "n_components": cfg.n_components,
            "random_state": cfg.random_state,
        },
    }
    return ManifoldEmbedding(embedding=Z, wells=sub_idx.tolist(), distance_matrix=D, model=model, info=info)
