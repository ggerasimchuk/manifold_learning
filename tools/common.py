"""Shared utilities reused by multiple PBM tools modules."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def collect_time_series(
    panel_long: pd.DataFrame,
    wells: Sequence[str],
    channel: str,
    horizon: int,
) -> np.ndarray:
    """Return a dense matrix with one time series per well.

    Parameters
    ----------
    panel_long:
        Long-format dataframe with columns ``well_name`` and ``t``.
    wells:
        Ordered collection of well identifiers to extract.
    channel:
        Name of the column to read from ``panel_long``.
    horizon:
        Maximum horizon (number of time steps) to allocate per series.

    Returns
    -------
    numpy.ndarray
        Array of shape ``[len(wells), horizon]`` filled with NaN if data is missing.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    rows = []
    for well in wells:
        subset = (
            panel_long.loc[panel_long["well_name"] == well, ["t", channel]]
            .sort_values("t")
        )
        values = np.full(horizon, np.nan, float)
        if subset.empty:
            rows.append(values)
            continue
        t = subset["t"].to_numpy(int)
        v = subset[channel].to_numpy(float)
        mask = (t >= 0) & (t < horizon)
        t = t[mask]
        v = v[: len(t)]
        values[t[: len(v)]] = v
        rows.append(values)
    return np.vstack(rows) if rows else np.empty((0, horizon))


def compute_cluster_medoids(
    Z: np.ndarray,
    labels: Sequence[int],
    metric: str = "euclidean",
) -> Dict[int, int]:
    """Compute medoid indices for each non-noise cluster.

    Parameters
    ----------
    Z:
        Embedding coordinates with shape ``[n_samples, n_components]``.
    labels:
        Cluster labels where ``-1`` denotes noise.
    metric:
        Distance metric passed to :func:`sklearn.metrics.pairwise_distances`.

    Returns
    -------
    dict
        Mapping ``cluster_id → medoid_index`` (global index into ``Z``).
    """
    medoids: Dict[int, int] = {}
    labels = np.asarray(labels)
    unique_clusters = [cl for cl in np.unique(labels) if cl >= 0]
    for cluster_id in unique_clusters:
        cluster_idx = np.where(labels == cluster_id)[0]
        if cluster_idx.size == 0:
            continue
        if cluster_idx.size == 1:
            medoids[cluster_id] = int(cluster_idx[0])
            continue
        distances = pairwise_distances(Z[cluster_idx], metric=metric)
        medoid_local = int(np.argmin(distances.sum(axis=0)))
        medoids[cluster_id] = int(cluster_idx[medoid_local])
    return medoids


