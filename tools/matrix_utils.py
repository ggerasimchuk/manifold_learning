from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _collect_matrix(panel_long: pd.DataFrame, wells: Sequence[str], channel: str, T: int) -> np.ndarray:
    """Собирает матрицу [n_series, T] по указанному каналу с NaN."""
    rows = []
    groups = panel_long.groupby("well_name")
    for w in wells:
        g = (
            groups.get_group(w)[["t", channel]].sort_values("t")
            if w in groups.groups
            else pd.DataFrame(columns=["t", channel])
        )
        v = np.full(T, np.nan, float)
        t = g["t"].to_numpy(int)
        vals = g[channel].to_numpy(float)
        mask = (t >= 0) & (t < T)
        t = t[mask]
        vals = vals[mask]
        v[t] = vals  # Предполагается, что t отсортирован по возрастанию
        rows.append(v)
    return np.vstack(rows) if rows else np.empty((0, T))
