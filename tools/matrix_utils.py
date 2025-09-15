from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _collect_matrix(panel_long: pd.DataFrame, wells: Sequence[str], channel: str, T: int) -> np.ndarray:
    """Собирает матрицу [n_series, T] по указанному каналу с NaN."""
    rows = []
    for w in wells:
        g = panel_long.loc[panel_long["well_name"] == w, ["t", channel]].sort_values("t")
        v = np.full(T, np.nan, float)
        t = g["t"].to_numpy(int)
        vals = g[channel].to_numpy(float)
        t = t[(t >= 0) & (t < T)]
        vals = vals[: len(t)]
        v[t[: len(vals)]] = vals
        rows.append(v)
    return np.vstack(rows) if rows else np.empty((0, T))
