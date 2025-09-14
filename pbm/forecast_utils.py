# -*- coding: utf-8 -*-
"""Utility helpers for forecast exports and plotting."""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_predictions_csv(
    Y_pred: np.ndarray,
    wells: List[str],
    name: str,
    out_dir: str,
    T_pref: int,
    T: int,
) -> str:
    """Save per-well predictions to CSV and return file path."""
    cols = [f"m{t}" for t in range(T_pref + 1, T + 1)]
    df = pd.DataFrame(Y_pred, columns=cols)
    df.insert(0, "well_name", wells)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pred_{name}.csv")
    df.to_csv(path, index=False)
    return path


def plot_example(
    idx: int,
    title: str,
    ytrue: np.ndarray,
    ypred: np.ndarray,
    out_dir: str,
    T_pref: int,
    T: int,
) -> str:
    """Plot true vs predicted suffix for a single well and save to file."""
    plt.figure()
    plt.plot(range(T_pref, T), ytrue[idx], label="true")
    plt.plot(range(T_pref, T), ypred[idx], label="pred")
    plt.title(title)
    plt.xlabel("month index")
    plt.ylabel("oil rate (r_oil_s)")
    plt.legend()
    fig_path = os.path.join(out_dir, f"{title.replace(' ','_').lower()}_{idx}.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    return fig_path
