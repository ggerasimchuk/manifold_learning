"""
Production Behavior Manifold (PBM) — Шаг 5: Визуализации и Отчёты
------------------------------------------------------------------
Зависимости (pip): numpy, pandas, matplotlib, (опц.) jinja2

Правила построения графиков (важно для окружения чат-ноутбука):
  - использовать matplotlib (не seaborn)
  - один график на фигуру (без субплотов)
  - не задавать явные цвета (использовать дефолтные)

Входы из предыдущих шагов:
  - out: результат preprocess_profiles (Шаг 1)
  - Z, sub_idx: embedding и индексы скважин, вошедшие в него (Шаг 3)
  - res: результат cluster_hdbscan или cluster_gmm_bic (Шаг 4) с полем df_map
  - protos: прототипы (build_cluster_prototypes)

Основные функции:
  * save_pbm_map(...) — карта PBM с кластерами/аномалиями
  * save_cluster_prototype_plots(...) — профили прототипов с IQR-заштриховкой (если доступна сырая матрица)
  * save_cluster_distribution_plot(...) — диаграмма распределения размеров кластеров
  * export_csv_summaries(...) — CSV-выгрузки (карта, сводка по кластерам, топ аномалий)
  * build_html_report(...) — простой HTML-отчёт с картинками и таблицами

Пример использования внизу файла (закомментирован).
"""
from __future__ import annotations

import os
import io
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # рендер без GUI
import matplotlib.pyplot as plt
from .matrix_utils import _collect_matrix

try:
    from tqdm import tqdm
    _TQDM = True
except Exception:
    _TQDM = False

# ------------------------- УТИЛИТЫ -------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------- КАРТА PBM ------------------------

def save_pbm_map(
    Z: np.ndarray,
    df_map: pd.DataFrame,
    out_dir: str,
    title: str = "PBM map (UMAP)",
    annotate_medoids: bool = True,
    mark_anomalies: bool = True,
    anomaly_col: str = "anomaly_score",
    color_clusters: bool = True,
    dpi: int = 160,
) -> str:
    """Сохраняет PNG с картой PBM (scatter), один график — одна фигура.
    Возвращает путь к PNG.
    """
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)
    # точки
    colors = None
    if color_clusters and "cluster" in df_map.columns:
        try:
            colors = df_map["cluster"].to_numpy()
        except Exception:
            colors = None
    sc = ax.scatter(Z[:, 0], Z[:, 1], s=12, c=colors)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # Медоиды кластеров (на координатах Z)
    if annotate_medoids and "cluster" in df_map.columns:
        from sklearn.metrics import pairwise_distances
        for cl, g in df_map[df_map["cluster"] >= 0].groupby("cluster"):
            idx = g.index.to_numpy()
            D = pairwise_distances(Z[idx], metric="euclidean")
            m_local = int(idx[np.argmin(D.sum(axis=0))])
            ax.text(Z[m_local, 0], Z[m_local, 1], f"C{int(cl)}", fontsize=9)

    # Аномалии (обводка поверх, если есть колонка)
    if mark_anomalies and anomaly_col in df_map.columns:
        q = df_map[anomaly_col].quantile(0.95)
        mask = df_map[anomaly_col] >= q
        ax.scatter(Z[mask, 0], Z[mask, 1], s=32, facecolors='none', edgecolors='k', linewidths=0.8)

    path = os.path.join(out_dir, "pbm_map.png")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# --------------- РАСПРЕДЕЛЕНИЕ КЛАСТЕРОВ ----------------

def save_cluster_distribution_plot(df_map: pd.DataFrame, out_dir: str, dpi: int = 160) -> str:
    _ensure_dir(out_dir)
    sizes = df_map[df_map["cluster"] >= 0]["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Cluster sizes")
    ax.bar(sizes.index.astype(str), sizes.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    path = os.path.join(out_dir, "cluster_sizes.png")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ------------------ ПРОФИЛИ ПРОТОТИПОВ -------------------

def save_cluster_prototype_plots(
    panel_long: pd.DataFrame,
    df_map: pd.DataFrame,
    protos: Dict[int, Dict[str, np.ndarray]],
    channels: Sequence[str],
    T: int,
    out_dir: str,
    dpi: int = 160,
) -> List[str]:
    """Для каждого кластера и канала рисует один график с прототипом и IQR-зоной.
    Возвращает список путей к PNG.
    """
    _ensure_dir(out_dir)
    paths: List[str] = []
    # подготовка списков скважин по кластеру
    cl2wells: Dict[int, List[str]] = {}
    for row in df_map.itertuples(index=False):
        if row.cluster >= 0:
            cl2wells.setdefault(int(row.cluster), []).append(str(row.well_name))

    for cl, ch_dict in protos.items():
        wells = cl2wells.get(cl, [])
        for ch in channels:
            proto = ch_dict.get(ch)
            if proto is None:
                continue
            # Матрица ряда для IQR
            M = _collect_matrix(panel_long, wells, ch, T)
            p25 = np.nanpercentile(M, 25, axis=0) if M.size else None
            p75 = np.nanpercentile(M, 75, axis=0) if M.size else None
            x = np.arange(len(proto))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title(f"Cluster {cl} — {ch}")
            ax.plot(x, proto)
            if M.size:
                ax.fill_between(x, p25, p75, alpha=0.25)
            ax.set_xlabel("t (months since start)")
            ax.set_ylabel(ch)
            path = os.path.join(out_dir, f"cluster_{cl}_{ch}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            paths.append(path)
    return paths


# ----------------------- CSV-ЭКСПОРТ ---------------------

def export_csv_summaries(
    df_map: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: str,
    top_anoms: int = 50,
) -> Dict[str, str]:
    _ensure_dir(out_dir)
    paths = {}
    p_map = os.path.join(out_dir, "pbm_map_points.csv")
    df_map.to_csv(p_map, index=False)
    paths["map_csv"] = p_map

    p_sum = os.path.join(out_dir, "cluster_summary.csv")
    summary.to_csv(p_sum, index=False)
    paths["summary_csv"] = p_sum

    if "anomaly_score" in df_map.columns:
        p_an = os.path.join(out_dir, "top_anomalies.csv")
        df_map.sort_values("anomaly_score", ascending=False).head(top_anoms).to_csv(p_an, index=False)
        paths["anomalies_csv"] = p_an
    return paths


# --------------------- HTML-ОТЧЁТ -----------------------

def build_html_report(
    out_dir: str,
    map_png: str,
    sizes_png: str,
    proto_pngs: Sequence[str],
    df_map: pd.DataFrame,
    summary: pd.DataFrame,
    title: str = "PBM Report",
    filename: str = "PBM_report.html",
) -> str:
    """Простой статический HTML-отчёт без внешних шаблонизаторов.
    Картинки и таблицы вставляются как <img> и <table>.
    """
    _ensure_dir(out_dir)
    def df_to_html(df: pd.DataFrame) -> str:
        # Минимальная чистка HTML
        return df.to_html(index=False, escape=True)

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>{}</title>".format(title),
        "<style>body{font-family:Arial, sans-serif; margin:24px;} h2{margin-top:28px;} img{max-width:100%; height:auto;} table{border-collapse:collapse;} td,th{border:1px solid #ddd; padding:6px;}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        "<h2>PBM Map</h2>",
        f"<img src='{os.path.relpath(map_png, out_dir)}' alt='PBM map'>",
        "<h2>Cluster Sizes</h2>",
        f"<img src='{os.path.relpath(sizes_png, out_dir)}' alt='Cluster sizes'>",
        "<h2>Cluster Summary</h2>",
        df_to_html(summary),
        "<h2>Top Anomalies</h2>",
        df_to_html(df_map.sort_values('anomaly_score', ascending=False).head(50) if 'anomaly_score' in df_map.columns else df_map.head(50)),
        "<h2>Cluster Prototypes</h2>",
    ]
    # галерея прототипов
    for p in proto_pngs:
        html.append(f"<img src='{os.path.relpath(p, out_dir)}' alt='{os.path.basename(p)}'>")
    html.append("</body></html>")

    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_path

