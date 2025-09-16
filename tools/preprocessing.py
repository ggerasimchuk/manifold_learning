# -*- coding: utf-8 -*-
"""
Production Behavior Manifold (PBM) — Предобработка профилей (Шаг 1)
------------------------------------------------------------------
Этот блок определяет конфиг и функции предобработки для датафрейма df
со столбцами:
['well_name', 'date', 'oil', 'water', 'gas', 'days_prod']

Основные шаги:
- Приведение к среднесуточным дебитам (учёт days_prod, иначе — дни в месяце)
- Единая месячная сетка и агрегация
- Фильтры качества и робастная обработка выбросов
- Сглаживание (Savitzky–Golay)
- Производные каналы (wc, gor), нормализация по пику
- Выравнивание времени (t=0 — старт производства)
- Усечение/паддинг до горизонта T
- Подготовка длинного и «tensor» представления

Код написан так, чтобы безопасно выполняться даже без df в окружении.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ----------------------------
# Конфигурация предобработки
# ----------------------------

@dataclass
class PreprocConfig:
    # Временная сетка
    freq: str = "MS"                 # ежемесячно: начало месяца
    T: int = 70                      # целевой горизонт в месяцах (паддинг/обрезка)
    min_profile_months: int = 12     # минимальная длина валидного профиля (по основному каналу)

    # Порог старта производства для выравнивания (t=0)
    # Абс. порог для r_oil_s (сглаженный среднесуточный дебит нефти)
    prod_threshold_abs: float = 0.01
    # Относительный порог от пика r_oil_s (например, 5% от пика)
    prod_threshold_rel: float = 0.01

    # Сглаживание (Savitzky–Golay)
    smooth_window: int = 5           # должно быть нечётным, корректируется автоматически
    smooth_poly: int = 2

    # Робастная «подрезка» выбросов (winsorize) по КАЖДОЙ скважине
    winsorize_low: float = 0.01      # перцентиль снизу
    winsorize_high: float = 0.99     # перцентиль сверху

    # Нормализация амплитуды (деление на пик r_oil_s)
    normalize_by_peak: bool = True
    eps: float = 1e-9

    # Ограничения и заполнение
    clamp_nonnegative: bool = True   # отрицательные объёмы/дебиты → NaN (позже можно 0)
    interpolate_gaps: bool = True    # интерполяция пропусков для сглаживания

    # Каналы, которые пойдут в итоговый тензор (в таком порядке)
    tensor_channels: Tuple[str, ...] = (
        "r_oil_norm", "wc", "gor", "dr_oil_norm"
    )

    # Основной канал, по которому считаем длину валидного профиля
    primary_channel: str = "r_oil_s"


# ----------------------------
# Утилиты
# ----------------------------

def _safe_quantiles(s: pd.Series, ql: float, qh: float) -> Tuple[float, float]:
    s_clean = s[np.isfinite(s)]
    if s_clean.empty:
        return (np.nan, np.nan)
    return (s_clean.quantile(ql), s_clean.quantile(qh))


def _winsorize_per_well(s: pd.Series, low: float, high: float) -> pd.Series:
    if s.isna().all():
        return s
    ql, qh = _safe_quantiles(s, low, high)
    if not np.isfinite(ql) or not np.isfinite(qh):
        return s
    return s.clip(lower=ql, upper=qh)


def _savgol_safe(y: pd.Series, window: int, poly: int, do_interpolate: bool = True) -> pd.Series:
    if y.isna().all():
        return y
    x = y.copy()
    if do_interpolate:
        x = x.interpolate(limit_direction="both")
    # безопасная корректировка окна
    n = len(x)
    w = max(3, min(window, n))
    if w % 2 == 0:
        w = max(3, w - 1)
    p = min(poly, w - 1)
    if p < 1:
        # слишком короткий ряд — вернём как есть
        return y if not do_interpolate else x
    try:
        arr = savgol_filter(x.values.astype(float), window_length=w, polyorder=p, mode="interp")
        out = pd.Series(arr, index=y.index)
        return out
    except Exception:
        # на всякий случай резервация
        return y if not do_interpolate else x


def _month_start(ts: pd.Series) -> pd.Series:
    # Приводим даты к началу месяца
    return ts.dt.to_period("M").dt.to_timestamp()


def _days_in_month(dt: pd.Timestamp) -> int:
    # номинальные дни в месяце (если days_prod нет)
    return (dt + pd.offsets.MonthEnd(0) - (dt - pd.offsets.MonthBegin(1))).days + 1 - 1  # безопасно


def _weighted_rate(sum_volume: float, sum_days: float, eps: float) -> float:
    if not np.isfinite(sum_days) or sum_days <= 0:
        return np.nan
    return float(sum_volume) / (float(sum_days) + eps)


# ----------------------------------
# Основная функция предобработки
# ----------------------------------

@dataclass(slots=True)
class PreprocessOutput(Mapping[str, object]):
    panel_long: pd.DataFrame
    X: np.ndarray
    wells_used: List[str]
    dropped_summary: pd.DataFrame
    config: Dict[str, object]
    tensor_channels: List[str]

    _KEYS: Tuple[str, ...] = (
        "panel_long",
        "X",
        "wells_used",
        "dropped_summary",
        "config",
        "tensor_channels",
    )

    def __getitem__(self, key: str) -> object:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def as_dict(self) -> Dict[str, object]:
        return {key: getattr(self, key) for key in self._KEYS}


def preprocess_profiles(
    df: pd.DataFrame,
    cfg: Optional[PreprocConfig] = None,
) -> PreprocessOutput:
    """
    Предобработка «сырых» данных до панели, выровненной по t=0 и ограниченной горизонтом T.
    Возвращает:
      - panel_long: long-таблица (well_name, t, date, каналы)
      - X: np.ndarray [n_wells, T, C] — тензор по cfg.tensor_channels
      - wells_used: список скважин, попавших в итог
      - dropped_summary: DataFrame со статистикой отбраковки
      - config: фактический конфиг (dict)

    Требуемые столбцы: well_name, date, oil, water, gas, days_prod
    """
    if cfg is None:
        cfg = PreprocConfig()

    required_cols = {"well_name", "date", "oil", "water", "gas", "days_prod"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют необходимые столбцы: {missing}")

    data = df.copy()

    # Типы
    data["well_name"] = data["well_name"].astype(str)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Фильтрация строк с невалидной датой/именем
    data = data[ data["date"].notna() & data["well_name"].notna() ].copy()

    # Приведение отрицательных объёмов/дней к NaN (если включено)
    for col in ["oil", "water", "gas", "days_prod"]:
        if col in data.columns and cfg.clamp_nonnegative:
            data.loc[data[col] < 0, col] = np.nan

    # Приведение к началу месяца
    data["month"] = _month_start(data["date"])

    # Если days_prod отсутствует или нулевой — используем номинальные дни месяца
    # (это компромисс: лучше, чем делить на 30 фиксированно)
    # Параллельно аккумулируем объёмы по месяцу
    grp = data.groupby(["well_name", "month"], as_index=False).agg(
        oil_sum=("oil", "sum"),
        water_sum=("water", "sum"),
        gas_sum=("gas", "sum"),
        days_sum=("days_prod", "sum"),
        valid_days=("days_prod", lambda s: np.sum(np.isfinite(s) & (s > 0)))
    )

    # Фоллбек дней: если нет валидных days_prod за месяц, берём номинал
    # (важно: если исходные числа — уже среднесуточные, эту логику нужно отключить)
    # При необходимости пользователь может отдельно указать флаг в cfg, но для простоты — авто-детект.
    def _fallback_days(row):
        if row["valid_days"] and row["valid_days"] > 0:
            return row["days_sum"]
        return float(_days_in_month(pd.Timestamp(row["month"])))

    grp["days_eff"] = grp.apply(_fallback_days, axis=1)

    # Среднесуточные дебиты за месяц (взвешенное по дням)
    eps = cfg.eps
    grp["r_oil"] = grp.apply(lambda r: _weighted_rate(r["oil_sum"], r["days_eff"], eps), axis=1)
    grp["r_water"] = grp.apply(lambda r: _weighted_rate(r["water_sum"], r["days_eff"], eps), axis=1)
    grp["r_gas"] = grp.apply(lambda r: _weighted_rate(r["gas_sum"], r["days_eff"], eps), axis=1)

    # Сортировка
    grp = grp.sort_values(["well_name", "month"]).reset_index(drop=True)

    # По-скважинная обработка: winsorize → сглаживание → производные каналы → нормализация → выравнивание/усечение
    records: List[pd.DataFrame] = []
    dropped: List[Tuple[str, str]] = []  # (well_name, reason)

    for well, g in grp.groupby("well_name", sort=False):
        g = g.copy().reset_index(drop=True)

        # Робастная подрезка выбросов по КАЖДОМУ каналу дебитов
        for col in ["r_oil", "r_water", "r_gas"]:
            g[col] = _winsorize_per_well(g[col], cfg.winsorize_low, cfg.winsorize_high)

        # Интерполяция пропусков (опционально) — только для сглаживания
        do_interp = cfg.interpolate_gaps

        # Сглаживание
        g["r_oil_s"] = _savgol_safe(g["r_oil"], cfg.smooth_window, cfg.smooth_poly, do_interp)
        g["r_water_s"] = _savgol_safe(g["r_water"], cfg.smooth_window, cfg.smooth_poly, do_interp)
        g["r_gas_s"] = _savgol_safe(g["r_gas"], cfg.smooth_window, cfg.smooth_poly, do_interp)

        # Неотрицательность после сглаживания (если нужно)
        if cfg.clamp_nonnegative:
            for col in ["r_oil_s", "r_water_s", "r_gas_s"]:
                g.loc[g[col] < 0, col] = 0.0

        # Производные каналы
        g["wc"] = g["r_water_s"] / (g["r_oil_s"] + g["r_water_s"] + eps)
        g["gor"] = g["r_gas_s"] / (g["r_oil_s"] + eps)

        # Нормализация по пику r_oil_s
        if cfg.normalize_by_peak:
            peak = np.nanmax(g["r_oil_s"].values) if len(g) else np.nan
            scale = peak if (np.isfinite(peak) and peak > eps) else 1.0
            g["r_oil_norm"] = g["r_oil_s"] / (scale + eps)
            g["r_water_norm"] = g["r_water_s"] / (scale + eps)
            g["r_gas_norm"] = g["r_gas_s"] / (scale + eps)
        else:
            g["r_oil_norm"] = g["r_oil_s"]
            g["r_water_norm"] = g["r_water_s"]
            g["r_gas_norm"] = g["r_gas_s"]

        # Первая разность (динамика)
        g["dr_oil_norm"] = g["r_oil_norm"].diff()

        # Определяем точку старта t=0
        # Условие: r_oil_s >= max(prod_threshold_abs, prod_threshold_rel * peak)
        peak_s = np.nanmax(g["r_oil_s"].values) if len(g) else np.nan
        rel_thr = cfg.prod_threshold_rel * (peak_s if np.isfinite(peak_s) else 0.0)
        start_thr = max(cfg.prod_threshold_abs, rel_thr)
        start_idx = int(np.argmax(g["r_oil_s"].values >= start_thr)) if np.any(g["r_oil_s"].values >= start_thr) else None

        if start_idx is None:
            # Нет видимого старта — отбрасываем
            dropped.append((well, "no_start_detected"))
            continue

        g = g.iloc[start_idx:].copy().reset_index(drop=True)

        # Длина валидного профиля по основному каналу
        valid_len = int(np.sum(np.isfinite(g[cfg.primary_channel])))
        if valid_len < cfg.min_profile_months:
            dropped.append((well, f"too_short(<{cfg.min_profile_months})"))
            continue

        # Усечение/паддинг до T
        g = g.iloc[: cfg.T].copy()
        g["t"] = np.arange(len(g))
        # Паддинг — не добавляем строки с NaN, просто оставим длину < T;
        # при сборке тензора недостающие t будут NaN.

        # Сохраняем
        g["well_name"] = well
        records.append(g)

    if not records:
        panel_long = pd.DataFrame(columns=[
            "well_name", "t", "month",
            "r_oil", "r_water", "r_gas",
            "r_oil_s", "r_water_s", "r_gas_s",
            "wc", "gor",
            "r_oil_norm", "r_water_norm", "r_gas_norm",
            "dr_oil_norm"
        ])
        X = np.empty((0, cfg.T, len(cfg.tensor_channels)), dtype=float)
        wells_used: List[str] = []
    else:
        panel_long = pd.concat(records, ignore_index=True)
        panel_long = panel_long.rename(columns={"month": "date"})

        # Сборка тензора [n_wells, T, C] по cfg.tensor_channels
        channels = list(cfg.tensor_channels)
        wells_used = panel_long["well_name"].unique().tolist()
        n = len(wells_used)
        C = len(channels)
        T = cfg.T
        X = np.full((n, T, C), np.nan, dtype=float)

        # Быстрый доступ к индексам
        well_to_idx = {w: i for i, w in enumerate(wells_used)}
        ch_to_idx = {ch: j for j, ch in enumerate(channels)}

        for w, g in panel_long.groupby("well_name", sort=False):
            i = well_to_idx[w]
            # гарантируем t в пределах [0, T-1]
            g = g[(g["t"] >= 0) & (g["t"] < T)]
            for ch in channels:
                if ch not in g.columns:
                    continue
                X[i, g["t"].astype(int).values, ch_to_idx[ch]] = g[ch].values

    # Сводка по отбросам
    if dropped:
        dropped_df = pd.DataFrame(dropped, columns=["well_name", "reason"])
        dropped_summary = dropped_df["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    else:
        dropped_summary = pd.DataFrame(columns=["reason", "count"])

    result = PreprocessOutput(
        panel_long=panel_long,
        X=X,
        wells_used=wells_used,
        dropped_summary=dropped_summary,
        config=asdict(cfg),
        tensor_channels=list(cfg.tensor_channels),
    )
    print("Preprocess complete.")
    print(f"  В итог попало скважин: {len(result.wells_used)}")
    if len(dropped_summary):
        print("  Отброшено (по причинам):")
        display(dropped_summary)
    else:
        print("  Отброшенных скважин нет.")
    return result


# # ----------------------------
# # Пример использования
# # ----------------------------
# if 'df' in globals():
#     print("Обнаружен df в окружении. Запускаю предобработку с дефолтным конфигом...")
#     cfg = PreprocConfig()
#     out = preprocess_profiles(df, cfg)
#     # Покажем заголовок long-таблицы как предварительный просмотр
#     head_preview = out["panel_long"].head(12).copy()
#     try:
#         # Если доступна вспомогательная функция отображения таблиц — используем её
#         import caas_jupyter_tools
#         caas_jupyter_tools.display_dataframe_to_user("Предпросмотр panel_long", head_preview)
#     except Exception:
#         # Иначе просто печатаем
#         print(head_preview)
# else:
#     print("Готово. Функции предобработки загружены. Для запуска вызовите:")
#     print("cfg = PreprocConfig(T=36, min_profile_months=12)")
#     print("out = preprocess_profiles(df, cfg)")
#     print("panel_long, X = out['panel_long'], out['X']")
