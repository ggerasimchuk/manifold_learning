# -*- coding: utf-8 -*-
"""
preprocessing.py

Назначение
---------
Единый и воспроизводимый препроцессинг профилей добычи по скважинам.
На вход подаются помесячные данные (объёмы или среднесуточные дебиты).
На выходе:
- panel_long: long-таблица, выровненная по t=0 (старт), со сглаженными и производными каналами
- X: тензор признаков формы [N, T, C] (float32), где N — число скважин, T — горизонт,
     C — число каналов из конфигурации
- mask_X: булева маска формы [N, T], какие месяцы по скважинам валидны
- wells: порядок скважин в тензоре
- quality_flags: метрики качества по скважинам
- dropped_df: причины исключения скважин, если таковые есть
- cfg_dict: фактическая конфигурация

Ключевые принципы
-----------------
1) Робастность: winsorize по перцентилям до сглаживания, eps в отношениях, контроль интерполяции.
2) Контроль интерполяции: линейная только для разрывов длиной ≤ max_gap; большие разрывы оставляем NaN.
3) Сглаживание: Savitzky–Golay с автоподбором окна (fallback на скользящую медиану).
4) Нормализация формы: по пику, по перцентилю (p95) или по среднему top-k.
5) Выравнивание по старту: порог abs/rel + требование «минимум k подряд» с гистерезисом.
6) Маски вместо жёсткого дропа коротких профилей: возвращается mask_X; padding нулями вне валидных зон.
7) Производительность: векторные шаги, downcast в float32, опциональный параллелизм по скважинам.
8) Журналирование: через logging, без print/display.

Требуемые столбцы входа
-----------------------
- well_name: str
- date: datetime64[ns] или строка, парсится в дату (будет приведена к началу месяца)
- oil, water, gas: неотрицательные величины
  * Если cfg.input_is_monthly_rates = False: это месячные объёмы (в тех же единицах),
    и нужен days_prod.
  * Если cfg.input_is_monthly_rates = True: это среднесуточные дебиты.
- days_prod: число отработанных дней в месяце (нужно, если переданы объёмы).
  Месяц признаётся валидным для расчёта дебитов, если days_prod / days_in_month ≥ cfg.days_min_fraction.
  Иначе — пропуск (NaN), без фоллбека на «номинальные» дни.

Замечания
---------
- Дубликаты внутри месяца обрабатываются стратегией cfg.dup_strategy: 'sum' | 'max' | 'mean' | 'error'.
- Список доступных каналов в panel_long:
  ['r_oil', 'r_water', 'r_gas',         # среднесуточные до сглаживания
   'r_oil_s', 'r_water_s', 'r_gas_s',   # сглаженные
   'r_oil_norm', 'wc', 'gor', 'dr_oil_norm'].
- В тензор X попадут только существующие в panel_long каналы из cfg.tensor_channels.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import logging
import numpy as np
import pandas as pd

# Savitzky–Golay
try:
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Параллелизм
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# Numba-ускорения
try:
    import numba as nb
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


# -------------------------- Логирование --------------------------

logger = logging.getLogger("preprocessing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -------------------------- Конфигурация --------------------------

@dataclass
class PreprocConfig:
    # Общие
    freq: str = "MS"
    T: int = 70
    tensor_channels: Tuple[str, ...] = ("r_oil_norm", "wc", "gor", "dr_oil_norm")
    nonneg_mode: str = "clip"        # 'clip' | 'none'
    dtype: str = "float32"

    # Вход и дни
    input_is_monthly_rates: bool = False
    days_min_fraction: float = 0.6

    # Дубликаты
    dup_strategy: str = "sum"        # 'sum' | 'max' | 'mean' | 'error'

    # Выбросы
    q_low: float = 0.01
    q_high: float = 0.99

    # Интерполяция
    do_interpolate: bool = True
    max_gap: int = 2                 # максимум подряд NaN для линейной интерполяции

    # Сглаживание
    sg_window: int = 7
    sg_poly: int = 2

    # Нормализация формы
    normalize_by: str = "p95"        # 'peak' | 'p95' | 'topk_mean'
    norm_quantile: float = 0.95
    norm_topk: int = 3
    min_norm_denom: float = 1e-6

    # Детект старта
    start_abs_thr: float = 1e-6
    start_rel_thr: float = 0.05
    start_min_consecutive: int = 2
    start_hysteresis: float = 0.1

    # Фильтры
    min_profile_len: int = 6

    # Разное
    diff_fill0: bool = True
    n_jobs: int = 1                   # -1: все ядра


# -------------------------- Константы --------------------------

_EPS = 1e-12
_FLOAT_DTYPE = np.float32


# -------------------------- Векторные утилиты --------------------------

def _ensure_month_start(dt: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt, errors="coerce")
    return s.values.astype("datetime64[M]").astype("datetime64[ns]")


def _days_in_month(series_dt: pd.Series) -> pd.Series:
    s = pd.to_datetime(series_dt, errors="coerce")
    return s.dt.days_in_month.fillna(0).astype(int)


def _enforce_nonneg_arr(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clip":
        np.maximum(x, 0.0, out=x)
    return x


def _winsorize_arr(x: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    if x.size == 0:
        return x
    # игнорируем NaN при оценке квантилей
    xn = x[np.isfinite(x)]
    if xn.size == 0:
        return x
    lo = np.nanquantile(xn, q_low)
    hi = np.nanquantile(xn, q_high)
    return np.clip(x, lo, hi, out=x.copy())


def _savgol_safe_arr(x: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = x.size
    if n == 0:
        return x
    w = max(3, window)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < poly + 2:
        w = poly + 3
        if w % 2 == 0:
            w += 1
        if w > n:
            # fallback: скользящая медиана
            k = min(7, n if n > 0 else 1)
            return _rolling_median_centered(x, k)
    if _HAVE_SCIPY:
        try:
            return savgol_filter(x, window_length=w, polyorder=poly, mode="interp")
        except Exception:
            pass
    # fallback
    k = min(7, n if n > 0 else 1)
    return _rolling_median_centered(x, k)


def _rolling_median_centered(x: np.ndarray, k: int) -> np.ndarray:
    # простой и быстрый fallback без pandas
    n = x.size
    y = np.empty_like(x)
    half = k // 2
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        y[i] = np.nanmedian(x[l:r])
    return y


def _normalize_series_arr(x: np.ndarray, method: str, q: float, topk: int, min_denom: float) -> Tuple[np.ndarray, float]:
    if method == "peak":
        denom = float(np.nanmax(x)) if np.isfinite(np.nanmax(x)) else 0.0
    elif method == "p95":
        denom = float(np.nanpercentile(x, q * 100.0)) if np.isfinite(np.nanpercentile(x, q * 100.0)) else 0.0
    elif method == "topk_mean":
        xf = x[np.isfinite(x)]
        if xf.size == 0:
            denom = 0.0
        else:
            k = int(max(1, min(topk, xf.size)))
            # top-k среднее через partial sort
            part = np.partition(xf, xf.size - k)[-k:]
            denom = float(np.mean(part))
    else:
        raise ValueError(f"Unknown normalize method: {method}")
    denom = max(denom, min_denom)
    return x / denom, denom


# -------------------------- Numba-ускоряемые ядра --------------------------

if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def _interpolate_limited_np(x: np.ndarray, max_gap: int) -> Tuple[np.ndarray, int]:
        y = x.copy()
        isn = np.isnan(y)
        n = y.size
        interp_count = 0
        start = -1
        for i in range(n):
            if isn[i] and start == -1:
                start = i
            last = (i == n - 1)
            if (not isn[i] or last) and start != -1:
                end = i if not isn[i] else i + 1
                gap_len = end - start
                left = start - 1
                right = end
                if gap_len <= max_gap and left >= 0 and right < n and not np.isnan(y[left]) and not np.isnan(y[right]):
                    # линейная интерполяция
                    x0 = float(y[left])
                    x1 = float(y[right])
                    for t in range(start, end):
                        alpha = (t - left) / (right - left)
                        y[t] = x0 + alpha * (x1 - x0)
                    interp_count += gap_len
                start = -1
        return y, interp_count

    @nb.njit(cache=True)
    def _detect_start_np(x: np.ndarray, abs_thr: float, rel_thr: float, min_consecutive: int, hysteresis: float) -> int:
        n = x.size
        peak = -np.inf
        for i in range(n):
            if x[i] > peak:
                peak = x[i]
        if not np.isfinite(peak):
            return -1
        thr_on = abs_thr
        rel = rel_thr * peak
        if rel > thr_on:
            thr_on = rel
        thr_off = thr_on * (1.0 - max(0.0, min(hysteresis, 1.0)))
        i = 0
        while i < n:
            if x[i] >= thr_on:
                ok = True
                # нужно min_consecutive-1 точек после i на уровне thr_off
                for j in range(i + 1, min(i + min_consecutive, n)):
                    if x[j] < thr_off:
                        ok = False
                        break
                if ok:
                    return i
            i += 1
        return -1

    @nb.njit(cache=True)
    def _count_large_gaps(isnan_mask: np.ndarray, max_gap: int) -> int:
        n = isnan_mask.size
        i = 0
        large = 0
        while i < n:
            if isnan_mask[i]:
                j = i
                while j < n and isnan_mask[j]:
                    j += 1
                if (j - i) > max_gap:
                    large += 1
                i = j
            else:
                i += 1
        return large

else:
    def _interpolate_limited_np(x: np.ndarray, max_gap: int) -> Tuple[np.ndarray, int]:
        y = x.copy()
        isn = np.isnan(y)
        n = y.size
        interp_count = 0
        start = -1
        for i in range(n):
            if isn[i] and start == -1:
                start = i
            last = (i == n - 1)
            if (not isn[i] or last) and start != -1:
                end = i if not isn[i] else i + 1
                gap_len = end - start
                left = start - 1
                right = end
                if gap_len <= max_gap and left >= 0 and right < n and not np.isnan(y[left]) and not np.isnan(y[right]):
                    # линейная интерполяция
                    x0 = float(y[left])
                    x1 = float(y[right])
                    for t in range(start, end):
                        alpha = (t - left) / (right - left)
                        y[t] = x0 + alpha * (x1 - x0)
                    interp_count += gap_len
                start = -1
        return y, interp_count

    def _detect_start_np(x: np.ndarray, abs_thr: float, rel_thr: float, min_consecutive: int, hysteresis: float) -> int:
        if x.size == 0 or not np.isfinite(x).any():
            return -1
        peak = float(np.nanmax(x))
        thr_on = max(abs_thr, rel_thr * peak)
        thr_off = thr_on * (1.0 - max(0.0, min(hysteresis, 1.0)))
        n = x.size
        i = 0
        while i < n:
            if x[i] >= thr_on:
                ok = True
                for j in range(i + 1, min(i + min_consecutive, n)):
                    if x[j] < thr_off:
                        ok = False
                        break
                if ok:
                    return i
            i += 1
        return -1

    def _count_large_gaps(isnan_mask: np.ndarray, max_gap: int) -> int:
        n = isnan_mask.size
        i = 0
        large = 0
        while i < n:
            if isnan_mask[i]:
                j = i
                while j < n and isnan_mask[j]:
                    j += 1
                if (j - i) > max_gap:
                    large += 1
                i = j
            else:
                i += 1
        return large


# -------------------------- Основная функция --------------------------

def preprocess_profiles(
    df: pd.DataFrame,
    cfg: PreprocConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], pd.DataFrame, pd.DataFrame, Dict]:
    """
    Полный препроцессинг входной панели скважин. Функционал идентичен предыдущей версии.
    Возвращает: panel_long, X, mask_X, wells, quality_flags, dropped_df, cfg_dict
    """
    required = {"well_name", "date", "oil", "water", "gas"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {missing}")
    if not cfg.input_is_monthly_rates and "days_prod" not in df.columns:
        raise ValueError("Нужен 'days_prod' при cfg.input_is_monthly_rates=False.")

    # ---- Базовая подготовка
    d = df.copy()
    d["well_name"] = d["well_name"].astype(str)
    d["date"] = _ensure_month_start(d["date"])

    # Дубликаты
    if cfg.dup_strategy != "error":
        agg = {
            "oil": "sum" if cfg.dup_strategy == "sum" else cfg.dup_strategy,
            "water": "sum" if cfg.dup_strategy == "sum" else cfg.dup_strategy,
            "gas": "sum" if cfg.dup_strategy == "sum" else cfg.dup_strategy,
        }
        if "days_prod" in d.columns:
            agg["days_prod"] = "sum" if cfg.dup_strategy == "sum" else cfg.dup_strategy
        d = (
            d.groupby(["well_name", "date"], as_index=False)
            .agg(agg)
            .sort_values(["well_name", "date"])
            .reset_index(drop=True)
        )
    else:
        dup_mask = d.duplicated(["well_name", "date"], keep=False)
        if dup_mask.any():
            raise ValueError("Дубликаты (well_name, date) при dup_strategy='error'.")

    # Дни месяца
    d["days_in_month"] = _days_in_month(d["date"])

    # Rate-каналы
    if cfg.input_is_monthly_rates:
        d["r_oil"] = d["oil"].astype(float)
        d["r_water"] = d["water"].astype(float)
        d["r_gas"] = d["gas"].astype(float)
        d["days_prod"] = d.get("days_prod", np.nan)
        d["valid_days_frac"] = np.where(np.isfinite(d["days_prod"]),
                                        d["days_prod"] / d["days_in_month"].clip(lower=1),
                                        np.nan)
    else:
        d["valid_days_frac"] = d["days_prod"] / d["days_in_month"].clip(lower=1)
        valid_mask = d["valid_days_frac"] >= float(cfg.days_min_fraction)
        denom = d["days_prod"].astype(float).clip(lower=1.0)
        d["r_oil"] = np.where(valid_mask, d["oil"].astype(float) / denom, np.nan)
        d["r_water"] = np.where(valid_mask, d["water"].astype(float) / denom, np.nan)
        d["r_gas"] = np.where(valid_mask, d["gas"].astype(float) / denom, np.nan)

    # Неотрицательность исходных дебитов
    for col in ("r_oil", "r_water", "r_gas"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d[col] = _enforce_nonneg_arr(d[col].to_numpy(copy=False), cfg.nonneg_mode)

    # ---- Индексация и группировка без лишних копий
    # Categorical ускоряет фильтры по имени скважины
    d["well_cat"] = pd.Categorical(d["well_name"])
    well_codes = d["well_cat"].cat.codes.to_numpy()
    well_names = d["well_cat"].cat.categories.tolist()
    wells = well_names  # порядок кодов совпадает с categories

    # Сохранить массивы для быстрых срезов
    dates = d["date"].to_numpy()
    days_frac = d["valid_days_frac"].to_numpy()
    r_oil = d["r_oil"].to_numpy(dtype=float, copy=False)
    r_water = d["r_water"].to_numpy(dtype=float, copy=False)
    r_gas = d["r_gas"].to_numpy(dtype=float, copy=False)

    # Индексы начала каждой группы скважины
    order = np.lexsort((dates, well_codes))
    well_codes_sorted = well_codes[order]
    dates_sorted = dates[order]
    days_frac_sorted = days_frac[order]
    r_oil_sorted = r_oil[order]
    r_water_sorted = r_water[order]
    r_gas_sorted = r_gas[order]

    # Находим границы групп
    change = np.empty_like(well_codes_sorted, dtype=bool)
    change[0] = True
    change[1:] = well_codes_sorted[1:] != well_codes_sorted[:-1]
    group_starts = np.flatnonzero(change)
    group_ends = np.concatenate([group_starts[1:], np.array([well_codes_sorted.size])])

    # ---- По-скважинная обработка на массивах
    panel_chunks: List[pd.DataFrame] = []
    quality_rows: List[Dict] = []
    dropped_rows: List[Dict] = []

    def process_slice(idx_well: int, s: int, e: int):
        wname = wells[idx_well]
        # срезы
        dt = dates_sorted[s:e]
        vdays = days_frac_sorted[s:e]
        oil0 = r_oil_sorted[s:e]
        wat0 = r_water_sorted[s:e]
        gas0 = r_gas_sorted[s:e]

        # Winsorize per series
        oil_w = _winsorize_arr(oil0, cfg.q_low, cfg.q_high)
        wat_w = _winsorize_arr(wat0, cfg.q_low, cfg.q_high)
        gas_w = _winsorize_arr(gas0, cfg.q_low, cfg.q_high)

        # Контролируемая интерполяция
        interp_counts = {"r_oil": 0, "r_water": 0, "r_gas": 0}
        if cfg.do_interpolate:
            oil_i, c1 = _interpolate_limited_np(oil_w, cfg.max_gap)
            wat_i, c2 = _interpolate_limited_np(wat_w, cfg.max_gap)
            gas_i, c3 = _interpolate_limited_np(gas_w, cfg.max_gap)
            interp_counts["r_oil"], interp_counts["r_water"], interp_counts["r_gas"] = c1, c2, c3
        else:
            oil_i, wat_i, gas_i = oil_w, wat_w, gas_w

        # Сглаживание: эрзац-интерполяция для остаточных NaN (ffill/bfill) только для фильтра
        def _prep_for_sg(x: np.ndarray) -> np.ndarray:
            y = x
            # быстрые ffill/bfill без pandas
            if np.isnan(y).any():
                # ffill
                last = np.nan
                for i in range(y.size):
                    if np.isnan(y[i]):
                        y[i] = last
                    else:
                        last = y[i]
                # bfill
                last = np.nan
                for i in range(y.size - 1, -1, -1):
                    if np.isnan(y[i]):
                        y[i] = last
                    else:
                        last = y[i]
                # если всё NaN, останутся NaN -> далее защита в _savgol_safe_arr
            return y

        oil_s = _savgol_safe_arr(_prep_for_sg(oil_i.copy()), cfg.sg_window, cfg.sg_poly)
        wat_s = _savgol_safe_arr(_prep_for_sg(wat_i.copy()), cfg.sg_window, cfg.sg_poly)
        gas_s = _savgol_safe_arr(_prep_for_sg(gas_i.copy()), cfg.sg_window, cfg.sg_poly)

        # Неотрицательность
        _enforce_nonneg_arr(oil_s, cfg.nonneg_mode)
        _enforce_nonneg_arr(wat_s, cfg.nonneg_mode)
        _enforce_nonneg_arr(gas_s, cfg.nonneg_mode)

        # Производные каналы
        wc = wat_s / (oil_s + wat_s + _EPS)
        gor = gas_s / (oil_s + _EPS)

        # Нормализация формы по oil_s
        oil_norm, denom = _normalize_series_arr(oil_s, cfg.normalize_by, cfg.norm_quantile, cfg.norm_topk, cfg.min_norm_denom)

        # Первая разность
        dr_oil_norm = np.empty_like(oil_norm)
        dr_oil_norm[0] = 0.0 if cfg.diff_fill0 else np.nan
        if oil_norm.size > 1:
            dr_oil_norm[1:] = np.diff(oil_norm)

        # Детект старта
        t0 = _detect_start_np(oil_s, cfg.start_abs_thr, cfg.start_rel_thr, cfg.start_min_consecutive, cfg.start_hysteresis)
        if t0 < 0:
            return None, dict(well_name=wname, reason="no_start_detected", valid_points=np.nan)

        # Качество
        mean_days = float(np.nanmean(vdays)) if np.isfinite(vdays).any() else np.nan
        large_gaps = _count_large_gaps(np.isnan(oil0), cfg.max_gap)
        peak = float(np.nanmax(oil_s)) if np.isfinite(oil_s).any() else 0.0
        p95 = float(np.nanpercentile(oil_s[~np.isnan(oil_s)], 95)) if np.isfinite(oil_s).any() else 0.0
        superpeak_ratio = (peak / (p95 + _EPS)) if p95 > 0 else np.nan
        has_superpeak = bool(superpeak_ratio > 1.5) if np.isfinite(superpeak_ratio) else False

        qrow = dict(
            well_name=wname,
            interp_count_oil=int(interp_counts["r_oil"]),
            interp_count_water=int(interp_counts["r_water"]),
            interp_count_gas=int(interp_counts["r_gas"]),
            mean_days_frac=float(mean_days) if mean_days == mean_days else np.nan,
            large_gaps=int(large_gaps),
            peak=float(peak),
            p95=float(p95),
            superpeak_ratio=float(superpeak_ratio) if np.isfinite(superpeak_ratio) else np.nan,
            has_superpeak=has_superpeak,
            norm_denom=float(denom),
            total_points=int(oil_s.size),
        )

        # Вырезка после t0 и формирование t
        oil_r = r_oil_sorted[s:e]  # для вывода r_* до сглаживания
        wat_r = r_water_sorted[s:e]
        gas_r = r_gas_sorted[s:e]

        # after
        oil_s_after = oil_s[t0:]
        n_after = oil_s_after.size
        t = np.arange(n_after, dtype=np.int32)
        # валидность времени — где oil_s не NaN
        valid = ~np.isnan(oil_s_after)

        # Минимальная длина
        if int(valid.sum()) < cfg.min_profile_len:
            return None, dict(well_name=wname, reason="too_short_after_t0", valid_points=int(valid.sum()))

        # Усечение до T
        Tcap = min(cfg.T, n_after)
        idx = slice(0, Tcap)

        data = {
            "well_name": np.repeat(wname, Tcap),
            "date": dt[t0: t0 + Tcap],
            "t": t[idx],
            "r_oil": oil_r[t0: t0 + Tcap],
            "r_water": wat_r[t0: t0 + Tcap],
            "r_gas": gas_r[t0: t0 + Tcap],
            "r_oil_s": oil_s[t0: t0 + Tcap],
            "r_water_s": wat_s[t0: t0 + Tcap],
            "r_gas_s": gas_s[t0: t0 + Tcap],
            "r_oil_norm": oil_norm[t0: t0 + Tcap],
            "wc": wc[t0: t0 + Tcap],
            "gor": gor[t0: t0 + Tcap],
            "dr_oil_norm": dr_oil_norm[t0: t0 + Tcap],
            "valid": valid[idx],
        }
        return pd.DataFrame(data), qrow

    # Параллельный запуск
    tasks = []
    for gi, (s, e) in enumerate(zip(group_starts, group_ends)):
        idx_well = int(well_codes_sorted[s])
        tasks.append((idx_well, s, e))

    if _HAVE_JOBLIB and cfg.n_jobs != 1:
        backend = "loky"  # процессы
        results = Parallel(n_jobs=cfg.n_jobs, backend=backend)(
            delayed(process_slice)(idx_w, s, e) for (idx_w, s, e) in tasks
        )
    else:
        results = [process_slice(idx_w, s, e) for (idx_w, s, e) in tasks]

    for res in results:
        panel_w, meta = res
        if panel_w is None:
            dropped_rows.append(meta)
        else:
            panel_chunks.append(panel_w)
            quality_rows.append(meta)

    if not panel_chunks:
        raise RuntimeError("После препроцессинга не осталось валидных скважин.")

    panel_long = pd.concat(panel_chunks, ignore_index=True)
    panel_long.rename(columns={"date": "month"}, inplace=True)

    quality_flags = pd.DataFrame(quality_rows, columns=[
        "well_name", "interp_count_oil", "interp_count_water", "interp_count_gas",
        "mean_days_frac", "large_gaps", "peak", "p95", "superpeak_ratio",
        "has_superpeak", "norm_denom", "total_points"
    ])
    dropped_df = pd.DataFrame(dropped_rows, columns=["well_name", "reason", "valid_points"])

    # ---- Сборка тензора X и mask_X
    channels = [c for c in cfg.tensor_channels if c in panel_long.columns]
    if not channels:
        raise ValueError("Нет доступных каналов для тензора X.")

    wells_valid = panel_long["well_name"].astype("category").cat.categories.tolist()
    well_codes_out = panel_long["well_name"].astype("category").cat.codes.to_numpy()
    N = len(wells_valid)
    Tcap = int(cfg.T)
    C = len(channels)

    X = np.zeros((N, Tcap, C), dtype=_FLOAT_DTYPE if cfg.dtype == "float32" else np.float64)
    mask_X = np.zeros((N, Tcap), dtype=bool)

    # быстрее заполнять по массивам
    t_vals = panel_long["t"].to_numpy()
    valid_vals = panel_long["valid"].to_numpy()
    for j, ch in enumerate(channels):
        vals = pd.to_numeric(panel_long[ch], errors="coerce").to_numpy()
        m = (t_vals >= 0) & (t_vals < Tcap) & np.isfinite(vals)
        if not m.any():
            continue
        ii = well_codes_out[m]
        tt = t_vals[m]
        X[ii, tt, j] = vals[m].astype(X.dtype, copy=False)

    # mask по valid
    m2 = (t_vals >= 0) & (t_vals < Tcap) & (valid_vals.astype(bool))
    ii = well_codes_out[m2]
    tt = t_vals[m2]
    mask_X[ii, tt] = True

    # Порядок и столбцы panel_long
    cols_order = ["well_name", "t", "month", "r_oil", "r_water", "r_gas",
                  "r_oil_s", "r_water_s", "r_gas_s",
                  "r_oil_norm", "wc", "gor", "dr_oil_norm", "valid"]
    keep_cols = [c for c in cols_order if c in panel_long.columns]
    panel_long = panel_long[keep_cols].copy()

    cfg_dict = asdict(cfg)
    logger.info("Сформировано: N=%d, T=%d, C=%d. Каналы: %s", N, Tcap, C, channels)

    # Приведение к требуемому dtype
    if cfg.dtype == "float32":
        X = X.astype(np.float32, copy=False)

    return panel_long, X, mask_X, wells_valid, quality_flags, dropped_df, cfg_dict


# -------------------------- Пример использования --------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    wells = ["A1", "B2", "C3"]
    months = pd.date_range("2020-01-01", periods=40, freq="MS")
    rows = []
    for w in wells:
        base = np.maximum(0, 100 * np.exp(-0.05 * np.arange(len(months))) + rng.normal(0, 5, len(months)))
        water = base * rng.uniform(0.3, 0.7)
        gas = base * rng.uniform(0.5, 1.5)
        for i, m in enumerate(months):
            rows.append(dict(
                well_name=w, date=m,
                oil=float(base[i] * 30),   # объём за месяц
                water=float(water[i] * 30),
                gas=float(gas[i] * 30),
                days_prod=int(rng.integers(20, 31)),
            ))
    df_demo = pd.DataFrame(rows)

    cfg = PreprocConfig(
        input_is_monthly_rates=False,
        days_min_fraction=0.7,
        T=24,
        normalize_by="p95",
        start_min_consecutive=2,
        max_gap=2,
        tensor_channels=("r_oil_norm", "wc", "gor", "dr_oil_norm"),
        n_jobs=-1,  # все ядра
    )
    panel_long, X, mask_X, wells_out, qf, dropped, cfg_used = preprocess_profiles(df_demo, cfg)
    print(panel_long.head())
    print("X shape:", X.shape, "mask shape:", mask_X.shape, "wells:", wells_out)
    print(qf.head())
    print(dropped.head())
