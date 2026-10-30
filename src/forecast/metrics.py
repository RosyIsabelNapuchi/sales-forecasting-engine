"""Métricas de error. WAPE elige el modelo; el error ANUAL es el que importa al presupuesto."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def _clean(actual: Sequence[float], pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    return a[valid], p[valid]


def wape(actual: Sequence[float], pred: Sequence[float]) -> float:
    a, p = _clean(actual, pred)
    if a.size == 0:
        return math.inf
    return float(np.abs(a - p).sum() / max(float(np.abs(a).sum()), 1e-9))


def smape(actual: Sequence[float], pred: Sequence[float]) -> float:
    a, p = _clean(actual, pred)
    if a.size == 0:
        return math.inf
    den = np.maximum(np.abs(a) + np.abs(p), 1e-9)
    return float(np.mean(2.0 * np.abs(a - p) / den))


def bias_pct(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Sesgo con signo: positivo = el modelo proyecta de más."""
    a, p = _clean(actual, pred)
    if a.size == 0:
        return math.nan
    return float((p - a).sum() / max(float(np.abs(a).sum()), 1e-9))


def annual_error(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Error del ACUMULADO: |suma pronóstico - suma real| / suma real.

    El WAPE mensual castiga que una venta caiga en marzo y no en abril. Para una meta
    anual eso se compensa dentro del año; este indicador mide el total del período.
    """
    a, p = _clean(actual, pred)
    if a.size == 0:
        return math.inf
    return float(abs(p.sum() - a.sum()) / max(abs(float(a.sum())), 1e-9))


def annual_bias(actual: Sequence[float], pred: Sequence[float]) -> float:
    a, p = _clean(actual, pred)
    if a.size == 0:
        return math.nan
    return float((p.sum() - a.sum()) / max(abs(float(a.sum())), 1e-9))


def annual_metrics(rows: pd.DataFrame, months: int = 12, pred_column: str = "pred_final") -> dict[str, float]:
    """Error anual por cada origen del backtest (primeros `months` meses de cada uno).

    Un solo origen puede salir bien por suerte; el promedio entre varios es lo que se puede prometer.
    """
    empty = {"error_anual_promedio": math.nan, "error_anual_holdout": math.nan,
             "error_anual_peor": math.nan, "sesgo_anual_promedio": math.nan, "origenes_anuales": 0}
    if rows.empty or pred_column not in rows:
        return empty
    window = rows[rows.horizon <= months]
    errors = []
    for _, group in window.groupby("origin_index", sort=True):
        if len(group) < months or abs(float(group.real.sum())) < 1e-9:
            continue
        errors.append((annual_error(group.real, group[pred_column]), annual_bias(group.real, group[pred_column])))
    if not errors:
        return empty
    values = [e for e, _ in errors]
    return {"error_anual_promedio": float(np.mean(values)), "error_anual_holdout": float(values[-1]),
            "error_anual_peor": float(np.max(values)), "sesgo_anual_promedio": float(np.mean([b for _, b in errors])),
            "origenes_anuales": len(errors)}
