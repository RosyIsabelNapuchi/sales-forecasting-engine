"""Registro de modelos: estadísticos, regresión de calendario, ML autoregresivo, intermitentes,
benchmarks (que nunca pueden ganar producción) y combinaciones.

Todos los modelos tienen la misma firma: fn(y, months, future) -> np.ndarray de len(future).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import wape

SEED = 42
ModelFn = Callable[[np.ndarray, pd.PeriodIndex, pd.PeriodIndex], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    complexity: int          # 0 benchmark · 1 simple · 2 intermedio · 3 complejo
    min_history: int
    fn: ModelFn
    intermittent_only: bool = False


# --------------------------------------------------------------------------- benchmarks
def m_naive(y, months, future):
    return np.repeat(float(y[-1]), len(future))


def _mean_window(window: int) -> ModelFn:
    def forecast(y, months, future):
        return np.repeat(float(np.mean(y[-window:])), len(future))
    return forecast


def m_seasonal_naive(y, months, future):
    if len(y) < 12:
        return _mean_window(min(6, len(y)))(y, months, future)
    hist, out = list(map(float, y)), []
    for _ in future:
        out.append(hist[-12])
        hist.append(hist[-12])
    return np.asarray(out)


# --------------------------------------------------------------------------- suavizamiento
def _ses_state(y: np.ndarray, alpha: float) -> tuple[float, float]:
    level, sse = float(y[0]), 0.0
    for value in y[1:]:
        sse += (float(value) - level) ** 2
        level = alpha * float(value) + (1.0 - alpha) * level
    return level, sse


def m_ses(y, months, future):
    states = [_ses_state(y, float(a)) for a in np.linspace(0.05, 0.95, 19)]
    level, _ = min(states, key=lambda s: s[1])
    return np.repeat(level, len(future))


def _holt_run(y, alpha, beta, phi):
    level = float(y[0])
    trend = float((np.mean(y[1:min(len(y), 7)]) - y[0]) / max(min(len(y), 6), 1))
    sse = 0.0
    for value in y[1:]:
        pred = level + phi * trend
        sse += (float(value) - pred) ** 2
        new_level = alpha * float(value) + (1.0 - alpha) * pred
        trend = beta * (new_level - level) + (1.0 - beta) * phi * trend
        level = new_level
    return level, trend, sse


def m_holt_damped(y, months, future):
    scale = max(float(np.mean(np.abs(y))), 1.0)
    res = minimize(lambda p: _holt_run(y / scale, *p)[2], x0=np.array([0.30, 0.10, 0.95]), method="L-BFGS-B",
                   bounds=[(0.01, 0.99), (0.01, 0.99), (0.80, 0.995)], options={"maxiter": 80})
    params = res.x if res.success else np.array([0.30, 0.10, 0.95])
    level, trend, _ = _holt_run(y, *params)
    phi, damp, out = float(params[2]), 0.0, []
    for step in range(1, len(future) + 1):
        damp += phi ** step
        out.append(level + damp * trend)
    return np.asarray(out)


def _hw_run(y, params):
    alpha, beta, gamma, phi = map(float, params)
    level = float(np.mean(y[:12]))
    trend = float((np.mean(y[12:24]) - level) / 12) if len(y) >= 24 else 0.0
    seasons = np.asarray(y[:12] - level, dtype=float).copy()
    sse = 0.0
    for pos in range(12, len(y)):
        idx = pos % 12
        seasonal = float(seasons[idx])
        pred = level + phi * trend + seasonal
        value = float(y[pos])
        sse += (value - pred) ** 2
        new_level = alpha * (value - seasonal) + (1.0 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1.0 - beta) * phi * trend
        seasons[idx] = gamma * (value - new_level) + (1.0 - gamma) * seasonal
        level = new_level
    return level, trend, seasons, sse


def m_holt_winters(y, months, future):
    scale = max(float(np.mean(np.abs(y))), 1.0)
    res = minimize(lambda p: _hw_run(y / scale, p)[3], x0=np.array([0.30, 0.08, 0.15, 0.95]), method="L-BFGS-B",
                   bounds=[(0.01, 0.99), (0.01, 0.70), (0.01, 0.70), (0.80, 0.995)], options={"maxiter": 100})
    params = res.x if res.success else np.array([0.30, 0.08, 0.15, 0.95])
    level, trend, seasons, _ = _hw_run(y, params)
    phi, damp, out = float(params[3]), 0.0, []
    for step in range(1, len(future) + 1):
        damp += phi ** step
        out.append(level + damp * trend + float(seasons[(len(y) + step - 1) % 12]))
    return np.asarray(out)


def m_theta(y, months, future):
    """Theta clásico: promedio de la recta de tendencia (theta-0) y SES sobre theta-2."""
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    theta2 = 2.0 * y - (intercept + slope * t)
    level = float(m_ses(theta2, months, future[:1])[0])
    t_future = np.arange(len(y), len(y) + len(future), dtype=float)
    return 0.5 * (intercept + slope * t_future) + 0.5 * level


# --------------------------------------------------------------------------- regresión de calendario
def _calendar(months: pd.PeriodIndex, start: int) -> np.ndarray:
    t = np.asarray([m.ordinal - start for m in months], dtype=float)
    month = np.asarray(months.month, dtype=float)
    cols = [t]
    for h in (1, 2, 3):
        cols += [np.sin(2 * np.pi * h * month / 12), np.cos(2 * np.pi * h * month / 12)]
    return np.column_stack(cols)


def m_ridge_calendar(y, months, future):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0 if len(y) < 60 else 30.0))
    model.fit(_calendar(months, months[0].ordinal), y)
    return np.asarray(model.predict(_calendar(future, months[0].ordinal)), dtype=float)


# --------------------------------------------------------------------------- ML autoregresivo
def _lag_row(hist: Sequence[float], month_number: int, t_value: float) -> list[float]:
    arr = np.asarray(hist, dtype=float)
    lags = [arr[-lag] if len(arr) >= lag else np.nan for lag in (1, 2, 3, 6, 12)]
    means = [np.mean(arr[-w:]) if len(arr) >= w else np.nan for w in (3, 6, 12)]
    std6 = np.std(arr[-6:]) if len(arr) >= 6 else np.nan
    return lags + means + [std6, t_value, np.sin(2 * np.pi * month_number / 12), np.cos(2 * np.pi * month_number / 12)]


def _recursive_ml(y, months, future, model):
    X, target = [], []
    for pos in range(12, len(y)):
        X.append(_lag_row(y[:pos], months[pos].month, pos / max(len(y), 1)))
        target.append(float(y[pos]))
    X, target = np.asarray(X, dtype=float), np.asarray(target, dtype=float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(target)
    if valid.sum() < 18:
        raise ValueError("Historia insuficiente para ML")
    model.fit(X[valid], target[valid])
    hist, out = list(map(float, y)), []
    for step, target_month in enumerate(future, start=1):
        value = float(model.predict(np.asarray([_lag_row(hist, target_month.month, (len(y) + step - 1) / len(y))]))[0])
        out.append(value)
        hist.append(value)
    return np.asarray(out)


def m_random_forest(y, months, future):
    return _recursive_ml(y, months, future, RandomForestRegressor(
        n_estimators=160, max_depth=5, min_samples_leaf=3, max_features=0.60, random_state=SEED, n_jobs=1))


def m_extra_trees(y, months, future):
    return _recursive_ml(y, months, future, ExtraTreesRegressor(
        n_estimators=120, max_depth=6, min_samples_leaf=3, max_features=0.80, random_state=SEED, n_jobs=1))


def m_hist_gbm(y, months, future):
    return _recursive_ml(y, months, future, HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.05, max_depth=3, min_samples_leaf=6, l2_regularization=10.0, random_state=SEED))


def m_gbm_huber(y, months, future):
    return _recursive_ml(y, months, future, GradientBoostingRegressor(
        loss="huber", n_estimators=120, learning_rate=0.04, max_depth=2, min_samples_leaf=4, random_state=SEED))


# --------------------------------------------------------------------------- demanda intermitente
def _croston_level(y, alpha=0.15):
    values = np.clip(np.asarray(y, dtype=float), 0.0, None)
    pos = np.flatnonzero(values > 0)
    if len(pos) == 0:
        return 0.0
    if len(pos) == 1:
        return float(values[pos[0]] / len(values))
    size, interval, last = float(values[pos[0]]), float(max(pos[1] - pos[0], 1)), int(pos[0])
    for p in pos[1:]:
        size = alpha * float(values[p]) + (1.0 - alpha) * size
        interval = alpha * float(p - last) + (1.0 - alpha) * interval
        last = int(p)
    return max(size / max(interval, 1e-9), 0.0)


def _croston_sba(y, alpha):
    return (1.0 - alpha / 2.0) * _croston_level(y, alpha)      # corrección de sesgo Syntetos-Boylan


def m_croston_sba(y, months, future):
    alpha = 0.15
    if len(y) >= 30:
        start = max(18, len(y) - 36)
        scored = [(wape(y[start:], [_croston_sba(y[:p], a) for p in range(start, len(y))]), a)
                  for a in (0.05, 0.10, 0.15, 0.20, 0.30)]
        alpha = min(scored)[1]
    return np.repeat(_croston_sba(y, alpha), len(future))


def _tsb_level(values, alpha, beta):
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    prob = float((values[:12] > 0).mean())
    positive = values[values > 0]
    size = float(positive[:3].mean()) if len(positive) else 0.0
    for v in values:
        occurred = float(v > 0)
        prob = alpha * occurred + (1.0 - alpha) * prob
        if occurred:
            size = beta * float(v) + (1.0 - beta) * size
    return max(prob * size, 0.0)


def m_tsb(y, months, future):
    alpha = beta = 0.15
    if len(y) >= 30:
        start = max(18, len(y) - 36)
        grid = (0.05, 0.10, 0.20, 0.30)
        scored = [(wape(y[start:], [_tsb_level(y[:p], a, b) for p in range(start, len(y))]), a, b) for a in grid for b in grid]
        _, alpha, beta = min(scored)
    return np.repeat(_tsb_level(y, alpha, beta), len(future))


# --------------------------------------------------------------------------- registro
MODEL_SPECS: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("ses", "suavizamiento", 1, 12, m_ses),
    ModelSpec("holt_damped", "suavizamiento", 1, 18, m_holt_damped),
    ModelSpec("holt_winters", "suavizamiento", 2, 24, m_holt_winters),
    ModelSpec("theta", "suavizamiento", 1, 18, m_theta),
    ModelSpec("ridge_calendar", "regresión", 2, 24, m_ridge_calendar),
    ModelSpec("random_forest", "ml", 3, 36, m_random_forest),
    ModelSpec("extra_trees", "ml", 3, 36, m_extra_trees),
    ModelSpec("hist_gbm", "ml", 3, 36, m_hist_gbm),
    ModelSpec("gbm_huber", "ml", 3, 36, m_gbm_huber),
    ModelSpec("croston_sba", "intermitente", 1, 24, m_croston_sba, True),
    ModelSpec("tsb", "intermitente", 1, 24, m_tsb, True),
]}

BENCHMARK_SPECS: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("benchmark_naive", "benchmark", 0, 2, m_naive),
    ModelSpec("benchmark_mean12", "benchmark", 0, 12, _mean_window(12)),
    ModelSpec("benchmark_seasonal_naive", "benchmark", 0, 12, m_seasonal_naive),
]}

COMBO_DEFS: dict[str, tuple[str, tuple[str, ...]]] = {
    "combo_suavizamiento": ("mean", ("ses", "holt_damped", "holt_winters", "theta")),
    "combo_robusto": ("median", ("holt_damped", "theta", "ridge_calendar", "extra_trees")),
    "combo_ml": ("median", ("random_forest", "extra_trees", "hist_gbm", "gbm_huber")),
    "combo_diverso": ("median", ("holt_damped", "theta", "ridge_calendar", "random_forest", "extra_trees")),
}

MODEL_COMPLEXITY = {n: s.complexity for n, s in {**MODEL_SPECS, **BENCHMARK_SPECS}.items()}
MODEL_COMPLEXITY.update({"combo_suavizamiento": 2, "combo_robusto": 2, "combo_ml": 3, "combo_diverso": 3})

MODEL_DESCRIPTIONS = {
    "benchmark_naive": "Benchmark: repite el último mes",
    "benchmark_mean12": "Benchmark: promedio de 12 meses",
    "benchmark_seasonal_naive": "Benchmark: mismo mes del año anterior",
    "ses": "Suavizado exponencial simple con alpha calibrado",
    "holt_damped": "Nivel y tendencia amortiguada",
    "holt_winters": "Holt-Winters aditivo amortiguado (solo si hay señal estacional)",
    "theta": "Método Theta",
    "ridge_calendar": "Ridge con tendencia y armónicos de calendario",
    "random_forest": "Random Forest autoregresivo",
    "extra_trees": "Extra Trees autoregresivo",
    "hist_gbm": "HistGradientBoosting autoregresivo",
    "gbm_huber": "Gradient Boosting con pérdida Huber (robusto a atípicos)",
    "croston_sba": "Croston SBA para demanda intermitente",
    "tsb": "TSB para demanda intermitente (probabilidad × tamaño)",
    "combo_suavizamiento": "Promedio de modelos de suavizamiento",
    "combo_robusto": "Mediana de Holt, Theta, Ridge y Extra Trees",
    "combo_ml": "Mediana de los modelos de ML",
    "combo_diverso": "Mediana de modelos de distintas familias",
}


def validate_registry() -> None:
    """Falla al importar si un combo usa un modelo inexistente o falta complejidad/descripción."""
    problems = []
    for combo, (_, members) in COMBO_DEFS.items():
        missing = [m for m in members if m not in MODEL_SPECS]
        if missing:
            problems.append(f"{combo}: miembros inexistentes {missing}")
    names = set(MODEL_SPECS) | set(BENCHMARK_SPECS) | set(COMBO_DEFS)
    if names - set(MODEL_COMPLEXITY):
        problems.append(f"sin complejidad: {sorted(names - set(MODEL_COMPLEXITY))}")
    if names - set(MODEL_DESCRIPTIONS):
        problems.append(f"sin descripción: {sorted(names - set(MODEL_DESCRIPTIONS))}")
    if problems:
        raise ValueError("Registro de modelos inconsistente: " + " | ".join(problems))


validate_registry()
