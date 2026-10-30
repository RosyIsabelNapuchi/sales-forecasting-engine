"""Perfil de la serie, backtest de origen móvil y selección de modelo.

Reglas de selección (en orden):
1. Puntaje = 0.55·WAPE ponderado por recencia + 0.20·error acumulado + 0.10·inestabilidad + 0.15·|sesgo|.
2. Regla de un error estándar: entre los modelos estadísticamente empatados con el mejor, gana el MÁS SIMPLE.
3. Margen contra la referencia estadística (SES / Holt / Theta): un modelo más complejo solo entra si
   mejora el puntaje al menos `margin`. Si no, se queda la referencia.
4. Los benchmarks se evalúan siempre, pero nunca pueden ganar: sirven para medir el valor agregado.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import bias_pct, smape, wape
from .models import BENCHMARK_SPECS, COMBO_DEFS, MODEL_COMPLEXITY, MODEL_SPECS

REFERENCE_MODELS = ("ses", "holt_damped", "theta")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 12 or np.std(a[valid]) < 1e-9 or np.std(b[valid]) < 1e-9:
        return 0.0
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def series_profile(y: np.ndarray, months: pd.PeriodIndex) -> dict[str, float]:
    """¿Qué forma tiene la serie? Variabilidad, ceros, autocorrelación, estacionalidad y atípicos."""
    values = np.asarray(y, dtype=float)
    mean_abs = max(float(np.mean(np.abs(values))), 1e-9)
    month_effect = pd.Series(values).groupby(np.asarray(months.month)).transform("mean").to_numpy()
    total_var = float(np.var(values))
    seasonal_strength = 0.0 if total_var < 1e-9 else float(np.clip(1 - np.var(values - month_effect) / total_var, 0, 1))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = float(q3 - q1)
    outliers = 0.0 if iqr <= 1e-9 else float(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean())
    return {
        "cv": float(np.std(values) / mean_abs),
        "zero_rate": float((values <= 0).mean()),
        "lag1": _safe_corr(values[1:], values[:-1]) if len(values) >= 13 else 0.0,
        "lag12": _safe_corr(values[12:], values[:-12]) if len(values) >= 24 else 0.0,
        "seasonal_strength": seasonal_strength,
        "outlier_rate": outliers,
    }


def candidate_model_names(y: np.ndarray, months: pd.PeriodIndex) -> list[str]:
    """Qué modelos tiene sentido probar según la forma de la serie (no se fuerza estacionalidad si no hay)."""
    profile = series_profile(y, months)
    names = [n for n in ("ses", "holt_damped", "theta", "ridge_calendar", "random_forest", "extra_trees")
             if len(y) >= MODEL_SPECS[n].min_history]
    if len(y) >= 24 and (profile["seasonal_strength"] >= 0.15 or abs(profile["lag12"]) >= 0.25):
        names.append("holt_winters")
    if len(y) >= 48 and profile["zero_rate"] < 0.25:
        names.append("hist_gbm")
    if len(y) >= 36 and (profile["outlier_rate"] >= 0.05 or profile["cv"] >= 0.60):
        names.append("gbm_huber")
    if profile["zero_rate"] >= 0.10:
        names += ["croston_sba", "tsb"]
    names += list(COMBO_DEFS)
    names += [n for n, s in BENCHMARK_SPECS.items() if len(y) >= s.min_history]
    return list(dict.fromkeys(names))


def safe_predict(name: str, y: np.ndarray, months: pd.PeriodIndex, future: pd.PeriodIndex,
                 allowed: set[str] | None = None) -> tuple[np.ndarray | None, str]:
    """Ejecuta un modelo y rechaza salidas inválidas en vez de romper el pipeline.

    Para un combo, `allowed` limita los miembros a los mismos que se evaluaron en el backtest.
    """
    if name in COMBO_DEFS:
        method, members = COMBO_DEFS[name]
        members = [m for m in members if allowed is None or m in allowed]
        preds = [p for p, _ in (safe_predict(m, y, months, future) for m in members) if p is not None]
        if len(preds) < 2:
            return None, "miembros_insuficientes"
        stack = np.vstack(preds)
        return (stack.mean(axis=0) if method == "mean" else np.median(stack, axis=0)), "ok"
    try:
        spec = MODEL_SPECS.get(name) or BENCHMARK_SPECS.get(name)
        if spec is None:
            return None, f"modelo_no_definido: {name}"
        pred = np.asarray(spec.fn(y.copy(), months, future), dtype=float)
        if pred.shape != (len(future),):
            return None, "longitud_invalida"
        if not np.isfinite(pred).all():
            return None, "no_finito"
        if np.max(np.abs(pred)) > 1e12:
            return None, "magnitud_invalida"
        return np.clip(pred, 0.0, None), "ok"
    except Exception as exc:                                    # noqa: BLE001 - se reporta como estado
        return None, f"{type(exc).__name__}: {str(exc)[:120]}"


def predictions_at_origin(y, months, origin, horizon, model_names):
    history, hist_months, future = y[:origin], months[:origin], months[origin:origin + horizon]
    preds, status = {}, {}
    for name in model_names:
        if name in COMBO_DEFS:
            continue
        pred, state = safe_predict(name, history, hist_months, future)
        status[name] = state
        if pred is not None:
            preds[name] = pred
    for name, (method, members) in COMBO_DEFS.items():
        if name not in model_names:
            continue
        available = [preds[m] for m in members if m in preds]
        if len(available) < 2:                                    # un ensamble exige 2 miembros reales
            status[name] = "miembros_insuficientes"
            continue
        stack = np.vstack(available)
        preds[name] = stack.mean(axis=0) if method == "mean" else np.median(stack, axis=0)
        status[name] = "ok"
    return preds, status


def rolling_backtest(scope: str, series: pd.Series, origins: Sequence[int], horizon: int,
                     model_names: Sequence[str] | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    y = series.to_numpy(dtype=float)
    months = pd.PeriodIndex(series.index, freq="M")
    model_names = list(model_names) if model_names is not None else candidate_model_names(y, months)
    rows, failures = [], {n: 0 for n in model_names}
    for origin in origins:
        preds, _ = predictions_at_origin(y, months, origin, horizon, model_names)
        for name in model_names:
            if name not in preds:
                failures[name] += 1
                continue
            for step, (target, actual, pred) in enumerate(
                    zip(months[origin:origin + horizon], y[origin:origin + horizon], preds[name]), start=1):
                rows.append({"scope": scope, "model": name, "origin_index": origin, "origin_period": months[origin - 1],
                             "target_period": target, "horizon": step, "real": float(actual), "pred": float(pred)})
    return pd.DataFrame(rows), failures


def choose_origins(n: int, horizon: int, min_train_long: int = 36, min_train_short: int = 24,
                   step: int = 3) -> tuple[list[int], int]:
    """Orígenes de desarrollo + un origen final de confirmación (holdout) que no participa en el ranking."""
    holdout_origin = n - horizon
    max_dev = holdout_origin - horizon
    if max_dev < min_train_short:
        raise ValueError(f"Historia insuficiente: {n} meses para horizonte {horizon}")
    min_train = min_train_long if max_dev >= min_train_long else min_train_short
    origins = list(range(min_train, max_dev + 1, step))
    if len(origins) < 3:
        origins = sorted(set(np.linspace(min_train, max_dev, num=3, dtype=int)))
    return origins, holdout_origin


def rank_models(backtest: pd.DataFrame, expected_folds: int, margin: float) -> tuple[pd.DataFrame, str, str, str]:
    rows = []
    for model, group in backtest.groupby("model"):
        folds = int(group.origin_index.nunique())
        if folds < max(2, math.ceil(expected_folds * 0.80)):
            continue
        per_fold = []
        for origin, fold in group.groupby("origin_index"):
            cumulative = abs(float(fold.pred.sum() - fold.real.sum())) / max(float(np.abs(fold.real).sum()), 1e-9)
            per_fold.append((int(origin), wape(fold.real, fold.pred), cumulative, abs(bias_pct(fold.real, fold.pred))))
        per_fold.sort()
        w = np.linspace(0.55, 1.00, len(per_fold))                # las ventanas recientes pesan más
        w = w / w.sum()
        w_wape = float(np.average([m[1] for m in per_fold], weights=w))
        w_cum = float(np.average([m[2] for m in per_fold], weights=w))
        w_bias = float(np.average([m[3] for m in per_fold], weights=w))
        stability = float(np.std([m[1] for m in per_fold]))
        fold_scores = np.asarray([0.65 * m[1] + 0.20 * m[2] + 0.15 * m[3] for m in per_fold])
        rows.append({
            "model": model,
            "benchmark_only": model in BENCHMARK_SPECS,
            "complexity": MODEL_COMPLEXITY.get(model, 3),
            "folds": folds,
            "wape_dev": wape(group.real, group.pred),
            "smape_dev": smape(group.real, group.pred),
            "bias_dev": bias_pct(group.real, group.pred),
            "score": 0.55 * w_wape + 0.20 * w_cum + 0.10 * stability + 0.15 * w_bias,
            "score_se": float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores))) if len(fold_scores) > 1 else 0.0,
        })
    rank = pd.DataFrame(rows)
    if rank.empty:
        raise ValueError("Ningún modelo completó suficientes pliegues")
    rank = rank.sort_values(["score", "complexity", "model"]).reset_index(drop=True)
    production = rank[~rank.benchmark_only]
    benchmarks = rank[rank.benchmark_only]

    refs = production[production.model.isin(REFERENCE_MODELS)]
    reference = str((refs if not refs.empty else production).iloc[0].model)
    best = production.iloc[0]
    near_best = production[production.score <= float(best.score) + float(best.score_se) + 1e-12]
    chosen_row = near_best.sort_values(["complexity", "score", "model"]).iloc[0]     # regla de 1 error estándar
    chosen = str(chosen_row.model)

    ref_score = float(production.loc[production.model == reference, "score"].iloc[0])
    improvement = 1.0 - float(chosen_row.score) / max(ref_score, 1e-9)
    if chosen != reference and improvement < margin:
        chosen = reference
        reason = f"se retiene la referencia {reference}: el retador mejora {improvement:.1%} < margen {margin:.0%}"
    else:
        reason = "mejor equilibrio entre WAPE reciente, error acumulado, estabilidad y sesgo"

    if not benchmarks.empty:
        bench_name, bench_score = str(benchmarks.iloc[0].model), float(benchmarks.iloc[0].score)
        rank["mejor_benchmark"] = bench_name
        rank["valor_agregado_vs_benchmark"] = 1.0 - rank.score / max(bench_score, 1e-9)
        added = float(rank.loc[rank.model == chosen, "valor_agregado_vs_benchmark"].iloc[0])
        reason += (f"; ALERTA: no supera al benchmark {bench_name} ({added:.1%})" if added < 0
                   else f"; mejora {added:.1%} vs benchmark {bench_name}")
    return rank, chosen, reference, reason
