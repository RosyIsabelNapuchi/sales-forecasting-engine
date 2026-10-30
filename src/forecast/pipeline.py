"""Pipeline por serie (línea o total): modelo base → señal de cartera → variables externas → escenarios.

Cada capa extra debe ganarse su lugar dos veces:
  1) mejorar por un margen en la parte de evaluación del desarrollo, y
  2) confirmarse en una ventana de holdout que NO participó en ningún ajuste.
Si no, se descarta y se registra el motivo. Al final, las líneas se reconcilian con el total.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .exogenous import ExogProvider, apply_exog_adjuster, fit_exog_adjuster
from .metrics import annual_metrics, bias_pct, wape
from .order_book import TOTAL_SCOPE, OrderBook, apply_ov_params, fit_ov_params
from .selection import choose_origins, rank_models, rolling_backtest, safe_predict, series_profile

LOG = logging.getLogger("forecast")


@dataclass(frozen=True)
class Config:
    min_train_long: int = 36
    min_train_short: int = 24
    cv_step: int = 3
    model_margin: float = 0.03              # un retador debe mejorar 3 % a la referencia estadística
    confirmation_margin: float = 0.02       # tolerancia de empate en la confirmación
    signal_confirmation_margin: float = 0.01
    ov_margin: float = 0.05                 # la cartera debe mejorar 5 % en desarrollo
    exog_margin: float = 0.08               # las exógenas deben mejorar 8 % (más exigente: más riesgo de sobreajuste)
    ov_min_coverage: float = 0.25
    ov_max_multiple: float = 2.0
    exog_max_vars: int = 3
    scenario_low_q: float = 0.10
    scenario_high_q: float = 0.90


@dataclass
class ScopeResult:
    scope: str
    history: pd.Series
    forecast: pd.DataFrame
    diagnostics: dict
    ranking: pd.DataFrame
    evaluation: pd.DataFrame = field(repr=False)


def confidence_label(wape_holdout: float, history_length: int, folds: int) -> str:
    if history_length >= 72 and folds >= 6 and wape_holdout <= 0.15:
        return "ALTA"
    if history_length >= 48 and folds >= 4 and wape_holdout <= 0.30:
        return "MEDIA"
    return "BAJA"


def add_scenarios(forecast: pd.DataFrame, evaluation: pd.DataFrame, history_scale: float, cfg: Config) -> pd.DataFrame:
    """Escenarios pesimista/optimista con los cuantiles del error relativo por horizonte (anchos no decrecientes)."""
    frame = forecast.copy()
    res = evaluation.copy()
    floor = max(history_scale * 0.15, 1.0)
    res["rel_error"] = ((res.real - res.pred_final) / np.maximum(np.abs(res.pred_final), floor)).clip(-0.8, 2.0)
    pooled_lo = float(res.rel_error.quantile(cfg.scenario_low_q)) if len(res) else -0.2
    pooled_hi = float(res.rel_error.quantile(cfg.scenario_high_q)) if len(res) else 0.2
    lows, highs = [], []
    for h in frame.horizon:
        v = res.loc[res.horizon == h, "rel_error"].dropna()
        lows.append(max(0.0, -(float(v.quantile(cfg.scenario_low_q)) if len(v) >= 4 else pooled_lo)))
        highs.append(max(0.0, float(v.quantile(cfg.scenario_high_q)) if len(v) >= 4 else pooled_hi))
    lows, highs = np.maximum.accumulate(lows), np.maximum.accumulate(highs)
    frame["pesimista"] = np.clip(frame.normal * (1 - lows), 0, None)
    frame["optimista"] = np.clip(frame.normal * (1 + highs), 0, None)
    return frame


def _rows_for(backtest: pd.DataFrame, model: str, book: OrderBook, scope: str) -> pd.DataFrame:
    frame = backtest[backtest.model == model].sort_values(["origin_index", "horizon"]).rename(columns={"pred": "pred_base"})
    return book.enrich(frame, scope)


def process_scope(scope: str, series: pd.Series, future: pd.PeriodIndex, book: OrderBook,
                  exog: ExogProvider, cfg: Config) -> ScopeResult:
    LOG.info("Procesando %s", "TOTAL" if scope == TOTAL_SCOPE else scope)
    y = series.to_numpy(dtype=float)
    months = pd.PeriodIndex(series.index, freq="M")
    horizon = len(future)
    origins, holdout_origin = choose_origins(len(series), horizon, cfg.min_train_long, cfg.min_train_short, cfg.cv_step)

    # ---------- 1. Torneo de modelos en desarrollo
    dev_bt, failures = rolling_backtest(scope, series, origins, horizon)
    rank, selected, reference, reason = rank_models(dev_bt, len(origins), cfg.model_margin)
    dev_selected = selected
    hold_bt, _ = rolling_backtest(scope, series, [holdout_origin], horizon, model_names=list(rank.model))
    hold_wape = {m: wape(g.real, g.pred) for m, g in hold_bt.groupby("model")}
    rank["wape_holdout"] = rank.model.map(hold_wape)

    # ---------- 2. Confirmación: solo desempata entre los empatados en desarrollo (no reabre el torneo)
    prod = rank[~rank.benchmark_only]
    best = prod.iloc[0]
    shortlist = prod[prod.score <= float(best.score) + float(best.score_se) + 1e-12].copy()
    shortlist = shortlist[np.isfinite(shortlist.wape_holdout)]
    confirmed = (str(shortlist.sort_values(["wape_holdout", "score", "complexity"]).iloc[0].model)
                 if not shortlist.empty else dev_selected)
    ref_hold = float(hold_wape.get(reference, math.inf))
    if confirmed == reference or float(hold_wape.get(confirmed, math.inf)) <= ref_hold * (1 + cfg.confirmation_margin):
        selected = confirmed
        if selected != dev_selected:
            reason += f"; la confirmación eligió {selected} dentro de los empatados"
    else:
        selected = reference
        reason += f"; retador rechazado en confirmación: empeora más de {cfg.confirmation_margin:.0%} vs {reference}"
    rank["seleccionado"] = rank.model.eq(selected)

    dev_rows = _rows_for(dev_bt, selected, book, scope)
    hold_rows = _rows_for(hold_bt, selected, book, scope)
    dev_origins = sorted(dev_rows.origin_index.unique())
    split = max(2, min(len(dev_origins) - 1, math.ceil(len(dev_origins) * 0.60)))
    eval_origins = set(dev_origins[split:])
    eval_start = series.index[min(eval_origins) - 1]
    stage_train = dev_rows[dev_rows.target_period <= eval_start].copy()        # nada del período de evaluación
    stage_eval = dev_rows[dev_rows.origin_index.isin(eval_origins)].copy()

    # ---------- 3. Señal de cartera de pedidos
    ov_train = fit_ov_params(stage_train, cfg.ov_min_coverage, cfg.ov_max_multiple)
    stage_train["pred_stage"], _ = apply_ov_params(stage_train, ov_train, cfg.ov_max_multiple)
    ov_eval_pred, ov_eval_applied = apply_ov_params(stage_eval, ov_train, cfg.ov_max_multiple)
    base_eval_wape = wape(stage_eval.real, stage_eval.pred_base)
    ov_improvement = 0.0
    if ov_eval_applied.any():
        ov_improvement = 1 - wape(stage_eval.real[ov_eval_applied], ov_eval_pred[ov_eval_applied]) / max(
            wape(stage_eval.real[ov_eval_applied], stage_eval.pred_base[ov_eval_applied]), 1e-9)
    ov_pass = bool(ov_train and any(w > 0 for w in ov_train["weights"].values()) and ov_eval_applied.sum() >= 4
                   and ov_improvement >= cfg.ov_margin and wape(stage_eval.real, ov_eval_pred) <= base_eval_wape * 1.01)
    ov_dev = fit_ov_params(dev_rows, cfg.ov_min_coverage, cfg.ov_max_multiple)
    hold_ov_pred, hold_ov_applied = apply_ov_params(hold_rows, ov_dev, cfg.ov_max_multiple)
    ov_conf = 1 - wape(hold_rows.real, hold_ov_pred) / max(wape(hold_rows.real, hold_rows.pred_base), 1e-9)
    ov_accepted = bool(ov_pass and hold_ov_applied.any() and ov_conf >= cfg.signal_confirmation_margin)
    if ov_accepted:
        ov_reason = "aceptada"
    elif not ov_train:
        ov_reason = "sin cobertura calibrable"
    elif not any(w > 0 for w in ov_train["weights"].values()):
        ov_reason = "peso óptimo 0 en todos los horizontes"
    elif ov_eval_applied.sum() < 4:
        ov_reason = f"solo {int(ov_eval_applied.sum())} meses elegibles"
    elif ov_improvement < cfg.ov_margin:
        ov_reason = f"mejora en desarrollo {ov_improvement:.1%} < margen {cfg.ov_margin:.0%}"
    else:
        ov_reason = f"no se confirmó en holdout ({ov_conf:.1%})"
    if not ov_accepted:
        stage_train["pred_stage"] = stage_train.pred_base
    stage_eval["pred_stage"] = ov_eval_pred if ov_accepted else stage_eval.pred_base.to_numpy()
    hold_rows["pred_stage"] = hold_ov_pred if ov_accepted else hold_rows.pred_base.to_numpy()
    dev_rows["pred_stage"] = apply_ov_params(dev_rows, ov_dev, cfg.ov_max_multiple)[0] if ov_accepted else dev_rows.pred_base

    # ---------- 4. Variables externas públicas
    ex_train = fit_exog_adjuster(stage_train, exog, scope, cfg.exog_max_vars)
    ex_eval_pred, ex_eval_delta = apply_exog_adjuster(stage_eval, exog, ex_train)
    ex_improvement = 1 - wape(stage_eval.real, ex_eval_pred) / max(wape(stage_eval.real, stage_eval.pred_stage), 1e-9)
    ex_pass = bool(ex_train and np.count_nonzero(np.abs(ex_eval_delta) > 1e-6) >= 4 and ex_improvement >= cfg.exog_margin)
    ex_dev = fit_exog_adjuster(dev_rows, exog, scope, cfg.exog_max_vars)
    hold_ex_pred, hold_ex_delta = apply_exog_adjuster(hold_rows, exog, ex_dev)
    ex_conf = 1 - wape(hold_rows.real, hold_ex_pred) / max(wape(hold_rows.real, hold_rows.pred_stage), 1e-9)
    ex_accepted = bool(ex_pass and np.count_nonzero(np.abs(hold_ex_delta) > 1e-6) >= 1 and ex_conf >= cfg.signal_confirmation_margin)
    stage_eval["pred_final"] = ex_eval_pred if ex_accepted else stage_eval.pred_stage.to_numpy()
    hold_rows["pred_final"] = hold_ex_pred if ex_accepted else hold_rows.pred_stage.to_numpy()

    # ---------- 5. Reentrenamiento con toda la historia y pronóstico
    final_base, state = safe_predict(selected, y, months, future, allowed=set(rank.model))
    if final_base is None:
        raise RuntimeError(f"{scope}: {selected} falló al reentrenar ({state})")
    calib = pd.concat([dev_rows.drop(columns="pred_stage"), hold_rows.drop(columns=["pred_stage", "pred_final"])],
                      ignore_index=True)
    ov_final = fit_ov_params(calib, cfg.ov_min_coverage, cfg.ov_max_multiple)
    current = pd.DataFrame({"origin_period": [series.index[-1]] * horizon, "target_period": future,
                            "horizon": np.arange(1, horizon + 1), "pred_base": final_base,
                            "booked_ov": book.current_amounts(scope, future)})
    cur_ov_pred, cur_ov_applied = apply_ov_params(current, ov_final, cfg.ov_max_multiple)
    current["pred_stage"] = cur_ov_pred if ov_accepted else final_base
    calib["pred_stage"] = apply_ov_params(calib, ov_final, cfg.ov_max_multiple)[0] if ov_accepted else calib.pred_base
    ex_final = fit_exog_adjuster(calib, exog, scope, cfg.exog_max_vars)
    cur_ex_pred, cur_ex_delta = apply_exog_adjuster(current, exog, ex_final)
    normal = cur_ex_pred if ex_accepted else current.pred_stage.to_numpy()

    forecast = pd.DataFrame({
        "scope": scope, "mes": future, "horizon": np.arange(1, horizon + 1), "modelo": selected,
        "base_estadistica": final_base, "cartera_abierta": current.booked_ov.to_numpy(),
        "despues_cartera": current.pred_stage.to_numpy(), "ajuste_exogeno": cur_ex_delta if ex_accepted else 0.0,
        "normal": np.clip(normal, 0, None), "cartera_aplicada": cur_ov_applied if ov_accepted else False})
    evaluation = pd.concat([stage_eval, hold_rows], ignore_index=True)
    forecast = add_scenarios(forecast, evaluation, float(np.median(np.abs(series.tail(24)))), cfg)

    profile = series_profile(y, months)
    hold_final_wape = wape(hold_rows.real, hold_rows.pred_final)
    added = rank.loc[rank.model == selected, "valor_agregado_vs_benchmark"]
    diagnostics = {
        "scope": scope, "historia_meses": len(series), "cv": profile["cv"], "fuerza_estacional": profile["seasonal_strength"],
        "pct_meses_cero": profile["zero_rate"], "tasa_atipicos": profile["outlier_rate"],
        "modelo": selected, "modelo_desarrollo": dev_selected, "modelo_referencia": reference, "razon": reason,
        "valor_agregado_vs_benchmark": float(added.iloc[0]) if len(added) else math.nan,
        "folds": len(origins), "wape_desarrollo": float(rank.loc[rank.model == selected, "wape_dev"].iloc[0]),
        "cartera_aceptada": ov_accepted, "cartera_motivo": ov_reason, "cartera_mejora_desarrollo": ov_improvement,
        "cartera_mejora_confirmacion": ov_conf,
        "exogenas_aceptadas": ex_accepted, "exogenas_mejora_desarrollo": ex_improvement, "exogenas_mejora_confirmacion": ex_conf,
        "grupos_exogenos": "+".join(exog.groups_for(scope)),
        "drivers": ", ".join(ex_final["features"]) if (ex_final and ex_accepted) else "",
        "wape_holdout_final": hold_final_wape, "sesgo_holdout": bias_pct(hold_rows.real, hold_rows.pred_final),
        **annual_metrics(evaluation),
        "confianza": confidence_label(hold_final_wape, len(series), len(origins)),
        "fallas_modelos": sum(failures.values()),
    }
    return ScopeResult(scope, series, forecast, diagnostics, rank.assign(scope=scope), evaluation.assign(scope=scope))


def reconcile(total: ScopeResult, lines: list[ScopeResult], other_share: float) -> pd.DataFrame:
    """Proporcional: ajusta las líneas (+ 'otras') para que cada mes sumen exactamente el total."""
    tot = total.forecast.set_index("mes")
    detail = pd.concat([r.forecast.assign(linea=r.scope) for r in lines], ignore_index=True)
    other = pd.DataFrame({"mes": tot.index, "horizon": tot.horizon.values, "modelo": "participacion_reciente",
                          "normal": tot.normal.values * other_share, "pesimista": tot.pesimista.values * other_share,
                          "optimista": tot.optimista.values * other_share, "linea": "OTRAS"})
    detail = pd.concat([detail, other], ignore_index=True)
    factors = tot.normal / detail.groupby("mes").normal.sum().replace(0, np.nan)
    if factors.isna().any():
        raise ValueError("No se puede reconciliar: la suma de líneas es cero")
    detail["factor_reconciliacion"] = detail.mes.map(factors)
    detail["normal_reconciliado"] = detail.normal * detail.factor_reconciliacion
    for col in ("pesimista", "optimista"):
        ratio = (tot[col] / tot.normal.replace(0, np.nan)).fillna(1.0)
        detail[f"{col}_reconciliado"] = detail.normal_reconciliado * detail.mes.map(ratio)
    return detail
