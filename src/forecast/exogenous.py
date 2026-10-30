"""Variables externas públicas (BCRP) como ajuste del residuo, sin fuga temporal.

- Cada línea solo puede usar los indicadores de los sectores a los que vende (mezcla sectorial).
- En un origen t solo se usa lo PUBLICADO hasta t - rezago_publicación (el BCRP publica con ~2 meses).
- El efecto se modela con rezago (por defecto 3 meses): un mes de pesca alta hoy mueve la venta después.
- El ajuste es un Ridge sobre el residuo del modelo base, con tope de ±35 %, y solo entra si
  mejora por un margen en desarrollo y se confirma en la ventana de holdout.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

LEVEL_COLUMNS_PREFIX = ("expect_",)
LEVEL_COLUMNS = {"cemento_yoy", "inflacion_yoy", "inversion_minera_yoy"}

GROUPS = {
    "total": ["expect_demanda_12m", "expect_economia_12m", "expect_sector_12m", "pbi_total", "pbi_mineria",
              "pbi_manufactura_no_primaria", "precio_cobre", "anchoveta_tn", "fx_usdpen"],
    "mineria": ["expect_demanda_12m", "expect_sector_12m", "inversion_minera_yoy", "pbi_mineria", "prod_cobre",
                "precio_cobre", "precio_oro", "precio_zinc", "import_bienes_capital", "fx_usdpen"],
    "pesca": ["anchoveta_tn", "pbi_pesca", "pesca_congelado_tn", "pesca_conservas_tn", "precio_harina_pescado",
              "expect_sector_12m", "fx_usdpen"],
    "industria": ["expect_demanda_12m", "expect_economia_12m", "pbi_manufactura_no_primaria", "electricidad_sin_mineria_gwh",
                  "import_insumos", "pbi_total", "fx_usdpen"],
    "construccion": ["pbi_construccion", "cemento_yoy", "expect_sector_12m", "import_bienes_capital", "pbi_total", "fx_usdpen"],
}
SECTOR_TO_GROUP = {"MINERIA": "mineria", "PESCA": "pesca", "INDUSTRIA": "industria", "CONSTRUCCION": "construccion"}


class ExogProvider:
    def __init__(self, raw: pd.DataFrame, sector_mix: dict[str, list[tuple[str, float]]],
                 effect_lag: int = 3, release_lag: int = 2, min_share: float = 0.12, max_secondary: int = 2):
        raw = raw.copy()
        raw["mes"] = pd.PeriodIndex(raw.mes, freq="M")
        raw = raw.drop_duplicates("mes", keep="last").set_index("mes").sort_index()
        self.features = pd.DataFrame(index=raw.index)
        self.feature_to_driver: dict[str, str] = {}
        for column in raw.columns:
            values = pd.to_numeric(raw[column], errors="coerce")
            if values.notna().sum() < 36:
                continue
            if column in LEVEL_COLUMNS or column.startswith(LEVEL_COLUMNS_PREFIX):
                transformed, name = values, f"{column}__nivel"
            else:
                transformed = values.pct_change(12, fill_method=None).replace([np.inf, -np.inf], np.nan)
                name = f"{column}__yoy"
            lo, hi = transformed.quantile([0.01, 0.99])
            self.features[name] = transformed.clip(lo, hi)
            self.feature_to_driver[name] = column
        self.sector_mix, self.effect_lag, self.release_lag = sector_mix, effect_lag, release_lag
        self.min_share, self.max_secondary = min_share, max_secondary

    def groups_for(self, scope: str) -> tuple[str, ...]:
        if scope not in self.sector_mix:
            return ("total",)
        mix = self.sector_mix[scope]
        used = [mix[0]] + [(s, v) for s, v in mix[1:] if v >= self.min_share][: self.max_secondary]
        return tuple(dict.fromkeys(SECTOR_TO_GROUP[s] for s, _ in used if s in SECTOR_TO_GROUP)) or ("total",)

    def candidate_features(self, scope: str) -> list[str]:
        allowed = {d for g in self.groups_for(scope) for d in GROUPS[g]}
        return [f for f, d in self.feature_to_driver.items() if d in allowed]

    def value(self, feature: str, origin: pd.Period, target: pd.Period) -> float:
        """Valor del indicador para `target`, usando SOLO datos publicados al momento `origin`."""
        series = self.features[feature]
        available = origin - self.release_lag
        lookup = target - self.effect_lag
        if lookup > available:
            # Aún no publicado: se usa el último dato disponible. Las variaciones interanuales y los
            # índices de expectativas ya no tienen estacionalidad, así que el último valor es mejor
            # predictor que el del mismo mes del año anterior.
            lookup = available
        if lookup in series.index and pd.notna(series.loc[lookup]):
            return float(series.loc[lookup])
        prior = series.loc[series.index <= min(lookup, available)].dropna()
        if prior.empty:
            return math.nan
        same = prior[prior.index.month == lookup.month]
        return float(same.iloc[-1] if not same.empty else prior.tail(12).median())

    def matrix(self, rows: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        return np.asarray([[self.value(f, o, t) for f in features] for o, t in zip(rows.origin_period, rows.target_period)])


def fit_exog_adjuster(rows: pd.DataFrame, provider: ExogProvider, scope: str, max_vars: int = 3) -> dict | None:
    features = provider.candidate_features(scope)
    if len(rows) < 18 or not features:
        return None
    X = provider.matrix(rows, features)
    residual = rows.real.to_numpy(dtype=float) - rows.pred_stage.to_numpy(dtype=float)
    valid = np.isfinite(residual) & np.isfinite(X).all(axis=1)
    if valid.sum() < 18 or np.std(residual[valid]) < 1e-6:
        return None
    Xv, rv = X[valid], residual[valid]
    corr = [0.0 if np.std(Xv[:, j]) < 1e-9 else abs(float(np.corrcoef(Xv[:, j], rv)[0, 1])) for j in range(Xv.shape[1])]
    selected: list[int] = []
    for j in np.argsort(np.nan_to_num(corr))[::-1]:
        if corr[j] < 0.10:
            continue
        if any(abs(float(np.corrcoef(Xv[:, j], Xv[:, k])[0, 1])) > 0.90 for k in selected):   # evita redundancia
            continue
        selected.append(int(j))
        if len(selected) >= max_vars:
            break
    if not selected:
        return None
    Xs = Xv[:, selected]
    mean, std = Xs.mean(axis=0), Xs.std(axis=0)
    std[std < 1e-9] = 1.0
    Z = (Xs - mean) / std
    split = min(max(12, int(len(Z) * 0.75)), len(Z) - 3)
    alpha = min(((float(np.mean(np.abs(rv[split:] - Ridge(alpha=a).fit(Z[:split], rv[:split]).predict(Z[split:])))), a)
                 for a in (10.0, 100.0, 1000.0, 10000.0)))[1]
    model = Ridge(alpha=alpha).fit(Z, rv)
    return {"features": [features[j] for j in selected], "mean": mean.tolist(), "std": std.tolist(),
            "coef": model.coef_.tolist(), "intercept": float(model.intercept_), "alpha": alpha,
            "scale": float(np.median(np.abs(rows.real)))}


def apply_exog_adjuster(rows: pd.DataFrame, provider: ExogProvider, params: dict | None) -> tuple[np.ndarray, np.ndarray]:
    base = rows.pred_stage.to_numpy(dtype=float)
    pred, delta = base.copy(), np.zeros(len(rows))
    if not params:
        return pred, delta
    X = provider.matrix(rows, params["features"])
    valid = np.isfinite(X).all(axis=1)
    if not valid.any():
        return pred, delta
    raw = ((X[valid] - np.asarray(params["mean"])) / np.asarray(params["std"])) @ np.asarray(params["coef"]) + params["intercept"]
    cap = np.maximum(np.abs(base[valid]) * 0.35, params["scale"] * 0.10)
    delta[valid] = np.clip(raw, -cap, cap)
    pred[valid] = np.clip(base[valid] + delta[valid], 0, None)
    return pred, delta
