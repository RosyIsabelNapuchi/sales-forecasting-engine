"""Cartera de pedidos (órdenes de venta abiertas) como señal adelantada.

Idea: si en el origen ya está pedido el X % de lo que históricamente termina vendiéndose en un mes,
la venta implícita es pedido / cobertura. Esa señal solo se usa si mejora el pronóstico base
por un margen en desarrollo Y se confirma en una ventana posterior.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import wape

TOTAL_SCOPE = "__TOTAL__"
REQUIRED = ["linea", "fecha_pedido", "fecha_entrega", "cantidad", "cantidad_pendiente", "monto_usd", "estado", "cancelada"]


def horizon_bucket(h: int) -> str:
    if h <= 4:
        return str(h)
    return "5-6" if h <= 6 else "7-9" if h <= 9 else "10-12" if h <= 12 else "13+"


class OrderBook:
    def __init__(self, raw: pd.DataFrame, cutoff: pd.Period, overdue_distribution: Sequence[float] = (0.5, 0.3, 0.2)):
        missing = [c for c in REQUIRED if c not in raw.columns]
        if missing:
            raise ValueError(f"Cartera de pedidos sin columnas: {missing}")
        ov = raw.copy()
        ov["doc_date"] = pd.to_datetime(ov.fecha_pedido, errors="coerce")
        due = pd.to_datetime(ov.fecha_entrega, errors="coerce")
        invalid_due = due.isna() | (due.dt.to_period("M") < ov.doc_date.dt.to_period("M"))
        ov["doc_month"] = ov.doc_date.dt.to_period("M")
        ov["due_month"] = due.mask(invalid_due, ov.doc_date).dt.to_period("M")
        ov["amount"] = pd.to_numeric(ov.monto_usd, errors="coerce")
        ov = ov[ov.doc_month.notna() & ov.amount.gt(0) & (ov.doc_month <= cutoff)].copy()
        self.invalid_due = int(invalid_due.sum())
        # Mismo universo para calibrar y para predecir: sin pedidos cancelados en ningún lado.
        self.cancelled_amount = float(ov.loc[ov.cancelada.astype(bool), "amount"].sum())
        ov = ov[~ov.cancelada.astype(bool)].copy()
        ratio = (pd.to_numeric(ov.cantidad_pendiente, errors="coerce")
                 / pd.to_numeric(ov.cantidad, errors="coerce").replace(0, np.nan)).clip(0, 1)
        ov["open_amount"] = ov.amount * ratio.fillna(1.0)
        self.cutoff = cutoff
        self.distribution = tuple(overdue_distribution)
        self._history: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for scope, frame in [(TOTAL_SCOPE, ov), *ov.groupby("linea")]:
            grouped = frame.groupby(["due_month", "doc_month"]).amount.sum().reset_index()
            for due_m, g in grouped.groupby("due_month"):
                g = g.sort_values("doc_month")
                self._history[(scope, due_m.ordinal)] = (np.asarray([d.ordinal for d in g.doc_month]),
                                                         g.amount.cumsum().to_numpy(dtype=float))
        self.current = ov[ov.estado.eq("ABIERTA")].copy()

    def booked(self, scope: str, origin: pd.Period, target: pd.Period) -> float:
        """Monto pedido para `target` que ya estaba registrado en `origin` (sin mirar el futuro)."""
        key = (scope, target.ordinal)
        if key not in self._history:
            return 0.0
        docs, cumulative = self._history[key]
        pos = int(np.searchsorted(docs, origin.ordinal, side="right") - 1)
        return float(cumulative[pos]) if pos >= 0 else 0.0

    def enrich(self, rows: pd.DataFrame, scope: str) -> pd.DataFrame:
        out = rows.copy()
        out["booked_ov"] = [self.booked(scope, o, t) for o, t in zip(out.origin_period, out.target_period)]
        return out

    def current_amounts(self, scope: str, future: pd.PeriodIndex) -> np.ndarray:
        frame = self.current if scope == TOTAL_SCOPE else self.current[self.current.linea == scope]
        amounts = np.zeros(len(future))
        if frame.empty:
            return amounts
        overdue = float(frame.loc[frame.due_month <= self.cutoff, "open_amount"].sum())
        for step, weight in enumerate(self.distribution):         # lo atrasado se reparte en los próximos meses
            if step < len(amounts):
                amounts[step] += overdue * weight
        on_time = frame[frame.due_month > self.cutoff].groupby("due_month").open_amount.sum()
        for pos, month in enumerate(future):
            amounts[pos] += float(on_time.get(month, 0.0))
        return amounts


def fit_ov_params(rows: pd.DataFrame, min_coverage: float = 0.25, max_multiple: float = 2.0) -> dict | None:
    """Calibra, por tramo de horizonte, la cobertura típica y el peso óptimo de la señal."""
    if rows.empty:
        return None
    frame = rows.assign(bucket=rows.horizon.map(horizon_bucket))
    valid = frame[(frame.booked_ov > 0) & (frame.real > 0)]
    if len(valid) < 8:
        return None
    coverages = {}
    for bucket, g in valid.groupby("bucket"):
        if len(g) >= 3:
            # Una cobertura de 3 % es ruido: dividir por ella multiplica el error por 33.
            coverages[bucket] = float(np.clip((g.booked_ov / g.real).clip(0.005, 3.0).median(), min_coverage, 2.0))
    if not coverages:
        return None
    weights = {}
    for bucket, coverage in coverages.items():
        g = frame[frame.bucket == bucket]
        candidates = []
        for weight in (0.0, 0.20, 0.35, 0.50, 0.65, 0.80):
            implied = np.clip(g.booked_ov / coverage, 0, np.maximum(g.pred_base * max_multiple, g.pred_base + 1))
            pred = np.where(g.booked_ov > 0, (1 - weight) * g.pred_base + weight * implied, g.pred_base)
            candidates.append((wape(g.real, pred), weight))
        base_err = candidates[0][0]
        best_err = min(c[0] for c in candidates)
        err, weight = min([c for c in candidates if c[0] <= best_err * 1.01 + 1e-12], key=lambda c: c[1])
        weights[bucket] = 0.0 if 1 - err / max(base_err, 1e-9) < 0.02 else float(weight)
    return {"weights": weights, "coverages": coverages}


def apply_ov_params(rows: pd.DataFrame, params: dict | None, max_multiple: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    base = rows.pred_base.to_numpy(dtype=float)
    pred, applied = base.copy(), np.zeros(len(rows), dtype=bool)
    if not params or not any(w > 0 for w in params["weights"].values()):
        return pred, applied
    for pos, (h, booked, b) in enumerate(zip(rows.horizon, rows.booked_ov, base)):
        bucket = horizon_bucket(int(h))
        coverage, weight = params["coverages"].get(bucket), params["weights"].get(bucket, 0.0)
        if coverage is None or weight <= 0 or booked <= 0:
            continue
        implied = float(np.clip(booked / coverage, 0.0, max(b * max_multiple, b + 1.0)))
        pred[pos] = (1 - weight) * b + weight * implied
        applied[pos] = True
    return pred, applied
