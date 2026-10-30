"""Pruebas de que ninguna señal mira el futuro."""
import numpy as np
import pandas as pd
import pytest

from forecast.exogenous import ExogProvider, apply_exog_adjuster, fit_exog_adjuster
from forecast.order_book import OrderBook


def test_order_book_ignores_orders_placed_after_origin():
    raw = pd.DataFrame({
        "linea": ["A", "A"], "fecha_pedido": ["2024-01-10", "2024-03-05"], "fecha_entrega": ["2024-04-01", "2024-04-15"],
        "cantidad": [1, 1], "cantidad_pendiente": [0, 0], "monto_usd": [100.0, 50.0],
        "estado": ["CERRADA", "CERRADA"], "cancelada": [False, False]})
    book = OrderBook(raw, cutoff=pd.Period("2024-06", "M"))
    target = pd.Period("2024-04", "M")
    assert book.booked("A", pd.Period("2024-01", "M"), target) == 100.0      # el pedido de marzo aún no existía
    assert book.booked("A", pd.Period("2024-03", "M"), target) == 150.0
    assert book.booked("A", pd.Period("2023-12", "M"), target) == 0.0


def test_cancelled_orders_are_excluded():
    raw = pd.DataFrame({"linea": ["A"], "fecha_pedido": ["2024-01-10"], "fecha_entrega": ["2024-02-01"], "cantidad": [1],
                        "cantidad_pendiente": [0], "monto_usd": [100.0], "estado": ["CERRADA"], "cancelada": [True]})
    assert OrderBook(raw, pd.Period("2024-06", "M")).booked("A", pd.Period("2024-01", "M"), pd.Period("2024-02", "M")) == 0


def make_provider(values):
    months = pd.period_range("2015-01", periods=len(values), freq="M").astype(str)
    raw = pd.DataFrame({"mes": months, "expect_demanda_12m": values})
    return ExogProvider(raw, sector_mix={}, effect_lag=3, release_lag=2)


def test_exogenous_value_respects_publication_lag():
    values = np.arange(120, dtype=float)                 # el valor es igual al número de mes: fácil de auditar
    p = make_provider(values)
    origin = pd.Period("2020-06", "M")                   # índice 65 → lo publicado llega hasta el índice 63
    far_target = origin + 6                              # pediría el índice 68: aún no publicado
    assert p.value("expect_demanda_12m__nivel", origin, far_target) == 63.0
    near_target = origin + 1                             # pide el índice 63: ya publicado
    assert p.value("expect_demanda_12m__nivel", origin, near_target) == 63.0
    assert p.value("expect_demanda_12m__nivel", origin, origin) == 62.0


def test_exogenous_adjuster_learns_a_real_relationship():
    rng = np.random.default_rng(1)
    driver = rng.normal(0, 1, 140).cumsum() * 0.3 + 50
    p = make_provider(driver)
    origins = pd.period_range("2017-01", periods=60, freq="M")
    rows = pd.DataFrame({"origin_period": origins, "target_period": origins + 1, "horizon": 1})
    x = p.matrix(rows, ["expect_demanda_12m__nivel"])[:, 0]
    base = np.full(len(rows), 1000.0)
    rows["real"] = base + 40 * (x - x.mean()) + rng.normal(0, 5, len(rows))
    rows["pred_stage"] = base
    params = fit_exog_adjuster(rows, p, scope="cualquiera")
    assert params is not None
    adjusted, _ = apply_exog_adjuster(rows, p, params)
    before = np.abs(rows.real - base).mean()
    after = np.abs(rows.real - adjusted).mean()
    assert after < 0.5 * before
