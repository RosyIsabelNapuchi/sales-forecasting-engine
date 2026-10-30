import numpy as np
import pandas as pd
import pytest

from forecast.pipeline import ScopeResult, reconcile
from forecast.synthetic import make_data


def _result(scope, normal):
    months = pd.period_range("2027-01", periods=len(normal), freq="M")
    fc = pd.DataFrame({"mes": months, "horizon": np.arange(1, len(normal) + 1), "modelo": "x",
                       "normal": normal, "pesimista": np.array(normal) * 0.9, "optimista": np.array(normal) * 1.1})
    return ScopeResult(scope, pd.Series(dtype=float), fc, {}, pd.DataFrame(), pd.DataFrame())


def test_reconciled_lines_sum_to_total():
    total = _result("TOTAL", [1000.0, 1100.0, 1200.0])
    lines = [_result("A", [500.0, 500.0, 600.0]), _result("B", [300.0, 450.0, 400.0])]
    detail = reconcile(total, lines, other_share=0.05)
    sums = detail.groupby("mes").normal_reconciliado.sum().to_numpy()
    assert np.allclose(sums, [1000.0, 1100.0, 1200.0])


def test_synthetic_data_is_reproducible_and_clean():
    exog = pd.read_csv(pytest.importorskip("pathlib").Path(__file__).parents[1] / "data" / "bcrp_indicadores.csv")
    a, b = make_data(exog, seed=3), make_data(exog, seed=3)
    assert a["ventas_mensuales"].equals(b["ventas_mensuales"])
    assert (a["ventas_mensuales"].venta_usd >= 0).all()
    ob = a["cartera_pedidos"]
    assert (pd.to_datetime(ob.fecha_pedido) <= pd.Timestamp("2026-08-31")).all()   # nada pedido después del cierre
