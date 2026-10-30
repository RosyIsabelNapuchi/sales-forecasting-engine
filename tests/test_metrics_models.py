import numpy as np
import pandas as pd
import pytest

from forecast.metrics import annual_error, bias_pct, wape
from forecast.models import BENCHMARK_SPECS, MODEL_SPECS, validate_registry
from forecast.selection import candidate_model_names, safe_predict


def seasonal_series(n=72, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2019-01", periods=n, freq="M")
    y = 100 + 0.5 * np.arange(n) + 15 * np.sin(2 * np.pi * months.month / 12) + rng.normal(0, 3, n)
    return y, months


def test_wape_and_annual_error():
    assert wape([100, 200], [110, 180]) == pytest.approx(30 / 300)
    # errores que se compensan dentro del año: WAPE alto, error anual cero
    assert wape([100, 100], [150, 50]) == pytest.approx(0.5)
    assert annual_error([100, 100], [150, 50]) == pytest.approx(0.0)


def test_bias_sign():
    assert bias_pct([100], [120]) > 0 > bias_pct([100], [80])


def test_registry_is_consistent():
    validate_registry()


@pytest.mark.parametrize("name", list(MODEL_SPECS) + list(BENCHMARK_SPECS) + ["combo_robusto"])
def test_every_model_returns_valid_forecast(name):
    y, months = seasonal_series()
    future = pd.period_range(months[-1] + 1, periods=16, freq="M")
    pred, state = safe_predict(name, y, months, future)
    assert state == "ok" and pred.shape == (16,) and np.isfinite(pred).all() and (pred >= 0).all()


def test_intermittent_models_only_when_many_zeros():
    y, months = seasonal_series()
    assert "croston_sba" not in candidate_model_names(y, months)
    y_int = np.where(np.arange(len(y)) % 4 == 0, y, 0.0)
    assert {"croston_sba", "tsb"} <= set(candidate_model_names(y_int, months))
