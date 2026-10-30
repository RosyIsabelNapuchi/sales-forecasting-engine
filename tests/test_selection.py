import numpy as np
import pandas as pd

from forecast.selection import rank_models


def fake_backtest(errors: dict[str, float], folds=6, seed=0):
    """Backtest artificial: cada modelo predice real*(1+error) con algo de ruido."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, err in errors.items():
        for origin in range(folds):
            for h in range(1, 7):
                real = 100.0
                rows.append({"model": model, "origin_index": origin, "horizon": h, "real": real,
                             "pred": real * (1 + err + rng.normal(0, 0.002))})
    return pd.DataFrame(rows)


def test_benchmark_never_selected_even_if_best():
    bt = fake_backtest({"benchmark_mean12": 0.01, "ses": 0.10, "random_forest": 0.12})
    _, chosen, _, reason = rank_models(bt, 6, margin=0.03)
    assert chosen == "ses"
    assert "ALERTA" in reason                     # se reporta que no supera al benchmark


def test_complex_model_needs_margin_over_reference():
    # el bosque mejora solo ~2 % frente a SES: no alcanza el margen de 3 %
    bt = fake_backtest({"ses": 0.100, "random_forest": 0.098})
    _, chosen, reference, _ = rank_models(bt, 6, margin=0.03)
    assert reference == "ses" and chosen == "ses"


def test_clear_winner_is_selected():
    bt = fake_backtest({"ses": 0.20, "random_forest": 0.05})
    _, chosen, _, _ = rank_models(bt, 6, margin=0.03)
    assert chosen == "random_forest"
