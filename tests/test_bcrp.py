import numpy as np
import pandas as pd

from forecast.bcrp import parse_payload, parse_period, to_float


def test_parse_period_variants():
    assert parse_period("Ene.2025") == pd.Period("2025-01", "M")
    assert parse_period("Set.2024") == pd.Period("2024-09", "M")
    assert parse_period("Dic25") == pd.Period("2025-12", "M")
    assert parse_period("2023-07") == pd.Period("2023-07", "M")
    assert parse_period("basura") is None


def test_to_float_handles_missing_and_commas():
    assert np.isnan(to_float("n.d."))
    assert to_float("1,5") == 1.5
    assert to_float("1,234.5") == 1234.5


def test_parse_payload_shape_of_bcrp_api():
    payload = {"config": {}, "periods": [{"name": "Ene.2024", "values": ["101.2"]},
                                         {"name": "Feb.2024", "values": ["n.d."]}]}
    s = parse_payload(payload)
    assert s.loc[pd.Period("2024-01", "M")] == 101.2 and np.isnan(s.loc[pd.Period("2024-02", "M")])
