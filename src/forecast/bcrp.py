"""Descarga de series públicas del BCRP (API de estadísticas) para actualizar data/bcrp_indicadores.csv."""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "set": 9,
         "oct": 10, "nov": 11, "dic": 12, "jan": 1, "apr": 4, "aug": 8, "dec": 12}


def to_float(value) -> float:
    s = str(value).strip().replace("\xa0", "").replace(" ", "") if value is not None else ""
    if s.lower() in {"", "n.d.", "nd", "nan", "null", "none", "-", "--"}:
        return np.nan
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "," in s:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_period(name: str) -> pd.Period | None:
    """Acepta 'Ene.2025', 'Ene25', 'Jan.2025', '2025-01'."""
    s = str(name).strip().lower().replace(".", "")
    m = re.search(r"(\d{4})\D*(\d{1,2})$", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return pd.Period(f"{m.group(1)}-{int(m.group(2)):02d}", freq="M")
    m = re.search(r"([a-záéíóúñ]{3})\s*[-/]?\s*(\d{2,4})", s)
    if m and m.group(1)[:3] in MESES:
        year = int(m.group(2))
        year += (2000 if year < 70 else 1900) if year < 100 else 0
        return pd.Period(f"{year}-{MESES[m.group(1)[:3]]:02d}", freq="M")
    return None


def parse_payload(payload: dict) -> pd.Series:
    rows = {}
    for p in payload.get("periods", []):
        period = parse_period(p.get("name", ""))
        if period is not None and p.get("values"):
            rows[period] = to_float(p["values"][0])
    if not rows:
        raise ValueError("Respuesta del BCRP sin períodos interpretables")
    return pd.Series(rows).sort_index()


def download(metadata_csv: Path, out_csv: Path, start: str = "2008-01", end: str | None = None, pause: float = 0.5) -> None:
    import requests

    end = end or str(pd.Timestamp.today().to_period("M"))
    s, e = pd.Period(start, "M"), pd.Period(end, "M")
    meta = pd.read_csv(metadata_csv)
    series = {}
    for row in meta.itertuples():
        url = f"{API}/{row.codigo}/json/{s.year}-{s.month}/{e.year}-{e.month}/esp"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        series[row.alias] = parse_payload(resp.json())
        time.sleep(pause)
    wide = pd.DataFrame(series)
    wide.index.name = "mes"
    wide.index = wide.index.astype(str)
    wide.reset_index().to_csv(out_csv, index=False)
