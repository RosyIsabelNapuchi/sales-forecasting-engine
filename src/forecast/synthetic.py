"""Datos SINTÉTICOS de una distribuidora B2B de repuestos industriales (10 líneas + 'otras').

- Las ventas son simuladas: tendencia, estacionalidad, ruido, atípicos, una línea intermitente (proyectos).
- Algunas líneas dependen, con 3 meses de rezago, de indicadores PÚBLICOS REALES del BCRP
  (data/bcrp_indicadores.csv). Así la prueba de variables externas tiene algo real que encontrar… o no.
- La cartera de pedidos se genera desde las ventas: las líneas de proyectos y servicios se piden con
  meses de anticipación (señal útil); los repuestos casi no (señal débil).
- Se simula también el futuro (sep-2026 a dic-2027) para construir la cartera abierta al cierre y
  poder comparar el pronóstico contra la "verdad" simulada.
Nada de esto proviene de una empresa real.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START, CUTOFF, END = "2016-01", "2026-08", "2027-12"

# nombre, mezcla sectorial, nivel mensual (USD), crecimiento anual, estacionalidad, ruido, driver BCRP, beta, % pedido, rezago pedido
LINES = [
    ("Rodamientos", [("MINERIA", .70), ("PESCA", .15), ("INDUSTRIA", .15)], 180e3, .04, "leve", .07, None, 0, .45, (0, 1)),
    ("Bombas", [("PESCA", .55), ("MINERIA", .30), ("INDUSTRIA", .15)], 140e3, .03, "pesca", .10, "pbi_pesca", .45, .50, (0, 2)),
    ("Valvulas", [("MINERIA", .50), ("CONSTRUCCION", .30), ("INDUSTRIA", .20)], 120e3, .05, None, .08, "pbi_construccion", .90, .45, (0, 1)),
    ("Sellos", [("PESCA", .40), ("INDUSTRIA", .40), ("MINERIA", .20)], 60e3, .02, "pesca", .09, None, 0, .40, (0, 1)),
    ("Transmision", [("MINERIA", .60), ("INDUSTRIA", .25), ("PESCA", .15)], 150e3, .05, "leve", .08, "prod_cobre", .80, .50, (0, 2)),
    ("Filtros", [("MINERIA", .75), ("PESCA", .14), ("INDUSTRIA", .11)], 200e3, .03, "leve", .05, None, 0, .40, (0, 1)),
    ("Instrumentacion", [("INDUSTRIA", .60), ("MINERIA", .40)], 50e3, .06, None, .22, None, 0, .55, (0, 2)),
    ("Servicios tecnicos", [("MINERIA", .60), ("PESCA", .20), ("INDUSTRIA", .20)], 90e3, .04, None, .14, None, 0, .80, (1, 3)),
    ("Proyectos", [("MINERIA", .60), ("CONSTRUCCION", .40)], 0, 0, None, 0, None, 0, .95, (3, 9)),
    ("Refrigeracion", [("PESCA", .60), ("INDUSTRIA", .40)], 70e3, .02, "pesca", .10, "anchoveta_tn", .25, .45, (0, 2)),
]
OTHER_SHARE = 0.04
PATTERNS = {
    "leve": [1.00, .95, 1.02, 1.00, 1.03, 1.00, .98, 1.00, 1.02, 1.00, .98, 1.02],
    "pesca": [1.05, .80, .80, 1.20, 1.35, 1.25, 1.10, .85, .80, .85, 1.10, 1.25],
}


def _driver_effect(exog: pd.DataFrame, driver: str, months: pd.PeriodIndex, lag: int = 3) -> np.ndarray:
    """Variación interanual REAL del indicador, con rezago; se prolonga el último dato hacia el futuro."""
    s = pd.to_numeric(exog.set_index("mes")[driver], errors="coerce")
    s.index = pd.PeriodIndex(s.index, freq="M")
    yoy = s.pct_change(12, fill_method=None).clip(-0.6, 0.6)
    lagged = yoy.reindex(months - lag).ffill().fillna(0.0)
    return lagged.to_numpy()


def make_data(exog: pd.DataFrame, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    months = pd.period_range(START, END, freq="M")
    t = np.arange(len(months)) / 12.0
    macro = _driver_effect(exog, "pbi_total", months)                    # caída de 2020 y ciclo general (real)
    sales, orders = [], []
    for name, mix, level, growth, pattern, noise, driver, beta, ordered_share, lead in LINES:
        if name == "Proyectos":                                           # demanda intermitente: hitos grandes
            y = np.where(rng.random(len(months)) < 0.30, rng.lognormal(np.log(380e3), 0.5, len(months)), 0.0)
            y *= np.exp(0.8 * macro)
        else:
            season = np.asarray(PATTERNS[pattern])[months.month - 1] if pattern else 1.0
            y = level * (1 + growth) ** t * season * np.exp(0.9 * macro) * rng.lognormal(0, noise, len(months))
            if driver:
                y *= np.exp(beta * _driver_effect(exog, driver, months))
            spikes = rng.random(len(months)) < 0.03                       # pedidos extraordinarios
            y[spikes] *= rng.uniform(1.4, 2.0, spikes.sum())
        y = np.round(y, 2)
        sales.append(pd.DataFrame({"mes": months.astype(str), "linea": name, "venta_usd": y}))
        # cartera de pedidos: cada venta pedida se registra con anticipación
        for m, amount in zip(months, y):
            if amount <= 0:
                continue
            booked = amount * ordered_share * rng.uniform(0.85, 1.0)
            for part in rng.dirichlet(np.ones(rng.integers(1, 4))) * booked:
                lead_m = int(rng.integers(lead[0], lead[1] + 1))
                doc = (m - lead_m).to_timestamp() + pd.Timedelta(days=int(rng.integers(0, 27)))
                due = m.to_timestamp() + pd.Timedelta(days=int(rng.integers(0, 27)))
                orders.append((name, doc, due, part, False))
            if rng.random() < 0.05:                                       # algunos pedidos se cancelan
                orders.append((name, (m - lead[1]).to_timestamp(), m.to_timestamp(), booked * 0.3, True))
    sales = pd.concat(sales, ignore_index=True)
    lines_total = sales.groupby("mes").venta_usd.sum()
    other = (lines_total * OTHER_SHARE / (1 - OTHER_SHARE) * rng.lognormal(0, 0.10, len(lines_total))).round(2)
    sales = pd.concat([sales, pd.DataFrame({"mes": other.index, "linea": "Otras", "venta_usd": other.values})],
                      ignore_index=True)

    ob = pd.DataFrame(orders, columns=["linea", "fecha_pedido", "fecha_entrega", "monto_usd", "cancelada"])
    cutoff_ts = pd.Period(CUTOFF, freq="M").to_timestamp(how="end")
    ob = ob[ob.fecha_pedido <= cutoff_ts].copy()                          # al cierre solo existen los pedidos ya colocados
    delivered = ob.fecha_entrega <= cutoff_ts
    ob["estado"] = np.where(delivered | ob.cancelada, "CERRADA", "ABIERTA")
    ob["cantidad"] = rng.integers(1, 40, len(ob))
    ob["cantidad_pendiente"] = np.where(ob.estado.eq("ABIERTA"), ob.cantidad, 0)
    ob["monto_usd"] = ob.monto_usd.round(2)
    ob = ob.sort_values("fecha_pedido").reset_index(drop=True)
    ob.insert(0, "pedido_id", [f"PV{i:06d}" for i in range(1, len(ob) + 1)])

    history = sales[sales.mes <= CUTOFF].reset_index(drop=True)
    truth = sales[sales.mes > CUTOFF].reset_index(drop=True)
    mix = pd.DataFrame([(n, s, v) for n, m, *_ in LINES for s, v in m], columns=["linea", "sector", "participacion"])
    return {"ventas_mensuales": history, "cartera_pedidos": ob, "mezcla_sectorial": mix, "verdad_futura": truth}


def write(data_dir: Path, seed: int = 7) -> dict[str, pd.DataFrame]:
    exog = pd.read_csv(data_dir / "bcrp_indicadores.csv")
    data = make_data(exog, seed)
    for name, df in data.items():
        df.to_csv(data_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    return data
