"""Pipeline completo: datos → selección de modelo por serie → cartera → exógenas → escenarios → reconciliación.

Uso:  python run.py            (unos 4 minutos)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from forecast.exogenous import ExogProvider  # noqa: E402
from forecast.metrics import annual_error, wape  # noqa: E402
from forecast.order_book import TOTAL_SCOPE, OrderBook  # noqa: E402
from forecast.pipeline import Config, process_scope, reconcile  # noqa: E402
from forecast.report import make_figures  # noqa: E402
from forecast.synthetic import CUTOFF, END, write  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "reports"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    data = write(DATA)                                   # regenera los datos sintéticos (semilla fija)
    sales, book_raw = data["ventas_mensuales"], data["cartera_pedidos"]
    cutoff = pd.Period(CUTOFF, freq="M")
    future = pd.period_range(cutoff + 1, END, freq="M")
    mix = {line: [(r.sector, r.participacion) for r in g.sort_values("participacion", ascending=False).itertuples()]
           for line, g in data["mezcla_sectorial"].groupby("linea")}
    exog = ExogProvider(pd.read_csv(DATA / "bcrp_indicadores.csv"), mix)
    book = OrderBook(book_raw, cutoff)
    cfg = Config()

    pivot = sales.pivot(index="mes", columns="linea", values="venta_usd")
    pivot.index = pd.PeriodIndex(pivot.index, freq="M")
    modeled = [c for c in pivot.columns if c != "Otras"]
    total = pivot.sum(axis=1)
    other_share = float(pivot["Otras"].tail(12).sum() / total.tail(12).sum())

    results = [process_scope(line, pivot[line], future, book, exog, cfg) for line in modeled]
    total_result = process_scope(TOTAL_SCOPE, total, future, book, exog, cfg)
    detail = reconcile(total_result, results, other_share)

    diag = pd.DataFrame([r.diagnostics for r in results + [total_result]])
    diag["scope"] = diag.scope.replace({TOTAL_SCOPE: "TOTAL"})
    ranking = pd.concat([r.ranking for r in results + [total_result]], ignore_index=True)
    detail["mes"] = detail.mes.astype(str)
    diag.to_csv(OUT / "diagnostico_por_serie.csv", index=False)
    ranking.to_csv(OUT / "ranking_modelos.csv", index=False)
    detail.to_csv(OUT / "forecast_detalle_reconciliado.csv", index=False)

    # ---- Comparación contra la "verdad" simulada (solo posible porque los datos son sintéticos)
    truth = data["verdad_futura"].pivot(index="mes", columns="linea", values="venta_usd")
    rows = []
    for line in modeled + ["TOTAL"]:
        real = truth.sum(axis=1) if line == "TOTAL" else truth[line]
        pred = (total_result.forecast.set_index(total_result.forecast.mes.astype(str)).normal if line == "TOTAL"
                else detail[detail.linea == line].set_index("mes").normal_reconciliado)
        pred = pred.reindex(real.index)
        y27 = real.index.str.startswith("2027")
        rows.append({"serie": line, "wape_mensual_16m": wape(real, pred), "error_anual_2027": annual_error(real[y27], pred[y27])})
    vs_truth = pd.DataFrame(rows)
    vs_truth.to_csv(OUT / "comparacion_vs_verdad_simulada.csv", index=False)

    y2027 = detail[detail.mes.str.startswith("2027")].groupby("linea")[
        ["normal_reconciliado", "pesimista_reconciliado", "optimista_reconciliado"]].sum().round(0)
    y2027.loc["TOTAL"] = y2027.sum()
    y2027.to_csv(OUT / "forecast_2027_por_linea.csv")
    make_figures(pivot, total, results, total_result, detail, diag, ranking, OUT / "figures")

    cols = ["scope", "modelo", "modelo_referencia", "valor_agregado_vs_benchmark", "cartera_aceptada", "exogenas_aceptadas",
            "drivers", "wape_holdout_final", "error_anual_promedio", "confianza"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    print("\n=== Diagnóstico por serie ===\n", diag[cols].round(3).to_string(index=False))
    print("\n=== Contra la verdad simulada ===\n", vs_truth.round(3).to_string(index=False))
    summary = {"lineas": len(modeled), "corte": CUTOFF, "horizonte_meses": len(future),
               "cartera_aceptada_en": diag.loc[diag.cartera_aceptada, "scope"].tolist(),
               "exogenas_aceptadas_en": diag.loc[diag.exogenas_aceptadas, "scope"].tolist(),
               "modelos_elegidos": diag.modelo.value_counts().to_dict()}
    (OUT / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
