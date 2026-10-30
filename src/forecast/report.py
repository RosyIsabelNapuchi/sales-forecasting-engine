"""Gráficos del reporte (paleta categórica validada; texto en tinta neutra)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _style() -> None:
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "legend.frameon": False, "lines.linewidth": 2, "font.size": 9.5})


def _k(x, _=None):
    return f"{x / 1e6:.1f} M" if abs(x) >= 1e6 else f"{x / 1e3:.0f} k"


def make_figures(pivot, total, results, total_result, detail, diag, ranking, out: Path) -> None:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    from matplotlib.ticker import FuncFormatter

    # 1. Total con escenarios
    f = total_result.forecast
    hist = total[total.index >= total.index[-1] - 47]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(hist.index.to_timestamp(), hist.values, color=BLUE, label="Histórico (sintético)")
    x = f.mes.dt.to_timestamp()
    ax.plot(x, f.normal, color=BLUE, linestyle="--", label="Pronóstico normal")
    ax.fill_between(x, f.pesimista, f.optimista, color=BLUE, alpha=0.18, linewidth=0, label="Pesimista – optimista")
    ax.yaxis.set_major_formatter(FuncFormatter(_k))
    d = total_result.diagnostics
    ax.set_title(f"Venta total: pronóstico a dic-2027 · modelo {d['modelo']} · confianza {d['confianza']}")
    ax.set_ylabel("USD por mes")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "01_total_escenarios.png", dpi=160)
    plt.close(fig)

    # 2. Fichas por línea (small multiples)
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.6), sharex=False)
    for ax, r in zip(axes.ravel(), results):
        h = r.history[r.history.index >= r.history.index[-1] - 35]
        g = detail[detail.linea == r.scope]
        gx = pd.PeriodIndex(g.mes, freq="M").to_timestamp()
        ax.plot(h.index.to_timestamp(), h.values, color=BLUE, linewidth=1.6)
        ax.plot(gx, g.normal_reconciliado, color=ORANGE, linewidth=1.8)
        ax.fill_between(gx, g.pesimista_reconciliado, g.optimista_reconciliado, color=ORANGE, alpha=0.18, linewidth=0)
        dg = r.diagnostics
        flags = " · ".join(t for t, ok in (("cartera", dg["cartera_aceptada"]), ("exógenas", dg["exogenas_aceptadas"])) if ok)
        ax.set_title(r.scope, loc="left", pad=18)
        ax.text(0.0, 1.02, f"{dg['modelo']} · confianza {dg['confianza']}" + (f" · + {flags}" if flags else ""),
                transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
        ax.yaxis.set_major_formatter(FuncFormatter(_k))
        ax.tick_params(axis="x", labelsize=7.5, rotation=30)
    fig.suptitle("Fichas por línea: últimos 36 meses (azul) y pronóstico reconciliado con escenarios (naranja)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "02_fichas_por_linea.png", dpi=150)
    plt.close(fig)

    # 3. Valor agregado frente al mejor benchmark
    dd = diag.sort_values("valor_agregado_vs_benchmark")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = [ORANGE if v < 0 else BLUE for v in dd.valor_agregado_vs_benchmark]
    ax.barh(dd.scope, dd.valor_agregado_vs_benchmark * 100, color=colors, height=0.6)
    for yv, (v, m) in enumerate(zip(dd.valor_agregado_vs_benchmark * 100, dd.modelo)):
        ax.text(max(v, 0) + 0.6, yv, f"{v:.0f} % · {m}", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Mejora del puntaje vs. el mejor benchmark (%)")
    ax.set_title("¿El modelo elegido le gana a lo simple? (naranja = no)")
    ax.grid(axis="y", visible=False)
    lo, hi = (dd.valor_agregado_vs_benchmark * 100).min(), (dd.valor_agregado_vs_benchmark * 100).max()
    ax.set_xlim(min(lo * 1.3, -5), hi * 1.9)
    fig.tight_layout()
    fig.savefig(out / "03_valor_agregado_vs_benchmark.png", dpi=160)
    plt.close(fig)
