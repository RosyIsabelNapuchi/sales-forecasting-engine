"""Actualiza data/bcrp_indicadores.csv con las series públicas del BCRP (requiere internet)."""
from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from forecast.bcrp import download  # noqa: E402

if __name__ == "__main__":
    download(ROOT / "data" / "bcrp_metadata.csv", ROOT / "data" / "bcrp_indicadores.csv")
    print("Listo: data/bcrp_indicadores.csv actualizado")
