"""Construye la población anual por cantón (2015-2026) a partir de dos fuentes del INEC.

1. Estimaciones y proyecciones de población distritales por sexo y edad 2000-2025
   (INEC, 2014). Da la población de cada cantón, pero está basada en el censo 2011
   y sobrestima la población reciente (unos 3 % en 2025).
2. Estimaciones y proyecciones subnacionales 2000-2050, panel por regiones del
   Ministerio de Salud (INEC, 2025). Basada en el censo 2022, pero solo llega a
   nivel de provincia.

Método: el peso de cada cantón dentro de su provincia se toma de la fuente 1 y se
aplica al total provincial de la fuente 2. Para 2026, que la fuente 1 no cubre, se
usa el peso de 2025.

Requiere `pdftotext` (poppler) para leer el PDF.

Uso:
    python scripts/construir_poblacion.py RUTA_PDF_DISTRITAL RUTA_XLSX_REGIONES
Salida:
    data/external/poblacion_cantonal.csv
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.oij import normalizar_texto  # noqa: E402

SALIDA = RAIZ / "data" / "external" / "poblacion_cantonal.csv"
ANIOS = range(2015, 2027)

# Número de cuadro de las proyecciones cantonales del PDF -> provincia
PROVINCIA_POR_CUADRO = (
    [("SAN JOSE", n) for n in range(14, 34)]
    + [("ALAJUELA", n) for n in range(34, 49)]
    + [("CARTAGO", n) for n in range(49, 57)]
    + [("HEREDIA", n) for n in range(57, 67)]
    + [("GUANACASTE", n) for n in range(67, 78)]
    + [("PUNTARENAS", n) for n in range(78, 89)]
    + [("LIMON", n) for n in range(89, 95)]
)

# Nombres del INEC 2014 -> nombres que usa el OIJ
RENOMBRAR = {
    "AGUIRRE": "QUEPOS",
    "VALVERDE VEGA": "SARCHI",
    "VAZQUEZ DE CORONADO": "VASQUEZ DE CORONADO",
}


def leer_proyecciones_cantonales(pdf: Path) -> pd.DataFrame:
    """Extrae la fila 'Total' de cada cuadro de proyecciones (2011-2025)."""
    texto = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    inicio = next(i for i, l in enumerate(texto) if i > 800 and "Población total proyectada" in l)
    datos: dict[int, dict] = {}
    cuadro = actual = anios = None
    for linea in texto[inicio:]:
        m = re.match(r"^\s*CUADRO\s+(\d+)\s*$", linea)
        if m:
            cuadro = int(m.group(1))
            continue
        m = re.match(r"^\s*(.+?)\. Población total proyectada", linea)
        if m and not linea.strip().startswith("Cuadro"):
            actual = cuadro
            datos.setdefault(actual, {"nombre": m.group(1).strip()})
            continue
        if actual is None:
            continue
        if re.match(r"^\s*(20\d\d\s+){3,}", linea.strip() + " "):
            anios = [int(a) for a in re.findall(r"20\d\d", linea)]
            continue
        if anios and linea.strip().startswith("Total"):
            valores = [int(v.replace(" ", "")) for v in re.split(r"\s{2,}", linea.strip())[1:]]
            if len(valores) == len(anios):
                datos[actual].update(dict(zip(anios, valores)))
            anios = None
    filas = []
    for provincia, n in PROVINCIA_POR_CUADRO:
        fila = datos[n]
        canton = normalizar_texto(fila["nombre"])
        filas.append({"Provincia": provincia, "Canton": RENOMBRAR.get(canton, canton),
                      **{a: fila[a] for a in range(2011, 2026)}})
    return pd.DataFrame(filas)


def leer_totales_provinciales(xlsx: Path) -> pd.DataFrame:
    """Población total por provincia y año del panel subnacional 2000-2050."""
    panel = pd.read_excel(xlsx, sheet_name="Hoja6")
    panel["Provincia"] = panel["Provincia"].str.replace(r"^\d+-", "", regex=True).map(normalizar_texto)
    panel["Poblacion"] = panel["Hombres"] + panel["Mujeres"]
    return panel.groupby(["Provincia", "Año"])["Poblacion"].sum().unstack()


def construir(pdf: Path, xlsx: Path) -> pd.DataFrame:
    cantones = leer_proyecciones_cantonales(pdf)
    provincias = leer_totales_provinciales(xlsx)
    anios_viejos = [a for a in ANIOS if a <= 2025]
    viejos = cantones.set_index(["Provincia", "Canton"])[anios_viejos]
    peso = viejos / viejos.groupby(level="Provincia").transform("sum")
    peso[2026] = peso[2025]
    salida = peso.copy()
    for anio in ANIOS:
        total = salida.index.get_level_values("Provincia").map(provincias[anio])
        salida[anio] = (peso[anio] * total).round().astype(int)
    largo = salida.reset_index().melt(["Provincia", "Canton"], var_name="Anio", value_name="Poblacion")
    return largo.sort_values(["Provincia", "Canton", "Anio"]).reset_index(drop=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tabla = construir(Path(sys.argv[1]), Path(sys.argv[2]))
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA, index=False)
    print(f"{tabla['Canton'].nunique()} cantones, {tabla['Anio'].min()}-{tabla['Anio'].max()} -> {SALIDA}")
