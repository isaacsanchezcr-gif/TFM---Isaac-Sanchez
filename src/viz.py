"""Estilo gráfico común y mapas coropléticos por cantón."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from shapely.geometry import shape

AZUL = "#1f3a5f"      # azul oscuro principal
AZUL_MEDIO = "#2a78d6"
GRIS = "#8a929b"
NARANJA = "#eb6834"   # solo para resaltar un dato puntual

# Rampa secuencial de un solo tono (azul), de claro a oscuro
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

ETIQUETAS_DELITO = {
    "HURTO": "Hurto",
    "ASALTO": "Asalto",
    "ROBO": "Robo",
    "ROBO DE VEHICULO": "Robo de vehículo",
    "TACHA DE VEHICULO": "Tacha de vehículo",
    "HOMICIDIO": "Homicidio",
}


def estilo() -> None:
    """Estilo sobrio para todas las figuras del proyecto."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#b8bec6",
        "axes.labelcolor": "#33393f",
        "axes.grid": True,
        "grid.color": "#e6e9ed",
        "grid.linewidth": 0.8,
        "xtick.color": "#4d5560",
        "ytick.color": "#4d5560",
        "axes.prop_cycle": mpl.cycler(color=[AZUL, NARANJA, "#1baf7a"]),
        "legend.frameon": False,
    })


def cargar_geometrias(ruta: str | Path) -> dict:
    """Devuelve {cantón: geometría shapely} a partir del GeoJSON del proyecto."""
    with open(ruta, encoding="utf-8") as f:
        geo = json.load(f)
    return {f["properties"]["Canton"]: shape(f["geometry"]) for f in geo["features"]}


def _poligonos(geom):
    return geom.geoms if geom.geom_type == "MultiPolygon" else [geom]


def mapa(ax, geometrias: dict, valores: pd.Series, cortes: list[float] | None = None,
         titulo: str = "", formato: str = "{:,.0f}", leyenda: bool = True) -> list[float]:
    """Mapa coroplético por cantón con clases por cuantiles (o cortes dados)."""
    valores = valores.reindex(list(geometrias))
    if cortes is None:
        cortes = list(np.unique(np.nanquantile(valores, np.linspace(0, 1, len(RAMPA) + 1))))
    n = len(cortes) - 1
    cmap = ListedColormap(RAMPA[-n:] if n < len(RAMPA) else RAMPA)
    norm = BoundaryNorm(cortes, cmap.N, clip=True)
    for canton, geom in geometrias.items():
        v = valores.get(canton)
        color = "#eceef1" if pd.isna(v) else cmap(norm(v))
        for poli in _poligonos(geom):
            x, y = poli.exterior.xy
            ax.fill(x, y, color=color, ec="white", lw=0.35)
    ax.set_xlim(-86.0, -82.5)
    ax.set_ylim(8.0, 11.25)
    ax.set_aspect("equal")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo)
    if leyenda:
        parches = [Patch(color=cmap(i), label=f"{formato.format(cortes[i])}–{formato.format(cortes[i + 1])}")
                   for i in range(n)]
        ax.legend(handles=parches, loc="lower left", fontsize=7, handlelength=1, borderaxespad=0)
    return cortes
