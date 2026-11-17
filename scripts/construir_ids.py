"""Prepara el Índice de Desarrollo Social 2023 (MIDEPLAN) y los límites cantonales.

Fuente: MIDEPLAN, Índice de Desarrollo Social 2023, capa cantonal en formato
shapefile (ids_2023_cantonal.zip), publicada en
https://sites.google.com/expedientesmideplan.go.cr/odnr/mapas-y-recursos

El shapefile trae 84 cantones (división vigente). Para encajar con las 81 series
del proyecto, Río Cuarto, Monteverde y Puerto Jiménez se unen a su cantón de
origen: su polígono se fusiona y sus índices no se usan (se conserva el valor
publicado para el cantón de origen, que es mucho más poblado).

Nombres de campo del shapefile (truncados a 10 caracteres por el formato DBF) y
dimensión que representan. La correspondencia se dedujo de los valores
extremos de cada campo y debe confirmarse con la metodología del MIDEPLAN:
    N_INDICE_3 -> económica      (máximos: Santa Ana, Escazú, Belén)
    N_INDICE_P -> participación  (mínimos: Limón, Golfito, Corredores)
    N_INDICE_S -> salud
    N_INDICE_E -> educativa
    N_INDICE_2 -> seguridad      (mínimos: Garabito, Limón, Matina)

Uso:
    python scripts/construir_ids.py RUTA_ZIP_O_CARPETA_SHAPEFILE
Salidas:
    data/external/ids_2023_cantonal.csv
    data/external/cantones.geojson   (81 cantones, WGS84, geometría simplificada)
"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile  # pyshp
from pyproj import Transformer
from shapely import transform
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.oij import CANTONES_NUEVOS, normalizar_texto  # noqa: E402

SALIDA_IDS = RAIZ / "data" / "external" / "ids_2023_cantonal.csv"
SALIDA_GEO = RAIZ / "data" / "external" / "cantones.geojson"

CAMPOS = {
    "C_DIGO_CA": "Codigo",
    "N_IDS_2023": "IDS",
    "N_INDICE_3": "IDS_economica",
    "N_INDICE_P": "IDS_participacion",
    "N_INDICE_S": "IDS_salud",
    "N_INDICE_E": "IDS_educativa",
    "N_INDICE_2": "IDS_seguridad",
    "QUINTIL": "IDS_quintil",
}

# Nombres del MIDEPLAN -> nombres que usa el OIJ
RENOMBRAR = {"LEON CORTES CASTRO": "LEON CORTES", "VAZQUEZ DE CORONADO": "VASQUEZ DE CORONADO"}

# Proyección oficial CRTM05 -> coordenadas geográficas
_CRTM05_A_WGS84 = Transformer.from_crs("EPSG:5367", "EPSG:4326", always_xy=True)


def a_wgs84(coords: np.ndarray) -> np.ndarray:
    return np.column_stack(_CRTM05_A_WGS84.transform(coords[:, 0], coords[:, 1]))


def abrir_shapefile(ruta: Path) -> shapefile.Reader:
    if ruta.suffix == ".zip":
        carpeta = Path(tempfile.mkdtemp())
        zipfile.ZipFile(ruta).extractall(carpeta)
        ruta = carpeta
    shp = next(ruta.rglob("*.shp"))
    return shapefile.Reader(str(shp.with_suffix("")), encoding="utf-8")


def construir(ruta: Path) -> tuple[pd.DataFrame, dict]:
    lector = abrir_shapefile(ruta)
    nombres = [f[0] for f in lector.fields[1:]]
    filas, geometrias = [], {}
    for registro in lector.shapeRecords():
        datos = dict(zip(nombres, registro.record))
        canton = normalizar_texto(datos["CANT_N"])
        canton = RENOMBRAR.get(canton, canton)
        fila = {"Provincia": normalizar_texto(datos["PROVINCIA"]), "Canton": canton}
        fila.update({nuevo: datos[viejo] for viejo, nuevo in CAMPOS.items()})
        filas.append(fila)
        geometrias[canton] = shape(registro.shape.__geo_interface__).buffer(0)

    ids = pd.DataFrame(filas)
    ids = ids[~ids["Canton"].isin(CANTONES_NUEVOS)].sort_values("Codigo").reset_index(drop=True)

    for nuevo, origen in CANTONES_NUEVOS.items():
        geometrias[origen] = unary_union([geometrias[origen], geometrias.pop(nuevo)])

    provincia = ids.set_index("Canton")["Provincia"]
    features = []
    for canton, geom in sorted(geometrias.items()):
        geom = transform(geom.simplify(150, preserve_topology=True), a_wgs84)
        features.append({
            "type": "Feature",
            "properties": {"Canton": canton, "Provincia": provincia[canton]},
            "geometry": json.loads(json.dumps(mapping(geom), default=float)),
        })
    return ids, {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ids, geo = construir(Path(sys.argv[1]))
    SALIDA_IDS.parent.mkdir(parents=True, exist_ok=True)
    ids.to_csv(SALIDA_IDS, index=False)
    with open(SALIDA_GEO, "w", encoding="utf-8") as f:
        json.dump(geo, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(ids)} cantones -> {SALIDA_IDS}")
    print(f"{len(geo['features'])} polígonos -> {SALIDA_GEO}")
