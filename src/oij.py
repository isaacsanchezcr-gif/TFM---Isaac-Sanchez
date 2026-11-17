"""Carga, limpieza y agregación de las Estadísticas Policiales del OIJ.

Fuente: Portal de Datos Abiertos del Poder Judicial de Costa Rica
https://datosabiertospj.poder-judicial.go.cr/dataset/estadisticas-policiales
Licencia: Creative Commons Attribution.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

COLUMNAS = [
    "Delito", "SubDelito", "Fecha", "Victima", "SubVictima", "Edad",
    "Genero", "Nacionalidad", "Provincia", "Canton", "Distrito",
]

DELITOS = ["HURTO", "ASALTO", "ROBO", "ROBO DE VEHICULO", "TACHA DE VEHICULO", "HOMICIDIO"]

# Cantones creados después de 2015. Para tener series comparables en todo el
# periodo se reasignan al cantón del que se segregaron.
CANTONES_NUEVOS = {
    "RIO CUARTO": "GRECIA",        # creado en 2017
    "MONTEVERDE": "PUNTARENAS",    # creado en 2021
    "PUERTO JIMENEZ": "GOLFITO",   # creado en 2022
}

VALORES_DESCONOCIDOS = {"", "DESCONOCIDO", "DESCONOCIDA", "IGNORADO"}


def normalizar_texto(valor: object) -> str:
    """Mayúsculas, sin tildes y sin espacios sobrantes."""
    if pd.isna(valor):
        return ""
    texto = unicodedata.normalize("NFD", str(valor).upper().strip())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto)


def _leer_csv_sin_encabezado(ruta: Path) -> pd.DataFrame:
    """Los CSV del OIJ no traen encabezado y algunas nacionalidades llevan comas
    sin comillas (p. ej. "CONGO, REPUBLICA DEMOCRATICA DEL"). Se reconstruyen a
    11 columnas uniendo los campos sobrantes en Nacionalidad."""
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            campos = linea.rstrip("\r\n").split(",")
            if len(campos) > len(COLUMNAS):
                campos = campos[:7] + [",".join(campos[7:-3])] + campos[-3:]
            filas.append(campos)
    return pd.DataFrame(filas, columns=COLUMNAS)


def leer_archivo(ruta: str | Path) -> pd.DataFrame:
    """Lee un archivo anual del OIJ (xls, xlsx o csv) con las columnas estándar."""
    ruta = Path(ruta)
    if ruta.suffix.lower() == ".csv":
        df = _leer_csv_sin_encabezado(ruta)
    else:
        df = pd.read_excel(ruta, dtype=str, keep_default_na=False)
    faltantes = set(COLUMNAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"{ruta.name}: faltan columnas {sorted(faltantes)}")
    return df[COLUMNAS]


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza texto, convierte fechas y armoniza cantones."""
    df = df.copy()
    texto = [c for c in COLUMNAS if c != "Fecha"]
    for col in texto:
        df[col] = df[col].map(normalizar_texto)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Canton_original"] = df["Canton"]
    df["Canton"] = df["Canton"].replace(CANTONES_NUEVOS)
    df["Anio"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.to_period("M")
    return df


def cargar_todos(carpeta: str | Path = "data/raw") -> pd.DataFrame:
    """Carga y limpia todos los archivos del OIJ de una carpeta.

    Si hay varios formatos del mismo año (xls y csv), usa uno solo para no
    duplicar registros: prioriza xlsx, luego xls, luego csv.
    """
    carpeta = Path(carpeta)
    prioridad = {".xlsx": 0, ".xls": 1, ".csv": 2}
    por_anio: dict[str, Path] = {}
    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in prioridad:
            continue
        anio = re.search(r"(20\d{2})", ruta.name)
        if not anio:
            continue
        actual = por_anio.get(anio.group(1))
        if actual is None or prioridad[ruta.suffix.lower()] < prioridad[actual.suffix.lower()]:
            por_anio[anio.group(1)] = ruta
    if not por_anio:
        raise FileNotFoundError(f"No hay archivos del OIJ en {carpeta}")
    partes = []
    for anio, ruta in sorted(por_anio.items()):
        parte = leer_archivo(ruta)
        parte["Archivo"] = ruta.name
        partes.append(parte)
    return limpiar(pd.concat(partes, ignore_index=True))


def perfil_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de valores vacíos o desconocidos y número de categorías por columna."""
    filas = []
    for col in COLUMNAS:
        serie = df[col]
        if col == "Fecha":
            desconocido = serie.isna().mean()
        else:
            desconocido = serie.isin(VALORES_DESCONOCIDOS).mean()
        filas.append({
            "columna": col,
            "categorias": serie.nunique(),
            "pct_vacio_o_desconocido": round(100 * desconocido, 2),
        })
    return pd.DataFrame(filas)


def ultimo_mes_completo(df: pd.DataFrame) -> pd.Period:
    """Último mes con datos hasta su último día (descarta meses en curso)."""
    ultima = df["Fecha"].max()
    mes = ultima.to_period("M")
    return mes if ultima.day == ultima.days_in_month else mes - 1


def agregar_mensual(df: pd.DataFrame, solo_meses_completos: bool = True) -> pd.DataFrame:
    """Cuenta delitos por cantón, mes y tipo de delito.

    Devuelve una rejilla completa: las combinaciones sin delitos aparecen con 0,
    lo que es imprescindible para modelar series temporales.
    """
    datos = df[~df["Canton"].isin(VALORES_DESCONOCIDOS) & df["Fecha"].notna()]
    if solo_meses_completos:
        datos = datos[datos["Mes"] <= ultimo_mes_completo(datos)]
    # En Costa Rica no hay dos cantones con el mismo nombre, así que el cantón
    # basta como clave; la provincia se asigna con su valor más frecuente.
    conteo = datos.groupby(["Canton", "Mes", "Delito"]).size().rename("Delitos").reset_index()
    cantones = datos.groupby("Canton")["Provincia"].agg(lambda s: s.mode().iat[0])
    # Solo meses con datos: si falta algún año en data/raw no se rellena con ceros falsos.
    meses = sorted(datos["Mes"].unique())
    rango = pd.period_range(min(meses), max(meses), freq="M")
    if len(meses) < len(rango):
        faltan = sorted({str(m.year) for m in set(rango) - set(meses)})
        print(f"Aviso: no hay datos para los años {', '.join(faltan)}; descárgalos con scripts/descargar_oij.py")
    rejilla = pd.MultiIndex.from_product(
        [cantones.index, meses, DELITOS], names=["Canton", "Mes", "Delito"]
    ).to_frame(index=False)
    rejilla["Provincia"] = rejilla["Canton"].map(cantones)
    salida = rejilla.merge(conteo, on=["Canton", "Mes", "Delito"], how="left")
    salida["Delitos"] = salida["Delitos"].fillna(0).astype(int)
    return salida[["Provincia", "Canton", "Mes", "Delito", "Delitos"]]
