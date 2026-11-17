"""Descarga los archivos anuales de Estadísticas Policiales del OIJ.

Usa la API CKAN del Portal de Datos Abiertos del Poder Judicial y guarda los
archivos Excel en data/raw/. Solo descarga los que aún no existen localmente.

Uso:
    python scripts/descargar_oij.py
    python scripts/descargar_oij.py --forzar   # vuelve a descargar todo
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests

PORTAL = "https://datosabiertospj.poder-judicial.go.cr"
DATASET = "estadisticas-policiales"
DESTINO = Path(__file__).resolve().parents[1] / "data" / "raw"
FORMATOS = {"xls", "xlsx"}


def listar_recursos() -> list[dict]:
    """Devuelve los recursos del dataset con formato Excel y año identificable."""
    url = f"{PORTAL}/api/3/action/package_show"
    respuesta = requests.get(url, params={"id": DATASET}, timeout=60)
    respuesta.raise_for_status()
    recursos = []
    for recurso in respuesta.json()["result"]["resources"]:
        formato = (recurso.get("format") or "").lower()
        nombre_archivo = recurso["url"].rsplit("/", 1)[-1]
        anio = re.search(r"(20\d{2})", nombre_archivo) or re.search(r"(20\d{2})", recurso.get("name", ""))
        if formato in FORMATOS and anio:
            recursos.append({
                "anio": anio.group(1),
                "url": recurso["url"],
                "archivo": nombre_archivo,
                "modificado": recurso.get("last_modified") or recurso.get("metadata_modified"),
            })
    return sorted(recursos, key=lambda r: r["anio"])


def descargar(forzar: bool = False) -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for recurso in listar_recursos():
        ruta = DESTINO / recurso["archivo"]
        if ruta.exists() and not forzar:
            print(f"{recurso['anio']}: ya existe ({ruta.name})")
            continue
        print(f"{recurso['anio']}: descargando {recurso['url']}")
        with requests.get(recurso["url"], stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(ruta, "wb") as f:
                for bloque in r.iter_content(chunk_size=1 << 20):
                    f.write(bloque)
    print(f"Archivos en {DESTINO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forzar", action="store_true", help="vuelve a descargar los archivos existentes")
    descargar(parser.parse_args().forzar)
