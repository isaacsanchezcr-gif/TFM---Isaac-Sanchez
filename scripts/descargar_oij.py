"""Descarga los archivos anuales de Estadísticas Policiales del OIJ.

Usa la API CKAN del Portal de Datos Abiertos del Poder Judicial y guarda los
archivos Excel en data/raw/. Descarga los archivos que faltan y vuelve a
descargar los que el portal haya modificado desde la última vez (el OIJ
actualiza los años ya publicados con denuncias tardías). Las fechas de
modificación conocidas se guardan en data/external/manifest_oij.json.

Uso:
    python scripts/descargar_oij.py
    python scripts/descargar_oij.py --forzar           # vuelve a descargar todo
    python scripts/descargar_oij.py --salida-github F  # escribe "cambios=true|false" en F (GitHub Actions)

Nota: el script no se ha podido probar contra el portal desde el entorno en que
se desarrolló; si la estructura de la API cambia, ajustar `listar_recursos`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parents[1]
PORTAL = "https://datosabiertospj.poder-judicial.go.cr"
DATASET = "estadisticas-policiales"
DESTINO = RAIZ / "data" / "raw"
MANIFIESTO = RAIZ / "data" / "external" / "manifest_oij.json"
FORMATOS = {"xls", "xlsx"}


def listar_recursos() -> list[dict]:
    """Recursos del dataset en formato Excel con año identificable."""
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
                "modificado": recurso.get("last_modified") or recurso.get("metadata_modified") or "",
            })
    return sorted(recursos, key=lambda r: r["anio"])


def descargar(forzar: bool = False) -> bool:
    """Descarga lo nuevo o modificado. Devuelve True si cambió algún archivo."""
    DESTINO.mkdir(parents=True, exist_ok=True)
    conocido = json.loads(MANIFIESTO.read_text()) if MANIFIESTO.exists() else {}
    cambios = False
    for recurso in listar_recursos():
        ruta = DESTINO / recurso["archivo"]
        igual = conocido.get(recurso["archivo"]) == recurso["modificado"]
        if ruta.exists() and igual and not forzar:
            print(f"{recurso['anio']}: sin cambios ({ruta.name})")
            continue
        print(f"{recurso['anio']}: descargando {recurso['url']}")
        with requests.get(recurso["url"], stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(ruta, "wb") as f:
                for bloque in r.iter_content(chunk_size=1 << 20):
                    f.write(bloque)
        conocido[recurso["archivo"]] = recurso["modificado"]
        cambios = cambios or not igual
    MANIFIESTO.write_text(json.dumps(conocido, indent=2, sort_keys=True))
    print("Hay datos nuevos" if cambios else "No hay datos nuevos")
    return cambios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forzar", action="store_true", help="vuelve a descargar todos los archivos")
    parser.add_argument("--salida-github", help="archivo $GITHUB_OUTPUT donde escribir si hubo cambios")
    args = parser.parse_args()
    hubo = descargar(args.forzar)
    if args.salida_github:
        with open(args.salida_github, "a") as f:
            f.write(f"cambios={'true' if hubo else 'false'}\n")
