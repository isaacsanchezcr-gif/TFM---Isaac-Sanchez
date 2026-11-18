"""Actualiza el sistema de alerta temprana con los últimos datos del OIJ.

Pasos:
1. (Opcional, --descargar) descarga del portal del Poder Judicial los archivos
   nuevos o modificados.
2. Reconstruye el panel cantón–delito–mes, incluido el mes siguiente al último
   mes completo publicado.
3. Reentrena el modelo de producción con la configuración elegida en el notebook 04.
4. Calcula el pronóstico del mes siguiente y sus explicaciones (SHAP).
5. Escribe los archivos que lee la aplicación en app/datos/.

Uso:
    python scripts/actualizar.py              # con los archivos que ya están en data/raw
    python scripts/actualizar.py --descargar  # descarga antes de actualizar
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import modelo  # noqa: E402

SALIDA = RAIZ / "app" / "datos"
MODELOS = RAIZ / "models"
MESES_HISTORICO = 36


def entrenar_produccion(X: pd.DataFrame, ultimo: pd.Period) -> lgb.Booster:
    config = json.loads((MODELOS / "lgbm_config.json").read_text())
    datos = X[X["Mes"] <= ultimo].dropna(subset=["lag_12", "media_12"])
    m = lgb.LGBMRegressor(
        objective="poisson", n_estimators=int(config["arboles"]), learning_rate=0.03,
        num_leaves=int(config["num_leaves"]), min_child_samples=int(config["min_child_samples"]),
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0,
    )
    m.fit(datos[modelo.VARIABLES], datos["Delitos"], init_score=modelo.offset(datos))
    m.booster_.save_model(str(MODELOS / "lgbm_produccion.txt"))
    return m.booster_


def actualizar(descargar: bool = False) -> dict:
    if descargar:
        from scripts.descargar_oij import descargar as bajar
        bajar()

    panel = modelo.construir_panel(RAIZ, incluir_mes_siguiente=True)
    ultimo = panel.dropna(subset=["Delitos"])["Mes"].max()
    siguiente = ultimo + 1
    X = modelo.construir_variables(panel, RAIZ, ultimo)
    booster = entrenar_produccion(X, ultimo)

    futuro = X[X["Mes"] == siguiente].copy()
    futuro["Prevision"] = modelo.predecir(booster, futuro)
    futuro["Riesgo"] = futuro["Prevision"] / (futuro["media_12"] + 0.5)
    contrib = modelo.contribuciones(booster, futuro).add_prefix("c_")
    futuro = pd.concat([futuro, contrib], axis=1)

    SALIDA.mkdir(parents=True, exist_ok=True)
    base = ["Canton", "Provincia", "Delito", "Mes", "Prevision", "Riesgo", "Poblacion"]
    columnas = base + [v for v in modelo.VARIABLES if v != "Delito_cat"] + list(contrib.columns)
    futuro[columnas].assign(Mes=futuro["Mes"].astype(str)).to_parquet(SALIDA / "pronostico.parquet", index=False)

    historico = panel[panel["Mes"].between(ultimo - (MESES_HISTORICO - 1), ultimo)]
    historico[["Canton", "Delito", "Mes", "Delitos"]].assign(Mes=historico["Mes"].astype(str)) \
        .to_parquet(SALIDA / "historico.parquet", index=False)

    meta = {
        "ultimo_mes_con_datos": str(ultimo),
        "mes_pronosticado": str(siguiente),
        "actualizado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "delitos_previstos": int(round(futuro["Prevision"].sum())),
        "media_12_nacional": int(round(futuro["media_12"].sum())),
    }
    (SALIDA / "metadatos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--descargar", action="store_true", help="descarga datos nuevos del OIJ antes de actualizar")
    print(json.dumps(actualizar(parser.parse_args().descargar), ensure_ascii=False, indent=2))
