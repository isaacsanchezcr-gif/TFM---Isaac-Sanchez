"""Vecindad entre cantones y autocorrelación espacial (Moran global y local).

Se implementa a mano, sin librerías especializadas, para que cada paso sea
transparente en la memoria del TFM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def matriz_vecindad(geometrias: dict, tolerancia: float = 1e-4) -> pd.DataFrame:
    """Matriz binaria de contigüidad tipo 'reina': dos cantones son vecinos si sus
    límites se tocan (en un tramo o en un punto). La tolerancia, en grados,
    absorbe pequeños huecos de la simplificación de los polígonos."""
    nombres = sorted(geometrias)
    ampliadas = {n: geometrias[n].buffer(tolerancia) for n in nombres}
    w = pd.DataFrame(0, index=nombres, columns=nombres, dtype=int)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            if ampliadas[a].intersects(ampliadas[b]):
                w.loc[a, b] = w.loc[b, a] = 1
    return w


def estandarizar_filas(w: pd.DataFrame) -> pd.DataFrame:
    """Cada fila suma 1: el 'rezago espacial' pasa a ser el promedio de los vecinos."""
    suma = w.sum(axis=1).replace(0, np.nan)
    return w.div(suma, axis=0).fillna(0)


def rezago_espacial(valores: pd.Series, w: pd.DataFrame) -> pd.Series:
    """Promedio de la variable en los cantones vecinos."""
    ws = estandarizar_filas(w)
    v = valores.reindex(ws.columns)
    return pd.Series(ws.values @ v.values, index=ws.index)


def moran_global(valores: pd.Series, w: pd.DataFrame, permutaciones: int = 999,
                 semilla: int = 42) -> dict:
    """I de Moran global con prueba de permutaciones.

    I > 0: los cantones vecinos tienen valores parecidos (agrupamiento).
    I ≈ E[I] = -1/(n-1): distribución espacial aleatoria.
    """
    ws = estandarizar_filas(w).values
    x = valores.reindex(w.index).values.astype(float)
    z = x - x.mean()
    n = len(z)

    def calcular(v):
        return (n / ws.sum()) * (v @ ws @ v) / (v @ v)

    observado = calcular(z)
    rng = np.random.default_rng(semilla)
    simulados = np.array([calcular(rng.permutation(z)) for _ in range(permutaciones)])
    p = (np.sum(simulados >= observado) + 1) / (permutaciones + 1)
    return {"I": observado, "esperado": -1 / (n - 1), "p_valor": p}


def moran_local(valores: pd.Series, w: pd.DataFrame, permutaciones: int = 999,
                semilla: int = 42, alfa: float = 0.05) -> pd.DataFrame:
    """Indicadores locales de asociación espacial (LISA).

    Clasifica cada cantón significativo como:
      Alto-Alto (foco), Bajo-Bajo (zona fría), Alto-Bajo o Bajo-Alto (atípicos).
    """
    ws = estandarizar_filas(w)
    x = valores.reindex(w.index).values.astype(float)
    z = (x - x.mean()) / x.std()
    rezago = ws.values @ z
    local = z * rezago

    rng = np.random.default_rng(semilla)
    n = len(z)
    p = np.ones(n)
    for i in range(n):
        vecinos = np.flatnonzero(ws.values[i])
        if len(vecinos) == 0:
            continue
        otros = np.delete(z, i)
        pesos = ws.values[i, vecinos]
        sim = np.array([z[i] * (pesos @ rng.choice(otros, len(vecinos), replace=False))
                        for _ in range(permutaciones)])
        mas_extremos = np.sum(sim >= local[i]) if local[i] >= 0 else np.sum(sim <= local[i])
        p[i] = (mas_extremos + 1) / (permutaciones + 1)

    cuadrante = np.select(
        [(z > 0) & (rezago > 0), (z < 0) & (rezago < 0), (z > 0) & (rezago < 0), (z < 0) & (rezago > 0)],
        ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto"],
        default="No significativo",
    )
    tipo = np.where(p < alfa, cuadrante, "No significativo")
    return pd.DataFrame({"z": z, "rezago": rezago, "I_local": local, "p_valor": p, "tipo": tipo},
                        index=w.index)
