"""Panel cantón–mes–delito, variables explicativas y métricas para la modelización.

Convención temporal: cada fila es un (cantón, delito, mes t) y todas las
variables se calculan con información disponible al cierre del mes t-1.
Así el modelo simula un pronóstico real hecho a principios de cada mes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import espacial, oij

PANDEMIA = (pd.Period("2020-03", "M"), pd.Period("2021-12", "M"))
IDS_VARIABLES = ["IDS_economica", "IDS_educativa", "IDS_salud", "IDS_participacion"]


def fines_de_semana(mes: pd.Period) -> int:
    dias = pd.date_range(mes.start_time, mes.end_time, freq="D")
    return int((dias.dayofweek >= 5).sum())


def construir_panel(raiz: Path) -> pd.DataFrame:
    """Panel completo con conteos mensuales y datos estructurales del cantón."""
    df = oij.cargar_todos(raiz / "data" / "raw")
    panel = oij.agregar_mensual(df)
    panel["Anio"] = panel["Mes"].dt.year

    ext = raiz / "data" / "external"
    pob = pd.read_csv(ext / "poblacion_cantonal.csv")
    ids = pd.read_csv(ext / "ids_2023_cantonal.csv")
    perfil = pd.read_csv(raiz / "data" / "processed" / "perfil_cantones.csv")

    panel = panel.merge(pob[["Canton", "Anio", "Poblacion"]], on=["Canton", "Anio"], how="left")
    panel = panel.merge(ids[["Canton"] + IDS_VARIABLES], on="Canton", how="left")
    panel = panel.merge(perfil[["Canton", "Grupo"]], on="Canton", how="left")
    return panel.sort_values(["Canton", "Delito", "Mes"]).reset_index(drop=True)


def _vecindad(raiz: Path) -> pd.DataFrame:
    aristas = pd.read_csv(raiz / "data" / "external" / "vecindad_cantonal.csv")
    cantones = sorted(set(aristas["Canton"]) | set(aristas["Vecino"]))
    w = pd.DataFrame(0, index=cantones, columns=cantones)
    for a, b in aristas.itertuples(index=False):
        w.loc[a, b] = w.loc[b, a] = 1
    return espacial.estandarizar_filas(w)


def indice_estacional_canton(panel: pd.DataFrame, hasta: pd.Period) -> pd.DataFrame:
    """Peso de cada mes del año en cada cantón (todos los delitos juntos),
    calculado solo con datos hasta `hasta` y sin los años de pandemia.
    Recoge efectos locales recurrentes como las fiestas cantonales."""
    base = panel[(panel["Mes"] <= hasta)
                 & ~panel["Mes"].between(*PANDEMIA)]
    m = base.groupby(["Canton", base["Mes"].dt.month])["Delitos"].sum().unstack(fill_value=0)
    indice = m.div(m.mean(axis=1).replace(0, np.nan), axis=0).fillna(1.0)
    # Suavizado hacia 1 en cantones con pocos delitos, para no sobreajustar ruido
    peso = (m.sum(axis=1) / (m.sum(axis=1) + 500)).values[:, None]
    indice = 1 + (indice - 1) * peso
    return indice.stack().rename("Estacional_canton").reset_index().rename(columns={"Mes": "MesNum"})


def construir_variables(panel: pd.DataFrame, raiz: Path, fin_entrenamiento: pd.Period) -> pd.DataFrame:
    """Añade al panel las variables explicativas (todas con datos hasta t-1)."""
    p = panel.copy()
    g = p.groupby(["Canton", "Delito"])["Delitos"]
    for k in (1, 2, 3, 12):
        p[f"lag_{k}"] = g.shift(k)
    p["media_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    p["media_12"] = g.transform(lambda s: s.shift(1).rolling(12).mean())
    p["tendencia"] = (p["media_3"] + 1) / (p["media_12"] + 1)
    p["tasa_lag_1"] = p["lag_1"] / p["Poblacion"] * 1e5
    p["tasa_media_12"] = p["media_12"] / p["Poblacion"] * 1e5

    # Rezago espacial: tasa media de los cantones vecinos en el mes anterior
    w = _vecindad(raiz)
    tasa = p.pivot_table(index=["Delito", "Mes"], columns="Canton", values="tasa_lag_1")
    tasa = tasa[w.columns]
    vecinos = pd.DataFrame(tasa.fillna(0).values @ w.values.T, index=tasa.index,
                           columns=pd.Index(w.index, name="Canton"))
    vecinos = vecinos.stack().rename("tasa_vecinos_lag_1").reset_index()
    p = p.merge(vecinos, on=["Delito", "Mes", "Canton"], how="left")

    # Contexto nacional del mismo delito en el mes anterior
    nacional = p.groupby(["Delito", "Mes"])["Delitos"].sum().groupby(level=0).shift(1)
    p = p.merge(nacional.rename("nacional_lag_1").reset_index(), on=["Delito", "Mes"], how="left")

    # Calendario
    p["MesNum"] = p["Mes"].dt.month
    meses = p["Mes"].drop_duplicates()
    p["fines_de_semana"] = p["Mes"].map(dict(zip(meses, meses.map(fines_de_semana))))
    p["dias_mes"] = p["Mes"].dt.days_in_month
    p["pandemia"] = p["Mes"].between(*PANDEMIA).astype(int)
    p = p.merge(indice_estacional_canton(panel, fin_entrenamiento), on=["Canton", "MesNum"], how="left")

    p["log_poblacion"] = np.log(p["Poblacion"])
    p["Delito_cat"] = pd.Categorical(p["Delito"], categories=oij.DELITOS)
    return p


VARIABLES = [
    "Delito_cat", "lag_1", "lag_2", "lag_3", "lag_12", "media_3", "media_12", "tendencia",
    "tasa_lag_1", "tasa_media_12", "tasa_vecinos_lag_1", "nacional_lag_1",
    "MesNum", "fines_de_semana", "dias_mes", "pandemia", "Estacional_canton",
    "log_poblacion", "Grupo",
] + IDS_VARIABLES


def metricas(y: pd.Series, pred: pd.Series) -> dict:
    """MAE, RMSE y WAPE (error absoluto total dividido entre el total real)."""
    e = pred - y
    return {
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "WAPE": float(np.abs(e).sum() / y.sum()),
    }


def sarima_un_paso(datos: pd.DataFrame, fin_train: pd.Period, ini_eval: pd.Period,
                   fin_eval: pd.Period | None = None) -> pd.DataFrame:
    """Pronóstico a un mes con un SARIMA(1,0,1)(1,0,0)12 por cada serie cantón–delito.

    Cada modelo se ajusta una vez con los datos hasta `fin_train` (log de los conteos,
    pandemia como variable exógena) y después se actualiza mes a mes con el dato
    observado, sin reestimar parámetros, como se haría en producción. Si una serie
    no converge, se usa la media de sus últimos 12 meses.
    """
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    datos = datos[["Canton", "Delito", "Mes", "Delitos", "pandemia"]]
    if fin_eval is not None:
        datos = datos[datos["Mes"] <= fin_eval]
    salida, fallos = [], 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (canton, delito), g in datos.groupby(["Canton", "Delito"]):
            g = g.set_index("Mes")
            y, ex = np.log1p(g["Delitos"].astype(float)), g[["pandemia"]]
            y_tr, y_ev = y[:fin_train], y[ini_eval:]
            try:
                r = SARIMAX(y_tr, exog=ex[:fin_train], order=(1, 0, 1), seasonal_order=(1, 0, 0, 12),
                            trend="c").fit(disp=False, maxiter=50)
                pred = []
                for i, mes in enumerate(y_ev.index):
                    pred.append(r.forecast(1, exog=ex.loc[[mes]]).iloc[0])
                    r = r.append(y_ev.iloc[[i]], exog=ex.loc[[mes]], refit=False)
                pred = np.expm1(np.array(pred))
            except Exception:
                fallos += 1
                pred = np.repeat(np.expm1(y_tr[-12:]).mean(), len(y_ev))
            salida.append(pd.DataFrame({"Canton": canton, "Delito": delito, "Mes": y_ev.index,
                                        "SARIMA": np.clip(pred, 0, None)}))
    print(f"Series sin convergencia: {fallos}")
    return pd.concat(salida, ignore_index=True)
