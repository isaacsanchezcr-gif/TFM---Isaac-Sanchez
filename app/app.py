"""Sistema de alerta temprana de criminalidad cantonal · aplicación Streamlit.

Ejecutar desde la raíz del repositorio:
    streamlit run app/app.py

Lee los archivos que genera `scripts/actualizar.py` en app/datos/ y el modelo de
producción en models/lgbm_produccion.txt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import modelo, oij  # noqa: E402

DATOS = RAIZ / "app" / "datos"
AZUL = "#1f3a5f"
AZUL_CLARO = "#9ec5f4"
ROJO = "#b8322f"
GRIS = "#8a929b"

ET = {
    "HURTO": "Hurto", "ASALTO": "Asalto", "ROBO": "Robo", "ROBO DE VEHICULO": "Robo de vehículo",
    "TACHA DE VEHICULO": "Tacha de vehículo", "HOMICIDIO": "Homicidio",
}
NOMBRE = {
    "media_12": "Nivel de los últimos 12 meses", "media_3": "Media de los últimos 3 meses",
    "lag_1": "Delitos del mes anterior", "lag_2": "Delitos de hace 2 meses", "lag_3": "Delitos de hace 3 meses",
    "lag_12": "Mismo mes del año anterior", "tendencia": "Tendencia reciente",
    "tasa_lag_1": "Tasa del mes anterior", "tasa_media_12": "Tasa de los últimos 12 meses",
    "tasa_vecinos_lag_1": "Tasa en cantones vecinos", "nacional_lag_1": "Total nacional del mes anterior",
    "Estacional_canton": "Estacionalidad local (fiestas)", "MesNum": "Mes del año",
    "fines_de_semana": "Fines de semana del mes", "dias_mes": "Días del mes", "pandemia": "Pandemia",
    "log_poblacion": "Población", "Grupo": "Perfil delictivo del cantón", "Delito_cat": "Tipo de delito",
    "IDS_economica": "IDS económico", "IDS_educativa": "IDS educativo", "IDS_salud": "IDS salud",
    "IDS_participacion": "IDS participación electoral",
}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

st.set_page_config(page_title="Alerta temprana de criminalidad · Costa Rica", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    h1, h2, h3 {color: #1f3a5f;}
    div[data-testid="stMetricValue"] {color: #1f3a5f;}
    .nota {color: #5b636d; font-size: 0.85rem;}
    </style>""",
    unsafe_allow_html=True,
)


# ---------- Datos ----------

@st.cache_data
def cargar():
    pron = pd.read_parquet(DATOS / "pronostico.parquet")
    hist = pd.read_parquet(DATOS / "historico.parquet")
    meta = json.loads((DATOS / "metadatos.json").read_text(encoding="utf-8"))
    geo = json.loads((RAIZ / "data" / "external" / "cantones.geojson").read_text(encoding="utf-8"))
    nombres = pd.read_csv(RAIZ / "data" / "external" / "ids_2023_cantonal.csv").set_index("Canton")["Nombre"]
    return pron, hist, meta, geo, nombres.to_dict()


@st.cache_resource
def cargar_modelo():
    return lgb.Booster(model_file=str(RAIZ / "models" / "lgbm_produccion.txt"))


def nombre_mes(periodo: str) -> str:
    anio, mes = periodo.split("-")
    return f"{MESES[int(mes) - 1]} de {anio}"


def titulo(canton: str) -> str:
    """Nombre oficial con tildes (del MIDEPLAN)."""
    return NOMBRES.get(canton, canton.title())


def mes_corto(periodo: str) -> str:
    anio, mes = periodo.split("-")
    return f"{MESES[int(mes) - 1][:3]} {anio[2:]}"


def motivos(fila: pd.Series, n: int = 3) -> str:
    c = pd.Series({v: fila[f"c_{v}"] for v in modelo.VARIABLES})
    c = c[c > 0.01].sort_values(ascending=False).head(n)
    return " · ".join(f"{NOMBRE[v]} (×{np.exp(x):.2f})" for v, x in c.items()) or "Sin factores destacados"


pron, hist, meta, geo, NOMBRES = cargar()
booster = cargar_modelo()
mes_txt = nombre_mes(meta["mes_pronosticado"])

# ---------- Encabezado ----------

st.title("Alerta temprana de criminalidad cantonal")
st.markdown(
    f"Pronóstico de delitos denunciados ante el OIJ para **{mes_txt}**, por cantón y tipo de delito. "
    f"<span class='nota'>Datos hasta {nombre_mes(meta['ultimo_mes_con_datos'])} · "
    f"actualizado {meta['actualizado']}</span>",
    unsafe_allow_html=True,
)

pestanas = st.tabs(["Panorama", "Consulta por cantón", "Simulador", "Acerca del sistema"])

# ---------- Panorama ----------

with pestanas[0]:
    delito = st.selectbox("Tipo de delito", ["Todos"] + oij.DELITOS,
                          format_func=lambda d: "Todos los delitos" if d == "Todos" else ET[d])
    datos = pron if delito == "Todos" else pron[pron["Delito"] == delito]

    previstos = datos["Prevision"].sum()
    habitual = datos["media_12"].sum()
    alertables = datos[datos["media_12"] >= 3]
    n_alertas = int((alertables["Riesgo"] >= 1.10).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Delitos previstos", f"{previstos:,.0f}".replace(",", "."),
              f"{previstos / habitual - 1:+.1%} frente a la media de 12 meses", delta_color="inverse")
    c2.metric("Media de los últimos 12 meses", f"{habitual:,.0f}".replace(",", "."))
    c3.metric("Alertas de aumento (≥ +10 %)", n_alertas)

    por_canton = datos.groupby("Canton").agg(Prevision=("Prevision", "sum"), Habitual=("media_12", "sum"),
                                             Poblacion=("Poblacion", "first"))
    por_canton["Tasa"] = por_canton["Prevision"] / por_canton["Poblacion"] * 1e5
    por_canton["Variacion"] = (por_canton["Prevision"] / (por_canton["Habitual"] + 0.5) - 1) * 100

    vista = st.radio("Mapa", ["Variación frente a lo habitual", "Tasa prevista por 100.000 habitantes"],
                     horizontal=True)
    if vista.startswith("Variación"):
        z, escala, rango, sufijo = por_canton["Variacion"].clip(-30, 30), \
            [[0, "#184f95"], [0.5, "#eceae6"], [1, ROJO]], (-30, 30), "%"
    else:
        z, escala, rango, sufijo = por_canton["Tasa"], [[0, "#eaf2fc"], [1, AZUL]], \
            (0, por_canton["Tasa"].quantile(0.95)), ""
    fig = go.Figure(go.Choroplethmap(
        geojson=geo, featureidkey="properties.Canton", locations=por_canton.index, z=z,
        zmin=rango[0], zmax=rango[1], colorscale=escala, marker_line_color="white", marker_line_width=0.6,
        customdata=np.stack([por_canton.index.map(titulo), por_canton["Prevision"], por_canton["Habitual"],
                             por_canton["Variacion"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>Previstos: %{customdata[1]:.0f}<br>"
                      "Media 12 meses: %{customdata[2]:.0f}<br>Variación: %{customdata[3]:+.0f} %<extra></extra>",
        colorbar=dict(title=sufijo, thickness=12, len=0.6),
    ))
    fig.update_layout(map=dict(style="white-bg", center=dict(lat=9.62, lon=-84.2), zoom=6.85),
                      height=560, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Las 10 alertas del mes")
    st.markdown("<span class='nota'>Cantones y delitos donde se espera la mayor subida frente a su media de 12 meses "
                "(solo series con al menos 3 casos al mes de media).</span>", unsafe_allow_html=True)
    top = alertables.sort_values("Riesgo", ascending=False).head(10)
    tabla = pd.DataFrame({
        "Cantón": top["Canton"].map(titulo), "Delito": top["Delito"].map(ET),
        "Media 12 meses": top["media_12"].round(1), "Previsto": top["Prevision"].round(1),
        "Variación": ((top["Riesgo"] - 1) * 100).round(0).astype(int).astype(str) + " %",
        "Principales motivos": [motivos(f) for _, f in top.iterrows()],
    })
    tabla.index = range(1, len(tabla) + 1)
    st.dataframe(tabla, width="stretch")

    st.subheader("Dónde se concentrará la carga de trabajo")
    carga = por_canton.sort_values("Prevision", ascending=False).head(10)
    st.bar_chart(pd.Series(carga["Prevision"].values, index=carga.index.map(titulo), name="Delitos previstos"),
                 color=AZUL, horizontal=True)

# ---------- Consulta por cantón ----------

with pestanas[1]:
    col1, col2 = st.columns(2)
    cantones = sorted(pron["Canton"].unique())
    canton = col1.selectbox("Cantón", cantones, index=cantones.index("SAN JOSE"), format_func=titulo)
    delito_c = col2.selectbox("Delito", oij.DELITOS, format_func=ET.get, key="delito_consulta")

    fila = pron[(pron["Canton"] == canton) & (pron["Delito"] == delito_c)].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Previstos en {mes_txt}", f"{fila['Prevision']:.1f}")
    m2.metric("Media de los últimos 12 meses", f"{fila['media_12']:.1f}")
    m3.metric("Variación prevista", f"{fila['Riesgo'] - 1:+.0%}")

    serie = hist[(hist["Canton"] == canton) & (hist["Delito"] == delito_c)]
    etiquetas = [mes_corto(m) for m in serie["Mes"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=etiquetas, y=serie["Delitos"], mode="lines+markers", name="Observado",
                             line=dict(color=AZUL, width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[etiquetas[-1], mes_corto(meta["mes_pronosticado"])],
                             y=[serie["Delitos"].iloc[-1], fila["Prevision"]], mode="lines+markers",
                             name="Pronóstico", line=dict(color=ROJO, width=2, dash="dot"), marker=dict(size=8)))
    fig.add_hline(y=fila["media_12"], line=dict(color=GRIS, dash="dash", width=1),
                  annotation_text="media 12 meses", annotation_position="top left")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=30, b=0), title=f"{ET[delito_c]} en {titulo(canton)}",
                      legend=dict(orientation="h", y=-0.2), yaxis_title="Delitos por mes",
                      plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, type="category", nticks=12)
    fig.update_yaxes(gridcolor="#e6e9ed")
    st.plotly_chart(fig, width="stretch")

    st.subheader("¿Por qué este pronóstico?")
    st.markdown("<span class='nota'>El modelo parte de la media de los últimos 12 meses y la corrige. "
                "Cada barra indica por cuánto multiplica una variable el pronóstico.</span>", unsafe_allow_html=True)
    c = pd.Series({NOMBRE[v]: fila[f"c_{v}"] for v in modelo.VARIABLES})
    c = c.reindex(c.abs().sort_values(ascending=False).index).head(8)[::-1]
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h",
                           marker_color=[ROJO if x > 0 else "#2a78d6" for x in c.values],
                           text=[f"×{np.exp(x):.2f}" for x in c.values], textposition="outside",
                           hovertemplate="%{y}: %{text}<extra></extra>"))
    lim = float(np.abs(c.values).max()) * 1.4 or 0.1
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[-lim, lim], zeroline=True,
                      zerolinecolor=GRIS, showticklabels=False, showgrid=False), plot_bgcolor="white",
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

# ---------- Simulador ----------

with pestanas[2]:
    st.markdown("Modifica los datos de entrada y el modelo recalcula el pronóstico al instante. "
                "Sirve para explorar escenarios: por ejemplo, qué pasaría si el último mes hubiera sido excepcionalmente malo "
                "o si los cantones vecinos estuvieran en alza.")
    col1, col2 = st.columns(2)
    canton_s = col1.selectbox("Cantón", cantones, index=cantones.index("SAN JOSE"), format_func=titulo, key="canton_sim")
    delito_s = col2.selectbox("Delito", oij.DELITOS, format_func=ET.get, key="delito_sim")
    base = pron[(pron["Canton"] == canton_s) & (pron["Delito"] == delito_s)].iloc[0]

    techo = max(10, int(base["media_12"] * 4) + 5)
    a, b = st.columns(2)
    lag_1 = a.slider("Delitos del mes anterior", 0, techo, int(base["lag_1"]))
    lag_2 = a.slider("Delitos de hace 2 meses", 0, techo, int(base["lag_2"]))
    lag_3 = a.slider("Delitos de hace 3 meses", 0, techo, int(base["lag_3"]))
    vecinos = b.slider("Tasa en cantones vecinos (por 100.000 hab.)", 0.0,
                       float(max(5.0, base["tasa_vecinos_lag_1"] * 4)), float(round(base["tasa_vecinos_lag_1"], 1)))
    estacional = b.slider("Estacionalidad local del mes (1 = mes normal)", 0.5, 2.5,
                          float(round(base["Estacional_canton"], 2)), 0.05)
    nacional = b.slider("Total nacional del mes anterior", 0, int(base["nacional_lag_1"] * 2) + 10,
                        int(base["nacional_lag_1"]))

    x = base.copy()
    x["lag_1"], x["lag_2"], x["lag_3"] = lag_1, lag_2, lag_3
    x["media_3"] = (lag_1 + lag_2 + lag_3) / 3
    x["tendencia"] = (x["media_3"] + 1) / (x["media_12"] + 1)
    x["tasa_lag_1"] = lag_1 / x["Poblacion"] * 1e5
    x["tasa_vecinos_lag_1"], x["Estacional_canton"], x["nacional_lag_1"] = vecinos, estacional, nacional
    entrada = pd.DataFrame([x])
    entrada["Delito_cat"] = pd.Categorical([delito_s], categories=oij.DELITOS)
    for v in modelo.VARIABLES:
        if v != "Delito_cat":
            entrada[v] = pd.to_numeric(entrada[v])
    nueva = float(modelo.predecir(booster, entrada)[0])

    r1, r2, r3 = st.columns(3)
    r1.metric("Pronóstico original", f"{base['Prevision']:.1f}")
    r2.metric("Pronóstico con tus datos", f"{nueva:.1f}", f"{nueva - base['Prevision']:+.1f}", delta_color="inverse")
    r3.metric("Frente a la media de 12 meses", f"{nueva / (base['media_12'] + 0.5) - 1:+.0%}")

# ---------- Acerca del sistema ----------

with pestanas[3]:
    st.markdown(f"""
**Qué hace.** Cada vez que el OIJ publica datos nuevos, el sistema reentrena un modelo LightGBM con todas las denuncias
desde 2015 y pronostica, para el mes siguiente, los delitos de cada tipo en cada uno de los 81 cantones (con Río Cuarto,
Monteverde y Puerto Jiménez integrados en su cantón de origen).

**Cómo se evaluó.** Entrenado con datos de 2015 a 2024 y evaluado en 21 meses que nunca vio (enero 2025 – septiembre 2026):

- Error (WAPE) del 34,5 %, frente al 36,4 % de la media de 12 meses y al 37,9 % de un SARIMA por serie.
  Incluso un modelo perfecto tendría alrededor de un 25 % de error, por el componente aleatorio de los delitos.
- De las 10 alertas mensuales de aumento, el 15 % fueron aumentos reales, frente al 7–8 % de los criterios simples.

**Fuentes.** Estadísticas Policiales del OIJ (Poder Judicial, licencia CC BY); proyecciones de población del INEC;
Índice de Desarrollo Social 2023 y límites cantonales del MIDEPLAN.

**Limitaciones.** Los datos son denuncias, no la criminalidad real. El último mes publicado puede estar algo
incompleto porque el OIJ añade denuncias tardías. Para el homicidio, con pocos casos por cantón, el azar domina.
Las alertas son una ayuda para priorizar la revisión, no una predicción segura.

<span class='nota'>Trabajo Fin de Máster · Máster en Business Intelligence y Data Science · Universidad Complutense de Madrid · Isaac Sánchez</span>
""", unsafe_allow_html=True)
