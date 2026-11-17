# Sistema predictivo de alerta temprana para la priorización cantonal de recursos de seguridad en Costa Rica

Trabajo Fin de Máster · Máster en Business Intelligence y Data Science · Universidad Complutense de Madrid

**Autor:** Isaac Sánchez

## Objetivo

Pronosticar los delitos denunciados por cantón, mes y tipo de delito en Costa Rica y, a partir de esas predicciones, priorizar en qué cantones conviene reforzar los recursos preventivos el mes siguiente. El sistema se despliega como una aplicación que muestra un mapa de riesgo, explica cada predicción y se actualiza cuando el OIJ publica datos nuevos.

## Datos

| Fuente | Uso | Licencia |
| --- | --- | --- |
| [Estadísticas Policiales del OIJ](https://datosabiertospj.poder-judicial.go.cr/dataset/estadisticas-policiales) (Poder Judicial) | Delitos denunciados 2015–2026 | Creative Commons Attribution |
| Estimaciones y proyecciones distritales 2000–2025 (INEC, 2014) y proyecciones subnacionales 2000–2050 (INEC, 2025) | Población por cantón y año 2015–2026, para tasas por habitante | Reproducción permitida citando la fuente |
| [Índice de Desarrollo Social 2023](https://sites.google.com/expedientesmideplan.go.cr/odnr/mapas-y-recursos) (MIDEPLAN), capa cantonal | IDS y sus cinco dimensiones por cantón; límites cantonales oficiales para mapas y análisis espacial | Publicación oficial de acceso libre |

Los archivos de datos no se suben al repositorio por su tamaño. Para reproducir el proyecto:

```bash
pip install -r requirements.txt
python scripts/descargar_oij.py      # descarga los archivos del OIJ en data/raw/
```

Las tablas auxiliares ya están construidas en `data/external/`: población por cantón (`scripts/construir_poblacion.py`), Índice de Desarrollo Social y límites cantonales en GeoJSON (`scripts/construir_ids.py`).

También pueden descargarse a mano desde el portal y guardarse en `data/raw/` con su nombre original.

## Estructura

```
data/raw/          datos originales (no versionados)
data/processed/    datos limpios y agregados (no versionados)
data/external/     tablas auxiliares pequeñas derivadas de fuentes oficiales (sí versionadas)
notebooks/         análisis paso a paso, numerados en orden de ejecución
src/               funciones reutilizables de carga, limpieza y modelado
scripts/           descarga y actualización de datos
app/               aplicación Streamlit
reports/figures/   figuras para la memoria
```

## Notebooks

| Notebook | Contenido | Estado |
| --- | --- | --- |
| `01_integracion_y_calidad` | Unión de años, perfil de calidad, armonización de cantones, tabla cantón–mes–delito | Hecho |
| `02_eda` | Tendencias, estacionalidad, mapas, relación con el IDS y eventos masivos | Hecho |
| `03_espacial_y_clustering` | Autocorrelación espacial y perfiles de cantones | Pendiente |
| `04_modelizacion` | Comparación de modelos con validación temporal | Pendiente |
| `05_interpretabilidad` | Explicación de predicciones con SHAP | Pendiente |

## Decisiones de preparación

- **Versiones de archivo:** de 2019 a 2024 el portal publica cada año en XLS y en XLSX. Se usa el XLSX, que es la versión más reciente y completa (el XLS de 2023 solo llega al 30 de agosto y el de 2024 al 16 de junio).
- **Cantones nuevos:** Río Cuarto (2017), Monteverde (2021) y Puerto Jiménez (2022) se reasignan a Grecia, Puntarenas y Golfito para mantener 81 series comparables en todo el periodo.
- **Meses incompletos:** el último mes publicado se excluye si no llega hasta su último día.
- **Registros sin cantón:** se excluyen de la tabla agregada (0,2 % del total).
- **Filas idénticas:** se conservan, porque sin identificador de caso pueden ser varias víctimas de un mismo hecho.
- **Género:** vacío en todos los registros; se descarta.
- **Población cantonal:** el INEC solo publica la proyección por cantón con base en el censo 2011 (llega a 2025 y sobrestima la población reciente en torno a un 3 %). La proyección nueva, basada en el censo 2022, solo llega a provincia. Se combinan: el peso de cada cantón dentro de su provincia sale de la proyección de 2014 y se aplica al total provincial de la de 2025. Para 2026 se mantiene el peso de 2025.
- **Índice de Desarrollo Social:** el shapefile del MIDEPLAN trae 84 cantones. Los polígonos de Río Cuarto, Monteverde y Puerto Jiménez se fusionan con su cantón de origen y se usa el índice publicado para este. La dimensión de seguridad se calcula con datos de criminalidad, así que no se usará como variable predictiva para evitar circularidad.
