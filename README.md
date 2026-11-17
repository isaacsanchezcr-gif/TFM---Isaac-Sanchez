# Sistema predictivo de alerta temprana para la priorización cantonal de recursos de seguridad en Costa Rica

Trabajo Fin de Máster · Máster en Business Intelligence y Data Science · Universidad Complutense de Madrid

**Autor:** Isaac Sánchez

## Objetivo

Pronosticar los delitos denunciados por cantón, mes y tipo de delito en Costa Rica y, a partir de esas predicciones, priorizar en qué cantones conviene reforzar los recursos preventivos el mes siguiente. El sistema se despliega como una aplicación que muestra un mapa de riesgo, explica cada predicción y se actualiza cuando el OIJ publica datos nuevos.

## Datos

| Fuente | Uso | Licencia |
| --- | --- | --- |
| [Estadísticas Policiales del OIJ](https://datosabiertospj.poder-judicial.go.cr/dataset/estadisticas-policiales) (Poder Judicial) | Delitos denunciados 2015–2026 | Creative Commons Attribution |
| Población cantonal (INEC) | Tasas por habitante | Pendiente |
| Índice de Desarrollo Social (MIDEPLAN) | Variables socioeconómicas | Pendiente |

Los archivos de datos no se suben al repositorio por su tamaño. Para reproducir el proyecto:

```bash
pip install -r requirements.txt
python scripts/descargar_oij.py      # descarga los archivos del OIJ en data/raw/
```

También pueden descargarse a mano desde el portal y guardarse en `data/raw/` con su nombre original.

## Estructura

```
data/raw/          datos originales (no versionados)
data/processed/    datos limpios y agregados (no versionados)
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
| `02_eda` | Tendencias, estacionalidad, mapas y anomalías | Pendiente |
| `03_espacial_y_clustering` | Autocorrelación espacial y perfiles de cantones | Pendiente |
| `04_modelizacion` | Comparación de modelos con validación temporal | Pendiente |
| `05_interpretabilidad` | Explicación de predicciones con SHAP | Pendiente |

## Decisiones de preparación

- **Cantones nuevos:** Río Cuarto (2017), Monteverde (2021) y Puerto Jiménez (2022) se reasignan a Grecia, Puntarenas y Golfito para mantener 81 series comparables en todo el periodo.
- **Meses incompletos:** el último mes publicado se excluye si no llega hasta su último día.
- **Registros sin cantón:** se excluyen de la tabla agregada (0,2 % del total).
- **Filas idénticas:** se conservan, porque sin identificador de caso pueden ser varias víctimas de un mismo hecho.
- **Género:** vacío en todos los registros; se descarta.
