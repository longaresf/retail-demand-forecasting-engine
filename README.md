# Retail Demand Forecasting Engine

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](#)
[![Tests](https://github.com/longaresf/retail-demand-forecasting-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/longaresf/retail-demand-forecasting-engine/actions/workflows/tests.yml)

Motor modular y reproducible para preparar series de demanda de retail, crear
predictores temporales, comparar baselines con Gradient Boosted Trees y validar
pronósticos con particiones cronológicas.

## Contexto de negocio

Un pronóstico fiable a nivel tienda-producto ayuda a equilibrar disponibilidad
y capital inmovilizado: reduce quiebres de stock y exceso de inventario, y
permite planificar reposición ante demanda estacional o intermitente. Las
métricas deben interpretarse junto al costo asimétrico de sobrepredecir y
subpredecir; ningún modelo sustituye la revisión de restricciones operativas.

El proyecto ofrece una base técnica para evaluar ese flujo, no una promesa de
precisión ni un modelo ya calibrado para una operación específica.

## Arquitectura y workflow

```text
CSV / fuente propia
       |
       v
preprocessing -> features -> backtesting -> baselines / XGBoost -> evaluate
       |             |              |                           |
  calidad,       calendario,   folds temporales            RMSE, MAE,
  fechas y       lags y medias  expanding-window             WAPE
  faltantes
```

- `src/preprocessing.py`: conversión y agregación de datos, reporte de nulos y
  puntuaciones de autocorrelación en períodos candidatos.
- `src/features.py`: calendario y atributos retardados, rolling y exponenciales.
  Los atributos de demanda se desplazan para no usar el objetivo actual.
- `src/models.py`: promedio móvil, media histórica, naive estacional y wrapper
  entrenable de XGBoost.
- `src/backtesting.py`: validación walk-forward de ventana expansiva, en orden
  temporal y sin mezclar observaciones futuras en entrenamiento.
- `src/evaluate.py`: RMSE, MAE, WAPE y tabla comparativa de modelos.
- `app.py`: dashboard Streamlit para explorar series y comparar XGBoost con la
  demanda observada en un período de prueba temporal.
- `tests/test_backtesting_and_models.py`: pruebas de orden cronológico,
  predicciones/métricas finitas y features causales.
- `data/sample_data.py`: generador determinista local; `notebooks/` contiene la
  exploración y visualización, no lógica de negocio.
- `.github/workflows/tests.yml`: ejecuta pytest para Python 3.10, 3.11 y 3.12.

## Metodología y backtesting

Las observaciones se ordenan cronológicamente y cada fold valida un bloque
futuro distinto. En walk-forward, el conjunto de entrenamiento se expande
después de cada bloque. Las transformaciones aprendidas por un modelo deben
ajustarse únicamente con el fold de entrenamiento correspondiente. Los
predictores de lags/ventanas se calculan con valores anteriores al objetivo.

En un caso real, se deben separar además un período final de prueba que no se
use para seleccionar hiperparámetros ni modelos. Ajusta la longitud de los
bloques al horizonte de reposición y evalúa por segmento tienda-producto y
costo de negocio. WAPE está expresado en porcentaje y no está definido si la
suma de valores reales absolutos es cero.

## Instalación y pruebas

```bash
git clone https://github.com/longaresf/retail-demand-forecasting-engine.git
cd retail-demand-forecasting-engine
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Las pruebas nuevas pueden ejecutarse de forma aislada:

```bash
python -m pytest -q tests/test_backtesting_and_models.py
```

Descarga opcional del dataset Kaggle e instrucciones de credenciales en
[data/README.md](data/README.md). No versiona datasets reales ni tokens.

## Aplicación web local

Desde la raíz del repositorio y con el entorno virtual activado:

```bash
streamlit run app.py
```

La aplicación abre un dashboard con datos sintéticos reproducibles de forma
predeterminada. También permite cargar un CSV con las columnas `date`, `store`,
`item` y `sales` (por ejemplo, `train.csv` de Store Item Demand Forecasting),
elegir una tienda y un producto, ajustar la longitud del bloque de prueba y
visualizar demanda histórica, predicción XGBoost y métricas RMSE/WAPE.

## Por qué validar sin fuga temporal

En retail, una partición aleatoria puede colocar días futuros en entrenamiento
y días pasados en prueba. Eso permite que el modelo aprenda promociones,
tendencias o patrones de calendario que todavía no estarían disponibles al
momento de reponer inventario, y produce métricas artificialmente optimistas.
El walk-forward de este motor ordena primero las observaciones, entrena con
fechas estrictamente anteriores al bloque de prueba y expande el histórico en
cada fold. Las features de demanda se desplazan para que lags y ventanas solo
usen valores previos. El dashboard evalúa un paso hacia adelante: se asume que,
al avanzar por el período de prueba, la venta del día anterior ya se observó.
Para una decisión multi-horizonte emitida en una sola fecha, se requiere una
validación recursiva distinta que no use ventas reales futuras.

## Ejemplo de uso

```python
from data.sample_data import generate_sample_data
from src.backtesting import walk_forward_split
from src.evaluate import compare_forecasts
from src.features import add_time_series_features
from src.models import XGBoostForecaster, moving_average_forecast

data = generate_sample_data(periods=400, stores=1, items=1)
features = add_time_series_features(
    data, group_cols=["store", "item"], lags=(1, 7, 28), windows=(7, 28)
)
feature_columns = [
    "day_of_week", "month", "sales_lag_1", "sales_lag_7",
    "sales_lag_28", "sales_rolling_mean_7", "sales_rolling_mean_28",
]

# Ejemplo ilustrativo de un fold; en producción define folds/horizonte de negocio.
train, validation = next(
    walk_forward_split(features, n_splits=1, test_size=28)
)
model = XGBoostForecaster().fit(train, feature_columns)
predictions = model.predict(validation)
baseline = moving_average_forecast(train["sales"], len(validation), window=7)
print(compare_forecasts(
    validation["sales"].to_numpy(),
    {"moving_average": baseline, "xgboost": predictions},
))
```

Las filas iniciales no tienen todos los lags disponibles: XGBoost puede tratar
valores ausentes en predictores, pero entrenamiento y evaluación deben mantener
el objetivo válido. Para pronósticos recursivos a varios pasos, genera cada
paso futuro actualizando lags únicamente con observaciones disponibles o
predicciones previas; el ejemplo puntúa filas con predictores observados.

## Exploración

Abre [notebooks/01_eda_exploratory_analysis.ipynb](notebooks/01_eda_exploratory_analysis.ipynb)
para inspeccionar el esquema, cobertura, tendencia, calendario y estacionalidad
del generador sintético. Reemplázalo por un dataset autorizado para obtener
conclusiones de negocio reales.

## Licencia

El proyecto está preparado para distribución bajo MIT. Confirma la política de
licenciamiento de tu organización antes de publicar o incorporar datos de
terceros.
