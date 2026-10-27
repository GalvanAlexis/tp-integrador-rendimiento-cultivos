# Rendimiento de cultivos por partido: ¿se viene una campaña mala?

Trabajo Práctico Integrador de **Aprendizaje Automático**.

**Repositorio:** https://github.com/GalvanAlexis/tp-integrador-rendimiento-cultivos

**Estado:** Entrega 4 terminada. Clasificadores entrenados, comparacion con la regresión+umbral realizada, app Flask operativa.

---

## Problema

Los productores y asesores agronómicos tienen que decidir insumos, cobertura de seguro, financiamiento y venta anticipada antes de conocer el rendimiento de la campaña. El clima explica una parte importante de la variación entre campañas, pero no toda.

El proyecto predice el rendimiento de un cultivo en un partido bonaerense antes de la cosecha y genera una alerta si la campaña se espera mala.

| Tarea | Pregunta | Salida |
|---|---|---|
| Regresión | ¿Qué rendimiento se espera? | kg/ha |
| Clasificación | ¿La campaña será mala? | Alerta sí/no |

La alerta se genera de dos maneras y luego se comparan:

1. **Clasificador directo:** predice la clase.
2. **Regresión + umbral:** predice el rendimiento y aplica el umbral.

## Decisión y usuario

- **Usuario:** productores, asesores agronómicos, cooperativas, aseguradoras y áreas municipales de producción.
- **Decisión:** contratar o reforzar cobertura, vender una parte de la cosecha a futuro, gestionar financiamiento o priorizar asistencia en los partidos con alerta.
- **Momento de la predicción:** fecha de corte fija por cultivo, al final del período crítico y antes de la cosecha.

| Cultivo | Corte |
|---|---|
| Trigo | 15/11 |
| Maíz | 31/01 |
| Girasol | 31/01 |
| Soja | 28/02 |

El corte se ubica cuando ya empezó el período crítico de cada cultivo. La justificación, las fuentes y las limitaciones están en el informe de la Entrega 2.

## Variable numérica y umbral

- **Variable numérica:** rendimiento en kg/ha.
- **Variable de modelado:** `ratio = rendimiento / promedio de las 5 campañas anteriores`.
- **Clase:** `campaña_mala = 1` si `ratio < 0,85`.

El umbral relativo absorbe diferencias de suelo y tendencia tecnológica entre partidos. El 0,85 se justifica frente a la Ley 26.509 de Emergencia Agropecuaria (50%), el seguro multirriesgo (60-70%) y la tasa de positivos que deja en train (18%). Los umbrales 0,80 y 0,90 se analizan como sensibilidad.

## Fuente de datos

**Estimaciones Agrícolas**, Datos Abiertos de la Provincia de Buenos Aires.

- Organismo: Ministerio de Economía. Secretaría de Agricultura, Ganadería y Pesca. Dirección de Estimaciones Agrícolas.
- Período: campañas 1969/70 a 2025/26.
- Licencia: CC BY 4.0.
- Catálogo: https://catalogo.datos.gba.gob.ar/dataset/estimaciones-agro
- Archivo: `datos/raw/estimaciones-agricolas-1969_2026.csv`

El clima se obtiene de **NASA POWER** (datos diarios desde 1981, reanálisis en grilla de aproximadamente 0,5°).

## Alcance de la Entrega 2

- Cultivos: **maíz, trigo total, girasol y soja total**.
- Se excluyen sorgo (alta proporción de rendimiento cero) y cebada cervecera (serie corta, termina en 2015/16).
- Se excluye la campaña 2025/26 por estar incompleta.
- Campañas de modelado: 1981/82 a 2024/25.
- Unidad: partido × cultivo × campaña.

## Estructura del repositorio

```text
proyecto/
├── README.md
├── requirements.txt
├── app.py
├── modelos/
│   ├── regresor.joblib
│   ├── regresor_metadata.json
│   ├── clasificador.joblib
│   └── clasificador_metadata.json
├── templates/
│   ├── index.html
│   └── predecir.html
├── public/
├── notebooks/
│   ├── 02_preparacion_y_baselines.ipynb
│   └── 03_regresion.ipynb
├── src/
│   ├── config.py
│   ├── descargar_datos.py
│   ├── preparar_datos.py
│   ├── baselines.py
│   ├── clima.py
│   ├── regresion.py
│   └── clasificacion.py
├── datos/
│   ├── README.md
│   ├── diccionario_variables.md
│   ├── raw/
│   └── procesado/
├── informe/
│   ├── entrega_1_propuesta.md
│   ├── entrega_2_preparacion_y_baselines.md
│   ├── entrega_3_regresion.md
│   ├── entrega_4_clasificacion.md
│   ├── bitacora.md
│   └── figuras/
└── tests/
```

## Inicio rápido

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.descargar_datos
python -m src.preparar_datos
python -m src.baselines
python -m src.clima --merge
python -m src.regresion
python -m src.clasificacion
```

App Flask:

```bash
py -m flask --app app run --debug
```

Notebooks:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/02_preparacion_y_baselines.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_regresion.ipynb
```

Tests:

```bash
python -m pytest
```

Clima:

```bash
python -m src.clima --sample
python -m src.clima --full
python -m src.clima --merge
```

La descarga completa ya se ejecutó: 107 partidos, 18.832 filas de clima y 15.179 filas de modelado con clima completo. La radiación se conserva en la tabla cruda pero se excluye del modelado por faltantes en 1981-1983; quedan nueve variables climáticas.

El notebook no descarga clima: lee `datos/procesado/dataset_modelado_con_clima.csv`, que está versionado. `--merge` lo regenera sin conexión a partir de `clima_features.csv`. Solo `--full` y `--sample` consultan la API de NASA POWER.

## Resultados principales de la Entrega 2

Dataset de modelado: **15.179 filas** (11.147 train, 1.828 validación, 2.204 test).

| Baseline (test) | MAE ratio | MAE kg/ha | Accuracy | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Dummy | 0,177 | 756 | 0,787 | 0,000 | 0,000 |
| Promedio 5 campañas | **0,154** | **626** | 0,787 | 0,000 | 0,000 |

Lectura:

- El promedio de 5 campañas es el baseline real del problema y supera al Dummy en test.
- En validación el Dummy gana por poco; en test pierde. El período de prueba es más malo que el de entrenamiento: el ratio medio pasa de 1,079 en train a 0,960 en test.
- Ambos baselines tienen **recall = 0**. Aciertan cerca del 79% y no detectan ninguna campaña mala. Es el ejemplo de "buen MAE, mala clasificación".
- La campaña 2022/23 concentra el **60,7%** de los casos malos del test. El modelo de la Clase 3 tiene que demostrar que la detecta con clima.
- Hay 12 ratios extremos (mayores a 3), todos causados por campañas previas sin cosecha. Se conservan y se agrega `prev5_ceros` como predictor.
- En maíz y soja el clima se relaciona con el ratio en la dirección esperada; en trigo y girasol la relación es casi nula.

## Resultados principales de la Entrega 3

Se compararon 136 configuraciones de árbol y KNN, con y sin clima, elegidas con validación. El regresor seleccionado es **KNN con clima, k = 300 y pesos por distancia**.

| Modelo (test) | MAE ratio | MAE kg/ha | RMSE kg/ha | R² kg/ha | Recall de la alerta |
|---|---:|---:|---:|---:|---:|
| Dummy por cultivo | 0,177 | 756 | 1.174 | 0,704 | 0,00 |
| Promedio 5 campañas | **0,154** | **626** | 981 | 0,793 | 0,00 |
| **KNN con clima** | 0,159 | 637 | **938** | **0,811** | **0,23** |

Lectura:

- En validación, el KNN supera a los dos baselines.
- En test, supera al Dummy pero no al promedio de 5 campañas en MAE. Sí lo supera en RMSE y R² en kg/ha, en la sequía 2022/23 y en soja.
- La causa es un sesgo de nivel: el test (ratio medio 0,96) fue más malo que cualquier período de entrenamiento (1,08).
- La precipitación de la ventana es la variable más importante. El clima explica bien la soja y algo del maíz, pero no el trigo ni el girasol.
- Por la regresión hacia la media, la alerta obtenida desde la regresión detecta solo el 23% de las campañas malas.

## Resultados principales de la Entrega 4

Se compararon 228 configuraciones de árbol y KNN como clasificadores directos. El criterio de selección es el menor costo total en validación (FN = 2 × FP). Campañas malas reales en test: 470 (21,3%).

| Modelo (test) | Recall | Precision | F1 | Costo | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy | 0,00 | — | 0,00 | 940 | 0 | 0 | 470 |
| Regresión + umbral (E3) | 0,23 | 0,77 | 0,35 | 758 | 107 | 32 | 363 |
| KNeighborsClassifier con_clima (seleccionado) | 0,08 | **0,95** | 0,14 | 870 | 36 | 2 | 434 |
| **DecisionTreeClassifier con_clima** (prof=5) | **0,67** | 0,42 | **0,51** | **752** | 315 | 442 | 155 |

Lectura:

- El árbol con_clima tiene el mayor recall (67%) y el menor costo total (752), por debajo de la regresión (758).
- El KNN fue seleccionado por costo de validación pero no generaliza al período de test (sequía 2022/23).
- El clasificador directo (árbol) supera al sistema de regresión + umbral en recall, F1 y costo total.

## Próximos pasos

1. Regenerar `modelos/regresor.joblib` con scikit-learn 1.9.0 para eliminar los warnings de versión.
2. Agregar un modelo por cultivo (especialmente soja y maíz, donde el clima explica más).
3. Recalibrar el nivel del regresor sin usar el test.

## Equipo

- Francisco Calahorra
- Federico Valdés
- Alexis Galván

Curso: Aprendizaje Automático 2026.

## Licencia

Código: MIT. Datos: CC BY 4.0, con atribución a la Dirección de Estimaciones Agrícolas de la Provincia de Buenos Aires.
