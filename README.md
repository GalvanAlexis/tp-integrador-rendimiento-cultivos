# Rendimiento de cultivos por partido: ¿se viene una campaña mala?

Trabajo Práctico Integrador de **Aprendizaje Automático**.

**Repositorio:** https://github.com/GalvanAlexis/tp-integrador-rendimiento-cultivos

**Estado:** Entrega 5. La app Flask está publicada en Vercel, el informe final está en [informe/informe_final.pdf](informe/informe_final.pdf) y la presentación en [presentacion/presentacion.html](presentacion/presentacion.html).

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
├── requirements-dev.txt
├── vercel.json
├── .python-version
├── .vercelignore
├── app.py
├── modelos/
│   ├── regresor.joblib
│   ├── regresor_metadata.json
│   ├── clasificador.joblib
│   └── clasificador_metadata.json
├── templates/
│   ├── base.html
│   ├── index.html
│   └── predecir.html
├── public/
├── notebooks/
│   ├── 02_preparacion_y_baselines.ipynb
│   ├── 03_regresion.ipynb
│   ├── 04_clasificacion.ipynb
│   └── entrenamiento.ipynb
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
│   ├── informe_final.md
│   ├── informe_final.pdf
│   ├── construir_pdf.py
│   ├── bitacora.md
│   └── figuras/
├── presentacion/
│   ├── plantilla.html
│   ├── presentacion.html
│   ├── presentacion.pdf
│   ├── figuras_presentacion.py
│   ├── construir_presentacion.py
│   └── figuras/
└── tests/
```

## Inicio rápido

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt

python -m src.descargar_datos
python -m src.preparar_datos
python -m src.baselines
python -m src.clima --merge
python -m src.regresion
python -m src.clasificacion
```

`requirements.txt` contiene solo lo necesario para la app en producción (Flask, joblib, numpy, pandas y scikit-learn). `requirements-dev.txt` agrega matplotlib, requests, pytest y Jupyter para reproducir notebooks, figuras y tests.

App Flask:

```bash
python -m flask --app app run --debug
```

Notebooks:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/02_preparacion_y_baselines.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_regresion.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/04_clasificacion.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/entrenamiento.ipynb
```

`notebooks/entrenamiento.ipynb` recorre las cinco etapas en un solo archivo, desde la descarga hasta la predicción con los modelos guardados. Es el que se entrega ejecutado desde cero.

Informe final:

```bash
python informe/construir_pdf.py
```

El informe se escribe en `informe/informe_final.md` y el script lo convierte a `informe/informe_final.pdf` con Chrome en modo headless. Requiere `markdown-it-py`, que está en `requirements-dev.txt`.

Presentación:

```bash
python presentacion/figuras_presentacion.py
python presentacion/construir_presentacion.py --pdf
```

`presentacion/plantilla.html` es el diseño de las 14 diapositivas y `construir_presentacion.py` arma `presentacion/presentacion.html` con las figuras incrustadas, así el archivo se abre con doble clic y funciona sin servidor. `--pdf` genera además `presentacion/presentacion.pdf`. Las figuras de la presentación se regeneran desde `datos/procesado/` con tipografía grande, porque las del informe están pensadas para leerse de cerca.

Controles: flechas para navegar, `F` pantalla completa, `S` muestra las notas del orador y `#7` en la URL abre la diapositiva 7.

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

Se compararon DummyClassifier, DecisionTreeClassifier, KNeighborsClassifier (con StandardScaler en el Pipeline) y GaussianNB, con y sin clima: 268 configuraciones, cada una con 19 umbrales de decisión. Se eligió por el menor costo total en validación (FN = 2 × FP). El clasificador seleccionado es un **árbol con clima, profundidad 5, hojas de al menos 50 filas y umbral 0,65**. Campañas malas reales en test: 470 (21,3%).

| Sistema (test) | Accuracy | Precisión | Recall | F1 | Costo | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy | 0,787 | — | 0,00 | 0,00 | 940 | 0 | 0 | 470 |
| Regresión + umbral (E3) | 0,821 | **0,77** | 0,23 | 0,35 | 758 | 107 | 32 | 363 |
| **Árbol con clima (seleccionado)** | **0,823** | 0,65 | **0,37** | **0,47** | **687** | 173 | 93 | 297 |

Lectura:

- Sin clima, ningún clasificador supera al Dummy. Con clima, solo el árbol se separa con claridad en validación (costo 443 contra 540).
- El clasificador directo detecta 66 campañas malas más que la regresión, con 61 falsas alarmas más, y reduce el costo un 9%. La regresión es más precisa, pero se pierde tres de cada cuatro campañas malas.
- La mitad de los falsos negativos son campañas apenas malas (ratio de 0,75 a 0,85). Los falsos positivos son campañas secas que terminaron bien.
- Cuando los dos sistemas alertan, el 89% de las alertas son correctas. La app usa el clasificador como alerta principal y la coincidencia con la regresión como señal de alta confianza.

Detalle en [informe/entrega_4_clasificacion.md](informe/entrega_4_clasificacion.md).

## Despliegue en Vercel

La app se publica como una única función de Python a partir de `app.py`; no entrena al arrancar, solo carga los pipelines y los CSV versionados.

- `vercel.json` fija `maxDuration` y el bundle de la función: incluye `templates/`, `modelos/`, `src/` y los tres CSV que usa la app.
- `.vercelignore` deja afuera notebooks, informes, tests, datos crudos y la caché de clima.
- `.python-version` fija Python 3.14, la misma versión de `requirements.txt`.
- No hay variables de entorno ni credenciales.

```bash
npm i -g vercel
vercel          # preview
vercel --prod   # produccion
```

**URL:** https://tp-integrador-rendimiento-cultivos.vercel.app

La app no entrena al arrancar: el primer request carga los dos pipelines y el CSV de histórico. El build incluye `templates/`, `modelos/`, `src/` y los tres CSV que usa la app.

El proyecto está vinculado al repositorio de GitHub: cada push a `main` redeploya producción y cada rama genera un preview deployment. El primer request después de un deploy tarda unos segundos porque carga los pipelines.

## Próximos pasos

1. ~~Desplegar la app Flask en Vercel.~~
2. Calibrar las probabilidades del clasificador con validación.
3. Evaluar modelos por grupo de cultivos (verano e invierno), donde el clima tiene un efecto distinto.

## Equipo

- Francisco Calahorra
- Federico Valdés
- Alexis Galván

Curso: Aprendizaje Automático 2026.

## Licencia

Código: MIT. Datos: CC BY 4.0, con atribución a la Dirección de Estimaciones Agrícolas de la Provincia de Buenos Aires.
